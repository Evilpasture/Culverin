#include "culverin_command_buffer.h"

/**
 * Internal helper to remove a body from the dense arrays.
 * Maintains a packed, contiguous array by swapping the last body into the hole.
 * MUST be called while holding SHADOW_LOCK.
 */
void world_remove_body_slot(PhysicsWorldObject *self, uint32_t slot) {
  uint32_t dense_idx = self->slot_to_dense[slot];
  auto last_dense = (uint32_t)self->count - 1;
  JPH_BodyID bid = self->body_ids[dense_idx];

  // 1. Cleanup Jolt Mapping
  if (bid != JPH_INVALID_BODY_ID) {
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);
    if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
      self->id_to_handle_map[j_idx] = 0;
    }
  }

  // 2. Swap-and-Pop
  if (dense_idx != last_dense) {
    // Type-safe Casts
    auto *pos = (PosStride *)self->positions;
    auto *prev_pos = (PosStride *)self->prev_positions;
    
    auto *rot = (AuxStride *)self->rotations;
    auto *prev_rot = (AuxStride *)self->prev_rotations;
    auto *lvel = (AuxStride *)self->linear_velocities;
    auto *avel = (AuxStride *)self->angular_velocities;

    // Struct Copy (Compiler handles size/alignment)
    pos[dense_idx] = pos[last_dense];
    prev_pos[dense_idx] = prev_pos[last_dense];

    rot[dense_idx] = rot[last_dense];
    prev_rot[dense_idx] = prev_rot[last_dense];
    lvel[dense_idx] = lvel[last_dense];
    avel[dense_idx] = avel[last_dense];

    // Metadata Copy
    self->body_ids[dense_idx] = self->body_ids[last_dense];
    self->user_data[dense_idx] = self->user_data[last_dense];
    self->categories[dense_idx] = self->categories[last_dense];
    self->masks[dense_idx] = self->masks[last_dense];
    self->material_ids[dense_idx] = self->material_ids[last_dense];

    // Fix Indirection
    uint32_t mover_slot = self->dense_to_slot[last_dense];
    self->slot_to_dense[mover_slot] = dense_idx;
    self->dense_to_slot[dense_idx] = mover_slot;
  }

  // 3. Finalize
  self->generations[slot]++;
  self->free_slots[self->free_count++] = slot;
  self->slot_states[slot] = SLOT_EMPTY;
  self->count--;
  self->view_shape[0] = (Py_ssize_t)self->count;
}

// Helper to grow queue
bool ensure_command_capacity(PhysicsWorldObject *self) {
  if (self->command_count >= self->command_capacity) {
    // Defensive: handle zero or uninitialized capacity
    size_t new_cap =
        (self->command_capacity == 0) ? 64 : self->command_capacity * 2;

    // Safety check: Prevent overflow on extreme counts
    if (new_cap > (SIZE_MAX / sizeof(PhysicsCommand))) {
      return false;
    }

    void *new_ptr =
        PyMem_RawRealloc(self->command_queue, new_cap * sizeof(PhysicsCommand));
    if (!new_ptr) {
      return false;
    }

    self->command_queue = (PhysicsCommand *)new_ptr;
    self->command_capacity = new_cap;
  }
  return true;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void flush_commands_internal(PhysicsWorldObject *self, PhysicsCommand *queue, size_t count) {
  if (count == 0) return;
  JPH_BodyInterface *bi = self->body_interface;

  // Pre-cast buffers for safe access inside the loop
  auto *shadow_pos = (PosStride *)self->positions;
  auto *shadow_prev_pos = (PosStride *)self->prev_positions;
  auto *shadow_rot = (AuxStride *)self->rotations;
  auto *shadow_prev_rot = (AuxStride *)self->prev_rotations;
  auto *shadow_lvel = (AuxStride *)self->linear_velocities;
  auto *shadow_avel = (AuxStride *)self->angular_velocities;

  for (size_t i = 0; i < count; i++) {
    PhysicsCommand *cmd = &queue[i];
    uint32_t header = cmd->header;
    CommandType type = CMD_GET_TYPE(header);
    uint32_t slot = CMD_GET_SLOT(header);

    SHADOW_LOCK(&self->shadow_lock);
    SlotState state = self->slot_states[slot];
    JPH_BodyID bid = JPH_INVALID_BODY_ID;
    uint32_t dense = 0;

    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
      dense = self->slot_to_dense[slot];
      bid = self->body_ids[dense];
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    if (type != CMD_CREATE_BODY && bid == JPH_INVALID_BODY_ID) continue;

    switch (type) {
    case CMD_CREATE_BODY: {
      JPH_BodyCreationSettings *s = cmd->create.settings;
      JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);
      JPH_BodyCreationSettings_Destroy(s);

      SHADOW_LOCK(&self->shadow_lock);
      if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        self->count--;
        self->slot_states[slot] = SLOT_EMPTY;
        self->generations[slot]++;
        self->free_slots[self->free_count++] = slot;
      } else {
        self->body_ids[self->slot_to_dense[slot]] = new_bid;
        uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
          self->id_to_handle_map[j_idx] = make_handle(slot, self->generations[slot]);
        }
        self->slot_states[slot] = SLOT_ALIVE;
      }
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_DESTROY_BODY: {
      JPH_BodyInterface_RemoveBody(bi, bid);
      JPH_BodyInterface_DestroyBody(bi, bid);
      
      SHADOW_LOCK(&self->shadow_lock);
      world_remove_body_slot(self, slot);
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_POS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      // Use JPH_Real from command directly
      p->x = cmd->pos.x; p->y = cmd->pos.y; p->z = cmd->pos.z; 
      JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      uint32_t d = self->slot_to_dense[slot];
      auto *shadow_pos = (PosStride *)self->positions;
      auto *shadow_ppos = (PosStride *)self->prev_positions;
      
      shadow_pos[d] = (PosStride){cmd->pos.x, cmd->pos.y, cmd->pos.z};
      shadow_ppos[d] = shadow_pos[d];
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_ROT: {
      JPH_STACK_ALLOC(JPH_Quat, q);
      // No conversion logic needed; already floats
      q->x = cmd->quat.x; 
      q->y = cmd->quat.y; 
      q->z = cmd->quat.z; 
      q->w = cmd->quat.w;
      
      JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      dense = self->slot_to_dense[slot];
      
      // AuxStride is float-based, so this is a direct assignment
      auto *shadow_rot = (AuxStride *)self->rotations;
      auto *shadow_prot = (AuxStride *)self->prev_rotations;
      
      shadow_rot[dense] = (AuxStride){cmd->quat.x, cmd->quat.y, cmd->quat.z, cmd->quat.w};
      shadow_prot[dense] = shadow_rot[dense];
      
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_TRNS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->transform.px; p->y = cmd->transform.py; p->z = cmd->transform.pz;
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->transform.rx; q->y = cmd->transform.ry; q->z = cmd->transform.rz; q->w = cmd->transform.rw;
      JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      dense = self->slot_to_dense[slot];
      PosStride new_p = {cmd->transform.px, cmd->transform.py, cmd->transform.pz};
      AuxStride new_q = {cmd->transform.rx, cmd->transform.ry, cmd->transform.rz, cmd->transform.rw};

      auto *shadow_pos = (PosStride *)self->positions;
      auto *shadow_prev_pos = (PosStride *)self->prev_positions;
      auto *shadow_rot = (AuxStride *)self->rotations;
      auto *shadow_prev_rot = (AuxStride *)self->prev_rotations;
      
      shadow_pos[dense] = new_p;
      shadow_prev_pos[dense] = new_p;
      shadow_rot[dense] = new_q;
      shadow_prev_rot[dense] = new_q;
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_LINVEL: {
      // Direct assignment to Jolt struct
      JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
      JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);

      SHADOW_LOCK(&self->shadow_lock);
      dense = self->slot_to_dense[slot];
      
      // Update shadow buffer (float-to-float)
      shadow_lvel[dense] = (AuxStride){v.x, v.y, v.z, 0.0f};
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_ANGVEL: {
      JPH_Vec3 v = {cmd->vec3f.x, cmd->vec3f.y, cmd->vec3f.z};
      JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);

      SHADOW_LOCK(&self->shadow_lock);
      dense = self->slot_to_dense[slot];
      
      shadow_avel[dense] = (AuxStride){v.x, v.y, v.z, 0.0f};
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_MOTION: {
      JPH_BodyInterface_SetMotionType(bi, bid, (JPH_MotionType)cmd->motion.motion_type, JPH_Activation_Activate);
      uint32_t layer = (cmd->motion.motion_type == 0) ? 0 : 1;
      JPH_BodyInterface_SetObjectLayer(bi, bid, (JPH_ObjectLayer)layer);
      break;
    }

    case CMD_ACTIVATE: JPH_BodyInterface_ActivateBody(bi, bid); break;
    case CMD_DEACTIVATE: JPH_BodyInterface_DeactivateBody(bi, bid); break;

    case CMD_SET_USER_DATA:
      SHADOW_LOCK(&self->shadow_lock);
      self->user_data[self->slot_to_dense[slot]] = cmd->user_data.user_data_val;
      SHADOW_UNLOCK(&self->shadow_lock);
      break;

    case CMD_SET_CCD: {
      JPH_MotionQuality qual = cmd->motion.motion_type ? JPH_MotionQuality_LinearCast : JPH_MotionQuality_Discrete;
      JPH_BodyInterface_SetMotionQuality(bi, bid, qual);
      break;
    }
    case CMD_TELEPORT: {
      // TODO: add teleport method
    }
    default: break;
    }
  }

  // Final count sync for MemoryViews
  SHADOW_LOCK(&self->shadow_lock);
  self->view_shape[0] = (Py_ssize_t)self->count;
  SHADOW_UNLOCK(&self->shadow_lock);
}

/**
 * Helper: Flushes pending commands while releasing shadow_lock to 
 * avoid stalling the world during heavy Jolt operations.
 */
void sync_and_flush_internal(PhysicsWorldObject *self) {
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    if (self->command_count == 0) return;

    // Capture queue and swap
    PhysicsCommand *captured_queue = self->command_queue;
    size_t captured_count = self->command_count;
    self->command_queue = NULL;
    self->command_count = 0;
    self->command_capacity = 0;

    // RELEASE SHADOW LOCK completely before Jolt/Flush
    SHADOW_UNLOCK(&self->shadow_lock);
    
    Py_BEGIN_ALLOW_THREADS
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    flush_commands_internal(self, captured_queue, captured_count);
    PyMem_RawFree(captured_queue);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

    // Re-acquire for the caller
    SHADOW_LOCK(&self->shadow_lock);
}

void clear_command_queue(PhysicsWorldObject *self) {
  if (!self->command_queue) {
    return;
  }

  for (size_t i = 0; i < self->command_count; i++) {
    PhysicsCommand *cmd = &self->command_queue[i];
    if (CMD_GET_TYPE(cmd->header) == CMD_CREATE_BODY) {
      // We own this pointer until it's consumed by Jolt
      if (cmd->create.settings) {
        JPH_BodyCreationSettings_Destroy(cmd->create.settings);
      }
    }
  }
  self->command_count = 0;
}