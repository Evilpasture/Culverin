#include "culverin_command_buffer.h"

/**
 * Internal helper to remove a body from the dense arrays.
 * Maintains a packed, contiguous array by swapping the last body into the hole.
 * MUST be called while holding SHADOW_LOCK.
 */
void world_remove_body_slot(PhysicsWorldObject *self, uint32_t slot) {
  uint32_t dense_idx = self->slot_to_dense[slot];
  uint32_t last_dense = (uint32_t)self->count - 1;
  JPH_BodyID bid = self->body_ids[dense_idx];
  if (bid != JPH_INVALID_BODY_ID) {
    uint32_t j_idx = JPH_ID_TO_INDEX(bid); // Use Macro
    if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
      self->id_to_handle_map[j_idx] = 0;
    }
  }

  // 1. If we aren't already the last element, move the last element into this
  // hole
  if (dense_idx != last_dense) {
    size_t dst = (size_t)dense_idx * 4;
    size_t src = (size_t)last_dense * 4;

    // Copy Shadow Buffers (16 bytes each)
    memcpy(&self->positions[dst], &self->positions[src], 16);
    memcpy(&self->rotations[dst], &self->rotations[src], 16);
    memcpy(&self->prev_positions[dst], &self->prev_positions[src], 16);
    memcpy(&self->prev_rotations[dst], &self->prev_rotations[src], 16);
    memcpy(&self->linear_velocities[dst], &self->linear_velocities[src], 16);
    memcpy(&self->angular_velocities[dst], &self->angular_velocities[src], 16);

    // Copy Metadata
    self->body_ids[dense_idx] = self->body_ids[last_dense];
    self->user_data[dense_idx] = self->user_data[last_dense];
    self->categories[dense_idx] = self->categories[last_dense];
    self->masks[dense_idx] = self->masks[last_dense];
    self->material_ids[dense_idx] = self->material_ids[last_dense];

    // Update Indirection Maps
    uint32_t mover_slot = self->dense_to_slot[last_dense];
    self->slot_to_dense[mover_slot] = dense_idx;
    self->dense_to_slot[dense_idx] = mover_slot;
  }

  // 2. Invalidate the slot
  self->generations[slot]++;
  self->free_slots[self->free_count++] = slot;
  self->slot_states[slot] = SLOT_EMPTY;

  // 3. Update World Counters
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

  for (size_t i = 0; i < count; i++) {
    PhysicsCommand *cmd = &queue[i];
    uint32_t header = cmd->header;
    CommandType type = CMD_GET_TYPE(header);
    uint32_t slot = CMD_GET_SLOT(header);

    // --- 1. DYNAMIC RESOLUTION (Brief Lock) ---
    // We must resolve the ID and Index right before processing.
    // This allows Create -> Destroy in the same frame to work.
    SHADOW_LOCK(&self->shadow_lock);
    SlotState state = self->slot_states[slot];
    JPH_BodyID bid = JPH_INVALID_BODY_ID;
    uint32_t dense_idx = 0;

    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
      dense_idx = self->slot_to_dense[slot];
      bid = self->body_ids[dense_idx];
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    // If the body doesn't exist and we aren't creating it, skip this command.
    if (type != CMD_CREATE_BODY && bid == JPH_INVALID_BODY_ID) {
      continue;
    }

    switch (type) {
    case CMD_CREATE_BODY: {
      JPH_BodyCreationSettings *s = cmd->create.settings;
      JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      if (UNLIKELY(new_bid == JPH_INVALID_BODY_ID)) {
        // Jolt Failed: Rollback the slot reservation
        self->count--;
        self->slot_states[slot] = SLOT_EMPTY;
        self->generations[slot]++;
        self->free_slots[self->free_count++] = slot;
      } else {
        // Success: Finalize the shadow buffers
        uint32_t d = self->slot_to_dense[slot];
        self->body_ids[d] = new_bid;
        
        uint32_t j_idx = JPH_ID_TO_INDEX(new_bid);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
          self->id_to_handle_map[j_idx] = make_handle(slot, self->generations[slot]);
        }
        self->slot_states[slot] = SLOT_ALIVE;
      }
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_BodyCreationSettings_Destroy(s);
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
      p->x = (JPH_Real)cmd->vec.x; p->y = (JPH_Real)cmd->vec.y; p->z = (JPH_Real)cmd->vec.z;
      JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      // RE-FETCH: Previous commands in this batch might have triggered a 
      // swap-and-pop, changing our dense index. We must re-fetch.
      uint32_t current_dense = self->slot_to_dense[slot];
      size_t off = (size_t)current_dense * 4;
      self->positions[off+0] = cmd->vec.x; self->positions[off+1] = cmd->vec.y; self->positions[off+2] = cmd->vec.z;
      self->prev_positions[off+0] = cmd->vec.x; self->prev_positions[off+1] = cmd->vec.y; self->prev_positions[off+2] = cmd->vec.z;
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_ROT: {
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->vec.x; q->y = cmd->vec.y; q->z = cmd->vec.z; q->w = cmd->vec.w;
      JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      uint32_t current_dense = self->slot_to_dense[slot];
      memcpy(&self->rotations[current_dense * 4], &cmd->vec, 16);
      memcpy(&self->prev_rotations[current_dense * 4], &cmd->vec, 16);
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_TRNS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = (JPH_Real)cmd->transform.px; p->y = (JPH_Real)cmd->transform.py; p->z = (JPH_Real)cmd->transform.pz;
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->transform.rx; q->y = cmd->transform.ry; q->z = cmd->transform.rz; q->w = cmd->transform.rw;
      JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);

      SHADOW_LOCK(&self->shadow_lock);
      uint32_t current_dense = self->slot_to_dense[slot];
      size_t off = (size_t)current_dense * 4;
      self->positions[off+0] = cmd->transform.px; self->positions[off+1] = cmd->transform.py; self->positions[off+2] = cmd->transform.pz;
      self->prev_positions[off+0] = cmd->transform.px; self->prev_positions[off+1] = cmd->transform.py; self->prev_positions[off+2] = cmd->transform.pz;
      memcpy(&self->rotations[off], &cmd->transform.rx, 16);
      memcpy(&self->prev_rotations[off], &cmd->transform.rx, 16);
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_LINVEL: {
      JPH_Vec3 v = {cmd->vec.x, cmd->vec.y, cmd->vec.z};
      JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);
      SHADOW_LOCK(&self->shadow_lock);
      uint32_t current_dense = self->slot_to_dense[slot];
      self->linear_velocities[current_dense * 4 + 0] = v.x;
      self->linear_velocities[current_dense * 4 + 1] = v.y;
      self->linear_velocities[current_dense * 4 + 2] = v.z;
      SHADOW_UNLOCK(&self->shadow_lock);
      break;
    }

    case CMD_SET_ANGVEL: {
      JPH_Vec3 v = {cmd->vec.x, cmd->vec.y, cmd->vec.z};
      JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);
      SHADOW_LOCK(&self->shadow_lock);
      uint32_t current_dense = self->slot_to_dense[slot];
      self->angular_velocities[current_dense * 4 + 0] = v.x;
      self->angular_velocities[current_dense * 4 + 1] = v.y;
      self->angular_velocities[current_dense * 4 + 2] = v.z;
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