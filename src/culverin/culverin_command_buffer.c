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

    // 1. If we aren't already the last element, move the last element into this hole
    if (dense_idx != last_dense) {
        size_t dst = (size_t)dense_idx * 4;
        size_t src = (size_t)last_dense * 4;

        // Copy Shadow Buffers (16 bytes each)
        memcpy(&self->positions[dst],          &self->positions[src],          16);
        memcpy(&self->rotations[dst],          &self->rotations[src],          16);
        memcpy(&self->prev_positions[dst],     &self->prev_positions[src],     16);
        memcpy(&self->prev_rotations[dst],     &self->prev_rotations[src],     16);
        memcpy(&self->linear_velocities[dst],  &self->linear_velocities[src],  16);
        memcpy(&self->angular_velocities[dst], &self->angular_velocities[src], 16);

        // Copy Metadata
        self->body_ids[dense_idx]     = self->body_ids[last_dense];
        self->user_data[dense_idx]    = self->user_data[last_dense];
        self->categories[dense_idx]   = self->categories[last_dense];
        self->masks[dense_idx]        = self->masks[last_dense];
        self->material_ids[dense_idx] = self->material_ids[last_dense];

        // Update Indirection Maps
        uint32_t mover_slot = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = dense_idx;
        self->dense_to_slot[dense_idx]  = mover_slot;
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

void flush_commands(PhysicsWorldObject *self) {
  if (self->command_count == 0) {
    return;
  }

  JPH_BodyInterface *bi = self->body_interface;

  for (size_t i = 0; i < self->command_count; i++) {
    PhysicsCommand *cmd = &self->command_queue[i];
    // Unpack Header
    uint32_t header = cmd->header;
    CommandType type = CMD_GET_TYPE(header);
    uint32_t slot = CMD_GET_SLOT(header);

    // --- Safety Checks ---
    if (type != CMD_CREATE_BODY) {
       if (self->slot_states[slot] != SLOT_ALIVE) continue;
    }

    // Resolve dense index
    uint32_t dense_idx = 0;
    JPH_BodyID bid = JPH_INVALID_BODY_ID;
    
    if (type != CMD_CREATE_BODY) {
      dense_idx = self->slot_to_dense[slot];
      bid = self->body_ids[dense_idx];
    }


    switch (type) {
    case CMD_CREATE_BODY: {
      JPH_BodyCreationSettings *s = cmd->create.settings;
      JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);

      if (new_bid == JPH_INVALID_BODY_ID) {
        self->slot_states[slot] = SLOT_EMPTY;
        self->generations[slot]++;
        self->free_slots[self->free_count++] = slot;
        JPH_BodyCreationSettings_Destroy(s);
        continue;
      }
      uint32_t gen = self->generations[slot];
      BodyHandle handle = make_handle(slot, gen);
      
      uint32_t jolt_idx = JPH_ID_TO_INDEX(new_bid);
      if (self->id_to_handle_map && jolt_idx < self->max_jolt_bodies) {
          self->id_to_handle_map[jolt_idx] = handle;
      }

      size_t new_dense = self->count;
      self->body_ids[new_dense] = new_bid;
      self->slot_to_dense[slot] = (uint32_t)new_dense;
      self->dense_to_slot[new_dense] = slot;
      self->user_data[new_dense] = cmd->create.user_data;

      JPH_STACK_ALLOC(JPH_RVec3, p);
      JPH_STACK_ALLOC(JPH_Quat, q);
      JPH_BodyInterface_GetPosition(bi, new_bid, p);
      JPH_BodyInterface_GetRotation(bi, new_bid, q);

      float fx = (float)p->x;
      float fy = (float)p->y;
      float fz = (float)p->z;
      
      size_t offset = new_dense * 4;
      
      self->positions[offset + 0] = fx;
      self->positions[offset + 1] = fy;
      self->positions[offset + 2] = fz;
      // Correctly preventing creation jitter
      self->prev_positions[offset + 0] = fx;
      self->prev_positions[offset + 1] = fy;
      self->prev_positions[offset + 2] = fz;

      self->rotations[offset + 0] = q->x;
      self->rotations[offset + 1] = q->y;
      self->rotations[offset + 2] = q->z;
      self->rotations[offset + 3] = q->w;
      memcpy(&self->prev_rotations[offset], &self->rotations[offset], 16);

      memset(&self->linear_velocities[offset], 0, 16);
      memset(&self->angular_velocities[offset], 0, 16);

      self->categories[new_dense] = cmd->create.category;
      self->masks[new_dense] = cmd->create.mask;
      self->material_ids[new_dense] = cmd->create.material_id;

      self->count++;
      self->slot_states[slot] = SLOT_ALIVE;
      JPH_BodyCreationSettings_Destroy(s);
      break;
    }

    case CMD_DESTROY_BODY: {
      JPH_BodyInterface_RemoveBody(bi, bid);
      JPH_BodyInterface_DestroyBody(bi, bid);
      world_remove_body_slot(self, slot);
      break;
    }

    case CMD_SET_POS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->vec.x;
      p->y = cmd->vec.y;
      p->z = cmd->vec.z;
      JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);
      
      size_t offset = (size_t)dense_idx * 4;
      self->positions[offset + 0] = cmd->vec.x;
      self->positions[offset + 1] = cmd->vec.y;
      self->positions[offset + 2] = cmd->vec.z;

      // FIX: Reset interpolation (Teleport)
      self->prev_positions[offset + 0] = cmd->vec.x;
      self->prev_positions[offset + 1] = cmd->vec.y;
      self->prev_positions[offset + 2] = cmd->vec.z;
      break;
    }

    case CMD_SET_ROT: {
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->vec.x;
      q->y = cmd->vec.y;
      q->z = cmd->vec.z;
      q->w = cmd->vec.w;
      JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);
      
      size_t offset = (size_t)dense_idx * 4;
      memcpy(&self->rotations[offset], &cmd->vec, 16);
      // FIX: Reset interpolation
      memcpy(&self->prev_rotations[offset], &cmd->vec, 16);
      break;
    }

    case CMD_SET_TRNS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->transform.px;
      p->y = cmd->transform.py;
      p->z = cmd->transform.pz;
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->transform.rx;
      q->y = cmd->transform.ry;
      q->z = cmd->transform.rz;
      q->w = cmd->transform.rw;

      JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);

      size_t offset = (size_t)dense_idx * 4;
      self->positions[offset + 0] = (float)p->x;
      self->positions[offset + 1] = (float)p->y;
      self->positions[offset + 2] = (float)p->z;
      memcpy(&self->rotations[offset], &cmd->transform.rx, 16);

      // FIX: Reset interpolation
      self->prev_positions[offset + 0] = (float)p->x;
      self->prev_positions[offset + 1] = (float)p->y;
      self->prev_positions[offset + 2] = (float)p->z;
      memcpy(&self->prev_rotations[offset], &cmd->transform.rx, 16);
      break;
    }

    case CMD_SET_LINVEL: {
        JPH_Vec3 v = {cmd->vec.x, cmd->vec.y, cmd->vec.z};
        JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);
        self->linear_velocities[dense_idx * 4 + 0] = cmd->vec.x;
        self->linear_velocities[dense_idx * 4 + 1] = cmd->vec.y;
        self->linear_velocities[dense_idx * 4 + 2] = cmd->vec.z;
        break;
    }

    case CMD_SET_ANGVEL: {
      JPH_Vec3 v = {cmd->vec.x, cmd->vec.y, cmd->vec.z};
      JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);
      self->angular_velocities[dense_idx * 4 + 0] = cmd->vec.x;
      self->angular_velocities[dense_idx * 4 + 1] = cmd->vec.y;
      self->angular_velocities[dense_idx * 4 + 2] = cmd->vec.z;
      break;
    }

    case CMD_SET_MOTION: {
      JPH_BodyInterface_SetMotionType(bi, bid,
                                    (JPH_MotionType)cmd->motion.motion_type,
                                    JPH_Activation_Activate);
      // Optional: If you use Layer 0 for Static and Layer 1 for Moving
      uint32_t layer = (cmd->motion.motion_type == 0) ? 0 : 1;
      JPH_BodyInterface_SetObjectLayer(bi, bid, (JPH_ObjectLayer)layer);
      break;
    }

    case CMD_ACTIVATE:
      JPH_BodyInterface_ActivateBody(bi, bid);
      break;
    case CMD_DEACTIVATE:
      JPH_BodyInterface_DeactivateBody(bi, bid);
      break;

    case CMD_SET_USER_DATA: {
      self->user_data[dense_idx] = cmd->user_data.user_data_val;
      break;
    }
    case CMD_SET_CCD: {
        JPH_MotionQuality qual = cmd->motion.motion_type ? 
                                 JPH_MotionQuality_LinearCast : 
                                 JPH_MotionQuality_Discrete;
        JPH_BodyInterface_SetMotionQuality(bi, bid, qual);
        break;
    }
    default:
      DEBUG_LOG("Warning: Invalid command during flush. Check flush_commands.");
      break;
    }
  }

  self->command_count = 0;
  self->view_shape[0] = (Py_ssize_t)self->count;
}

void clear_command_queue(PhysicsWorldObject *self) {
    if (!self->command_queue) return;

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