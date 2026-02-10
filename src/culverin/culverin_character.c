#include "culverin.h"

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL
char_on_contact_validate(void *userData, const JPH_CharacterVirtual *character,
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
  return true; // Usually true, unless you want to walk through certain bodies
}

static void record_character_contact(CharacterObject *self, JPH_BodyID bodyID2, 
                                     const JPH_RVec3 *pos, const JPH_Vec3 *norm, 
                                     ContactEventType type) {
    PhysicsWorldObject *world = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    BodyHandle h2 = 0;
    
    if (world->id_to_handle_map && j_idx < world->max_jolt_bodies) {
        h2 = world->id_to_handle_map[j_idx];
    }
    if (h2 == 0) return; // Ignore unmapped bodies (like internal Jolt helpers)

    BodyHandle h1 = self->handle;
    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = (uint32_t)type;
        
        // Consistent ordering for Python set logic
        if (h1 < h2) { ev->body1 = h1; ev->body2 = h2; }
        else { ev->body1 = h2; ev->body2 = h1; }

        ev->px = (float)pos->x; ev->py = (float)pos->y; ev->pz = (float)pos->z;
        ev->nx = norm->x; ev->ny = norm->y; ev->nz = norm->z;
        ev->impulse = 1.0f; // Logical trigger strength
        ev->sliding_speed_sq = 0.0f;
        
        // Look up material of the object we hit
        uint32_t slot2 = (uint32_t)(h2 & 0xFFFFFFFF);
        uint32_t dense2 = world->slot_to_dense[slot2];
        ev->mat1 = 0; // Characters don't have materials yet
        ev->mat2 = world->material_ids[dense2];

        atomic_thread_fence(memory_order_release);
    }
}

static void report_char_vs_char(CharacterObject *self, const JPH_CharacterVirtual *other, 
                                const JPH_Vec3 *normal, const JPH_RVec3 *pos, 
                                ContactEventType type) {
    PhysicsWorldObject *world = self->world;
    BodyHandle h1 = self->handle;
    
    // 1. Get Inner Body ID
    JPH_BodyID other_bid = JPH_CharacterVirtual_GetInnerBodyID(other);
    
    // 2. Direct Jolt Lookup (Bypasses our map, which might be too small for Virtual IDs)
    uint64_t userdata = JPH_BodyInterface_GetUserData(world->body_interface, other_bid);
    BodyHandle h2 = (BodyHandle)userdata;
    
    if (h2 == 0) return; // Not a known Culverin object

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = (uint32_t)type;
        
        // Canonicalize Order
        if (h1 < h2) { ev->body1 = h1; ev->body2 = h2; } 
        else { ev->body1 = h2; ev->body2 = h1; }

        ev->sliding_speed_sq = 0.0f;
        ev->nx = normal->x; ev->ny = normal->y; ev->nz = normal->z;
        ev->px = (float)pos->x; ev->py = (float)pos->y; ev->pz = (float)pos->z;
        ev->impulse = 1.0f;
        ev->mat1 = 0; ev->mat2 = 0; 

        atomic_thread_fence(memory_order_release);
    }
}
static void JPH_API_CALL char_on_character_contact_added(
    void *userData, const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, JPH_SubShapeID subShapeID2,
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {
    
    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;
    
    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;
    
    report_char_vs_char(self, otherCharacter, contactNormal, contactPosition, EVENT_ADDED);
}

static void apply_character_impulse(CharacterObject *self, JPH_BodyID bodyID2, const JPH_Vec3 *contactNormal) {
    // 1. Thread-Safe Member Access
    float vx = atomic_load_explicit((&self->last_vx), memory_order_relaxed);
    float vy = atomic_load_explicit((&self->last_vy), memory_order_relaxed);
    float vz = atomic_load_explicit((&self->last_vz), memory_order_relaxed);
    float strength = atomic_load_explicit((&self->push_strength), memory_order_relaxed);

    JPH_BodyInterface *bi = self->world->body_interface;

    // 2. Ignore Sensors & Non-Dynamic Bodies
    if (JPH_BodyInterface_IsSensor(bi, bodyID2) ||
        JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
        return;
    }

    // 3. Calculate Pushing Force
    float dot = vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

    if (dot > 0.1f) {
        float factor = dot * strength;
        const float max_impulse = 5000.0f;
        if (factor > max_impulse) factor = max_impulse;

        JPH_Vec3 impulse;
        impulse.x = contactNormal->x * factor;
        
        // Flatten Y Response (allow kicking up, suppress crushing down)
        float y_push = contactNormal->y * factor;
        impulse.y = (y_push > 0.0f) ? y_push : 0.0f;
        
        impulse.z = contactNormal->z * factor;

        JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
        JPH_BodyInterface_ActivateBody(bi, bodyID2);
    }
}

// --- Updated Added Callback ---
static void JPH_API_CALL char_on_contact_added(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  CharacterObject *self = (CharacterObject *)userData;
  if (!self) return;

  // Record Event
  record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_ADDED);

  // Apply Impulse
  apply_character_impulse(self, bodyID2, contactNormal);
}


static void JPH_API_CALL char_on_contact_persisted(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;

    CharacterObject *self = (CharacterObject *)userData;
    if (!self) return;

    // Record Event
    record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_PERSISTED);

    // Apply Impulse (CRITICAL FIX)
    apply_character_impulse(self, bodyID2, contactNormal);
}

static void JPH_API_CALL char_on_contact_removed(void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;

    PhysicsWorldObject *world = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    
    BodyHandle h1 = self->handle;
    BodyHandle h2 = 0;
    if (world->id_to_handle_map && j_idx < world->max_jolt_bodies) {
        h2 = world->id_to_handle_map[j_idx];
    }
    if (h2 == 0) return;

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = EVENT_REMOVED;
        ev->body1 = (h1 < h2) ? h1 : h2;
        ev->body2 = (h1 < h2) ? h2 : h1;
        // Geometry is zeroed for removal
        memset(&ev->px, 0, sizeof(float) * 8); 
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_character_contact_persisted(
    void *userData, const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, JPH_SubShapeID subShapeID2,
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;

    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;

    report_char_vs_char(self, otherCharacter, contactNormal, contactPosition, EVENT_PERSISTED);
}

static void JPH_API_CALL char_on_character_contact_removed(void *userData, const JPH_CharacterVirtual *character, const JPH_CharacterID otherCharacterID, JPH_SubShapeID subShapeID2) {
    CharacterObject *self = (CharacterObject *)userData;
    PhysicsWorldObject *world = self->world;

    BodyHandle h1 = self->handle;
    // We have to use the CharacterID to find the handle. 
    // Jolt CharacterIDs usually map to the inner BodyID.
    uint32_t j_idx = JPH_ID_TO_INDEX(otherCharacterID);
    BodyHandle h2 = world->id_to_handle_map[j_idx];

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = EVENT_REMOVED;
        ev->body1 = (h1 < h2) ? h1 : h2;
        ev->body2 = (h1 < h2) ? h2 : h1;
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_adjust_velocity(
    void *userData, const JPH_CharacterVirtual *character, const JPH_Body *body2, 
    JPH_Vec3 *ioLinearVelocity, JPH_Vec3 *ioAngularVelocity) {
    
    // Usually, we want the default behavior (character follows the body).
    // TODO: add logic here if you want the character to "slip" on certain materials.
}

// Map the procs
const JPH_CharacterContactListener_Procs char_listener_procs = {
    .OnContactValidate = char_on_contact_validate,
    .OnContactAdded = char_on_contact_added,
    .OnAdjustBodyVelocity = char_on_adjust_velocity, // ADDED
    .OnContactPersisted = char_on_contact_persisted, // CHANGED from char_on_contact_added
    .OnContactRemoved = char_on_contact_removed,      // ADDED
    .OnCharacterContactValidate = NULL,               // Default True is fine
    .OnCharacterContactAdded = char_on_character_contact_added,
    .OnCharacterContactPersisted = char_on_character_contact_persisted, // ADDED
    .OnCharacterContactRemoved = char_on_character_contact_removed,     // ADDED
    .OnContactSolve = NULL                            // Advanced, keep NULL
};

PyObject *Character_move(CharacterObject *self, PyObject *args,
                                PyObject *kwds) {
  float vx = 0;
  float vy = 0;
  float vz = 0;
  float dt = 0;
  static char *kwlist[] = {"velocity", "dt", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &vx, &vy, &vz,
                                   &dt)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);

  // 1. RE-ENTRANCY GUARD
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  // 2. ATOMIC INPUT STORAGE
  // Satisfies the memory model for the lock-free contact callbacks
  atomic_store_explicit(&self->last_vx, vx, memory_order_relaxed);
  atomic_store_explicit(&self->last_vy, vy, memory_order_relaxed);
  atomic_store_explicit(&self->last_vz, vz, memory_order_relaxed);

  // 3. CAPTURE PRE-MOVE STATE (For Interpolation)
  JPH_STACK_ALLOC(JPH_RVec3, current_pos);
  JPH_STACK_ALLOC(JPH_Quat, current_rot);
  JPH_CharacterVirtual_GetPosition(self->character, current_pos);
  JPH_CharacterVirtual_GetRotation(self->character, current_rot);

  // Update object-local prev state
  self->prev_px = (float)current_pos->x;
  self->prev_py = (float)current_pos->y;
  self->prev_pz = (float)current_pos->z;
  self->prev_rx = current_rot->x;
  self->prev_ry = current_rot->y;
  self->prev_rz = current_rot->z;
  self->prev_rw = current_rot->w;

  // Sync to GLOBAL shadow buffers so get_render_state() works
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  uint32_t dense_idx = self->world->slot_to_dense[slot];
  size_t off = (size_t)dense_idx * 4;

  memcpy(&self->world->prev_positions[off], &self->world->positions[off],
         12); // Copy x,y,z
  memcpy(&self->world->prev_rotations[off], &self->world->rotations[off], 16);

  // --- 4. JOLT EXECUTION ---
  JPH_Vec3 v = {vx, vy, vz};
  JPH_CharacterVirtual_SetLinearVelocity(self->character, &v);

  JPH_STACK_ALLOC(JPH_ExtendedUpdateSettings, update_settings);

  // 1. Zero out the memory first
  memset(update_settings, 0, sizeof(JPH_ExtendedUpdateSettings));

  // 2. Set Jolt's standard defaults (these are non-zero in the C++ core)
  update_settings->stickToFloorStepDown.x = 0.0f;
  update_settings->stickToFloorStepDown.y = -0.5f; // How far to look for floor
  update_settings->stickToFloorStepDown.z = 0.0f;

  update_settings->walkStairsStepUp.x = 0.0f;
  update_settings->walkStairsStepUp.y = 0.4f; // Maximum step height
  update_settings->walkStairsStepUp.z = 0.0f;

  update_settings->walkStairsMinStepForward = 0.02f;
  update_settings->walkStairsStepForwardTest = 0.15f;
  update_settings->walkStairsCosAngleForwardContact = 0.996f; // ~5 degrees

  update_settings->walkStairsStepDownExtra.x = 0.0f;
  update_settings->walkStairsStepDownExtra.y = 0.0f;
  update_settings->walkStairsStepDownExtra.z = 0.0f;

  // Now execute
  JPH_CharacterVirtual_ExtendedUpdate(self->character, dt, update_settings, 1,
                                      self->world->system, self->body_filter,
                                      self->shape_filter);

  // 5. POST-MOVE SYNC
  // Retrieve the final position after collision resolution
  JPH_CharacterVirtual_GetPosition(self->character, current_pos);
  JPH_CharacterVirtual_GetRotation(self->character, current_rot);

  // Update current positions for queries and rendering
  self->world->positions[off + 0] = (float)current_pos->x;
  self->world->positions[off + 1] = (float)current_pos->y;
  self->world->positions[off + 2] = (float)current_pos->z;

  self->world->rotations[off + 0] = current_rot->x;
  self->world->rotations[off + 1] = current_rot->y;
  self->world->rotations[off + 2] = current_rot->z;
  self->world->rotations[off + 3] = current_rot->w;

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

PyObject *Character_get_position(CharacterObject *self,
                                        PyObject *Py_UNUSED(ignored)) {
  // 1. Aligned stack storage for SIMD
  JPH_STACK_ALLOC(JPH_RVec3, pos);

  // 2. Lock for consistency (ensure we aren't reading mid-step)
  SHADOW_LOCK(&self->world->shadow_lock);
  JPH_CharacterVirtual_GetPosition(self->character, pos);
  SHADOW_UNLOCK(&self->world->shadow_lock);

  PyObject *ret = PyTuple_New(3);
  if (!ret) {
    return NULL;
  }

  // Use the double precision provided by RVec3
  PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble(pos->x));
  PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble(pos->y));
  PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(pos->z));

  return ret;
}

PyObject *Character_set_position(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  static char *kwlist[] = {"pos", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)", kwlist, &x, &y, &z)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  // 1. Update Jolt (Aligned)
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)x;
  pos->y = (double)y;
  pos->z = (double)z;
  JPH_CharacterVirtual_SetPosition(self->character, pos);

  // 2. Update Shadow Buffers (Reset Interpolation to prevent streaks)
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  uint32_t dense_idx = self->world->slot_to_dense[slot];
  size_t off = (size_t)dense_idx * 4;

  self->world->positions[off + 0] = x;
  self->world->positions[off + 1] = y;
  self->world->positions[off + 2] = z;

  self->world->prev_positions[off + 0] = x;
  self->world->prev_positions[off + 1] = y;
  self->world->prev_positions[off + 2] = z;

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

PyObject *Character_set_rotation(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
  static char *kwlist[] = {"rot", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ffff)", kwlist, &x, &y, &z,
                                   &w)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  // 1. Update Jolt
  JPH_STACK_ALLOC(JPH_Quat, q);
  q->x = x;
  q->y = y;
  q->z = z;
  q->w = w;
  JPH_CharacterVirtual_SetRotation(self->character, q);

  // 2. Update Shadow Buffers
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  uint32_t dense_idx = self->world->slot_to_dense[slot];
  size_t off = (size_t)dense_idx * 4;

  memcpy(&self->world->rotations[off], q, 16);
  memcpy(&self->world->prev_rotations[off], q, 16);

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

PyObject *Character_is_grounded(CharacterObject *self, PyObject *args) {
  SHADOW_LOCK(&self->world->shadow_lock);
  // No need for GUARD_STEPPING here as it's a non-destructive status check,
  // but holding the lock ensures the character hasn't been deallocated.
  JPH_GroundState state =
      JPH_CharacterBase_GetGroundState((JPH_CharacterBase *)self->character);
  SHADOW_UNLOCK(&self->world->shadow_lock);

  if (state == JPH_GroundState_OnGround ||
      state == JPH_GroundState_OnSteepGround) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject *Character_set_strength(CharacterObject *self, PyObject *args) {
  float strength = 0.0f;
  if (!PyArg_ParseTuple(args, "f", &strength)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);
  BLOCK_UNTIL_NOT_QUERYING(self->world);

  // 1. Update Atomic for Jolt worker threads
  atomic_store_explicit(&self->push_strength, strength, memory_order_relaxed);

  // 2. Update Jolt internal state
  JPH_CharacterVirtual_SetMaxStrength(self->character, strength);

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

// Change signature to take PyObject* arg directly
PyObject *Character_get_render_transform(CharacterObject *self,
                                                PyObject *arg) {
  // --- OPTIMIZATION 1: Fast Argument Parsing ---
  double alpha_dbl = PyFloat_AsDouble(arg);
  if (alpha_dbl == -1.0 && PyErr_Occurred()) {
    return NULL;
  }

  float alpha = (float)alpha_dbl;
  if (alpha < 0.0f) {
    alpha = 0.0f;
  } else if (alpha > 1.0f) {
    alpha = 1.0f;
  }

  // --- 1. ALIGNED STACK ALLOCATION ---
  // Mandatory for SIMD safety
  JPH_STACK_ALLOC(JPH_RVec3, cur_p);
  JPH_STACK_ALLOC(JPH_Quat, cur_r);

  // --- 2. CONSISTENT SNAPSHOT (Locked) ---
  SHADOW_LOCK(&self->world->shadow_lock);

  // We snapshot both the 'prev' state and 'current' state together
  float p_px = self->prev_px;
  float p_py = self->prev_py;
  float p_pz = self->prev_pz;
  float p_rx = self->prev_rx;
  float p_ry = self->prev_ry;
  float p_rz = self->prev_rz;
  float p_rw = self->prev_rw;

  JPH_CharacterVirtual_GetPosition(self->character, cur_p);
  JPH_CharacterVirtual_GetRotation(self->character, cur_r);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  // --- 3. MATH (Unlocked) ---
  // Position LERP
  float px = p_px + ((float)cur_p->x - p_px) * alpha;
  float py = p_py + ((float)cur_p->y - p_py) * alpha;
  float pz = p_pz + ((float)cur_p->z - p_pz) * alpha;

  // Rotation NLERP
  float dot =
      p_rx * cur_r->x + p_ry * cur_r->y + p_rz * cur_r->z + p_rw * cur_r->w;
  float q2x = cur_r->x;
  float q2y = cur_r->y;
  float q2z = cur_r->z;
  float q2w = cur_r->w;
  if (dot < 0.0f) {
    q2x = -q2x;
    q2y = -q2y;
    q2z = -q2z;
    q2w = -q2w;
  }

  float rx = p_rx + (q2x - p_rx) * alpha;
  float ry = p_ry + (q2y - p_ry) * alpha;
  float rz = p_rz + (q2z - p_rz) * alpha;
  float rw = p_rw + (q2w - p_rw) * alpha;

  float mag_sq = rx * rx + ry * ry + rz * rz + rw * rw;
  if (mag_sq > 1e-9f) {
    float inv_len = 1.0f / sqrtf(mag_sq);
    rx *= inv_len;
    ry *= inv_len;
    rz *= inv_len;
    rw *= inv_len;
  } else {
    rx = 0.0f;
    ry = 0.0f;
    rz = 0.0f;
    rw = 1.0f;
  }

  // --- 4. HARDENED MANUAL CONSTRUCTION ---
  PyObject *pos = PyTuple_New(3);
  PyObject *rot = PyTuple_New(4);
  PyObject *out = PyTuple_New(2);

  if (!pos || !rot || !out) {
    Py_XDECREF(pos);
    Py_XDECREF(rot);
    Py_XDECREF(out);
    return PyErr_NoMemory();
  }

  // Safely fill tuples (PyFloat_FromDouble can fail, but unlikely)
  // If these return NULL, the whole return 'out' will be cleaned up by the
  // user's logic
  PyTuple_SET_ITEM(pos, 0, PyFloat_FromDouble(px));
  PyTuple_SET_ITEM(pos, 1, PyFloat_FromDouble(py));
  PyTuple_SET_ITEM(pos, 2, PyFloat_FromDouble(pz));

  PyTuple_SET_ITEM(rot, 0, PyFloat_FromDouble(rx));
  PyTuple_SET_ITEM(rot, 1, PyFloat_FromDouble(ry));
  PyTuple_SET_ITEM(rot, 2, PyFloat_FromDouble(rz));
  PyTuple_SET_ITEM(rot, 3, PyFloat_FromDouble(rw));

  PyTuple_SET_ITEM(out, 0, pos);
  PyTuple_SET_ITEM(out, 1, rot);

  return out;
}