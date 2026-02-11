#include "culverin_character.h"
#include "culverin.h"
#include "culverin_filters.h"
#include "culverin_physics_world_internal.h"

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

// NEW: GC Traverse/Clear for Character
int Character_traverse(CharacterObject *self, visitproc visit,
                              void *arg) {
  Py_VISIT(self->world);
  return 0;
}
int Character_clear(CharacterObject *self) {
  Py_CLEAR(self->world);
  return 0;
}

void Character_dealloc(CharacterObject *self) {
  PyObject_GC_UnTrack(self);

  if (!self->world) {
    goto finalize;
  }
  // We must wait for the physics step to finish before touching the Jolt Character.
  // Otherwise, Jolt might be using this pointer in a worker thread.
  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);
  SHADOW_UNLOCK(&self->world->shadow_lock);

  // 1. REMOVE FROM JOLT MANAGER (Unlocked)
  // This is safe because the manager is thread-safe for removal.
  if (self->world->char_vs_char_manager && self->character) {
    JPH_CharacterVsCharacterCollisionSimple_RemoveCharacter(
        self->world->char_vs_char_manager, self->character);
  }

  // 2. WORLD REGISTRY CLEANUP (Locked)
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  SHADOW_LOCK(&self->world->shadow_lock);
  // Re-check stepping just to be paranoid/consistent, though strictly we handled it above. It costs nothing.
  BLOCK_UNTIL_NOT_STEPPING(self->world); 

  // GUARD: We must wait for queries to finish. Dealloc cannot return error,
  // so we must block. Since queries are fast, this is a very short wait.
  BLOCK_UNTIL_NOT_QUERYING(self->world);

  world_remove_body_slot(self->world, slot);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  // 3. JOLT RESOURCE DESTRUCTION (Unlocked)
  // No lock held here to avoid AB-BA deadlock with contact callbacks
  if (self->character) {
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)self->character);
  }
  if (self->listener) {
    JPH_CharacterContactListener_Destroy(self->listener);
  }
  if (self->body_filter) {
    JPH_BodyFilter_Destroy(self->body_filter);
  }
  if (self->shape_filter) {
    JPH_ShapeFilter_Destroy(self->shape_filter);
  }
  if (self->bp_filter) {
    JPH_BroadPhaseLayerFilter_Destroy(self->bp_filter);
  }
  if (self->obj_filter) {
    JPH_ObjectLayerFilter_Destroy(self->obj_filter);
  }

finalize:
  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Helper 1: Jolt-side allocation and Collision Manager linking
static JPH_CharacterVirtual *alloc_j_char(PhysicsWorldObject *self, 
                                          PositionVector pos,
                                          CharacterParams params) { // Reduced to 2 conceptual arguments
    
    // Position parameters are now accessed via pos.px, pos.py, pos.pz
    // Size parameters are now accessed via params.height, params.radius, etc.
    
    float half_h = fmaxf((params.height - 2.0f * params.radius) * 0.5f, 0.1f);
    JPH_CapsuleShapeSettings *ss =
        JPH_CapsuleShapeSettings_Create(half_h, params.radius);
    JPH_Shape *shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(ss);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);
    if (!shape) return NULL;

    JPH_CharacterVirtualSettings settings;
    JPH_CharacterVirtualSettings_Init(&settings);
    settings.base.shape = shape;
    settings.base.maxSlopeAngle = params.max_slope * (JPH_M_PI / 180.0f);

    JPH_CharacterVirtual *j_char = JPH_CharacterVirtual_Create(
        &settings, 
        &(JPH_RVec3){(double)pos.px, (double)pos.py, (double)pos.pz},
        &(JPH_Quat){0, 0, 0, 1}, 1, self->system);

    JPH_Shape_Destroy(shape);
    if (!j_char) return NULL;

    if (self->char_vs_char_manager) {
        JPH_CharacterVsCharacterCollisionSimple_AddCharacter(
            self->char_vs_char_manager, j_char);
        JPH_CharacterVirtual_SetCharacterVsCharacterCollision(
            j_char, self->char_vs_char_manager);
    }
    return j_char;
}

// Helper 2: Shadow Buffer Registration (Atomic Commit)
static void register_char(PhysicsWorldObject *self, CharacterObject *obj,
                          JPH_CharacterVirtual *j_char, uint32_t slot) {
  SHADOW_LOCK(&self->shadow_lock);

  BodyHandle h = make_handle(slot, self->generations[slot]);
  obj->handle = h;

  uint32_t dense_idx = (uint32_t)self->count;
  JPH_BodyID bid = JPH_CharacterVirtual_GetInnerBodyID(j_char);
  uint32_t j_idx = JPH_ID_TO_INDEX(bid);

  if (j_idx < self->max_jolt_bodies) {
    self->id_to_handle_map[j_idx] = h;
  }

  self->body_ids[dense_idx] = bid;
  self->slot_to_dense[slot] = dense_idx;
  self->dense_to_slot[dense_idx] = slot;
  self->slot_states[slot] = SLOT_ALIVE;
  self->user_data[dense_idx] = 0;
  self->count++;
  self->view_shape[0] = (Py_ssize_t)self->count;

  JPH_BodyInterface_SetUserData(self->body_interface, bid, (uint64_t)h);
  SHADOW_UNLOCK(&self->shadow_lock);
}

// Helper 3: Filter and Listener serialization (Trampoline Lock)
static void setup_char_filters(CharacterObject *obj) {
  SHADOW_LOCK(&g_jph_trampoline_lock);
  JPH_CharacterContactListener_SetProcs(&char_listener_procs);
  obj->listener = JPH_CharacterContactListener_Create(obj);
  JPH_BodyFilter_SetProcs(&global_bf_procs);
  obj->body_filter = JPH_BodyFilter_Create(NULL);
  JPH_ShapeFilter_SetProcs(&global_sf_procs);
  obj->shape_filter = JPH_ShapeFilter_Create(NULL);
  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  JPH_CharacterVirtual_SetListener(obj->character, obj->listener);
  obj->bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  obj->obj_filter = JPH_ObjectLayerFilter_Create(NULL);
}

// Main Orchestrator
PyObject *PhysicsWorld_create_character(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  float px = 0, py = 0, pz = 0, height = 1.8f, radius = 0.4f, step_h = 0.4f,
        slope = 45.0f;
  static char *kwlist[] = {"pos",         "height",    "radius",
                           "step_height", "max_slope", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)|ffff", kwlist, &px, &py,
                                   &pz, &height, &radius, &step_h, &slope))
    return NULL;

  // 1. Slot Reservation
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);
  if (self->free_count == 0 &&
      PhysicsWorld_resize(self, self->capacity * 2) < 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return NULL;
  }
  uint32_t char_slot = self->free_slots[--self->free_count];
  self->slot_states[char_slot] = SLOT_PENDING_CREATE;
  SHADOW_UNLOCK(&self->shadow_lock);

  // 2. Resource Allocation
  PositionVector pos_vec = {px, py, pz};
  CharacterParams char_params = {height, radius, slope};

  JPH_CharacterVirtual *j_char = alloc_j_char(self, pos_vec, char_params);
  if (!j_char) 
    goto fail_jolt;

  CharacterObject *obj = (CharacterObject *)PyObject_GC_New(
      CharacterObject,
      (PyTypeObject *)get_culverin_state(PyType_GetModule(Py_TYPE(self)))
          ->CharacterType);
  if (!obj)
    goto fail_py;

  // 3. Initialization
  obj->world = (PhysicsWorldObject *)Py_NewRef(self);
  obj->character = j_char;
  atomic_store(&obj->push_strength, 200.0f);
  atomic_store(&obj->last_vx, 0.0f);
  atomic_store(&obj->last_vy, 0.0f);
  atomic_store(&obj->last_vz, 0.0f);
  obj->prev_px = px;
  obj->prev_py = py;
  obj->prev_pz = pz;
  obj->prev_rx = 0.0f;
  obj->prev_ry = 0.0f;
  obj->prev_rz = 0.0f;
  obj->prev_rw = 1.0f;
  obj->listener = NULL;
  obj->body_filter = NULL;
  obj->shape_filter = NULL;
  obj->bp_filter = NULL;
  obj->obj_filter = NULL;

  // 4. Registration & Filter Setup
  register_char(self, obj, j_char, char_slot);
  setup_char_filters(obj);

  PyObject_GC_Track((PyObject *)obj);
  return (PyObject *)obj;

fail_py:
  JPH_CharacterBase_Destroy((JPH_CharacterBase *)j_char);
fail_jolt:
  SHADOW_LOCK(&self->shadow_lock);
  self->slot_states[char_slot] = SLOT_EMPTY;
  self->free_slots[self->free_count++] = char_slot;
  SHADOW_UNLOCK(&self->shadow_lock);
  return NULL;
}