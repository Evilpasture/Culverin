#include "culverin.h"

// Global lock for JPH callbacks
static ShadowMutex
    g_jph_trampoline_lock; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// --- Handle Helper ---
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
  return ((uint64_t)gen << 32) | (uint64_t)slot;
}

static inline bool unpack_handle(PhysicsWorldObject *self, BodyHandle h,
                                 uint32_t *slot) {
  *slot = (uint32_t)(h & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(h >> 32);

  if (*slot >= self->slot_capacity) {
    return false;
  }
  return self->generations[*slot] == gen;
}

// --- Debug Buffer Helpers ---
static void debug_buffer_ensure(DebugBuffer* buf, size_t count_needed) {
    if (buf->count + count_needed > buf->capacity) {
        size_t new_cap = (buf->capacity == 0) ? 4096 : buf->capacity * 2;
        while (buf->count + count_needed > new_cap) new_cap *= 2;
        
        void* new_ptr = PyMem_RawRealloc(buf->data, new_cap * sizeof(DebugVertex));
        if (!new_ptr) return; // Silent fail on OOM for debug info
        
        buf->data = (DebugVertex*)new_ptr;
        buf->capacity = new_cap;
    }
}

static void debug_buffer_push(DebugBuffer* buf, DebugCoordinates pos, uint32_t color) {
    if (buf->count >= buf->capacity) return; // Safety
    buf->data[buf->count].x = pos.x;
    buf->data[buf->count].y = pos.y;
    buf->data[buf->count].z = pos.z;
    buf->data[buf->count].color = color;
    buf->count++;
}

static void debug_buffer_free(DebugBuffer* buf) {
    if (buf->data) PyMem_RawFree(buf->data);
    buf->data = NULL;
    buf->count = 0;
    buf->capacity = 0;
}

/**
 * Internal helper to remove a body from the dense arrays.
 * Maintains a packed, contiguous array by swapping the last body into the hole.
 * MUST be called while holding SHADOW_LOCK.
 */
static void world_remove_body_slot(PhysicsWorldObject *self, uint32_t slot) {
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

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL
char_on_contact_validate(void *userData, const JPH_CharacterVirtual *character,
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
  return true; // Usually true, unless you want to walk through certain bodies
}

static JPH_ValidateResult JPH_API_CALL on_contact_validate(
    void *userData, const JPH_Body *body1, const JPH_Body *body2,
    const JPH_RVec3 *baseOffset, const JPH_CollideShapeResult *result) {
  PhysicsWorldObject *self = (PhysicsWorldObject *)userData;

  // 1. Extract Slots
  BodyHandle h1 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body1);
  BodyHandle h2 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body2);
  uint32_t slot1 = (uint32_t)(h1 & 0xFFFFFFFF);
  uint32_t slot2 = (uint32_t)(h2 & 0xFFFFFFFF);

  // 2. Bitmask Filter
  uint32_t idx1 = self->slot_to_dense[slot1];
  uint32_t idx2 = self->slot_to_dense[slot2];

  uint32_t cat1 = self->categories[idx1];
  uint32_t mask1 = self->masks[idx1];
  uint32_t cat2 = self->categories[idx2];
  uint32_t mask2 = self->masks[idx2];

  // 3. Logic: If either doesn't want to hit the other's category, reject.
  if (!(cat1 & mask2) || !(cat2 & mask1)) {
    return JPH_ValidateResult_RejectContact; // <--- This stops the PHYSICS
                                             // solver
  }

  return JPH_ValidateResult_AcceptContact;
}

// --- Query Filters ---
static bool JPH_API_CALL filter_allow_all_bp(void *userData,
                                             JPH_BroadPhaseLayer layer) {
  return true; // Allow ray to see all broadphase regions
}
static bool JPH_API_CALL filter_allow_all_obj(void *userData,
                                              JPH_ObjectLayer layer) {
  return true; // Allow ray to see all object layers (0 and 1)
}

static bool JPH_API_CALL filter_true_body(void *userData, JPH_BodyID bodyID) {
  return true;
}
static bool JPH_API_CALL filter_true_shape(void *userData,
                                           const JPH_Shape *shape,
                                           const JPH_SubShapeID *id) {
  return true;
}

static const JPH_BodyFilter_Procs global_bf_procs = {.ShouldCollide =
                                                         filter_true_body};
static const JPH_ShapeFilter_Procs global_sf_procs = {.ShouldCollide =
                                                          filter_true_shape};

static void JPH_API_CALL char_on_character_contact_added(
//NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void *userData, const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, JPH_SubShapeID subShapeID2,
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {

  // 1. Resolve Self
  CharacterObject *self = (CharacterObject *)userData;
  if (!self || !self->world) {
    return;
  }
  PhysicsWorldObject *world = self->world;

  // 2. Define Physics Interaction
  // canPushCharacter: Allows 'character' to push 'otherCharacter'
  // canReceiveImpulses: Allows the characters to exchange momentum
  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  // 3. GENERATIONAL EVENT REPORTING (Lock-Free)
  // Character-vs-Character collisions are handled in a special pass by Jolt.
  // To make them show up in Python's get_contact_events(), we record them
  // manually.

  // ID 1: Our own immutable handle
  BodyHandle h1 = self->handle;

  // ID 2: Retrieve the handle of the other character.
  // We get the inner BodyID, then read the 'Handle' we stamped into its
  // UserData.
  JPH_BodyID other_bid = JPH_CharacterVirtual_GetInnerBodyID(otherCharacter);
  BodyHandle h2 = (BodyHandle)JPH_BodyInterface_GetUserData(
      world->body_interface, other_bid);

  // 4. Atomic Reservation in the Global Event Buffer
  size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1,
                                         memory_order_relaxed);

  if (idx < world->contact_max_capacity) {
    ContactEvent *ev = &world->contact_buffer[idx];
    ev->type = EVENT_ADDED;
    // Ensure consistent ordering (h1 < h2) to avoid duplicate entries
    // if both characters are being updated in the same step.
    if (h1 < h2) {
      ev->body1 = h1;
      ev->body2 = h2;
    } else {
      ev->body1 = h2;
      ev->body2 = h1;
    }

    ev->sliding_speed_sq =
        0.0f; // No tangential speed for character-character contacts, for now.

    // normal points from Character to otherCharacter
    ev->nx = contactNormal->x;
    ev->ny = contactNormal->y;
    ev->nz = contactNormal->z;

    ev->px = (float)contactPosition->x;
    ev->py = (float)contactPosition->y;
    ev->pz = (float)contactPosition->z;

    // CharacterVirtual doesn't easily provide mass-based closing impulses here,
    // so we provide a "Contact Strength" of 0 or 1 for logical triggers.
    ev->impulse = 1.0f;

    // 5. Release Fence
    // Synchronizes this write with the Python thread's 'acquire' in
    // get_contact_events
    atomic_thread_fence(memory_order_release);
  }
}

// Callback: Handle the collision settings AND Apply Impulse
static void JPH_API_CALL char_on_contact_added(
//NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

  // 1. Safe Defaults
  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  // 2. Resolve Character Object
  if (!userData) {
    return;
  }
  CharacterObject *self = (CharacterObject *)userData;

  // 3. Thread-Safe Member Access
  // We use relaxed atomics here. If the main thread is mid-write,
  // we just want 'a' valid value, we don't need strict synchronization.
  float vx = atomic_load_explicit((&self->last_vx), memory_order_relaxed);
  float vy = atomic_load_explicit((&self->last_vy), memory_order_relaxed);
  float vz = atomic_load_explicit((&self->last_vz), memory_order_relaxed);
  float strength =
      atomic_load_explicit((&self->push_strength), memory_order_relaxed);

  // 4. Jolt Interface (Safe to call from worker threads)
  JPH_BodyInterface *bi = self->world->body_interface;

  // 5. Ignore Sensors & Non-Dynamic Bodies
  // These calls are safe because Jolt's internal locks protect the BodyManager
  if (JPH_BodyInterface_IsSensor(bi, bodyID2) ||
      JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
    ioSettings->canPushCharacter = false;
    ioSettings->canReceiveImpulses = false;
    return;
  }

  // 6. Calculate Pushing Force
  // contactNormal points from Character to Body2
  float dot =
      vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

  // Threshold prevents micro-jitter and pushing objects just by grazing them
  if (dot > 0.1f) {
    float factor = dot * strength;

    // Safety Cap: Prevent "Physics Nukes" from velocity spikes
    const float max_impulse = 5000.0f;
    if (factor > max_impulse) {
      factor = max_impulse;
    }

    JPH_Vec3 impulse;
    impulse.x = contactNormal->x * factor;

    // Flatten Y Response:
    // We allow upward pushes (kicking an object up),
    // but we suppress downward pushes (preventing the character from
    // crushing dynamic floors or applying massive gravity-doubling forces).
    float y_push = contactNormal->y * factor;
    impulse.y = (y_push > 0.0f) ? y_push : 0.0f;

    impulse.z = contactNormal->z * factor;

    // 7. Apply to Jolt
    // This is a thread-safe call into Jolt's internal command queue/locking
    // system
    JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
    JPH_BodyInterface_ActivateBody(bi, bodyID2);
  }
}

// Helper to find an arbitrary vector perpendicular to 'in'
static void vec3_get_perpendicular(const JPH_Vec3 *in, JPH_Vec3 *out) {
  if (fabsf(in->x) > fabsf(in->z)) {
    out->x = -in->y;
    out->y = in->x;
    out->z = 0.0f; // Cross(in, Z)
  } else {
    out->x = 0.0f;
    out->y = -in->z;
    out->z = in->y; // Cross(in, X)
  }
  // Normalize
  float len = sqrtf(out->x * out->x + out->y * out->y + out->z * out->z);
  if (len > 1e-6f) {
    float inv = 1.0f / len;
    out->x *= inv;
    out->y *= inv;
    out->z *= inv;
  } else {
    // Fallback if 'in' is zero
    out->x = 1.0f;
    out->y = 0.0f;
    out->z = 0.0f;
  }
}

// Helper to rotate a vector by a quaternion manually (v' = q * v * q^-1)
static inline void manual_vec3_rotate_by_quat(const JPH_Vec3 *v,
                                              const JPH_Quat *q,
                                              JPH_Vec3 *out) {
  float tx = 2.0f * (q->y * v->z - q->z * v->y);
  float ty = 2.0f * (q->z * v->x - q->x * v->z);
  float tz = 2.0f * (q->x * v->y - q->y * v->x);

  float cx = q->y * tz - q->z * ty;
  float cy = q->z * tx - q->x * tz;
  float cz = q->x * ty - q->y * tx;

  out->x = v->x + q->w * tx + cx;
  out->y = v->y + q->w * ty + cy;
  out->z = v->z + q->w * tz + cz;
}

// Helper for quaternion multiplication (q_out = q_a * q_b)
static inline void manual_quat_multiply(const JPH_Quat *a, const JPH_Quat *b,
                                        JPH_Quat *out) {
  out->x = a->w * b->x + a->x * b->w + a->y * b->z - a->z * b->y;
  out->y = a->w * b->y - a->x * b->z + a->y * b->w + a->z * b->x;
  out->z = a->w * b->z + a->x * b->y - a->y * b->x + a->z * b->w;
  out->w = a->w * b->w - a->x * b->x - a->y * b->y - a->z * b->z;
}

// --- Jolt Debug Callbacks ---
static void JPH_API_CALL OnDebugDrawLine(void *userData, const JPH_RVec3 *from,
                                         const JPH_RVec3 *to, JPH_Color color) {
  PhysicsWorldObject *self = (PhysicsWorldObject *)userData;
  debug_buffer_ensure(&self->debug_lines, 2);
  debug_buffer_push(
      &self->debug_lines,
      (DebugCoordinates){(float)from->x, (float)from->y, (float)from->z},
      color);
  debug_buffer_push(
      &self->debug_lines,
      (DebugCoordinates){(float)to->x, (float)to->y, (float)to->z}, color);
}

static void JPH_API_CALL
OnDebugDrawTriangle(void *userData, const JPH_RVec3 *v1, const JPH_RVec3 *v2,
//NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    const JPH_RVec3 *v3, JPH_Color color,
                    JPH_DebugRenderer_CastShadow castShadow) {
  PhysicsWorldObject *self = (PhysicsWorldObject *)userData;
  debug_buffer_ensure(&self->debug_triangles, 3);
  debug_buffer_push(
      &self->debug_triangles,
      (DebugCoordinates){(float)v1->x, (float)v1->y, (float)v1->z}, color);
  debug_buffer_push(
      &self->debug_triangles,
      (DebugCoordinates){(float)v2->x, (float)v2->y, (float)v2->z}, color);
  debug_buffer_push(
      &self->debug_triangles,
      (DebugCoordinates){(float)v3->x, (float)v3->y, (float)v3->z}, color);
}

static void JPH_API_CALL OnDebugDrawText(void *userData,
                                         const JPH_RVec3 *position,
//NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                         const char *str, JPH_Color color,
                                         float height) {
  // Text is hard to batch efficiently to Python bytes.
  // Usually ignored or printed to stdout.
}

static const JPH_DebugRenderer_Procs debug_procs = {
    .DrawLine = OnDebugDrawLine,
    .DrawTriangle = OnDebugDrawTriangle,
    .DrawText3D = OnDebugDrawText
};

// --- Internal Contact Helper ---
static void process_contact_manifold(PhysicsWorldObject *self, 
                                     const JPH_Body *body1, const JPH_Body *body2, 
                                     const JPH_ContactManifold *manifold,
                                     ContactEventType type) {
  
  // Fast Pointer-based UserData retrieval (No Jolt locks)
  BodyHandle h1 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body1);
  BodyHandle h2 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body2);

  uint32_t slot1 = (uint32_t)(h1 & 0xFFFFFFFF);
  uint32_t slot2 = (uint32_t)(h2 & 0xFFFFFFFF);

  // Safety: Ensure slot is within our shadow buffer range
  if (slot1 >= self->slot_capacity || slot2 >= self->slot_capacity) return;

  uint32_t idx1 = self->slot_to_dense[slot1];
  uint32_t idx2 = self->slot_to_dense[slot2];

  // Bitmask Filter
  if (!(self->categories[idx1] & self->masks[idx2]) || 
      !(self->categories[idx2] & self->masks[idx1])) return;

  size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);
  if (idx >= self->contact_max_capacity) return;

  ContactEvent *ev = &self->contact_buffer[idx];
  ev->type = (uint32_t)type;

  JPH_STACK_ALLOC(JPH_Vec3, n);
  JPH_ContactManifold_GetWorldSpaceNormal(manifold, n);

  bool swapped = (h1 > h2);
  if (!swapped) {
    ev->body1 = h1; ev->body2 = h2;
  } else {
    ev->body1 = h2; ev->body2 = h1;
    n->x = -n->x; n->y = -n->y; n->z = -n->z;
  }
  ev->nx = n->x; ev->ny = n->y; ev->nz = n->z;

  JPH_STACK_ALLOC(JPH_RVec3, p);
  JPH_ContactManifold_GetWorldSpaceContactPointOn1(manifold, 0, p);
  ev->px = (float)p->x; ev->py = (float)p->y; ev->pz = (float)p->z;

  // Impulse math skipped for sensors to prevent Static Body access violations
  if (JPH_Body_IsSensor((JPH_Body*)body1) || JPH_Body_IsSensor((JPH_Body*)body2)) {
      ev->impulse = 0.0f;
      ev->sliding_speed_sq = 0.0f;
  } else {
      JPH_Vec3 v1 = {0,0,0}, v2 = {0,0,0};
      if (JPH_Body_GetMotionType((JPH_Body*)body1) != JPH_MotionType_Static) 
          JPH_Body_GetLinearVelocity((JPH_Body*)body1, &v1);
      if (JPH_Body_GetMotionType((JPH_Body*)body2) != JPH_MotionType_Static) 
          JPH_Body_GetLinearVelocity((JPH_Body*)body2, &v2);

      float dvx = swapped ? (v2.x - v1.x) : (v1.x - v2.x);
      float dvy = swapped ? (v2.y - v1.y) : (v1.y - v2.y);
      float dvz = swapped ? (v2.z - v1.z) : (v1.z - v2.z);

      float dot = dvx * ev->nx + dvy * ev->ny + dvz * ev->nz;
      ev->impulse = fabsf(dot);
      ev->sliding_speed_sq = (dvx*dvx + dvy*dvy + dvz*dvz) - (dot*dot);
  }

  ev->mat1 = self->material_ids[idx1];
  ev->mat2 = self->material_ids[idx2];

  atomic_thread_fence(memory_order_release);
}

// --- Global Contact Listener ---
// 1. ADDED
static void JPH_API_CALL on_contact_added(void *userData, const JPH_Body *body1,
                                          const JPH_Body *body2,
                                          const JPH_ContactManifold *manifold,
                                          JPH_ContactSettings *settings) {
    process_contact_manifold((PhysicsWorldObject *)userData, body1, body2, manifold, EVENT_ADDED);
}

// 2. PERSISTED (Uses same helper, different type ID)
static void JPH_API_CALL on_contact_persisted(void *userData, const JPH_Body *body1,
                                              const JPH_Body *body2,
                                              const JPH_ContactManifold *manifold,
                                              JPH_ContactSettings *settings) {
    process_contact_manifold((PhysicsWorldObject *)userData, body1, body2, manifold, EVENT_PERSISTED);
}

// 3. REMOVED (Simpler logic, no manifold)
static void JPH_API_CALL on_contact_removed(void *userData, const JPH_SubShapeIDPair *pair) {
    PhysicsWorldObject *self = (PhysicsWorldObject *)userData;

    // Use indices from BodyIDs to look up handles in our private map
    uint32_t i1 = JPH_ID_TO_INDEX(pair->Body1ID);
    uint32_t i2 = JPH_ID_TO_INDEX(pair->Body2ID);

    BodyHandle h1 = 0, h2 = 0;
    if (self->id_to_handle_map) {
        if (i1 < self->max_jolt_bodies) h1 = self->id_to_handle_map[i1];
        if (i2 < self->max_jolt_bodies) h2 = self->id_to_handle_map[i2];
    }
    
    if (h1 == 0 || h2 == 0) return; // Ignore unmapped bodies

    size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx >= self->contact_max_capacity) return;

    ContactEvent *ev = &self->contact_buffer[idx];
    ev->type = EVENT_REMOVED;
    ev->body1 = (h1 < h2) ? h1 : h2;
    ev->body2 = (h1 < h2) ? h2 : h1;

    // Zero geometry for removal
    memset(&ev->px, 0, sizeof(float) * 8); 

    atomic_thread_fence(memory_order_release);
}

// Fixed get_contact_events to be safer with locking
static PyObject *PhysicsWorld_get_contact_events(PhysicsWorldObject *self,
                                                 PyObject *args) {
  // 1. Enter the lock to check the state machine
  SHADOW_LOCK(&self->shadow_lock);

  // GUARD: Ensure we aren't mid-step and prevent a new step from starting
  BLOCK_UNTIL_NOT_STEPPING(self);

  // 2. Acquire index (Memory Visibility)
  size_t count =
      atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
  if (count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyList_New(0);
  }

  if (count > self->contact_max_capacity) {
    count = self->contact_max_capacity;
  }

  // 3. Fast Copy (Hold lock for the shortest possible time)
  ContactEvent *scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
  if (!scratch) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // Snapshot the buffer data
  memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));

  // Clear the buffer index for the next frame
  atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

  // 4. EXIT the lock immediately
  SHADOW_UNLOCK(&self->shadow_lock);

  // 5. Slow Python Work (Done while the next physics step can run in parallel!)
  PyObject *list = PyList_New((Py_ssize_t)count);
  if (!list) {
    PyMem_RawFree(scratch);
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    // Create a 4-item tuple: (ID1, ID2, ImpactStrength, SlidingStrengthSq)
    PyObject *item = PyTuple_New(4);
    if (!item) {
      Py_DECREF(list);
      PyMem_RawFree(scratch);
      return NULL;
    }

    // PyTuple_SET_ITEM "steals" the reference, so no extra DECREF needed on
    // these creators
    PyTuple_SET_ITEM(item, 0, PyLong_FromUnsignedLongLong(scratch[i].body1));
    PyTuple_SET_ITEM(item, 1, PyLong_FromUnsignedLongLong(scratch[i].body2));
    PyTuple_SET_ITEM(item, 2, PyFloat_FromDouble(scratch[i].impulse));
    PyTuple_SET_ITEM(item, 3, PyFloat_FromDouble(scratch[i].sliding_speed_sq));

    PyList_SET_ITEM(list, (Py_ssize_t)i, item);
  }

  PyMem_RawFree(scratch);
  return list;
}

static PyObject *PhysicsWorld_get_contact_events_ex(PhysicsWorldObject *self, PyObject *args) {
    // 1. Acquire Lock & Copy Data
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    
    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyList_New(0);
    }

    if (count > self->contact_max_capacity) {
        count = self->contact_max_capacity;
    }

    // Allocate scratch buffer
    ContactEvent *scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
    if (!scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // Copy data and release lock immediately
    memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 2. Static Keys (Allocated ONCE)
    // We intentionally "leak" these references for the lifetime of the app.
    // This provides a massive speedup and stability fix.
    static PyObject *k_bodies = NULL;
    static PyObject *k_pos = NULL;
    static PyObject *k_norm = NULL;
    static PyObject *k_str = NULL;
    static PyObject *k_slide = NULL;
    static PyObject *k_mat = NULL;
    static PyObject *k_type = NULL;

    if (!k_bodies) {
        k_bodies = PyUnicode_InternFromString("bodies");
        k_pos = PyUnicode_InternFromString("position");
        k_norm = PyUnicode_InternFromString("normal");
        k_str = PyUnicode_InternFromString("strength");
        k_slide = PyUnicode_InternFromString("slide_sq");
        k_mat = PyUnicode_InternFromString("materials");
        k_type = PyUnicode_InternFromString("type");

        // Paranoid check: if any failed during init, clean up and fail
        if (!k_bodies || !k_pos || !k_norm || !k_str || !k_slide || !k_mat || !k_type) {
            PyMem_RawFree(scratch);
            return PyErr_NoMemory();
        }
    }

    // 3. Build Python List
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) {
        PyMem_RawFree(scratch);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEvent *e = &scratch[i];

        PyObject *dict = PyDict_New();
        if (!dict) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, (Py_ssize_t)i, Py_None);
            continue;
        }

        // 1. Bodies (u64, u64)
        PyObject *b_tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(b_tuple, 0, PyLong_FromUnsignedLongLong(e->body1));
        PyTuple_SET_ITEM(b_tuple, 1, PyLong_FromUnsignedLongLong(e->body2));
        PyDict_SetItem(dict, k_bodies, b_tuple);
        Py_DECREF(b_tuple);

        // 2. Position (f, f, f)
        PyObject *p_tuple = PyTuple_New(3);
        PyTuple_SET_ITEM(p_tuple, 0, PyFloat_FromDouble(e->px));
        PyTuple_SET_ITEM(p_tuple, 1, PyFloat_FromDouble(e->py));
        PyTuple_SET_ITEM(p_tuple, 2, PyFloat_FromDouble(e->pz));
        PyDict_SetItem(dict, k_pos, p_tuple);
        Py_DECREF(p_tuple);

        // 3. Normal (f, f, f)
        PyObject *n_tuple = PyTuple_New(3);
        PyTuple_SET_ITEM(n_tuple, 0, PyFloat_FromDouble(e->nx));
        PyTuple_SET_ITEM(n_tuple, 1, PyFloat_FromDouble(e->ny));
        PyTuple_SET_ITEM(n_tuple, 2, PyFloat_FromDouble(e->nz));
        PyDict_SetItem(dict, k_norm, n_tuple);
        Py_DECREF(n_tuple);

        // 4. Materials (u32, u32)
        PyObject *m_tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(m_tuple, 0, PyLong_FromUnsignedLong(e->mat1));
        PyTuple_SET_ITEM(m_tuple, 1, PyLong_FromUnsignedLong(e->mat2));
        PyDict_SetItem(dict, k_mat, m_tuple);
        Py_DECREF(m_tuple);

        // 5. Strength (float)
        PyObject *s_val = PyFloat_FromDouble(e->impulse);
        PyDict_SetItem(dict, k_str, s_val);
        Py_DECREF(s_val);

        // 6. Sliding Speed (float)
        PyObject *sl_val = PyFloat_FromDouble(e->sliding_speed_sq);
        PyDict_SetItem(dict, k_slide, sl_val);
        Py_DECREF(sl_val);

        // 7. Event Type (int)
        PyObject *t_val = PyLong_FromUnsignedLong(e->type);
        PyDict_SetItem(dict, k_type, t_val);
        Py_DECREF(t_val);

        // Steals ref to dict
        PyList_SET_ITEM(list, (Py_ssize_t)i, dict);
    }

    // REMOVED: Py_DECREF(keys) - we keep them alive statically now.
    
    PyMem_RawFree(scratch);
    return list;
}
// ContactEvent layout (packed, little-endian):
// uint64 body1, uint64 body2
// float32 px, py, pz
// float32 nx, ny, nz
// float32 impulse
static PyObject *PhysicsWorld_get_contact_events_raw(PhysicsWorldObject *self,
                                                     PyObject *args) {
  // 1. Phase Guard
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  // 2. Atomic Acquire (Publication Visibility)
  size_t count =
      atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);

  if (count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    // Return empty view
    PyObject *empty = PyBytes_FromStringAndSize("", 0);
    PyObject *view = PyMemoryView_FromObject(empty);
    Py_DECREF(empty);
    return view;
  }

  if (count > self->contact_max_capacity) {
    count = self->contact_max_capacity;
  }

  // 3. Snapshot Data
  // We copy into a PyBytes object. This is fast (memcpy) and
  // ensures the data remains valid even after the next step() resets the
  // buffer.
  size_t bytes_size = count * sizeof(ContactEvent);
  PyObject *raw_bytes = PyBytes_FromStringAndSize((char *)self->contact_buffer,
                                                  (Py_ssize_t)bytes_size);

  // 4. Reset Index for next frame
  atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

  SHADOW_UNLOCK(&self->shadow_lock);

  if (!raw_bytes) {
    return NULL;
  }

  // 5. Wrap in MemoryView
  // This allows the user to use np.frombuffer(events, dtype=...) without extra
  // copies
  PyObject *view = PyMemoryView_FromObject(raw_bytes);
  Py_DECREF(raw_bytes);
  return view;
}

// Map the procs
static const JPH_CharacterContactListener_Procs char_listener_procs = {
    .OnContactValidate = char_on_contact_validate,
    .OnContactAdded = char_on_contact_added,
    .OnAdjustBodyVelocity = NULL,
    .OnContactPersisted = char_on_contact_added,
    .OnContactRemoved = NULL,
    .OnCharacterContactValidate = NULL,
    .OnCharacterContactAdded = char_on_character_contact_added,
    .OnCharacterContactPersisted = NULL,
    .OnCharacterContactRemoved = NULL,
    .OnContactSolve = NULL};

static const JPH_ContactListener_Procs contact_procs = {
    .OnContactValidate = on_contact_validate,
    .OnContactAdded = on_contact_added,
    .OnContactPersisted = on_contact_persisted,
    .OnContactRemoved = on_contact_removed};

// --- Helper: Shape Caching (Internal) ---
static JPH_Shape *find_or_create_shape(PhysicsWorldObject *self, int type,
                                       const float *params) {
  // 1. HARDENED KEY CONSTRUCTION
  // We zero out the key first so that unused parameters for a specific
  // shape type (like p2-p4 for a Sphere) don't cause cache misses.
  ShapeKey key;
  memset(&key, 0, sizeof(ShapeKey));
  key.type = (uint32_t)type;

  switch (type) {
  case 0: // BOX: Uses 3 params (half-extents)
    key.p1 = params[0];
    key.p2 = params[1];
    key.p3 = params[2];
    break;
  case 1: // SPHERE: Uses 1 param (radius)
    key.p1 = params[0];
    break;
  case 2: // CAPSULE: Uses 2 params (half-height, radius)
  case 3: // CYLINDER: Uses 2 params (half-height, radius)
    key.p1 = params[0];
    key.p2 = params[1];
    break;
  case 4: // PLANE: Uses 4 params (nx, ny, nz, d)
    key.p1 = params[0];
    key.p2 = params[1];
    key.p3 = params[2];
    key.p4 = params[3];
    break;
  default:
    break;
  }

  // 2. CACHE LOOKUP
  for (size_t i = 0; i < self->shape_cache_count; i++) {
    ShapeKey *entry_key = &self->shape_cache[i].key;

    // Explicit comparison avoids padding/alignment issues and handles -0.0 vs
    // 0.0 correctly
    if (entry_key->type == key.type && entry_key->p1 == key.p1 &&
        entry_key->p2 == key.p2 && entry_key->p3 == key.p3 &&
        entry_key->p4 == key.p4) {
      return self->shape_cache[i].shape;
    }
  }

  // 3. SHAPE CREATION (Only if not found)
  JPH_Shape *shape = NULL;
  if (type == 0) {
    JPH_Vec3 he = {key.p1, key.p2, key.p3};
    JPH_BoxShapeSettings *s = JPH_BoxShapeSettings_Create(&he, 0.05f);
    shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 1) {
    JPH_SphereShapeSettings *s = JPH_SphereShapeSettings_Create(key.p1);
    shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 2) {
    JPH_CapsuleShapeSettings *s =
        JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
    shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 3) {
    JPH_CylinderShapeSettings *s =
        JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
    shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 4) {
    JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
    // Note: Planes in Jolt often require a half-extent (1000.0f) to define
    // their "active" area
    JPH_PlaneShapeSettings *s =
        JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
    shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  }

  if (!shape) {
    return NULL;
  }

  // 4. CACHE EXPANSION
  if (self->shape_cache_count >= self->shape_cache_capacity) {
    size_t new_cap =
        (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
    // Note: PyMem_RawRealloc is safe here because this is called under
    // SHADOW_LOCK and is not inside the Jolt step.
    void *new_ptr =
        PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
    if (!new_ptr) {
      JPH_Shape_Destroy(shape);
      PyErr_NoMemory();
      return NULL;
    }
    self->shape_cache = (ShapeEntry *)new_ptr;
    self->shape_cache_capacity = new_cap;
  }

  // 5. STORAGE
  self->shape_cache[self->shape_cache_count].key = key;
  self->shape_cache[self->shape_cache_count].shape = shape;
  self->shape_cache_count++;

  return shape;
}

static void free_constraints(PhysicsWorldObject *self) {
  if (self->constraints) {
      for (size_t i = 0; i < self->constraint_capacity; i++) {
          if (!self->constraints[i]) continue;

          bool is_alive = !self->constraint_states || self->constraint_states[i] == SLOT_ALIVE;
          if (is_alive) {
              if (self->system) {
                  JPH_PhysicsSystem_RemoveConstraint(self->system, self->constraints[i]);
              }
              JPH_Constraint_Destroy(self->constraints[i]);
          }
          self->constraints[i] = NULL;
      }
      PyMem_RawFree((void *)self->constraints);
      self->constraints = NULL;
  }
  PyMem_RawFree(self->constraint_generations); self->constraint_generations = NULL;
  PyMem_RawFree(self->free_constraint_slots);  self->free_constraint_slots = NULL;
  PyMem_RawFree(self->constraint_states);      self->constraint_states = NULL;
}

static void free_shape_cache(PhysicsWorldObject *self) {
  if (!self->shape_cache) return;

  for (size_t i = 0; i < self->shape_cache_count; i++) {
      if (self->shape_cache[i].shape) {
          JPH_Shape_Destroy(self->shape_cache[i].shape);
      }
  }
  PyMem_RawFree(self->shape_cache);
  self->shape_cache = NULL;
  self->shape_cache_count = 0;
}

static void free_shadow_buffers(PhysicsWorldObject *self) {
  PyMem_RawFree(self->positions);          self->positions = NULL;
  PyMem_RawFree(self->rotations);          self->rotations = NULL;
  PyMem_RawFree(self->prev_positions);     self->prev_positions = NULL;
  PyMem_RawFree(self->prev_rotations);     self->prev_rotations = NULL;
  PyMem_RawFree(self->linear_velocities);  self->linear_velocities = NULL;
  PyMem_RawFree(self->angular_velocities); self->angular_velocities = NULL;
  PyMem_RawFree(self->body_ids);           self->body_ids = NULL;
  PyMem_RawFree(self->generations);        self->generations = NULL;
  PyMem_RawFree(self->slot_to_dense);      self->slot_to_dense = NULL;
  PyMem_RawFree(self->dense_to_slot);      self->dense_to_slot = NULL;
  PyMem_RawFree(self->free_slots);         self->free_slots = NULL;
  PyMem_RawFree(self->slot_states);        self->slot_states = NULL;
  PyMem_RawFree(self->command_queue);      self->command_queue = NULL;
  PyMem_RawFree(self->user_data);          self->user_data = NULL;
  PyMem_RawFree(self->categories);         self->categories = NULL;
  PyMem_RawFree(self->masks);              self->masks = NULL;
  PyMem_RawFree(self->material_ids);       self->material_ids = NULL;
  PyMem_RawFree(self->materials);          self->materials = NULL;
}

static void clear_command_queue(PhysicsWorldObject *self) {
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

// --- Helper: Resource Cleanup (Idempotent) ---
// SAFETY:
// - Must not be called while PhysicsSystem is stepping
// - Must not be called from a Jolt callback
// - Must not race with Python memoryview access
static void PhysicsWorld_free_members(PhysicsWorldObject *self) {
  // Clear pending commands
  clear_command_queue(self);
  PyMem_RawFree(self->command_queue);
  self->command_queue = NULL;
  // 1. Constraints (Must go before PhysicsSystem)
  free_constraints(self);

  // 2. Jolt Core Systems
  if (self->system) {
    JPH_PhysicsSystem_Destroy(self->system);
    self->system = NULL;
  }
  if (self->char_vs_char_manager) {
    JPH_CharacterVsCharacterCollision_Destroy(self->char_vs_char_manager);
    self->char_vs_char_manager = NULL;
  }
  if (self->job_system) {
    JPH_JobSystem_Destroy(self->job_system);
    self->job_system = NULL;
  }

  // 3. Debug Utilities
  if (self->debug_renderer) {
    JPH_DebugRenderer_Destroy(self->debug_renderer);
    self->debug_renderer = NULL;
  }
  debug_buffer_free(&self->debug_lines);
  debug_buffer_free(&self->debug_triangles);

  // 4. Shape Cache
  free_shape_cache(self);

  // 5. Contact Listener & Buffers
  if (self->contact_listener) {
    JPH_ContactListener_Destroy(self->contact_listener);
    self->contact_listener = NULL;
  }
  PyMem_RawFree(self->contact_buffer);
  self->contact_buffer = NULL;

  // 6. Native Memory Buffers
  free_shadow_buffers(self);

  // 7. Cleanup remaining pointers
  self->bp_interface = NULL;
  self->pair_filter = NULL;
  self->bp_filter = NULL;
  PyMem_RawFree(self->id_to_handle_map);
  self->id_to_handle_map = NULL;

  FREE_LOCK(self->shadow_lock);
}

// --- Lifecycle: Deallocation ---
static void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
  PhysicsWorld_free_members(self);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Lifecycle: Initialization ---
// helper: Initialize settings via Python helper
static int init_settings(PhysicsWorldObject *self, PyObject *settings_dict,
                         float *gx, float *gy, float *gz, int *max_bodies,
                         int *max_pairs) {
  PyObject *st_module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(st_module);
  PyObject *val_func = PyObject_GetAttrString(st->helper, "validate_settings");
  if (!val_func)
    return -1;

  PyObject *norm = PyObject_CallFunctionObjArgs(
      val_func, settings_dict ? settings_dict : Py_None, NULL);
  Py_DECREF(val_func);
  if (!norm)
    return -1;

  float slop;
  int ok = PyArg_ParseTuple(norm, "ffffii", gx, gy, gz, &slop, max_bodies,
                            max_pairs);
  Py_DECREF(norm);
  return ok ? 0 : -1;
}

// helper: Initialize Jolt Core Systems
static int init_jolt_core(PhysicsWorldObject *self, WorldLimits limits, GravityVector gravity) {
  JobSystemThreadPoolConfig job_cfg = {
      .maxJobs = 1024, .maxBarriers = 8, .numThreads = -1};
  self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
  self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(2, 2);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 0, 0);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 1, 1);
  self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(2);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 0);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1);
  self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(
      self->bp_interface, 2, self->pair_filter, 2);

  JPH_PhysicsSystemSettings phys_settings = {
        .maxBodies = (uint32_t)limits.max_bodies,
        .maxBodyPairs = (uint32_t)limits.max_pairs, // Now safe
        .maxContactConstraints = 1024*1024,
        .broadPhaseLayerInterface = self->bp_interface,
        .objectLayerPairFilter = self->pair_filter,
        .objectVsBroadPhaseLayerFilter = self->bp_filter};

  self->system = JPH_PhysicsSystem_Create(&phys_settings);
  self->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
  JPH_PhysicsSystem_SetGravity(self->system, &(JPH_Vec3){gravity.gx, gravity.gy, gravity.gz});
  self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);
  return 0;
}

// helper: Allocate shadow buffers and indirection maps
static int allocate_buffers(PhysicsWorldObject *self, int max_bodies) {
  self->capacity = (size_t)max_bodies;
  if (self->capacity < self->count + 128)
    self->capacity = self->count + 1024;
  self->max_jolt_bodies = (uint32_t)max_bodies;
  self->slot_capacity = self->capacity;

  self->positions = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->rotations = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->prev_positions = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->prev_rotations = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->linear_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->angular_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->body_ids = PyMem_RawMalloc(self->capacity * sizeof(JPH_BodyID));
  self->user_data = PyMem_RawCalloc(self->capacity, sizeof(uint64_t));
  self->categories = PyMem_RawMalloc(self->capacity * sizeof(uint32_t));
  self->masks = PyMem_RawMalloc(self->capacity * sizeof(uint32_t));
  self->material_ids = PyMem_RawCalloc(self->capacity, sizeof(uint32_t));
  self->id_to_handle_map =
      PyMem_RawCalloc(self->max_jolt_bodies, sizeof(BodyHandle));
  self->generations = PyMem_RawCalloc(self->slot_capacity, sizeof(uint32_t));
  self->slot_to_dense = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->dense_to_slot = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->free_slots = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->slot_states = PyMem_RawCalloc(self->slot_capacity, sizeof(uint8_t));
  self->command_queue = PyMem_RawMalloc(64 * sizeof(PhysicsCommand));
  self->command_capacity = 64;

  if (!self->positions || !self->id_to_handle_map || !self->command_queue ||
      !self->slot_states)
    return -1;

  for (size_t i = 0; i < self->capacity; i++) {
    self->categories[i] = 0xFFFF;
    self->masks[i] = 0xFFFF;
  }
  return 0;
}

// helper: Iterate over baked Python data to create initial Jolt bodies
static int load_baked_scene(PhysicsWorldObject *self, PyObject *baked) {
  float *f_pos = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 1));
  float *f_rot = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 2));
  float *f_shape = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 3));
  unsigned char *u_mot =
      (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 4));
  unsigned char *u_layer =
      (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 5));
  uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, 6));

  for (size_t i = 0; i < self->count; i++) {
    float params[4] = {f_shape[i * 5 + 1], f_shape[i * 5 + 2],
                       f_shape[i * 5 + 3], f_shape[i * 5 + 4]};
    JPH_Shape *shape = find_or_create_shape(self, (int)f_shape[i * 5], params);
    if (!shape)
      return -1;

    JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){f_pos[i * 4], f_pos[i * 4 + 1], f_pos[i * 4 + 2]},
        &(JPH_Quat){f_rot[i * 4], f_rot[i * 4 + 1], f_rot[i * 4 + 2],
                    f_rot[i * 4 + 3]},
        (JPH_MotionType)u_mot[i], (JPH_ObjectLayer)u_layer[i]);

    self->generations[i] = 1;
    JPH_BodyCreationSettings_SetUserData(creation,
                                         (uint64_t)make_handle((uint32_t)i, 1));
    if (u_mot[i] == 2)
      JPH_BodyCreationSettings_SetAllowSleeping(creation, true);

    self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(
        self->body_interface, creation, JPH_Activation_Activate);
    uint32_t j_idx = JPH_ID_TO_INDEX(self->body_ids[i]);
    if (j_idx < self->max_jolt_bodies)
      self->id_to_handle_map[j_idx] = make_handle((uint32_t)i, 1);

    self->slot_to_dense[i] = self->dense_to_slot[i] = (uint32_t)i;
    self->slot_states[i] = SLOT_ALIVE;
    self->user_data[i] = u_data[i];
    JPH_BodyCreationSettings_Destroy(creation);
  }
  return 0;
}

static int verify_abi_alignment(JPH_BodyInterface *bi) {
  JPH_BoxShapeSettings *bs =
      JPH_BoxShapeSettings_Create(&(JPH_Vec3){1, 1, 1}, 0.0f);
  JPH_Shape *shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(bs);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)bs);
  if (!shape)
    return -1;

  JPH_BodyCreationSettings *bcs = JPH_BodyCreationSettings_Create3(
      shape, &(JPH_RVec3){10.0, 20.0, 30.0}, &(JPH_Quat){0, 0, 0, 1},
      JPH_MotionType_Static, 0);
  JPH_Shape_Destroy(shape);
  if (!bcs)
    return -1;

  JPH_BodyID bid =
      JPH_BodyInterface_CreateAndAddBody(bi, bcs, JPH_Activation_Activate);
  JPH_BodyCreationSettings_Destroy(bcs);

  JPH_STACK_ALLOC(JPH_RVec3, p_check);
  JPH_BodyInterface_GetPosition(bi, bid, p_check);
  JPH_BodyInterface_RemoveBody(bi, bid);
  JPH_BodyInterface_DestroyBody(bi, bid);

  if (fabs(p_check->x - 10.0) > 0.1 || fabs(p_check->y - 20.0) > 0.1) {
    PyErr_SetString(PyExc_RuntimeError, "JoltC ABI Mismatch: Precision issue.");
    return -1;
  }
  return 0;
}

// Orchestrator function
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                             PyObject *kwds) {
  PyObject *settings_dict = NULL, *bodies_list = NULL, *baked = NULL;
  float gx, gy, gz;
  int max_bodies, max_pairs;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO",
                                   (char *[]){"settings", "bodies", NULL},
                                   &settings_dict, &bodies_list))
    return -1;

  // 1. Initial State
  memset(((char *)self) + offsetof(PhysicsWorldObject, system), 0,
         sizeof(PhysicsWorldObject) - offsetof(PhysicsWorldObject, system));
  INIT_LOCK(self->shadow_lock);
  self->debug_renderer = JPH_DebugRenderer_Create(self);
  JPH_DebugRenderer_SetProcs(&debug_procs);
  atomic_init(&self->is_stepping, false);

  // 2. Settings & Jolt Init
  if (init_settings(self, settings_dict, &gx, &gy, &gz, &max_bodies,
                    &max_pairs) < 0)
    goto fail;
  WorldLimits limits = {max_bodies, max_pairs};
  GravityVector gravity = {gx, gy, gz};
  if (init_jolt_core(self, limits, gravity) < 0) 
    goto fail;

  if (verify_abi_alignment(self->body_interface) < 0)
    goto fail;

  self->contact_max_capacity = 4096;
  self->contact_buffer = PyMem_RawMalloc(4096 * sizeof(ContactEvent));
  atomic_init(&self->contact_atomic_idx, 0);
  JPH_ContactListener_SetProcs(&contact_procs);
  self->contact_listener = JPH_ContactListener_Create(self);
  JPH_PhysicsSystem_SetContactListener(self->system, self->contact_listener);

  // 3. Bake & Buffers
  if (bodies_list && bodies_list != Py_None) {
    PyObject *st_helper =
        get_culverin_state(PyType_GetModule(Py_TYPE(self)))->helper;
    PyObject *bake_func = PyObject_GetAttrString(st_helper, "bake_scene");
    baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
    Py_XDECREF(bake_func);
    if (!baked)
      goto fail;
    self->count = PyLong_AsSize_t(PyTuple_GetItem(baked, 0));
  }

  if (allocate_buffers(self, max_bodies) < 0)
    goto fail;

  // 4. Constraints & Data Loading
  self->constraint_capacity = 256;
  self->constraints =
      (JPH_Constraint **)PyMem_RawCalloc(256, sizeof(JPH_Constraint *));
  self->constraint_generations = PyMem_RawCalloc(256, sizeof(uint32_t));
  self->free_constraint_slots = PyMem_RawMalloc(256 * sizeof(uint32_t));
  self->constraint_states = PyMem_RawCalloc(256, sizeof(uint8_t));
  if (!self->constraints || !self->free_constraint_slots)
    goto fail;

  for (uint32_t i = 0; i < 256; i++) {
    self->constraint_generations[i] = 1;
    self->free_constraint_slots[i] = i;
  }
  self->free_constraint_count = 256;

  if (baked && load_baked_scene(self, baked) < 0)
    goto fail;
  Py_XDECREF(baked);

  for (uint32_t i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity;
       i++) {
    self->generations[i] = 1;
    self->free_slots[self->free_count++] = i;
  }

  culverin_sync_shadow_buffers(self);
  return 0;

fail:
  Py_XDECREF(baked);
  PhysicsWorld_free_members(self);
  return -1;
}

static PyObject *PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
    uint64_t h;
    float x, y, z;

    // --- MANUALLY PARSE ARGS (FAST PATH) ---
    // If no keywords are provided and we have exactly 4 arguments
    if (LIKELY(kwds == NULL && PyTuple_GET_SIZE(args) == 4)) {
        PyObject *py_h = PyTuple_GET_ITEM(args, 0);
        PyObject *py_x = PyTuple_GET_ITEM(args, 1);
        PyObject *py_y = PyTuple_GET_ITEM(args, 2);
        PyObject *py_z = PyTuple_GET_ITEM(args, 3);

        h = PyLong_AsUnsignedLongLong(py_h);
        x = (float)PyFloat_AsDouble(py_x);
        y = (float)PyFloat_AsDouble(py_y);
        z = (float)PyFloat_AsDouble(py_z);

        // Check if conversion failed (e.g. user passed a string or None)
        if (UNLIKELY(PyErr_Occurred())) {
            return NULL;
        }
    } else {
        // --- FALLBACK TO STANDARD PARSING (SLOW PATH) ---
        static char *kwlist[] = {"handle", "x", "y", "z", NULL};
        if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &h, &x, &y, &z)) {
            return NULL;
        }
    }

    if (UNLIKELY(!isfinite(x) || !isfinite(y) || !isfinite(z))) {
        PyErr_SetString(PyExc_ValueError, "Impulse components must be finite (no NaN/Inf)");
        return NULL;
    }

    // --- EXECUTION ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    uint32_t slot = 0;
    // Check liveness and generation
    if (UNLIKELY(!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // Single lookup of indices
    uint32_t dense_idx = self->slot_to_dense[slot];
    JPH_BodyID bid = self->body_ids[dense_idx];
    
    JPH_Vec3 imp = {x, y, z};
    
    // Jolt thread-safe application
    JPH_BodyInterface_AddImpulse(self->body_interface, bid, &imp);
    JPH_BodyInterface_ActivateBody(self->body_interface, bid);

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_apply_impulse_at(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  uint64_t h;
  float ix, iy, iz; // Impulse
  float px, py, pz; // Position
  
  // --- MANUALLY PARSE ARGS (FAST PATH) ---
  // Optimizing for positional-only calls: handle, ix, iy, iz, px, py, pz
  if (LIKELY(kwds == NULL && PyTuple_GET_SIZE(args) == 7)) {
      PyObject *p_h = PyTuple_GET_ITEM(args, 0);
      PyObject *p_ix = PyTuple_GET_ITEM(args, 1);
      PyObject *p_iy = PyTuple_GET_ITEM(args, 2);
      PyObject *p_iz = PyTuple_GET_ITEM(args, 3);
      PyObject *p_px = PyTuple_GET_ITEM(args, 4);
      PyObject *p_py = PyTuple_GET_ITEM(args, 5);
      PyObject *p_pz = PyTuple_GET_ITEM(args, 6);

      h = PyLong_AsUnsignedLongLong(p_h);
      ix = (float)PyFloat_AsDouble(p_ix);
      iy = (float)PyFloat_AsDouble(p_iy);
      iz = (float)PyFloat_AsDouble(p_iz);
      px = (float)PyFloat_AsDouble(p_px);
      py = (float)PyFloat_AsDouble(p_py);
      pz = (float)PyFloat_AsDouble(p_pz);

      if (UNLIKELY(PyErr_Occurred())) {
          return NULL;
      }
  } else {
      // --- FALLBACK TO STANDARD PARSING (SLOW PATH) ---
      static char *kwlist[] = {"handle", "ix", "iy", "iz", "px", "py", "pz", NULL};
      if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffffff", kwlist, &h, &ix, &iy, &iz, &px, &py, &pz)) {
        return NULL;
      }
  }
  
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (UNLIKELY(!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
  
  // Single lookup for body ID
  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

  // JPH_Vec3 is float[3], JPH_RVec3 is double[3]
  JPH_Vec3 imp = {ix, iy, iz};
  JPH_RVec3 pos = {(double)px, (double)py, (double)pz};
  
  // Apply Impulse (Immediate, relies on Jolt internal locking)
  JPH_BodyInterface_AddImpulse2(self->body_interface, bid, &imp, &pos);
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// Helper 1: Run the Raycast (Encapsulates all Jolt Filter/Lock/Cleanup
// boilerplate)
static bool execute_raycast_query(PhysicsWorldObject *self,
                                  JPH_BodyID ignore_bid,
                                  const JPH_RVec3 *origin,
                                  const JPH_Vec3 *direction,
                                  JPH_RayCastResult *hit) {
  bool has_hit;

  // 1. LOCK TRAMPOLINE
  SHADOW_LOCK(&g_jph_trampoline_lock);

  // 2. Filter Setup
  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide =
                                                  filter_allow_all_bp};
  JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide =
                                               filter_allow_all_obj};
  JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  CastShapeFilter filter_ctx = {.ignore_id = ignore_bid};
  JPH_BodyFilter_Procs filter_procs = {.ShouldCollide = CastShape_BodyFilter};
  JPH_BodyFilter *bf = JPH_BodyFilter_Create(&filter_ctx);
  JPH_BodyFilter_SetProcs(&filter_procs);

  // 3. Execution
  const JPH_NarrowPhaseQuery *query =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
  has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, bp_f,
                                         obj_f, bf);

  // 4. Restore & Cleanup
  JPH_BodyFilter_SetProcs(&global_bf_procs);
  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  JPH_BodyFilter_Destroy(bf);
  JPH_BroadPhaseLayerFilter_Destroy(bp_f);
  JPH_ObjectLayerFilter_Destroy(obj_f);

  return has_hit;
}

// Helper 2: Extract World Space Normal after hit
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
static void extract_hit_normal(PhysicsWorldObject *self, JPH_BodyID bodyID,
                               JPH_SubShapeID subShapeID2,
                               const JPH_RVec3 *origin, const JPH_Vec3 *ray_dir,
                               float fraction, JPH_Vec3 *normal_out) {
  const JPH_BodyLockInterface *lock_iface =
      JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockRead lock;
  JPH_BodyLockInterface_LockRead(lock_iface, bodyID, &lock);

  if (lock.body) {
    JPH_RVec3 hit_p = {origin->x + ray_dir->x * fraction,
                       origin->y + ray_dir->y * fraction,
                       origin->z + ray_dir->z * fraction};
    JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, subShapeID2, &hit_p,
                                        normal_out);
  } else {
    normal_out->x = 0;
    normal_out->y = 1;
    normal_out->z = 0;
  }
  JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);
}

// Main Orchestrator
static PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *args,
                                      PyObject *kwds) {
  float sx, sy, sz, dx, dy, dz, max_dist = 1000.0f;
  uint64_t ignore_h = 0;
  static char *kwlist[] = {"start", "direction", "max_dist", "ignore", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|fK", kwlist, &sx,
                                   &sy, &sz, &dx, &dy, &dz, &max_dist,
                                   &ignore_h))
    return NULL;

  PyObject *result = NULL;

  // --- 1. Validation & Pre-calc ---
  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f)
    Py_RETURN_NONE;

  float mag = sqrtf(mag_sq);
  float scale = max_dist / mag;

  JPH_STACK_ALLOC(JPH_RVec3, origin);
  origin->x = sx;
  origin->y = sy;
  origin->z = sz;
  JPH_STACK_ALLOC(JPH_Vec3, direction);
  direction->x = dx * scale;
  direction->y = dy * scale;
  direction->z = dz * scale;
  JPH_STACK_ALLOC(JPH_RayCastResult, hit);
  memset(hit, 0, sizeof(JPH_RayCastResult));

  // --- 2. Query Setup (Lock/Wait/Increment) ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);

  JPH_BodyID ignore_bid = 0;
  uint32_t ignore_slot = 0;
  if (ignore_h != 0 && unpack_handle(self, ignore_h, &ignore_slot)) {
    ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 3. Execution (Unlocked, Trampoline Locked internally) ---
  bool has_hit =
      execute_raycast_query(self, ignore_bid, origin, direction, hit);

  if (!has_hit)
    goto exit;

  // --- 4. Hit Result Extraction ---
  JPH_Vec3 normal;
  extract_hit_normal(self, hit->bodyID, hit->subShapeID2, origin, direction,
                     hit->fraction, &normal);

  // --- 5. Resolve Handle (Shadow Locked) ---
  SHADOW_LOCK(&self->shadow_lock);
  BodyHandle handle = (BodyHandle)JPH_BodyInterface_GetUserData(
      self->body_interface, hit->bodyID);
  uint32_t slot = (uint32_t)(handle & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(handle >> 32);

  if (slot < self->slot_capacity && self->generations[slot] == gen &&
      self->slot_states[slot] == SLOT_ALIVE) {
    result = Py_BuildValue("Kf(fff)", handle, hit->fraction, normal.x, normal.y,
                           normal.z);
  }
  SHADOW_UNLOCK(&self->shadow_lock);

exit:
  // Decrement active query counter
  atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);

  return result ? result : Py_None;
}

static PyObject *PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  Py_buffer b_starts;
  Py_buffer b_dirs;
  float max_dist = 1000.0f;
  static char *kwlist[] = {"starts", "directions", "max_dist", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*y*|f", kwlist, &b_starts,
                                   &b_dirs, &max_dist)) {
    return NULL;
  }

  // 1. Validation
  if (b_starts.len != b_dirs.len || (b_starts.len % (3 * sizeof(float)) != 0)) {
    PyBuffer_Release(&b_starts);
    PyBuffer_Release(&b_dirs);
    PyErr_SetString(PyExc_ValueError,
                    "Buffers must be equal size and multiples of 3*float32");
    return NULL;
  }

  size_t count = b_starts.len / (3 * sizeof(float));
  PyObject *result_bytes = PyBytes_FromStringAndSize(
      NULL, (Py_ssize_t)(count * sizeof(RayCastBatchResult)));
  if (!result_bytes) {
    PyBuffer_Release(&b_starts);
    PyBuffer_Release(&b_dirs);
    return PyErr_NoMemory();
  }

  RayCastBatchResult *results =
      (RayCastBatchResult *)PyBytes_AsString(result_bytes);
  float *f_starts = (float *)b_starts.buf;
  float *f_dirs = (float *)b_dirs.buf;

  // 2. Lock & Phase Guard
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  // This atomic prevents step() or resize() from running while we are in the
  // C++ loop
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  // 3. Multithreaded Execution (GIL Released)
  Py_BEGIN_ALLOW_THREADS

      // We lock the trampoline once for the whole batch
      SHADOW_LOCK(&g_jph_trampoline_lock);

  const JPH_NarrowPhaseQuery *query =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
  const JPH_BodyLockInterface *lock_iface =
      JPH_PhysicsSystem_GetBodyLockInterface(self->system);

  // Filter setup
  JPH_BroadPhaseLayerFilter *bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(
      &(JPH_BroadPhaseLayerFilter_Procs){.ShouldCollide = filter_allow_all_bp});
  JPH_ObjectLayerFilter *obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(
      &(JPH_ObjectLayerFilter_Procs){.ShouldCollide = filter_allow_all_obj});
  JPH_BodyFilter *body_filter = JPH_BodyFilter_Create(NULL);
  JPH_BodyFilter_SetProcs(
      &(JPH_BodyFilter_Procs){.ShouldCollide = filter_true_body});

  for (size_t i = 0; i < count; i++) {
    size_t off = i * 3;
    RayCastBatchResult *res = &results[i];
    memset(res, 0, sizeof(RayCastBatchResult));

    JPH_RVec3 origin = {(double)f_starts[off], (double)f_starts[off + 1],
                        (double)f_starts[off + 2]};

    // Normalize and scale direction
    float dx = f_dirs[off];
    float dy = f_dirs[off + 1];
    float dz = f_dirs[off + 2];
    float mag_sq = dx * dx + dy * dy + dz * dz;
    // Check mag_sq to avoid sqrt(0) and division by zero
    if (mag_sq < 1e-12f) {
      results[i].handle = 0; // No hit for zero-length ray
      continue;
    }
    float mag = sqrtf(mag_sq);
    float scale = max_dist / mag;
    JPH_Vec3 direction = {dx * scale, dy * scale, dz * scale};

    JPH_RayCastResult hit;
    memset(&hit, 0, sizeof(hit));

    if (JPH_NarrowPhaseQuery_CastRay(query, &origin, &direction, &hit,
                                     bp_filter, obj_filter, body_filter)) {
      res->handle =
          JPH_BodyInterface_GetUserData(self->body_interface, hit.bodyID);
      res->fraction = hit.fraction;
      res->subShapeID = hit.subShapeID2;

      // --- Material ID Lookup ---
      // Safe to read shadow buffers here because active_queries prevents
      // step() or resize() from mutating them while we run.
      uint32_t slot = (uint32_t)(res->handle & 0xFFFFFFFF);
      uint32_t gen = (uint32_t)(res->handle >> 32);

      if (slot < self->slot_capacity && self->generations[slot] == gen) {
        uint32_t dense = self->slot_to_dense[slot];
        res->material_id = self->material_ids[dense];
      } else {
        res->material_id = 0;
      }

      // World normal extraction
      JPH_BodyLockRead lock;
      JPH_BodyLockInterface_LockRead(lock_iface, hit.bodyID, &lock);
      if (lock.body) {
        JPH_RVec3 hit_p = {
            origin.x + (double)direction.x * (double)hit.fraction,
            origin.y + (double)direction.y * (double)hit.fraction,
            origin.z + (double)direction.z * (double)hit.fraction};
        JPH_Vec3 norm;
        JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, hit.subShapeID2, &hit_p,
                                            &norm);
        res->nx = norm.x;
        res->ny = norm.y;
        res->nz = norm.z;
        res->px = (float)hit_p.x;
        res->py = (float)hit_p.y;
        res->pz = (float)hit_p.z;
      }
      JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);
    }
  }

  JPH_BodyFilter_Destroy(body_filter);
  JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  JPH_ObjectLayerFilter_Destroy(obj_filter);
  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  Py_END_ALLOW_THREADS

      // 4. Cleanup
      atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);
  PyBuffer_Release(&b_starts);
  PyBuffer_Release(&b_dirs);

  return result_bytes;
}

static PyObject *PhysicsWorld_apply_buoyancy(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float surface_y = 0.0f;
  float buoyancy = 1.0f;
  float lin_drag = 0.5f;
  float ang_drag = 0.5f;
  float dt = 1.0f / 60.0f;
  float vx = 0;
  float vy = 0;
  float vz = 0;

  static char *kwlist[] = {
      "handle",       "surface_y", "buoyancy",       "linear_drag",
      "angular_drag", "dt",        "fluid_velocity", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kf|ffff(fff)", kwlist,
                                   &handle_raw, &surface_y, &buoyancy,
                                   &lin_drag, &ang_drag, &dt, &vx, &vy, &vz)) {
    return NULL;
  }

  // --- 1. RESOLUTION PHASE (Locked) ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, handle_raw, &slot) ||
      self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_FALSE;
  }

  // Capture the BodyID and pointer to interface
  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
  JPH_BodyInterface *bi = self->body_interface;
  JPH_PhysicsSystem *system = self->system;

  // RELEASE the lock early. The body will not be destroyed until
  // the next world.step() call because deletions are deferred.
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. EXECUTION PHASE (Unlocked & GIL-Friendly) ---

  // Wake up the body so it continues to simulate in the fluid
  JPH_BodyInterface_ActivateBody(bi, bid);

  JPH_Vec3 gravity;
  JPH_PhysicsSystem_GetGravity(system, &gravity);

  JPH_STACK_ALLOC(JPH_RVec3, surf_pos);
  surf_pos->x = 0;
  surf_pos->y = (double)surface_y;
  surf_pos->z = 0;

  JPH_STACK_ALLOC(JPH_Vec3, surf_norm);
  surf_norm->x = 0;
  surf_norm->y = 1.0f;
  surf_norm->z = 0;

  JPH_STACK_ALLOC(JPH_Vec3, fluid_vel);
  fluid_vel->x = vx;
  fluid_vel->y = vy;
  fluid_vel->z = vz;

  // Heavy lifting done GIL-free if Python were configured that way,
  // but crucially, it's done without blocking the Shadow Buffers.
  bool submerged = JPH_BodyInterface_ApplyBuoyancyImpulse(
      bi, bid, surf_pos, surf_norm, buoyancy, lin_drag, ang_drag, fluid_vel,
      &gravity, dt);

  return PyBool_FromLong(submerged);
}

static PyObject *PhysicsWorld_apply_buoyancy_batch(PhysicsWorldObject *self,
                                                   PyObject *args, PyObject *kwds) {
  Py_buffer h_view = {0};
  float surface_y = 0.0f;
  float buoyancy = 1.0f;
  float lin_drag = 0.5f;
  float ang_drag = 0.5f;
  float dt = 1.0f / 60.0f;
  float vx = 0;
  float vy = 0;
  float vz = 0;

  static char *kwlist[] = {
      "handles",      "surface_y",   "buoyancy", "linear_drag",
      "angular_drag", "dt",          "fluid_velocity", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*|fffff(fff)", kwlist, &h_view,
                                   &surface_y, &buoyancy, &lin_drag, &ang_drag,
                                   &dt, &vx, &vy, &vz)) {
    return NULL;
  }

  // 1. Validation
  if (h_view.itemsize != 8) {
    PyBuffer_Release(&h_view);
    return PyErr_Format(PyExc_ValueError, 
        "Handle buffer must be uint64 (itemsize=8), got %zd", h_view.itemsize);
  }
  
  size_t count = h_view.len / 8;
  if (count == 0) {
    PyBuffer_Release(&h_view);
    Py_RETURN_NONE;
  }

  // 2. Allocate Temp Storage for BodyIDs
  // We do this to avoid holding the SHADOW_LOCK during the heavy Jolt math loop
  JPH_BodyID *ids = (JPH_BodyID *)PyMem_RawMalloc(count * sizeof(JPH_BodyID));
  if (!ids) {
    PyBuffer_Release(&h_view);
    return PyErr_NoMemory();
  }

  uint64_t *handles = (uint64_t *)h_view.buf;
  size_t valid_count = 0;

  // 3. RESOLUTION PHASE (Locked)
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  for (size_t i = 0; i < count; i++) {
    uint32_t slot = 0;
    // Fast unpack inline
    uint64_t h = handles[i];
    slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {
      uint32_t dense = self->slot_to_dense[slot];
      ids[valid_count++] = self->body_ids[dense];
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);
  
  PyBuffer_Release(&h_view); // Done with Python object

  // 4. EXECUTION PHASE (Unlocked)
  // Jolt is thread-safe for these calls.
  JPH_BodyInterface *bi = self->body_interface;
  JPH_PhysicsSystem *sys = self->system;
  
  JPH_Vec3 gravity;
  JPH_PhysicsSystem_GetGravity(sys, &gravity);

  JPH_STACK_ALLOC(JPH_RVec3, surf_pos);
  surf_pos->x = 0;
  surf_pos->y = (double)surface_y;
  surf_pos->z = 0;

  JPH_STACK_ALLOC(JPH_Vec3, surf_norm);
  surf_norm->x = 0;
  surf_norm->y = 1.0f;
  surf_norm->z = 0;

  JPH_STACK_ALLOC(JPH_Vec3, fluid_vel);
  fluid_vel->x = vx;
  fluid_vel->y = vy;
  fluid_vel->z = vz;

  for (size_t i = 0; i < valid_count; i++) {
      JPH_BodyID bid = ids[i];
      // Wake up
      JPH_BodyInterface_ActivateBody(bi, bid);
      // Apply
      JPH_BodyInterface_ApplyBuoyancyImpulse(
          bi, bid, surf_pos, surf_norm, buoyancy, lin_drag, ang_drag, 
          fluid_vel, &gravity, dt);
  }

  PyMem_RawFree(ids);
  Py_RETURN_NONE;
}

// Callback: Called by Jolt when a hit is found during the sweep
static float CastShape_ClosestCollector(void *context,
                                        const JPH_ShapeCastResult *result) {
  CastShapeContext *ctx = (CastShapeContext *)context;

  // We only care about the closest hit (smallest fraction)
  if (result->fraction < ctx->hit.fraction) {
    ctx->hit = *result;
    ctx->has_hit = true;
  }

  // Returning the fraction tells Jolt to ignore any future hits further than
  // this one
  return result->fraction;
}

// Helper 1: Parse shape parameters from Python tuple or float
static void parse_shape_params(PyObject *py_size, float s[4]) {
  memset(s, 0, sizeof(float) * 4);
  if (!py_size || py_size == Py_None)
    return;

  if (PyTuple_Check(py_size)) {
    Py_ssize_t sz_len = PyTuple_Size(py_size);
    for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
      PyObject *item = PyTuple_GetItem(py_size, i);
      if (PyNumber_Check(item))
        s[i] = (float)PyFloat_AsDouble(item);
    }
  } else if (PyNumber_Check(py_size)) {
    s[0] = (float)PyFloat_AsDouble(py_size);
  }
}

// Helper 2: Internal logic to run the actual query under trampoline locks
static void shapecast_execute_internal(PhysicsWorldObject *self,
                                       const JPH_Shape *shape,
                                       const JPH_RMat4 *transform,
                                       const JPH_Vec3 *sweep_dir,
                                       JPH_BodyID ignore_bid,
                                       CastShapeContext *ctx) {
  SHADOW_LOCK(&g_jph_trampoline_lock);

  JPH_BroadPhaseLayerFilter_Procs bp_p = {.ShouldCollide = filter_allow_all_bp};
  JPH_BroadPhaseLayerFilter *bp_f = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_p);

  JPH_ObjectLayerFilter_Procs obj_p = {.ShouldCollide = filter_allow_all_obj};
  JPH_ObjectLayerFilter *obj_f = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_p);

  CastShapeFilter filter_ctx = {.ignore_id = ignore_bid};
  JPH_BodyFilter_Procs bf_p = {.ShouldCollide = CastShape_BodyFilter};
  JPH_BodyFilter *bf = JPH_BodyFilter_Create(&filter_ctx);
  JPH_BodyFilter_SetProcs(&bf_p);

  JPH_STACK_ALLOC(JPH_ShapeCastSettings, settings);
  JPH_ShapeCastSettings_Init(settings);
  settings->backFaceModeTriangles = JPH_BackFaceMode_IgnoreBackFaces;
  settings->backFaceModeConvex = JPH_BackFaceMode_IgnoreBackFaces;

  JPH_RVec3 base_offset = {0, 0, 0};
  const JPH_NarrowPhaseQuery *nq =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  JPH_NarrowPhaseQuery_CastShape(nq, shape, transform, sweep_dir, settings,
                                 &base_offset, CastShape_ClosestCollector, ctx,
                                 bp_f, obj_f, bf, NULL);

  JPH_BodyFilter_SetProcs(&global_bf_procs);
  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  JPH_BodyFilter_Destroy(bf);
  JPH_BroadPhaseLayerFilter_Destroy(bp_f);
  JPH_ObjectLayerFilter_Destroy(obj_f);
}

// Main Orchestrator
static PyObject *PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                        PyObject *args, PyObject *kwds) {
  int shape_type = 0;
  float px, py, pz, rx, ry, rz, rw, dx, dy, dz = NAN;
  PyObject *py_size = NULL;
  uint64_t ignore_h = 0;
  static char *kwlist[] = {"shape", "pos",    "rot", "dir",
                           "size",  "ignore", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "i(fff)(ffff)(fff)O|K", kwlist,
                                   &shape_type, &px, &py, &pz, &rx, &ry, &rz,
                                   &rw, &dx, &dy, &dz, &py_size, &ignore_h))
    return NULL;

  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f)
    Py_RETURN_NONE;

  float s[4];
  parse_shape_params(py_size, s);

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  JPH_Shape *shape = find_or_create_shape(self, shape_type, s);
  if (!shape) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_RuntimeError, "Invalid shape parameters");
  }

  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  JPH_BodyID ignore_bid = 0;
  uint32_t ignore_slot;
  if (ignore_h && unpack_handle(self, ignore_h, &ignore_slot)) {
    ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, &(JPH_Quat){rx, ry, rz, rw},
                                &(JPH_RVec3){px, py, pz});
  JPH_Vec3 sweep_dir = {dx, dy, dz};

  CastShapeContext ctx = {.has_hit = false};
  ctx.hit.fraction = 1.0f;

  shapecast_execute_internal(self, shape, transform, &sweep_dir, ignore_bid,
                             &ctx);

  PyObject *result = NULL;
  if (ctx.has_hit) {
    float nx = -ctx.hit.penetrationAxis.x, ny = -ctx.hit.penetrationAxis.y,
          nz = -ctx.hit.penetrationAxis.z;
    float n_len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (n_len > 1e-6f) {
      nx /= n_len;
      ny /= n_len;
      nz /= n_len;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BodyHandle h = (BodyHandle)JPH_BodyInterface_GetUserData(
        self->body_interface, ctx.hit.bodyID2);
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    if (slot < self->slot_capacity &&
        self->generations[slot] == (uint32_t)(h >> 32) &&
        self->slot_states[slot] == SLOT_ALIVE) {
      result = Py_BuildValue(
          "Kf(fff)(fff)", h, ctx.hit.fraction, ctx.hit.contactPointOn2.x,
          ctx.hit.contactPointOn2.y, ctx.hit.contactPointOn2.z, nx, ny, nz);
    }
    SHADOW_UNLOCK(&self->shadow_lock);
  }

  atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);
  return result ? result : Py_None;
}

// Helper to grow queue
static bool ensure_command_capacity(PhysicsWorldObject *self) {
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

static void flush_commands(PhysicsWorldObject *self) {
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

// Constraints

// Initialize defaults to avoid garbage data
static void params_init(ConstraintParams *p) {
  p->px = 0;
  p->py = 0;
  p->pz = 0;
  p->ax = 0;
  p->ay = 1;
  p->az = 0; // Default Up axis
  p->limit_min = -FLT_MAX;
  p->limit_max = FLT_MAX;
  p->half_cone_angle = 0.0f;
}

// --- 1. Python Parsers ---

static int parse_point_params(PyObject *args, ConstraintParams *p) {
  if (!args || args == Py_None) {
    return 1; // Use defaults (0,0,0)
  }
  return PyArg_ParseTuple(args, "fff", &p->px, &p->py, &p->pz);
}

static int parse_hinge_params(PyObject *args, ConstraintParams *p) {
  p->limit_min = -JPH_M_PI;
  p->limit_max = JPH_M_PI; // Hinge defaults
  if (!args) {
    return 1;
  }
  // (Pivot), (Axis), [Min, Max]
  return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax,
                          &p->ay, &p->az, &p->limit_min, &p->limit_max);
}

static int parse_slider_params(PyObject *args, ConstraintParams *p) {
  // Slider axis defaults to X usually, but Y is fine. Limits default to free.
  if (!args) {
    return 1;
  }
  return PyArg_ParseTuple(args, "(fff)(fff)|ff", &p->px, &p->py, &p->pz, &p->ax,
                          &p->ay, &p->az, &p->limit_min, &p->limit_max);
}

static int parse_cone_params(PyObject *args, ConstraintParams *p) {
  if (!args) {
    return 1;
  }
  // (Pivot), (TwistAxis), HalfAngle
  return PyArg_ParseTuple(args, "(fff)(fff)f", &p->px, &p->py, &p->pz, &p->ax,
                          &p->ay, &p->az, &p->half_cone_angle);
}

static int parse_distance_params(PyObject *args, ConstraintParams *p) {
  p->limit_min = 0.0f;
  p->limit_max = 10.0f;
  if (!args) {
    return 1;
  }
  // Min, Max
  return PyArg_ParseTuple(args, "ff", &p->limit_min, &p->limit_max);
}

// --- 2. Jolt Creator Helpers ---

static JPH_Constraint *create_fixed(const ConstraintParams *p, JPH_Body *b1,
                                    JPH_Body *b2) {
  JPH_FixedConstraintSettings s;
  JPH_FixedConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.autoDetectPoint = true;
  return (JPH_Constraint *)JPH_FixedConstraint_Create(&s, b1, b2);
}

static JPH_Constraint *create_point(const ConstraintParams *p, JPH_Body *b1,
                                    JPH_Body *b2) {
  JPH_PointConstraintSettings s;
  JPH_PointConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;
  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;
  return (JPH_Constraint *)JPH_PointConstraint_Create(&s, b1, b2);
}

static JPH_Constraint *create_hinge(const ConstraintParams *p, JPH_Body *b1,
                                    JPH_Body *b2) {
  JPH_HingeConstraintSettings s;
  JPH_HingeConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;

  // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
  if (len_sq < 1e-9f) {
    axis.x = 0.0f;
    axis.y = 1.0f;
    axis.z = 0.0f;
  } else {
    JPH_Vec3_Normalize(&axis, &axis);
  }

  JPH_Vec3 norm;
  vec3_get_perpendicular(&axis, &norm);

  s.hingeAxis1 = axis;
  s.hingeAxis2 = axis;
  s.normalAxis1 = norm;
  s.normalAxis2 = norm;
  s.limitsMin = p->limit_min;
  s.limitsMax = p->limit_max;

  return (JPH_Constraint *)JPH_HingeConstraint_Create(&s, b1, b2);
}

static JPH_Constraint *create_slider(const ConstraintParams *p, JPH_Body *b1,
                                     JPH_Body *b2) {
  JPH_SliderConstraintSettings s;
  JPH_SliderConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;
  s.autoDetectPoint = false;

  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;

  // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
  if (len_sq < 1e-9f) {
    axis.x = 0.0f;
    axis.y = 1.0f;
    axis.z = 0.0f;
  } else {
    JPH_Vec3_Normalize(&axis, &axis);
  }

  JPH_Vec3 norm;
  vec3_get_perpendicular(&axis, &norm);

  s.sliderAxis1 = axis;
  s.sliderAxis2 = axis;
  s.normalAxis1 = norm;
  s.normalAxis2 = norm;
  s.limitsMin = p->limit_min;
  s.limitsMax = p->limit_max;

  return (JPH_Constraint *)JPH_SliderConstraint_Create(&s, b1, b2);
}

static JPH_Constraint *create_cone(const ConstraintParams *p, JPH_Body *b1,
                                   JPH_Body *b2) {
  JPH_ConeConstraintSettings s;
  JPH_ConeConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  s.point1.x = p->px;
  s.point1.y = p->py;
  s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;

  // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
  if (len_sq < 1e-9f) {
    axis.x = 0.0f;
    axis.y = 1.0f;
    axis.z = 0.0f;
  } else {
    JPH_Vec3_Normalize(&axis, &axis);
  }

  s.twistAxis1 = axis;
  s.twistAxis2 = axis;
  s.halfConeAngle = p->half_cone_angle;

  return (JPH_Constraint *)JPH_ConeConstraint_Create(&s, b1, b2);
}

static JPH_Constraint *create_distance(const ConstraintParams *p, JPH_Body *b1,
                                       JPH_Body *b2) {
  JPH_DistanceConstraintSettings s;
  JPH_DistanceConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  // Check if the user provided a specific pivot point
  if (fabsf(p->px) > 1e-6f || fabsf(p->py) > 1e-6f || fabsf(p->pz) > 1e-6f) {
    s.point1.x = p->px;
    s.point1.y = p->py;
    s.point1.z = p->pz;
    s.point2 = s.point1;
  } else {
    // Fallback: Default to current body centers if no pivot was provided
    JPH_Body_GetPosition(b1, &s.point1);
    JPH_Body_GetPosition(b2, &s.point2);
  }

  s.minDistance = p->limit_min;
  s.maxDistance = p->limit_max;

  return (JPH_Constraint *)JPH_DistanceConstraint_Create(&s, b1, b2);
}

static void free_new_buffers(NewBuffers *nb) {
    PyMem_RawFree(nb->pos);  PyMem_RawFree(nb->rot);
    PyMem_RawFree(nb->ppos); PyMem_RawFree(nb->prot);
    PyMem_RawFree(nb->lvel); PyMem_RawFree(nb->avel);
    PyMem_RawFree(nb->bids); PyMem_RawFree(nb->udat);
    PyMem_RawFree(nb->gens); PyMem_RawFree(nb->s2d);
    PyMem_RawFree(nb->d2s);  PyMem_RawFree(nb->stat);
    PyMem_RawFree(nb->free); PyMem_RawFree(nb->cats);
    PyMem_RawFree(nb->masks); PyMem_RawFree(nb->mats);
}

static int alloc_new_buffers(NewBuffers *nb, size_t cap) {
    memset(nb, 0, sizeof(NewBuffers));
    size_t f4 = cap * 4 * sizeof(float);
    
    nb->pos = PyMem_RawMalloc(f4);  nb->rot = PyMem_RawMalloc(f4);
    nb->ppos = PyMem_RawMalloc(f4); nb->prot = PyMem_RawMalloc(f4);
    nb->lvel = PyMem_RawMalloc(f4); nb->avel = PyMem_RawMalloc(f4);

    nb->bids  = PyMem_RawMalloc(cap * sizeof(JPH_BodyID));
    nb->udat  = PyMem_RawMalloc(cap * sizeof(uint64_t));
    nb->gens  = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->s2d   = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->d2s   = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->stat  = PyMem_RawMalloc(cap * sizeof(uint8_t));
    nb->free  = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->cats  = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->masks = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->mats  = PyMem_RawMalloc(cap * sizeof(uint32_t));

    if (!nb->pos || !nb->rot || !nb->ppos || !nb->prot || !nb->lvel || !nb->avel ||
        !nb->bids || !nb->udat || !nb->gens || !nb->s2d || !nb->d2s || !nb->stat ||
        !nb->free || !nb->cats || !nb->masks || !nb->mats) {
        free_new_buffers(nb);
        return -1;
    }
    return 0;
}

static void migrate_and_init(PhysicsWorldObject *self, NewBuffers *nb, size_t new_cap) {
    size_t stride = 4 * sizeof(float);
    if (self->count > 0) {
        memcpy(nb->pos,  self->positions,         self->count * stride);
        memcpy(nb->rot,  self->rotations,         self->count * stride);
        memcpy(nb->ppos, self->prev_positions,    self->count * stride);
        memcpy(nb->prot, self->prev_rotations,    self->count * stride);
        memcpy(nb->lvel, self->linear_velocities, self->count * stride);
        memcpy(nb->avel, self->angular_velocities,self->count * stride);
        memcpy(nb->bids, self->body_ids,          self->count * sizeof(JPH_BodyID));
        memcpy(nb->udat, self->user_data,         self->count * sizeof(uint64_t));
        memcpy(nb->cats, self->categories,        self->count * sizeof(uint32_t));
        memcpy(nb->masks,self->masks,             self->count * sizeof(uint32_t));
        memcpy(nb->mats, self->material_ids,      self->count * sizeof(uint32_t));
    }

    memcpy(nb->gens, self->generations, self->slot_capacity * sizeof(uint32_t));
    memcpy(nb->s2d,  self->slot_to_dense, self->slot_capacity * sizeof(uint32_t));
    memcpy(nb->d2s,  self->dense_to_slot, self->slot_capacity * sizeof(uint32_t));
    memcpy(nb->stat, self->slot_states,   self->slot_capacity * sizeof(uint8_t));
    memcpy(nb->free, self->free_slots,    self->free_count * sizeof(uint32_t));

    for (size_t i = self->slot_capacity; i < new_cap; i++) {
        nb->gens[i] = 1;
        nb->stat[i] = SLOT_EMPTY;
        nb->free[self->free_count++] = (uint32_t)i;
    }
}

static int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
    // 1. Validation
    if (self->view_export_count > 0) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize while views are exported.");
        return -1;
    }
    BLOCK_UNTIL_NOT_QUERYING(self);
    if (new_capacity <= self->capacity) return 0;

    // 2. Transactional Allocation
    NewBuffers nb;
    if (alloc_new_buffers(&nb, new_capacity) < 0) {
        PyErr_NoMemory();
        return -1;
    }

    // 3. Data Migration
    migrate_and_init(self, &nb, new_capacity);

    // 4. Commit: Free OLD, assign NEW
    PyMem_RawFree(self->positions);          self->positions = nb.pos;
    PyMem_RawFree(self->rotations);          self->rotations = nb.rot;
    PyMem_RawFree(self->prev_positions);     self->prev_positions = nb.ppos;
    PyMem_RawFree(self->prev_rotations);     self->prev_rotations = nb.prot;
    PyMem_RawFree(self->linear_velocities);  self->linear_velocities = nb.lvel;
    PyMem_RawFree(self->angular_velocities); self->angular_velocities = nb.avel;
    
    PyMem_RawFree(self->body_ids);           self->body_ids = nb.bids;
    PyMem_RawFree(self->user_data);          self->user_data = nb.udat;
    PyMem_RawFree(self->generations);        self->generations = nb.gens;
    PyMem_RawFree(self->slot_to_dense);      self->slot_to_dense = nb.s2d;
    PyMem_RawFree(self->dense_to_slot);      self->dense_to_slot = nb.d2s;
    PyMem_RawFree(self->slot_states);        self->slot_states = nb.stat;
    PyMem_RawFree(self->free_slots);         self->free_slots = nb.free;
    PyMem_RawFree(self->categories);         self->categories = nb.cats;
    PyMem_RawFree(self->masks);              self->masks = nb.masks;
    PyMem_RawFree(self->material_ids);       self->material_ids = nb.mats;

    self->capacity = new_capacity;
    self->slot_capacity = new_capacity;
    return 0;
}

static PyObject *PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                PyObject *args,
                                                PyObject *kwds) {
  int type = 0;
  uint64_t h1 = 0;
  uint64_t h2 = 0;
  PyObject *params = NULL;
  static char *kwlist[] = {"type", "body1", "body2", "params", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iKK|O", kwlist, &type, &h1, &h2,
                                   &params)) {
    return NULL;
  }

  // NEW: Explicitly forbid self-constraints (Jolt requires two distinct bodies)
  if (h1 == h2) {
    PyErr_SetString(PyExc_ValueError,
                    "Cannot create a constraint between a body and itself");
    return NULL;
  }

  ConstraintParams p;
  params_init(&p);
  int parse_ok = 1;
  switch (type) {
  case CONSTRAINT_FIXED:
    break;
  case CONSTRAINT_POINT:
    parse_ok = parse_point_params(params, &p);
    break;
  case CONSTRAINT_HINGE:
    parse_ok = parse_hinge_params(params, &p);
    break;
  case CONSTRAINT_SLIDER:
    parse_ok = parse_slider_params(params, &p);
    break;
  case CONSTRAINT_CONE:
    parse_ok = parse_cone_params(params, &p);
    break;
  case CONSTRAINT_DISTANCE:
    parse_ok = parse_distance_params(params, &p);
    break;
  default:
    PyErr_SetString(PyExc_ValueError, "Unknown constraint type");
    return NULL;
  }
  if (!parse_ok) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t s1 = 0;
  uint32_t s2 = 0;
  if (!unpack_handle(self, h1, &s1) || self->slot_states[s1] != SLOT_ALIVE ||
      !unpack_handle(self, h2, &s2) || self->slot_states[s2] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid body handles");
    return NULL;
  }

  JPH_BodyID bid1 = self->body_ids[self->slot_to_dense[s1]];
  JPH_BodyID bid2 = self->body_ids[self->slot_to_dense[s2]];

  if (self->free_constraint_count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_MemoryError, "Max constraints reached");
    return NULL;
  }
  uint32_t c_slot = self->free_constraint_slots[--self->free_constraint_count];
  SHADOW_UNLOCK(&self->shadow_lock);

  const JPH_BodyLockInterface *lock_iface =
      JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockWrite lock1;
  JPH_BodyLockWrite lock2;

  // Sort for Deadlock Prevention
  if (bid1 < bid2) {
    JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
    JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
  } else {
    JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
    JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
  }

  JPH_Constraint *constraint = NULL;
  if (lock1.body && lock2.body) {
    // Resolve pointers (Safe because h1 != h2 guaranteed earlier)
    JPH_Body *b1 =
        (JPH_Body_GetID(lock1.body) == bid1) ? lock1.body : lock2.body;
    JPH_Body *b2 =
        (JPH_Body_GetID(lock1.body) == bid2) ? lock1.body : lock2.body;

    switch (type) {
    case CONSTRAINT_FIXED:
      constraint = create_fixed(&p, b1, b2);
      break;
    case CONSTRAINT_POINT:
      constraint = create_point(&p, b1, b2);
      break;
    case CONSTRAINT_HINGE:
      constraint = create_hinge(&p, b1, b2);
      break;
    case CONSTRAINT_SLIDER:
      constraint = create_slider(&p, b1, b2);
      break;
    case CONSTRAINT_CONE:
      constraint = create_cone(&p, b1, b2);
      break;
    case CONSTRAINT_DISTANCE:
      constraint = create_distance(&p, b1, b2);
      break;
    default:
      break; // Already handled above
    }
  }

  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock1);
  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock2);

  if (!constraint) {
    SHADOW_LOCK(&self->shadow_lock);
    self->free_constraint_slots[self->free_constraint_count++] = c_slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_RuntimeError,
                    "Jolt failed to create constraint instance");
    return NULL;
  }

  JPH_PhysicsSystem_AddConstraint(self->system, constraint);

  SHADOW_LOCK(&self->shadow_lock);
  self->constraints[c_slot] = constraint;
  self->constraint_states[c_slot] = SLOT_ALIVE;
  uint32_t gen = self->constraint_generations[c_slot];
  ConstraintHandle handle = ((uint64_t)gen << 32) | c_slot;
  SHADOW_UNLOCK(&self->shadow_lock);

  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject *PhysicsWorld_destroy_constraint(PhysicsWorldObject *self,
                                                 PyObject *args,
                                                 PyObject *kwds) {
  uint64_t h = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
    return NULL;
  }

  JPH_Constraint *c_to_destroy = NULL;

  // --- 1. RESOLUTION PHASE (Inside Shadow Lock) ---
  SHADOW_LOCK(&self->shadow_lock);

  // Guard against both Physics Step AND active Queries
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(h >> 32);

  // Validate identity
  if (slot >= self->constraint_capacity ||
      self->constraint_generations[slot] != gen ||
      self->constraint_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
    return NULL;
  }

  // Capture the pointer and IMMEDIATELY invalidate the slot
  c_to_destroy = self->constraints[slot];

  self->constraints[slot] = NULL;
  self->constraint_states[slot] = SLOT_EMPTY;
  self->constraint_generations[slot]++; // Increment generation to invalidate
                                        // stale handles
  self->free_constraint_slots[self->free_constraint_count++] = slot;

  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT DESTRUCTION PHASE (Outside Shadow Lock) ---
  // No Shadow-vs-Jolt deadlocks possible here!
  if (c_to_destroy) {
    // Automatic Body Wake-up
    // This is a "nice to have" - prevents objects from hanging in the air
    // when the joint holding them is deleted.
    if (JPH_Constraint_GetType(c_to_destroy) ==
        JPH_ConstraintType_TwoBodyConstraint) {
      JPH_TwoBodyConstraint *tbc = (JPH_TwoBodyConstraint *)c_to_destroy;
      JPH_Body *b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
      JPH_Body *b2 = JPH_TwoBodyConstraint_GetBody2(tbc);

      // JPH_BodyInterface_ActivateBody is thread-safe
      if (b1) {
        JPH_BodyInterface_ActivateBody(self->body_interface,
                                       JPH_Body_GetID(b1));
      }
      if (b2) {
        JPH_BodyInterface_ActivateBody(self->body_interface,
                                       JPH_Body_GetID(b2));
      }
    }

    // Remove and Destroy
    JPH_PhysicsSystem_RemoveConstraint(self->system, c_to_destroy);
    JPH_Constraint_Destroy(c_to_destroy);
  }

  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_save_state(PhysicsWorldObject *self,
                                         PyObject *Py_UNUSED(unused)) {
  SHADOW_LOCK(&self->shadow_lock);

  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // 1. Unambiguous Size Calculation
  size_t header_size = sizeof(size_t) /* count */ + sizeof(double) /* time */ +
                       sizeof(size_t) /* slot_capacity */;

  size_t dense_stride = 4 * sizeof(float); // 16 bytes
  size_t dense_size = self->count * 4 /* arrays */ * dense_stride;

  size_t mapping_size =
      self->slot_capacity *
      (sizeof(uint32_t) * 3 /* gen, s2d, d2s */ + sizeof(uint8_t) /* states */
      );

  size_t total_size = header_size + dense_size + mapping_size;
  PyObject *bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_size);
  if (!bytes) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return NULL;
  }

  char *ptr = PyBytes_AsString(bytes);

  // 2. Encode Header
  memcpy(ptr, &self->count, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, &self->time, sizeof(double));
  ptr += sizeof(double);
  memcpy(ptr, &self->slot_capacity, sizeof(size_t));
  ptr += sizeof(size_t);

  // 3. Encode Dense Buffers
  memcpy(ptr, self->positions, self->count * dense_stride);
  ptr += self->count * dense_stride;
  memcpy(ptr, self->rotations, self->count * dense_stride);
  ptr += self->count * dense_stride;
  memcpy(ptr, self->linear_velocities, self->count * dense_stride);
  ptr += self->count * dense_stride;
  memcpy(ptr, self->angular_velocities, self->count * dense_stride);
  ptr += self->count * dense_stride;

  // 4. Encode Mapping Tables
  memcpy(ptr, self->generations, self->slot_capacity * sizeof(uint32_t));
  ptr += self->slot_capacity * sizeof(uint32_t);
  memcpy(ptr, self->slot_to_dense, self->slot_capacity * sizeof(uint32_t));
  ptr += self->slot_capacity * sizeof(uint32_t);
  memcpy(ptr, self->dense_to_slot, self->slot_capacity * sizeof(uint32_t));
  ptr += self->slot_capacity * sizeof(uint32_t);
  memcpy(ptr, self->slot_states, self->slot_capacity);

  SHADOW_UNLOCK(&self->shadow_lock);
  return bytes;
}

static PyObject *PhysicsWorld_load_state(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  Py_buffer view;
  static char *kwlist[] = {"state", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*", kwlist, &view)) {
    return NULL;
  }

  // IMMEDIATE SNAPSHOT (Unlocked, GIL held)
  // We copy the data to the C heap so we can release the Python object
  // before we start yielding/waiting.
  void *local_state_copy = PyMem_RawMalloc(view.len);
  if (!local_state_copy) {
    PyBuffer_Release(&view);
    return PyErr_NoMemory();
  }
  memcpy(local_state_copy, view.buf, view.len);
  size_t total_len = (size_t)view.len;

  // Release the Python buffer immediately. We don't need it anymore.
  PyBuffer_Release(&view);

  SHADOW_LOCK(&self->shadow_lock);

  // 1. CONCURRENCY GUARD
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // 2. HEADER VALIDATION
  if ((size_t)view.len < (sizeof(size_t) * 2 + sizeof(double))) {
    goto size_fail;
  }

  char *ptr = (char *)local_state_copy;
  if (total_len < (sizeof(size_t) * 2 + sizeof(double))) {
    goto size_fail;
  }
  size_t saved_count = 0;
  size_t saved_slot_cap = 0;
  double saved_time = NAN;

  memcpy(&saved_count, ptr, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(&saved_time, ptr, sizeof(double));
  ptr += sizeof(double);
  memcpy(&saved_slot_cap, ptr, sizeof(size_t));
  ptr += sizeof(size_t);

  // CRITICAL: Prevent memory corruption by verifying slot capacity matches
  // exactly
  if (saved_slot_cap != self->slot_capacity) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_Format(PyExc_ValueError,
                 "Capacity mismatch: World is %zu, Snapshot is %zu",
                 self->slot_capacity, saved_slot_cap);
    return NULL;
  }

  // 3. FULL SIZE VALIDATION
  size_t dense_stride = 16;
  size_t expected = (sizeof(size_t) * 2 + sizeof(double)) +
                    (saved_count * 4 * dense_stride) +
                    (saved_slot_cap * (4 * 3 + 1));

  if ((size_t)view.len != expected) {
    goto size_fail;
  }

  // 4. RESTORE SHADOW STATE
  self->count = saved_count;
  self->time = saved_time;
  self->view_shape[0] = (Py_ssize_t)self->count;

  memcpy(self->positions, ptr, self->count * dense_stride);
  ptr += self->count * dense_stride;
  memcpy(self->rotations, ptr, self->count * dense_stride);
  ptr += self->count * dense_stride;
  memcpy(self->linear_velocities, ptr, self->count * dense_stride);
  ptr += self->count * dense_stride;
  memcpy(self->angular_velocities, ptr, self->count * dense_stride);
  ptr += self->count * dense_stride;

  memcpy(self->generations, ptr, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(self->slot_to_dense, ptr, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(self->dense_to_slot, ptr, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(self->slot_states, ptr, self->slot_capacity);

  // 5. HANDLE INVALIDATION (Option A)
  // We increment every generation. This ensures that any Python handle created
  // BEFORE the load becomes invalid AFTER the load.
  for (size_t i = 0; i < self->slot_capacity; i++) {
    self->generations[i]++;
  }

  // 6. REBUILD FREE LIST
  self->free_count = 0;
  for (uint32_t i = 0; i < (uint32_t)self->slot_capacity; i++) {
    if (self->slot_states[i] == SLOT_EMPTY) {
      self->free_slots[self->free_count++] = i;
    }
  }

  // 7. HANDLE TOPOLOGY SHRINK
  // If the saved state has fewer bodies than the current Jolt system,
  // we must deactivate the "extra" bodies that are no longer in the dense map.
  JPH_BodyInterface *bi = self->body_interface;
  for (size_t i = self->count; i < self->capacity; i++) {
    if (self->body_ids[i] != JPH_INVALID_BODY_ID) {
      JPH_BodyInterface_DeactivateBody(bi, self->body_ids[i]);
    }
  }

  SHADOW_UNLOCK(&self->shadow_lock);

  // 8. JOLT SYNC (Unlocked to prevent deadlocks)
  for (size_t i = 0; i < self->count; i++) {
    JPH_BodyID bid = self->body_ids[i];
    if (bid == JPH_INVALID_BODY_ID) {
      continue;
    }

    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = (double)self->positions[i * 4];
    p->y = (double)self->positions[i * 4 + 1];
    p->z = (double)self->positions[i * 4 + 2];
    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = self->rotations[i * 4];
    q->y = self->rotations[i * 4 + 1];
    q->z = self->rotations[i * 4 + 2];
    q->w = self->rotations[i * 4 + 3];

    JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q,
                                             JPH_Activation_Activate);
    JPH_BodyInterface_SetLinearVelocity(
        bi, bid, (JPH_Vec3 *)&self->linear_velocities[i * 4]);
    JPH_BodyInterface_SetAngularVelocity(
        bi, bid, (JPH_Vec3 *)&self->angular_velocities[i * 4]);

    // Re-Sync UserData to the newly incremented generations
    BodyHandle new_h = make_handle(self->dense_to_slot[i],
                                   self->generations[self->dense_to_slot[i]]);
    JPH_BodyInterface_SetUserData(bi, bid, (uint64_t)new_h);
  }

  PyMem_RawFree(local_state_copy);
  Py_RETURN_NONE;

size_fail:
  SHADOW_UNLOCK(&self->shadow_lock);
  PyMem_RawFree(local_state_copy);
  PyErr_SetString(PyExc_ValueError,
                  "Snapshot buffer truncated or capacity mismatch");
  return NULL;
}

static PyObject *PhysicsWorld_step(PhysicsWorldObject *self, PyObject *args) {
  float dt = 1.0f / 60.0f;
  if (UNLIKELY(
          !PyArg_ParseTuple(args, "|f", &dt))) { // UNLIKELY fail to parse args
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // 1. RE-ENTRANCY GUARD
  if (UNLIKELY(atomic_load_explicit(
          &self->is_stepping,
          memory_order_relaxed))) { // UNLIKELY concurrent call
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_RuntimeError, "Concurrent step detected");
    return NULL;
  }
  BLOCK_UNTIL_NOT_QUERYING(self);
  atomic_store_explicit(&self->is_stepping, true, memory_order_relaxed);

  // 2. BUFFER MANAGEMENT (Reset Phase)
  if (UNLIKELY(!self->contact_buffer)) { // UNLIKELY re-allocation needed
    self->contact_max_capacity = 4096;
    self->contact_buffer =
        PyMem_RawMalloc(self->contact_max_capacity * sizeof(ContactEvent));
    if (UNLIKELY(!self->contact_buffer)) { // UNLIKELY OOM
      atomic_store_explicit(&self->is_stepping, false, memory_order_relaxed);
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
    }
  }
  atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

  // 3. FLUSH COMMANDS
  // Note: Assuming command_count > 0 is LIKELY if user queues anything
  flush_commands(self);

  // Snapshot state for interpolation (always done)
  memcpy(self->prev_positions, self->positions, self->count * 16);
  memcpy(self->prev_rotations, self->rotations, self->count * 16);

  SHADOW_UNLOCK(&self->shadow_lock);

  // 4. JOLT UPDATE (Unlocked)
  Py_BEGIN_ALLOW_THREADS JPH_PhysicsSystem_Update(self->system, dt, 1,
                                                  self->job_system);
  Py_END_ALLOW_THREADS

      // 5. ACQUIRE FENCE (Consumer Phase)
      atomic_thread_fence(memory_order_acquire);

  SHADOW_LOCK(&self->shadow_lock);

  // 6. SYNC SHADOW BUFFERS
  culverin_sync_shadow_buffers(self);

  // 7. FINALIZE COUNT
  size_t count =
      atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);

  if (UNLIKELY(count > self->contact_max_capacity)) { // UNLIKELY overflow
    count = self->contact_max_capacity;
  }
  self->contact_count = count;

  atomic_store_explicit(&self->is_stepping, false, memory_order_relaxed);
  self->time += (double)dt;

  SHADOW_UNLOCK(&self->shadow_lock);

  Py_RETURN_NONE;
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
static PyObject *PhysicsWorld_create_character(PhysicsWorldObject *self,
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

static PyObject *PhysicsWorld_create_convex_hull(PhysicsWorldObject *self,
                                                 PyObject *args, PyObject *kwds) {
  float px = 0;
  float py = 0;
  float pz = 0;
  float rx = 0;
  float ry = 0;
  float rz = 0;
  float rw = 1.0f;
  
  Py_buffer points_view = {0};
  uint64_t user_data = 0;
  
  int motion_type = 2; // Dynamic by default for hulls!
  float mass = -1.0f;
  uint32_t category = 0xFFFF;
  uint32_t mask = 0xFFFF;
  uint32_t material_id = 0;
  float friction = 0.2f;
  float restitution = 0.0f;
  int use_ccd = 0;

  static char *kwlist[] = {
      "pos", "rot", "points", 
      "motion", "mass", "user_data", 
      "category", "mask", "material_id", 
      "friction", "restitution", "ccd", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "(fff)(ffff)y*|ifKIIffp", kwlist, 
          &px, &py, &pz, &rx, &ry, &rz, &rw, 
          &points_view, &motion_type, &mass, &user_data,
          &category, &mask, &material_id, 
          &friction, &restitution, &use_ccd)) {
    return NULL;
  }

  // 1. Buffer Validation
  if (points_view.len % (3 * sizeof(float)) != 0) {
    PyBuffer_Release(&points_view);
    return PyErr_Format(PyExc_ValueError, "Points buffer must be 3 * float32");
  }

  size_t num_points = points_view.len / (3 * sizeof(float));
  if (num_points < 3) {
    PyBuffer_Release(&points_view);
    return PyErr_Format(PyExc_ValueError, "Convex Hull requires at least 3 points");
  }

  // 2. Convert to Jolt format
  // We copy to a temporary C array because JPH_Vec3 alignment might differ from packed floats
  JPH_Vec3* jolt_points = PyMem_RawMalloc(num_points * sizeof(JPH_Vec3));
  if (!jolt_points) {
      PyBuffer_Release(&points_view);
      return PyErr_NoMemory();
  }

  float* raw_floats = (float*)points_view.buf;
  for (size_t i = 0; i < num_points; i++) {
      jolt_points[i].x = raw_floats[i*3 + 0];
      jolt_points[i].y = raw_floats[i*3 + 1];
      jolt_points[i].z = raw_floats[i*3 + 2];
  }
  PyBuffer_Release(&points_view); // Done with Python object

  // 3. Create Shape (Unlocked - Heavy Math)
  // 0.05f is the standard convex radius "shrink" to improve performance
  JPH_ConvexHullShapeSettings* hull_settings = 
      JPH_ConvexHullShapeSettings_Create(jolt_points, (uint32_t)num_points, 0.05f);
  
  PyMem_RawFree(jolt_points); // Free temp buffer

  if (!hull_settings) {
      return PyErr_Format(PyExc_RuntimeError, "Failed to allocate Hull Settings");
  }

  JPH_Shape* shape = (JPH_Shape*)JPH_ConvexHullShapeSettings_CreateShape(hull_settings);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)hull_settings);

  if (!shape) {
      return PyErr_Format(PyExc_RuntimeError, 
          "Failed to build Convex Hull. Points might be coplanar or degenerate.");
  }

  // 4. World Registration (Locked)
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_Shape_Destroy(shape); // Clean up orphaned shape
      return NULL;
    }
  }

  uint32_t slot = self->free_slots[--self->free_count];
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)px; pos->y = (double)py; pos->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx; rot->y = ry; rot->z = rz; rot->w = rw;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, (JPH_MotionType)motion_type, 
      (motion_type == 0) ? 0 : 1); // Layer 0 for Static, 1 for Moving
  JPH_Shape_Destroy(shape);

  // Mass Override
  if (mass > 0.0f) {
      JPH_MassProperties mp;
      JPH_Shape_GetMassProperties(shape, &mp);
      float scale = mass / mp.mass;
      mp.mass = mass;
      for(int i=0; i<3; i++) {
          mp.inertia.column[i].x *= scale;
          mp.inertia.column[i].y *= scale;
          mp.inertia.column[i].z *= scale;
      }
      JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
      JPH_BodyCreationSettings_SetOverrideMassProperties(
          settings, JPH_OverrideMassProperties_CalculateInertia);
  }

  JPH_BodyCreationSettings_SetFriction(settings, friction);
  JPH_BodyCreationSettings_SetRestitution(settings, restitution);
  if (use_ccd) JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);

  uint32_t gen = self->generations[slot];
  BodyHandle handle = make_handle(slot, gen);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  if (!ensure_command_capacity(self)) {
      JPH_BodyCreationSettings_Destroy(settings);
      settings = NULL;
      JPH_Shape_Destroy(shape); 
      shape = NULL;
      self->slot_states[slot] = SLOT_EMPTY;
      self->free_slots[self->free_count++] = slot;
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
  cmd->create.settings = settings;
  cmd->create.user_data = user_data;
  cmd->create.category = category;
  cmd->create.mask = mask;
  cmd->create.material_id = material_id;

  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLongLong(handle);
}

// Helper 1: Build the Jolt Compound Shape from the Python parts list
static JPH_Shape *init_compound_shape(PhysicsWorldObject *self,
                                      PyObject *parts) {
  JPH_StaticCompoundShapeSettings *compound_settings =
      JPH_StaticCompoundShapeSettings_Create();
  JPH_CompoundShapeSettings *base_settings =
      (JPH_CompoundShapeSettings *)compound_settings;

  Py_ssize_t num_parts = PyList_Size(parts);
  for (Py_ssize_t i = 0; i < num_parts; i++) {
    PyObject *item = PyList_GetItem(parts, i);
    if (!PyTuple_Check(item) || PyTuple_Size(item) != 4)
      goto fail;

    PyObject *p_pos = PyTuple_GetItem(item, 0);
    PyObject *p_rot = PyTuple_GetItem(item, 1);
    int type = (int)PyLong_AsLong(PyTuple_GetItem(item, 2));
    PyObject *p_size = PyTuple_GetItem(item, 3);

    // Parse Vectors and Params
    JPH_Vec3 local_p = {0};
    JPH_Quat local_q = {0, 0, 0, 1};
    float params[4] = {0};

    if (PyTuple_Check(p_pos) && PyTuple_Size(p_pos) == 3) {
      local_p.x = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 0));
      local_p.y = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 1));
      local_p.z = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 2));
    }
    if (PyTuple_Check(p_rot) && PyTuple_Size(p_rot) == 4) {
      local_q.x = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 0));
      local_q.y = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 1));
      local_q.z = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 2));
      local_q.w = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 3));
    }
    if (PyTuple_Check(p_size)) {
      for (int j = 0; j < 4 && j < PyTuple_Size(p_size); j++)
        params[j] = (float)PyFloat_AsDouble(PyTuple_GetItem(p_size, j));
    } else {
      params[0] = (float)PyFloat_AsDouble(p_size);
    }

    JPH_Shape *sub_shape = find_or_create_shape(self, type, params);
    if (!sub_shape)
      goto fail;

    JPH_CompoundShapeSettings_AddShape2(base_settings, &local_p, &local_q,
                                        sub_shape, 0);
  }

  JPH_Shape *final_shape =
      (JPH_Shape *)JPH_StaticCompoundShape_Create(compound_settings);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)compound_settings);
  return final_shape;

fail:
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)compound_settings);
  return NULL;
}

// Helper 2: Apply physics properties (mass, friction, etc) to creation settings
static void apply_body_creation_props(JPH_BodyCreationSettings *settings,
                                      JPH_Shape *shape, 
                                      BodyCreationProps props) {
    if (props.mass > 0.0f) {
        JPH_MassProperties mp;
        JPH_Shape_GetMassProperties(shape, &mp);
        if (mp.mass > 1e-6f) {
            float scale = props.mass / mp.mass;
            mp.mass = props.mass;
            for (int i = 0; i < 3; i++) {
                mp.inertia.column[i].x *= scale;
                mp.inertia.column[i].y *= scale;
                mp.inertia.column[i].z *= scale;
            }
            JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
            JPH_BodyCreationSettings_SetOverrideMassProperties(
                settings, JPH_OverrideMassProperties_CalculateInertia);
        }
    }
    
    if (props.is_sensor)
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    
    if (props.use_ccd)
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
        
    JPH_BodyCreationSettings_SetFriction(settings, props.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, props.restitution);
}

// Orchestrator
static PyObject *PhysicsWorld_create_compound_body(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
  float px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0, rw = 1.0f, mass = -1.0f,
        friction = 0.2f, restitution = 0.0f;
  int motion_type = 2, is_sensor = 0, use_ccd = 0;
  uint64_t user_data = 0;
  uint32_t category = 0xFFFF, mask = 0xFFFF, material_id = 0;
  PyObject *parts = NULL;
  static char *kwlist[] = {"pos",  "rot",         "parts",     "motion",
                           "mass", "user_data",   "is_sensor", "category",
                           "mask", "material_id", "friction",  "restitution",
                           "ccd",  NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "(fff)(ffff)O|ifKpIIffp", kwlist, &px, &py, &pz, &rx, &ry,
          &rz, &rw, &parts, &motion_type, &mass, &user_data, &is_sensor,
          &category, &mask, &material_id, &friction, &restitution, &use_ccd))
    return NULL;

  if (!PyList_Check(parts))
    return PyErr_Format(PyExc_TypeError, "Parts must be a list");

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  if (self->free_count == 0 &&
      PhysicsWorld_resize(self, self->capacity * 2) < 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return NULL;
  }

  // Ref Count = 1 (Created)
  JPH_Shape *final_shape = init_compound_shape(self, parts);
  if (!final_shape) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_RuntimeError, "Failed to create Compound Shape");
  }

  uint32_t slot = self->free_slots[--self->free_count];
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  // Ref Count -> 2 (Held by Settings)
  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      final_shape, &(JPH_RVec3){(double)px, (double)py, (double)pz},
      &(JPH_Quat){rx, ry, rz, rw}, (JPH_MotionType)motion_type,
      (motion_type == 0) ? 0 : 1);

  // FIX: Release our local reference. 
  // The 'settings' object now owns the shape. 
  // When 'settings' is destroyed in flush_commands, it releases the shape.
  // If the body was created, the Body owns the shape.
  JPH_Shape_Destroy(final_shape); 

  BodyCreationProps props = {
    .mass = mass,
    .friction = friction,
    .restitution = restitution,
    .is_sensor = is_sensor,
    .use_ccd = use_ccd
  };

  apply_body_creation_props(settings, final_shape, props);
  JPH_BodyCreationSettings_SetUserData(
      settings, (uint64_t)make_handle(slot, self->generations[slot]));

  if (!ensure_command_capacity(self)) {
    JPH_BodyCreationSettings_Destroy(settings); 
    // ^ This releases the shape ref (Ref -> 0), effectively destroying the shape correctly.
    
    self->slot_states[slot] = SLOT_EMPTY;
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
  cmd->create.settings = settings;
  cmd->create.user_data = user_data;
  cmd->create.category = category;
  cmd->create.mask = mask;
  cmd->create.material_id = material_id;

  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLongLong(
      make_handle(slot, self->generations[slot]));
}

// Helper 1: Resolve material properties based on ID and explicit overrides
static MaterialSettings resolve_material_params(PhysicsWorldObject *self,
                                                uint32_t material_id,
                                                MaterialSettings input) {
    // 1. Start with Jolt Defaults
    float f = 0.2f, r = 0.0f;

    // 2. Lookup Registry Defaults
    if (material_id > 0) {
        SHADOW_LOCK(&self->shadow_lock);
        for (size_t i = 0; i < self->material_count; i++) {
            if (self->materials[i].id == material_id) {
                f = self->materials[i].friction;
                r = self->materials[i].restitution;
                break;
            }
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }

    // 3. Apply Overrides (if input values are non-negative)
    MaterialSettings resolved;
    resolved.friction = (input.friction >= 0.0f) ? input.friction : f;
    resolved.restitution = (input.restitution >= 0.0f) ? input.restitution : r;

    return resolved;
}

// Helper 2: Parse the size object (tuple or float) into a 4-float array
static void parse_body_size(PyObject *py_size, float s[4]) {
  s[0] = 1.0f;
  s[1] = 1.0f;
  s[2] = 1.0f;
  s[3] = 0.0f; // Defaults
  if (!py_size || py_size == Py_None)
    return;
  if (PyTuple_Check(py_size)) {
    Py_ssize_t sz_len = PyTuple_Size(py_size);
    for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
      PyObject *item = PyTuple_GetItem(py_size, i);
      if (PyNumber_Check(item))
        s[i] = (float)PyFloat_AsDouble(item);
    }
  } else if (PyNumber_Check(py_size)) {
    s[0] = (float)PyFloat_AsDouble(py_size);
  }
}

// Helper 3: Apply mass, sensor, CCD, and sleeping settings to the creation
// struct
static void configure_body_settings(JPH_BodyCreationSettings *settings,
                                    JPH_Shape *shape, 
                                    BodyConfig cfg) {
  // Use the members of the struct instead of loose variables
  if (cfg.is_sensor)
    JPH_BodyCreationSettings_SetIsSensor(settings, true);
  
  if (cfg.use_ccd)
    JPH_BodyCreationSettings_SetMotionQuality(settings,
                                              JPH_MotionQuality_LinearCast);
  
  if (cfg.motion_type == 2) // MOTION_DYNAMIC
    JPH_BodyCreationSettings_SetAllowSleeping(settings, true);

  JPH_BodyCreationSettings_SetFriction(settings, cfg.friction);
  JPH_BodyCreationSettings_SetRestitution(settings, cfg.restitution);

  if (cfg.mass > 0.0f) {
    JPH_MassProperties mp;
    JPH_Shape_GetMassProperties(shape, &mp);
    float scale = cfg.mass / fmaxf(mp.mass, 1e-6f);
    mp.mass = cfg.mass;
    for (int i = 0; i < 3; i++) {
      mp.inertia.column[i].x *= scale;
      mp.inertia.column[i].y *= scale;
      mp.inertia.column[i].z *= scale;
    }
    JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
    JPH_BodyCreationSettings_SetOverrideMassProperties(
        settings, JPH_OverrideMassProperties_CalculateInertia);
  }
}

// Main Orchestrator
static PyObject *PhysicsWorld_create_body(PhysicsWorldObject *self,
                                          PyObject *args, PyObject *kwds) {
  float px = 0.0f, py = 0.0f, pz = 0.0f, rx = 0.0f, ry = 0.0f, rz = 0.0f,
        rw = 1.0f, mass = -1.0f, friction = -1.0f, restitution = -1.0f;
  int shape_type = 0, motion_type = 2, is_sensor = 0, use_ccd = 0;
  uint32_t category = 0xFFFF, mask = 0xFFFF, material_id = 0;
  unsigned long long user_data = 0;
  PyObject *py_size = NULL;
  static char *kwlist[] = {
      "pos",       "rot",         "size",        "shape",    "motion",
      "user_data", "is_sensor",   "mass",        "category", "mask",
      "friction",  "restitution", "material_id", "ccd",      NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "|(fff)(ffff)OiiKpfIIffIp", kwlist, &px, &py, &pz, &rx,
          &ry, &rz, &rw, &py_size, &shape_type, &motion_type, &user_data,
          &is_sensor, &mass, &category, &mask, &friction, &restitution,
          &material_id, &use_ccd))
    return NULL;

  if (shape_type == 4 && motion_type != 0) {
    return PyErr_Format(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
  }

  MaterialSettings input = { .friction = friction, .restitution = restitution };

  // Resolve
  MaterialSettings mat = resolve_material_params(self, material_id, input);
  float s[4];
  parse_body_size(py_size, s);

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  if (self->free_count == 0 &&
      PhysicsWorld_resize(self, self->capacity * 2) < 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return NULL;
  }

  JPH_Shape *shape = find_or_create_shape(self, shape_type, s);
  if (!shape) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_RuntimeError, "Failed to create shape");
  }

  uint32_t slot = self->free_slots[--self->free_count];
  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, &(JPH_RVec3){(double)px, (double)py, (double)pz},
      &(JPH_Quat){rx, ry, rz, rw}, (JPH_MotionType)motion_type,
      (motion_type == 0) ? 0 : 1);

  BodyConfig config = {
    .mass = mass,
    .friction = friction,
    .restitution = restitution,
    .is_sensor = is_sensor,
    .use_ccd = use_ccd,
    .motion_type = motion_type
  };

  configure_body_settings(settings, shape, config);
  JPH_BodyCreationSettings_SetUserData(
      settings, (uint64_t)make_handle(slot, self->generations[slot]));

  if (!ensure_command_capacity(self)) {
    JPH_BodyCreationSettings_Destroy(settings);
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
  cmd->create.settings = settings;
  cmd->create.user_data = (uint64_t)user_data;
  cmd->create.category = category;
  cmd->create.mask = mask;
  cmd->create.material_id = material_id;

  self->slot_states[slot] = SLOT_PENDING_CREATE;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLongLong(
      make_handle(slot, self->generations[slot]));
}

/**
 * Helper 1: Build the Jolt triangle array while verifying index bounds.
 */
static JPH_IndexedTriangle *build_mesh_triangles(const uint32_t *raw,
                                                 uint32_t tri_count,
                                                 uint32_t vertex_count) {
  JPH_IndexedTriangle *jolt_tris = (JPH_IndexedTriangle *)PyMem_RawMalloc(
      tri_count * sizeof(JPH_IndexedTriangle));
  if (!jolt_tris) {
    PyErr_NoMemory();
    return NULL;
  }

  for (uint32_t t = 0; t < tri_count; t++) {
    uint32_t i1 = raw[t * 3 + 0], i2 = raw[t * 3 + 1], i3 = raw[t * 3 + 2];

    if (i1 >= vertex_count || i2 >= vertex_count || i3 >= vertex_count) {
      PyMem_RawFree(jolt_tris);
      PyErr_Format(PyExc_ValueError, "Mesh index out of range: %u/%u/%u >= %u",
                   i1, i2, i3, vertex_count);
      return NULL;
    }

    jolt_tris[t].i1 = i1;
    jolt_tris[t].i2 = i2;
    jolt_tris[t].i3 = i3;
    jolt_tris[t].materialIndex = 0;
    jolt_tris[t].userData = 0;
  }
  return jolt_tris;
}

/**
 * Helper 2: Encapsulate Jolt Mesh creation (Settings -> BVH build -> Shape).
 */
static JPH_Shape *build_mesh_shape(const void *v_data, uint32_t v_count,
                                   JPH_IndexedTriangle *tris,
                                   uint32_t t_count) {
  JPH_MeshShapeSettings *mss =
      JPH_MeshShapeSettings_Create2((JPH_Vec3 *)v_data, v_count, tris, t_count);
  if (!mss) {
    PyErr_SetString(PyExc_RuntimeError, "Jolt MeshSettings allocation failed");
    return NULL;
  }

  JPH_Shape *shape = (JPH_Shape *)JPH_MeshShapeSettings_CreateShape(mss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)mss);

  if (!shape) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Jolt Mesh BVH build failed (Triangle data degenerate?)");
  }
  return shape;
}

/**
 * Main Orchestrator
 */
static PyObject *PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  Py_buffer v_view = {0}, i_view = {0};
  float px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0, rw = 1.0f;
  unsigned long long user_data = 0;
  uint32_t cat = 0xFFFF, mask = 0xFFFF;
  static char *kwlist[] = {"pos",       "rot",      "vertices", "indices",
                           "user_data", "category", "mask",     NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(ffff)y*y*|KII", kwlist,
                                   &px, &py, &pz, &rx, &ry, &rz, &rw, &v_view,
                                   &i_view, &user_data, &cat, &mask))
    return NULL;

  // 1. Validation
  if (v_view.len % 12 != 0 || i_view.len % 12 != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "Buffer size mismatch: must be multiples of 12 bytes");
    goto cleanup;
  }

  uint32_t v_count = (uint32_t)(v_view.len / 12);
  uint32_t t_count = (uint32_t)(i_view.len / 12);

  // 2. Triangle processing
  JPH_IndexedTriangle *tris =
      build_mesh_triangles((uint32_t *)i_view.buf, t_count, v_count);
  if (!tris)
    goto cleanup;

  // 3. Jolt Shape Build
  JPH_Shape *shape = build_mesh_shape(v_view.buf, v_count, tris, t_count);
  PyMem_RawFree(tris);
  if (!shape)
    goto cleanup;

  // 4. World Reservation & Command Queuing
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  if (self->free_count == 0 &&
      PhysicsWorld_resize(self, self->capacity + 1024) < 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    goto cleanup;
  }

  uint32_t slot = self->free_slots[--self->free_count];
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, &(JPH_RVec3){(double)px, (double)py, (double)pz},
      &(JPH_Quat){rx, ry, rz, rw}, JPH_MotionType_Static, 0);

  JPH_Shape_Destroy(shape); 

  BodyHandle handle = make_handle(slot, self->generations[slot]);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  if (!ensure_command_capacity(self)) {
    JPH_BodyCreationSettings_Destroy(settings);
    self->slot_states[slot] = SLOT_EMPTY;
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_NoMemory();
    goto cleanup;
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
  cmd->create.settings = settings;
  cmd->create.user_data = user_data;
  cmd->create.category = cat;
  cmd->create.mask = mask;

  SHADOW_UNLOCK(&self->shadow_lock);
  PyBuffer_Release(&v_view);
  PyBuffer_Release(&i_view);
  return PyLong_FromUnsignedLongLong(handle);

cleanup:
  if (v_view.obj)
    PyBuffer_Release(&v_view);
  if (i_view.obj)
    PyBuffer_Release(&i_view);
  return NULL;
}

static PyObject *PhysicsWorld_destroy_body(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // 1. MUTATION GUARD
  // Prevents modifying topology while Jolt is stepping or while
  // background threads are querying the world state.
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // 2. HANDLE RESOLUTION
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  // 3. DEFERRED DESTRUCTION
  // We check if it's ALIVE or PENDING_CREATE.
  // If it's already PENDING_DESTROY, we do nothing (idempotent).
  if (self->slot_states[slot] == SLOT_ALIVE ||
      self->slot_states[slot] == SLOT_PENDING_CREATE) {

    if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header = CMD_HEADER(CMD_DESTROY_BODY, slot);

    // Mark the slot immediately. This ensures that any logic
    // running between now and the next step() treats this body as "gone".
    self->slot_states[slot] = SLOT_PENDING_DESTROY;
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// Helper macro to get a float attribute, decref it, and handle errors
#define GET_FLOAT_ATTR(obj, name, target)                                      \
  do {                                                                         \
    PyObject *attr = PyObject_GetAttrString(obj, name);                        \
    if (attr) {                                                                \
      double _v = PyFloat_AsDouble(attr);                                      \
      Py_DECREF(attr);                                                         \
      if (!PyErr_Occurred())                                                   \
        (target) = (float)_v;                                                  \
    }                                                                          \
    PyErr_Clear();                                                             \
  } while (0)

// --- Low-complexity helper to fetch attributes with a fallback ---
static float get_py_float_attr(PyObject *obj, const char *name,
                               float default_val) {
  if (!obj || obj == Py_None) {
    return default_val;
  }

  float result = default_val;
  PyObject *attr = PyObject_GetAttrString(obj, name);

  if (attr) {
    double v = PyFloat_AsDouble(attr);
    if (!PyErr_Occurred()) {
      result = (float)v;
    }
    Py_DECREF(attr);
  }

  // Clear any errors (like AttributeError) to allow fallback to default
  PyErr_Clear();
  return result;
}

// vroom vroom
// this is paperwork and i did surgery in the core
// --- Internal Helpers to reduce complexity ---

// --- Reusable helper for Vec3 parsing (Complexity: 2) ---
static int parse_py_vec3(PyObject *obj, Vec3f *out) {
  // 1. Initial validation
  if (!obj || !PySequence_Check(obj) || PySequence_Size(obj) != 3) {
    return 0;
  }

  float results[3];
  for (int i = 0; i < 3; i++) {
    PyObject *item = PySequence_GetItem(obj, i);
    if (!item) {
      return 0;
    }

    results[i] = (float)PyFloat_AsDouble(item);
    Py_DECREF(item);

    if (UNLIKELY(PyErr_Occurred())) {
      return 0;
    }
  }

  // 3. Assignment to struct members
  out->x = results[0];
  out->y = results[1];
  out->z = results[2];

  return 1;
}

// --- Refactored Wheel Creation (Complexity: 2) ---
static JPH_WheelSettings *create_single_wheel(PyObject *w_dict,
                                              JPH_LinearCurve *f_curve) {
  Vec3f pos;

  // 1. Parse Position using helper
  if (!parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
    PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 floats");
    return NULL;
  }

  // 2. Parse Float Attributes using consistent helper
  float radius = get_py_float_attr(w_dict, "radius", 0.4f);
  float width = get_py_float_attr(w_dict, "width", 0.2f);

  // 3. Jolt Object Setup
  JPH_WheelSettingsWV *w = JPH_WheelSettingsWV_Create();
  // A standard wheel has an inertia of about 0.1 to 0.5
  JPH_WheelSettingsWV_SetInertia(w, 0.5f);
  JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings *)w, 0.05f);
  JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings *)w, 0.3f); // 30cm travel
  JPH_SpringSettings spring = {0};
  spring.mode = JPH_SpringMode_FrequencyAndDamping;
  spring.frequencyOrStiffness = 4.0f; // Strong enough to hold the car
  spring.damping = 0.7f;
  JPH_WheelSettings_SetSuspensionSpring((JPH_WheelSettings *)w, &spring);
  // The axis the wheel pivots around for steering
  JPH_WheelSettings_SetSteeringAxis((JPH_WheelSettings *)w, &(JPH_Vec3){0, 1.0f, 0});
  
  // The 'Up' direction for the wheel geometry
  JPH_WheelSettings_SetWheelUp((JPH_WheelSettings *)w, &(JPH_Vec3){0, 1.0f, 0});
  
  // The 'Forward' direction (the way it rolls)
  JPH_WheelSettings_SetWheelForward((JPH_WheelSettings *)w, &(JPH_Vec3){0, 0, 1.0f});
  
  // Suspension direction (the way the shock absorber moves) - usually opposite to Up
  JPH_WheelSettings_SetSuspensionDirection((JPH_WheelSettings *)w, &(JPH_Vec3){0, -1.0f, 0});
  JPH_WheelSettingsWV_SetMaxBrakeTorque(w, 1500.0f); 
  if (pos.z > 0.1f) {
      JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.5f);
      JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, 0.0f);
  } else {
      JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.0f);
      JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, 4000.0f);
  }
  JPH_WheelSettings_SetPosition((JPH_WheelSettings *)w,
                                &(JPH_Vec3){pos.x, pos.y, pos.z});
  JPH_WheelSettings_SetRadius((JPH_WheelSettings *)w, radius);
  JPH_WheelSettings_SetWidth((JPH_WheelSettings *)w, width);

  JPH_WheelSettingsWV_SetLongitudinalFriction(w, f_curve);
  JPH_WheelSettingsWV_SetLateralFriction(w, f_curve);

  // Steering logic (Simple branch)
  JPH_WheelSettingsWV_SetMaxSteerAngle(w, (pos.z > 0.1f) ? 0.5f : 0.0f);

  return (JPH_WheelSettings *)w;
}

static void
setup_vehicle_differentials(JPH_WheeledVehicleControllerSettings *v_ctrl,
                            const char *drive_str, uint32_t num_wheels) {
  if (strcmp(drive_str, "FWD") == 0) {
    JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
  } else if (strcmp(drive_str, "AWD") == 0 && num_wheels >= 4) {
    JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
    JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 2, 3);
  } else { // RWD
    uint32_t i1 = (num_wheels >= 4) ? 2 : 0;
    uint32_t i2 = (num_wheels >= 4) ? 3 : 1;
    JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, (int)i1,
                                                         (int)i2);
  }
}

// --- Internal Helpers for Vehicle Construction ---

typedef struct {
  JPH_LinearCurve *f_curve;
  JPH_LinearCurve *t_curve;
  JPH_WheelSettings **w_settings;
  JPH_WheeledVehicleControllerSettings *v_ctrl;
  JPH_VehicleTransmissionSettings *v_trans_set;
  JPH_VehicleCollisionTesterRay *tester;
  JPH_VehicleConstraint *j_veh;
  bool is_added_to_world;
} VehicleResources;

static void cleanup_vehicle_resources(VehicleResources *r, uint32_t num_wheels,
                                      PhysicsWorldObject *self) {
  if (r->j_veh) {
    // If it was already added to Jolt, we MUST remove it before destroying it
    if (r->is_added_to_world && self && self->system) {
      JPH_PhysicsSystem_RemoveStepListener(
          self->system, JPH_VehicleConstraint_AsPhysicsStepListener(r->j_veh));
      JPH_PhysicsSystem_RemoveConstraint(self->system,
                                         (JPH_Constraint *)r->j_veh);
    }
    JPH_Constraint_Destroy((JPH_Constraint *)r->j_veh);
  }

  if (r->tester) {
    JPH_VehicleCollisionTester_Destroy((JPH_VehicleCollisionTester *)r->tester);
  }
  if (r->v_trans_set) {
    JPH_VehicleTransmissionSettings_Destroy(r->v_trans_set);
  }
  if (r->v_ctrl) {
    JPH_VehicleControllerSettings_Destroy(
        (JPH_VehicleControllerSettings *)r->v_ctrl);
  }

  if (r->w_settings) {
    for (uint32_t i = 0; i < num_wheels; i++) {
      if (r->w_settings[i]) {
        JPH_WheelSettings_Destroy(r->w_settings[i]);
      }
    }
    PyMem_RawFree((void *)r->w_settings);
  }

  if (r->f_curve) {
    JPH_LinearCurve_Destroy(r->f_curve);
  }
  if (r->t_curve) {
    JPH_LinearCurve_Destroy(r->t_curve);
  }
}

// --- Sub-helper: Engine Configuration ---
static void setup_engine(JPH_WheeledVehicleControllerSettings *v_ctrl,
                         JPH_LinearCurve *t_curve, PyObject *py_engine) {
  JPH_VehicleEngineSettings eng_set;
  JPH_VehicleEngineSettings_Init(&eng_set);

  // Flat execution: no nesting, no hidden macro branches
  eng_set.maxTorque = get_py_float_attr(py_engine, "max_torque", 500.0f);
  eng_set.maxRPM = get_py_float_attr(py_engine, "max_rpm", 7000.0f);
  eng_set.minRPM = get_py_float_attr(py_engine, "min_rpm", 1000.0f);
  eng_set.inertia = get_py_float_attr(py_engine, "inertia", 0.5f);

  eng_set.normalizedTorque = t_curve;

  JPH_WheeledVehicleControllerSettings_SetEngine(v_ctrl, &eng_set);
}

// --- Sub-helper: Transmission Configuration ---
static void setup_transmission(JPH_WheeledVehicleControllerSettings *v_ctrl,
                               JPH_VehicleTransmissionSettings *v_trans_set,
                               PyObject *py_trans) {
  // Determine mode
  int t_mode = 1; // Default Manual
  PyObject *o_mode = (py_trans && py_trans != Py_None)
                         ? PyObject_GetAttrString(py_trans, "mode")
                         : NULL;
  if (o_mode) {
    t_mode = (int)PyLong_AsLong(o_mode);
    Py_DECREF(o_mode);
  }
  PyErr_Clear();

  JPH_VehicleTransmissionSettings_SetMode(v_trans_set,
                                          (JPH_TransmissionMode)t_mode);
  JPH_VehicleTransmissionSettings_SetClutchStrength(
      v_trans_set, get_py_float_attr(py_trans, "clutch_strength", 2000.0f));

  // --- Define default gear ratios to prevent Division by Zero ---
  // If py_trans doesn't provide ratios, we must provide defaults.
  float forward_gears[] = {2.8f, 1.75f, 1.3f, 1.0f, 0.8f};
  JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, forward_gears, 5);
  
  float reverse_gears[] = {-3.0f};
  JPH_VehicleTransmissionSettings_SetReverseGearRatios(v_trans_set, reverse_gears, 1);

  JPH_WheeledVehicleControllerSettings_SetTransmission(v_ctrl, v_trans_set);
}

// --- Main coordinate function (Complexity: 1) ---
static void configure_drivetrain(VehicleResources *r, PyObject *py_engine,
                                 PyObject *py_trans, const char *drive_str,
                                 uint32_t num_wheels) {

  setup_engine(r->v_ctrl, r->t_curve, py_engine);

  setup_transmission(r->v_ctrl, r->v_trans_set, py_trans);

  setup_vehicle_differentials(r->v_ctrl, drive_str, num_wheels);
}

// --- Main Function ---

static PyObject *PhysicsWorld_create_vehicle(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  uint64_t chassis_h = 0;
  PyObject *py_wheels = NULL;
  PyObject *py_engine = NULL;
  PyObject *py_trans = NULL;
  char *drive_str = "RWD";
  static char *kwlist[] = {"chassis", "wheels",       "drive",
                           "engine",  "transmission", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KO|sOO", kwlist, &chassis_h,
                                   &py_wheels, &drive_str, &py_engine,
                                   &py_trans)) {
    return NULL;
  }

  if (!PyList_Check(py_wheels) || PyList_Size(py_wheels) < 2) {
    PyErr_SetString(PyExc_ValueError,
                    "Wheels must be a list of at least 2 dictionaries");
    return NULL;
  }
  uint32_t num_wheels = (uint32_t)PyList_Size(py_wheels);

  // 1. Resolve Body (Double-Locking Pattern)
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);
  flush_commands(self);
  uint32_t slot = 0;
  if (!unpack_handle(self, chassis_h, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_ValueError, "Invalid chassis handle");
  }
  JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
  SHADOW_UNLOCK(&self->shadow_lock);

  const JPH_BodyLockInterface *lock_iface =
      JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockWrite lock;
  JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);
  if (!lock.body) {
    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    return PyErr_Format(PyExc_RuntimeError, "Could not lock chassis body");
  }

  // 2. Initialize Resources
  VehicleResources r = {0};
  r.f_curve = JPH_LinearCurve_Create();
  JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
  JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, 1.0f);
  r.t_curve = JPH_LinearCurve_Create();
  JPH_LinearCurve_AddPoint(r.t_curve, 0.0f, 1.0f);
  JPH_LinearCurve_AddPoint(r.t_curve, 1.0f, 1.0f);
  r.w_settings = (JPH_WheelSettings **)PyMem_RawCalloc(
      num_wheels, sizeof(JPH_WheelSettings *));
  r.v_ctrl = JPH_WheeledVehicleControllerSettings_Create();
  r.v_trans_set = JPH_VehicleTransmissionSettings_Create();

  // 3. Generate Wheels
  for (uint32_t i = 0; i < num_wheels; i++) {
    r.w_settings[i] =
        create_single_wheel(PyList_GetItem(py_wheels, i), r.f_curve);
    if (!r.w_settings[i]) {
      cleanup_vehicle_resources(&r, num_wheels, self);
      JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
      return NULL;
    }
  }

  // 4. Setup Drivetrain & Assembly
  configure_drivetrain(&r, py_engine, py_trans, drive_str, num_wheels);

  JPH_VehicleConstraintSettings v_set;
  JPH_VehicleConstraintSettings_Init(&v_set);
  v_set.wheelsCount = num_wheels;
  v_set.wheels = r.w_settings;
  v_set.controller = (JPH_VehicleControllerSettings *)r.v_ctrl;

  r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
  if (!r.j_veh) {
    PyErr_SetString(PyExc_RuntimeError, "Jolt vehicle creation failed");
    cleanup_vehicle_resources(&r, num_wheels, self);
    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    return NULL;
  }

  r.tester =
      r.tester = JPH_VehicleCollisionTesterRay_Create(0, &(JPH_Vec3){0, 1.0f, 0}, 2.0f);
  JPH_VehicleConstraint_SetVehicleCollisionTester(
      r.j_veh, (JPH_VehicleCollisionTester *)r.tester);

  // 5. World Insertion
  SHADOW_LOCK(&self->shadow_lock);
  // We must ensure no queries started while we were busy parsing Python dicts!
  BLOCK_UNTIL_NOT_STEPPING(self); // Safety check
  BLOCK_UNTIL_NOT_QUERYING(self);
  JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint *)r.j_veh);
  JPH_PhysicsSystem_AddStepListener(
      self->system, JPH_VehicleConstraint_AsPhysicsStepListener(r.j_veh));
  r.is_added_to_world = true;
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);

  // --- 6. Python Wrapper (FIXED) ---
  CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
  VehicleObject *obj = (VehicleObject *)PyObject_New(
      VehicleObject, (PyTypeObject *)st->VehicleType);

  if (!obj) {
    cleanup_vehicle_resources(&r, num_wheels, self);
    return NULL;
  }

  // Assign individual fields to preserve PyObject_HEAD
  obj->vehicle = r.j_veh;
  obj->tester = (JPH_VehicleCollisionTester *)r.tester;
  obj->world = self;
  obj->num_wheels = num_wheels;
  obj->current_gear = 0;
  obj->wheel_settings = r.w_settings;
  obj->controller_settings = (JPH_VehicleControllerSettings *)r.v_ctrl;
  obj->transmission_settings = r.v_trans_set;
  obj->friction_curve = r.f_curve;
  obj->torque_curve = r.t_curve;

  // IMPORTANT: Keep the world alive as long as the vehicle exists
  Py_INCREF(self);

  return (PyObject *)obj;
}

// --- Tracked Vehicle Implementation ---

static JPH_WheelSettings *create_track_wheel(PyObject *w_dict) {
  Vec3f pos;
  if (!parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
    PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 floats");
    return NULL; 
  }

  float radius = get_py_float_attr(w_dict, "radius", 0.4f);
  float width = get_py_float_attr(w_dict, "width", 0.2f);
  float suspension_len = get_py_float_attr(w_dict, "suspension", 0.5f);
  
  // Tracked wheels use a float, not a curve
  float friction = get_py_float_attr(w_dict, "friction", 1.0f);

  JPH_WheelSettingsTV *w = JPH_WheelSettingsTV_Create();
  
  JPH_WheelSettings_SetPosition((JPH_WheelSettings *)w, &(JPH_Vec3){pos.x, pos.y, pos.z});
  JPH_WheelSettings_SetRadius((JPH_WheelSettings *)w, radius);
  JPH_WheelSettings_SetWidth((JPH_WheelSettings *)w, width);
  
  JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings *)w, 0.05f);
  JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings *)w, suspension_len); 

  // FIXED: Pass 'friction' (float) instead of 'f_curve'
  JPH_WheelSettingsTV_SetLongitudinalFriction(w, friction);
  JPH_WheelSettingsTV_SetLateralFriction(w, friction);

  return (JPH_WheelSettings *)w;
}

// Helper 1: Setup Engine, Transmission, and Controller settings
static JPH_TrackedVehicleControllerSettings *
init_tracked_controller_settings(TrackedEngineConfig config,
                                 JPH_VehicleTransmissionSettings **out_trans) {

  JPH_TrackedVehicleControllerSettings *t_ctrl =
      JPH_TrackedVehicleControllerSettings_Create();

  JPH_VehicleEngineSettings eng;
  JPH_VehicleEngineSettings_Init(&eng);
  
  // Use members from the config struct
  eng.maxTorque = config.torque;
  eng.maxRPM = config.max_rpm;
  eng.minRPM = config.min_rpm;
  
  JPH_TrackedVehicleControllerSettings_SetEngine(t_ctrl, &eng);

  JPH_VehicleTransmissionSettings *trans =
      JPH_VehicleTransmissionSettings_Create();
  JPH_VehicleTransmissionSettings_SetMode(trans, JPH_TransmissionMode_Auto);

  // Default Gears
  float gears[] = {2.0f, 1.4f, 1.0f, 0.7f};
  JPH_VehicleTransmissionSettings_SetGearRatios(trans, gears, 4);
  float reverse[] = {-1.5f};
  JPH_VehicleTransmissionSettings_SetReverseGearRatios(trans, reverse, 1);

  JPH_TrackedVehicleControllerSettings_SetTransmission(t_ctrl, trans);
  *out_trans = trans;
  return t_ctrl;
}

// Helper 2: Parse track dictionaries and map wheels to tracks
static void parse_tracks_config(JPH_TrackedVehicleControllerSettings *t_ctrl,
                                PyObject *py_tracks, uint32_t ***ptr_list,
                                int *num_out) {
  Py_ssize_t num_tracks = PyList_Size(py_tracks);
  if (num_tracks > 2)
    num_tracks = 2;
  *num_out = (int)num_tracks;
  *ptr_list = (uint32_t **)PyMem_RawCalloc(num_tracks, sizeof(uint32_t *));

  for (int t = 0; t < num_tracks; t++) {
    PyObject *track_dict = PyList_GetItem(py_tracks, t);
    JPH_VehicleTrackSettings track_set;
    JPH_VehicleTrackSettings_Init(&track_set);

    PyObject *py_idxs = PyDict_GetItemString(track_dict, "indices");
    if (py_idxs && PyList_Check(py_idxs)) {
      uint32_t count = (uint32_t)PyList_Size(py_idxs);
      uint32_t *indices = PyMem_RawMalloc(count * sizeof(uint32_t));
      (*ptr_list)[t] = indices;
      for (uint32_t k = 0; k < count; k++) {
        indices[k] = (uint32_t)PyLong_AsLong(PyList_GetItem(py_idxs, k));
      }
      track_set.wheels = indices;
      track_set.wheelsCount = count;
    }

    PyObject *py_driven = PyDict_GetItemString(track_dict, "driven_wheel");
    if (py_driven)
      track_set.drivenWheel = (uint32_t)PyLong_AsUnsignedLong(py_driven);

    JPH_TrackedVehicleControllerSettings_SetTrack(t_ctrl, (uint32_t)t,
                                                  &track_set);
  }
}

// Orchestrator
static PyObject *PhysicsWorld_create_tracked_vehicle(PhysicsWorldObject *self,
                                                     PyObject *args,
                                                     PyObject *kwds) {
  uint64_t chassis_h = 0;
  PyObject *py_wheels = NULL, *py_tracks = NULL;
  float max_rpm = 6000.0f, min_rpm = 500.0f, max_torque = 5000.0f;
  static char *kwlist[] = {"chassis",    "wheels",  "tracks",
                           "max_torque", "max_rpm", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KOO|ff", kwlist, &chassis_h,
                                   &py_wheels, &py_tracks, &max_torque,
                                   &max_rpm))
    return NULL;
  if (!PyList_Check(py_wheels) || !PyList_Check(py_tracks))
    return PyErr_Format(PyExc_TypeError, "Inputs must be lists");

  // 1. Resolve Chassis
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  flush_commands(self);
  uint32_t slot = 0;
  // FIX: Check SLOT_ALIVE to prevent attaching to a dying body
  if (!unpack_handle(self, chassis_h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_ValueError, "Invalid or stale chassis handle");
  }
  JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
  SHADOW_UNLOCK(&self->shadow_lock);

  const JPH_BodyLockInterface *lock_iface =
      JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockWrite lock;
  JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);
  if (!lock.body)
    return PyErr_Format(PyExc_RuntimeError, "Could not lock chassis body");

  // 2. Resource Management
  VehicleResources r = {0};
  r.f_curve = JPH_LinearCurve_Create();
  JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
  JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, 1.0f);

  uint32_t num_wheels = (uint32_t)PyList_Size(py_wheels);
  r.w_settings = (JPH_WheelSettings **)PyMem_RawCalloc(
      num_wheels, sizeof(JPH_WheelSettings *));

  for (uint32_t i = 0; i < num_wheels; i++) {
    r.w_settings[i] = create_track_wheel(PyList_GetItem(py_wheels, i));
    if (!r.w_settings[i]) {
      cleanup_vehicle_resources(&r, num_wheels, self);
      JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
      return NULL;
    }
  }

  // 3. Controller & Tracks
  JPH_VehicleTransmissionSettings *v_trans = NULL;
  TrackedEngineConfig eng_cfg = {
      .torque = max_torque,
      .max_rpm = max_rpm,
      .min_rpm = min_rpm
  };

  JPH_TrackedVehicleControllerSettings *t_ctrl =
      init_tracked_controller_settings(eng_cfg, &v_trans);
  r.v_ctrl = (JPH_WheeledVehicleControllerSettings *)t_ctrl;
  r.v_trans_set = v_trans;

  uint32_t **track_indices_ptrs = NULL;
  int num_tracks = 0;
  parse_tracks_config(t_ctrl, py_tracks, &track_indices_ptrs, &num_tracks);

  // 4. Assembly
  JPH_VehicleConstraintSettings v_set;
  JPH_VehicleConstraintSettings_Init(&v_set);
  v_set.wheelsCount = num_wheels;
  v_set.wheels = r.w_settings;
  v_set.controller = (JPH_VehicleControllerSettings *)t_ctrl;

  r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
  
  // FIX: Leak cleanup on failure
  if (!r.j_veh) {
      cleanup_vehicle_resources(&r, num_wheels, self);
      JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
      
      if (track_indices_ptrs) {
          for(int i = 0; i < num_tracks; i++) {
            PyMem_RawFree(track_indices_ptrs[i]);
          }
          PyMem_RawFree((void *)track_indices_ptrs);
      }
      
      return PyErr_Format(PyExc_RuntimeError, "Failed to create Tracked Vehicle Constraint");
  }

  r.tester = JPH_VehicleCollisionTesterRay_Create(1, &(JPH_Vec3){0, 1.0f, 0}, 2.0f);
  // FIX: Check tester creation
  if (!r.tester) {
      cleanup_vehicle_resources(&r, num_wheels, self); // Clean up j_veh too
      JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
      if (track_indices_ptrs) {
          for(int i=0; i<num_tracks; i++) PyMem_RawFree(track_indices_ptrs[i]);
          PyMem_RawFree((void *)track_indices_ptrs);
      }
      return PyErr_NoMemory();
  }

  JPH_VehicleConstraint_SetVehicleCollisionTester(
      r.j_veh, (JPH_VehicleCollisionTester *)r.tester);

  // 5. Commit
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint *)r.j_veh);
  JPH_PhysicsSystem_AddStepListener(
      self->system, JPH_VehicleConstraint_AsPhysicsStepListener(r.j_veh));
  r.is_added_to_world = true;
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);

  // Free temp track index arrays (they were copied into Jolt settings)
  if (track_indices_ptrs) {
      for (int i = 0; i < num_tracks; i++)
        PyMem_RawFree(track_indices_ptrs[i]);
      PyMem_RawFree((void *)track_indices_ptrs);
  }

  // 6. Python Return
  VehicleObject *obj = (VehicleObject *)PyObject_New(
      VehicleObject,
      (PyTypeObject *)get_culverin_state(PyType_GetModule(Py_TYPE(self)))
          ->VehicleType);
  if (!obj) {
      // Very unlikely at this stage, but technically possible
      // We rely on Python GC to eventually clean up if we crash here, 
      // but strictly we should destroy the Jolt constraint.
      // This requires complex unlocking logic.
      return NULL;
  }

  obj->vehicle = r.j_veh;
  obj->tester = (JPH_VehicleCollisionTester *)r.tester;
  obj->world = self;
  obj->num_wheels = num_wheels;
  obj->wheel_settings = r.w_settings;
  obj->controller_settings = (JPH_VehicleControllerSettings *)t_ctrl;
  obj->transmission_settings = r.v_trans_set;
  obj->friction_curve = r.f_curve;
  obj->torque_curve = NULL;

  Py_INCREF(self);
  return (PyObject *)obj;
}

// Helper: Set Tank Input
static PyObject *Vehicle_set_tank_input(VehicleObject *self, PyObject *args, PyObject *kwds) {
    float left = 0.0f;
    float right = 0.0f;
    float brake = 0.0f;
    
    static char *kwlist[] = {"left", "right", "brake", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ff|f", kwlist, &left, &right, &brake)) {
        return NULL;
    }

    if (!self->vehicle || !self->world) Py_RETURN_NONE;

    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // WAKE UP the tank so it can react to throttle
    JPH_BodyID bid = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));
    JPH_BodyInterface_ActivateBody(self->world->body_interface, bid);
    
    JPH_TrackedVehicleController* t_ctrl = (JPH_TrackedVehicleController*)JPH_VehicleConstraint_GetController(self->vehicle);
    
    // Convert Inputs: Jolt uses (Forward, LeftRatio, RightRatio, Brake)
    float forward = (left + right) / 2.0f;
    
    // Fix Pivot Turn (Forward=0 but L/R opposed)
    if (fabsf(forward) < 0.01f && fabsf(left - right) > 0.01f) {
        forward = 1.0f; 
    }
    // Fix Reverse
    if (forward < 0.0f) {
        // When engine reverses, we must flip ratios to keep logic consistent
        JPH_TrackedVehicleController_SetDriverInput(t_ctrl, forward, -left, -right, brake);
    } else {
        JPH_TrackedVehicleController_SetDriverInput(t_ctrl, forward, left, right, brake);
    }
    
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_position(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  // Allow PENDING_CREATE so users can move bodies they just created
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_ValueError, "Invalid handle");
      return NULL;
  }
  
  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_ValueError, "Stale handle");
      return NULL;
  }

  // Queue it instead of immediate execution
  if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_POS, slot);
  cmd->vec.x = x;
  cmd->vec.y = y;
  cmd->vec.z = z;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_rotation(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN, y = NAN, z = NAN, w = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", "w", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffff", kwlist, &handle_raw, 
                                   &x, &y, &z, &w)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  // Use unpack_handle to verify generation
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  // Allow rotation on bodies that are alive OR just about to be created
  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Handle is stale or body is being destroyed");
    return NULL;
  }

  // Ensure queue has space
  if (!ensure_command_capacity(self)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // Queue the command
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_ROT, slot);
  cmd->vec.x = x;
  cmd->vec.y = y;
  cmd->vec.z = z;
  cmd->vec.w = w;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                  PyObject *args,
                                                  PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  // Note: We check for ALIVE OR PENDING_CREATE now
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_ValueError, "Invalid handle");
      return NULL;
  }

  // Ensure queue space
  if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
  }

  // Queue the command
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_LINVEL, slot);
  cmd->vec.x = x;
  cmd->vec.y = y;
  cmd->vec.z = z;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_ValueError, "Invalid handle");
      return NULL;
  }

  if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_ANGVEL, slot);
  cmd->vec.x = x;
  cmd->vec.y = y;
  cmd->vec.z = z;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // FIX: Consistency Guard
  BLOCK_UNTIL_NOT_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) ||
      self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
  long mt = (long)JPH_BodyInterface_GetMotionType(self->body_interface, bid);

  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromLong(mt);
}

static PyObject *PhysicsWorld_set_motion_type(PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  int motion_type = 0;
  static char *kwlist[] = {"handle", "motion", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Ki", kwlist, &handle_raw,
                                   &motion_type)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  // Allow modifying bodies created in the current frame
  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Handle is stale or body is being destroyed");
    return NULL;
  }

  if (!ensure_command_capacity(self)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // Queue the command
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_MOTION, slot);
  cmd->motion.motion_type = motion_type;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  unsigned long long data = 0;
  static char *kwlist[] = {"handle", "data", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KK", kwlist, &handle_raw,
                                   &data)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Handle is stale or invalid");
    return NULL;
  }

  if (!ensure_command_capacity(self)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // Queue the command
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_USER_DATA, slot);
  cmd->user_data.user_data_val = (uint64_t)data;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_user_data(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // 1. FIX: Ensure indices aren't shifting while we read
  BLOCK_UNTIL_NOT_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) ||
      self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE; // Macro: Incref Py_None and return
  }

  uint64_t val = self->user_data[self->slot_to_dense[slot]];

  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLongLong(val);
}

static PyObject *PhysicsWorld_activate(PhysicsWorldObject *self, PyObject *args,
                                       PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Body is stale or invalid");
    return NULL;
  }

  if (!ensure_command_capacity(self)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_ACTIVATE, slot);

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_deactivate(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Body is stale or invalid");
    return NULL;
  }

  if (!ensure_command_capacity(self)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_DEACTIVATE, slot);
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_transform(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float px = 0, py = 0, pz = 0;
  float rx = 0, ry = 0, rz = 0, rw = 1.0f;
  static char *kwlist[] = {"handle", "pos", "rot", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K(fff)(ffff)", kwlist,
                                   &handle_raw, &px, &py, &pz, &rx, &ry, &rz,
                                   &rw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  // Use unpack_handle to verify identity and generation
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  // Support modifying bodies created in the same frame
  uint8_t state = self->slot_states[slot];
  if (state != SLOT_ALIVE && state != SLOT_PENDING_CREATE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Body handle is stale or invalid");
    return NULL;
  }

  if (!ensure_command_capacity(self)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // Queue CMD_SET_TRNS
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_TRNS, slot);
  
  // Pack position
  cmd->transform.px = px;
  cmd->transform.py = py;
  cmd->transform.pz = pz;
  
  // Pack rotation
  cmd->transform.rx = rx;
  cmd->transform.ry = ry;
  cmd->transform.rz = rz;
  cmd->transform.rw = rw;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_ccd(PhysicsWorldObject *self,
                                      PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  int enabled = 0;
  static char *kwlist[] = {"handle", "enabled", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kp", kwlist, &handle_raw, &enabled)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) ||
      self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
  }

  // Reuse the 'motion_type' field in the union since it's just an int
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_SET_CCD, slot);
  cmd->motion.motion_type = enabled; // 1 = LinearCast, 0 = Discrete

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// Unified hit collector for both Broad and Narrow phase overlaps
static void overlap_record_hit(OverlapContext *ctx, JPH_BodyID bid) {
  // 1. Grow buffer if needed (Native memory, no GIL required)
  if (ctx->count >= ctx->capacity) {
    size_t new_cap = (ctx->capacity == 0) ? 32 : ctx->capacity * 2;
    uint64_t *new_ptr = PyMem_RawRealloc(ctx->hits, new_cap * sizeof(uint64_t));
    if (!new_ptr) {
      return; // Drop hit on OOM (safer than crashing)
    }
    ctx->hits = new_ptr;
    ctx->capacity = new_cap;
  }

  // 2. Retrieve the baked Handle from Jolt UserData
  // This handle contains the Generation + Slot at the time of creation.
  ctx->hits[ctx->count++] =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);
}

static float OverlapCallback_Narrow(void *context,
                                    const JPH_CollideShapeResult *result) {
  overlap_record_hit((OverlapContext *)context, result->bodyID2);
  return 1.0f; // Continue looking for more hits
}

static float OverlapCallback_Broad(void *context, const JPH_BodyID result_bid) {
  overlap_record_hit((OverlapContext *)context, result_bid);
  return 1.0f; // Continue
}

static PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float radius = 1.0f;
  static char *kwlist[] = {"center", "radius", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &x, &y, &z,
                                   &radius)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  OverlapContext ctx = {.world = self, .hits = NULL, .count = 0, .capacity = 0};

  // Jolt Resources
  JPH_Shape *shape = NULL;
  JPH_BroadPhaseLayerFilter *bp_filter = NULL;
  JPH_ObjectLayerFilter *obj_filter = NULL;
  JPH_BodyFilter *body_filter = NULL;

  // --- 1. PHASE GUARD (Blocking) ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  // Reserve query slot
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. RESOURCE PREP ---
  JPH_SphereShapeSettings *ss = JPH_SphereShapeSettings_Create(radius);
  if (!ss) {
    PyErr_NoMemory();
    goto cleanup;
  }
  shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);
  if (!shape) {
    PyErr_NoMemory();
    goto cleanup;
  }

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)x;
  pos->y = (double)y;
  pos->z = (double)z;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = 0;
  rot->y = 0;
  rot->z = 0;
  rot->w = 1;
  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, rot, pos);
  JPH_STACK_ALLOC(JPH_Vec3, scale);
  scale->x = 1.0f;
  scale->y = 1.0f;
  scale->z = 1.0f;
  JPH_STACK_ALLOC(JPH_RVec3, base_offset);
  base_offset->x = 0;
  base_offset->y = 0;
  base_offset->z = 0;
  JPH_STACK_ALLOC(JPH_CollideShapeSettings, settings);
  JPH_CollideShapeSettings_Init(settings);

  // --- 3. FILTER SETUP & EXECUTION (Serialized) ---
  // We MUST lock the trampoline because SetProcs modifies global state.
  SHADOW_LOCK(&g_jph_trampoline_lock);

  // BroadPhase: Allow All
  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide =
                                                  filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  // ObjectLayer: Allow All
  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide =
                                               filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  // BodyFilter: Default (True)
  JPH_BodyFilter_Procs bf_procs = {.ShouldCollide = filter_true_body};
  body_filter = JPH_BodyFilter_Create(NULL);
  JPH_BodyFilter_SetProcs(&bf_procs);

  const JPH_NarrowPhaseQuery *nq =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  JPH_NarrowPhaseQuery_CollideShape(nq, shape, scale, transform, settings,
                                    base_offset, OverlapCallback_Narrow, &ctx,
                                    bp_filter, obj_filter, body_filter, NULL);

  // Restore Defaults & Unlock
  // (filter_true_body is effectively the default, but we set it explicitly
  // above)
  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  // --- 4. VALIDATION (Locked) ---
  ret_val = PyList_New(0);
  if (!ret_val) {
    goto cleanup;
  }

  SHADOW_LOCK(&self->shadow_lock);
  for (size_t i = 0; i < ctx.count; i++) {
    uint64_t h = ctx.hits[i];
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {
      PyObject *py_h = PyLong_FromUnsignedLongLong(h);
      if (py_h) {
        PyList_Append(ret_val, py_h);
        Py_DECREF(py_h);
      }
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

cleanup:
  // Release query slot
  atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);

  if (shape) {
    JPH_Shape_Destroy(shape);
  }
  if (bp_filter) {
    JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  }
  if (obj_filter) {
    JPH_ObjectLayerFilter_Destroy(obj_filter);
  }
  if (body_filter) {
    JPH_BodyFilter_Destroy(body_filter);
  }

  if (ctx.hits) {
    PyMem_RawFree(ctx.hits);
  }

  return ret_val;
}

static PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  float min_x = NAN;
  float min_y = NAN;
  float min_z = NAN;
  float max_x = NAN;
  float max_y = NAN;
  float max_z = NAN;
  static char *kwlist[] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, &min_x,
                                   &min_y, &min_z, &max_x, &max_y, &max_z)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  OverlapContext ctx = {.world = self, .hits = NULL, .count = 0, .capacity = 0};

  JPH_BroadPhaseLayerFilter *bp_filter = NULL;
  JPH_ObjectLayerFilter *obj_filter = NULL;

  // --- 1. PHASE GUARD ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT PREP ---
  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x;
  box->min.y = min_y;
  box->min.z = min_z;
  box->max.x = max_x;
  box->max.y = max_y;
  box->max.z = max_z;

  // --- 3. FILTER & EXECUTION (Serialized) ---
  SHADOW_LOCK(&g_jph_trampoline_lock);

  JPH_BroadPhaseLayerFilter_Procs bp_procs = {.ShouldCollide =
                                                  filter_allow_all_bp};
  bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

  JPH_ObjectLayerFilter_Procs obj_procs = {.ShouldCollide =
                                               filter_allow_all_obj};
  obj_filter = JPH_ObjectLayerFilter_Create(NULL);
  JPH_ObjectLayerFilter_SetProcs(&obj_procs);

  const JPH_BroadPhaseQuery *bq =
      JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);
  JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx,
                                   bp_filter, obj_filter);

  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  // --- 4. VALIDATION ---
  ret_val = PyList_New(0);
  if (!ret_val) {
    goto cleanup;
  }

  SHADOW_LOCK(&self->shadow_lock);
  for (size_t i = 0; i < ctx.count; i++) {
    uint64_t h = ctx.hits[i];
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    if (slot < self->slot_capacity && self->generations[slot] == gen &&
        self->slot_states[slot] == SLOT_ALIVE) {
      PyObject *py_h = PyLong_FromUnsignedLongLong(h);
      if (py_h) {
        PyList_Append(ret_val, py_h);
        Py_DECREF(py_h);
      }
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

cleanup:
  atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_release);

  if (bp_filter) {
    JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
  }
  if (obj_filter) {
    JPH_ObjectLayerFilter_Destroy(obj_filter);
  }
  if (ctx.hits) {
    PyMem_RawFree(ctx.hits);
  }

  return ret_val;
}

static PyObject *PhysicsWorld_get_index(PhysicsWorldObject *self,
                                        PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
  }
  uint32_t idx = self->slot_to_dense[slot];
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLong(idx);
}

static PyObject *PhysicsWorld_is_alive(PhysicsWorldObject *self, PyObject *args,
                                       PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  bool alive = false;
  if (unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    uint8_t state = self->slot_states[slot];
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
      alive = true;
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  if (alive) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *make_view(PhysicsWorldObject *self, void *ptr) {
  if (!ptr) {
    Py_RETURN_NONE;
  }

  // 1. Capture State Under Lock
  SHADOW_LOCK(&self->shadow_lock);

  // We capture the current count at the moment the view is exported
  size_t current_count = self->count;

  // Increment export count to prevent resize() from moving this pointer
  self->view_export_count++;

  SHADOW_UNLOCK(&self->shadow_lock);

  // 2. Setup Local Buffer Metadata
  // These are copied by Python into the memoryview object.
  // We use 4 floats per body (stride is 16 bytes).
  Py_ssize_t local_shape[1] = {(Py_ssize_t)(current_count * 4)};
  Py_ssize_t local_strides[1] = {(Py_ssize_t)sizeof(float)};

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = ptr;
  buf.obj = (PyObject *)self; // Ownership link
  Py_INCREF(self);

  buf.len = local_shape[0] * (Py_ssize_t)sizeof(float);
  buf.readonly = 1;
  buf.itemsize = sizeof(float);
  buf.format = "f";
  buf.ndim = 1;
  buf.shape = local_shape;
  buf.strides = local_strides;

  // 3. Create MemoryView
  PyObject *mv = PyMemoryView_FromBuffer(&buf);

  if (!mv) {
    // Clean up on failure
    SHADOW_LOCK(&self->shadow_lock);
    if (self->view_export_count > 0) {
      self->view_export_count--;
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    Py_DECREF(self); // Drop the ownership link ref
    return NULL;
  }

  return mv;
}

static PyObject *PhysicsWorld_get_active_indices(PhysicsWorldObject *self,
                                                 PyObject *args) {
  SHADOW_LOCK(&self->shadow_lock);
  size_t count = self->count;
  if (count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyBytes_FromStringAndSize(NULL, 0);
  }

  // 1. Snapshot the BodyIDs while locked (Fast)
  JPH_BodyID *id_scratch =
      (JPH_BodyID *)PyMem_RawMalloc(count * sizeof(JPH_BodyID));
  if (!id_scratch) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }
  memcpy(id_scratch, self->body_ids, count * sizeof(JPH_BodyID));
  SHADOW_UNLOCK(&self->shadow_lock);

  // 2. Query activity state WHILE UNLOCKED (Deadlock safe)
  uint32_t *results = (uint32_t *)PyMem_RawMalloc(count * sizeof(uint32_t));
  size_t active_count = 0;
  JPH_BodyInterface *bi = self->body_interface;

  for (size_t i = 0; i < count; i++) {
    if (id_scratch[i] != JPH_INVALID_BODY_ID &&
        JPH_BodyInterface_IsActive(bi, id_scratch[i])) {
      results[active_count++] = (uint32_t)i;
    }
  }

  // 3. Construct Python object and cleanup
  PyObject *bytes_obj = PyBytes_FromStringAndSize(
      (char *)results, (Py_ssize_t)(active_count * sizeof(uint32_t)));
  PyMem_RawFree(id_scratch);
  PyMem_RawFree(results);
  return bytes_obj;
}

static PyObject *get_user_data_buffer(PhysicsWorldObject *self, void *c) {
  if (!self->user_data) {
    Py_RETURN_NONE;
  }

  SHADOW_LOCK(&self->shadow_lock);
  size_t current_count = self->count;
  self->view_export_count++;
  SHADOW_UNLOCK(&self->shadow_lock);

  // Use stack-allocated metadata to prevent cross-thread corruption
  Py_ssize_t local_shape[1] = {(Py_ssize_t)current_count};
  Py_ssize_t local_stride = sizeof(uint64_t);

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = self->user_data;
  buf.obj = (PyObject *)self;
  Py_INCREF(self);

  buf.len = (Py_ssize_t)(current_count * sizeof(uint64_t));
  buf.readonly = 1;
  buf.itemsize = sizeof(uint64_t);
  buf.format = "Q";
  buf.ndim = 1;
  buf.shape = local_shape;
  buf.strides = &local_stride;

  PyObject *mv = PyMemoryView_FromBuffer(&buf);
  if (!mv) {
    SHADOW_LOCK(&self->shadow_lock);
    if (self->view_export_count > 0) {
      self->view_export_count--;
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_DECREF(self);
    return NULL;
  }
  return mv;
}

static PyObject *PhysicsWorld_get_render_state(PhysicsWorldObject *self,
                                               PyObject *args) {
  float alpha = NAN;
  if (!PyArg_ParseTuple(args, "f", &alpha)) {
    return NULL;
  }

  alpha = fmaxf(0.0f, fminf(1.0f, alpha));

  SHADOW_LOCK(&self->shadow_lock);
  size_t count = self->count;
  size_t total_bytes = count * 7 * sizeof(float);

  // 1. Allocate the Python Bytes object immediately with NULL.
  // This reserves the memory inside the Python Heap.
  PyObject *bytes_obj =
      PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_bytes);
  if (!bytes_obj) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // 2. Get the direct pointer to the Bytes object's internal buffer.
  float *out = (float *)PyBytes_AsString(bytes_obj);

  for (size_t i = 0; i < count; i++) {
    size_t src = i * 4;
    size_t dst = i * 7;

    // Position Lerp
    out[dst + 0] =
        self->prev_positions[src + 0] +
        (self->positions[src + 0] - self->prev_positions[src + 0]) * alpha;
    out[dst + 1] =
        self->prev_positions[src + 1] +
        (self->positions[src + 1] - self->prev_positions[src + 1]) * alpha;
    out[dst + 2] =
        self->prev_positions[src + 2] +
        (self->positions[src + 2] - self->prev_positions[src + 2]) * alpha;

    // Rotation NLerp
    float q1x = self->prev_rotations[src + 0];
    float q1y = self->prev_rotations[src + 1];
    float q1z = self->prev_rotations[src + 2];
    float q1w = self->prev_rotations[src + 3];
    float q2x = self->rotations[src + 0];
    float q2y = self->rotations[src + 1];
    float q2z = self->rotations[src + 2];
    float q2w = self->rotations[src + 3];

    float dot = q1x * q2x + q1y * q2y + q1z * q2z + q1w * q2w;
    if (dot < 0.0f) {
      q2x = -q2x;
      q2y = -q2y;
      q2z = -q2z;
      q2w = -q2w;
    }

    float rx = q1x + (q2x - q1x) * alpha;
    float ry = q1y + (q2y - q1y) * alpha;
    float rz = q1z + (q2z - q1z) * alpha;
    float rw = q1w + (q2w - q1w) * alpha;

    float inv_len = 1.0f / sqrtf(rx * rx + ry * ry + rz * rz + rw * rw);
    out[dst + 3] = rx * inv_len;
    out[dst + 4] = ry * inv_len;
    out[dst + 5] = rz * inv_len;
    out[dst + 6] = rw * inv_len;
  }

  SHADOW_UNLOCK(&self->shadow_lock);

  // Return the object directly to Python
  return bytes_obj;
}

// --- Vehicles Methods ---

static PyObject *Vehicle_set_input(VehicleObject *self, PyObject *args,
                                   PyObject *kwds) {
  float forward = 0.0f, right = 0.0f, brake = 0.0f, handbrake = 0.0f;
  static char *kwlist[] = {"forward", "right", "brake", "handbrake", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ffff", kwlist, &forward, &right, &brake, &handbrake)) 
    return NULL;

  SHADOW_LOCK(&self->world->shadow_lock);
  // Re-entrancy guard
  if (UNLIKELY(self->world->is_stepping || !self->vehicle)) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  JPH_WheeledVehicleController *controller = (JPH_WheeledVehicleController *)JPH_VehicleConstraint_GetController(self->vehicle);
  JPH_BodyInterface *bi = self->world->body_interface;
  JPH_BodyID chassis_id = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));

  // 1. Wake Body
  JPH_BodyInterface_ActivateBody(bi, chassis_id);

  // 2. Query Velocity and Rotation for Speed Calculation
  JPH_STACK_ALLOC(JPH_Vec3, linear_vel);
  JPH_BodyInterface_GetLinearVelocity(bi, chassis_id, linear_vel);
  JPH_STACK_ALLOC(JPH_Quat, chassis_q);
  JPH_BodyInterface_GetRotation(bi, chassis_id, chassis_q);

  // Calculate World Forward
  JPH_Vec3 world_fwd;
  manual_vec3_rotate_by_quat(&(JPH_Vec3){0, 0, 1.0f}, chassis_q, &world_fwd);
  float speed = (linear_vel->x * world_fwd.x) + (linear_vel->y * world_fwd.y) + (linear_vel->z * world_fwd.z);

  // 3. Logic: Determine Intended Movement
  float input_throttle = 0.0f;
  float input_brake = brake;
  int requested_direction = 0; // 1: Forward, -1: Reverse, 0: Neutral

  if (forward > 0.01f) {
    if (speed < -0.5f) { input_brake = 1.0f; } // Moving backward, pressing forward -> Brake
    else { input_throttle = forward; requested_direction = 1; }
  } else if (forward < -0.01f) {
    if (speed > 0.5f) { input_brake = 1.0f; } // Moving forward, pressing backward -> Brake
    else { input_throttle = fabsf(forward); requested_direction = -1; }
  }

  // 4. Transmission Management (Undeclared Function fix)
  JPH_VehicleTransmission *trans = (JPH_VehicleTransmission *)JPH_WheeledVehicleController_GetTransmission(controller);
  if (trans) {
    int current_gear = JPH_VehicleTransmission_GetCurrentGear(trans);

    // Smart Gear Switching:
    // We only call SetGear when switching between Forward/Reverse/Neutral.
    // If we are already in a forward gear (1,2,3...) we let Jolt's Auto-trans handle it.
    if (requested_direction == 1 && current_gear <= 0) {
        // Shift from Neutral/Reverse to Drive
        JPH_VehicleTransmission_Set(trans, 1, 1.0f);
    } 
    else if (requested_direction == -1 && current_gear >= 0) {
        // Shift from Neutral/Drive to Reverse
        JPH_VehicleTransmission_Set(trans, -1, 1.0f);
    }
    else if (requested_direction == 0 && current_gear != 0 && fabsf(speed) < 0.2f) {
        // Shift to Neutral when stopped and no input
        JPH_VehicleTransmission_Set(trans, 0, 0.0f);
    }
  }

  // 5. Apply to Jolt
  JPH_WheeledVehicleController_SetDriverInput(controller, input_throttle, right, input_brake, handbrake);

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *Vehicle_get_wheel_transform(VehicleObject *self,
                                             PyObject *args) {
  uint32_t index = 0;
  if (!PyArg_ParseTuple(args, "I", &index)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  if (!self->vehicle || index >= self->num_wheels) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_Vec3 right = {1.0f, 0.0f, 0.0f};
  JPH_Vec3 up = {0.0f, 1.0f, 0.0f};

  // Get Transform in Double Precision
  JPH_VehicleConstraint_GetWheelWorldTransform(self->vehicle, index, &right,
                                               &up, transform);

  // --- CRITICAL FIX: Layout Mapping ---

  // 1. Position: In Double Precision, this is the 'column3' member
  // (RVec3/doubles)
  double px = transform->column3.x;
  double py = transform->column3.y;
  double pz = transform->column3.z;

  // 2. Rotation: These are the first 3 columns (Vec4/floats in RMat44)
  // We copy them to a standard Mat4 to extract the quaternion.
  JPH_STACK_ALLOC(JPH_Mat4, rot_only_mat);
  JPH_Mat4_Identity(rot_only_mat);

  // Safe struct copy of Vec4/Vec3 columns
  rot_only_mat->column[0] = transform->column[0];
  rot_only_mat->column[1] = transform->column[1];
  rot_only_mat->column[2] = transform->column[2];

  JPH_STACK_ALLOC(JPH_Quat, q);
  JPH_Mat4_GetQuaternion(rot_only_mat, q);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  // Safe Python construction
  PyObject *py_pos = Py_BuildValue("(ddd)", px, py, pz);
  PyObject *py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);

  if (!py_pos || !py_rot) {
    Py_XDECREF(py_pos);
    Py_XDECREF(py_rot);
    return NULL;
  }

  PyObject *result = PyTuple_Pack(2, py_pos, py_rot);
  Py_DECREF(py_pos);
  Py_DECREF(py_rot);
  return result;
}

static PyObject *Vehicle_get_wheel_local_transform(VehicleObject *self,
                                                   PyObject *args) {
  uint32_t index = 0;
  if (!PyArg_ParseTuple(args, "I", &index)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  // Re-entry guard: Ensure we aren't stepping while querying
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  if (!self->vehicle || index >= self->num_wheels) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  const JPH_Wheel *w_ptr = JPH_VehicleConstraint_GetWheel(self->vehicle, index);
  const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w_ptr);
  JPH_Vec3 local_pos_check;
  JPH_WheelSettings_GetPosition(ws, &local_pos_check);

  // If wheel is on the left (x < 0), we flip the right vector
  JPH_Vec3 right = { (local_pos_check.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f };
  JPH_Vec3 up = { 0.0f, 1.0f, 0.0f };

  JPH_STACK_ALLOC(JPH_Mat4, local_transform);
  // Initialize to identity just in case the API call fails or partially writes
  JPH_Mat4_Identity(local_transform);

  JPH_VehicleConstraint_GetWheelLocalTransform(self->vehicle, index, &right,
                                               &up, local_transform);

  float lx = local_transform->column[3].x;
  float ly = local_transform->column[3].y;
  float lz = local_transform->column[3].z;

  JPH_STACK_ALLOC(JPH_Quat, q);
  JPH_Mat4_GetQuaternion(local_transform, q);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  PyObject *py_pos = Py_BuildValue("(fff)", lx, ly, lz);
  PyObject *py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);

  if (!py_pos || !py_rot) {
    Py_XDECREF(py_pos);
    Py_XDECREF(py_rot);
    return NULL;
  }

  PyObject *result = PyTuple_Pack(2, py_pos, py_rot);
  Py_DECREF(py_pos);
  Py_DECREF(py_rot);

  return result;
}

static PyObject *Vehicle_get_debug_state(VehicleObject *self,
                                         PyObject *Py_UNUSED(ignored)) {
  // 1. LOCK AND GUARD
  // We need the world lock to ensure the vehicle pointer is stable
  // and the physics step isn't currently mutating these values.
  SHADOW_LOCK(&self->world->shadow_lock);

  if (self->world->is_stepping) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    // Debug prints usually shouldn't raise Python errors,
    // so we just log a warning and return.
    DEBUG_LOG("Warning: Cannot get debug state while physics is stepping.");
    Py_RETURN_NONE;
  }

  if (!self->vehicle) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    DEBUG_LOG("Warning: Vehicle has been destroyed.");
    Py_RETURN_NONE;
  }

  // 2. RESOLVE JOLT COMPONENTS
  JPH_WheeledVehicleController *controller =
      (JPH_WheeledVehicleController *)JPH_VehicleConstraint_GetController(
          self->vehicle);
  if (!controller) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  const JPH_VehicleEngine *engine =
      JPH_WheeledVehicleController_GetEngine(controller);
  const JPH_VehicleTransmission *trans =
      JPH_WheeledVehicleController_GetTransmission(controller);

  // 3. CAPTURE INPUTS
  float in_fwd = JPH_WheeledVehicleController_GetForwardInput(controller);
  float in_brk = JPH_WheeledVehicleController_GetBrakeInput(controller);

  // 4. CAPTURE DRIVETRAIN
  float rpm = JPH_VehicleEngine_GetCurrentRPM(engine);
  float engine_torque = JPH_VehicleEngine_GetTorque(engine, in_fwd);
  int gear = JPH_VehicleTransmission_GetCurrentGear(trans);
  float clutch = JPH_VehicleTransmission_GetClutchFriction(trans);

  DEBUG_LOG("=== VEHICLE DEBUG STATE ===");
  DEBUG_LOG("  Inputs: Fwd=%.2f | Brk=%.2f", in_fwd, in_brk);
  DEBUG_LOG("  Engine: %.2f RPM | Torque: %.2f Nm", rpm, engine_torque);
  DEBUG_LOG("  Trans : Gear %d | Clutch Friction: %.2f", gear, clutch);

  // 5. CAPTURE WHEEL STATE
  for (uint32_t i = 0; i < self->num_wheels; i++) {
    const JPH_Wheel *w = JPH_VehicleConstraint_GetWheel(self->vehicle, i);
    const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w);

    bool contact = JPH_Wheel_HasContact(w);
    float susp_len = JPH_Wheel_GetSuspensionLength(w);
    float ang_vel = JPH_Wheel_GetAngularVelocity(w);
    float radius = JPH_WheelSettings_GetRadius(ws);

    float tire_speed = ang_vel * radius;
    float long_lambda = JPH_Wheel_GetLongitudinalLambda(w);
    float lat_lambda = JPH_Wheel_GetLateralLambda(w);

    DEBUG_LOG("  Wheel %u: %s", i, contact ? "GROUND" : "AIR   ");
    DEBUG_LOG("    Susp: %.3fm | AngVel: %.2f rad/s | SurfSpd: %.2f m/s",
              susp_len, ang_vel, tire_speed);
    DEBUG_LOG("    Trac: Long=%.2f | Lat=%.2f", long_lambda, lat_lambda);
  }
  DEBUG_LOG("===========================");

  // 6. UNLOCK
  SHADOW_UNLOCK(&self->world->shadow_lock);

  Py_RETURN_NONE;
}

// --- Vehicle GC Support ---
static int Vehicle_traverse(VehicleObject *self, visitproc visit, void *arg) {
  Py_VISIT(self->world);
  return 0;
}

static int Vehicle_clear(VehicleObject *self) {
  Py_CLEAR(self->world);
  return 0;
}

// --- Explicit Destroy (Clean up Jolt resources) ---
static PyObject *Vehicle_destroy(VehicleObject *self,
                                 PyObject *Py_UNUSED(ignored)) {
  if (!self->world) {
    Py_RETURN_NONE;
  }

  SHADOW_LOCK(&self->world->shadow_lock);

  BLOCK_UNTIL_NOT_STEPPING(self->world);
  BLOCK_UNTIL_NOT_QUERYING(self->world);

  if (!self->vehicle) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  JPH_VehicleConstraint *j_veh = self->vehicle;
  JPH_VehicleCollisionTester *tester = self->tester;
  JPH_VehicleControllerSettings *v_ctrl = self->controller_settings;
  JPH_VehicleTransmissionSettings *v_trans = self->transmission_settings;
  JPH_WheelSettings **wheels = self->wheel_settings;
  JPH_LinearCurve *f_curve = self->friction_curve;
  JPH_LinearCurve *t_curve = self->torque_curve;
  uint32_t wheel_count = self->num_wheels;

  self->vehicle = NULL;
  self->tester = NULL;
  self->controller_settings = NULL;
  self->transmission_settings = NULL;
  self->wheel_settings = NULL;
  self->friction_curve = NULL;
  self->torque_curve = NULL;

  SHADOW_UNLOCK(&self->world->shadow_lock);

  // Safe to destroy Jolt objects now that we are unlocked and removed from struct
  if (j_veh) {
    JPH_PhysicsStepListener *step_listener =
        JPH_VehicleConstraint_AsPhysicsStepListener(j_veh);
    JPH_PhysicsSystem_RemoveStepListener(self->world->system, step_listener);

    JPH_PhysicsSystem_RemoveConstraint(self->world->system,
                                       (JPH_Constraint *)j_veh);
    JPH_Constraint_Destroy((JPH_Constraint *)j_veh);
  }

  if (tester) JPH_VehicleCollisionTester_Destroy(tester);
  if (v_ctrl) JPH_VehicleControllerSettings_Destroy(v_ctrl);
  if (v_trans) JPH_VehicleTransmissionSettings_Destroy(v_trans);

  if (wheels) {
    for (uint32_t i = 0; i < wheel_count; i++) {
      if (wheels[i]) JPH_WheelSettings_Destroy(wheels[i]);
    }
    PyMem_RawFree((void *)wheels);
  }

  if (f_curve) JPH_LinearCurve_Destroy(f_curve);
  if (t_curve) JPH_LinearCurve_Destroy(t_curve);

  Py_RETURN_NONE;
}

// --- Python Deallocation ---
static void Vehicle_dealloc(VehicleObject *self) {
  PyObject_GC_UnTrack(self);

  // Attempt to clean up. If the world is stepping, this might fail
  // to remove the constraint from Jolt immediately.
  // However, in standard cleanup paths, the world will be idle.
  Vehicle_destroy(self, NULL);

  // If destroy failed because the world was busy, the pointers
  // are still in the struct. This is a leak, but a safe one (no crash).

  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Character Methods ---

static void Character_dealloc(CharacterObject *self) {
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

// NEW: GC Traverse/Clear for Character
static int Character_traverse(CharacterObject *self, visitproc visit,
                              void *arg) {
  Py_VISIT(self->world);
  return 0;
}
static int Character_clear(CharacterObject *self) {
  Py_CLEAR(self->world);
  return 0;
}

static PyObject *Character_move(CharacterObject *self, PyObject *args,
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

static PyObject *Character_get_position(CharacterObject *self,
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

static PyObject *Character_set_position(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = NAN;
  float y = NAN;
  float z = NAN;
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

static PyObject *Character_set_rotation(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = NAN;
  float y = NAN;
  float z = NAN;
  float w = NAN;
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

static PyObject *Character_is_grounded(CharacterObject *self, PyObject *args) {
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

static PyObject *Character_set_strength(CharacterObject *self, PyObject *args) {
  float strength = NAN;
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
static PyObject *Character_get_render_transform(CharacterObject *self,
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

static PyObject *PhysicsWorld_set_collision_filter(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
  uint64_t handle_raw = 0;
  uint32_t category = 0;
  uint32_t mask = 0;
  static char *kwlist[] = {"handle", "category", "mask", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KII", kwlist, &handle_raw,
                                   &category, &mask)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // Block to prevent race conditions during physics step
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) ||
      self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  // Direct write to dense arrays (Thread-safe because we hold the lock and
  // engine is not stepping)
  uint32_t dense = self->slot_to_dense[slot];
  self->categories[dense] = category;
  self->masks[dense] = mask;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// --- Skeleton Implementation ---

static void Skeleton_dealloc(SkeletonObject *self) {
  if (self->skeleton) {
    JPH_Skeleton_Destroy(self->skeleton);
  }
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Skeleton_new(PyTypeObject *type, PyObject *args,
                              PyObject *kwds) {
  SkeletonObject *self = (SkeletonObject *)type->tp_alloc(type, 0);
  if (self) {
    self->skeleton = JPH_Skeleton_Create();
    if (!self->skeleton) {
      Py_DECREF(self);
      return PyErr_NoMemory();
    }
  }
  return (PyObject *)self;
}

static PyObject *Skeleton_add_joint(SkeletonObject *self, PyObject *args) {
  const char *name = NULL;
  int parent_idx = -1; // Default to root
  if (!PyArg_ParseTuple(args, "s|i", &name, &parent_idx)) {
    return NULL;
  }

  int idx = (int)JPH_Skeleton_AddJoint2(self->skeleton, name, parent_idx);
  return PyLong_FromLong(idx);
}

static PyObject *Skeleton_get_joint_index(SkeletonObject *self,
                                          PyObject *args) {
  const char *name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  int idx = JPH_Skeleton_GetJointIndex(self->skeleton, name);
  return PyLong_FromLong(idx);
}

// --- Ragdoll Settings Implementation ---

static void RagdollSettings_dealloc(RagdollSettingsObject *self) {
  if (self->settings) {
    JPH_RagdollSettings_Destroy(self->settings);
  }
  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *PhysicsWorld_create_ragdoll_settings(PhysicsWorldObject *self,
                                                      PyObject *args) {
  SkeletonObject *py_skel = NULL;
  if (!PyArg_ParseTuple(
          args, "O!",
          get_culverin_state(PyType_GetModule(Py_TYPE(self)))->SkeletonType,
          &py_skel)) {
    return NULL;
  }

  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);

  RagdollSettingsObject *obj = (RagdollSettingsObject *)PyObject_New(
      RagdollSettingsObject, (PyTypeObject *)st->RagdollSettingsType);
  if (!obj) {
    return NULL;
  }

  obj->settings = JPH_RagdollSettings_Create();
  JPH_RagdollSettings_SetSkeleton(obj->settings, py_skel->skeleton);

  obj->world = self;
  Py_INCREF(self);

  return (PyObject *)obj;
}

static PyObject *RagdollSettings_add_part(RagdollSettingsObject *self,
                                          PyObject *args, PyObject *kwds) {
  int joint_idx = 0;
  int parent_idx = 0;
  int shape_type = 0;
  float mass = 10.0f;
  PyObject *py_size = NULL;
  PyObject *py_pos = NULL;

  float twist_min = -0.1f;
  float twist_max = 0.1f;
  float cone_angle = 0.0f;
  float cx = 1;
  float cy = 0;
  float cz = 0;
  float nx = 0;
  float ny = 1;
  float nz = 0;

  static char *kwlist[] = {
      "joint_index",  "shape_type", "size",      "mass",
      "parent_index", "twist_min",  "twist_max", "cone_angle",
      "axis",         "normal",     "pos",       NULL // Added "pos"
  };

  // Added "O" at the end of the format string for the position tuple
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "iiOfi|fff(fff)(fff)O", kwlist, &joint_idx, &shape_type,
          &py_size, &mass, &parent_idx, &twist_min, &twist_max, &cone_angle,
          &cx, &cy, &cz, &nx, &ny, &nz, &py_pos)) {
    return NULL;
  }

  float s[4] = {1, 1, 1, 0};
  if (py_size && PyTuple_Check(py_size)) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(py_size) && i < 4; i++) {
      s[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(py_size, i));
    }
  } else if (py_size && PyNumber_Check(py_size)) {
    s[0] = (float)PyFloat_AsDouble(py_size);
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  JPH_Shape *shape = find_or_create_shape(self->world, shape_type, s);
  SHADOW_UNLOCK(&self->world->shadow_lock);

  if (!shape) {
    return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");
  }

  int current_cap = JPH_RagdollSettings_GetPartCount(self->settings);
  JPH_Skeleton *skel =
      (JPH_Skeleton *)JPH_RagdollSettings_GetSkeleton(self->settings);
  int skel_count = JPH_Skeleton_GetJointCount(skel);

  // Ensure capacity matches skeleton
  if (current_cap < skel_count) {
    JPH_RagdollSettings_ResizeParts(self->settings, skel_count);
  }

  if (joint_idx >= skel_count) {
    return PyErr_Format(PyExc_IndexError, "Joint index out of bounds");
  }

  JPH_RagdollSettings_SetPartShape(self->settings, joint_idx, shape);
  JPH_RagdollSettings_SetPartMassProperties(self->settings, joint_idx, mass);
  JPH_RagdollSettings_SetPartObjectLayer(self->settings, joint_idx, 1);
  JPH_RagdollSettings_SetPartMotionType(self->settings, joint_idx,
                                        JPH_MotionType_Dynamic);

  if (py_pos && PyTuple_Check(py_pos) && PyTuple_Size(py_pos) == 3) {
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = PyFloat_AsDouble(PyTuple_GetItem(py_pos, 0));
    p->y = PyFloat_AsDouble(PyTuple_GetItem(py_pos, 1));
    p->z = PyFloat_AsDouble(PyTuple_GetItem(py_pos, 2));
    JPH_RagdollSettings_SetPartPosition(self->settings, joint_idx, p);
  }

  if (parent_idx >= 0) {
    JPH_SwingTwistConstraintSettings cs;
    JPH_SwingTwistConstraintSettings_Init(&cs);

    cs.base.enabled = true;

    JPH_Vec3 twist_axis = {cx, cy, cz};
    JPH_Vec3 plane_normal = {nx, ny, nz};

    cs.position1.x = 0;
    cs.position1.y = 0;
    cs.position1.z = 0;
    cs.position2.x = 0;
    cs.position2.y = 0;
    cs.position2.z = 0;

    cs.twistAxis1 = twist_axis;
    cs.twistAxis2 = twist_axis;
    cs.planeAxis1 = plane_normal;
    cs.planeAxis2 = plane_normal;

    cs.normalHalfConeAngle = cone_angle;
    cs.planeHalfConeAngle = cone_angle;
    cs.twistMinAngle = twist_min;
    cs.twistMaxAngle = twist_max;

    JPH_RagdollSettings_SetPartToParent(self->settings, joint_idx, &cs);
  }

  Py_RETURN_NONE;
}

static PyObject *RagdollSettings_stabilize(RagdollSettingsObject *self,
                                           PyObject *args) {
  if (JPH_RagdollSettings_Stabilize(self->settings)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

// --- Ragdoll Instance Implementation ---

static void Ragdoll_dealloc(RagdollObject *self) {
  if (self->world && self->ragdoll) {
    SHADOW_LOCK(&self->world->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self->world);
    BLOCK_UNTIL_NOT_QUERYING(self->world);

    JPH_Ragdoll_RemoveFromPhysicsSystem(self->ragdoll, true);

    // Validate pointers before iteration to prevent corruption
    if (self->body_slots && self->world->slot_states) {
        for (size_t i = 0; i < self->body_count; i++) {
          uint32_t slot = self->body_slots[i];
          
          // Boundary check
          if (slot >= self->world->slot_capacity) continue;

          if (self->world->slot_states[slot] != SLOT_ALIVE) {
            continue; 
          }
          world_remove_body_slot(self->world, slot);
        }
    }
    SHADOW_UNLOCK(&self->world->shadow_lock);
  }

  if (self->ragdoll) {
    JPH_Ragdoll_Destroy(self->ragdoll);
  }
  if (self->body_slots) {
    PyMem_RawFree(self->body_slots);
    self->body_slots = NULL; // Prevent double-free in weird recursion cases
  }

  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Skeleton_finalize(SkeletonObject *self, PyObject *args) {
  JPH_Skeleton_CalculateParentJointIndices(self->skeleton);
  if (!JPH_Skeleton_AreJointsCorrectlyOrdered(self->skeleton)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Skeleton joints are out of order (parent must be added before child)");
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_create_ragdoll(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  RagdollSettingsObject *py_settings = NULL;
  float px = 0;
  float py = 0;
  float pz = 0;
  float rx = 0;
  float ry = 0;
  float rz = 0;
  float rw = 1;
  uint64_t user_data = 0;
  uint32_t category = 0xFFFF;
  uint32_t mask = 0xFFFF;
  uint32_t material_id = 0;

  static char *kwlist[] = {"settings", "pos",  "rot",         "user_data",
                           "category", "mask", "material_id", NULL};

  // FIX 1: Added third "I" to format string for material_id
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "O!(fff)|(ffff)KIII", kwlist,
          get_culverin_state(PyType_GetModule(Py_TYPE(self)))
              ->RagdollSettingsType,
          &py_settings, &px, &py, &pz, &rx, &ry, &rz, &rw, &user_data,
          &category, &mask, &material_id)) {
    return NULL;
  }

  JPH_RagdollSettings_CalculateBodyIndexToConstraintIndex(
      py_settings->settings);
  JPH_RagdollSettings_CalculateConstraintIndexToBodyIdxPair(
      py_settings->settings);

  JPH_Ragdoll *j_rag = JPH_RagdollSettings_CreateRagdoll(
      py_settings->settings, self->system, 0, user_data);
  if (!j_rag) {
    return PyErr_Format(PyExc_RuntimeError,
                        "Jolt failed to create Ragdoll instance");
  }

  int joint_count = JPH_Skeleton_GetJointCount(
      JPH_RagdollSettings_GetSkeleton(py_settings->settings));
  JPH_Mat4 *neutral_matrices =
      (JPH_Mat4 *)PyMem_RawMalloc(joint_count * sizeof(JPH_Mat4));
  JPH_STACK_ALLOC(JPH_RVec3, dummy_root);
  JPH_Ragdoll_GetPose2(j_rag, dummy_root, neutral_matrices, true);

  JPH_STACK_ALLOC(JPH_RVec3, root_pos);
  root_pos->x = px;
  root_pos->y = py;
  root_pos->z = pz;
  JPH_STACK_ALLOC(JPH_Quat, root_q);
  root_q->x = rx;
  root_q->y = ry;
  root_q->z = rz;
  root_q->w = rw;
  JPH_Ragdoll_SetPose2(j_rag, root_pos, neutral_matrices, true);

  int body_count = JPH_Ragdoll_GetBodyCount(j_rag);
  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);
  RagdollObject *obj = (RagdollObject *)PyObject_New(
      RagdollObject, (PyTypeObject *)st->RagdollType);

  if (!obj) {
    JPH_Ragdoll_Destroy(j_rag);
    PyMem_RawFree(neutral_matrices); // FIX 2: Prevent leak if allocation fails
    return NULL;
  }

  obj->ragdoll = j_rag;
  obj->world = self;
  Py_INCREF(self);
  obj->body_count = (size_t)body_count;
  obj->body_slots = (uint32_t *)PyMem_RawMalloc(body_count * sizeof(uint32_t));

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  if (self->free_count < (size_t)body_count) {
    if (PhysicsWorld_resize(self, self->capacity + body_count + 128) < 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        // Clean up the Jolt ragdoll since we can't track it
        JPH_Ragdoll_Destroy(j_rag);
        PyMem_RawFree(neutral_matrices);
        Py_DECREF(obj); 
        return NULL;
    }
  }

  for (int i = 0; i < body_count; i++) {
    JPH_BodyID bid = JPH_Ragdoll_GetBodyID(j_rag, i);
    uint32_t slot = self->free_slots[--self->free_count];
    obj->body_slots[i] = slot;
    uint32_t dense = (uint32_t)self->count;

    JPH_STACK_ALLOC(JPH_Vec3, local_offset);
    JPH_Mat4_GetTranslation(&neutral_matrices[i], local_offset);

    // FIX 3: Apply root rotation to limb offset for perfect world positioning
    JPH_STACK_ALLOC(JPH_Vec3, rotated_offset);
    manual_vec3_rotate_by_quat(local_offset, root_q, rotated_offset);

    self->positions[dense * 4 + 0] = (px + rotated_offset->x);
    self->positions[dense * 4 + 1] = (py + rotated_offset->y);
    self->positions[dense * 4 + 2] = (pz + rotated_offset->z);
    self->positions[dense * 4 + 3] = 0.0f;

    JPH_STACK_ALLOC(JPH_Quat, local_q);
    JPH_Mat4_GetQuaternion(&neutral_matrices[i], local_q);
    JPH_STACK_ALLOC(JPH_Quat, world_q);
    manual_quat_multiply(root_q, local_q, world_q);
    memcpy(&self->rotations[(size_t)dense * 4], world_q, 16);

    // Sync metadata
    memcpy(&self->prev_positions[(size_t)dense * 4],
           &self->positions[(size_t)dense * 4], 16);
    memcpy(&self->prev_rotations[(size_t)dense * 4],
           &self->rotations[(size_t)dense * 4], 16);

    // FIX 4: Zero out velocities to prevent "Exploding Launch" from stale
    // memory
    memset(&self->linear_velocities[(size_t)dense * 4], 0, 16);
    memset(&self->angular_velocities[(size_t)dense * 4], 0, 16);

    self->body_ids[dense] = bid;
    self->slot_to_dense[slot] = dense;
    self->dense_to_slot[dense] = slot;
    self->slot_states[slot] = SLOT_ALIVE;
    self->user_data[dense] = user_data;
    self->categories[dense] = category;
    self->masks[dense] = mask;
    self->material_ids[dense] = material_id;
    self->count++;

    JPH_BodyInterface_SetUserData(
        self->body_interface, bid,
        (uint64_t)make_handle(slot, self->generations[slot]));
  }

  JPH_Ragdoll_AddToPhysicsSystem(j_rag, JPH_Activation_Activate, true);
  SHADOW_UNLOCK(&self->shadow_lock);

  PyMem_RawFree(neutral_matrices);
  return (PyObject *)obj;
}

static PyObject *Ragdoll_drive_to_pose(RagdollObject *self, PyObject *args,
                                       PyObject *kwds) {
  float root_x = NAN;
  float root_y = NAN;
  float root_z = NAN;
  float rx = NAN;
  float ry = NAN;
  float rz = NAN;
  float rw = NAN;
  PyObject *py_matrices = NULL;

  static char *kwlist[] = {"root_pos", "root_rot", "matrices", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(ffff)O", kwlist, &root_x,
                                   &root_y, &root_z, &rx, &ry, &rz, &rw,
                                   &py_matrices)) {
    return NULL;
  }

  if (!self->ragdoll || !self->world) {
    Py_RETURN_NONE; // Or error
  }

  const JPH_RagdollSettings *settings =
      JPH_Ragdoll_GetRagdollSettings(self->ragdoll);
  JPH_Skeleton *skel =
      (JPH_Skeleton *)JPH_RagdollSettings_GetSkeleton(settings);
  int joint_count = JPH_Skeleton_GetJointCount(skel);

  // 1. Validate Buffer Size BEFORE access
  Py_buffer view;
  if (PyObject_GetBuffer(py_matrices, &view, PyBUF_SIMPLE) < 0) {
    return NULL;
  }
  
  // JPH_Mat4 is 16 floats = 64 bytes.
  size_t required_size = (size_t)joint_count * 64; 
  if ((size_t)view.len < required_size) {
      PyBuffer_Release(&view);
      return PyErr_Format(PyExc_ValueError, 
          "Matrices buffer too small. Expected %zu bytes for %d joints, got %zd",
          required_size, joint_count, view.len);
  }

  // 2. Access Matrix Buffer
  JPH_Mat4 *matrices = (JPH_Mat4 *)view.buf;

  // 3. Construct Pose Object
  JPH_SkeletonPose *pose = JPH_SkeletonPose_Create();
  JPH_SkeletonPose_SetSkeleton(pose, skel);

  JPH_STACK_ALLOC(JPH_RVec3, r_pos);
  r_pos->x = (double)root_x;
  r_pos->y = (double)root_y;
  r_pos->z = (double)root_z;
  JPH_SkeletonPose_SetRootOffset(pose, r_pos);

  // 4. EXECUTE TELEPORT (Shadow Locked)
  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  JPH_Ragdoll_SetPose2(self->ragdoll, r_pos, matrices, true);

  JPH_SkeletonPose_SetJointMatrices(pose, matrices, joint_count);
  JPH_Ragdoll_Activate(self->ragdoll, true);
  JPH_Ragdoll_DriveToPoseUsingMotors(self->ragdoll, pose);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  PyBuffer_Release(&view);
  JPH_SkeletonPose_Destroy(pose);
  Py_RETURN_NONE;
}

static PyObject *Ragdoll_get_body_ids(RagdollObject *self, PyObject *args) {
  // Helper to get the Body Handles of the parts so users can manipulate
  // specific limbs
  PyObject *list = PyList_New((Py_ssize_t)self->body_count);
  SHADOW_LOCK(&self->world->shadow_lock);
  for (size_t i = 0; i < self->body_count; i++) {
    uint32_t slot = self->body_slots[i];
    if (self->world->slot_states[slot] == SLOT_ALIVE) {
      uint32_t gen = self->world->generations[slot];
      PyList_SET_ITEM(list, i,
                      PyLong_FromUnsignedLongLong(make_handle(slot, gen)));
    } else {
      Py_INCREF(Py_None);
      PyList_SET_ITEM(list, i, Py_None);
    }
  }
  SHADOW_UNLOCK(&self->world->shadow_lock);
  return list;
}

static PyObject *Ragdoll_get_debug_info(RagdollObject *self,
                                        PyObject *Py_UNUSED(ignored)) {
  if (!self->ragdoll || !self->world) {
    Py_RETURN_NONE;
  }

  int body_count = JPH_Ragdoll_GetBodyCount(self->ragdoll);
  PyObject *list = PyList_New(body_count);

  JPH_BodyInterface *bi = self->world->body_interface;

  SHADOW_LOCK(&self->world->shadow_lock);
  for (int i = 0; i < body_count; i++) {
    JPH_BodyID bid = JPH_Ragdoll_GetBodyID(self->ragdoll, i);

    JPH_STACK_ALLOC(JPH_RVec3, pos);
    JPH_STACK_ALLOC(JPH_Quat, rot);
    JPH_STACK_ALLOC(JPH_Vec3, vel);

    JPH_BodyInterface_GetPosition(bi, bid, pos);
    JPH_BodyInterface_GetRotation(bi, bid, rot);
    JPH_BodyInterface_GetLinearVelocity(bi, bid, vel);

    PyObject *dict = PyDict_New();
    PyDict_SetItemString(dict, "index", PyLong_FromLong(i));
    PyDict_SetItemString(dict, "pos",
                         Py_BuildValue("(ddd)", pos->x, pos->y, pos->z));
    PyDict_SetItemString(dict, "vel",
                         Py_BuildValue("(fff)", vel->x, vel->y, vel->z));

    PyList_SET_ITEM(list, i, dict);
  }
  SHADOW_UNLOCK(&self->world->shadow_lock);

  return list;
}

static PyObject *PhysicsWorld_register_material(PhysicsWorldObject *self,
                                                PyObject *args,
                                                PyObject *kwds) {
  uint32_t id = 0;
  float friction = 0.5f;
  float restitution = 0.0f;
  // Note: 'combine' requires complex callback logic.
  // For this foundation, we rely on Jolt's standard combination (Geometric Mean
  // for Friction).

  static char *kwlist[] = {"id", "friction", "restitution", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "I|ff", kwlist, &id, &friction,
                                   &restitution)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // 1. Check for existing (Update)
  for (size_t i = 0; i < self->material_count; i++) {
    if (self->materials[i].id == id) {
      self->materials[i].friction = friction;
      self->materials[i].restitution = restitution;
      SHADOW_UNLOCK(&self->shadow_lock);
      Py_RETURN_NONE;
    }
  }

  // 2. Grow capacity if needed
  if (self->material_count >= self->material_capacity) {
    size_t new_cap =
        (self->material_capacity == 0) ? 16 : self->material_capacity * 2;
    MaterialData *new_ptr =
        PyMem_RawRealloc(self->materials, new_cap * sizeof(MaterialData));
    if (!new_ptr) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
    }
    self->materials = new_ptr;
    self->material_capacity = new_cap;
  }

  // 3. Add new
  self->materials[self->material_count].id = id;
  self->materials[self->material_count].friction = friction;
  self->materials[self->material_count].restitution = restitution;
  self->material_count++;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_create_heightfield(PhysicsWorldObject *self,
                                                 PyObject *args,
                                                 PyObject *kwds) {
  float px = 0;
  float py = 0;
  float pz = 0;
  float rx = 0;
  float ry = 0;
  float rz = 0;
  float rw = 1.0f;
  float sx = 1.0f;
  float sy = 1.0f;
  float sz = 1.0f;
  int grid_size = 0;
  Py_buffer h_view = {0};
  uint64_t user_data = 0;
  uint32_t category = 0xFFFF;
  uint32_t mask = 0xFFFF;
  uint32_t material_id = 0;
  float friction = 0.2f;
  float restitution = 0.0f;

  static char *kwlist[] = {"pos",         "rot",       "scale",       "heights",
                           "grid_size",   "user_data", "category",    "mask",
                           "material_id", "friction",  "restitution", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "(fff)(ffff)(fff)y*i|KIIIff", kwlist, &px, &py, &pz, &rx,
          &ry, &rz, &rw, &sx, &sy, &sz, &h_view, &grid_size, &user_data,
          &category, &mask, &material_id, &friction, &restitution)) {
    return NULL;
  }

  if (h_view.len !=
      (Py_ssize_t)((Py_ssize_t)grid_size * grid_size * sizeof(float))) {
    PyBuffer_Release(&h_view);
    return PyErr_Format(PyExc_ValueError,
                        "Height buffer size mismatch. Expected %d floats.",
                        grid_size * grid_size);
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity + 1024) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyBuffer_Release(&h_view);
      return NULL;
    }
  }

  JPH_Vec3 offset = {0, 0, 0};
  JPH_Vec3 scale = {sx, sy, sz};

  // FIX: Added NULL for materialIndices
  JPH_HeightFieldShapeSettings *hf_settings =
      JPH_HeightFieldShapeSettings_Create((float *)h_view.buf, &offset, &scale,
                                          (uint32_t)grid_size, NULL);

  PyBuffer_Release(&h_view);

  if (!hf_settings) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  JPH_Shape *shape =
      (JPH_Shape *)JPH_HeightFieldShapeSettings_CreateShape(hf_settings);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hf_settings);

  if (!shape) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_RuntimeError,
                        "Failed to create HeightField shape");
  }

  uint32_t slot = self->free_slots[--self->free_count];
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)px;
  pos->y = (double)py;
  pos->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx;
  rot->y = ry;
  rot->z = rz;
  rot->w = rw;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, JPH_MotionType_Static, 0);

  JPH_Shape_Destroy(shape);

  JPH_BodyCreationSettings_SetFriction(settings, friction);
  JPH_BodyCreationSettings_SetRestitution(settings, restitution);

  uint32_t gen = self->generations[slot];
  BodyHandle handle = make_handle(slot, gen);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  if (!ensure_command_capacity(self)) {
    JPH_BodyCreationSettings_Destroy(settings);
    self->slot_states[slot] = SLOT_EMPTY;
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
  cmd->create.settings = settings;
  cmd->create.user_data = user_data;
  cmd->create.category = category;
  cmd->create.mask = mask;
  cmd->create.material_id = material_id;

  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject* PhysicsWorld_get_debug_data(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    int draw_shapes = 1;
    int draw_constraints = 1;
    int draw_bounding_box = 0;
    int draw_centers = 0;
    int wireframe = 1;

    static char* kwlist[] = {"shapes", "constraints", "bbox", "centers", "wireframe", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ppppp", kwlist, 
        &draw_shapes, &draw_constraints, &draw_bounding_box, &draw_centers, &wireframe)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 1. Reset Buffer Counts (Reuse memory)
    self->debug_lines.count = 0;
    self->debug_triangles.count = 0;

    // 2. Configure Draw Settings (For Bodies Only)
    JPH_DrawSettings settings;
    JPH_DrawSettings_InitDefault(&settings);
    settings.drawShape = draw_shapes;
    settings.drawShapeWireframe = wireframe;
    settings.drawBoundingBox = draw_bounding_box;
    settings.drawCenterOfMassTransform = draw_centers;
    
    // 3. Draw Bodies
    if (draw_shapes || draw_bounding_box || draw_centers) {
        JPH_PhysicsSystem_DrawBodies(self->system, &settings, self->debug_renderer, NULL);
    }
    
    // 4. Draw Constraints (Explicit Calls)
    if (draw_constraints) {
        JPH_PhysicsSystem_DrawConstraints(self->system, self->debug_renderer);
        JPH_PhysicsSystem_DrawConstraintLimits(self->system, self->debug_renderer);
    }

    // 5. Export to Python Bytes
    // We snapshot the C-arrays into Python immutable bytes objects.
    PyObject* lines_bytes = PyBytes_FromStringAndSize(
        (char*)self->debug_lines.data, 
        (Py_ssize_t)(self->debug_lines.count * sizeof(DebugVertex))
    );
    PyObject* tris_bytes = PyBytes_FromStringAndSize(
        (char*)self->debug_triangles.data, 
        (Py_ssize_t)(self->debug_triangles.count * sizeof(DebugVertex))
    );

    SHADOW_UNLOCK(&self->shadow_lock);

    if (!lines_bytes || !tris_bytes) {
        Py_XDECREF(lines_bytes);
        Py_XDECREF(tris_bytes);
        return PyErr_NoMemory();
    }

    PyObject* ret = PyTuple_Pack(2, lines_bytes, tris_bytes);
    Py_DECREF(lines_bytes);
    Py_DECREF(tris_bytes);
    return ret;
}

/* --- Immutable Getters (Safe without locks) --- */

static PyObject *Vehicle_get_wheel_count(VehicleObject *self, void *closure) {
  // num_wheels is set at creation and never changes
  return PyLong_FromUnsignedLong(self->num_wheels);
}

static PyObject *Character_get_handle(CharacterObject *self, void *closure) {
  // handle is immutable for the life of this Character instance
  return PyLong_FromUnsignedLongLong(self->handle);
}

/* --- Shadow Buffer Getters (Safe via hardened make_view) --- */

static PyObject *get_positions(PhysicsWorldObject *self, void *c) {
  // make_view internally acquires SHADOW_LOCK and snapshots count
  return make_view(self, self->positions);
}

static PyObject *get_rotations(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->rotations);
}

static PyObject *get_velocities(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->linear_velocities);
}

static PyObject *get_angular_velocities(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->angular_velocities);
}

/* --- Mutable Metadata Getters (Hardened with Locks) --- */

static PyObject *get_count(PhysicsWorldObject *self, void *c) {
  SHADOW_LOCK(&self->shadow_lock);
  size_t val = self->count;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromSize_t(val);
}

static PyObject *get_time(PhysicsWorldObject *self, void *c) {
  SHADOW_LOCK(&self->shadow_lock);
  double val = self->time;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyFloat_FromDouble(val);
}

// Buffer Release Slot
static void PhysicsWorld_releasebuffer(PhysicsWorldObject *self,
                                       Py_buffer *view) {
  SHADOW_LOCK(&self->shadow_lock);
  if (self->view_export_count > 0) {
    self->view_export_count--;
  }
  SHADOW_UNLOCK(&self->shadow_lock);
}

// --- Type Definition ---

static const PyGetSetDef PhysicsWorld_getset[] = {
    {"positions", (getter)get_positions, NULL, NULL, NULL},
    {"rotations", (getter)get_rotations, NULL, NULL, NULL},
    {"velocities", (getter)get_velocities, NULL, NULL, NULL},
    {"angular_velocities", (getter)get_angular_velocities, NULL, NULL, NULL},
    {"count", (getter)get_count, NULL, NULL, NULL},
    {"time", (getter)get_time, NULL, NULL, NULL},
    {"user_data", (getter)get_user_data_buffer, NULL, NULL, NULL},
    {NULL}};

static const PyGetSetDef Character_getset[] = {
    {"handle", (getter)Character_get_handle, NULL,
     "The unique physics handle for this character.", NULL},
    {NULL}};

static const PyGetSetDef Vehicle_getset[] = {
    {"wheel_count", (getter)Vehicle_get_wheel_count, NULL,
     "Number of wheels attached to this vehicle.", NULL},
    {NULL}};

static const PyMethodDef PhysicsWorld_methods[] = {
    // --- Lifecycle ---
    {"step", (PyCFunction)PhysicsWorld_step, METH_VARARGS, NULL},
    {"create_body", (PyCFunction)PhysicsWorld_create_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)PhysicsWorld_destroy_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_mesh_body", (PyCFunction)PhysicsWorld_create_mesh_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_constraint", (PyCFunction)PhysicsWorld_create_constraint,
     METH_VARARGS | METH_KEYWORDS,
     "Create a constraint between two bodies. Params depend on type."},
    {"destroy_constraint", (PyCFunction)PhysicsWorld_destroy_constraint,
     METH_VARARGS | METH_KEYWORDS,
     "Remove and destroy a constraint by handle."},
    {"create_vehicle", (PyCFunction)PhysicsWorld_create_vehicle,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_tracked_vehicle", (PyCFunction)PhysicsWorld_create_tracked_vehicle, 
     METH_VARARGS | METH_KEYWORDS, "Create a tank-style vehicle."},
    {"create_ragdoll_settings",
     (PyCFunction)PhysicsWorld_create_ragdoll_settings, METH_VARARGS,
     "Create settings bound to this world"},
    {"create_ragdoll", (PyCFunction)PhysicsWorld_create_ragdoll,
     METH_VARARGS | METH_KEYWORDS, "Instantiate a ragdoll"},
    {"create_heightfield", (PyCFunction)PhysicsWorld_create_heightfield,
     METH_VARARGS | METH_KEYWORDS,
     "Create a static terrain from a height grid."},
   {"create_convex_hull", (PyCFunction)PhysicsWorld_create_convex_hull,
     METH_VARARGS | METH_KEYWORDS,
     "Create a body from a point cloud. Points are wrapped in a convex shell."},
   {"create_compound_body", (PyCFunction)PhysicsWorld_create_compound_body,
     METH_VARARGS | METH_KEYWORDS,
     "Create a body made of multiple primitives. parts=[((x,y,z), (rx,ry,rz,rw), type, size), ...]"},

    // --- Interaction ---
    {"apply_impulse", (PyCFunction)PhysicsWorld_apply_impulse,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"apply_impulse_at", (PyCFunction)PhysicsWorld_apply_impulse_at, 
     METH_VARARGS | METH_KEYWORDS, "Apply impulse at world position."},
    {"apply_buoyancy", (PyCFunction)PhysicsWorld_apply_buoyancy,
     METH_VARARGS | METH_KEYWORDS, "Apply fluid forces to a body."},
    {"apply_buoyancy_batch", (PyCFunction)PhysicsWorld_apply_buoyancy_batch,
     METH_VARARGS | METH_KEYWORDS, "Apply buoyancy to a list of bodies. handles must be a buffer of uint64."},
    {"set_position", (PyCFunction)PhysicsWorld_set_position,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_rotation", (PyCFunction)PhysicsWorld_set_rotation,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_linear_velocity", (PyCFunction)PhysicsWorld_set_linear_velocity,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_angular_velocity", (PyCFunction)PhysicsWorld_set_angular_velocity,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_transform", (PyCFunction)PhysicsWorld_set_transform,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_collision_filter", (PyCFunction)PhysicsWorld_set_collision_filter,
     METH_VARARGS | METH_KEYWORDS, "Dynamically update collision bitmasks."},
    {"register_material", (PyCFunction)PhysicsWorld_register_material,
     METH_VARARGS | METH_KEYWORDS, "Define properties for a material ID."},

    // --- Motion Control ---
    {"get_motion_type", (PyCFunction)PhysicsWorld_get_motion_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_motion_type", (PyCFunction)PhysicsWorld_set_motion_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"activate", (PyCFunction)PhysicsWorld_activate,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"deactivate", (PyCFunction)PhysicsWorld_deactivate,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_ccd", (PyCFunction)PhysicsWorld_set_ccd,
     METH_VARARGS | METH_KEYWORDS, "Enable/Disable Continuous Collision Detection."},

    // --- Queries ---
    {"raycast", (PyCFunction)PhysicsWorld_raycast, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"raycast_batch", (PyCFunction)PhysicsWorld_raycast_batch,
     METH_VARARGS | METH_KEYWORDS, "Execute multiple raycasts efficiently."},
    {"shapecast", (PyCFunction)PhysicsWorld_shapecast,
     METH_VARARGS | METH_KEYWORDS,
     "Sweeps a shape along a direction vector. Returns (Handle, Fraction, "
     "ContactPoint, Normal) or None."},
    {"overlap_sphere", (PyCFunction)PhysicsWorld_overlap_sphere,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"overlap_aabb", (PyCFunction)PhysicsWorld_overlap_aabb,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Utilities ---
    {"get_index", (PyCFunction)PhysicsWorld_get_index,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_alive", (PyCFunction)PhysicsWorld_is_alive,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_active_indices", (PyCFunction)PhysicsWorld_get_active_indices,
     METH_NOARGS,
     "Returns a bytes object containing uint32 indices of all active bodies."},
    {"get_render_state", (PyCFunction)PhysicsWorld_get_render_state,
     METH_VARARGS,
     "Returns a packed bytes object of interpolated positions and rotations "
     "(3+4 floats per body)."},
    {"get_debug_data", (PyCFunction)PhysicsWorld_get_debug_data, METH_VARARGS | METH_KEYWORDS, 
     "Returns (lines_bytes, triangles_bytes). Each vertex is 16 bytes: [x, y, z, color_u32]."},

    // --- User Data ---
    {"get_user_data", (PyCFunction)PhysicsWorld_get_user_data,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_user_data", (PyCFunction)PhysicsWorld_set_user_data,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // -- Event Logic ---
    {"get_contact_events", (PyCFunction)PhysicsWorld_get_contact_events,
     METH_NOARGS, NULL},
    {"get_contact_events_ex", (PyCFunction)PhysicsWorld_get_contact_events_ex,
     METH_NOARGS, "Get rich collision data as dicts"},
    {"get_contact_events_raw", (PyCFunction)PhysicsWorld_get_contact_events_raw,
     METH_NOARGS, "Get raw collision buffer as memoryview"},

    // --- State & Advanced ---
    {"save_state", (PyCFunction)PhysicsWorld_save_state, METH_NOARGS, NULL},
    {"load_state", (PyCFunction)PhysicsWorld_load_state,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_character", (PyCFunction)PhysicsWorld_create_character,
     METH_VARARGS | METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}};

static const PyMethodDef Character_methods[] = {
    {"move", (PyCFunction)Character_move, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_position", (PyCFunction)Character_get_position, METH_NOARGS, NULL},
    {"set_position", (PyCFunction)Character_set_position,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_rotation", (PyCFunction)Character_set_rotation,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_grounded", (PyCFunction)Character_is_grounded, METH_NOARGS, NULL},
    {"set_strength", (PyCFunction)Character_set_strength, METH_VARARGS,
     "Set the max pushing force"},
    {"get_render_transform", (PyCFunction)Character_get_render_transform,
     METH_O,
     "Returns interpolated ((x,y,z), (rx,ry,rz,rw)) based on alpha [0-1]."},
    {NULL}};

static const PyMethodDef Vehicle_methods[] = {
    {"set_input", (PyCFunction)Vehicle_set_input, METH_VARARGS | METH_KEYWORDS,
     "Set driver inputs: forward [-1..1], right [-1..1], brake [0..1], "
     "handbrake [0..1]"},
    {"set_tank_input", (PyCFunction)Vehicle_set_tank_input, METH_VARARGS | METH_KEYWORDS, 
     "Set inputs for tracked vehicle: (left, right, brake)."},
    {"get_wheel_transform", (PyCFunction)Vehicle_get_wheel_transform,
     METH_VARARGS, "Get world-space transform of a wheel by index."},
    {"get_wheel_local_transform",
     (PyCFunction)Vehicle_get_wheel_local_transform, METH_VARARGS,
     "Get local-space transform of a wheel by index."},
    {"destroy", (PyCFunction)Vehicle_destroy, METH_NOARGS,
     "Manually remove the vehicle from physics."},
    {"get_debug_state", (PyCFunction)Vehicle_get_debug_state, METH_NOARGS,
     "Print drivetrain and wheel status to stderr"},
    {NULL}};

static const PyMethodDef Skeleton_methods[] = {
    {"add_joint", (PyCFunction)Skeleton_add_joint, METH_VARARGS,
     "Add joint(name, parent_idx=-1)"},
    {"get_joint_index", (PyCFunction)Skeleton_get_joint_index, METH_VARARGS,
     "Get index by name"},
    {"finalize", (PyCFunction)Skeleton_finalize, METH_NOARGS,
     "Bake skeleton hierarchy"},
    {NULL}};

static const PyMethodDef Ragdoll_methods[] = {
    {"drive_to_pose", (PyCFunction)Ragdoll_drive_to_pose,
     METH_VARARGS | METH_KEYWORDS, "Drive motors to target pose"},
    {"get_body_handles", (PyCFunction)Ragdoll_get_body_ids, METH_NOARGS,
     "Get list of body handles"},
    {"get_debug_info", (PyCFunction)Ragdoll_get_debug_info, METH_NOARGS,
     "Returns list of dicts for each part"},
    {NULL}};

static const PyMethodDef RagdollSettings_methods[] = {
    {"add_part", (PyCFunction)RagdollSettings_add_part,
     METH_VARARGS | METH_KEYWORDS, "Config part"},
    {"stabilize", (PyCFunction)RagdollSettings_stabilize, METH_NOARGS,
     "Auto-detect collisions"},
    {NULL}};

static const PyType_Slot PhysicsWorld_slots[] = {
    {Py_tp_new, PyType_GenericNew},
    {Py_tp_init, PhysicsWorld_init},
    {Py_tp_dealloc, PhysicsWorld_dealloc},
    {Py_tp_methods, (PyMethodDef *)PhysicsWorld_methods},
    {Py_tp_getset, (PyGetSetDef *)PhysicsWorld_getset},
    {Py_bf_releasebuffer, PhysicsWorld_releasebuffer},
    {0, NULL},
};

static const PyType_Slot Character_slots[] = {
    {Py_tp_dealloc, Character_dealloc},
    {Py_tp_traverse, Character_traverse},
    {Py_tp_clear, Character_clear},
    {Py_tp_methods, (PyMethodDef *)Character_methods},
    {Py_tp_getset, (PyGetSetDef *)Character_getset},
    {0, NULL},
};

static const PyType_Slot Vehicle_slots[] = {
    {Py_tp_dealloc, Vehicle_dealloc},
    {Py_tp_traverse, Vehicle_traverse},
    {Py_tp_clear, Vehicle_clear},
    {Py_tp_methods, (PyMethodDef *)Vehicle_methods},
    {Py_tp_getset, (PyGetSetDef *)Vehicle_getset},
    {0, NULL},
};

static const PyType_Slot Skeleton_slots[] = {
    {Py_tp_new, Skeleton_new}, // We defined this
    {Py_tp_dealloc, Skeleton_dealloc},
    {Py_tp_methods, (PyMethodDef *)Skeleton_methods},
    {0, NULL},
};

static const PyType_Spec Skeleton_spec = {
    .name = "culverin._culverin_c.Skeleton",
    .basicsize = sizeof(SkeletonObject),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = (PyType_Slot *)Skeleton_slots,
};

static const PyType_Slot RagdollSettings_slots[] = {
    {Py_tp_dealloc, RagdollSettings_dealloc},
    {Py_tp_methods, (PyMethodDef *)RagdollSettings_methods},
    {0, NULL},
};

static const PyType_Spec PhysicsWorld_spec = {
    .name = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(PhysicsWorldObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = (PyType_Slot *)PhysicsWorld_slots,
};

static const PyType_Spec Character_spec = {
    .name = "culverin._culverin_c.Character",
    .basicsize = sizeof(CharacterObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots = (PyType_Slot *)Character_slots,
};

static const PyType_Spec Vehicle_spec = {
    .name = "culverin._culverin_c.Vehicle",
    .basicsize = sizeof(VehicleObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots = (PyType_Slot *)Vehicle_slots,
};

static const PyType_Spec RagdollSettings_spec = {
    .name = "culverin._culverin_c.RagdollSettings",
    .basicsize = sizeof(RagdollSettingsObject),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = (PyType_Slot *)RagdollSettings_slots,
};

static const PyType_Slot Ragdoll_slots[] = {
    {Py_tp_dealloc, Ragdoll_dealloc},
    {Py_tp_methods, (PyMethodDef *)Ragdoll_methods},
    {0, NULL},
};

static const PyType_Spec Ragdoll_spec = {
    .name = "culverin._culverin_c.Ragdoll",
    .basicsize = sizeof(RagdollObject),
    .flags = Py_TPFLAGS_DEFAULT,
    .slots = (PyType_Slot *)Ragdoll_slots,
};

// --- Module Initialization ---

// 1. Logic for registering types (from the previous refactor)
static int init_types(PyObject *m, CulverinState *st) {
  struct {
    PyType_Spec *spec;
    PyObject **slot;
    const char *name;
  } types[] = {
      {(PyType_Spec *)&PhysicsWorld_spec, &st->PhysicsWorldType,
       "PhysicsWorld"},
      {(PyType_Spec *)&Character_spec, &st->CharacterType, "Character"},
      {(PyType_Spec *)&Vehicle_spec, &st->VehicleType, "Vehicle"},
      {(PyType_Spec *)&RagdollSettings_spec, &st->RagdollSettingsType,
       "RagdollSettings"},
      {(PyType_Spec *)&Ragdoll_spec, &st->RagdollType, "Ragdoll"},
      {(PyType_Spec *)&Skeleton_spec, &st->SkeletonType, "Skeleton"}};

  for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
    PyObject *type = PyType_FromModuleAndSpec(m, types[i].spec, NULL);
    if (!type) {
      return -1;
    }
    if (PyModule_AddObject(m, types[i].name, type) < 0) {
      Py_DECREF(type);
      return -1;
    }
    Py_INCREF(type);
    *types[i].slot = type;
  }
  return 0;
}

// 2. Logic for constants
static int init_constants(PyObject *m) {
  static const struct {
    const char *name;
    int value;
  } consts[] = {{"SHAPE_BOX", 0},         {"SHAPE_SPHERE", 1},
                {"SHAPE_CAPSULE", 2},     {"SHAPE_CYLINDER", 3},
                {"SHAPE_PLANE", 4},       {"SHAPE_MESH", 5},
                {"SHAPE_HEIGHTFIELD", 6}, {"SHAPE_CONVEX_HULL", 7},
                {"MOTION_STATIC", 0},     {"MOTION_KINEMATIC", 1},
                {"MOTION_DYNAMIC", 2},    {"CONSTRAINT_FIXED", 0},
                {"CONSTRAINT_POINT", 1},  {"CONSTRAINT_HINGE", 2},
                {"CONSTRAINT_SLIDER", 3}, {"CONSTRAINT_DISTANCE", 4},
                {"CONSTRAINT_CONE", 5},   {"EVENT_ADDED", 0},
                {"EVENT_PERSISTED", 1},   {"EVENT_REMOVED", 2}};
  for (size_t i = 0; i < sizeof(consts) / sizeof(consts[0]); i++) {
    if (PyModule_AddIntConstant(m, consts[i].name, consts[i].value) < 0) {
      return -1;
    }
  }
  return 0;
}

// 3. Main Entry (Complexity: ~5)
static int culverin_exec(PyObject *m) {
  CulverinState *st = get_culverin_state(m);

  if (!JPH_Init()) {
    PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
    return -1;
  }

  // Initialize the GLOBAL lock for Jolt trampolines
  INIT_LOCK(g_jph_trampoline_lock);
#if PY_VERSION_HEX < 0x030D0000
  if (!g_jph_trampoline_lock) {
    PyErr_NoMemory();
    return -1;
  }
#endif

  st->helper = PyImport_ImportModule("culverin._culverin");
  if (!st->helper) {
    return -1;
  }

  if (init_types(m, st) < 0) {
    return -1;
  }
  if (init_constants(m) < 0) {
    return -1;
  }

  return 0;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
  CulverinState *st = get_culverin_state(m);
  Py_VISIT(st->helper);
  Py_VISIT(st->PhysicsWorldType);
  Py_VISIT(st->CharacterType);
  Py_VISIT(st->VehicleType);
  Py_VISIT(st->RagdollSettingsType);
  Py_VISIT(st->RagdollType);
  Py_VISIT(st->SkeletonType);
  return 0;
}

static int culverin_clear(PyObject *m) {
  CulverinState *st = get_culverin_state(m);
  Py_CLEAR(st->helper);
  Py_CLEAR(st->PhysicsWorldType);
  Py_CLEAR(st->CharacterType);
  Py_CLEAR(st->VehicleType);
  Py_CLEAR(st->RagdollSettingsType);
  Py_CLEAR(st->RagdollType);
  Py_CLEAR(st->SkeletonType);
  return 0;
}

static const PyModuleDef_Slot culverin_slots[] = {
    {Py_mod_exec, culverin_exec},
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyModuleDef culverin_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_culverin_c",
    .m_doc = "Culverin Physics Engine Core",
    .m_size = sizeof(CulverinState),
    .m_slots = (PyModuleDef_Slot *)culverin_slots,
    .m_traverse = culverin_traverse,
    .m_clear = culverin_clear,
};

PyMODINIT_FUNC PyInit__culverin_c(void) {
  return PyModuleDef_Init(&culverin_module);
}