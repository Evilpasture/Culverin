#include "culverin.h"


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

// Character helpers
// Callback: Can the character collide with this object?
static bool JPH_API_CALL
char_on_contact_validate(void *userData, const JPH_CharacterVirtual *character,
                         JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
  return true; // Usually true, unless you want to walk through certain bodies
}

// --- Query Filters ---
static bool JPH_API_CALL filter_allow_all_bp(void* userData, JPH_BroadPhaseLayer layer) {
    return true; // Allow ray to see all broadphase regions
}
static bool JPH_API_CALL filter_allow_all_obj(void* userData, JPH_ObjectLayer layer) {
    return true; // Allow ray to see all object layers (0 and 1)
}

static bool JPH_API_CALL filter_true_body(void *userData, JPH_BodyID bodyID) { return true; }
static bool JPH_API_CALL filter_true_shape(void *userData, const JPH_Shape *shape, const JPH_SubShapeID *id) { return true; }

static const JPH_BodyFilter_Procs global_bf_procs = { .ShouldCollide = filter_true_body };
static const JPH_ShapeFilter_Procs global_sf_procs = { .ShouldCollide = filter_true_shape };

static void JPH_API_CALL char_on_character_contact_added(
    void *userData, const JPH_CharacterVirtual *character, const JPH_CharacterVirtual *otherCharacter,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {
    
    // 1. Resolve Self
    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;
    PhysicsWorldObject *world = self->world;

    // 2. Define Physics Interaction
    // canPushCharacter: Allows 'character' to push 'otherCharacter'
    // canReceiveImpulses: Allows the characters to exchange momentum
    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;

    // 3. GENERATIONAL EVENT REPORTING (Lock-Free)
    // Character-vs-Character collisions are handled in a special pass by Jolt.
    // To make them show up in Python's get_contact_events(), we record them manually.

    // ID 1: Our own immutable handle
    BodyHandle h1 = self->handle;

    // ID 2: Retrieve the handle of the other character.
    // We get the inner BodyID, then read the 'Handle' we stamped into its UserData.
    JPH_BodyID other_bid = JPH_CharacterVirtual_GetInnerBodyID(otherCharacter);
    BodyHandle h2 = (BodyHandle)JPH_BodyInterface_GetUserData(world->body_interface, other_bid);

    // 4. Atomic Reservation in the Global Event Buffer
    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);

    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];

        // Ensure consistent ordering (h1 < h2) to avoid duplicate entries 
        // if both characters are being updated in the same step.
        if (h1 < h2) {
            ev->body1 = h1;
            ev->body2 = h2;
        } else {
            ev->body1 = h2;
            ev->body2 = h1;
        }

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
        // Synchronizes this write with the Python thread's 'acquire' in get_contact_events
        atomic_thread_fence(memory_order_release);
    }
}

// Callback: Handle the collision settings AND Apply Impulse
static void JPH_API_CALL char_on_contact_added(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {
  
  // 1. Safe Defaults
  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  // 2. Resolve Character Object
  if (!userData) return;
  CharacterObject *self = (CharacterObject *)userData;
  
  // 3. Thread-Safe Member Access
  // We use relaxed atomics here. If the main thread is mid-write, 
  // we just want 'a' valid value, we don't need strict synchronization.
  float vx = atomic_load_explicit((_Atomic float*)&self->last_vx, memory_order_relaxed);
  float vy = atomic_load_explicit((_Atomic float*)&self->last_vy, memory_order_relaxed);
  float vz = atomic_load_explicit((_Atomic float*)&self->last_vz, memory_order_relaxed);
  float strength = atomic_load_explicit((_Atomic float*)&self->push_strength, memory_order_relaxed);

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
  float dot = vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

  // Threshold prevents micro-jitter and pushing objects just by grazing them
  if (dot > 0.1f) {
    float factor = dot * strength;
    
    // Safety Cap: Prevent "Physics Nukes" from velocity spikes
    const float max_impulse = 5000.0f; 
    if (factor > max_impulse) factor = max_impulse;

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
    // This is a thread-safe call into Jolt's internal command queue/locking system
    JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
    JPH_BodyInterface_ActivateBody(bi, bodyID2);
  }
}

// Helper to find an arbitrary vector perpendicular to 'in'
static void vec3_get_perpendicular(const JPH_Vec3* in, JPH_Vec3* out) {
    if (fabsf(in->x) > fabsf(in->z)) {
        out->x = -in->y; out->y = in->x; out->z = 0.0f; // Cross(in, Z)
    } else {
        out->x = 0.0f; out->y = -in->z; out->z = in->y; // Cross(in, X)
    }
    // Normalize
    float len = sqrtf(out->x * out->x + out->y * out->y + out->z * out->z);
    if (len > 1e-6f) {
        float inv = 1.0f / len;
        out->x *= inv; out->y *= inv; out->z *= inv;
    } else {
        // Fallback if 'in' is zero
        out->x = 1.0f; out->y = 0.0f; out->z = 0.0f;
    }
}

// --- Global Contact Listener ---
static void JPH_API_CALL on_contact_added(void* userData, const JPH_Body* body1, const JPH_Body* body2, const JPH_ContactManifold* manifold, JPH_ContactSettings* settings) {
    PhysicsWorldObject* self = (PhysicsWorldObject*)userData;

    // 1. ATOMIC RESERVATION (Wait-Free)
    // "idx" is our exclusive slot. No other thread will write here.
    size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);

    // 2. BOUNDS CHECK
    if (idx >= self->contact_max_capacity) {
        return; // Buffer full, drop event
    }

    ContactEvent* ev = &self->contact_buffer[idx];

    // 3. RETRIEVE IMMUTABLE HANDLES
    // We read the handles stamped at creation. 
    // Even if the main thread has destroyed the body and incremented the generation
    // in the shadow array, these handles reflect the bodies *as they existed* 
    // when the collision occurred.
    ev->body1 = (BodyHandle)JPH_Body_GetUserData((JPH_Body*)body1);
    ev->body2 = (BodyHandle)JPH_Body_GetUserData((JPH_Body*)body2);

    // 4. GATHER GEOMETRY (No Locks)
    JPH_Vec3 n;
    JPH_ContactManifold_GetWorldSpaceNormal(manifold, &n);
    ev->nx = n.x; ev->ny = n.y; ev->nz = n.z;

    if (JPH_ContactManifold_GetPointCount(manifold) > 0) {
        JPH_RVec3 p;
        JPH_ContactManifold_GetWorldSpaceContactPointOn1(manifold, 0, &p);
        ev->px = (float)p.x; ev->py = (float)p.y; ev->pz = (float)p.z;
    } else {
        ev->px = 0.0f; ev->py = 0.0f; ev->pz = 0.0f;
    }

    // 5. CALCULATE IMPULSE (No Locks)
    JPH_Vec3 v1, v2;
    JPH_Body_GetLinearVelocity((JPH_Body*)body1, &v1);
    JPH_Body_GetLinearVelocity((JPH_Body*)body2, &v2);
    
    float closing_speed = (v1.x - v2.x) * n.x + 
                          (v1.y - v2.y) * n.y + 
                          (v1.z - v2.z) * n.z;
    ev->impulse = fabsf(closing_speed);

    // 6. PUBLICATION FENCE (Critical for Weak Memory Models)
    // Ensures all writes to 'ev' above are visible to any thread that
    // later observes 'contact_atomic_idx > idx'.
    atomic_thread_fence(memory_order_release);
}

// Fixed get_contact_events to be safer with locking
static PyObject* PhysicsWorld_get_contact_events(PhysicsWorldObject* self, PyObject* args) {
    // 1. Enter the lock to check the state machine
    SHADOW_LOCK(&self->shadow_lock);
    
    // GUARD: Ensure we aren't mid-step and prevent a new step from starting
    if (self->is_stepping) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot read events while physics is stepping.");
        return NULL;
    }

    // 2. Acquire index (Memory Visibility)
    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyList_New(0);
    }

    if (count > self->contact_max_capacity) count = self->contact_max_capacity;

    // 3. Fast Copy (Hold lock for the shortest possible time)
    ContactEvent* scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
    if (!scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }
    
    // This is the only part that MUST be synchronized with step()
    memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));
    
    // Clear the buffer for the next frame
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

    // 4. EXIT the lock immediately
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. Slow Python Work (Done while the next physics step can run in parallel!)
    PyObject* list = PyList_New((Py_ssize_t)count);
    for (size_t i = 0; i < count; i++) {
        PyObject* item = PyTuple_New(2);
        PyTuple_SET_ITEM(item, 0, PyLong_FromUnsignedLongLong(scratch[i].body1));
        PyTuple_SET_ITEM(item, 1, PyLong_FromUnsignedLongLong(scratch[i].body2));
        PyList_SET_ITEM(list, (Py_ssize_t)i, item);
    }

    PyMem_RawFree(scratch);
    return list;
}

static PyObject* PhysicsWorld_get_contact_events_ex(PhysicsWorldObject* self, PyObject* args) {
    // 1. Enter lock and check if step() is running
    SHADOW_LOCK(&self->shadow_lock);
    
    if (self->is_stepping) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot get contact events while physics is stepping.");
        return NULL;
    }

    // 2. Acquire index with memory visibility
    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    if (count == 0) { 
        SHADOW_UNLOCK(&self->shadow_lock); 
        return PyList_New(0); 
    }

    // Defensive clamping
    if (count > self->contact_max_capacity) {
        count = self->contact_max_capacity;
    }

    // 3. Fast Scratch Copy
    ContactEvent* scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
    if (!scratch) { 
        SHADOW_UNLOCK(&self->shadow_lock); 
        return PyErr_NoMemory(); 
    }
    
    // Copy from the lock-free contact_buffer
    memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));
    
    // Reset the index so the next physics step starts at 0
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. Build Python Objects (Outside the lock)
    // Optimization: Pre-intern the dictionary keys
    PyObject* k_bodies = PyUnicode_InternFromString("bodies");
    PyObject* k_pos    = PyUnicode_InternFromString("position");
    PyObject* k_norm   = PyUnicode_InternFromString("normal");
    PyObject* k_str    = PyUnicode_InternFromString("strength");

    if (!k_bodies || !k_pos || !k_norm || !k_str) {
        Py_XDECREF(k_bodies); Py_XDECREF(k_pos); 
        Py_XDECREF(k_norm); Py_XDECREF(k_str);
        PyMem_RawFree(scratch);
        return NULL;
    }

    PyObject* list = PyList_New((Py_ssize_t)count);
    if (!list) {
        Py_DECREF(k_bodies); Py_DECREF(k_pos); 
        Py_DECREF(k_norm); Py_DECREF(k_str);
        PyMem_RawFree(scratch);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEvent* e = &scratch[i];
        
        PyObject* dict = PyDict_New();
        if (!dict) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, (Py_ssize_t)i, Py_None);
            continue;
        }

        // 1. Bodies Tuple (u64, u64)
        PyObject* b_tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(b_tuple, 0, PyLong_FromUnsignedLongLong(e->body1));
        PyTuple_SET_ITEM(b_tuple, 1, PyLong_FromUnsignedLongLong(e->body2));
        PyDict_SetItem(dict, k_bodies, b_tuple); 
        Py_DECREF(b_tuple); 

        // 2. Position Tuple (f, f, f)
        PyObject* p_tuple = PyTuple_New(3);
        PyTuple_SET_ITEM(p_tuple, 0, PyFloat_FromDouble(e->px));
        PyTuple_SET_ITEM(p_tuple, 1, PyFloat_FromDouble(e->py));
        PyTuple_SET_ITEM(p_tuple, 2, PyFloat_FromDouble(e->pz));
        PyDict_SetItem(dict, k_pos, p_tuple);
        Py_DECREF(p_tuple);

        // 3. Normal Tuple (f, f, f)
        PyObject* n_tuple = PyTuple_New(3);
        PyTuple_SET_ITEM(n_tuple, 0, PyFloat_FromDouble(e->nx));
        PyTuple_SET_ITEM(n_tuple, 1, PyFloat_FromDouble(e->ny));
        PyTuple_SET_ITEM(n_tuple, 2, PyFloat_FromDouble(e->nz));
        PyDict_SetItem(dict, k_norm, n_tuple);
        Py_DECREF(n_tuple);

        // 4. Strength Float (Closing speed)
        PyObject* s_val = PyFloat_FromDouble(e->impulse);
        PyDict_SetItem(dict, k_str, s_val);
        Py_DECREF(s_val);

        PyList_SET_ITEM(list, (Py_ssize_t)i, dict);
    }

    // Cleanup Interned Keys
    Py_DECREF(k_bodies);
    Py_DECREF(k_pos);
    Py_DECREF(k_norm);
    Py_DECREF(k_str);

    PyMem_RawFree(scratch);
    return list;
}

// ContactEvent layout (packed, little-endian):
// uint64 body1, uint64 body2
// float32 px, py, pz
// float32 nx, ny, nz
// float32 impulse
static PyObject* PhysicsWorld_get_contact_events_raw(PhysicsWorldObject* self, PyObject* args) {
    // 1. Phase Guard
    SHADOW_LOCK(&self->shadow_lock);
    GUARD_STEPPING(self);

    // 2. Atomic Acquire (Publication Visibility)
    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    
    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        // Return empty view
        PyObject* empty = PyBytes_FromStringAndSize("", 0);
        PyObject* view = PyMemoryView_FromObject(empty);
        Py_DECREF(empty);
        return view;
    }

    if (count > self->contact_max_capacity) {
        count = self->contact_max_capacity;
    }

    // 3. Snapshot Data
    // We copy into a PyBytes object. This is fast (memcpy) and 
    // ensures the data remains valid even after the next step() resets the buffer.
    size_t bytes_size = count * sizeof(ContactEvent);
    PyObject* raw_bytes = PyBytes_FromStringAndSize((char*)self->contact_buffer, (Py_ssize_t)bytes_size);
    
    // 4. Reset Index for next frame
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    
    SHADOW_UNLOCK(&self->shadow_lock);

    if (!raw_bytes) return NULL;

    // 5. Wrap in MemoryView
    // This allows the user to use np.frombuffer(events, dtype=...) without extra copies
    PyObject* view = PyMemoryView_FromObject(raw_bytes);
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

static JPH_ContactListener_Procs contact_procs = {
    .OnContactValidate = NULL, 
    .OnContactAdded = on_contact_added,
    .OnContactPersisted = NULL,
    .OnContactRemoved = NULL
};

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
      key.p1 = params[0]; key.p2 = params[1]; key.p3 = params[2];
      break;
    case 1: // SPHERE: Uses 1 param (radius)
      key.p1 = params[0];
      break;
    case 2: // CAPSULE: Uses 2 params (half-height, radius)
    case 3: // CYLINDER: Uses 2 params (half-height, radius)
      key.p1 = params[0]; key.p2 = params[1];
      break;
    case 4: // PLANE: Uses 4 params (nx, ny, nz, d)
      key.p1 = params[0]; key.p2 = params[1]; key.p3 = params[2]; key.p4 = params[3];
      break;
    default: break;
  }

  // 2. CACHE LOOKUP
  // Linear search is fine for typical game usage (usually < 100 unique shapes)
  for (size_t i = 0; i < self->shape_cache_count; i++) {
    if (memcmp(&self->shape_cache[i].key, &key, sizeof(ShapeKey)) == 0) {
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
    JPH_CapsuleShapeSettings *s = JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
    shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 3) {
    JPH_CylinderShapeSettings *s = JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
    shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 4) {
    JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
    // Note: Planes in Jolt often require a half-extent (1000.0f) to define their "active" area
    JPH_PlaneShapeSettings *s = JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
    shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  }

  if (!shape) return NULL;

  // 4. CACHE EXPANSION
  if (self->shape_cache_count >= self->shape_cache_capacity) {
    size_t new_cap = (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
    // Note: PyMem_RawRealloc is safe here because this is called under SHADOW_LOCK 
    // and is not inside the Jolt step.
    void *new_ptr = PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
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

// --- Helper: Resource Cleanup (Idempotent) ---
// SAFETY:
// - Must not be called while PhysicsSystem is stepping
// - Must not be called from a Jolt callback
// - Must not race with Python memoryview access
static void PhysicsWorld_free_members(PhysicsWorldObject *self) {
  // --- 0. Pre-allocation Safety ---
  // If the object was partially allocated, some arrays might be NULL.
  // We check array existence before any loops.

  // --- 1. Vehicles (Special Constraints) ---
  // Vehicles must be destroyed before the system because they act as constraints
  // and step-listeners.
  // Note: Since VehicleObjects are Python-owned, they usually handle their own 
  // Jolt cleanup in Vehicle_dealloc, but we ensure the system is still 
  // valid while they exist.

  // --- 2. Constraints (Standard) ---
  // CRITICAL: Constraints must be destroyed BEFORE the PhysicsSystem.
  // This ensures Jolt handles the reference counting while the BodyManager is alive.
  if (self->constraints) {
    for (size_t i = 0; i < self->constraint_capacity; i++) {
      // Only destroy if the pointer exists AND the state array confirms it is alive.
      // We check self->constraint_states existence to handle partial init failure.
      if (self->constraints[i]) {
        if (!self->constraint_states || self->constraint_states[i] == SLOT_ALIVE) {
          // If the system is still alive, Jolt recommends removing before destroying
          if (self->system) {
            JPH_PhysicsSystem_RemoveConstraint(self->system, self->constraints[i]);
          }
          JPH_Constraint_Destroy(self->constraints[i]);
        }
        self->constraints[i] = NULL;
      }
    }
    PyMem_RawFree((void *)self->constraints);
    self->constraints = NULL;
  }
  
  if (self->constraint_generations) { PyMem_RawFree(self->constraint_generations); self->constraint_generations = NULL; }
  if (self->free_constraint_slots) { PyMem_RawFree(self->free_constraint_slots); self->free_constraint_slots = NULL; }
  if (self->constraint_states) { PyMem_RawFree(self->constraint_states); self->constraint_states = NULL; }

  // --- 3. Jolt Systems ---
  // Now that constraints are gone, it is safe to tear down the system.
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

  // --- 4. Filters & Interfaces ---
  // Still missing Destroy APIs in JoltC as of current version.
  if (self->bp_interface) self->bp_interface = NULL;
  if (self->pair_filter)   self->pair_filter = NULL;
  if (self->bp_filter)     self->bp_filter = NULL;

  // --- 5. Shape Cache ---
  if (self->shape_cache) {
    for (size_t i = 0; i < self->shape_cache_count; i++) {
      if (self->shape_cache[i].shape) {
        JPH_Shape_Destroy(self->shape_cache[i].shape);
      }
    }
    PyMem_RawFree(self->shape_cache);
    self->shape_cache = NULL;
    self->shape_cache_count = 0;
  }

  // --- 6. Contact Listener & Atomic Buffer ---
  if (self->contact_listener) { 
      JPH_ContactListener_Destroy(self->contact_listener); 
      self->contact_listener = NULL;
  }
  if (self->contact_buffer) { 
      PyMem_RawFree(self->contact_buffer); 
      self->contact_buffer = NULL;
  }

  // --- 7. Core Shadow Buffers ---
  if (self->positions) { PyMem_RawFree(self->positions); self->positions = NULL; }
  if (self->rotations) { PyMem_RawFree(self->rotations); self->rotations = NULL; }
  if (self->prev_positions) { PyMem_RawFree(self->prev_positions); self->prev_positions = NULL; }
  if (self->prev_rotations) { PyMem_RawFree(self->prev_rotations); self->prev_rotations = NULL; }
  if (self->linear_velocities) { PyMem_RawFree(self->linear_velocities); self->linear_velocities = NULL; }
  if (self->angular_velocities) { PyMem_RawFree(self->angular_velocities); self->angular_velocities = NULL; }
  
  if (self->body_ids) { PyMem_RawFree(self->body_ids); self->body_ids = NULL; }
  if (self->generations) { PyMem_RawFree(self->generations); self->generations = NULL; }
  if (self->slot_to_dense) { PyMem_RawFree(self->slot_to_dense); self->slot_to_dense = NULL; }
  if (self->dense_to_slot) { PyMem_RawFree(self->dense_to_slot); self->dense_to_slot = NULL; }
  if (self->free_slots) { PyMem_RawFree(self->free_slots); self->free_slots = NULL; }
  if (self->slot_states) { PyMem_RawFree(self->slot_states); self->slot_states = NULL; }
  if (self->command_queue) { PyMem_RawFree(self->command_queue); self->command_queue = NULL; }
  if (self->user_data) { PyMem_RawFree(self->user_data); self->user_data = NULL; }

#if PY_VERSION_HEX < 0x030D0000
  if (self->shadow_lock) {
    PyThread_free_lock(self->shadow_lock);
    self->shadow_lock = NULL;
  }
#endif
}

// --- Lifecycle: Deallocation ---
static void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
  PhysicsWorld_free_members(self);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Lifecycle: Initialization ---
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                             PyObject *kwds) {
  static char *kwlist[] = {"settings", "bodies", NULL};
  PyObject *settings_dict = NULL;
  PyObject *bodies_list = NULL;
  
  PyObject *val_func = NULL;
  PyObject *norm_settings = NULL;
  PyObject *bake_func = NULL;
  PyObject *baked = NULL;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &settings_dict,
                                   &bodies_list)) {
    return -1;
  }
  
  // 1. Explicitly zero counters (prevent UB)
  self->view_export_count = 0;
  self->free_count = 0;
  self->command_count = 0;
  self->contact_count = 0;
  self->shape_cache_count = 0;
  self->constraint_count = 0;
  self->free_constraint_count = 0;
  self->count = 0;
  self->time = 0.0;
  
  // Initialize Pointers to NULL for safe fail-cleanup
  self->system = NULL;
  self->job_system = NULL;
  self->bp_interface = NULL;
  self->pair_filter = NULL;
  self->bp_filter = NULL;
  self->contact_listener = NULL;
  self->contact_buffer = NULL; // Updated Name
  self->char_vs_char_manager = NULL;
  self->positions = NULL;

  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);

  val_func = PyObject_GetAttrString(st->helper, "validate_settings");
  if (!val_func) goto fail;

  norm_settings = PyObject_CallFunctionObjArgs(val_func, settings_dict ? settings_dict : Py_None, NULL);
  if (!norm_settings) goto fail;

  float gx, gy, gz, slop;
  int max_bodies, max_pairs;
  if (!PyArg_ParseTuple(norm_settings, "ffffii", &gx, &gy, &gz, &slop, &max_bodies, &max_pairs)) goto fail;
  Py_CLEAR(norm_settings); 

  self->is_stepping = false;

  // 2. Jolt Systems Initialization
  JobSystemThreadPoolConfig job_cfg = {.maxJobs = 1024, .maxBarriers = 8, .numThreads = -1};
  self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
  
  self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(2, 2);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 0, 0);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 1, 1);

  self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(2);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 0);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1);
  
  self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(self->bp_interface, 2, self->pair_filter, 2);

  JPH_PhysicsSystemSettings phys_settings = {
      .maxBodies = (uint32_t)max_bodies, // Use validated setting
      .maxBodyPairs = (uint32_t)max_pairs,
      .maxContactConstraints = 102400,
      .broadPhaseLayerInterface = self->bp_interface,
      .objectLayerPairFilter = self->pair_filter,
      .objectVsBroadPhaseLayerFilter = self->bp_filter};
  
  self->system = JPH_PhysicsSystem_Create(&phys_settings);
  self->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();

  JPH_Vec3 gravity = {gx, gy, gz};
  JPH_PhysicsSystem_SetGravity(self->system, &gravity);
  self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);

  // 3. ATOMIC Contact Buffer Setup
  self->contact_max_capacity = 4096; // Sane default batch size
  self->contact_buffer = PyMem_RawMalloc(self->contact_max_capacity * sizeof(ContactEvent));
  if (!self->contact_buffer) { PyErr_NoMemory(); goto fail; }
  
  // Initialize the C11 atomic counter
  atomic_init(&self->contact_atomic_idx, 0);

  JPH_ContactListener_SetProcs(&contact_procs);
  self->contact_listener = JPH_ContactListener_Create(self); 
  JPH_PhysicsSystem_SetContactListener(self->system, self->contact_listener);

  // 4. ABI / Alignment Check
  {
      JPH_BoxShapeSettings* bs = JPH_BoxShapeSettings_Create(&(JPH_Vec3){1,1,1}, 0.0f);
      JPH_Shape* shape = (JPH_Shape*)JPH_BoxShapeSettings_CreateShape(bs);
      JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)bs);
      if (!shape) { PyErr_NoMemory(); goto fail; }

      JPH_BodyCreationSettings* bcs = JPH_BodyCreationSettings_Create3(
          shape, &(JPH_RVec3){10.0, 20.0, 30.0}, &(JPH_Quat){0,0,0,1}, JPH_MotionType_Static, 0);
      JPH_Shape_Destroy(shape);
      if (!bcs) { PyErr_NoMemory(); goto fail; }

      JPH_BodyID bid = JPH_BodyInterface_CreateAndAddBody(self->body_interface, bcs, JPH_Activation_Activate);
      JPH_BodyCreationSettings_Destroy(bcs);

      JPH_STACK_ALLOC(JPH_RVec3, p_check);
      JPH_BodyInterface_GetPosition(self->body_interface, bid, p_check);
      JPH_BodyInterface_RemoveBody(self->body_interface, bid);
      JPH_BodyInterface_DestroyBody(self->body_interface, bid);

      // Verify double precision alignment
      if (fabs(p_check->x - 10.0) > 0.1 || fabs(p_check->y - 20.0) > 0.1) {
          PyErr_SetString(PyExc_RuntimeError, "JoltC ABI Mismatch: Precision issue.");
          goto fail;
      }
  }

  // 5. Bake Scene
  size_t baked_count = 0;
  if (bodies_list && bodies_list != Py_None) {
    bake_func = PyObject_GetAttrString(st->helper, "bake_scene");
    baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
    if (!baked) goto fail;
    baked_count = PyLong_AsSize_t(PyTuple_GetItem(baked, 0));
  }
  Py_XDECREF(bake_func); 

  self->count = baked_count;
  self->capacity = (size_t)max_bodies;
  if (self->capacity < self->count + 128) self->capacity = self->count + 1024;

  // 6. Native Buffer Allocations
  self->positions = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->rotations = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->prev_positions = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->prev_rotations = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->linear_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->angular_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->body_ids = PyMem_RawMalloc(self->capacity * sizeof(JPH_BodyID));
  self->user_data = PyMem_RawCalloc(self->capacity, sizeof(uint64_t));
  
  self->slot_capacity = self->capacity;
  self->generations = PyMem_RawCalloc(self->slot_capacity, sizeof(uint32_t));
  self->slot_to_dense = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->dense_to_slot = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->free_slots = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->slot_states = PyMem_RawCalloc(self->slot_capacity, sizeof(uint8_t));
  
  self->command_capacity = 64;
  self->command_queue = PyMem_RawMalloc(self->command_capacity * sizeof(PhysicsCommand));

  self->constraint_capacity = 256; 
  self->constraints = (JPH_Constraint **)PyMem_RawCalloc(self->constraint_capacity, sizeof(JPH_Constraint*));
  self->constraint_generations = PyMem_RawCalloc(self->constraint_capacity, sizeof(uint32_t));
  self->free_constraint_slots = PyMem_RawMalloc(self->constraint_capacity * sizeof(uint32_t));
  self->constraint_states = PyMem_RawCalloc(self->constraint_capacity, sizeof(uint8_t));

  if (!self->positions || !self->rotations || !self->prev_positions ||
    !self->prev_rotations || !self->linear_velocities || !self->angular_velocities ||
    !self->body_ids || !self->user_data ||
    !self->generations || !self->slot_to_dense || !self->dense_to_slot ||
    !self->free_slots || !self->slot_states ||
    !self->command_queue ||
    !self->constraints || !self->constraint_generations ||
    !self->free_constraint_slots || !self->constraint_states) {
    PyErr_NoMemory();
    goto fail;
  }


  // Init Indices
  for(uint32_t i=0; i<self->constraint_capacity; i++) {
      self->constraint_generations[i] = 1;
      self->free_constraint_slots[i] = i;
  }
  self->free_constraint_count = self->constraint_capacity;

  // 7. Apply Baked Data
  if (baked) {
    float *f_pos = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 1));
    float *f_rot = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 2));
    float *f_shape = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 3));
    unsigned char *u_mot = (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 4));
    unsigned char *u_layer = (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 5));
    uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, 6));

    for (size_t i = 0; i < self->count; i++) {
      JPH_STACK_ALLOC(JPH_RVec3, body_pos);
      body_pos->x = f_pos[i * 4]; body_pos->y = f_pos[i * 4 + 1]; body_pos->z = f_pos[i * 4 + 2];
      JPH_STACK_ALLOC(JPH_Quat, body_rot);
      body_rot->x = f_rot[i * 4]; body_rot->y = f_rot[i * 4 + 1]; body_rot->z = f_rot[i * 4 + 2]; body_rot->w = f_rot[i * 4 + 3];

      float params[4] = {f_shape[i * 5 + 1], f_shape[i * 5 + 2], f_shape[i * 5 + 3], f_shape[i * 5 + 4]};
      JPH_Shape *shape = find_or_create_shape(self, (int)f_shape[i * 5], params);

      if (!shape) {
        PyErr_SetString(PyExc_Warning, "Failed to create shape");
        goto fail;
      }

      if (shape) {
        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(shape, body_pos, body_rot, (JPH_MotionType)u_mot[i], (JPH_ObjectLayer)u_layer[i]);
        
        // FIX 1: Set Full Generational Handle in UserData
        self->generations[i] = 1;
        BodyHandle h = make_handle((uint32_t)i, 1);
        JPH_BodyCreationSettings_SetUserData(creation, (uint64_t)h);

        if (u_mot[i] == 2) JPH_BodyCreationSettings_SetAllowSleeping(creation, true);

        self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(self->body_interface, creation, JPH_Activation_Activate);
        JPH_BodyCreationSettings_Destroy(creation);

        self->slot_to_dense[i] = (uint32_t)i;
        self->dense_to_slot[i] = (uint32_t)i;
        self->slot_states[i] = SLOT_ALIVE;
        self->user_data[i] = u_data[i];
      }
    }
    Py_CLEAR(baked);
  }

  // Init Free Slots
  for (uint32_t i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity; i++) {
    self->generations[i] = 1;
    self->free_slots[self->free_count++] = i;
  }
  
  culverin_sync_shadow_buffers(self);
  return 0;

fail:
  PhysicsWorld_free_members(self); 
  Py_XDECREF(val_func);
  Py_XDECREF(norm_settings);
  Py_XDECREF(bake_func);
  Py_XDECREF(baked);
  return -1;
}

static PyObject *PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &h, &x, &y,
                                   &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  GUARD_STEPPING(self); 
  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
  JPH_Vec3 imp = {x, y, z};
  JPH_BodyInterface_AddImpulse(self->body_interface,
                               self->body_ids[self->slot_to_dense[slot]], &imp);
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    float sx, sy, sz, dx, dy, dz, max_dist = 1000.0f;
    uint64_t ignore_h = 0;
    static char *kwlist[] = {"start", "direction", "max_dist", "ignore", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|fK", kwlist, 
                                     &sx, &sy, &sz, 
                                     &dx, &dy, &dz, 
                                     &max_dist, &ignore_h)) {
        return NULL;
    }

    PyObject* result = NULL;
    bool query_active = false;
    JPH_BodyID ignore_bid = 0;

    // --- 1. ENTRY & LOCKING ---
    SHADOW_LOCK(&self->shadow_lock);
    if (self->is_stepping) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot raycast while physics is stepping");
        return NULL;
    }

    atomic_fetch_add(&self->active_queries, 1);
    query_active = true;

    if (ignore_h != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, ignore_h, &ignore_slot)) {
            ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 2. VALIDATION ---
    float mag_sq = dx*dx + dy*dy + dz*dz;
    if (mag_sq < 1e-9f) goto exit; 

    // --- 3. JOLT EXECUTION ---
    float mag = sqrtf(mag_sq);
    float scale = max_dist / mag;

    JPH_STACK_ALLOC(JPH_RVec3, origin);
    origin->x = sx; origin->y = sy; origin->z = sz;
    JPH_STACK_ALLOC(JPH_Vec3, direction);
    direction->x = dx * scale; direction->y = dy * scale; direction->z = dz * scale;
    JPH_STACK_ALLOC(JPH_RayCastResult, hit);
    memset(hit, 0, sizeof(JPH_RayCastResult));

    // --- SETUP FILTERS ---
    JPH_BroadPhaseLayerFilter_Procs bp_procs = { .ShouldCollide = filter_allow_all_bp };
    JPH_BroadPhaseLayerFilter* bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
    JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

    JPH_ObjectLayerFilter_Procs obj_procs = { .ShouldCollide = filter_allow_all_obj };
    JPH_ObjectLayerFilter* obj_filter = JPH_ObjectLayerFilter_Create(NULL);
    JPH_ObjectLayerFilter_SetProcs(&obj_procs);

    CastShapeFilter filter_ctx = { .ignore_id = ignore_bid };
    JPH_BodyFilter_Procs filter_procs = { .ShouldCollide = CastShape_BodyFilter };
    JPH_BodyFilter* body_filter = JPH_BodyFilter_Create(&filter_ctx);

    SHADOW_LOCK(&g_jph_trampoline_lock);
    JPH_BodyFilter_SetProcs(&filter_procs);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    
    // Pass the layer filters here
    bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, 
                                                bp_filter, obj_filter, body_filter);

    JPH_BodyFilter_SetProcs(&global_bf_procs);
    SHADOW_UNLOCK(&g_jph_trampoline_lock);

    JPH_BodyFilter_Destroy(body_filter);
    JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    JPH_ObjectLayerFilter_Destroy(obj_filter);

    if (!has_hit) goto exit;

    // --- 4. EXTRACT GEOMETRY ---
    JPH_Vec3 normal = {0, 1, 0}; 
    const JPH_BodyLockInterface* lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockRead lock;
    JPH_BodyLockInterface_LockRead(lock_iface, hit->bodyID, &lock);
    if (lock.body) {
        JPH_RVec3 hit_p = { origin->x + direction->x * hit->fraction, 
                           origin->y + direction->y * hit->fraction, 
                           origin->z + direction->z * hit->fraction };
        JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, hit->subShapeID2, &hit_p, &normal);
    }
    JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);

    // --- 5. RESOLVE IDENTITY (Consistent with shapecast) ---
    SHADOW_LOCK(&self->shadow_lock);
    // UserData is the baked 64-bit handle
    BodyHandle handle = (BodyHandle)JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
    uint32_t slot = (uint32_t)(handle & 0xFFFFFFFF);
    uint32_t gen  = (uint32_t)(handle >> 32);

    if (slot < self->slot_capacity && 
        self->generations[slot] == gen && 
        self->slot_states[slot] == SLOT_ALIVE) 
    {
        result = Py_BuildValue("Kf(fff)", handle, hit->fraction, normal.x, normal.y, normal.z);
    }
    SHADOW_UNLOCK(&self->shadow_lock);

exit:
    if (query_active) atomic_fetch_sub(&self->active_queries, 1);

    if (result) return result;
    Py_RETURN_NONE;
}

// Callback: Called by Jolt when a hit is found during the sweep
static float CastShape_ClosestCollector(void* context, const JPH_ShapeCastResult* result) {
    CastShapeContext* ctx = (CastShapeContext*)context;
    
    // We only care about the closest hit (smallest fraction)
    if (result->fraction < ctx->hit.fraction) {
        ctx->hit = *result;
        ctx->has_hit = true;
    }
    
    // Returning the fraction tells Jolt to ignore any future hits further than this one
    return result->fraction; 
}

static PyObject* PhysicsWorld_shapecast(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    int shape_type;
    float px, py, pz;
    float rx, ry, rz, rw;
    float dx, dy, dz;
    PyObject* py_size = NULL;
    uint64_t ignore_h = 0; 
    static char *kwlist[] = {"shape", "pos", "rot", "dir", "size", "ignore", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i(fff)(ffff)(fff)O|K", kwlist, 
                                     &shape_type, &px, &py, &pz, 
                                     &rx, &ry, &rz, &rw, 
                                     &dx, &dy, &dz, &py_size, &ignore_h)) {
        return NULL;
    }

    PyObject* result = NULL;
    bool query_active = false;
    JPH_BodyID ignore_bid = 0;
    JPH_Shape* shape = NULL;

    // --- 1. VALIDATE DIRECTION ---
    float mag_sq = dx*dx + dy*dy + dz*dz;
    if (mag_sq < 1e-9f) goto exit; 

    // --- 2. PREPARE SHAPE & TRANSFORM ---
    float s[4] = {0, 0, 0, 0};
    if (py_size && PyTuple_Check(py_size)) {
        Py_ssize_t sz_len = PyTuple_Size(py_size);
        for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
            PyObject *item = PyTuple_GetItem(py_size, i);
            if (PyNumber_Check(item)) s[i] = (float)PyFloat_AsDouble(item);
        }
    } else if (py_size && PyNumber_Check(py_size)) {
        s[0] = (float)PyFloat_AsDouble(py_size);
    }

    // --- 3. PHASE GUARD & RESOURCE RESOLUTION ---
    SHADOW_LOCK(&self->shadow_lock);
    if (self->is_stepping) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot shapecast while physics is stepping");
        return NULL;
    }

    shape = find_or_create_shape(self, shape_type, s);
    if (!shape) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Invalid shape parameters");
        return NULL;
    }

    atomic_fetch_add(&self->active_queries, 1);
    query_active = true;

    if (ignore_h != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, ignore_h, &ignore_slot)) {
            ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 4. EXECUTE JOLT QUERY (SERIALIZED VTABLE) ---
    JPH_STACK_ALLOC(JPH_RMat4, transform);
    JPH_Quat q = {rx, ry, rz, rw};
    JPH_RVec3 p = {(double)px, (double)py, (double)pz};
    JPH_RMat4_RotationTranslation(transform, &q, &p);
    
    JPH_Vec3 sweep_dir = {dx, dy, dz};

    JPH_STACK_ALLOC(JPH_ShapeCastSettings, settings);
    JPH_ShapeCastSettings_Init(settings);
    settings->backFaceModeTriangles = JPH_BackFaceMode_IgnoreBackFaces;
    settings->backFaceModeConvex = JPH_BackFaceMode_IgnoreBackFaces;

    // --- SETUP FILTERS (The Missing Piece) ---
    JPH_BroadPhaseLayerFilter_Procs bp_procs = { .ShouldCollide = filter_allow_all_bp };
    JPH_BroadPhaseLayerFilter* bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
    JPH_BroadPhaseLayerFilter_SetProcs(&bp_procs);

    JPH_ObjectLayerFilter_Procs obj_procs = { .ShouldCollide = filter_allow_all_obj };
    JPH_ObjectLayerFilter* obj_filter = JPH_ObjectLayerFilter_Create(NULL);
    JPH_ObjectLayerFilter_SetProcs(&obj_procs);

    CastShapeFilter filter_ctx = { .ignore_id = ignore_bid };
    JPH_BodyFilter_Procs filter_procs = { .ShouldCollide = CastShape_BodyFilter };
    JPH_BodyFilter* body_filter = JPH_BodyFilter_Create(&filter_ctx);

    SHADOW_LOCK(&g_jph_trampoline_lock);
    JPH_BodyFilter_SetProcs(&filter_procs);

    CastShapeContext ctx = { .has_hit = false };
    ctx.hit.fraction = 1.0f;

    JPH_RVec3 base_offset = {0, 0, 0};
    const JPH_NarrowPhaseQuery* nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    
    // Pass bp_filter and obj_filter
    JPH_NarrowPhaseQuery_CastShape(nq, shape, transform, &sweep_dir, settings, &base_offset, 
                                   CastShape_ClosestCollector, &ctx, 
                                   bp_filter, obj_filter, body_filter, NULL);

    JPH_BodyFilter_SetProcs(&global_bf_procs);
    SHADOW_UNLOCK(&g_jph_trampoline_lock);

    JPH_BodyFilter_Destroy(body_filter);
    JPH_BroadPhaseLayerFilter_Destroy(bp_filter);
    JPH_ObjectLayerFilter_Destroy(obj_filter);

    if (!ctx.has_hit) goto exit;

    // --- 5. RESOLVE IDENTITY & GEOMETRY ---
    float nx = -ctx.hit.penetrationAxis.x;
    float ny = -ctx.hit.penetrationAxis.y;
    float nz = -ctx.hit.penetrationAxis.z;
    float n_len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (n_len > 1e-6f) {
        float inv = 1.0f / n_len;
        nx *= inv; ny *= inv; nz *= inv;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BodyHandle handle = (BodyHandle)JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
    uint32_t slot = (uint32_t)(handle & 0xFFFFFFFF);
    uint32_t gen  = (uint32_t)(handle >> 32);

    if (slot < self->slot_capacity && 
        self->generations[slot] == gen && 
        self->slot_states[slot] == SLOT_ALIVE) 
    {
        result = Py_BuildValue("Kf(fff)(fff)", 
            handle, 
            ctx.hit.fraction, 
            (float)ctx.hit.contactPointOn2.x, 
            (float)ctx.hit.contactPointOn2.y, 
            (float)ctx.hit.contactPointOn2.z,
            nx, ny, nz
        );
    }
    SHADOW_UNLOCK(&self->shadow_lock);

exit:
    if (query_active) {
        atomic_fetch_sub(&self->active_queries, 1);
    }

    if (result) return result;
    Py_RETURN_NONE;
}

// Helper to grow queue
static bool ensure_command_capacity(PhysicsWorldObject *self) {
  if (self->command_count >= self->command_capacity) {
    // Defensive: handle zero or uninitialized capacity
    size_t new_cap = (self->command_capacity == 0) ? 64 : self->command_capacity * 2;
    
    // Safety check: Prevent overflow on extreme counts
    if (new_cap > (SIZE_MAX / sizeof(PhysicsCommand))) {
        return false;
    }

    void *new_ptr = PyMem_RawRealloc(self->command_queue, new_cap * sizeof(PhysicsCommand));
    if (!new_ptr) {
      return false;
    }
    
    self->command_queue = (PhysicsCommand *)new_ptr;
    self->command_capacity = new_cap;
  }
  return true;
}

static void flush_commands(PhysicsWorldObject *self) {
  if (self->command_count == 0) return;

  JPH_BodyInterface *bi = self->body_interface;

  for (size_t i = 0; i < self->command_count; i++) {
    PhysicsCommand *cmd = &self->command_queue[i];
    uint32_t slot = cmd->slot;

    uint32_t dense_idx = 0;
    JPH_BodyID bid = 0;
    if (cmd->type != CMD_CREATE_BODY) {
      dense_idx = self->slot_to_dense[slot];
      bid = self->body_ids[dense_idx];
    }

    switch (cmd->type) {
    case CMD_CREATE_BODY: {
      JPH_BodyCreationSettings *s = cmd->data.create.settings;
      JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);

      // Handle Jolt spawn failure
      if (new_bid == JPH_INVALID_BODY_ID) {
          self->slot_states[slot] = SLOT_EMPTY;
          self->generations[slot]++;
          self->free_slots[self->free_count++] = slot;
          JPH_BodyCreationSettings_Destroy(s);
          continue; 
      }

      size_t new_dense = self->count;
      self->body_ids[new_dense] = new_bid;
      self->slot_to_dense[slot] = (uint32_t)new_dense;
      self->dense_to_slot[new_dense] = slot;
      self->user_data[new_dense] = cmd->data.create.user_data;

      JPH_STACK_ALLOC(JPH_RVec3, p);
      JPH_STACK_ALLOC(JPH_Quat, q);
      JPH_BodyInterface_GetPosition(bi, new_bid, p);
      JPH_BodyInterface_GetRotation(bi, new_bid, q);

      float fx = (float)p->x, fy = (float)p->y, fz = (float)p->z;
      self->positions[new_dense * 4 + 0] = fx;
      self->positions[new_dense * 4 + 1] = fy;
      self->positions[new_dense * 4 + 2] = fz;
      // Sync prev to prevent frame-0 jitter
      self->prev_positions[new_dense * 4 + 0] = fx;
      self->prev_positions[new_dense * 4 + 1] = fy;
      self->prev_positions[new_dense * 4 + 2] = fz;

      self->rotations[new_dense * 4 + 0] = q->x;
      self->rotations[new_dense * 4 + 1] = q->y;
      self->rotations[new_dense * 4 + 2] = q->z;
      self->rotations[new_dense * 4 + 3] = q->w;
      memcpy(&self->prev_rotations[new_dense * 4], &self->rotations[new_dense * 4], 16);

      self->count++;
      self->slot_states[slot] = SLOT_ALIVE;
      JPH_BodyCreationSettings_Destroy(s);
      break;
    }

    case CMD_DESTROY_BODY: {
      JPH_BodyInterface_RemoveBody(bi, bid);
      JPH_BodyInterface_DestroyBody(bi, bid);

      size_t last_dense = self->count - 1;
      if (dense_idx != last_dense) {
        // Correct Swap-and-Pop: Move all shadow data
        memcpy(&self->positions[dense_idx * 4], &self->positions[last_dense * 4], 16);
        memcpy(&self->rotations[dense_idx * 4], &self->rotations[last_dense * 4], 16);
        memcpy(&self->prev_positions[dense_idx * 4], &self->prev_positions[last_dense * 4], 16);
        memcpy(&self->prev_rotations[dense_idx * 4], &self->prev_rotations[last_dense * 4], 16);
        memcpy(&self->linear_velocities[dense_idx * 4], &self->linear_velocities[last_dense * 4], 16);
        memcpy(&self->angular_velocities[dense_idx * 4], &self->angular_velocities[last_dense * 4], 16);
        
        self->body_ids[dense_idx] = self->body_ids[last_dense];
        self->user_data[dense_idx] = self->user_data[last_dense];

        uint32_t mover_slot = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = (uint32_t)dense_idx;
        self->dense_to_slot[dense_idx] = mover_slot;
      }

      self->generations[slot]++; 
      self->free_slots[self->free_count++] = slot;
      self->slot_states[slot] = SLOT_EMPTY;
      self->count--;
      break;
    }

    case CMD_SET_POS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->data.vec.x; p->y = cmd->data.vec.y; p->z = cmd->data.vec.z;
      JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);
      self->positions[dense_idx * 4 + 0] = cmd->data.vec.x;
      self->positions[dense_idx * 4 + 1] = cmd->data.vec.y;
      self->positions[dense_idx * 4 + 2] = cmd->data.vec.z;
      break;
    }

    case CMD_SET_ROT: {
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->data.vec.x; q->y = cmd->data.vec.y; q->z = cmd->data.vec.z; q->w = cmd->data.vec.w;
      JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);
      memcpy(&self->rotations[dense_idx * 4], &cmd->data.vec, 16);
      break;
    }

    case CMD_SET_TRNS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->data.transform.px; p->y = cmd->data.transform.py; p->z = cmd->data.transform.pz;
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->data.transform.rx; q->y = cmd->data.transform.ry; q->z = cmd->data.transform.rz; q->w = cmd->data.transform.rw;

      JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);

      self->positions[dense_idx * 4 + 0] = (float)p->x;
      self->positions[dense_idx * 4 + 1] = (float)p->y;
      self->positions[dense_idx * 4 + 2] = (float)p->z;
      memcpy(&self->rotations[dense_idx * 4], &cmd->data.transform.rx, 16);
      break;
    }

    case CMD_SET_MOTION: {
      JPH_BodyInterface_SetMotionType(bi, bid, (JPH_MotionType)cmd->data.motion_type, JPH_Activation_Activate);
      // Logic Note: User should manually update layer if switching between Static (0) and Moving (1)
      break;
    }

    case CMD_ACTIVATE:   JPH_BodyInterface_ActivateBody(bi, bid); break;
    case CMD_DEACTIVATE: JPH_BodyInterface_DeactivateBody(bi, bid); break;

    case CMD_SET_USER_DATA: {
      self->user_data[dense_idx] = cmd->data.user_data_val;
      break;
    }
    default: break;
    }
  }

  self->command_count = 0;
  self->view_shape[0] = (Py_ssize_t)self->count;
}

// Constraints
// --- 1. Unified Parameter Struct ---
typedef struct {
    float px, py, pz;       // Pivot / Point
    float ax, ay, az;       // Axis (Hinge/Slider/Cone)
    float limit_min;        // Min Angle or Min Distance
    float limit_max;        // Max Angle or Max Distance
    float half_cone_angle;  // Cone specific
} ConstraintParams;

// Initialize defaults to avoid garbage data
static void params_init(ConstraintParams* p) {
    p->px = 0; p->py = 0; p->pz = 0;
    p->ax = 0; p->ay = 1; p->az = 0; // Default Up axis
    p->limit_min = -FLT_MAX;
    p->limit_max = FLT_MAX;
    p->half_cone_angle = 0.0f;
}

// --- 2. Python Parsers ---

static int parse_point_params(PyObject* args, ConstraintParams* p) {
    if (!args || args == Py_None) return 1; // Use defaults (0,0,0)
    return PyArg_ParseTuple(args, "fff", &p->px, &p->py, &p->pz);
}

static int parse_hinge_params(PyObject* args, ConstraintParams* p) {
    p->limit_min = -JPH_M_PI; p->limit_max = JPH_M_PI; // Hinge defaults
    if (!args) return 1;
    // (Pivot), (Axis), [Min, Max]
    return PyArg_ParseTuple(args, "(fff)(fff)|ff", 
        &p->px, &p->py, &p->pz, 
        &p->ax, &p->ay, &p->az, 
        &p->limit_min, &p->limit_max);
}

static int parse_slider_params(PyObject* args, ConstraintParams* p) {
    // Slider axis defaults to X usually, but Y is fine. Limits default to free.
    if (!args) return 1;
    return PyArg_ParseTuple(args, "(fff)(fff)|ff", 
        &p->px, &p->py, &p->pz, 
        &p->ax, &p->ay, &p->az, 
        &p->limit_min, &p->limit_max);
}

static int parse_cone_params(PyObject* args, ConstraintParams* p) {
    if (!args) return 1;
    // (Pivot), (TwistAxis), HalfAngle
    return PyArg_ParseTuple(args, "(fff)(fff)f", 
        &p->px, &p->py, &p->pz, 
        &p->ax, &p->ay, &p->az, 
        &p->half_cone_angle);
}

static int parse_distance_params(PyObject* args, ConstraintParams* p) {
    p->limit_min = 0.0f; p->limit_max = 10.0f;
    if (!args) return 1;
    // Min, Max
    return PyArg_ParseTuple(args, "ff", &p->limit_min, &p->limit_max);
}

// --- 3. Jolt Creator Helpers ---

static JPH_Constraint* create_fixed(const ConstraintParams* p, JPH_Body* b1, JPH_Body* b2) {
    JPH_FixedConstraintSettings s;
    JPH_FixedConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.autoDetectPoint = true; 
    return (JPH_Constraint*)JPH_FixedConstraint_Create(&s, b1, b2);
}

static JPH_Constraint* create_point(const ConstraintParams* p, JPH_Body* b1, JPH_Body* b2) {
    JPH_PointConstraintSettings s;
    JPH_PointConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space = JPH_ConstraintSpace_WorldSpace;
    s.point1.x = p->px; s.point1.y = p->py; s.point1.z = p->pz;
    s.point2 = s.point1;
    return (JPH_Constraint*)JPH_PointConstraint_Create(&s, b1, b2);
}

static JPH_Constraint* create_hinge(const ConstraintParams* p, JPH_Body* b1, JPH_Body* b2) {
    JPH_HingeConstraintSettings s;
    JPH_HingeConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space = JPH_ConstraintSpace_WorldSpace;
    
    s.point1.x = p->px; s.point1.y = p->py; s.point1.z = p->pz;
    s.point2 = s.point1;

    JPH_Vec3 axis = {p->ax, p->ay, p->az};
    float len_sq = axis.x*axis.x + axis.y*axis.y + axis.z*axis.z;
    
    // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
    if (len_sq < 1e-9f) {
        axis.x = 0.0f; axis.y = 1.0f; axis.z = 0.0f;
    } else {
        JPH_Vec3_Normalize(&axis, &axis);
    }
    
    JPH_Vec3 norm;
    vec3_get_perpendicular(&axis, &norm);

    s.hingeAxis1 = axis; s.hingeAxis2 = axis;
    s.normalAxis1 = norm; s.normalAxis2 = norm;
    s.limitsMin = p->limit_min;
    s.limitsMax = p->limit_max;
    
    return (JPH_Constraint*)JPH_HingeConstraint_Create(&s, b1, b2);
}

static JPH_Constraint* create_slider(const ConstraintParams* p, JPH_Body* b1, JPH_Body* b2) {
    JPH_SliderConstraintSettings s;
    JPH_SliderConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space = JPH_ConstraintSpace_WorldSpace;
    s.autoDetectPoint = false;

    s.point1.x = p->px; s.point1.y = p->py; s.point1.z = p->pz;
    s.point2 = s.point1;

    JPH_Vec3 axis = {p->ax, p->ay, p->az};
    float len_sq = axis.x*axis.x + axis.y*axis.y + axis.z*axis.z;
    
    // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
    if (len_sq < 1e-9f) {
        axis.x = 0.0f; axis.y = 1.0f; axis.z = 0.0f;
    } else {
        JPH_Vec3_Normalize(&axis, &axis);
    }
    
    JPH_Vec3 norm;
    vec3_get_perpendicular(&axis, &norm);

    s.sliderAxis1 = axis; s.sliderAxis2 = axis;
    s.normalAxis1 = norm; s.normalAxis2 = norm;
    s.limitsMin = p->limit_min;
    s.limitsMax = p->limit_max;

    return (JPH_Constraint*)JPH_SliderConstraint_Create(&s, b1, b2);
}

static JPH_Constraint* create_cone(const ConstraintParams* p, JPH_Body* b1, JPH_Body* b2) {
    JPH_ConeConstraintSettings s;
    JPH_ConeConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space = JPH_ConstraintSpace_WorldSpace;

    s.point1.x = p->px; s.point1.y = p->py; s.point1.z = p->pz;
    s.point2 = s.point1;

    JPH_Vec3 axis = {p->ax, p->ay, p->az};
    float len_sq = axis.x*axis.x + axis.y*axis.y + axis.z*axis.z;
    
    // SAFETY: If axis is zero, default to "UP" to prevent NaN explosion
    if (len_sq < 1e-9f) {
        axis.x = 0.0f; axis.y = 1.0f; axis.z = 0.0f;
    } else {
        JPH_Vec3_Normalize(&axis, &axis);
    }

    s.twistAxis1 = axis; 
    s.twistAxis2 = axis;
    s.halfConeAngle = p->half_cone_angle;

    return (JPH_Constraint*)JPH_ConeConstraint_Create(&s, b1, b2);
}

static JPH_Constraint* create_distance(const ConstraintParams* p, JPH_Body* b1, JPH_Body* b2) {
    JPH_DistanceConstraintSettings s;
    JPH_DistanceConstraintSettings_Init(&s);
    s.base.enabled = true;
    s.space = JPH_ConstraintSpace_WorldSpace;
    
    // Check if the user provided a specific pivot point
    if (fabsf(p->px) > 1e-6f || fabsf(p->py) > 1e-6f || fabsf(p->pz) > 1e-6f) {
        s.point1.x = p->px; s.point1.y = p->py; s.point1.z = p->pz;
        s.point2 = s.point1;
    } else {
        // Fallback: Default to current body centers if no pivot was provided
        JPH_Body_GetPosition(b1, &s.point1);
        JPH_Body_GetPosition(b2, &s.point2);
    }
    
    s.minDistance = p->limit_min;
    s.maxDistance = p->limit_max;

    return (JPH_Constraint*)JPH_DistanceConstraint_Create(&s, b1, b2);
}

static int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
    if (self->view_export_count > 0) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize world while memory views are exported.");
        return -1;
    }
    // We check this inside the SHADOW_LOCK (which the caller must hold).
    // Because raycast/shapecast increment this counter WHILE holding the lock,
    // we are guaranteed that if this is 0, no new queries can start 
    // until we release the lock.
    if (atomic_load_explicit(&self->active_queries, memory_order_relaxed) > 0) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot resize world while queries (raycast/shapecast) are active.");
        return -1;
    }
    if (new_capacity <= self->capacity) return 0;

    // 1. Allocate all new buffers to TEMPORARY pointers first
    void* p_pos = PyMem_RawRealloc(self->positions, new_capacity * 16);
    void* p_rot = PyMem_RawRealloc(self->rotations, new_capacity * 16);
    void* p_ppos = PyMem_RawRealloc(self->prev_positions, new_capacity * 16);
    void* p_prot = PyMem_RawRealloc(self->prev_rotations, new_capacity * 16);
    void* p_lvel = PyMem_RawRealloc(self->linear_velocities, new_capacity * 16);
    void* p_avel = PyMem_RawRealloc(self->angular_velocities, new_capacity * 16);
    void* p_bids = PyMem_RawRealloc(self->body_ids, new_capacity * sizeof(JPH_BodyID));
    void* p_udat = PyMem_RawRealloc(self->user_data, new_capacity * sizeof(uint64_t));
    void* p_gens = PyMem_RawRealloc(self->generations, new_capacity * sizeof(uint32_t));
    void* p_s2d  = PyMem_RawRealloc(self->slot_to_dense, new_capacity * sizeof(uint32_t));
    void* p_d2s  = PyMem_RawRealloc(self->dense_to_slot, new_capacity * sizeof(uint32_t));
    void* p_stat = PyMem_RawRealloc(self->slot_states, new_capacity * sizeof(uint8_t));
    void* p_free = PyMem_RawRealloc(self->free_slots, new_capacity * sizeof(uint32_t));

    // 2. Check if ANY allocation failed
    if (!p_pos || !p_rot || !p_ppos || !p_prot || !p_lvel || !p_avel || 
        !p_bids || !p_udat || !p_gens || !p_s2d || !p_d2s || !p_stat || !p_free) {
        // Note: PyMem_RawRealloc does not free the original pointer on failure.
        // We just return -1; the struct still points to the old (valid) memory.
        PyErr_NoMemory();
        return -1;
    }

    // 3. COMMIT all pointers at once
    self->positions = p_pos; self->rotations = p_rot;
    self->prev_positions = p_ppos; self->prev_rotations = p_prot;
    self->linear_velocities = p_lvel; self->angular_velocities = p_avel;
    self->body_ids = p_bids; self->user_data = p_udat;
    self->generations = p_gens; self->slot_to_dense = p_s2d;
    self->dense_to_slot = p_d2s; self->slot_states = p_stat;
    self->free_slots = p_free;

    // 4. Initialize new slots
    for (size_t i = self->slot_capacity; i < new_capacity; i++) {
        self->generations[i] = 1;
        self->slot_states[i] = SLOT_EMPTY;
        self->free_slots[self->free_count++] = (uint32_t)i;
    }

    self->capacity = new_capacity;
    self->slot_capacity = new_capacity;
    return 0;
}

static PyObject* PhysicsWorld_create_constraint(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    int type;
    uint64_t h1, h2;
    PyObject* params = NULL;
    static char *kwlist[] = {"type", "body1", "body2", "params", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iKK|O", kwlist, &type, &h1, &h2, &params)) {
        return NULL;
    }

    // NEW: Explicitly forbid self-constraints (Jolt requires two distinct bodies)
    if (h1 == h2) {
        PyErr_SetString(PyExc_ValueError, "Cannot create a constraint between a body and itself");
        return NULL;
    }

    ConstraintParams p;
    params_init(&p);
    // ... (parsing logic same as before) ...
    int parse_ok = 1;
    switch (type) {
        case CONSTRAINT_FIXED:    break;
        case CONSTRAINT_POINT:    parse_ok = parse_point_params(params, &p); break;
        case CONSTRAINT_HINGE:    parse_ok = parse_hinge_params(params, &p); break;
        case CONSTRAINT_SLIDER:   parse_ok = parse_slider_params(params, &p); break;
        case CONSTRAINT_CONE:     parse_ok = parse_cone_params(params, &p); break;
        case CONSTRAINT_DISTANCE: parse_ok = parse_distance_params(params, &p); break;
        default: PyErr_SetString(PyExc_ValueError, "Unknown constraint type"); return NULL;
    }
    if (!parse_ok) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    if (self->is_stepping || atomic_load(&self->active_queries) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot create constraint while world is busy");
        return NULL;
    }

    uint32_t s1, s2;
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

    const JPH_BodyLockInterface* lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock1, lock2;
    
    // Sort for Deadlock Prevention
    if (bid1 < bid2) {
        JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
        JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
    } else {
        JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);
        JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
    }

    JPH_Constraint* constraint = NULL;
    if (lock1.body && lock2.body) {
        // Resolve pointers (Safe because h1 != h2 guaranteed earlier)
        JPH_Body *b1 = (JPH_Body_GetID(lock1.body) == bid1) ? lock1.body : lock2.body;
        JPH_Body *b2 = (JPH_Body_GetID(lock1.body) == bid2) ? lock1.body : lock2.body;

        switch (type) {
            case CONSTRAINT_FIXED:    constraint = create_fixed(&p, b1, b2); break;
            case CONSTRAINT_POINT:    constraint = create_point(&p, b1, b2); break;
            case CONSTRAINT_HINGE:    constraint = create_hinge(&p, b1, b2); break;
            case CONSTRAINT_SLIDER:   constraint = create_slider(&p, b1, b2); break;
            case CONSTRAINT_CONE:     constraint = create_cone(&p, b1, b2); break;
            case CONSTRAINT_DISTANCE: constraint = create_distance(&p, b1, b2); break;
            default: break; // Already handled above
        }
    }

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock1);
    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock2);

    if (!constraint) {
        SHADOW_LOCK(&self->shadow_lock);
        self->free_constraint_slots[self->free_constraint_count++] = c_slot;
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Jolt failed to create constraint instance");
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

static PyObject* PhysicsWorld_destroy_constraint(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t h = 0;
    static char *kwlist[] = {"handle", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
        return NULL;
    }

    JPH_Constraint* c_to_destroy = NULL;

    // --- 1. RESOLUTION PHASE (Inside Shadow Lock) ---
    SHADOW_LOCK(&self->shadow_lock);

    // Guard against both Physics Step AND active Queries
    if (self->is_stepping || atomic_load(&self->active_queries) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot destroy constraint while world is busy (stepping or querying)");
        return NULL;
    }

    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    // Validate identity
    if (slot >= self->constraint_capacity || 
        self->constraint_generations[slot] != gen || 
        self->constraint_states[slot] != SLOT_ALIVE) 
    {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale constraint handle");
        return NULL;
    }

    // Capture the pointer and IMMEDIATELY invalidate the slot
    c_to_destroy = self->constraints[slot];
    
    self->constraints[slot] = NULL;
    self->constraint_states[slot] = SLOT_EMPTY;
    self->constraint_generations[slot]++; // Increment generation to invalidate stale handles
    self->free_constraint_slots[self->free_constraint_count++] = slot;

    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 2. JOLT DESTRUCTION PHASE (Outside Shadow Lock) ---
    // No Shadow-vs-Jolt deadlocks possible here!
    if (c_to_destroy) {
        // Automatic Body Wake-up
        // This is a "nice to have" - prevents objects from hanging in the air 
        // when the joint holding them is deleted.
        if (JPH_Constraint_GetType(c_to_destroy) == JPH_ConstraintType_TwoBodyConstraint) {
            JPH_TwoBodyConstraint* tbc = (JPH_TwoBodyConstraint*)c_to_destroy;
            JPH_Body* b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
            JPH_Body* b2 = JPH_TwoBodyConstraint_GetBody2(tbc);
            
            // JPH_BodyInterface_ActivateBody is thread-safe
            if (b1) JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
            if (b2) JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
        }

        // Remove and Destroy
        JPH_PhysicsSystem_RemoveConstraint(self->system, c_to_destroy);
        JPH_Constraint_Destroy(c_to_destroy);
    }

    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_save_state(PhysicsWorldObject *self, PyObject *Py_UNUSED(unused)) {
    SHADOW_LOCK(&self->shadow_lock);
    
    if (self->is_stepping || atomic_load(&self->active_queries) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot save state while world is busy");
        return NULL;
    }

    // 1. Unambiguous Size Calculation
    size_t header_size = sizeof(size_t) /* count */ + 
                         sizeof(double) /* time */ + 
                         sizeof(size_t) /* slot_capacity */;

    size_t dense_stride = 4 * sizeof(float); // 16 bytes
    size_t dense_size = self->count * 4 /* arrays */ * dense_stride; 

    size_t mapping_size = self->slot_capacity * (
        sizeof(uint32_t) * 3 /* gen, s2d, d2s */ + 
        sizeof(uint8_t)  /* states */
    );

    size_t total_size = header_size + dense_size + mapping_size;
    PyObject *bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_size);
    if (!bytes) { SHADOW_UNLOCK(&self->shadow_lock); return NULL; }
    
    char *ptr = PyBytes_AsString(bytes);

    // 2. Encode Header
    memcpy(ptr, &self->count, sizeof(size_t));         ptr += sizeof(size_t);
    memcpy(ptr, &self->time, sizeof(double));          ptr += sizeof(double);
    memcpy(ptr, &self->slot_capacity, sizeof(size_t)); ptr += sizeof(size_t);

    // 3. Encode Dense Buffers
    memcpy(ptr, self->positions, self->count * dense_stride);          ptr += self->count * dense_stride;
    memcpy(ptr, self->rotations, self->count * dense_stride);          ptr += self->count * dense_stride;
    memcpy(ptr, self->linear_velocities, self->count * dense_stride);  ptr += self->count * dense_stride;
    memcpy(ptr, self->angular_velocities, self->count * dense_stride); ptr += self->count * dense_stride;

    // 4. Encode Mapping Tables
    memcpy(ptr, self->generations, self->slot_capacity * sizeof(uint32_t));   ptr += self->slot_capacity * sizeof(uint32_t);
    memcpy(ptr, self->slot_to_dense, self->slot_capacity * sizeof(uint32_t)); ptr += self->slot_capacity * sizeof(uint32_t);
    memcpy(ptr, self->dense_to_slot, self->slot_capacity * sizeof(uint32_t)); ptr += self->slot_capacity * sizeof(uint32_t);
    memcpy(ptr, self->slot_states, self->slot_capacity);

    SHADOW_UNLOCK(&self->shadow_lock);
    return bytes;
}

static PyObject *PhysicsWorld_load_state(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    Py_buffer view;
    static char *kwlist[] = {"state", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*", kwlist, &view)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);

    // 1. CONCURRENCY GUARD
    if (self->is_stepping || atomic_load(&self->active_queries) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_RuntimeError, "Cannot load state while world is busy");
        return NULL;
    }

    // 2. HEADER VALIDATION
    if ((size_t)view.len < (sizeof(size_t) * 2 + sizeof(double))) goto size_fail;

    char *ptr = (char *)view.buf;
    size_t saved_count, saved_slot_cap;
    double saved_time;

    memcpy(&saved_count, ptr, sizeof(size_t));    ptr += sizeof(size_t);
    memcpy(&saved_time, ptr, sizeof(double));     ptr += sizeof(double);
    memcpy(&saved_slot_cap, ptr, sizeof(size_t)); ptr += sizeof(size_t);

    // CRITICAL: Prevent memory corruption by verifying slot capacity matches exactly
    if (saved_slot_cap != self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyBuffer_Release(&view);
        PyErr_Format(PyExc_ValueError, "Capacity mismatch: World is %zu, Snapshot is %zu", 
                     self->slot_capacity, saved_slot_cap);
        return NULL;
    }

    // 3. FULL SIZE VALIDATION
    size_t dense_stride = 16;
    size_t expected = (sizeof(size_t) * 2 + sizeof(double)) + 
                      (saved_count * 4 * dense_stride) + 
                      (saved_slot_cap * (4 * 3 + 1));

    if ((size_t)view.len != expected) goto size_fail;

    // 4. RESTORE SHADOW STATE
    self->count = saved_count;
    self->time = saved_time;
    self->view_shape[0] = (Py_ssize_t)self->count;

    memcpy(self->positions, ptr, self->count * dense_stride);          ptr += self->count * dense_stride;
    memcpy(self->rotations, ptr, self->count * dense_stride);          ptr += self->count * dense_stride;
    memcpy(self->linear_velocities, ptr, self->count * dense_stride);  ptr += self->count * dense_stride;
    memcpy(self->angular_velocities, ptr, self->count * dense_stride); ptr += self->count * dense_stride;

    memcpy(self->generations, ptr, self->slot_capacity * 4);   ptr += self->slot_capacity * 4;
    memcpy(self->slot_to_dense, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(self->dense_to_slot, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
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
        if (bid == JPH_INVALID_BODY_ID) continue;

        JPH_STACK_ALLOC(JPH_RVec3, p);
        p->x = (double)self->positions[i * 4]; p->y = (double)self->positions[i * 4+1]; p->z = (double)self->positions[i * 4+2];
        JPH_STACK_ALLOC(JPH_Quat, q);
        q->x = self->rotations[i * 4]; q->y = self->rotations[i * 4+1]; q->z = self->rotations[i * 4+2]; q->w = self->rotations[i * 4+3];

        JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);
        JPH_BodyInterface_SetLinearVelocity(bi, bid, (JPH_Vec3*)&self->linear_velocities[i*4]);
        JPH_BodyInterface_SetAngularVelocity(bi, bid, (JPH_Vec3*)&self->angular_velocities[i*4]);
        
        // Re-Sync UserData to the newly incremented generations
        BodyHandle new_h = make_handle(self->dense_to_slot[i], self->generations[self->dense_to_slot[i]]);
        JPH_BodyInterface_SetUserData(bi, bid, (uint64_t)new_h);
    }

    PyBuffer_Release(&view);
    Py_RETURN_NONE;

size_fail:
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError, "Snapshot buffer truncated or capacity mismatch");
    return NULL;
}

static PyObject* PhysicsWorld_step(PhysicsWorldObject* self, PyObject* args) {
    float dt = 1.0f/60.0f;
    if (!PyArg_ParseTuple(args, "|f", &dt)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);

    // 1. RE-ENTRANCY GUARD
    if (self->is_stepping) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Concurrent step detected");
        return NULL;
    }
    // NEW: Wait for active raycasts/shapecasts to finish
    // This prevents us from deleting bodies while a raycast is looking at them.
    if (atomic_load(&self->active_queries) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot step while queries (raycast/shapecast) are active");
        return NULL;
    }
    self->is_stepping = true;

    // 2. BUFFER MANAGEMENT (Reset Phase)
    // Safe because is_stepping=true ensures no other thread is inside step(),
    // and Jolt is not running yet, so no callbacks are firing.
    if (!self->contact_buffer) {
        self->contact_max_capacity = 4096; // Fixed size batch buffer
        self->contact_buffer = PyMem_RawMalloc(self->contact_max_capacity * sizeof(ContactEvent));
        if (!self->contact_buffer) {
          self->is_stepping = false;
          SHADOW_UNLOCK(&self->shadow_lock);
          return PyErr_NoMemory();
        }
    }
    // RESET the atomic index for the new frame
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

    // 3. FLUSH COMMANDS
    // This updates Jolt bodies and sets their UserData handles.
    flush_commands(self);

    // Snapshot state for interpolation
    memcpy(self->prev_positions, self->positions, self->count * 16);
    memcpy(self->prev_rotations, self->rotations, self->count * 16);

    SHADOW_UNLOCK(&self->shadow_lock);

    // 4. JOLT UPDATE (The Producer Phase)
    // The main thread sleeps here. Jolt worker threads run and fire on_contact_added.
    Py_BEGIN_ALLOW_THREADS
    JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);
    Py_END_ALLOW_THREADS
    
    // 5. ACQUIRE FENCE (The Consumer Phase)
    // Ensure we see all data written by the worker threads before we read the index.
    // While allow_threads implies a barrier, explicit acquire pairs with the callback's release.
    atomic_thread_fence(memory_order_acquire);

    SHADOW_LOCK(&self->shadow_lock);
    
    // 6. SYNC SHADOW BUFFERS
    culverin_sync_shadow_buffers(self);

    // 7. FINALIZE COUNT
    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    if (count > self->contact_max_capacity) {
        count = self->contact_max_capacity;
    }
    self->contact_count = count; // Publish to Python-facing field

    self->is_stepping = false;
    self->time += (double)dt;
    
    SHADOW_UNLOCK(&self->shadow_lock);
    
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_create_character(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  float px = 0, py = 0, pz = 0, height = 1.8f, radius = 0.4f, step_height = 0.4f, max_slope = 45.0f;
  static char *kwlist[] = {"pos", "height", "radius", "step_height", "max_slope", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)|ffff", kwlist, &px, &py, &pz, 
                                   &height, &radius, &step_height, &max_slope)) {
    return NULL;
  }

  // Locals for cleanup
  JPH_CharacterVirtual *j_char = NULL;
  CharacterObject *obj = NULL;
  uint32_t char_slot = 0xFFFFFFFF;
  bool slot_reserved = false;

  // --- 1. SLOT RESERVATION ---
  SHADOW_LOCK(&self->shadow_lock);
  if (self->is_stepping || atomic_load(&self->active_queries) > 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_RuntimeError, "Cannot create character while world is busy");
      return NULL;
  }

  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return NULL;
    }
  }
  char_slot = self->free_slots[--self->free_count];
  self->slot_states[char_slot] = SLOT_PENDING_CREATE;
  slot_reserved = true;
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT CHARACTER CREATION (Unlocked) ---
  float half_h = (height - 2.0f * radius) * 0.5f;
  if (half_h < 0.1f) half_h = 0.1f;

  JPH_CapsuleShapeSettings *ss = JPH_CapsuleShapeSettings_Create(half_h, radius);
  JPH_Shape *shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);
  if (!shape) goto fail;

  JPH_CharacterVirtualSettings settings;
  JPH_CharacterVirtualSettings_Init(&settings); 
  settings.base.shape = shape;
  settings.base.maxSlopeAngle = max_slope * (JPH_M_PI / 180.0f);
  
  JPH_STACK_ALLOC(JPH_RVec3, pos_aligned);
  pos_aligned->x = (double)px; pos_aligned->y = (double)py; pos_aligned->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot_aligned);
  rot_aligned->x = 0; rot_aligned->y = 0; rot_aligned->z = 0; rot_aligned->w = 1;

  // Use Layer 1 (Moving)
  j_char = JPH_CharacterVirtual_Create(&settings, pos_aligned, rot_aligned, 1, self->system);
  JPH_Shape_Destroy(shape);
  if (!j_char) goto fail;

  if (self->char_vs_char_manager) {
      JPH_CharacterVsCharacterCollisionSimple_AddCharacter(self->char_vs_char_manager, j_char);
      JPH_CharacterVirtual_SetCharacterVsCharacterCollision(j_char, self->char_vs_char_manager);
  }

  // --- 3. PYTHON WRAPPER ALLOCATION ---
  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);
  obj = (CharacterObject *)PyObject_GC_New(CharacterObject, (PyTypeObject *)st->CharacterType);
  if (!obj) goto fail;

  // Initialize fields (Atomics)
  atomic_store_explicit(&obj->push_strength, 200.0f, memory_order_relaxed);
  atomic_store_explicit(&obj->last_vx, 0.0f, memory_order_relaxed);
  atomic_store_explicit(&obj->last_vy, 0.0f, memory_order_relaxed);
  atomic_store_explicit(&obj->last_vz, 0.0f, memory_order_relaxed);
  
  obj->world = self;
  obj->character = j_char;
  obj->prev_px = px; obj->prev_py = py; obj->prev_pz = pz;
  obj->prev_rx = 0.0f; obj->prev_ry = 0.0f; obj->prev_rz = 0.0f; obj->prev_rw = 1.0f;
  obj->listener = NULL; obj->body_filter = NULL; obj->shape_filter = NULL;
  obj->bp_filter = NULL; obj->obj_filter = NULL;

  // --- 4. WORLD REGISTRATION (Atomic Commit) ---
  SHADOW_LOCK(&self->shadow_lock);
  
  uint32_t gen = self->generations[char_slot];
  BodyHandle handle = ((uint64_t)gen << 32) | (uint64_t)char_slot;
  obj->handle = handle;

  uint32_t dense_idx = (uint32_t)self->count;
  self->body_ids[dense_idx] = JPH_CharacterVirtual_GetInnerBodyID(j_char);
  self->slot_to_dense[char_slot] = dense_idx;
  self->dense_to_slot[dense_idx] = char_slot;
  self->slot_states[char_slot] = SLOT_ALIVE;
  self->user_data[dense_idx] = 0; 
  self->count++;
  self->view_shape[0] = (Py_ssize_t)self->count;

  // Stamp identity into Jolt LAST to ensure dense maps are ready if a callback fires
  JPH_BodyInterface_SetUserData(self->body_interface, self->body_ids[dense_idx], (uint64_t)handle);
  
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 5. GLOBAL PROC SERIALIZATION & FILTER INIT ---
  SHADOW_LOCK(&g_jph_trampoline_lock);
  
  JPH_CharacterContactListener_SetProcs(&char_listener_procs);
  obj->listener = JPH_CharacterContactListener_Create(obj);
  
  JPH_BodyFilter_SetProcs(&global_bf_procs);
  obj->body_filter = JPH_BodyFilter_Create(NULL);

  JPH_ShapeFilter_SetProcs(&global_sf_procs);
  obj->shape_filter = JPH_ShapeFilter_Create(NULL);

  SHADOW_UNLOCK(&g_jph_trampoline_lock);

  JPH_CharacterVirtual_SetListener(j_char, obj->listener);
  obj->bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  obj->obj_filter = JPH_ObjectLayerFilter_Create(NULL);

  Py_INCREF(self);
  PyObject_GC_Track((PyObject *)obj);
  return (PyObject *)obj;

fail:
  if (obj) {
      // If we got here, registration didn't happen, so we just DECREF and the 
      // destructor (Character_dealloc) will handle j_char/filters.
      Py_DECREF(obj);
  } else {
      if (j_char) JPH_CharacterBase_Destroy((JPH_CharacterBase *)j_char);
      if (slot_reserved) {
          SHADOW_LOCK(&self->shadow_lock);
          self->slot_states[char_slot] = SLOT_EMPTY;
          self->free_slots[self->free_count++] = char_slot;
          SHADOW_UNLOCK(&self->shadow_lock);
      }
  }
  return NULL;
}

static PyObject *PhysicsWorld_create_body(PhysicsWorldObject *self,
                                          PyObject *args, PyObject *kwds) {
  float px = 0.0f; float py = 0.0f; float pz = 0.0f;
  float rx = 0.0f; float ry = 0.0f; float rz = 0.0f; float rw = 1.0f;
  float s[4] = {1.0f, 1.0f, 1.0f, 0.0f}; 
  int shape_type = 0; int motion_type = 2;                   
  unsigned long long user_data = 0;
  int is_sensor = 0; 

  PyObject *py_size = NULL;
  static char *kwlist[] = {"pos", "rot", "size", "shape", "motion", "user_data", "is_sensor", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|(fff)(ffff)OiiKp", kwlist, 
        &px, &py, &pz, &rx, &ry, &rz, &rw, &py_size, &shape_type, &motion_type, &user_data, &is_sensor)) {
        return NULL;
    }

  if (py_size && PyTuple_Check(py_size)) {
    Py_ssize_t sz_len = PyTuple_Size(py_size);
    for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
      PyObject *item = PyTuple_GetItem(py_size, i);
      if (PyNumber_Check(item)) {
        s[i] = (float)PyFloat_AsDouble(item);
      }
    }
  }

  if (shape_type == 4 && motion_type != 0) { 
    PyErr_SetString(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  
  // 1. Thread Safety Guard
  GUARD_STEPPING(self);

  // 2. Resize Logic
  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_MemoryError, "Failed to resize PhysicsWorld");
      return NULL;
    }
  }

  uint32_t slot = self->free_slots[--self->free_count];
  JPH_Shape *shape = find_or_create_shape(self, shape_type, s);
  if (!shape) {
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_RuntimeError, "Failed to create shape.");
    return NULL;
  }

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)px; pos->y = (double)py; pos->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx; rot->y = ry; rot->z = rz; rot->w = rw;

  uint32_t layer = (motion_type == 0) ? 0 : 1;
  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, (JPH_MotionType)motion_type, (JPH_ObjectLayer)layer);
  if (is_sensor) JPH_BodyCreationSettings_SetIsSensor(settings, true);

  if (motion_type == 2) JPH_BodyCreationSettings_SetAllowSleeping(settings, true);

  // 3. CRITICAL: Embed the Full Handle (Slot + Gen) into Jolt UserData
  // This allows the lock-free callback to identify the body correctly.
  uint32_t gen = self->generations[slot];
  BodyHandle handle = make_handle(slot, gen);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  if (!ensure_command_capacity(self)) {
    JPH_BodyCreationSettings_Destroy(settings);
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->type = CMD_CREATE_BODY;
  cmd->slot = slot;
  cmd->data.create.settings = settings;
  cmd->data.create.user_data = (uint64_t)user_data; // This is the Python-level UserData
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  SHADOW_UNLOCK(&self->shadow_lock);

  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject *PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  // 1. SAFE INITIALIZATION
  Py_buffer v_view = {0}; 
  Py_buffer i_view = {0};
  float px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0, rw = 1.0f;
  unsigned long long user_data = 0;
  static char *kwlist[] = {"pos", "rot", "vertices", "indices", "user_data", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(ffff)y*y*|K", kwlist, &px,
                                   &py, &pz, &rx, &ry, &rz, &rw, &v_view,
                                   &i_view, &user_data)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  JPH_Shape *shape = NULL;
  uint32_t slot = 0xFFFFFFFF;
  bool slot_reserved = false;

  // 2. BUFFER ALIGNMENT VALIDATION
  if (v_view.len % (3 * sizeof(float)) != 0) {
      PyErr_SetString(PyExc_ValueError, "Vertex buffer size mismatch (must be 3*float32 per vertex)");
      goto cleanup;
  }
  if (i_view.len % (3 * sizeof(uint32_t)) != 0) {
      PyErr_SetString(PyExc_ValueError, "Index buffer size mismatch (must be 3*uint32 per triangle)");
      goto cleanup;
  }

  uint32_t vertex_count = (uint32_t)(v_view.len / (3 * sizeof(float)));
  uint32_t tri_count = (uint32_t)(i_view.len / (3 * sizeof(uint32_t)));

  // 3. JOLT MESH BUILDING (Triangle Copy + Index Bounds Check)
  JPH_IndexedTriangle *jolt_tris = (JPH_IndexedTriangle *)PyMem_RawMalloc(
      tri_count * sizeof(JPH_IndexedTriangle));
  if (!jolt_tris) {
    ret_val = PyErr_NoMemory();
    goto cleanup;
  }

  uint32_t *raw_indices = (uint32_t *)i_view.buf;
  for (uint32_t t = 0; t < tri_count; t++) {
    uint32_t i1 = raw_indices[t * 3 + 0];
    uint32_t i2 = raw_indices[t * 3 + 1];
    uint32_t i3 = raw_indices[t * 3 + 2];

    // CRITICAL: Prevent Jolt from reading past v_view.buf
    if (i1 >= vertex_count || i2 >= vertex_count || i3 >= vertex_count) {
        PyMem_RawFree(jolt_tris);
        PyErr_Format(PyExc_ValueError, "Mesh index out of range: %u/%u/%u >= %u", 
                     i1, i2, i3, vertex_count);
        goto cleanup;
    }

    jolt_tris[t].i1 = i1;
    jolt_tris[t].i2 = i2;
    jolt_tris[t].i3 = i3;
    jolt_tris[t].materialIndex = 0;
    jolt_tris[t].userData = 0;
  }

  JPH_MeshShapeSettings *mss = JPH_MeshShapeSettings_Create2(
      (JPH_Vec3 *)v_view.buf, vertex_count, jolt_tris, tri_count);
  PyMem_RawFree(jolt_tris);

  if (!mss) {
    PyErr_SetString(PyExc_RuntimeError, "Jolt MeshSettings allocation failed");
    goto cleanup;
  }

  shape = (JPH_Shape *)JPH_MeshShapeSettings_CreateShape(mss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)mss);

  if (!shape) {
    PyErr_SetString(PyExc_RuntimeError, "Jolt Mesh BVH build failed (Triangle data degenerate?)");
    goto cleanup;
  }

  // 4. MUTATION PHASE (Locked)
  SHADOW_LOCK(&self->shadow_lock);
  
  // Guard with Acquire ordering to match query entry/exit barriers
  if (self->is_stepping || atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_RuntimeError, "Cannot create mesh while world is busy (stepping or querying)");
      goto cleanup;
  }
  
  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity + 1024) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      goto cleanup; 
    }
  }

  slot = self->free_slots[--self->free_count];
  self->slot_states[slot] = SLOT_PENDING_CREATE;
  slot_reserved = true;

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)px; pos->y = (double)py; pos->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx; rot->y = ry; rot->z = rz; rot->w = rw;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, JPH_MotionType_Static, 0 
  );

  if (!settings) {
    self->slot_states[slot] = SLOT_EMPTY;
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    ret_val = PyErr_NoMemory();
    goto cleanup;
  }

  // Identity Publishing
  uint32_t gen = self->generations[slot];
  BodyHandle handle = make_handle(slot, gen);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  if (!ensure_command_capacity(self)) {
    JPH_BodyCreationSettings_Destroy(settings);
    self->slot_states[slot] = SLOT_EMPTY;
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    ret_val = PyErr_NoMemory();
    goto cleanup;
  }

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->type = CMD_CREATE_BODY;
  cmd->slot = slot;
  cmd->data.create.settings = settings;
  cmd->data.create.user_data = (uint64_t)user_data;

  SHADOW_UNLOCK(&self->shadow_lock);
  
  ret_val = PyLong_FromUnsignedLongLong(handle);

cleanup:
  if (shape) JPH_Shape_Destroy(shape);
  if (v_view.obj) PyBuffer_Release(&v_view);
  if (i_view.obj) PyBuffer_Release(&i_view);
  return ret_val; 
}

static PyObject *PhysicsWorld_destroy_body(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  SHADOW_LOCK(&self->shadow_lock);

  // 1. MUTATION GUARD
  // Prevents modifying topology while Jolt is stepping or while 
  // background threads are querying the world state.
  if (self->is_stepping || atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_RuntimeError, "Cannot destroy body while world is busy (stepping or querying)");
      return NULL;
  }

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
    cmd->type = CMD_DESTROY_BODY;
    cmd->slot = slot;

    // Mark the slot immediately. This ensures that any logic 
    // running between now and the next step() treats this body as "gone".
    self->slot_states[slot] = SLOT_PENDING_DESTROY;
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}


// Helper macro to get a float attribute, decref it, and handle errors
#define GET_FLOAT_ATTR(obj, name, target) do {            \
    PyObject* attr = PyObject_GetAttrString(obj, name);  \
    if (attr) {                                          \
        double _v = PyFloat_AsDouble(attr);              \
        Py_DECREF(attr);                                 \
        if (!PyErr_Occurred()) target = (float)_v;       \
    }                                                     \
    PyErr_Clear();                                       \
} while(0)

// vroom vroom
// this is paperwork and i did surgery in the core
static PyObject* PhysicsWorld_create_vehicle(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t chassis_h = 0;
    PyObject* py_wheels = NULL; 
    char* drive_str = "RWD";
    PyObject* py_engine = NULL;
    PyObject* py_trans = NULL;

    static char *kwlist[] = {"chassis", "wheels", "drive", "engine", "transmission", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KO|sOO", kwlist, 
        &chassis_h, &py_wheels, &drive_str, &py_engine, &py_trans)) return NULL;
    
    if (!PyList_Check(py_wheels)) {
        PyErr_SetString(PyExc_TypeError, "wheels must be a list of dictionaries");
        return NULL;
    }

    Py_ssize_t num_wheels = PyList_Size(py_wheels);
    if (num_wheels < 2) {
        PyErr_SetString(PyExc_ValueError, "Vehicle must have at least 2 wheels");
        return NULL;
    }

    // --- 1. CHASSIS RESOLUTION & INITIAL GUARD ---
    SHADOW_LOCK(&self->shadow_lock);
    if (self->is_stepping || atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot create vehicle while world is busy");
        return NULL;
    }

    flush_commands(self); 
    
    uint32_t slot;
    if (!unpack_handle(self, chassis_h, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid chassis handle");
        return NULL;
    }
    JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 2. JOLT BODY LOCKING ---
    const JPH_BodyLockInterface* lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock;
    JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);
    bool body_locked = true; 

    if (!lock.body) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
        body_locked = false;
        PyErr_SetString(PyExc_RuntimeError, "Could not lock chassis body");
        return NULL;
    }

    // --- 3. RESOURCE TRACKING (Zero-init for safe cleanup) ---
    JPH_LinearCurve* f_curve = NULL;
    JPH_LinearCurve* t_curve = NULL;
    JPH_WheelSettings** w_settings = NULL;
    JPH_WheeledVehicleControllerSettings* v_ctrl = NULL;
    JPH_VehicleTransmissionSettings* v_trans_set = NULL;
    JPH_VehicleCollisionTesterRay* tester = NULL;
    JPH_VehicleConstraint* j_veh = NULL;
    bool constraint_added = false;

    // curves
    f_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(f_curve, 0.0f, 1.0f); JPH_LinearCurve_AddPoint(f_curve, 1.0f, 1.0f);
    t_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(t_curve, 0.0f, 1.0f); JPH_LinearCurve_AddPoint(t_curve, 1.0f, 1.0f);

    // wheels
    w_settings = (JPH_WheelSettings**)PyMem_RawCalloc(num_wheels, sizeof(JPH_WheelSettings*));
    if (!w_settings) { PyErr_NoMemory(); goto error_cleanup; }

    for (Py_ssize_t i = 0; i < num_wheels; i++) {
        PyObject* w_dict = PyList_GetItem(py_wheels, i);
        if (!PyDict_Check(w_dict)) { PyErr_SetString(PyExc_TypeError, "Each wheel entry must be a dict"); goto error_cleanup; }

        PyObject *o_pos = PyDict_GetItemString(w_dict, "pos");
        float px, py, pz, radius = 0.4f, width = 0.2f;

        // FIXED: Generic sequence parsing for 'pos' (Accepts lists and tuples)
        if (!o_pos || !PySequence_Check(o_pos) || PySequence_Size(o_pos) != 3) {
            PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 floats");
            goto error_cleanup;
        }
        PyObject* p0 = PySequence_GetItem(o_pos, 0); px = (float)PyFloat_AsDouble(p0); Py_DECREF(p0);
        PyObject* p1 = PySequence_GetItem(o_pos, 1); py = (float)PyFloat_AsDouble(p1); Py_DECREF(p1);
        PyObject* p2 = PySequence_GetItem(o_pos, 2); pz = (float)PyFloat_AsDouble(p2); Py_DECREF(p2);

        GET_FLOAT_ATTR(w_dict, "radius", radius);
        GET_FLOAT_ATTR(w_dict, "width", width);

        JPH_WheelSettingsWV* w = JPH_WheelSettingsWV_Create();
        JPH_WheelSettings_SetPosition((JPH_WheelSettings*)w, &(JPH_Vec3){px, py, pz});
        JPH_WheelSettings_SetRadius((JPH_WheelSettings*)w, radius);
        JPH_WheelSettings_SetWidth((JPH_WheelSettings*)w, width);
        JPH_WheelSettingsWV_SetLongitudinalFriction(w, f_curve);
        JPH_WheelSettingsWV_SetLateralFriction(w, f_curve);
        if (i < 2) JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.5f);
        w_settings[i] = (JPH_WheelSettings*)w;
    }

    // --- 4. ENGINE & TRANSMISSION ---
    v_ctrl = JPH_WheeledVehicleControllerSettings_Create();
    
    float torque = 500.0f, max_rpm = 7000.0f, min_rpm = 1000.0f, inertia = 0.5f;
    if (py_engine && py_engine != Py_None) {
        GET_FLOAT_ATTR(py_engine, "max_torque", torque);
        GET_FLOAT_ATTR(py_engine, "max_rpm", max_rpm);
        GET_FLOAT_ATTR(py_engine, "min_rpm", min_rpm);
        GET_FLOAT_ATTR(py_engine, "inertia", inertia);
    }
    JPH_VehicleEngineSettings eng_set; JPH_VehicleEngineSettings_Init(&eng_set);
    eng_set.maxTorque = torque; eng_set.maxRPM = max_rpm; eng_set.minRPM = min_rpm;
    eng_set.inertia = inertia; eng_set.normalizedTorque = t_curve; 
    JPH_WheeledVehicleControllerSettings_SetEngine(v_ctrl, &eng_set);

    v_trans_set = JPH_VehicleTransmissionSettings_Create();
    int t_mode = 1; float clutch = 2000.0f;
    if (py_trans && py_trans != Py_None) {
        PyObject* o_mode = PyObject_GetAttrString(py_trans, "mode");
        if (o_mode) { t_mode = (int)PyLong_AsLong(o_mode); Py_DECREF(o_mode); }
        GET_FLOAT_ATTR(py_trans, "clutch_strength", clutch);
    }
    JPH_VehicleTransmissionSettings_SetMode(v_trans_set, (JPH_TransmissionMode)t_mode);
    JPH_VehicleTransmissionSettings_SetClutchStrength(v_trans_set, clutch); 
    
    // FIXED: Default Reverse Gear Ratio
    float rev_gears[] = { -3.0f };
    JPH_VehicleTransmissionSettings_SetReverseGearRatios(v_trans_set, rev_gears, 1);

    JPH_WheeledVehicleControllerSettings_SetTransmission(v_ctrl, v_trans_set);

    // Differentials (Patched JoltC helper)
    if (strcmp(drive_str, "FWD") == 0) {
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
    } else if (strcmp(drive_str, "AWD") == 0 && num_wheels >= 4) {
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 2, 3);
    } else { // RWD
        uint32_t i1 = (num_wheels >= 4) ? 2 : 0;
        uint32_t i2 = (num_wheels >= 4) ? 3 : 1;
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, i1, i2);
    }

    // --- 5. CONSTRAINT ASSEMBLY ---
    JPH_VehicleConstraintSettings v_set; JPH_VehicleConstraintSettings_Init(&v_set);
    v_set.wheelsCount = (uint32_t)num_wheels; 
    v_set.wheels = w_settings; 
    v_set.controller = (JPH_VehicleControllerSettings*)v_ctrl;
    v_set.up.x = 0; v_set.up.y = 1; v_set.up.z = 0;
    v_set.forward.x = 0; v_set.forward.y = 0; v_set.forward.z = 1;

    j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
    if (!j_veh) { PyErr_SetString(PyExc_RuntimeError, "Jolt vehicle construction failed"); goto error_cleanup; }

    // Collision Tester attached BEFORE insertion
    tester = JPH_VehicleCollisionTesterRay_Create(1, &(JPH_Vec3){0, 1, 0}, 1.0f);
    JPH_VehicleConstraint_SetVehicleCollisionTester(j_veh, (JPH_VehicleCollisionTester*)tester);

    // --- 6. INSERTION (Shadow Locked) ---
    SHADOW_LOCK(&self->shadow_lock);
    if (self->is_stepping || atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "World became busy during insertion");
        goto error_cleanup;
    }

    JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint*)j_veh);
    constraint_added = true;
    JPH_PhysicsSystem_AddStepListener(self->system, JPH_VehicleConstraint_AsPhysicsStepListener(j_veh));
    SHADOW_UNLOCK(&self->shadow_lock);

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    body_locked = false;

    // --- 7. PYTHON WRAPPER ---
    PyObject *module = PyType_GetModule(Py_TYPE(self));
    CulverinState *st = get_culverin_state(module);
    VehicleObject *obj = (VehicleObject *)PyObject_New(VehicleObject, (PyTypeObject *)st->VehicleType);
    if (!obj) goto error_cleanup;

    obj->vehicle = j_veh; obj->tester = (JPH_VehicleCollisionTester*)tester; obj->world = self;
    obj->num_wheels = (uint32_t)num_wheels; obj->current_gear = 1; 
    obj->wheel_settings = w_settings; obj->controller_settings = (JPH_VehicleControllerSettings*)v_ctrl;
    obj->transmission_settings = v_trans_set; obj->friction_curve = f_curve; obj->torque_curve = t_curve;
    
    Py_INCREF(self);
    return (PyObject *)obj;

error_cleanup:
    if (body_locked) JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    if (j_veh) { 
        if (constraint_added) {
            SHADOW_LOCK(&self->shadow_lock);
            JPH_PhysicsSystem_RemoveConstraint(self->system, (JPH_Constraint*)j_veh); 
            SHADOW_UNLOCK(&self->shadow_lock);
        }
        JPH_Constraint_Destroy((JPH_Constraint*)j_veh); 
    }
    if (tester) JPH_VehicleCollisionTester_Destroy(tester);
    if (v_trans_set) JPH_VehicleTransmissionSettings_Destroy(v_trans_set);
    if (v_ctrl) JPH_VehicleControllerSettings_Destroy((JPH_VehicleControllerSettings*)v_ctrl);
    if (w_settings) {
        for (Py_ssize_t i = 0; i < num_wheels; i++) if (w_settings[i]) JPH_WheelSettings_Destroy(w_settings[i]);
        PyMem_RawFree(w_settings);
    }
    if (f_curve) JPH_LinearCurve_Destroy(f_curve);
    if (t_curve) JPH_LinearCurve_Destroy(t_curve);
    return NULL;
}

static PyObject *PhysicsWorld_set_position(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN; float y = NAN; float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x, &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  GUARD_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  uint32_t idx = self->slot_to_dense[slot];
  JPH_STACK_ALLOC(JPH_RVec3, p);
  p->x = (double)x; p->y = (double)y; p->z = (double)z;

  // 1. Update Jolt (Thread-safe)
  JPH_BodyInterface_SetPosition(self->body_interface, self->body_ids[idx], p, JPH_Activation_Activate);

  // 2. Update Shadow Buffer
  self->positions[idx * 4 + 0] = x;
  self->positions[idx * 4 + 1] = y;
  self->positions[idx * 4 + 2] = z;

  // 3. FIX: Reset interpolation state to prevent "teleport streaks"
  self->prev_positions[idx * 4 + 0] = x;
  self->prev_positions[idx * 4 + 1] = y;
  self->prev_positions[idx * 4 + 2] = z;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_rotation(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x, y, z, w;
  static char *kwlist[] = {"handle", "x", "y", "z", "w", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffff", kwlist, &handle_raw, &x, &y, &z, &w)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock); 
  GUARD_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_STACK_ALLOC(JPH_Quat, q);
  q->x = x; q->y = y; q->z = z; q->w = w;

  // 1. Update Jolt
  JPH_BodyInterface_SetRotation(self->body_interface, self->body_ids[dense_idx], q, JPH_Activation_Activate);

  // 2. Update Current Shadow
  self->rotations[dense_idx * 4 + 0] = x;
  self->rotations[dense_idx * 4 + 1] = y;
  self->rotations[dense_idx * 4 + 2] = z;
  self->rotations[dense_idx * 4 + 3] = w;

  // 3. FIX: Update Previous Shadow to reset interpolation
  memcpy(&self->prev_rotations[dense_idx * 4], &self->rotations[dense_idx * 4], 16);

  SHADOW_UNLOCK(&self->shadow_lock); 
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                  PyObject *args,
                                                  PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN; float y = NAN; float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  GUARD_STEPPING(self);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }
  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_BodyID bid = self->body_ids[dense_idx];

  JPH_Vec3 v = {x, y, z};
  JPH_BodyInterface_SetLinearVelocity(self->body_interface, bid, &v);
  self->linear_velocities[dense_idx * 4 + 0] = x;
  self->linear_velocities[dense_idx * 4 + 1] = y;
  self->linear_velocities[dense_idx * 4 + 2] = z;
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN; float y = NAN; float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  GUARD_STEPPING(self);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }
  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_BodyID bid = self->body_ids[dense_idx];

  JPH_Vec3 v = {x, y, z};
  JPH_BodyInterface_SetAngularVelocity(self->body_interface, bid, &v);
  self->angular_velocities[dense_idx * 4 + 0] = x;
  self->angular_velocities[dense_idx * 4 + 1] = y;
  self->angular_velocities[dense_idx * 4 + 2] = z;
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  SHADOW_LOCK(&self->shadow_lock);
  
  // FIX: Consistency Guard
  GUARD_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
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
  GUARD_STEPPING(self);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }
  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
  JPH_BodyInterface_SetMotionType(self->body_interface, bid,
                                  (JPH_MotionType)motion_type,
                                  JPH_Activation_Activate);
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  unsigned long long data = 0;
  static char *kwlist[] = {"handle", "data", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KK", kwlist, &handle_raw, &data)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  
  // 1. FIX: Prevent mutation while step() is rearranging the dense array
  GUARD_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  // 2. Map to the dense index (safe because we are locked and not stepping)
  uint32_t dense = self->slot_to_dense[slot];
  self->user_data[dense] = (uint64_t)data;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_user_data(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  SHADOW_LOCK(&self->shadow_lock);
  
  // 1. FIX: Ensure indices aren't shifting while we read
  GUARD_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
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
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  SHADOW_LOCK(&self->shadow_lock);
  GUARD_STEPPING(self);                                      

  uint32_t slot = 0;
  // Consistency Check: Unpack AND verify the slot is currently ALIVE
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  // Thread-safe call into Jolt internal state
  JPH_BodyInterface_ActivateBody(self->body_interface,
                                 self->body_ids[self->slot_to_dense[slot]]);

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_deactivate(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  SHADOW_LOCK(&self->shadow_lock);
  GUARD_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyInterface_DeactivateBody(self->body_interface,
                                   self->body_ids[self->slot_to_dense[slot]]);

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
                                   &handle_raw, &px, &py, &pz, 
                                   &rx, &ry, &rz, &rw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  
  // 1. Mutation Guard
  GUARD_STEPPING(self);

  // 2. Handle Resolution & Liveness check
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_BodyID bid = self->body_ids[dense_idx];

  // 3. Jolt Sync (Immediate)
  // Because callbacks are lock-free, we can call this safely under shadow_lock.
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)px; pos->y = (double)py; pos->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx; rot->y = ry; rot->z = rz; rot->w = rw;

  JPH_BodyInterface_SetPositionAndRotation(self->body_interface, bid, pos, rot, 
                                           JPH_Activation_Activate);

  // 4. Current Shadow Buffer Update
  size_t off = (size_t)dense_idx * 4;
  self->positions[off + 0] = px;
  self->positions[off + 1] = py;
  self->positions[off + 2] = pz;

  self->rotations[off + 0] = rx;
  self->rotations[off + 1] = ry;
  self->rotations[off + 2] = rz;
  self->rotations[off + 3] = rw;

  // 5. Previous Shadow Buffer Update (Reset Interpolation)
  // This prevents the "Visual Streak" on the frame the body is moved.
  memcpy(&self->prev_positions[off], &self->positions[off], 16);
  memcpy(&self->prev_rotations[off], &self->rotations[off], 16);

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// Unified hit collector for both Broad and Narrow phase overlaps
static void overlap_record_hit(OverlapContext *ctx, JPH_BodyID bid) {
    // 1. Grow buffer if needed (Native memory, no GIL required)
    if (ctx->count >= ctx->capacity) {
        size_t new_cap = (ctx->capacity == 0) ? 32 : ctx->capacity * 2;
        uint64_t *new_ptr = PyMem_RawRealloc(ctx->hits, new_cap * sizeof(uint64_t));
        if (!new_ptr) return; // Drop hit on OOM (safer than crashing)
        ctx->hits = new_ptr;
        ctx->capacity = new_cap;
    }

    // 2. Retrieve the baked Handle from Jolt UserData
    // This handle contains the Generation + Slot at the time of creation.
    ctx->hits[ctx->count++] = (uint64_t)JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);
}

static float OverlapCallback_Narrow(void *context, const JPH_CollideShapeResult *result) {
    overlap_record_hit((OverlapContext *)context, result->bodyID2);
    return 1.0f; // Continue looking for more hits
}

static float OverlapCallback_Broad(void *context, const JPH_BodyID result_bid) {
    overlap_record_hit((OverlapContext *)context, result_bid);
    return 1.0f; // Continue
}

static PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  float x = 0.0f, y = 0.0f, z = 0.0f, radius = 1.0f;
  static char *kwlist[] = {"center", "radius", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &x, &y, &z, &radius)) return NULL;

  // Trackers for safe cleanup
  PyObject *ret_val = NULL; 
  OverlapContext ctx = { .world = self, .hits = NULL, .count = 0, .capacity = 0 };
  bool query_active = false;
  JPH_Shape *shape = NULL;

  // --- 1. PHASE GUARD ---
  SHADOW_LOCK(&self->shadow_lock);
  if (self->is_stepping) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_RuntimeError, "Cannot overlap_sphere while physics is stepping");
      return NULL;
  }
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  query_active = true;
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT RESOURCE PREP (Unlocked) ---
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
  pos->x = (double)x; pos->y = (double)y; pos->z = (double)z;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = 0; rot->y = 0; rot->z = 0; rot->w = 1;
  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, rot, pos);
  JPH_STACK_ALLOC(JPH_Vec3, scale);
  scale->x = 1.0f; scale->y = 1.0f; scale->z = 1.0f;
  JPH_STACK_ALLOC(JPH_RVec3, base_offset);
  base_offset->x = 0; base_offset->y = 0; base_offset->z = 0;
  JPH_STACK_ALLOC(JPH_CollideShapeSettings, settings);
  JPH_CollideShapeSettings_Init(settings);

  // --- 3. EXECUTE QUERY (Unlocked) ---
  const JPH_NarrowPhaseQuery *nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
  JPH_NarrowPhaseQuery_CollideShape(
      nq, shape, scale, transform, settings, base_offset,
      OverlapCallback_Narrow, &ctx, NULL, NULL, NULL, NULL
  );

  // --- 4. VALIDATION & LIST CONSTRUCTION (Locked) ---
  ret_val = PyList_New(0);
  if (!ret_val) goto cleanup;

  SHADOW_LOCK(&self->shadow_lock);
  for (size_t i = 0; i < ctx.count; i++) {
      uint64_t h = ctx.hits[i];
      uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
      uint32_t gen  = (uint32_t)(h >> 32);

      if (slot < self->slot_capacity && 
          self->generations[slot] == gen && 
          self->slot_states[slot] == SLOT_ALIVE) 
      {
          PyObject *py_h = PyLong_FromUnsignedLongLong(h);
          if (py_h) {
              PyList_Append(ret_val, py_h);
              Py_DECREF(py_h);
          }
      }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

cleanup:
  if (query_active) atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_relaxed);
  if (shape) JPH_Shape_Destroy(shape);
  if (ctx.hits) PyMem_RawFree(ctx.hits);
  return ret_val; 
}

static PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  float min_x, min_y, min_z, max_x, max_y, max_z;
  static char *kwlist[] = {"min", "max", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, 
                                   &min_x, &min_y, &min_z, &max_x, &max_y, &max_z)) {
    return NULL;
  }

  PyObject *ret_val = NULL;
  OverlapContext ctx = { .world = self, .hits = NULL, .count = 0, .capacity = 0 };
  bool query_active = false;

  // --- 1. PHASE GUARD ---
  SHADOW_LOCK(&self->shadow_lock);
  if (self->is_stepping) {
      SHADOW_UNLOCK(&self->shadow_lock);
      PyErr_SetString(PyExc_RuntimeError, "Cannot overlap_aabb while physics is stepping");
      return NULL;
  }
  atomic_fetch_add_explicit(&self->active_queries, 1, memory_order_relaxed);
  query_active = true;
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. JOLT PREP ---
  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x; box->min.y = min_y; box->min.z = min_z;
  box->max.x = max_x; box->max.y = max_y; box->max.z = max_z;

  // --- 3. EXECUTE BROADPHASE (Unlocked) ---
  const JPH_BroadPhaseQuery *bq = JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);
  JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx, NULL, NULL);

  // --- 4. VALIDATION & CONSTRUCTION (Locked) ---
  ret_val = PyList_New(0);
  if (!ret_val) goto cleanup;

  SHADOW_LOCK(&self->shadow_lock);
  for (size_t i = 0; i < ctx.count; i++) {
      uint64_t h = ctx.hits[i];
      uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
      uint32_t gen  = (uint32_t)(h >> 32);

      if (slot < self->slot_capacity && 
          self->generations[slot] == gen && 
          self->slot_states[slot] == SLOT_ALIVE) 
      {
          PyObject *py_h = PyLong_FromUnsignedLongLong(h);
          if (py_h) {
              PyList_Append(ret_val, py_h);
              Py_DECREF(py_h);
          }
      }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

cleanup:
  if (query_active) atomic_fetch_sub_explicit(&self->active_queries, 1, memory_order_relaxed);
  if (ctx.hits) PyMem_RawFree(ctx.hits);
  return ret_val;
}

static PyObject *PhysicsWorld_get_index(PhysicsWorldObject *self,
                                        PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) return NULL;
  
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

  if (alive) Py_RETURN_TRUE;
  Py_RETURN_FALSE;
}

static PyObject *make_view(PhysicsWorldObject *self, void *ptr) {
  if (!ptr) Py_RETURN_NONE;

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
  Py_ssize_t local_shape[1] = { (Py_ssize_t)(current_count * 4) };
  Py_ssize_t local_strides[1] = { (Py_ssize_t)sizeof(float) };

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
  PyObject* mv = PyMemoryView_FromBuffer(&buf);
  
  if (!mv) {
      // Clean up on failure
      SHADOW_LOCK(&self->shadow_lock);
      if (self->view_export_count > 0) self->view_export_count--;
      SHADOW_UNLOCK(&self->shadow_lock);
      
      Py_DECREF(self); // Drop the ownership link ref
      return NULL;
  }
  
  return mv;
}

static PyObject* PhysicsWorld_get_active_indices(PhysicsWorldObject* self, PyObject* args) {
    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->count;
    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyBytes_FromStringAndSize(NULL, 0);
    }

    // 1. Snapshot the BodyIDs while locked (Fast)
    JPH_BodyID* id_scratch = (JPH_BodyID*)PyMem_RawMalloc(count * sizeof(JPH_BodyID));
    if (!id_scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }
    memcpy(id_scratch, self->body_ids, count * sizeof(JPH_BodyID));
    SHADOW_UNLOCK(&self->shadow_lock);

    // 2. Query activity state WHILE UNLOCKED (Deadlock safe)
    uint32_t* results = (uint32_t*)PyMem_RawMalloc(count * sizeof(uint32_t));
    size_t active_count = 0;
    JPH_BodyInterface* bi = self->body_interface;

    for (size_t i = 0; i < count; i++) {
        if (id_scratch[i] != JPH_INVALID_BODY_ID && JPH_BodyInterface_IsActive(bi, id_scratch[i])) {
            results[active_count++] = (uint32_t)i;
        }
    }

    // 3. Construct Python object and cleanup
    PyObject* bytes_obj = PyBytes_FromStringAndSize((char*)results, active_count * sizeof(uint32_t));
    PyMem_RawFree(id_scratch);
    PyMem_RawFree(results);
    return bytes_obj;
}

static PyObject *get_user_data_buffer(PhysicsWorldObject *self, void *c) {
  if (!self->user_data) Py_RETURN_NONE;

  SHADOW_LOCK(&self->shadow_lock);
  size_t current_count = self->count;
  self->view_export_count++;
  SHADOW_UNLOCK(&self->shadow_lock);

  // Use stack-allocated metadata to prevent cross-thread corruption
  Py_ssize_t local_shape[1] = { (Py_ssize_t)current_count };
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

  PyObject* mv = PyMemoryView_FromBuffer(&buf);
  if (!mv) {
      SHADOW_LOCK(&self->shadow_lock);
      if (self->view_export_count > 0) self->view_export_count--;
      SHADOW_UNLOCK(&self->shadow_lock);
      Py_DECREF(self);
      return NULL;
  }
  return mv;
}

static PyObject* PhysicsWorld_get_render_state(PhysicsWorldObject* self, PyObject* args) {
    float alpha;
    if (!PyArg_ParseTuple(args, "f", &alpha)) return NULL;

    alpha = (alpha < 0.0f) ? 0.0f : (alpha > 1.0f ? 1.0f : alpha);

    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->count;
    size_t total_bytes = count * 7 * sizeof(float);

    // 1. Allocate the Python Bytes object immediately with NULL.
    // This reserves the memory inside the Python Heap.
    PyObject* bytes_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_bytes);
    if (!bytes_obj) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // 2. Get the direct pointer to the Bytes object's internal buffer.
    float* out = (float*)PyBytes_AsString(bytes_obj);

    for (size_t i = 0; i < count; i++) {
        size_t src = i * 4;
        size_t dst = i * 7;

        // Position Lerp
        out[dst+0] = self->prev_positions[src+0] + (self->positions[src+0] - self->prev_positions[src+0]) * alpha;
        out[dst+1] = self->prev_positions[src+1] + (self->positions[src+1] - self->prev_positions[src+1]) * alpha;
        out[dst+2] = self->prev_positions[src+2] + (self->positions[src+2] - self->prev_positions[src+2]) * alpha;

        // Rotation NLerp
        float q1x = self->prev_rotations[src+0], q1y = self->prev_rotations[src+1], q1z = self->prev_rotations[src+2], q1w = self->prev_rotations[src+3];
        float q2x = self->rotations[src+0], q2y = self->rotations[src+1], q2z = self->rotations[src+2], q2w = self->rotations[src+3];

        float dot = q1x*q2x + q1y*q2y + q1z*q2z + q1w*q2w;
        if (dot < 0.0f) { q2x = -q2x; q2y = -q2y; q2z = -q2z; q2w = -q2w; }

        float rx = q1x + (q2x - q1x) * alpha;
        float ry = q1y + (q2y - q1y) * alpha;
        float rz = q1z + (q2z - q1z) * alpha;
        float rw = q1w + (q2w - q1w) * alpha;

        float inv_len = 1.0f / sqrtf(rx*rx + ry*ry + rz*rz + rw*rw);
        out[dst+3] = rx * inv_len;
        out[dst+4] = ry * inv_len;
        out[dst+5] = rz * inv_len;
        out[dst+6] = rw * inv_len;
    }

    SHADOW_UNLOCK(&self->shadow_lock);

    // Return the object directly to Python
    return bytes_obj;
}

// --- Vehicles Methods ---

static PyObject* Vehicle_set_input(VehicleObject* self, PyObject* args, PyObject* kwds) {
    float forward, right, brake, handbrake;
    static char *kwlist[] = {"forward", "right", "brake", "handbrake", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ffff", kwlist, &forward, &right, &brake, &handbrake)) return NULL;

    SHADOW_LOCK(&self->world->shadow_lock);
    
    // 1. RE-ENTRANCY GUARD
    if (self->world->is_stepping) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot set vehicle input during physics step");
        return NULL;
    }

    if (!self->vehicle) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Vehicle instance is invalid or destroyed");
        return NULL;
    }

    // 2. RESOLVE JOLT COMPONENTS
    JPH_WheeledVehicleController* controller = (JPH_WheeledVehicleController*)JPH_VehicleConstraint_GetController(self->vehicle);
    if (!controller) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Vehicle controller is missing");
        return NULL;
    }

    const JPH_Body* chassis = JPH_VehicleConstraint_GetVehicleBody(self->vehicle);
    JPH_VehicleTransmission* transmission = (JPH_VehicleTransmission*)JPH_WheeledVehicleController_GetTransmission(controller);
    
    // Wake up the car if it went to sleep
    JPH_BodyInterface_ActivateBody(self->world->body_interface, JPH_Body_GetID(chassis));

    // 3. SPEED & DIRECTION CALCULATIONS (ALIGNED)
    JPH_STACK_ALLOC(JPH_Vec3, linear_vel);
    JPH_Body_GetLinearVelocity((JPH_Body *)chassis, linear_vel);
    
    JPH_STACK_ALLOC(JPH_RMat4, world_transform);
    JPH_Body_GetWorldTransform(chassis, world_transform);
    
    // Forward vector is Z-axis (Column 2). Note: Jolt uses column-major matrices.
    JPH_Vec3 world_fwd = { 
        (float)world_transform->column[2].x, 
        (float)world_transform->column[2].y, 
        (float)world_transform->column[2].z 
    }; 
    
    // Dot product for forward speed
    float speed = linear_vel->x * world_fwd.x + linear_vel->y * world_fwd.y + linear_vel->z * world_fwd.z;

    // 4. SMART INPUT STATE MACHINE
    float input_throttle = 0.0f;
    float input_brake = brake; 
    float clutch_friction = 1.0f; 
    int target_gear = self->current_gear;

    // Apply deadzone
    if (fabsf(forward) < 0.01f) forward = 0.0f;

    if (forward > 0.01f) {
        // DRIVE LOGIC
        if (speed < -0.5f) {
            // Moving backward, user pressed forward: BRAKE
            input_brake = 1.0f;
            input_throttle = 0.0f;
        } else {
            input_throttle = forward;
            target_gear = 1; 
            clutch_friction = 1.0f;
        }
    } 
    else if (forward < -0.01f) {
        // REVERSE LOGIC
        if (speed > 0.5f) {
            // Moving forward, user pressed backward: BRAKE
            input_brake = 1.0f;
            input_throttle = 0.0f;
        } else {
            input_throttle = fabsf(forward);
            target_gear = -1; 
            clutch_friction = 1.0f;
        }
    } 
    else {
        // NEUTRAL / COASTING
        input_throttle = 0.0f;
        target_gear = 0;      
        clutch_friction = 0.0f; 
        
        // Gentle rolling resistance
        if (input_brake < 0.01f) input_brake = 0.05f; 
    }

    // 5. APPLY TO JOLT
    self->current_gear = target_gear;
    JPH_VehicleTransmission_Set(transmission, self->current_gear, clutch_friction);

    JPH_WheeledVehicleController_SetDriverInput(
        controller,
        input_throttle,
        right,
        input_brake,
        handbrake
    );

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject* Vehicle_get_wheel_transform(VehicleObject* self, PyObject* args) {
    uint32_t index;
    if (!PyArg_ParseTuple(args, "I", &index)) return NULL;

    SHADOW_LOCK(&self->world->shadow_lock);
    if (self->world->is_stepping) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot query wheel transform during physics step");
        return NULL;
    }
    if (!self->vehicle || index >= self->num_wheels) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    JPH_STACK_ALLOC(JPH_RMat4, transform);
    JPH_Vec3 right = {1.0f, 0.0f, 0.0f};
    JPH_Vec3 up = {0.0f, 1.0f, 0.0f};

    JPH_VehicleConstraint_GetWheelWorldTransform(self->vehicle, index, &right, &up, transform);

    // --- CRITICAL FIX: Layout Mapping ---
    
    // 1. Position: In Double Precision, this is the 'column3' member (RVec3/doubles)
    double px = transform->column3.x;
    double py = transform->column3.y;
    double pz = transform->column3.z;

    // 2. Rotation: These are the first 3 columns (Vec4/floats)
    JPH_STACK_ALLOC(JPH_Mat4, rot_only_mat);
    JPH_Mat4_Identity(rot_only_mat);
    rot_only_mat->column[0] = transform->column[0];
    rot_only_mat->column[1] = transform->column[1];
    rot_only_mat->column[2] = transform->column[2];
    
    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_Mat4_GetQuaternion(rot_only_mat, q);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // Safe Python construction
    PyObject* py_pos = Py_BuildValue("(ddd)", px, py, pz);
    PyObject* py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);
    
    if (!py_pos || !py_rot) {
        Py_XDECREF(py_pos); Py_XDECREF(py_rot);
        return NULL;
    }

    PyObject* result = PyTuple_Pack(2, py_pos, py_rot);
    Py_DECREF(py_pos); Py_DECREF(py_rot);
    return result;
}

static PyObject* Vehicle_get_wheel_local_transform(VehicleObject* self, PyObject* args) {
    uint32_t index;
    if (!PyArg_ParseTuple(args, "I", &index)) return NULL;

    SHADOW_LOCK(&self->world->shadow_lock);
    if (!self->vehicle || index >= self->num_wheels) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    // 1. Use JPH_Mat4 (Standard 4x4 float matrix)
    JPH_STACK_ALLOC(JPH_Mat4, local_transform);
    
    JPH_Vec3 right = {1.0f, 0.0f, 0.0f};
    JPH_Vec3 up = {0.0f, 1.0f, 0.0f};

    // Jolt fills the float matrix
    JPH_VehicleConstraint_GetWheelLocalTransform(self->vehicle, index, &right, &up, local_transform);

    // 2. Extract Translation (column[3] is the 4th column in Mat4)
    float lx = local_transform->column[3].x;
    float ly = local_transform->column[3].y;
    float lz = local_transform->column[3].z;

    // 3. Extract Rotation
    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_Mat4_GetQuaternion(local_transform, q);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // Build the Python response: ((lx, ly, lz), (rx, ry, rz, rw))
    PyObject* py_pos = Py_BuildValue("(fff)", lx, ly, lz);
    PyObject* py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);
    
    if (!py_pos || !py_rot) {
        Py_XDECREF(py_pos); Py_XDECREF(py_rot);
        return NULL;
    }

    PyObject* result = PyTuple_Pack(2, py_pos, py_rot);
    Py_DECREF(py_pos); Py_DECREF(py_rot);
    
    return result;
}

static PyObject* Vehicle_get_debug_state(VehicleObject* self, PyObject* Py_UNUSED(ignored)) {
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
    JPH_WheeledVehicleController* controller = (JPH_WheeledVehicleController*)JPH_VehicleConstraint_GetController(self->vehicle);
    if (!controller) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    const JPH_VehicleEngine* engine = JPH_WheeledVehicleController_GetEngine(controller);
    const JPH_VehicleTransmission* trans = JPH_WheeledVehicleController_GetTransmission(controller);
    
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
        const JPH_Wheel* w = JPH_VehicleConstraint_GetWheel(self->vehicle, i);
        const JPH_WheelSettings* ws = JPH_Wheel_GetSettings(w);
        
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
static PyObject* Vehicle_destroy(VehicleObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->world) Py_RETURN_NONE; 

    // --- 1. CAPTURE & INVALIDATE PHASE (Locked) ---
    SHADOW_LOCK(&self->world->shadow_lock);

    // GUARD: Prevents mutation while Jolt is busy stepping or querying
    if (self->world->is_stepping || atomic_load_explicit(&self->world->active_queries, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Cannot destroy vehicle while world is busy (stepping or querying)");
        return NULL;
    }

    if (!self->vehicle) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    // Capture Jolt pointers to local stack variables
    JPH_VehicleConstraint* j_veh = self->vehicle;
    JPH_VehicleCollisionTester* tester = self->tester;
    JPH_VehicleControllerSettings* v_ctrl = self->controller_settings;
    JPH_VehicleTransmissionSettings* v_trans = self->transmission_settings;
    JPH_WheelSettings** wheels = self->wheel_settings;
    JPH_LinearCurve* f_curve = self->friction_curve;
    JPH_LinearCurve* t_curve = self->torque_curve;
    uint32_t wheel_count = self->num_wheels;

    // NULLify the struct immediately so no other thread can enter this block
    self->vehicle = NULL;
    self->tester = NULL;
    self->controller_settings = NULL;
    self->transmission_settings = NULL;
    self->wheel_settings = NULL;
    self->friction_curve = NULL;
    self->torque_curve = NULL;

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // --- 2. JOLT CLEANUP PHASE (Unlocked) ---
    // No Shadow-vs-Jolt deadlocks possible here!
    
    if (j_veh) {
        // Step Listener must be removed first
        JPH_PhysicsStepListener* step_listener = JPH_VehicleConstraint_AsPhysicsStepListener(j_veh);
        JPH_PhysicsSystem_RemoveStepListener(self->world->system, step_listener);

        JPH_PhysicsSystem_RemoveConstraint(self->world->system, (JPH_Constraint*)j_veh);
        JPH_Constraint_Destroy((JPH_Constraint*)j_veh);
    }

    if (tester) JPH_VehicleCollisionTester_Destroy(tester);
    if (v_ctrl) JPH_VehicleControllerSettings_Destroy(v_ctrl);
    if (v_trans) JPH_VehicleTransmissionSettings_Destroy(v_trans);

    if (wheels) {
        for (uint32_t i = 0; i < wheel_count; i++) {
            if (wheels[i]) JPH_WheelSettings_Destroy(wheels[i]);
        }
        PyMem_RawFree(wheels);
    }

    if (f_curve) JPH_LinearCurve_Destroy(f_curve);
    if (t_curve) JPH_LinearCurve_Destroy(t_curve);

    Py_RETURN_NONE;
}

// --- Python Deallocation ---
static void Vehicle_dealloc(VehicleObject* self) {
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

  if (!self->world) goto finalize;

  // 1. REMOVE FROM JOLT MANAGER (Unlocked)
  // This is safe because the manager is thread-safe for removal.
  if (self->world->char_vs_char_manager && self->character) {
      JPH_CharacterVsCharacterCollisionSimple_RemoveCharacter(self->world->char_vs_char_manager, self->character);
  }

  // 2. WORLD REGISTRY CLEANUP (Locked)
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  SHADOW_LOCK(&self->world->shadow_lock);
  
  // GUARD: We must wait for queries to finish. Dealloc cannot return error,
  // so we must block. Since queries are fast, this is a very short wait.
  while (atomic_load_explicit(&self->world->active_queries, memory_order_acquire) > 0) {
      // Busy wait or yield.
  }

  uint32_t dense_idx = self->world->slot_to_dense[slot];
  uint32_t last_dense = (uint32_t)self->world->count - 1;
  
  if (dense_idx != last_dense) {
      // --- THE FIX: COMPLETE SWAP AND POP ---
      // Move EVERY shadow array element to maintain pack integrity.
      size_t dst = (size_t)dense_idx * 4;
      size_t src = (size_t)last_dense * 4;

      memcpy(&self->world->positions[dst],          &self->world->positions[src],          16);
      memcpy(&self->world->rotations[dst],          &self->world->rotations[src],          16);
      memcpy(&self->world->prev_positions[dst],     &self->world->prev_positions[src],     16);
      memcpy(&self->world->prev_rotations[dst],     &self->world->prev_rotations[src],     16);
      memcpy(&self->world->linear_velocities[dst],  &self->world->linear_velocities[src],  16);
      memcpy(&self->world->angular_velocities[dst], &self->world->angular_velocities[src], 16);
      
      self->world->body_ids[dense_idx] = self->world->body_ids[last_dense];
      self->world->user_data[dense_idx] = self->world->user_data[last_dense];

      // Update Indirection Map
      uint32_t mover_slot = self->world->dense_to_slot[last_dense];
      self->world->slot_to_dense[mover_slot] = dense_idx;
      self->world->dense_to_slot[dense_idx] = mover_slot;
  }
  
  // Recycle Slot
  self->world->generations[slot]++;
  self->world->free_slots[self->world->free_count++] = slot;
  self->world->slot_states[slot] = SLOT_EMPTY;
  self->world->count--;
  self->world->view_shape[0] = (Py_ssize_t)self->world->count;
  
  SHADOW_UNLOCK(&self->world->shadow_lock);

  // 3. JOLT RESOURCE DESTRUCTION (Unlocked)
  // No lock held here to avoid AB-BA deadlock with contact callbacks
  if (self->character) {
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)self->character);
  }
  if (self->listener) {
    JPH_CharacterContactListener_Destroy(self->listener);
  }
  if (self->body_filter)  JPH_BodyFilter_Destroy(self->body_filter);
  if (self->shape_filter) JPH_ShapeFilter_Destroy(self->shape_filter);
  if (self->bp_filter)    JPH_BroadPhaseLayerFilter_Destroy(self->bp_filter);
  if (self->obj_filter)   JPH_ObjectLayerFilter_Destroy(self->obj_filter);

finalize:
  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// NEW: GC Traverse/Clear for Character
static int Character_traverse(CharacterObject *self, visitproc visit, void *arg) {
    Py_VISIT(self->world);
    return 0;
}
static int Character_clear(CharacterObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

static PyObject *Character_move(CharacterObject *self, PyObject *args,
                                PyObject *kwds) {
  float vx = 0, vy = 0, vz = 0, dt = 0;
  static char *kwlist[] = {"velocity", "dt", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &vx, &vy, &vz, &dt)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  
  // 1. RE-ENTRANCY GUARD
  if (self->world->is_stepping) {
      SHADOW_UNLOCK(&self->world->shadow_lock);
      PyErr_SetString(PyExc_RuntimeError, "Cannot move character during physics step");
      return NULL;
  }

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
  self->prev_px = (float)current_pos->x; self->prev_py = (float)current_pos->y; self->prev_pz = (float)current_pos->z;
  self->prev_rx = current_rot->x; self->prev_ry = current_rot->y; self->prev_rz = current_rot->z; self->prev_rw = current_rot->w;

  // Sync to GLOBAL shadow buffers so get_render_state() works
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  uint32_t dense_idx = self->world->slot_to_dense[slot];
  size_t off = (size_t)dense_idx * 4;

  memcpy(&self->world->prev_positions[off], &self->world->positions[off], 12); // Copy x,y,z
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
  update_settings->walkStairsStepUp.y = 0.4f;  // Maximum step height
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

static PyObject *Character_get_position(CharacterObject *self, PyObject *Py_UNUSED(ignored)) {
  // 1. Aligned stack storage for SIMD
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  
  // 2. Lock for consistency (ensure we aren't reading mid-step)
  SHADOW_LOCK(&self->world->shadow_lock);
  JPH_CharacterVirtual_GetPosition(self->character, pos);
  SHADOW_UNLOCK(&self->world->shadow_lock);

  PyObject *ret = PyTuple_New(3);
  if (!ret) return NULL;

  // Use the double precision provided by RVec3
  PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble(pos->x));
  PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble(pos->y));
  PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(pos->z));

  return ret;
}

static PyObject *Character_set_position(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x, y, z;
  static char *kwlist[] = {"pos", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)", kwlist, &x, &y, &z)) return NULL;

  SHADOW_LOCK(&self->world->shadow_lock);
  GUARD_STEPPING(self->world);

  // 1. Update Jolt (Aligned)
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)x; pos->y = (double)y; pos->z = (double)z;
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
  float x, y, z, w;
  static char *kwlist[] = {"rot", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ffff)", kwlist, &x, &y, &z, &w)) return NULL;

  SHADOW_LOCK(&self->world->shadow_lock);
  GUARD_STEPPING(self->world);

  // 1. Update Jolt
  JPH_STACK_ALLOC(JPH_Quat, q);
  q->x = x; q->y = y; q->z = z; q->w = w;
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
  JPH_GroundState state = JPH_CharacterBase_GetGroundState((JPH_CharacterBase *)self->character);
  SHADOW_UNLOCK(&self->world->shadow_lock);

  if (state == JPH_GroundState_OnGround || state == JPH_GroundState_OnSteepGround) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *Character_set_strength(CharacterObject *self, PyObject *args) {
  float strength;
  if (!PyArg_ParseTuple(args, "f", &strength)) return NULL;

  SHADOW_LOCK(&self->world->shadow_lock);
  GUARD_STEPPING(self->world);

  // 1. Update Atomic for Jolt worker threads
  atomic_store_explicit(&self->push_strength, strength, memory_order_relaxed);

  // 2. Update Jolt internal state
  JPH_CharacterVirtual_SetMaxStrength(self->character, strength);

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

// Change signature to take PyObject* arg directly
static PyObject* Character_get_render_transform(CharacterObject* self, PyObject* arg) {
    // --- OPTIMIZATION 1: Fast Argument Parsing ---
    double alpha_dbl = PyFloat_AsDouble(arg);
    if (alpha_dbl == -1.0 && PyErr_Occurred()) return NULL;
    
    float alpha = (float)alpha_dbl;
    if (alpha < 0.0f) alpha = 0.0f; else if (alpha > 1.0f) alpha = 1.0f;

    // --- 1. ALIGNED STACK ALLOCATION ---
    // Mandatory for SIMD safety
    JPH_STACK_ALLOC(JPH_RVec3, cur_p);
    JPH_STACK_ALLOC(JPH_Quat, cur_r);

    // --- 2. CONSISTENT SNAPSHOT (Locked) ---
    SHADOW_LOCK(&self->world->shadow_lock);
    
    // We snapshot both the 'prev' state and 'current' state together
    float p_px = self->prev_px, p_py = self->prev_py, p_pz = self->prev_pz;
    float p_rx = self->prev_rx, p_ry = self->prev_ry, p_rz = self->prev_rz, p_rw = self->prev_rw;
    
    JPH_CharacterVirtual_GetPosition(self->character, cur_p);
    JPH_CharacterVirtual_GetRotation(self->character, cur_r);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // --- 3. MATH (Unlocked) ---
    // Position LERP
    float px = p_px + ((float)cur_p->x - p_px) * alpha;
    float py = p_py + ((float)cur_p->y - p_py) * alpha;
    float pz = p_pz + ((float)cur_p->z - p_pz) * alpha;

    // Rotation NLERP
    float dot = p_rx*cur_r->x + p_ry*cur_r->y + p_rz*cur_r->z + p_rw*cur_r->w;
    float q2x = cur_r->x, q2y = cur_r->y, q2z = cur_r->z, q2w = cur_r->w;
    if (dot < 0.0f) { q2x = -q2x; q2y = -q2y; q2z = -q2z; q2w = -q2w; }

    float rx = p_rx + (q2x - p_rx) * alpha;
    float ry = p_ry + (q2y - p_ry) * alpha;
    float rz = p_rz + (q2z - p_rz) * alpha;
    float rw = p_rw + (q2w - p_rw) * alpha;

    float mag_sq = rx*rx + ry*ry + rz*rz + rw*rw;
    if (mag_sq > 1e-9f) {
        float inv_len = 1.0f / sqrtf(mag_sq);
        rx *= inv_len; ry *= inv_len; rz *= inv_len; rw *= inv_len;
    } else {
        rx = 0.0f; ry = 0.0f; rz = 0.0f; rw = 1.0f;
    }

    // --- 4. HARDENED MANUAL CONSTRUCTION ---
    PyObject* pos = PyTuple_New(3);
    PyObject* rot = PyTuple_New(4);
    PyObject* out = PyTuple_New(2);

    if (!pos || !rot || !out) {
        Py_XDECREF(pos); Py_XDECREF(rot); Py_XDECREF(out);
        return PyErr_NoMemory();
    }

    // Safely fill tuples (PyFloat_FromDouble can fail, but unlikely)
    // If these return NULL, the whole return 'out' will be cleaned up by the user's logic
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

/* --- Immutable Getters (Safe without locks) --- */

static PyObject* Vehicle_get_wheel_count(VehicleObject* self, void* closure) {
    // num_wheels is set at creation and never changes
    return PyLong_FromUnsignedLong(self->num_wheels);
}

static PyObject* Character_get_handle(CharacterObject* self, void* closure) {
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
static void PhysicsWorld_releasebuffer(PhysicsWorldObject *self, Py_buffer *view) {
    SHADOW_LOCK(&self->shadow_lock);
    if (self->view_export_count > 0) self->view_export_count--;
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
    {"handle", (getter)Character_get_handle, NULL, "The unique physics handle for this character.", NULL},
    {NULL}
};

static const PyGetSetDef Vehicle_getset[] = {
    {"wheel_count", (getter)Vehicle_get_wheel_count, NULL, "Number of wheels attached to this vehicle.", NULL},
    {NULL}
};

static const PyMethodDef PhysicsWorld_methods[] = {
    // --- Lifecycle ---
    {"step", (PyCFunction)PhysicsWorld_step, METH_VARARGS, NULL},
    {"create_body", (PyCFunction)PhysicsWorld_create_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)PhysicsWorld_destroy_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_mesh_body", (PyCFunction)PhysicsWorld_create_mesh_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_constraint", (PyCFunction)PhysicsWorld_create_constraint, METH_VARARGS | METH_KEYWORDS, 
    "Create a constraint between two bodies. Params depend on type."},
    {"destroy_constraint", (PyCFunction)PhysicsWorld_destroy_constraint, METH_VARARGS | METH_KEYWORDS, 
    "Remove and destroy a constraint by handle."},
    {"create_vehicle", (PyCFunction)PhysicsWorld_create_vehicle, METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Interaction ---
    {"apply_impulse", (PyCFunction)PhysicsWorld_apply_impulse,
     METH_VARARGS | METH_KEYWORDS, NULL},
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

    // --- Motion Control ---
    {"get_motion_type", (PyCFunction)PhysicsWorld_get_motion_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_motion_type", (PyCFunction)PhysicsWorld_set_motion_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"activate", (PyCFunction)PhysicsWorld_activate,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"deactivate", (PyCFunction)PhysicsWorld_deactivate,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Queries ---
    {"raycast", (PyCFunction)PhysicsWorld_raycast, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"shapecast", (PyCFunction)PhysicsWorld_shapecast, METH_VARARGS | METH_KEYWORDS, 
    "Sweeps a shape along a direction vector. Returns (Handle, Fraction, ContactPoint, Normal) or None."},
    {"overlap_sphere", (PyCFunction)PhysicsWorld_overlap_sphere,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"overlap_aabb", (PyCFunction)PhysicsWorld_overlap_aabb,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Utilities ---
    {"get_index", (PyCFunction)PhysicsWorld_get_index,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_alive", (PyCFunction)PhysicsWorld_is_alive,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_active_indices", (PyCFunction)PhysicsWorld_get_active_indices, METH_NOARGS, 
     "Returns a bytes object containing uint32 indices of all active bodies."},
     {"get_render_state", (PyCFunction)PhysicsWorld_get_render_state, METH_VARARGS, 
     "Returns a packed bytes object of interpolated positions and rotations (3+4 floats per body)."},

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
    {"get_render_transform", (PyCFunction)Character_get_render_transform, METH_O, 
     "Returns interpolated ((x,y,z), (rx,ry,rz,rw)) based on alpha [0-1]."},
    {NULL}};

static const PyMethodDef Vehicle_methods[] = {
    {"set_input", (PyCFunction)Vehicle_set_input, METH_VARARGS | METH_KEYWORDS, 
     "Set driver inputs: forward [-1..1], right [-1..1], brake [0..1], handbrake [0..1]"},
    {"get_wheel_transform", (PyCFunction)Vehicle_get_wheel_transform, METH_VARARGS, 
     "Get world-space transform of a wheel by index."},
    {"get_wheel_local_transform", (PyCFunction)Vehicle_get_wheel_local_transform, METH_VARARGS, 
     "Get local-space transform of a wheel by index."},
    {"destroy", (PyCFunction)Vehicle_destroy, METH_NOARGS, "Manually remove the vehicle from physics."},
    {"get_debug_state", (PyCFunction)Vehicle_get_debug_state, METH_NOARGS, "Print drivetrain and wheel status to stderr"},
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

// --- Module Initialization ---

#define CREATE_TYPE(name) do {                        \
    st->name##Type = PyType_FromModuleAndSpec(        \
        m, (PyType_Spec *)&name##_spec, NULL);        \
    if (!st->name##Type) return -1;                   \
    if (PyModule_AddObject(m, #name, st->name##Type) < 0) { \
        Py_DECREF(st->name##Type);                    \
        return -1;                                   \
    }                                                 \
    Py_INCREF(st->name##Type);                        \
} while (0)

#define ADD_CONSTANT(name, value) do {                 \
    if (PyModule_AddIntConstant(m, #name, value) < 0) { \
        return -1;                                   \
    }                                                 \
} while (0)

static int culverin_exec(PyObject *m) {
  CulverinState *st = get_culverin_state(m);
  if (!JPH_Init()) {
    PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
    return -1;
  }

#if PY_VERSION_HEX < 0x030D0000
  self->shadow_lock = PyThread_allocate_lock();
  if (!self->shadow_lock) { PyErr_NoMemory(); goto fail; }
  g_jph_trampoline_lock = PyThread_allocate_lock();
  if (!g_jph_trampoline_lock) { PyErr_NoMemory(); goto fail; }
#endif

  st->helper = PyImport_ImportModule("culverin._culverin");
  if (!st->helper) return -1;

  CREATE_TYPE(PhysicsWorld);
  CREATE_TYPE(Character);
  CREATE_TYPE(Vehicle);

  ADD_CONSTANT(SHAPE_BOX, 0);
  ADD_CONSTANT(SHAPE_SPHERE, 1);
  ADD_CONSTANT(SHAPE_CAPSULE, 2);
  ADD_CONSTANT(SHAPE_CYLINDER, 3);
  ADD_CONSTANT(SHAPE_PLANE, 4);
  ADD_CONSTANT(SHAPE_MESH, 5);

  ADD_CONSTANT(MOTION_STATIC, 0);
  ADD_CONSTANT(MOTION_KINEMATIC, 1);
  ADD_CONSTANT(MOTION_DYNAMIC, 2);

  ADD_CONSTANT(CONSTRAINT_FIXED, 0);
  ADD_CONSTANT(CONSTRAINT_POINT, 1);
  ADD_CONSTANT(CONSTRAINT_HINGE, 2);
  ADD_CONSTANT(CONSTRAINT_SLIDER, 3);
  ADD_CONSTANT(CONSTRAINT_DISTANCE, 4);
  ADD_CONSTANT(CONSTRAINT_CONE, 5);

  return 0;
}

static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
  CulverinState *st = get_culverin_state(m);
  Py_VISIT(st->helper);
  Py_VISIT(st->PhysicsWorldType);
  Py_VISIT(st->CharacterType);
  Py_VISIT(st->VehicleType);
  return 0;
}

static int culverin_clear(PyObject *m) {
  CulverinState *st = get_culverin_state(m);
  Py_CLEAR(st->helper);
  Py_CLEAR(st->PhysicsWorldType);
  Py_CLEAR(st->CharacterType);
  Py_CLEAR(st->VehicleType);
  return 0;
}

static const PyModuleDef_Slot culverin_slots[] = {
    {Py_mod_exec, culverin_exec},
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};

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