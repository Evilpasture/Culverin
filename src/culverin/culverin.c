#include "culverin.h"

#include <math.h>
#include <stddef.h>

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


static bool JPH_API_CALL filter_true_body(void *userData, JPH_BodyID bodyID) { return true; }
static bool JPH_API_CALL filter_true_shape(void *userData, const JPH_Shape *shape, const JPH_SubShapeID *id) { return true; }

static const JPH_BodyFilter_Procs global_bf_procs = { .ShouldCollide = filter_true_body };
static const JPH_ShapeFilter_Procs global_sf_procs = { .ShouldCollide = filter_true_shape };

static void JPH_API_CALL char_on_character_contact_added(
    void *userData, const JPH_CharacterVirtual *character, const JPH_CharacterVirtual *otherCharacter,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {
    
    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;
}

// Callback: Handle the collision settings AND Apply Impulse
static void JPH_API_CALL char_on_contact_added(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {
  
  // 1. Safe Defaults
  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  // 2. Safety Checks
  if (!userData) return;
  CharacterObject *self = (CharacterObject *)userData;
  
  if (!self->world || !self->world->body_interface) return;
  JPH_BodyInterface *bi = self->world->body_interface;

  // 3. Ignore Sensors
  // Sensors shouldn't physically push back or be pushed by locomotion impulses
  if (JPH_BodyInterface_IsSensor(bi, bodyID2)) {
      ioSettings->canPushCharacter = false;
      ioSettings->canReceiveImpulses = false;
      return;
  }

  // 4. Only interact with Dynamic bodies
  if (JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
    return;
  }

  float vx = self->last_vx;
  float vy = self->last_vy;
  float vz = self->last_vz;

  // 5. Projection: Normal points FROM Character TO Body
  float dot = vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

  // 6. Apply Impulse with Thresholds
  // Threshold > 0.1f prevents micro-jitter when touching objects at rest
  if (dot > 0.1f) {
    float factor = dot * self->push_strength;
    
    // Safety Cap: Prevent physics explosions if velocity spikes
    const float max_impulse = 10000.0f; 
    if (factor > max_impulse) factor = max_impulse;

    JPH_Vec3 impulse;
    impulse.x = contactNormal->x * factor;

    // Flatten Y: Only allow upward pushes (lifting/kicking), ignore downward (crushing)
    // This prevents the character from applying massive force to the floor or objects they stand on
    float y_push = contactNormal->y * factor;
    impulse.y = (y_push > 0.0f) ? y_push : 0.0f;
    
    impulse.z = contactNormal->z * factor;

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

    // 1. Get User Data (Slots)
    uint64_t slot1 = JPH_Body_GetUserData((JPH_Body*)body1);
    uint64_t slot2 = JPH_Body_GetUserData((JPH_Body*)body2);

    // 2. Bound Check (Safety)
    if (slot1 >= self->slot_capacity || slot2 >= self->slot_capacity) return;

    // 3. Construct Handles (ID + Generation)
    BodyHandle h1 = make_handle((uint32_t)slot1, self->generations[slot1]);
    BodyHandle h2 = make_handle((uint32_t)slot2, self->generations[slot2]);

    SHADOW_LOCK(&self->shadow_lock);
    
    // 4. Resize Event Buffer if full
    if (self->contact_count >= self->contact_capacity) {
        size_t new_cap = (self->contact_capacity == 0) ? 64 : self->contact_capacity * 2;
        void* new_ptr = PyMem_RawRealloc(self->contact_events, new_cap * sizeof(ContactEvent));
        if (!new_ptr) { 
            SHADOW_UNLOCK(&self->shadow_lock); 
            return; 
        }
        self->contact_events = (ContactEvent*)new_ptr;
        self->contact_capacity = new_cap;
    }

    ContactEvent* ev = &self->contact_events[self->contact_count];
    ev->body1 = h1;
    ev->body2 = h2;

    // 5. Get Contact Normal
    JPH_Vec3 n;
    JPH_ContactManifold_GetWorldSpaceNormal(manifold, &n);
    ev->nx = n.x; ev->ny = n.y; ev->nz = n.z;

    // 6. Get Contact Point
    if (JPH_ContactManifold_GetPointCount(manifold) > 0) {
        JPH_RVec3 p;
        JPH_ContactManifold_GetWorldSpaceContactPointOn1(manifold, 0, &p);
        ev->px = (float)p.x; 
        ev->py = (float)p.y; 
        ev->pz = (float)p.z;
    } else {
        ev->px = 0.0f; ev->py = 0.0f; ev->pz = 0.0f;
    }

    // --- 7. CALCULATE IMPACT SPEED ---
    JPH_Vec3 v1, v2;
    JPH_Body_GetLinearVelocity((JPH_Body*)body1, &v1);
    JPH_Body_GetLinearVelocity((JPH_Body*)body2, &v2);

    float rv_x = v1.x - v2.x;
    float rv_y = v1.y - v2.y;
    float rv_z = v1.z - v2.z;

    float closing_speed = rv_x * n.x + rv_y * n.y + rv_z * n.z;
    ev->impulse = fabsf(closing_speed);

    self->contact_count++;
    SHADOW_UNLOCK(&self->shadow_lock);
}

// Fixed get_contact_events to be safer with locking
static PyObject* PhysicsWorld_get_contact_events(PhysicsWorldObject* self, PyObject* args) {
    // 1. Snapshot size
    SHADOW_LOCK(&self->shadow_lock);
    if (self->contact_count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyList_New(0);
    }
    
    size_t count = self->contact_count;
    ContactEvent* scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
    if (!scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }
    
    memcpy(scratch, self->contact_events, count * sizeof(ContactEvent));
    SHADOW_UNLOCK(&self->shadow_lock);

    // 2. Build Python list OUTSIDE lock
    PyObject* list = PyList_New((Py_ssize_t)count);
    if (!list) {
        PyMem_RawFree(scratch);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        PyObject* v1 = PyLong_FromUnsignedLongLong(scratch[i].body1);
        PyObject* v2 = PyLong_FromUnsignedLongLong(scratch[i].body2);
        if (!v1 || !v2) {
            Py_XDECREF(v1); Py_XDECREF(v2); Py_DECREF(list);
            PyMem_RawFree(scratch);
            return NULL;
        }

        PyObject* item = PyTuple_New(2);
        PyTuple_SET_ITEM(item, 0, v1);
        PyTuple_SET_ITEM(item, 1, v2);
        PyList_SET_ITEM(list, (Py_ssize_t)i, item);
    }

    PyMem_RawFree(scratch);
    return list;
}

static PyObject* PhysicsWorld_get_contact_events_ex(PhysicsWorldObject* self, PyObject* args) {
    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->contact_count;
    if (count == 0) { 
        SHADOW_UNLOCK(&self->shadow_lock); 
        return PyList_New(0); 
    }

    ContactEvent* scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
    if (!scratch) { 
        SHADOW_UNLOCK(&self->shadow_lock); 
        return PyErr_NoMemory(); 
    }
    
    memcpy(scratch, self->contact_events, count * sizeof(ContactEvent));
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- OPTIMIZATION 1: Pre-allocate Keys ---
    // Using InternFromString is slightly faster for subsequent lookups
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
        // Cleanup keys and scratch on failure
        Py_DECREF(k_bodies); Py_DECREF(k_pos); 
        Py_DECREF(k_norm); Py_DECREF(k_str);
        PyMem_RawFree(scratch);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEvent* e = &scratch[i];
        
        // --- OPTIMIZATION 2: Direct Dict Construction ---
        PyObject* dict = PyDict_New();
        if (!dict) continue; // Should handle error, but skip for speed/simplicity here

        // 1. Bodies Tuple (u64, u64)
        PyObject* b_tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(b_tuple, 0, PyLong_FromUnsignedLongLong(e->body1));
        PyTuple_SET_ITEM(b_tuple, 1, PyLong_FromUnsignedLongLong(e->body2));
        PyDict_SetItem(dict, k_bodies, b_tuple); 
        Py_DECREF(b_tuple); // Dict_SetItem increments ref, so we drop ours

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

        // 4. Strength Float
        PyObject* s_val = PyFloat_FromDouble(e->impulse);
        PyDict_SetItem(dict, k_str, s_val);
        Py_DECREF(s_val);

        // Add to List (Steals Reference)
        PyList_SET_ITEM(list, (Py_ssize_t)i, dict);
    }

    // Cleanup Keys
    Py_DECREF(k_bodies);
    Py_DECREF(k_pos);
    Py_DECREF(k_norm);
    Py_DECREF(k_str);

    PyMem_RawFree(scratch);
    return list;
}

// ContactEvent layout (packed, little-endian):
// uint64 body1
// uint64 body2
// float32 px, py, pz
// float32 nx, ny, nz
// float32 impulse
static PyObject* PhysicsWorld_get_contact_events_raw(PhysicsWorldObject* self, PyObject* args) {
    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->contact_count;
    if (count == 0) { SHADOW_UNLOCK(&self->shadow_lock); return PyMemoryView_FromObject(PyBytes_FromStringAndSize("", 0)); }

    size_t bytes_size = count * sizeof(ContactEvent);
    PyObject* raw_bytes = PyBytes_FromStringAndSize((char*)self->contact_events, (Py_ssize_t)bytes_size);
    SHADOW_UNLOCK(&self->shadow_lock);

    if (!raw_bytes) return NULL;
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
  ShapeKey key = {(uint32_t)type, params[0], params[1], params[2], params[3]};
  for (size_t i = 0; i < self->shape_cache_count; i++) {
    ShapeKey *k = &self->shape_cache[i].key;
    if (k->type == key.type && k->p1 == key.p1 && k->p2 == key.p2 &&
        k->p3 == key.p3 && k->p4 == key.p4) {
      return self->shape_cache[i].shape;
    }
  }

  JPH_Shape *shape = NULL;
  if (type == 0) { // BOX
    JPH_Vec3 he = {key.p1, key.p2, key.p3};
    JPH_BoxShapeSettings *s = JPH_BoxShapeSettings_Create(&he, 0.05f);
    shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 1) { // SPHERE
    JPH_SphereShapeSettings *s = JPH_SphereShapeSettings_Create(key.p1);
    shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 2) { // CAPSULE
    JPH_CapsuleShapeSettings *s =
        JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
    shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 3) { // CYLINDER
    JPH_CylinderShapeSettings *s =
        JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
    shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 4) { // PLANE
    JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
    JPH_PlaneShapeSettings *s =
        JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
    shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  }

  if (!shape) return NULL;

  if (self->shape_cache_count >= self->shape_cache_capacity) {
    size_t new_cap = (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
    void *new_ptr = PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
    if (!new_ptr) {
      JPH_Shape_Destroy(shape);
      PyErr_NoMemory();
      return NULL;
    }
    self->shape_cache = (ShapeEntry *)new_ptr;
    self->shape_cache_capacity = new_cap;
  }

  self->shape_cache[self->shape_cache_count].key = key;
  self->shape_cache[self->shape_cache_count].shape = shape;
  self->shape_cache_count++;

  return shape;
}

// --- Helper: Resource Cleanup (Idempotent) ---
static void PhysicsWorld_free_members(PhysicsWorldObject *self) {
  // 1. Jolt Systems
  // Note: Destroy order matters. System uses the filters/interfaces, so destroy System first.
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

  // 2. Filters & Interfaces
  // WARNING: joltc.h does not expose Destroy functions for these tables.
  // This implies a memory leak in the C bindings for these specific objects.
  // We set them to NULL to prevent use-after-free logic, but we cannot free the C++ memory.
  // Best to assume that Jolt hopefully cleans them up when the program exits.
  if (self->bp_interface) { 
      // JPH_BroadPhaseLayerInterface_Destroy(self->bp_interface); // Missing in API
      self->bp_interface = NULL; 
  }
  if (self->pair_filter) { 
      // JPH_ObjectLayerPairFilter_Destroy(self->pair_filter); // Missing in API
      self->pair_filter = NULL; 
  }
  if (self->bp_filter) { 
      // JPH_ObjectVsBroadPhaseLayerFilter_Destroy(self->bp_filter); // Missing in API
      self->bp_filter = NULL; 
  }

  // 3. Shape Cache
  if (self->shape_cache) {
    for (size_t i = 0; i < self->shape_cache_count; i++) {
      if (self->shape_cache[i].shape) {
        JPH_Shape_Destroy(self->shape_cache[i].shape);
      }
    }
    PyMem_RawFree(self->shape_cache);
    self->shape_cache = NULL;
  }

  // 4. Contact Listener & Events
  if (self->contact_listener) { 
      JPH_ContactListener_Destroy(self->contact_listener); 
      self->contact_listener = NULL;
  }
  if (self->contact_events) { 
      PyMem_RawFree(self->contact_events); 
      self->contact_events = NULL;
  }

  // 5. Constraints
  if (self->constraints) {
    for(size_t i=0; i<self->constraint_capacity; i++) {
        if(self->constraints[i]) {
            JPH_Constraint_Destroy(self->constraints[i]);
            self->constraints[i] = NULL;
        }
    }
    PyMem_RawFree((void *)self->constraints);
    self->constraints = NULL;
  }
  if (self->constraint_generations) { PyMem_RawFree(self->constraint_generations); self->constraint_generations = NULL; }
  if (self->free_constraint_slots) { PyMem_RawFree(self->free_constraint_slots); self->free_constraint_slots = NULL; }
  if (self->constraint_states) { PyMem_RawFree(self->constraint_states); self->constraint_states = NULL; }

  // 6. Data Arrays
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
  
  // Clean Locals for fail label
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
  
  // Ensure pointers are NULL so free_members works correctly on fail
  self->system = NULL;
  self->job_system = NULL;
  self->bp_interface = NULL;
  self->pair_filter = NULL;
  self->bp_filter = NULL;
  self->contact_listener = NULL;
  self->contact_events = NULL;
  self->char_vs_char_manager = NULL;
  self->positions = NULL; // etc... (tp_alloc usually zeros, but be safe)

  PyObject *module = PyType_GetModule(Py_TYPE(self));
  if (!module) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to get module");
    return -1;
  }

  CulverinState *st = get_culverin_state(module);
  if (!st) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to get module state");
    return -1;
  }

  val_func = PyObject_GetAttrString(st->helper, "validate_settings");
  if (!val_func) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to get settings validator");
    goto fail;
  }

  norm_settings = PyObject_CallFunctionObjArgs(
      val_func, settings_dict ? settings_dict : Py_None, NULL);
  if (!norm_settings) goto fail;

  float gx = NAN, gy = NAN, gz = NAN, slop = NAN;
  int max_bodies = 0, max_pairs = 0;
  
  if (!PyArg_ParseTuple(norm_settings, "ffffii", &gx, &gy, &gz, &slop,
                      &max_bodies, &max_pairs)) {
    PyErr_SetString(PyExc_RuntimeError, "validate_settings returned invalid data");
    goto fail;
  }
  Py_CLEAR(norm_settings); 

  // 2. Jolt Systems Initialization
  JobSystemThreadPoolConfig job_cfg = {
      .maxJobs = 1024, .maxBarriers = 8, .numThreads = -1};
  self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
  if (!self->job_system) { PyErr_SetString(PyExc_RuntimeError, "Failed init JobSystem"); goto fail; }

  self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(2, 2);
  if (!self->bp_interface) { PyErr_NoMemory(); goto fail; }

  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 0, 0);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 1, 1);

  self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(2);
  if (!self->pair_filter) { PyErr_NoMemory(); goto fail; }
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 0);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1);
  
  self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(
      self->bp_interface, 2, self->pair_filter, 2);
  if (!self->bp_filter) { PyErr_NoMemory(); goto fail; }

  JPH_PhysicsSystemSettings phys_settings = {
      .maxBodies = 1000000,
      .maxBodyPairs = 1000000,
      .maxContactConstraints = 102400,
      .broadPhaseLayerInterface = self->bp_interface,
      .objectLayerPairFilter = self->pair_filter,
      .objectVsBroadPhaseLayerFilter = self->bp_filter};
  
  self->system = JPH_PhysicsSystem_Create(&phys_settings);
  if (!self->system) { PyErr_SetString(PyExc_RuntimeError, "Failed init PhysicsSystem"); goto fail; }

  self->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
  if (!self->char_vs_char_manager) { PyErr_NoMemory(); goto fail; }

  JPH_Vec3 gravity = {gx, gy, gz};
  JPH_PhysicsSystem_SetGravity(self->system, &gravity);
  self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);

  // 3. Contact Events & Listener
  self->contact_capacity = 64;
  self->contact_events = PyMem_RawMalloc(self->contact_capacity * sizeof(ContactEvent));
  if (!self->contact_events) { PyErr_NoMemory(); goto fail; }

  JPH_ContactListener_SetProcs(&contact_procs);
  self->contact_listener = JPH_ContactListener_Create(self); 
  if (!self->contact_listener) { PyErr_NoMemory(); goto fail; }
  JPH_PhysicsSystem_SetContactListener(self->system, self->contact_listener);

#if PY_VERSION_HEX < 0x030D0000
  self->shadow_lock = PyThread_allocate_lock();
  if (!self->shadow_lock) { PyErr_NoMemory(); goto fail; }
#endif

  // 4. ABI Check
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

  // 5. Bake Scene (Optional)
  size_t baked_count = 0;
  if (bodies_list && bodies_list != Py_None) {
    bake_func = PyObject_GetAttrString(st->helper, "bake_scene");
    if (!bake_func) { PyErr_SetString(PyExc_RuntimeError, "Missing bake_scene"); goto fail; }
    
    baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
    if (!baked) goto fail; // Exception set by call

    // --- Hardened Validation of bake_scene output ---
    if (!PyTuple_Check(baked) || PyTuple_Size(baked) < 7) {
        PyErr_SetString(PyExc_RuntimeError, "bake_scene returned invalid tuple (size < 7)");
        goto fail;
    }
    
    // Check Count
    PyObject* py_count = PyTuple_GetItem(baked, 0);
    if (!PyLong_Check(py_count)) {
        PyErr_SetString(PyExc_TypeError, "bake_scene count is not an integer");
        goto fail;
    }
    baked_count = PyLong_AsSize_t(py_count);
    if (PyErr_Occurred()) goto fail;

    // Check Buffers Types and Sizes
    struct { int idx; size_t stride; const char* name; } checks[] = {
        {1, 16, "positions"}, {2, 16, "rotations"}, {3, 20, "shapes"},
        {4, 1, "motion"}, {5, 1, "layer"}, {6, 8, "userdata"}
    };

    for (int k=0; k<6; k++) {
        PyObject* buf = PyTuple_GetItem(baked, checks[k].idx);
        if (!PyBytes_Check(buf)) {
            PyErr_Format(PyExc_TypeError, "bake_scene item %d (%s) is not bytes", k+1, checks[k].name);
            goto fail;
        }
        if (PyBytes_Size(buf) < (Py_ssize_t)(baked_count * checks[k].stride)) {
            PyErr_Format(PyExc_ValueError, "bake_scene buffer %s too small for count %zu", checks[k].name, baked_count);
            goto fail;
        }
    }
  }
  Py_CLEAR(bake_func); 

  if (max_bodies <= 0 || max_bodies > 1000000) {
    PyErr_SetString(PyExc_ValueError, "max_bodies out of sane range");
    goto fail;
  }

  self->count = baked_count;
  self->capacity = (size_t)max_bodies;
  if (self->capacity < 128) self->capacity = 128;
  if (self->capacity < self->count) self->capacity = self->count + 1024;

  // 6. Allocations (Native Buffers)
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

  // Allocation Check
  if (!self->positions || !self->rotations || !self->body_ids ||
      !self->linear_velocities || !self->angular_velocities ||
      !self->user_data || !self->generations || !self->slot_to_dense ||
      !self->dense_to_slot || !self->free_slots || !self->slot_states ||
      !self->command_queue || !self->prev_positions || !self->prev_rotations ||
      !self->constraints || !self->constraint_generations ||
      !self->free_constraint_slots || !self->constraint_states) {
    PyErr_NoMemory();
    goto fail;
  }

  // Init Constraint Slots
  for(uint32_t i=0; i<self->constraint_capacity; i++) {
      self->constraint_generations[i] = 1;
      self->free_constraint_slots[i] = i;
  }
  self->free_constraint_count = self->constraint_capacity;

  // 7. Apply Baked Data
  if (baked) {
    // Pointers are safe due to earlier checks
    float *f_pos = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 1));
    float *f_rot = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 2));
    float *f_shape = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 3));
    unsigned char *u_mot = (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 4));
    unsigned char *u_layer = (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 5));
    uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, 6));

    JPH_BodyInterface *bi = self->body_interface;

    for (size_t i = 0; i < self->count; i++) {
      JPH_STACK_ALLOC(JPH_RVec3, body_pos);
      body_pos->x = (double)f_pos[i * 4];
      body_pos->y = (double)f_pos[i * 4 + 1];
      body_pos->z = (double)f_pos[i * 4 + 2];

      JPH_STACK_ALLOC(JPH_Quat, body_rot);
      body_rot->x = f_rot[i * 4];
      body_rot->y = f_rot[i * 4 + 1];
      body_rot->z = f_rot[i * 4 + 2];
      body_rot->w = f_rot[i * 4 + 3];

      float params[4] = {f_shape[i * 5 + 1], f_shape[i * 5 + 2],
                         f_shape[i * 5 + 3], f_shape[i * 5 + 4]};
      
      JPH_Shape *shape = find_or_create_shape(self, (int)f_shape[i * 5], params);

      if (shape) {
        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
            shape, body_pos, body_rot, (JPH_MotionType)u_mot[i],
            (JPH_ObjectLayer)u_layer[i]);

        if (!creation) {
            // Partial Failure: Log it, mark invalid, but continue
            DEBUG_LOG("Warning: Failed to create settings for body %zu", i);
            self->body_ids[i] = JPH_INVALID_BODY_ID;
            continue;
        }

        JPH_BodyCreationSettings_SetUserData(creation, (uint64_t)i);
        if (u_mot[i] == 2) {
          JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
        }

        self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(
            bi, creation, JPH_Activation_Activate);
        JPH_BodyCreationSettings_Destroy(creation);

        self->generations[i] = 1;
        self->slot_to_dense[i] = (uint32_t)i;
        self->dense_to_slot[i] = (uint32_t)i;
        self->slot_states[i] = SLOT_ALIVE;
        self->user_data[i] = u_data[i];
      } else {
        DEBUG_LOG("Warning: Failed to create shape for body %zu", i);
        self->body_ids[i] = JPH_INVALID_BODY_ID;
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
  PhysicsWorld_free_members(self); // Clean native resources explicitly
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
  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
  JPH_Vec3 imp = {x, y, z};
  JPH_BodyInterface_AddImpulse(self->body_interface,
                               self->body_ids[self->slot_to_dense[slot]], &imp);
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    float sx, sy, sz;
    float dx, dy, dz;
    float max_dist = 1000.0f;
    uint64_t ignore_h = 0; // NEW: Optional handle to ignore
    static char *kwlist[] = {"start", "direction", "max_dist", "ignore", NULL};

    // Updated format string to "|fK" for max_dist and ignore_h
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|fK", kwlist, 
                                     &sx, &sy, &sz, 
                                     &dx, &dy, &dz, 
                                     &max_dist, &ignore_h)) {
        return NULL;
    }

    float mag_sq = dx*dx + dy*dy + dz*dz;
    if (mag_sq < 1e-9f) Py_RETURN_NONE;
    
    float mag = sqrtf(mag_sq);
    float scale = max_dist / mag;

    JPH_STACK_ALLOC(JPH_RVec3, origin);
    memset(origin, 0, sizeof(JPH_RVec3));
    origin->x = sx; origin->y = sy; origin->z = sz;

    JPH_STACK_ALLOC(JPH_Vec3, direction);
    direction->x = dx * scale;
    direction->y = dy * scale;
    direction->z = dz * scale;

    JPH_STACK_ALLOC(JPH_RayCastResult, hit);
    memset(hit, 0, sizeof(JPH_RayCastResult));

    // --- NEW: Setup Ignore Filter ---
    JPH_BodyID ignore_bid = 0;
    if (ignore_h != 0) {
        uint32_t ignore_slot;
        SHADOW_LOCK(&self->shadow_lock);
        if (unpack_handle(self, ignore_h, &ignore_slot)) {
            ignore_bid = self->body_ids[self->slot_to_dense[ignore_slot]];
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }

    // Reuse the filter context/logic we defined for shapecast
    CastShapeFilter filter_ctx = { .ignore_id = ignore_bid };
    JPH_BodyFilter_Procs filter_procs = { .ShouldCollide = CastShape_BodyFilter };
    JPH_BodyFilter* body_filter = JPH_BodyFilter_Create(&filter_ctx);
    JPH_BodyFilter_SetProcs(&filter_procs);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    
    // Pass the body_filter as the last argument
    bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, NULL, NULL, body_filter);

    JPH_BodyFilter_Destroy(body_filter);

    if (!has_hit) Py_RETURN_NONE;

    // --- Normal Extraction ---
    JPH_Vec3 normal = {0, 1, 0}; 
    const JPH_BodyLockInterface* lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockRead lock;
    JPH_BodyLockInterface_LockRead(lock_iface, hit->bodyID, &lock);

    if (lock.body) {
        JPH_RVec3 hit_pos;
        hit_pos.x = origin->x + direction->x * hit->fraction;
        hit_pos.y = origin->y + direction->y * hit->fraction;
        hit_pos.z = origin->z + direction->z * hit->fraction;
        JPH_Body_GetWorldSpaceSurfaceNormal(lock.body, hit->subShapeID2, &hit_pos, &normal);
    }
    JPH_BodyLockInterface_UnlockRead(lock_iface, &lock);

    // --- Resolve Handle ---
    SHADOW_LOCK(&self->shadow_lock);
    uint64_t slot_idx = JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
    
    if (slot_idx >= self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    uint32_t gen = self->generations[slot_idx];
    BodyHandle handle = make_handle((uint32_t)slot_idx, gen);
    SHADOW_UNLOCK(&self->shadow_lock);

    // Faster return construction
    PyObject* norm_tuple = PyTuple_New(3);
    PyTuple_SET_ITEM(norm_tuple, 0, PyFloat_FromDouble(normal.x));
    PyTuple_SET_ITEM(norm_tuple, 1, PyFloat_FromDouble(normal.y));
    PyTuple_SET_ITEM(norm_tuple, 2, PyFloat_FromDouble(normal.z));

    PyObject* result = PyTuple_New(3);
    PyTuple_SET_ITEM(result, 0, PyLong_FromUnsignedLongLong(handle));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(hit->fraction));
    PyTuple_SET_ITEM(result, 2, norm_tuple);

    return result;
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
    uint64_t ignore_h = 0; // NEW: Optional body to ignore
    static char *kwlist[] = {"shape", "pos", "rot", "dir", "size", "ignore", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "i(fff)(ffff)(fff)O|K", kwlist, 
                                     &shape_type, &px, &py, &pz, 
                                     &rx, &ry, &rz, &rw, 
                                     &dx, &dy, &dz, &py_size, &ignore_h)) {
        return NULL;
    }

    // 1. Prepare Shape
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

    JPH_Shape* shape = find_or_create_shape(self, shape_type, s);
    if (!shape) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid shape for shapecast");
        return NULL;
    }

    // 2. Setup Transform & Direction
    JPH_STACK_ALLOC(JPH_RMat4, transform);
    JPH_Quat q = {rx, ry, rz, rw};
    JPH_RVec3 p = {(double)px, (double)py, (double)pz};
    JPH_RMat4_RotationTranslation(transform, &q, &p);
    
    JPH_Vec3 sweep_dir = {dx, dy, dz};

    // 3. Settings & Context
    JPH_STACK_ALLOC(JPH_ShapeCastSettings, settings);
    JPH_ShapeCastSettings_Init(settings);
    // CRITICAL: Ignore back-faces to avoid getting stuck inside geometry
    settings->backFaceModeTriangles = JPH_BackFaceMode_IgnoreBackFaces;
    settings->backFaceModeConvex = JPH_BackFaceMode_IgnoreBackFaces;

    CastShapeContext ctx = { .has_hit = false };
    ctx.hit.fraction = 1.0f;

    // 4. Setup Ignore Filter
    CastShapeFilter filter_ctx = { .ignore_id = 0 };
    if (ignore_h != 0) {
        uint32_t ignore_slot;
        if (unpack_handle(self, ignore_h, &ignore_slot)) {
            filter_ctx.ignore_id = self->body_ids[self->slot_to_dense[ignore_slot]];
        }
    }

    JPH_BodyFilter_Procs filter_procs = { .ShouldCollide = CastShape_BodyFilter };
    JPH_BodyFilter* body_filter = JPH_BodyFilter_Create(&filter_ctx);
    JPH_BodyFilter_SetProcs(&filter_procs);

    // 5. Execute
    JPH_RVec3 base_offset = {0, 0, 0};
    const JPH_NarrowPhaseQuery* nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    
    JPH_NarrowPhaseQuery_CastShape(nq, shape, transform, &sweep_dir, settings, &base_offset, 
                                   CastShape_ClosestCollector, &ctx, 
                                   NULL, NULL, body_filter, NULL);

    JPH_BodyFilter_Destroy(body_filter);

    if (!ctx.has_hit) Py_RETURN_NONE;

    // 6. Handle & Normal Unpacking
    SHADOW_LOCK(&self->shadow_lock);
    uint64_t slot_idx = JPH_BodyInterface_GetUserData(self->body_interface, ctx.hit.bodyID2);
    if (slot_idx >= self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }
    BodyHandle handle = make_handle((uint32_t)slot_idx, self->generations[slot_idx]);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 7. Calculate Normalized Surface Normal
    // Jolt penetrationAxis points from Body2 to Shape1. Negate for Surface Normal.
    float nx = -ctx.hit.penetrationAxis.x;
    float ny = -ctx.hit.penetrationAxis.y;
    float nz = -ctx.hit.penetrationAxis.z;
    float n_len = sqrtf(nx*nx + ny*ny + nz*nz);
    if (n_len > 1e-6f) {
        float inv = 1.0f / n_len;
        nx *= inv; ny *= inv; nz *= inv;
    }

    PyObject* contact = Py_BuildValue("fff", (float)ctx.hit.contactPointOn2.x, 
                                             (float)ctx.hit.contactPointOn2.y, 
                                             (float)ctx.hit.contactPointOn2.z);
    PyObject* normal = Py_BuildValue("fff", nx, ny, nz);
    
    PyObject* result = PyTuple_New(4);
    PyTuple_SET_ITEM(result, 0, PyLong_FromUnsignedLongLong(handle));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(ctx.hit.fraction));
    PyTuple_SET_ITEM(result, 2, contact);
    PyTuple_SET_ITEM(result, 3, normal);

    return result;
}

// Helper to grow queue
static bool ensure_command_capacity(PhysicsWorldObject *self) {
  if (self->command_count >= self->command_capacity) {
    size_t new_cap = self->command_capacity * 2;
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

      JPH_BodyID new_bid =
          JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);

      size_t new_dense = self->count;
      self->body_ids[new_dense] = new_bid;
      self->slot_to_dense[slot] = (uint32_t)new_dense;
      self->dense_to_slot[new_dense] = slot;
      self->user_data[new_dense] = cmd->data.create.user_data;

      JPH_STACK_ALLOC(JPH_RVec3, p);
      JPH_STACK_ALLOC(JPH_Quat, q);
      JPH_BodyInterface_GetPosition(bi, new_bid, p);
      JPH_BodyInterface_GetRotation(bi, new_bid, q);

      self->positions[new_dense * 4 + 0] = (float)p->x;
      self->positions[new_dense * 4 + 1] = (float)p->y;
      self->positions[new_dense * 4 + 2] = (float)p->z;
      self->positions[new_dense * 4 + 3] = 0.0f;

      self->rotations[new_dense * 4 + 0] = q->x;
      self->rotations[new_dense * 4 + 1] = q->y;
      self->rotations[new_dense * 4 + 2] = q->z;
      self->rotations[new_dense * 4 + 3] = q->w;

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
        memcpy(&self->positions[(size_t)dense_idx * 4],
               &self->positions[last_dense * 4], 16);
        memcpy(&self->rotations[(size_t)dense_idx * 4],
               &self->rotations[last_dense * 4], 16);
        memcpy(&self->linear_velocities[(size_t)dense_idx * 4],
               &self->linear_velocities[last_dense * 4], 16);
        memcpy(&self->angular_velocities[(size_t)dense_idx * 4],
               &self->angular_velocities[last_dense * 4], 16);
        self->body_ids[dense_idx] = self->body_ids[last_dense];

        uint32_t mover_slot = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = dense_idx;
        self->dense_to_slot[dense_idx] = mover_slot;
        self->user_data[dense_idx] = self->user_data[last_dense];
      }

      self->generations[slot]++;
      self->free_slots[self->free_count++] = slot;
      self->slot_states[slot] = SLOT_EMPTY;
      self->count--;
      break;
    }

    case CMD_SET_POS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->data.vec.x;
      p->y = cmd->data.vec.y;
      p->z = cmd->data.vec.z;
      JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);
      self->positions[dense_idx * 4 + 0] = cmd->data.vec.x;
      self->positions[dense_idx * 4 + 1] = cmd->data.vec.y;
      self->positions[dense_idx * 4 + 2] = cmd->data.vec.z;
      break;
    }

    case CMD_SET_ROT: {
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->data.vec.x;
      q->y = cmd->data.vec.y;
      q->z = cmd->data.vec.z;
      q->w = cmd->data.vec.w;
      JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);
      self->rotations[dense_idx * 4 + 0] = q->x;
      self->rotations[dense_idx * 4 + 1] = q->y;
      self->rotations[dense_idx * 4 + 2] = q->z;
      self->rotations[dense_idx * 4 + 3] = q->w;
      break;
    }

    case CMD_SET_TRNS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->data.transform.px;
      p->y = cmd->data.transform.py;
      p->z = cmd->data.transform.pz;
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->data.transform.rx;
      q->y = cmd->data.transform.ry;
      q->z = cmd->data.transform.rz;
      q->w = cmd->data.transform.rw;

      JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q,
                                               JPH_Activation_Activate);

      self->positions[dense_idx * 4 + 0] = (float)p->x;
      self->positions[dense_idx * 4 + 1] = (float)p->y;
      self->positions[dense_idx * 4 + 2] = (float)p->z;
      self->rotations[dense_idx * 4 + 0] = q->x;
      self->rotations[dense_idx * 4 + 1] = q->y;
      self->rotations[dense_idx * 4 + 2] = q->z;
      self->rotations[dense_idx * 4 + 3] = q->w;
      break;
    }

    case CMD_SET_LINVEL: {
      JPH_Vec3 v = {cmd->data.vec.x, cmd->data.vec.y, cmd->data.vec.z};
      JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);
      self->linear_velocities[dense_idx * 4 + 0] = v.x;
      self->linear_velocities[dense_idx * 4 + 1] = v.y;
      self->linear_velocities[dense_idx * 4 + 2] = v.z;
      break;
    }

    case CMD_SET_ANGVEL: {
      JPH_Vec3 v = {cmd->data.vec.x, cmd->data.vec.y, cmd->data.vec.z};
      JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);
      self->angular_velocities[dense_idx * 4 + 0] = v.x;
      self->angular_velocities[dense_idx * 4 + 1] = v.y;
      self->angular_velocities[dense_idx * 4 + 2] = v.z;
      break;
    }

    case CMD_SET_MOTION: {
      JPH_BodyInterface_SetMotionType(bi, bid,
                                      (JPH_MotionType)cmd->data.motion_type,
                                      JPH_Activation_Activate);
      break;
    }

    case CMD_ACTIVATE: {
      JPH_BodyInterface_ActivateBody(bi, bid);
      break;
    }

    case CMD_DEACTIVATE: {
      JPH_BodyInterface_DeactivateBody(bi, bid);
      break;
    }
    case CMD_SET_USER_DATA: {
      uint32_t dense_idx = self->slot_to_dense[slot];
      self->user_data[dense_idx] = cmd->data.user_data_val;
      break;
    }
    default: {
      continue;
    }
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
    JPH_Vec3_Normalize(&axis, &axis);
    
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
    JPH_Vec3_Normalize(&axis, &axis);
    
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
    JPH_Vec3_Normalize(&axis, &axis);

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
    
    // Auto-detect anchor points based on current body positions
    JPH_Body_GetPosition(b1, &s.point1);
    JPH_Body_GetPosition(b2, &s.point2);
    
    s.minDistance = p->limit_min;
    s.maxDistance = p->limit_max;

    return (JPH_Constraint*)JPH_DistanceConstraint_Create(&s, b1, b2);
}

static int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
  // NEW: Safe Resize Check
  if (self->view_export_count > 0) {
      PyErr_SetString(PyExc_BufferError, "Cannot resize world while memory views are exported.");
      return -1;
  }

  if (new_capacity <= self->capacity) return 0;

  self->positions = PyMem_RawRealloc(self->positions, new_capacity * 16);
  self->rotations = PyMem_RawRealloc(self->rotations, new_capacity * 16);
  self->prev_positions = PyMem_RawRealloc(self->prev_positions, new_capacity * 16);
  self->prev_rotations = PyMem_RawRealloc(self->prev_rotations, new_capacity * 16);
  self->linear_velocities = PyMem_RawRealloc(self->linear_velocities, new_capacity * 16);
  self->angular_velocities = PyMem_RawRealloc(self->angular_velocities, new_capacity * 16);
  self->body_ids = PyMem_RawRealloc(self->body_ids, new_capacity * sizeof(JPH_BodyID));
  self->user_data = PyMem_RawRealloc(self->user_data, new_capacity * sizeof(uint64_t));

  self->generations = PyMem_RawRealloc(self->generations, new_capacity * sizeof(uint32_t));
  self->slot_to_dense = PyMem_RawRealloc(self->slot_to_dense, new_capacity * sizeof(uint32_t));
  self->dense_to_slot = PyMem_RawRealloc(self->dense_to_slot, new_capacity * sizeof(uint32_t));
  self->slot_states = PyMem_RawRealloc(self->slot_states, new_capacity * sizeof(uint8_t));

  uint32_t *new_free_slots = PyMem_RawRealloc(self->free_slots, new_capacity * sizeof(uint32_t));

  if (!self->positions || !self->rotations || !self->prev_positions || !self->prev_rotations || !new_free_slots) {
    PyErr_NoMemory();
    return -1;
  }
  self->free_slots = new_free_slots;

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

    DEBUG_LOG("Creating Constraint Type %d between %llu and %llu", type, h1, h2);

    // --- 1. Parse Parameters ---
    ConstraintParams p;
    params_init(&p);
    int parse_ok = 1;

    switch (type) {
        case CONSTRAINT_FIXED:    /* No params */ break;
        case CONSTRAINT_POINT:    parse_ok = parse_point_params(params, &p); break;
        case CONSTRAINT_HINGE:    parse_ok = parse_hinge_params(params, &p); break;
        case CONSTRAINT_SLIDER:   parse_ok = parse_slider_params(params, &p); break;
        case CONSTRAINT_CONE:     parse_ok = parse_cone_params(params, &p); break;
        case CONSTRAINT_DISTANCE: parse_ok = parse_distance_params(params, &p); break;
        default:
            PyErr_SetString(PyExc_ValueError, "Unknown constraint type");
            return NULL;
    }

    if (!parse_ok) return NULL; // PyArg_ParseTuple sets the error

    // --- 2. Handle Resolution & Memory Check ---
    SHADOW_LOCK(&self->shadow_lock);
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

    // --- 3. Jolt Body Locking ---
    const JPH_BodyLockInterface* lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock1, lock2;
    JPH_BodyLockInterface_LockWrite(lock_iface, bid1, &lock1);
    JPH_BodyLockInterface_LockWrite(lock_iface, bid2, &lock2);

    if (!lock1.body || !lock2.body) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock1);
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock2);
        SHADOW_LOCK(&self->shadow_lock);
        self->free_constraint_slots[self->free_constraint_count++] = c_slot;
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Failed to lock bodies");
        return NULL;
    }

    // --- 4. Constraint Creation ---
    JPH_Constraint* constraint = NULL;
    switch (type) {
        case CONSTRAINT_FIXED:    constraint = create_fixed(&p, lock1.body, lock2.body); break;
        case CONSTRAINT_POINT:    constraint = create_point(&p, lock1.body, lock2.body); break;
        case CONSTRAINT_HINGE:    constraint = create_hinge(&p, lock1.body, lock2.body); break;
        case CONSTRAINT_SLIDER:   constraint = create_slider(&p, lock1.body, lock2.body); break;
        case CONSTRAINT_CONE:     constraint = create_cone(&p, lock1.body, lock2.body); break;
        case CONSTRAINT_DISTANCE: constraint = create_distance(&p, lock1.body, lock2.body); break;
        default: break; // Already handled above
    }

    // --- 5. Cleanup & Registration ---
    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock1);
    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock2);

    if (!constraint) {
        SHADOW_LOCK(&self->shadow_lock);
        self->free_constraint_slots[self->free_constraint_count++] = c_slot;
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Jolt failed to create constraint");
        return NULL;
    }

    JPH_PhysicsSystem_AddConstraint(self->system, constraint);
    
    SHADOW_LOCK(&self->shadow_lock);
    self->constraints[c_slot] = constraint;
    self->constraint_states[c_slot] = SLOT_ALIVE;
    uint32_t gen = self->constraint_generations[c_slot];
    ConstraintHandle handle = ((uint64_t)gen << 32) | c_slot;
    SHADOW_UNLOCK(&self->shadow_lock);

    DEBUG_LOG("Constraint registered: Handle %llu at Slot %u", handle, c_slot);

    return PyLong_FromUnsignedLongLong(handle);
}

static PyObject* PhysicsWorld_destroy_constraint(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t h = 0;
    static char *kwlist[] = {"handle", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);

    // 1. Unpack Handle manually (since the helper is for bodies)
    uint32_t slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);

    // 2. Validate Slot & Generation
    if (slot >= self->constraint_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid constraint handle (slot out of bounds)");
        return NULL;
    }

    if (self->constraint_generations[slot] != gen) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid constraint handle (stale generation)");
        return NULL;
    }

    // 3. Check State
    if (self->constraint_states[slot] == SLOT_ALIVE) {
        JPH_Constraint* c = self->constraints[slot];
        
        // 4. Jolt Cleanup
        // Safe to call even if simulation is running (Jolt locks internally)
        if (c) {
            // NEW: Automatically wake up bodies attached to this constraint
            // This prevents "floating" bodies when joints break.
            JPH_ConstraintType c_type = JPH_Constraint_GetType(c);
            if (c_type == JPH_ConstraintType_TwoBodyConstraint) {
                JPH_TwoBodyConstraint* tbc = (JPH_TwoBodyConstraint*)c;
                JPH_Body* b1 = JPH_TwoBodyConstraint_GetBody1(tbc);
                JPH_Body* b2 = JPH_TwoBodyConstraint_GetBody2(tbc);
                if (b1) JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
                if (b2) JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
            }
            JPH_PhysicsSystem_RemoveConstraint(self->system, c);
            JPH_Constraint_Destroy(c);
        }

        // 5. Recycle Slot
        self->constraints[slot] = NULL;
        self->constraint_states[slot] = SLOT_EMPTY;
        self->constraint_generations[slot]++; // Increment gen to invalidate old handles
        self->free_constraint_slots[self->free_constraint_count++] = slot;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_save_state(PhysicsWorldObject *self,
                                         PyObject *Py_UNUSED(unused)) {
  SHADOW_LOCK(&self->shadow_lock);
  size_t meta_size = sizeof(size_t) * 3 + sizeof(double);
  size_t dense_size = self->count * 4 * sizeof(float) * 4; 
  size_t mapping_size =
      self->slot_capacity * (sizeof(uint32_t) * 4 + sizeof(uint8_t));

  size_t total_size = meta_size + dense_size + mapping_size;

  PyObject *bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_size);
  char *ptr = PyBytes_AsString(bytes);

  memcpy(ptr, &self->count, sizeof(size_t)); ptr += sizeof(size_t);
  memcpy(ptr, &self->time, sizeof(double)); ptr += sizeof(double);

  memcpy(ptr, self->positions, self->count * 16); ptr += self->count * 16;
  memcpy(ptr, self->rotations, self->count * 16); ptr += self->count * 16;
  memcpy(ptr, self->linear_velocities, self->count * 16); ptr += self->count * 16;
  memcpy(ptr, self->angular_velocities, self->count * 16); ptr += self->count * 16;

  memcpy(ptr, self->generations, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
  memcpy(ptr, self->slot_to_dense, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
  memcpy(ptr, self->dense_to_slot, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
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
  SHADOW_LOCK(&self->shadow_lock);

  char *ptr = (char *)view.buf;
  size_t saved_count = 0;

  if (view.len < (Py_ssize_t)sizeof(size_t)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError, "Snapshot too small");
    return NULL;
  }
  memcpy(&saved_count, ptr, sizeof(size_t)); ptr += sizeof(size_t);

  if (saved_count > self->capacity) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError, "Snapshot count exceeds capacity");
    return NULL;
  }

  size_t required_size = sizeof(size_t) + sizeof(double);
  required_size += saved_count * 16 * 4 * sizeof(float);
  required_size += self->slot_capacity * (3 * sizeof(uint32_t) + 1 * sizeof(uint8_t));

  if ((size_t)view.len < required_size) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError, "Snapshot buffer truncated");
    return NULL;
  }

  self->count = saved_count;
  self->view_shape[0] = (Py_ssize_t)self->count;

  memcpy(&self->time, ptr, sizeof(double)); ptr += sizeof(double);

  memcpy(self->positions, ptr, self->count * 16); ptr += self->count * 16;
  memcpy(self->rotations, ptr, self->count * 16); ptr += self->count * 16;
  memcpy(self->linear_velocities, ptr, self->count * 16); ptr += self->count * 16;
  memcpy(self->angular_velocities, ptr, self->count * 16); ptr += self->count * 16;

  memcpy(self->generations, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
  memcpy(self->slot_to_dense, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
  memcpy(self->dense_to_slot, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
  memcpy(self->slot_states, ptr, self->slot_capacity);

  for (size_t i = 0; i < self->count; i++) {
    JPH_BodyID bid = self->body_ids[i];
    if (bid == JPH_INVALID_BODY_ID) continue;

    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = (double)self->positions[i * 4];
    p->y = (double)self->positions[i * 4 + 1];
    p->z = (double)self->positions[i * 4 + 2];

    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = self->rotations[i * 4];
    q->y = self->rotations[i * 4 + 1];
    q->z = self->rotations[i * 4 + 2];
    q->w = self->rotations[i * 4 + 3];

    JPH_BodyInterface_SetPositionAndRotation(self->body_interface, bid, p, q, JPH_Activation_Activate);
    JPH_Vec3 lv = {self->linear_velocities[i * 4], self->linear_velocities[i * 4 + 1], self->linear_velocities[i * 4 + 2]};
    JPH_BodyInterface_SetLinearVelocity(self->body_interface, bid, &lv);
    JPH_Vec3 av = {self->angular_velocities[i * 4], self->angular_velocities[i * 4 + 1], self->angular_velocities[i * 4 + 2]};
    JPH_BodyInterface_SetAngularVelocity(self->body_interface, bid, &av);
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  PyBuffer_Release(&view);
  Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_step(PhysicsWorldObject* self, PyObject* args) {
    float dt = 1.0f/60.0f;
    if (!PyArg_ParseTuple(args, "|f", &dt)) return NULL;

    Py_BEGIN_ALLOW_THREADS
    SHADOW_LOCK(&self->shadow_lock);
    
    // 1. Snapshot CURRENT state to PREVIOUS state
    // We copy the entire dense array. This is very fast (memcpy).
    memcpy(self->prev_positions, self->positions, self->count * 4 * sizeof(float));
    memcpy(self->prev_rotations, self->rotations, self->count * 4 * sizeof(float));

    // 2. Flush commands (creates new bodies, etc.)
    flush_commands(self);
    self->contact_count = 0;
    
    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. Jolt Physics Update (Updates internal Jolt state)
    JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);
    
    SHADOW_LOCK(&self->shadow_lock);
    // 4. Sync Jolt state back to CURRENT buffers
    culverin_sync_shadow_buffers(self);
    SHADOW_UNLOCK(&self->shadow_lock);
    
    Py_END_ALLOW_THREADS
    self->time += (double)dt;

    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_create_character(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  float px = 0;
  float py = 0;
  float pz = 0;
  float height = 1.8f;
  float radius = 0.4f;
  float step_height = 0.4f;
  float max_slope = 45.0f;
  static char *kwlist[] = {"pos",         "height",    "radius",
                           "step_height", "max_slope", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)|ffff", kwlist, &px, &py,
                                   &pz, &height, &radius, &step_height,
                                   &max_slope)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return NULL;
    }
  }
  uint32_t char_slot = self->free_slots[--self->free_count];
  SHADOW_UNLOCK(&self->shadow_lock);

  float half_h = (height - 2.0f * radius) * 0.5f;
  if (half_h < 0.1f) half_h = 0.1f;

  JPH_CapsuleShapeSettings *ss = JPH_CapsuleShapeSettings_Create(half_h, radius);
  JPH_Shape *shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);

  if (!shape) return PyErr_NoMemory();

  JPH_CharacterVirtualSettings settings;
  memset(&settings, 0, sizeof(JPH_CharacterVirtualSettings));
  JPH_CharacterVirtualSettings_Init(&settings); 
  settings.base.shape = shape;
  settings.base.up.x = 0; settings.base.up.y = 1; settings.base.up.z = 0;
  settings.base.supportingVolume.normal.x = 0;
  settings.base.supportingVolume.normal.y = 1;
  settings.base.supportingVolume.normal.z = 0;
  settings.base.supportingVolume.distance = -1.0e10f;

  float slope_rad = max_slope * (3.14159f / 180.0f);
  settings.base.maxSlopeAngle = slope_rad;
  settings.base.enhancedInternalEdgeRemoval = true;
  settings.mass = 70.0f;
  settings.maxStrength = 500.0f;
  settings.characterPadding = 0.02f;
  settings.penetrationRecoverySpeed = 1.0f;
  settings.predictiveContactDistance = 0.1f;
  settings.maxCollisionIterations = 5;
  settings.maxConstraintIterations = 15;
  settings.minTimeRemaining = 0.0001f;
  settings.collisionTolerance = 0.001f;

  JPH_STACK_ALLOC(JPH_RVec3, pos_aligned);
  pos_aligned->x = (double)px; pos_aligned->y = (double)py; pos_aligned->z = (double)pz;
  JPH_STACK_ALLOC(JPH_Quat, rot_aligned);
  rot_aligned->x = 0; rot_aligned->y = 0; rot_aligned->z = 0; rot_aligned->w = 1;

  JPH_CharacterVirtual *j_char =
      JPH_CharacterVirtual_Create(&settings, pos_aligned, rot_aligned, 0, self->system);
  JPH_Shape_Destroy(shape);

  if (!j_char) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create CharacterVirtual");
    return NULL;
  }

  if (self->char_vs_char_manager) {
        // 1. Add this character to the manager's list
        JPH_CharacterVsCharacterCollisionSimple_AddCharacter(self->char_vs_char_manager, j_char);
        // 2. Tell this character to resolve against others in the manager
        JPH_CharacterVirtual_SetCharacterVsCharacterCollision(j_char, self->char_vs_char_manager);
    }

  // Resolve the internal BodyID
  JPH_BodyID internal_bid = JPH_CharacterVirtual_GetInnerBodyID(j_char);

  // Set UserData on the internal body so queries return this slot index
  JPH_BodyInterface *bi = self->body_interface;
  JPH_BodyInterface_SetUserData(bi, internal_bid, (uint64_t)char_slot);

  // CRITICAL: Register the character's body in the slot map
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t dense_idx = (uint32_t)self->count;
  self->body_ids[dense_idx] = internal_bid;
  self->slot_to_dense[char_slot] = dense_idx;
  self->dense_to_slot[dense_idx] = char_slot;
  self->slot_states[char_slot] = SLOT_ALIVE;
  uint32_t gen = self->generations[char_slot];
  self->count++;
  self->view_shape[0] = (Py_ssize_t)self->count;
  SHADOW_UNLOCK(&self->shadow_lock);

  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);
  PyObject *char_type = st->CharacterType;

  // NEW: Use GC-aware allocator
  CharacterObject *obj = (CharacterObject *)PyObject_GC_New(CharacterObject, (PyTypeObject *)char_type);
  if (!obj) {
      JPH_CharacterBase_Destroy((JPH_CharacterBase *)j_char);
      return NULL;
  }

  obj->character = j_char;
  obj->world = self;

  // Link the slot to the Character object so we can retrieve it
  obj->handle = make_handle(char_slot, 1);
  DEBUG_LOG("Creating Character: Slot %u, Handle %llu", char_slot, obj->handle);

  // Set 'Previous' to 'Current' so frame 0 doesn't jitter
  obj->prev_px = px;
  obj->prev_py = py;
  obj->prev_pz = pz;
  // Assuming Identity rotation for start, or convert 'rot' if provided
  obj->prev_rx = 0.0f;
  obj->prev_ry = 0.0f;
  obj->prev_rz = 0.0f;
  obj->prev_rw = 1.0f;

  obj->push_strength = 200.0f;
  obj->last_vx = 0;
  obj->last_vy = 0;
  obj->last_vz = 0;
  Py_INCREF(self);

  JPH_CharacterContactListener_SetProcs(&char_listener_procs);
  obj->listener = JPH_CharacterContactListener_Create(obj);
  JPH_CharacterVirtual_SetListener(j_char, obj->listener);

  // Filter Setup using GLOBAL pointers
  JPH_BodyFilter_SetProcs(&global_bf_procs);
  obj->body_filter = JPH_BodyFilter_Create(NULL);

  JPH_ShapeFilter_SetProcs(&global_sf_procs);
  obj->shape_filter = JPH_ShapeFilter_Create(NULL);

  obj->bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  obj->obj_filter = JPH_ObjectLayerFilter_Create(NULL);

  // NEW: Track with GC
  PyObject_GC_Track((PyObject *)obj);

  return (PyObject *)obj;
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
  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
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

  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)slot);
  if (motion_type == 2) JPH_BodyCreationSettings_SetAllowSleeping(settings, true);

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
  cmd->data.create.user_data = (uint64_t)user_data;
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  uint32_t gen = self->generations[slot];
  BodyHandle handle = make_handle(slot, gen);
  SHADOW_UNLOCK(&self->shadow_lock);

  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject *PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  float px = NAN; float py = NAN; float pz = NAN;
  float rx = NAN; float ry = NAN; float rz = NAN; float rw = NAN;
  Py_buffer v_view; Py_buffer i_view;
  unsigned long long user_data = 0;
  static char *kwlist[] = {"pos", "rot", "vertices", "indices", "user_data", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(ffff)y*y*|K", kwlist, &px,
                                   &py, &pz, &rx, &ry, &rz, &rw, &v_view,
                                   &i_view, &user_data)) {
    return NULL;
  }

  uint32_t vertex_count = (uint32_t)(v_view.len / (3 * sizeof(float)));
  uint32_t index_count = (uint32_t)(i_view.len / sizeof(uint32_t));
  uint32_t tri_count = index_count / 3;

  JPH_IndexedTriangle *jolt_tris = (JPH_IndexedTriangle *)PyMem_RawMalloc(
      tri_count * sizeof(JPH_IndexedTriangle));
  if (!jolt_tris) {
    PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
    return PyErr_NoMemory();
  }

  uint32_t *raw_indices = (uint32_t *)i_view.buf;
  for (uint32_t t = 0; t < tri_count; t++) {
    jolt_tris[t].i1 = raw_indices[t * 3 + 0];
    jolt_tris[t].i2 = raw_indices[t * 3 + 1];
    jolt_tris[t].i3 = raw_indices[t * 3 + 2];
    jolt_tris[t].materialIndex = 0;
    jolt_tris[t].userData = 0;
  }

  JPH_MeshShapeSettings *mss = JPH_MeshShapeSettings_Create2(
      (JPH_Vec3 *)v_view.buf, vertex_count, jolt_tris, tri_count);

  if (!mss) {
    PyMem_RawFree(jolt_tris); PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
    PyErr_SetString(PyExc_RuntimeError, "Jolt failed to create MeshShapeSettings");
    return NULL;
  }

  JPH_Shape *shape = (JPH_Shape *)JPH_MeshShapeSettings_CreateShape(mss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)mss);
  PyMem_RawFree(jolt_tris);

  if (!shape) {
    PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
    PyErr_SetString(PyExc_RuntimeError, "Jolt failed to build Mesh BVH");
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity + 1024) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_Shape_Destroy(shape);
      PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
      return NULL; 
    }
  }

  uint32_t slot = self->free_slots[--self->free_count];
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = px; pos->y = py; pos->z = pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx; rot->y = ry; rot->z = rz; rot->w = rw;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, JPH_MotionType_Static, 0 
  );
  JPH_Shape_Destroy(shape);
  if (!settings) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
    return PyErr_NoMemory();
  }
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)slot);

  ensure_command_capacity(self);
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->type = CMD_CREATE_BODY;
  cmd->slot = slot;
  cmd->data.create.settings = settings;
  cmd->data.create.user_data = (uint64_t)user_data;
  self->slot_states[slot] = SLOT_PENDING_CREATE;
  BodyHandle handle = make_handle(slot, self->generations[slot]);

  SHADOW_UNLOCK(&self->shadow_lock);
  PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject *PhysicsWorld_destroy_body(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  if (self->slot_states[slot] == SLOT_ALIVE ||
      self->slot_states[slot] == SLOT_PENDING_CREATE) {

    if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
    }
    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->type = CMD_DESTROY_BODY;
    cmd->slot = slot;
    self->slot_states[slot] = SLOT_PENDING_DESTROY;
  }
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// vroom vroom
// this is paperwork and i did surgery in the core
static PyObject* PhysicsWorld_create_vehicle(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t chassis_h = 0;
    PyObject* py_wheels = NULL; 
    static char *kwlist[] = {"chassis", "wheels", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "KO", kwlist, &chassis_h, &py_wheels)) return NULL;
    
    if (!PyList_Check(py_wheels)) {
        PyErr_SetString(PyExc_TypeError, "wheels must be a list");
        return NULL;
    }

    DEBUG_LOG("--- Create Vehicle Start ---");

    // 1. Resolve Chassis
    SHADOW_LOCK(&self->shadow_lock);
    flush_commands(self); 
    JPH_PhysicsSystem_OptimizeBroadPhase(self->system);
    
    uint32_t slot;
    if (!unpack_handle(self, chassis_h, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid chassis handle");
        return NULL;
    }
    JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
    SHADOW_UNLOCK(&self->shadow_lock);

    const JPH_BodyLockInterface* lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock;
    JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);
    
    bool is_locked = true; 

    if (!lock.body) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
        is_locked = false; 
        PyErr_SetString(PyExc_RuntimeError, "Could not lock chassis body");
        return NULL;
    }

    // --- RESOURCE TRACKING ---
    JPH_LinearCurve* f_curve = NULL;
    JPH_LinearCurve* t_curve = NULL;
    JPH_WheelSettings** w_settings = NULL;
    JPH_WheeledVehicleControllerSettings* v_ctrl = NULL;
    JPH_VehicleTransmissionSettings* trans = NULL;
    JPH_VehicleCollisionTesterRay* tester = NULL;
    JPH_VehicleConstraint* j_veh = NULL;

    Py_ssize_t num_wheels = PyList_Size(py_wheels);
    if (num_wheels < 4) { // FIX: Require 4 wheels for the 4WD setup below
        // If < 4, we'd need dynamic differential logic. For this demo, assume 4.
        // Or fallback to FWD if < 4. But let's enforce 4 for stability.
        // Actually, let's just warn and use FWD if < 4 to prevent crash.
    }

    // 1. Curves
    f_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(f_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(f_curve, 1.0f, 1.0f);

    t_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(t_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(t_curve, 1.0f, 1.0f);

    // 2. Wheels
    w_settings = (JPH_WheelSettings**)PyMem_RawMalloc(num_wheels * sizeof(JPH_WheelSettings*));
    if (!w_settings) { PyErr_NoMemory(); goto error_cleanup; }
    memset(w_settings, 0, num_wheels * sizeof(JPH_WheelSettings*));

    for (Py_ssize_t i = 0; i < num_wheels; i++) {
        PyObject* w_dict = PyList_GetItem(py_wheels, i);
        if (!PyDict_Check(w_dict)) { PyErr_SetString(PyExc_TypeError, "Wheel not dict"); goto error_cleanup; }

        PyObject* o_pos = PyDict_GetItemString(w_dict, "pos");
        PyObject* o_rad = PyDict_GetItemString(w_dict, "radius");
        PyObject* o_wid = PyDict_GetItemString(w_dict, "width");

        float px, py, pz, radius, width;
        if (!o_pos || !PyArg_ParseTuple(o_pos, "fff", &px, &py, &pz)) { PyErr_SetString(PyExc_ValueError, "Invalid pos"); goto error_cleanup; }
        if (!o_rad || !PyFloat_Check(o_rad)) { PyErr_SetString(PyExc_ValueError, "Invalid radius"); goto error_cleanup; }
        if (!o_wid || !PyFloat_Check(o_wid)) { PyErr_SetString(PyExc_ValueError, "Invalid width"); goto error_cleanup; }
        
        radius = (float)PyFloat_AsDouble(o_rad);
        width = (float)PyFloat_AsDouble(o_wid);

        JPH_WheelSettingsWV* w = JPH_WheelSettingsWV_Create();
        JPH_WheelSettings_SetPosition((JPH_WheelSettings*)w, &(JPH_Vec3){px, py, pz});
        JPH_WheelSettings_SetWheelForward((JPH_WheelSettings*)w, &(JPH_Vec3){0, 0, 1}); 
        JPH_WheelSettings_SetWheelUp((JPH_WheelSettings*)w, &(JPH_Vec3){0, 1, 0});
        JPH_WheelSettings_SetSteeringAxis((JPH_WheelSettings*)w, &(JPH_Vec3){0, 1, 0});

        JPH_WheelSettings_SetRadius((JPH_WheelSettings*)w, radius);
        JPH_WheelSettings_SetWidth((JPH_WheelSettings*)w, width);
        JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings*)w, 0.1f);
        JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings*)w, 0.5f); 
        
        JPH_SpringSettings spring;
        spring.mode = JPH_SpringMode_FrequencyAndDamping;
        spring.frequencyOrStiffness = 2.0f; 
        spring.damping = 0.5f;
        JPH_WheelSettings_SetSuspensionSpring((JPH_WheelSettings*)w, &spring);

        JPH_WheelSettingsWV_SetLongitudinalFriction(w, f_curve);
        JPH_WheelSettingsWV_SetLateralFriction(w, f_curve);
        JPH_WheelSettingsWV_SetMaxBrakeTorque(w, 5000.0f);
        JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, 8000.0f);
        JPH_WheelSettingsWV_SetInertia(w, 0.5f);

        if (i < 2) { 
            // Set front wheels to steer up to 30 degrees (in radians)
            float max_steer = 30.0f * (JPH_M_PI / 180.0f);
            JPH_WheelSettingsWV_SetMaxSteerAngle(w, max_steer);
        } else {
            // Rear wheels do not steer
            JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.0f);
        }

        w_settings[i] = (JPH_WheelSettings*)w;
    }

    // 3. Drivetrain
    v_ctrl = JPH_WheeledVehicleControllerSettings_Create();
    
    JPH_VehicleEngineSettings eng;
    JPH_VehicleEngineSettings_Init(&eng);
    eng.maxTorque = 600.0f;
    eng.minRPM = 1000.0f;
    eng.maxRPM = 7000.0f;
    eng.inertia = 0.5f; 
    eng.normalizedTorque = t_curve; 
    JPH_WheeledVehicleControllerSettings_SetEngine(v_ctrl, &eng);

    trans = JPH_VehicleTransmissionSettings_Create();
    JPH_VehicleTransmissionSettings_SetMode(trans, JPH_TransmissionMode_Manual);
    // FIX: Forward gears ONLY. Gear 1 is index 0.
    float forward_gears[] = { 4.0f, 2.0f, 1.5f, 1.1f, 0.9f }; 
    JPH_VehicleTransmissionSettings_SetGearRatios(trans, forward_gears, 5);
    
    // FIX: Reverse gears separately.
    float reverse_gears[] = { -3.5f };
    JPH_VehicleTransmissionSettings_SetReverseGearRatios(trans, reverse_gears, 1);
    JPH_VehicleTransmissionSettings_SetClutchStrength(trans, 10000.0f); 
    
    JPH_WheeledVehicleControllerSettings_SetTransmission(v_ctrl, trans);

    // FIX: Use the new helper to create 4WD setup
    // This ensures Jolt receives valid differential structs via C++ vector push_back
    if (num_wheels >= 4) {
        // Front Diff (Wheels 0, 1)
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
        // Rear Diff (Wheels 2, 3)
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 2, 3);
    } else {
        // FWD
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
    }

    // FIX: 4WD Setup (2 Differentials)
    // Diff 0: Wheels 0 & 1 (Front)
    // Diff 1: Wheels 2 & 3 (Rear)
    // This ensures torque reaches all wheels.
    int diff_count = (num_wheels >= 4) ? 2 : 1;
    JPH_VehicleDifferentialSettings diffs[2]; // Stack allocation safe here
    
    JPH_VehicleDifferentialSettings_Init(&diffs[0]);
    diffs[0].leftWheel = 0;
    diffs[0].rightWheel = 1;
    diffs[0].differentialRatio = 3.5f;
    diffs[0].limitedSlipRatio = 1.4f;
    diffs[0].engineTorqueRatio = (diff_count == 2) ? 0.5f : 1.0f; // Split torque

    if (diff_count == 2) {
        JPH_VehicleDifferentialSettings_Init(&diffs[1]);
        diffs[1].leftWheel = 2;
        diffs[1].rightWheel = 3;
        diffs[1].differentialRatio = 3.5f;
        diffs[1].limitedSlipRatio = 1.4f;
        diffs[1].engineTorqueRatio = 0.5f;
    }

    JPH_WheeledVehicleControllerSettings_SetDifferentials(v_ctrl, diffs, diff_count);

    // 4. Constraint
    JPH_VehicleConstraintSettings v_set;
    JPH_VehicleConstraintSettings_Init(&v_set);
    v_set.wheelsCount = (uint32_t)num_wheels; 
    v_set.wheels = w_settings; 
    v_set.controller = (JPH_VehicleControllerSettings*)v_ctrl;
    v_set.up.x = 0; v_set.up.y = 1; v_set.up.z = 0;
    v_set.forward.x = 0; v_set.forward.y = 0; v_set.forward.z = 1;
    v_set.maxPitchRollAngle = 1.04f; 

    j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
    if (!j_veh) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create Jolt VehicleConstraint");
        goto error_cleanup;
    }

    JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint*)j_veh);
    JPH_PhysicsStepListener* step_listener = JPH_VehicleConstraint_AsPhysicsStepListener(j_veh);
    JPH_PhysicsSystem_AddStepListener(self->system, step_listener);

    tester = JPH_VehicleCollisionTesterRay_Create(1, &(JPH_Vec3){0, 1, 0}, 0.5f);
    JPH_VehicleConstraint_SetVehicleCollisionTester(j_veh, (JPH_VehicleCollisionTester*)tester);

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    is_locked = false;

    // 5. Wrap Object
    PyObject *module = PyType_GetModule(Py_TYPE(self));
    CulverinState *st = get_culverin_state(module);
    VehicleObject *obj = (VehicleObject *)PyObject_New(VehicleObject, (PyTypeObject *)st->VehicleType);
    
    if (!obj) goto error_cleanup;

    obj->vehicle = j_veh;
    obj->tester = (JPH_VehicleCollisionTester*)tester;
    obj->world = self;
    obj->num_wheels = (uint32_t)num_wheels;
    obj->current_gear = 1; 
    
    obj->wheel_settings = w_settings;
    obj->controller_settings = (JPH_VehicleControllerSettings*)v_ctrl;
    obj->transmission_settings = trans; 
    obj->friction_curve = f_curve;
    obj->torque_curve = t_curve;
    
    Py_INCREF(self);
    return (PyObject *)obj;

error_cleanup:
    if (is_locked) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    }
    if (j_veh) { 
        JPH_PhysicsSystem_RemoveConstraint(self->system, (JPH_Constraint*)j_veh);
        JPH_Constraint_Destroy((JPH_Constraint*)j_veh); 
    }
    if (tester) { JPH_VehicleCollisionTester_Destroy((JPH_VehicleCollisionTester*)tester); }
    if (trans) { JPH_VehicleTransmissionSettings_Destroy(trans); }
    if (v_ctrl) { JPH_VehicleControllerSettings_Destroy((JPH_VehicleControllerSettings*)v_ctrl); }
    if (w_settings) {
        for (Py_ssize_t i = 0; i < num_wheels; i++) {
            if (w_settings[i]) JPH_WheelSettings_Destroy(w_settings[i]);
        }
        PyMem_RawFree(w_settings);
    }
    if (f_curve) { JPH_LinearCurve_Destroy(f_curve); }
    if (t_curve) { JPH_LinearCurve_Destroy(t_curve); }
    
    return NULL;
}

static PyObject *PhysicsWorld_set_position(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN; float y = NAN; float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
  uint32_t idx = self->slot_to_dense[slot];
  JPH_STACK_ALLOC(JPH_RVec3, p);
  p->x = x; p->y = y; p->z = z;
  JPH_BodyInterface_SetPosition(self->body_interface, self->body_ids[idx], p,
                                JPH_Activation_Activate);
  self->positions[idx * 4 + 0] = x;
  self->positions[idx * 4 + 1] = y;
  self->positions[idx * 4 + 2] = z;
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_rotation(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN; float y = NAN; float z = NAN; float w = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", "w", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffff", kwlist, &handle_raw, &x,
                                   &y, &z, &w)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock); 
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }
  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_BodyID bid = self->body_ids[dense_idx];

  JPH_STACK_ALLOC(JPH_Quat, q);
  q->x = x; q->y = y; q->z = z; q->w = w;
  JPH_BodyInterface_SetRotation(self->body_interface, bid, q,
                                JPH_Activation_Activate);
  self->rotations[dense_idx * 4 + 0] = x;
  self->rotations[dense_idx * 4 + 1] = y;
  self->rotations[dense_idx * 4 + 2] = z;
  self->rotations[dense_idx * 4 + 3] = w;
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
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
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
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KK", kwlist, &handle_raw,
                                   &data)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
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
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
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

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }
  JPH_BodyInterface_ActivateBody(self->body_interface,
                                 self->body_ids[self->slot_to_dense[slot]]);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_deactivate(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }
  JPH_BodyInterface_DeactivateBody(self->body_interface,
                                   self->body_ids[self->slot_to_dense[slot]]);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_transform(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float px = NAN; float py = NAN; float pz = NAN;
  float rx = NAN; float ry = NAN; float rz = NAN; float rw = NAN;
  static char *kwlist[] = {"handle", "pos", "rot", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K(fff)(ffff)", kwlist,
                                   &handle_raw, &px, &py, &pz, &rx, &ry, &rz,
                                   &rw)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  if (ensure_command_capacity(self)) {
    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->type = CMD_SET_TRNS;
    cmd->slot = slot;
    cmd->data.transform.px = px;
    cmd->data.transform.py = py;
    cmd->data.transform.pz = pz;
    cmd->data.transform.rx = rx;
    cmd->data.transform.ry = ry;
    cmd->data.transform.rz = rz;
    cmd->data.transform.rw = rw;
  }
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// Helper to deduce handle and append to list
static void append_hit(QueryContext *ctx, JPH_BodyID bid) {
  uint64_t slot =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);

  if (slot >= ctx->world->slot_capacity) return;

  uint32_t gen = ctx->world->generations[slot];
  BodyHandle h = make_handle((uint32_t)slot, gen);

  PyObject *py_h = PyLong_FromUnsignedLongLong(h);
  PyList_Append(ctx->result_list, py_h);
  Py_DECREF(py_h);
}

static float OverlapCallback_Narrow(void *context,
                                    const JPH_CollideShapeResult *result) {
  OverlapContext *ctx = (OverlapContext *)context;
  uint64_t slot = JPH_BodyInterface_GetUserData(ctx->world->body_interface,
                                                result->bodyID2);
  if (slot < ctx->world->slot_capacity) {
    uint32_t gen = ctx->world->generations[slot];
    BodyHandle h = make_handle((uint32_t)slot, gen);

    PyObject *py_h = PyLong_FromUnsignedLongLong(h);
    PyList_Append(ctx->result_list, py_h);
    Py_DECREF(py_h);
  }
  return 1.0f;
}

static float OverlapCallback_Broad(void *context, const JPH_BodyID result) {
  OverlapContext *ctx = (OverlapContext *)context;
  uint64_t slot =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, result);

  if (slot < ctx->world->slot_capacity) {
    uint32_t gen = ctx->world->generations[slot];
    BodyHandle h = make_handle((uint32_t)slot, gen);

    PyObject *py_h = PyLong_FromUnsignedLongLong(h);
    PyList_Append(ctx->result_list, py_h);
    Py_DECREF(py_h);
  }
  return 1.0f;
}

static PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  float x = NAN; float y = NAN; float z = NAN; float radius = NAN;
  static char *kwlist[] = {"center", "radius", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &x, &y, &z,
                                   &radius)) {
    return NULL;
  }

  JPH_SphereShapeSettings *ss = JPH_SphereShapeSettings_Create(radius);
  if (!ss) return PyErr_NoMemory();

  JPH_Shape *shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);

  if (!shape) return PyErr_NoMemory();

  PyObject *results = PyList_New(0);
  if (!results) {
    JPH_Shape_Destroy(shape);
    return NULL;
  }

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = x; pos->y = y; pos->z = z;
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

  QueryContext ctx = {self, results};
  const JPH_NarrowPhaseQuery *nq =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  SHADOW_LOCK(&self->shadow_lock);
  JPH_NarrowPhaseQuery_CollideShape(
      nq, shape, scale, transform, settings, base_offset,
      OverlapCallback_Narrow, &ctx, NULL, NULL, NULL, NULL
  );
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_Shape_Destroy(shape);
  return results;
}

static PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  float min_x = NAN; float min_y = NAN; float min_z = NAN;
  float max_x = NAN; float max_y = NAN; float max_z = NAN;
  static char *kwlist[] = {"min", "max", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, &min_x,
                                   &min_y, &min_z, &max_x, &max_y, &max_z)) {
    return NULL;
  }

  PyObject *results = PyList_New(0);
  if (!results) return NULL;

  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x; box->min.y = min_y; box->min.z = min_z;
  box->max.x = max_x; box->max.y = max_y; box->max.z = max_z;

  QueryContext ctx = {self, results};
  const JPH_BroadPhaseQuery *bq =
      JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);

  SHADOW_LOCK(&self->shadow_lock);
  JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx,
                                   NULL, NULL
  );
  SHADOW_UNLOCK(&self->shadow_lock);
  return results;
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

// NEW: Use-After-Free Protection for MemoryViews
static PyObject *make_view(PhysicsWorldObject *self, void *ptr) {
  if (!ptr) Py_RETURN_NONE;

  // 1. Manually Increment Count (Under Lock)
  SHADOW_LOCK(&self->shadow_lock);
  self->view_export_count++;
  SHADOW_UNLOCK(&self->shadow_lock);

  // 2. Setup Buffer
  self->view_shape[0] = (Py_ssize_t)(self->count * 4);
  self->view_strides[0] = (Py_ssize_t)sizeof(float);

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = ptr;
  
  // 3. Set Owner to self. PyMemoryView_FromBuffer will INCREF this.
  // When it is destroyed, it calls bf_releasebuffer on self.
  buf.obj = (PyObject *)self;
  Py_INCREF(self);

  buf.len = (Py_ssize_t)(self->count * 4 * sizeof(float));
  buf.readonly = 1;
  buf.itemsize = sizeof(float);
  buf.format = "f";
  buf.ndim = 1;                    
  buf.shape = &self->view_shape[0]; 
  buf.strides = &self->view_strides[0];

  PyObject* mv = PyMemoryView_FromBuffer(&buf);
  if (!mv) {
      // Rollback if creation failed
      SHADOW_LOCK(&self->shadow_lock);
      self->view_export_count--;
      SHADOW_UNLOCK(&self->shadow_lock);
      return NULL;
  }
  return mv;
}

static PyObject* PhysicsWorld_get_active_indices(PhysicsWorldObject* self, PyObject* args) {
    SHADOW_LOCK(&self->shadow_lock);
    uint32_t* indices = (uint32_t*)PyMem_RawMalloc(self->count * sizeof(uint32_t));
    if (!indices) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    size_t active_count = 0;
    JPH_BodyInterface* bi = self->body_interface;

    for (size_t i = 0; i < self->count; i++) {
        JPH_BodyID bid = self->body_ids[i];
        if (bid != JPH_INVALID_BODY_ID && JPH_BodyInterface_IsActive(bi, bid)) {
            indices[active_count++] = (uint32_t)i;
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    PyObject* bytes_obj = PyBytes_FromStringAndSize((char*)indices, active_count * sizeof(uint32_t));
    PyMem_RawFree(indices);
    return bytes_obj;
}

static PyObject *get_user_data_buffer(PhysicsWorldObject *self, void *c) {
  if (!self->user_data) Py_RETURN_NONE;

  SHADOW_LOCK(&self->shadow_lock);
  self->view_export_count++;
  SHADOW_UNLOCK(&self->shadow_lock);

  self->view_shape[0] = (Py_ssize_t)self->count;
  self->view_shape[1] = 1; 

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = self->user_data;
  buf.obj = (PyObject *)self;
  Py_INCREF(self);
  buf.len = (Py_ssize_t)(self->count * sizeof(uint64_t));
  buf.readonly = 1;
  buf.itemsize = sizeof(uint64_t);
  buf.format = "Q"; 
  buf.ndim = 1;
  buf.shape = &self->view_shape[0];
  buf.strides = &buf.itemsize;

  PyObject* mv = PyMemoryView_FromBuffer(&buf);
  if (!mv) {
      SHADOW_LOCK(&self->shadow_lock);
      self->view_export_count--;
      SHADOW_UNLOCK(&self->shadow_lock);
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

    if (!self->vehicle) {
        PyErr_SetString(PyExc_RuntimeError, "Vehicle has been destroyed");
        return NULL;
    }

    JPH_WheeledVehicleController* controller = (JPH_WheeledVehicleController*)JPH_VehicleConstraint_GetController(self->vehicle);
    const JPH_Body* chassis = JPH_VehicleConstraint_GetVehicleBody(self->vehicle);
    JPH_VehicleTransmission* transmission = (JPH_VehicleTransmission*)JPH_WheeledVehicleController_GetTransmission(controller);
    
    // Ensure the chassis is active so physics run
    JPH_BodyInterface_ActivateBody(self->world->body_interface, JPH_Body_GetID(chassis));

    // 1. Get Forward Speed
    JPH_Vec3 linear_vel;
    JPH_Body_GetLinearVelocity((JPH_Body *)chassis, &linear_vel);
    JPH_RMat4 world_transform;
    JPH_Body_GetWorldTransform(chassis, &world_transform);
    // Forward vector is Z-axis (Column 2)
    JPH_Vec3 world_fwd = { (float)world_transform.column[2].x, (float)world_transform.column[2].y, (float)world_transform.column[2].z }; 
    float speed = linear_vel.x * world_fwd.x + linear_vel.y * world_fwd.y + linear_vel.z * world_fwd.z;

    // 2. State Machine for Gears and Clutch
    float input_throttle = 0.0f;
    float input_brake = brake; 
    float clutch_friction = 1.0f; // Default engaged
    int target_gear = self->current_gear;

    // Deadzone
    if (fabsf(forward) < 0.05f) forward = 0.0f;

    if (forward > 0.01f) {
        // User wants to go FORWARD
        if (speed < -0.5f) {
            // But we are moving backward: Apply BRAKE
            input_brake = 1.0f;
            input_throttle = 0.0f;
        } else {
            // Stopped or moving forward: DRIVE
            input_throttle = forward;
            target_gear = 1; 
            clutch_friction = 1.0f;
        }
    } 
    else if (forward < -0.01f) {
        // User wants to go BACKWARD
        if (speed > 0.5f) {
            // But we are moving forward: Apply BRAKE
            input_brake = 1.0f;
            input_throttle = 0.0f;
        } else {
            // Stopped or moving backward: REVERSE
            input_throttle = fabsf(forward);
            target_gear = -1; 
            clutch_friction = 1.0f;
        }
    } 
    else {
        // NO INPUT: Neutralize to prevent the "Idle Crawl"
        input_throttle = 0.0f;
        target_gear = 0;      // Neutral
        clutch_friction = 0.0f; // Disengage engine from wheels
        
        // Rolling resistance: If we aren't braking, apply 5% brake to eventually stop
        if (input_brake < 0.01f) input_brake = 0.05f; 
    }

    // 3. Apply to Jolt
    self->current_gear = target_gear;
    JPH_VehicleTransmission_Set(transmission, self->current_gear, clutch_friction);

    JPH_WheeledVehicleController_SetDriverInput(
        controller,
        input_throttle,
        right,
        input_brake,
        handbrake
    );
    
    Py_RETURN_NONE;
}

static PyObject* Vehicle_get_wheel_transform(VehicleObject* self, PyObject* args) {
    uint32_t index;
    if (!PyArg_ParseTuple(args, "I", &index)) return NULL;
    if (index >= self->num_wheels) return PyErr_Format(PyExc_IndexError, "Wheel index %u out of range", index);

    // Aligned double-precision matrix for Jolt data
    JPH_STACK_ALLOC(JPH_RMat4, transform);
    
    // Basis vectors
    JPH_Vec3 right = {1.0f, 0.0f, 0.0f};
    JPH_Vec3 up = {0.0f, 1.0f, 0.0f};

    JPH_VehicleConstraint_GetWheelWorldTransform(
        self->vehicle,
        index,
        &right,
        &up,
        transform
    );

    // FIX 1: Extract Position from the Double Precision column3
    double px = transform->column3.x;
    double py = transform->column3.y;
    double pz = transform->column3.z;

    // FIX 2: Correct variable names for rotation matrix
    // Columns 0, 1, 2 of JPH_RMat4 are JPH_Vec4 (floats), so they fit in JPH_Mat4
    JPH_STACK_ALLOC(JPH_Mat4, rot_only_mat);
    JPH_Mat4_Identity(rot_only_mat);
    rot_only_mat->column[0] = transform->column[0];
    rot_only_mat->column[1] = transform->column[1];
    rot_only_mat->column[2] = transform->column[2];
    
    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_Mat4_GetQuaternion(rot_only_mat, q);

    // Return ( (x,y,z), (x,y,z,w) )
    // Note: Py_BuildValue uses 'd' for double, 'f' for float
    PyObject* py_pos = Py_BuildValue("(ddd)", px, py, pz);
    PyObject* py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);
    PyObject* result = PyTuple_Pack(2, py_pos, py_rot);
    
    Py_DECREF(py_pos);
    Py_DECREF(py_rot);
    return result;
}

static PyObject* Vehicle_get_debug_state(VehicleObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->vehicle) Py_RETURN_NONE;

    JPH_WheeledVehicleController* controller = (JPH_WheeledVehicleController*)JPH_VehicleConstraint_GetController(self->vehicle);
    const JPH_VehicleEngine* engine = JPH_WheeledVehicleController_GetEngine(controller);
    const JPH_VehicleTransmission* trans = JPH_WheeledVehicleController_GetTransmission(controller);
    
    // 1. Inputs
    float in_fwd = JPH_WheeledVehicleController_GetForwardInput(controller);
    float in_brk = JPH_WheeledVehicleController_GetBrakeInput(controller);

    // 2. Drivetrain State
    float rpm = JPH_VehicleEngine_GetCurrentRPM(engine);
    
    // FIX: Pass actual throttle input to calculate generated torque. 
    // Passing 0.0f returns idle torque (usually 0).
    float engine_torque = JPH_VehicleEngine_GetTorque(engine, in_fwd); 
    
    int gear = JPH_VehicleTransmission_GetCurrentGear(trans);
    float clutch = JPH_VehicleTransmission_GetClutchFriction(trans);

    DEBUG_LOG("=== VEHICLE DEBUG STATE ===");
    DEBUG_LOG("  Inputs: Fwd=%.2f | Brk=%.2f", in_fwd, in_brk);
    DEBUG_LOG("  Engine: %.2f RPM | Torque: %.2f Nm", rpm, engine_torque);
    DEBUG_LOG("  Trans : Gear %d | Clutch Friction: %.2f", gear, clutch);

    // 3. Wheel State
    for (uint32_t i = 0; i < self->num_wheels; i++) {
        const JPH_Wheel* w = JPH_VehicleConstraint_GetWheel(self->vehicle, i);
        const JPH_WheelSettings* ws = JPH_Wheel_GetSettings(w);
        
        bool contact = JPH_Wheel_HasContact(w);
        float susp_len = JPH_Wheel_GetSuspensionLength(w);
        float ang_vel = JPH_Wheel_GetAngularVelocity(w);
        float radius = JPH_WheelSettings_GetRadius(ws);
        
        // Calculate linear speed of tire surface (m/s)
        float tire_speed = ang_vel * radius;

        // Lambda correlates to the force applied by the solver to the ground
        float long_lambda = JPH_Wheel_GetLongitudinalLambda(w);
        float lat_lambda = JPH_Wheel_GetLateralLambda(w);

        DEBUG_LOG("  Wheel %u: %s", i, contact ? "GROUND" : "AIR   ");
        DEBUG_LOG("    Susp: %.3fm | AngVel: %.2f rad/s | SurfSpd: %.2f m/s", 
                  susp_len, ang_vel, tire_speed);
        DEBUG_LOG("    Trac: Long=%.2f | Lat=%.2f", long_lambda, lat_lambda);
    }
    DEBUG_LOG("===========================");

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

    DEBUG_LOG("Destroying Vehicle Instance...");
    SHADOW_LOCK(&self->world->shadow_lock);

    // 1. Remove Constraint & Step Listener
    if (self->vehicle) {
        // Step Listener removal must happen BEFORE destroying the constraint
        JPH_PhysicsStepListener* step_listener = JPH_VehicleConstraint_AsPhysicsStepListener(self->vehicle);
        JPH_PhysicsSystem_RemoveStepListener(self->world->system, step_listener);

        JPH_PhysicsSystem_RemoveConstraint(self->world->system, (JPH_Constraint*)self->vehicle);
        JPH_Constraint_Destroy((JPH_Constraint*)self->vehicle);
        self->vehicle = NULL;
    }

    // 2. Destroy Collision Tester
    if (self->tester) {
        JPH_VehicleCollisionTester_Destroy(self->tester);
        self->tester = NULL;
    }

    // 3. Destroy Settings (Reverse creation order safe)
    if (self->controller_settings) {
        JPH_VehicleControllerSettings_Destroy(self->controller_settings);
        self->controller_settings = NULL;
    }
    
    // NEW: Destroy Transmission Settings explicitly
    if (self->transmission_settings) {
        JPH_VehicleTransmissionSettings_Destroy(self->transmission_settings);
        self->transmission_settings = NULL;
    }

    if (self->wheel_settings) {
        for (uint32_t i = 0; i < self->num_wheels; i++) {
            if (self->wheel_settings[i]) {
                JPH_WheelSettings_Destroy(self->wheel_settings[i]);
            }
        }
        PyMem_RawFree((void *)self->wheel_settings);
        self->wheel_settings = NULL;
    }

    if (self->friction_curve) { JPH_LinearCurve_Destroy(self->friction_curve); self->friction_curve = NULL; }
    if (self->torque_curve) { JPH_LinearCurve_Destroy(self->torque_curve); self->torque_curve = NULL; }

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

// --- Python Deallocation ---
static void Vehicle_dealloc(VehicleObject* self) {
    PyObject_GC_UnTrack(self);
    
    // Call destroy to clean up native resources
    Vehicle_destroy(self, NULL);
    
    // Decref the World (kept alive by Vehicle)
    Py_XDECREF(self->world);
    
    // Free the python object itself
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Character Methods ---

static void Character_dealloc(CharacterObject *self) {
  PyObject_GC_UnTrack(self);

  // Must remove from manager before Jolt object is destroyed
  if (self->world && self->world->char_vs_char_manager && self->character) {
      JPH_CharacterVsCharacterCollisionSimple_RemoveCharacter(self->world->char_vs_char_manager, self->character);
  }

  // Recycle the slot in the world
  uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
  SHADOW_LOCK(&self->world->shadow_lock);
  
  // Note: We don't need to manually remove the internal body from Jolt 
  // because JPH_CharacterBase_Destroy handles that. 
  // We just need to cleanup our dense map.
  uint32_t dense_idx = self->world->slot_to_dense[slot];
  uint32_t last_dense = (uint32_t)self->world->count - 1;
  
  if (dense_idx != last_dense) {
      // Swap and Pop logic to keep array dense
      self->world->body_ids[dense_idx] = self->world->body_ids[last_dense];
      uint32_t mover_slot = self->world->dense_to_slot[last_dense];
      self->world->slot_to_dense[mover_slot] = dense_idx;
      self->world->dense_to_slot[dense_idx] = mover_slot;
  }
  
  self->world->generations[slot]++;
  self->world->free_slots[self->world->free_count++] = slot;
  self->world->slot_states[slot] = SLOT_EMPTY;
  self->world->count--;
  self->world->view_shape[0] = (Py_ssize_t)self->world->count;
  
  SHADOW_UNLOCK(&self->world->shadow_lock);

  if (self->character) {
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)self->character);
  }
  if (self->listener) {
    JPH_CharacterContactListener_Destroy(self->listener);
  }
  if (self->body_filter) JPH_BodyFilter_Destroy(self->body_filter);
  if (self->shape_filter) JPH_ShapeFilter_Destroy(self->shape_filter);
  if (self->bp_filter) JPH_BroadPhaseLayerFilter_Destroy(self->bp_filter);
  if (self->obj_filter) JPH_ObjectLayerFilter_Destroy(self->obj_filter);

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
  float vx = NAN; float vy = NAN; float vz = NAN; float dt = NAN;
  static char *kwlist[] = {"velocity", "dt", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &vx, &vy, &vz,
                                   &dt)) {
    return NULL;
  }

  self->last_vx = vx; self->last_vy = vy; self->last_vz = vz;

  JPH_RVec3 current_pos;
  JPH_Quat current_rot;
  JPH_CharacterVirtual_GetPosition(self->character, &current_pos);
  JPH_CharacterVirtual_GetRotation(self->character, &current_rot);

  self->prev_px = (float)current_pos.x;
  self->prev_py = (float)current_pos.y;
  self->prev_pz = (float)current_pos.z;
  self->prev_rx = current_rot.x;
  self->prev_ry = current_rot.y;
  self->prev_rz = current_rot.z;
  self->prev_rw = current_rot.w;

  JPH_Vec3 v = {vx, vy, vz};
  JPH_CharacterVirtual_SetLinearVelocity(self->character, &v);

  JPH_STACK_ALLOC(JPH_ExtendedUpdateSettings, update_settings);
 *update_settings = (JPH_ExtendedUpdateSettings){0}; // C99 compound literal to zero the struct
  update_settings->stickToFloorStepDown.x = 0;
  update_settings->stickToFloorStepDown.y = -0.5f;
  update_settings->stickToFloorStepDown.z = 0;
  update_settings->walkStairsStepUp.x = 0;
  update_settings->walkStairsStepUp.y = 0.4f;
  update_settings->walkStairsStepUp.z = 0;
  update_settings->walkStairsMinStepForward = 0.02f;
  update_settings->walkStairsStepForwardTest = 0.15f;
  update_settings->walkStairsCosAngleForwardContact = 0.996f; 
  update_settings->walkStairsStepDownExtra.x = 0;
  update_settings->walkStairsStepDownExtra.y = 0;
  update_settings->walkStairsStepDownExtra.z = 0;

  JPH_CharacterVirtual_ExtendedUpdate(self->character, dt, update_settings, 1,
                                        self->world->system, self->body_filter,
                                        self->shape_filter);
  Py_RETURN_NONE;
}

static PyObject *Character_get_position(CharacterObject *self, PyObject *Py_UNUSED(ignored)) {
  JPH_RVec3 pos;
  JPH_CharacterVirtual_GetPosition(self->character, &pos);

  // 1. Allocate Tuple
  PyObject *ret = PyTuple_New(3);
  if (!ret) return NULL;

  // 2. Fill Tuple
  // Note: We remove the (float) cast. Python floats are doubles.
  // If Jolt is running in double precision, this preserves that precision.
  // PyTuple_SET_ITEM "steals" the reference, so no DECREF needed.
  PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble(pos.x));
  PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble(pos.y));
  PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(pos.z));

  return ret;
}

static PyObject *Character_set_position(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = NAN; float y = NAN; float z = NAN;
  static char *kwlist[] = {"pos", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)", kwlist, &x, &y, &z)) {
    return NULL;
  }
  JPH_RVec3 pos = {(double)x, (double)y, (double)z};
  JPH_CharacterVirtual_SetPosition(self->character, &pos);
  Py_RETURN_NONE;
}

static PyObject *Character_set_rotation(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = NAN; float y = NAN; float z = NAN; float w = NAN;
  static char *kwlist[] = {"rot", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ffff)", kwlist, &x, &y, &z,
                                   &w)) {
    return NULL;
  }
  JPH_Quat q = {x, y, z, w};
  JPH_CharacterVirtual_SetRotation(self->character, &q);
  Py_RETURN_NONE;
}

static PyObject *Character_is_grounded(CharacterObject *self, PyObject *args) {
  JPH_GroundState state =
      JPH_CharacterBase_GetGroundState((JPH_CharacterBase *)self->character);
  if (state == JPH_GroundState_OnGround ||
      state == JPH_GroundState_OnSteepGround) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *Character_set_strength(CharacterObject *self, PyObject *args) {
  float strength = NAN;
  if (!PyArg_ParseTuple(args, "f", &strength)) return NULL;
  self->push_strength = strength;
  JPH_CharacterVirtual_SetMaxStrength(self->character, strength);
  Py_RETURN_NONE;
}

// 1. Change signature to take PyObject* arg directly
static PyObject* Character_get_render_transform(CharacterObject* self, PyObject* arg) {
    // --- OPTIMIZATION 1: Fast Argument Parsing ---
    // METH_O passes the object directly. We simply convert it to double.
    double alpha_dbl = PyFloat_AsDouble(arg);
    
    // Check for conversion error (e.g. if arg wasn't a number)
    if (alpha_dbl == -1.0 && PyErr_Occurred()) {
        return NULL;
    }
    
    float alpha = (float)alpha_dbl;

    // 1. Clamp Alpha
    if (alpha < 0.0f) alpha = 0.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    // 2. Get Current State (Direct Jolt Access)
    JPH_RVec3 cur_p;
    JPH_Quat cur_r;
    JPH_CharacterVirtual_GetPosition(self->character, &cur_p);
    JPH_CharacterVirtual_GetRotation(self->character, &cur_r);

    // 3. Position LERP
    float px = self->prev_px + ((float)cur_p.x - self->prev_px) * alpha;
    float py = self->prev_py + ((float)cur_p.y - self->prev_py) * alpha;
    float pz = self->prev_pz + ((float)cur_p.z - self->prev_pz) * alpha;

    // 4. Rotation NLERP
    float q1x = self->prev_rx; float q1y = self->prev_ry; float q1z = self->prev_rz; float q1w = self->prev_rw;
    float q2x = cur_r.x;       float q2y = cur_r.y;       float q2z = cur_r.z;       float q2w = cur_r.w;

    float dot = q1x*q2x + q1y*q2y + q1z*q2z + q1w*q2w;
    if (dot < 0.0f) { q2x = -q2x; q2y = -q2y; q2z = -q2z; q2w = -q2w; }

    float rx = q1x + (q2x - q1x) * alpha;
    float ry = q1y + (q2y - q1y) * alpha;
    float rz = q1z + (q2z - q1z) * alpha;
    float rw = q1w + (q2w - q1w) * alpha;

    float mag_sq = rx*rx + ry*ry + rz*rz + rw*rw;
    if (mag_sq > 1e-9f) {
        float inv_len = 1.0f / sqrtf(mag_sq);
        rx *= inv_len; ry *= inv_len; rz *= inv_len; rw *= inv_len;
    } else {
        rx = 0.0f; ry = 0.0f; rz = 0.0f; rw = 1.0f;
    }

    // --- OPTIMIZATION 2: Manual Object Construction ---
    // This is faster than Py_BuildValue. 
    // PyTuple_SET_ITEM "steals" the reference, so we don't need to DECREF the floats.
    
    // Create Position Tuple (x, y, z)
    PyObject* pos = PyTuple_New(3);
    PyTuple_SET_ITEM(pos, 0, PyFloat_FromDouble(px));
    PyTuple_SET_ITEM(pos, 1, PyFloat_FromDouble(py));
    PyTuple_SET_ITEM(pos, 2, PyFloat_FromDouble(pz));

    // Create Rotation Tuple (x, y, z, w)
    PyObject* rot = PyTuple_New(4);
    PyTuple_SET_ITEM(rot, 0, PyFloat_FromDouble(rx));
    PyTuple_SET_ITEM(rot, 1, PyFloat_FromDouble(ry));
    PyTuple_SET_ITEM(rot, 2, PyFloat_FromDouble(rz));
    PyTuple_SET_ITEM(rot, 3, PyFloat_FromDouble(rw));

    // Create Result Tuple (pos, rot)
    PyObject* out = PyTuple_New(2);
    PyTuple_SET_ITEM(out, 0, pos);
    PyTuple_SET_ITEM(out, 1, rot);

    return out;
}

static PyObject* Character_get_handle(CharacterObject* self, void* closure) {
    return PyLong_FromUnsignedLongLong(self->handle);
}

static PyObject *get_positions(PhysicsWorldObject *self, void *c) {
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
static PyObject *get_count(PhysicsWorldObject *self, void *c) {
  return PyLong_FromSize_t(self->count);
}
static PyObject *get_time(PhysicsWorldObject *self, void *c) {
  return PyFloat_FromDouble(self->time);
}

// NEW: Buffer Release Slot
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