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

// Callback: Handle the collision settings AND Apply Impulse
static void JPH_API_CALL char_on_contact_added(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {
  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  CharacterObject *self = (CharacterObject *)userData;
  JPH_BodyInterface *bi = self->world->body_interface;

  if (JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
    return;
  }

  float vx = self->last_vx;
  float vy = self->last_vy;
  float vz = self->last_vz;

  // Normal points FROM Character TO Body.
  float dot =
      vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

  if (dot > 0.01f) {
    float factor = dot * self->push_strength;
    JPH_Vec3 impulse;
    impulse.x = contactNormal->x * factor;

    // Flatten Y: Only allow upward pushes (lifting), ignore downward (friction)
    float y_push = contactNormal->y * factor;
    impulse.y = (y_push > 0.0f) ? y_push : 0.0f;
    impulse.z = contactNormal->z * factor;

    JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
    JPH_BodyInterface_ActivateBody(bi, bodyID2);
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
    .OnCharacterContactAdded = NULL,
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

// --- Lifecycle: Deallocation ---
static void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
  if (self->system) {
    JPH_PhysicsSystem_Destroy(self->system);
  }
  if (self->shape_cache) {
    for (size_t i = 0; i < self->shape_cache_count; i++) {
      if (self->shape_cache[i].shape) {
        JPH_Shape_Destroy(self->shape_cache[i].shape);
      }
    }
    PyMem_RawFree(self->shape_cache);
  }
  if (self->job_system) {
    JPH_JobSystem_Destroy(self->job_system);
  }
  if (self->contact_listener) { JPH_ContactListener_Destroy(self->contact_listener); }
  if (self->contact_events) { PyMem_RawFree(self->contact_events); }

  PyMem_RawFree(self->positions);
  PyMem_RawFree(self->rotations);
  PyMem_RawFree(self->prev_positions);
  PyMem_RawFree(self->prev_rotations);
  PyMem_RawFree(self->linear_velocities);
  PyMem_RawFree(self->angular_velocities);
  PyMem_RawFree(self->body_ids);
  PyMem_RawFree(self->generations);
  PyMem_RawFree(self->slot_to_dense);
  PyMem_RawFree(self->dense_to_slot);
  PyMem_RawFree(self->free_slots);
  PyMem_RawFree(self->slot_states);
  PyMem_RawFree(self->command_queue);
  PyMem_RawFree(self->user_data);

#if PY_VERSION_HEX < 0x030D0000
  if (self->shadow_lock)
    PyThread_free_lock(self->shadow_lock);
#endif
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Lifecycle: Initialization ---
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                             PyObject *kwds) {
  static char *kwlist[] = {"settings", "bodies", NULL};
  PyObject *settings_dict = NULL;
  PyObject *bodies_list = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &settings_dict,
                                   &bodies_list)) {
    return -1;
  }
  
  // NEW: Initialize MemoryView Counter
  self->view_export_count = 0;

  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);

  PyObject *val_func = PyObject_GetAttrString(st->helper, "validate_settings");
  PyObject *norm_settings = PyObject_CallFunctionObjArgs(
      val_func, settings_dict ? settings_dict : Py_None, NULL);
  Py_DECREF(val_func);
  if (!norm_settings) {
    return -1;
  }

  float gx = NAN;
  float gy = NAN;
  float gz = NAN;
  float slop = NAN;
  int max_bodies = 0;
  int max_pairs = 0;
  PyArg_ParseTuple(norm_settings, "ffffii", &gx, &gy, &gz, &slop, &max_bodies,
                   &max_pairs);
  Py_DECREF(norm_settings);

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
      .maxBodies = 1000000,
      .maxBodyPairs = 1000000,
      .maxContactConstraints = 102400,
      .broadPhaseLayerInterface = self->bp_interface,
      .objectLayerPairFilter = self->pair_filter,
      .objectVsBroadPhaseLayerFilter = self->bp_filter};
  self->system = JPH_PhysicsSystem_Create(&phys_settings);
  JPH_Vec3 gravity = {gx, gy, gz};
  JPH_PhysicsSystem_SetGravity(self->system, &gravity);
  self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);

  // Init Contact Event Buffer
    self->contact_events = PyMem_RawMalloc(64 * sizeof(ContactEvent));
    self->contact_capacity = 64;
    self->contact_count = 0;

    // Create and Register Listener
    JPH_ContactListener_SetProcs(&contact_procs);
    self->contact_listener = JPH_ContactListener_Create(self); 
    JPH_PhysicsSystem_SetContactListener(self->system, self->contact_listener);

#if PY_VERSION_HEX < 0x030D0000
  self->shadow_lock = PyThread_allocate_lock();
#endif

  // --- ABI Safety Check (Double Precision) ---
  {
      JPH_BoxShapeSettings* bs = JPH_BoxShapeSettings_Create(&(JPH_Vec3){1,1,1}, 0.0f);
      JPH_Shape* shape = (JPH_Shape*)JPH_BoxShapeSettings_CreateShape(bs);
      JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)bs);

      JPH_BodyCreationSettings* bcs = JPH_BodyCreationSettings_Create3(
          shape, 
          &(JPH_RVec3){10.0, 20.0, 30.0}, 
          &(JPH_Quat){0,0,0,1}, 
          JPH_MotionType_Static, 0
      );
      JPH_Shape_Destroy(shape);
      JPH_BodyID bid = JPH_BodyInterface_CreateAndAddBody(self->body_interface, bcs, JPH_Activation_Activate);
      JPH_BodyCreationSettings_Destroy(bcs);

      JPH_STACK_ALLOC(JPH_RVec3, p_check);
      JPH_BodyInterface_GetPosition(self->body_interface, bid, p_check);
      
      JPH_BodyInterface_RemoveBody(self->body_interface, bid);
      JPH_BodyInterface_DestroyBody(self->body_interface, bid);

      // We expect 10.0, 20.0, 30.0
      // If JPH_RVec3 struct alignment is mismatched between ext and lib, this will read garbage.
      if (fabs(p_check->x - 10.0) > 0.1 || fabs(p_check->y - 20.0) > 0.1) {
          PyErr_SetString(PyExc_RuntimeError, 
            "JoltC ABI Mismatch: Library expects different floating point precision (Double vs Float). "
            "Recompile extension with/without JPH_DOUBLE_PRECISION.");
          return -1;
      }
  }

  PyObject *baked = NULL;
  size_t baked_count = 0;
  if (bodies_list && bodies_list != Py_None) {
    PyObject *bake_func = PyObject_GetAttrString(st->helper, "bake_scene");
    baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
    Py_DECREF(bake_func);
    if (!baked) {
      return -1;
    }
    baked_count = PyLong_AsSize_t(PyTuple_GetItem(baked, 0));
  }

  self->count = baked_count;
  self->capacity = (size_t)max_bodies;
  if (self->capacity < 128) self->capacity = 128;
  if (self->capacity < self->count) self->capacity = self->count + 1024;

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
  self->command_queue = PyMem_RawMalloc(64 * sizeof(PhysicsCommand));
  self->command_capacity = 64;
  self->command_count = 0;

  if (!self->positions || !self->rotations || !self->body_ids ||
      !self->linear_velocities || !self->angular_velocities ||
      !self->user_data || !self->generations || !self->slot_to_dense ||
      !self->dense_to_slot || !self->free_slots || !self->slot_states ||
      !self->command_queue || !self->prev_positions || !self->prev_rotations) {

    Py_XDECREF(baked);
    PyErr_NoMemory();
    return -1;
  }

  if (baked) {
    float *f_pos = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 1));
    float *f_rot = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 2));
    float *f_shape = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 3));
    unsigned char *u_mot =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 4));
    unsigned char *u_layer =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 5));
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
      JPH_Shape *shape =
          find_or_create_shape(self, (int)f_shape[i * 5], params);

      if (shape) {
        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
            shape, body_pos, body_rot, (JPH_MotionType)u_mot[i],
            (JPH_ObjectLayer)u_layer[i]);

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
        self->body_ids[i] = JPH_INVALID_BODY_ID;
      }
    }
    Py_DECREF(baked);
  }
  for (uint32_t i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity;
       i++) {
    self->generations[i] = 1;
    self->free_slots[self->free_count++] = i;
  }
  culverin_sync_shadow_buffers(self);
  return 0;
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
    static char *kwlist[] = {"start", "direction", "max_dist", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|f", kwlist, 
                                     &sx, &sy, &sz, 
                                     &dx, &dy, &dz, 
                                     &max_dist)) {
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

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit, NULL, NULL, NULL);

    if (!has_hit) Py_RETURN_NONE;

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

    SHADOW_LOCK(&self->shadow_lock);
    uint64_t slot_idx = JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);
    
    if (slot_idx >= self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    uint32_t gen = self->generations[slot_idx];
    BodyHandle handle = make_handle((uint32_t)slot_idx, gen);
    SHADOW_UNLOCK(&self->shadow_lock);

    return Py_BuildValue("Kf(fff)", handle, hit->fraction, normal.x, normal.y, normal.z);
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

  JPH_RVec3 pos = {(double)px, (double)py, (double)pz};
  JPH_Quat rot = {0, 0, 0, 1};

  JPH_CharacterVirtual *j_char =
      JPH_CharacterVirtual_Create(&settings, &pos, &rot, 0, self->system);
  JPH_Shape_Destroy(shape);

  if (!j_char) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create CharacterVirtual");
    return NULL;
  }

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

  obj->body_filter = JPH_BodyFilter_Create(NULL);
  obj->shape_filter = JPH_ShapeFilter_Create(NULL);
  obj->bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  obj->obj_filter = JPH_ObjectLayerFilter_Create(NULL);

  // NEW: Track with GC
  PyObject_GC_Track((PyObject *)obj);

  return (PyObject *)obj;
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

// --- Character Methods ---

static void Character_dealloc(CharacterObject *self) {
  // NEW: GC Untrack
  PyObject_GC_UnTrack(self);

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

  JPH_ExtendedUpdateSettings update_settings;
  memset(&update_settings, 0, sizeof(JPH_ExtendedUpdateSettings));
  update_settings.stickToFloorStepDown.x = 0;
  update_settings.stickToFloorStepDown.y = -0.5f;
  update_settings.stickToFloorStepDown.z = 0;
  update_settings.walkStairsStepUp.x = 0;
  update_settings.walkStairsStepUp.y = 0.4f;
  update_settings.walkStairsStepUp.z = 0;
  update_settings.walkStairsMinStepForward = 0.02f;
  update_settings.walkStairsStepForwardTest = 0.15f;
  update_settings.walkStairsCosAngleForwardContact = 0.996f; 
  update_settings.walkStairsStepDownExtra.x = 0;
  update_settings.walkStairsStepDownExtra.y = 0;
  update_settings.walkStairsStepDownExtra.z = 0;

  JPH_CharacterVirtual_ExtendedUpdate(self->character, dt, &update_settings, 1,
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

static const PyMethodDef PhysicsWorld_methods[] = {
    // --- Lifecycle ---
    {"step", (PyCFunction)PhysicsWorld_step, METH_VARARGS, NULL},
    {"create_body", (PyCFunction)PhysicsWorld_create_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)PhysicsWorld_destroy_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_mesh_body", (PyCFunction)PhysicsWorld_create_mesh_body,
     METH_VARARGS | METH_KEYWORDS, NULL},

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
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, // NEW: GC Flag
    .slots = (PyType_Slot *)Character_slots,
};

// --- Module Initialization ---

static int culverin_exec(PyObject *m) {
  CulverinState *st = get_culverin_state(m);
  JPH_Init();

  st->helper = PyImport_ImportModule("culverin._culverin");
  if (!st->helper) return -1;

  st->PhysicsWorldType = PyType_FromModuleAndSpec(m, (PyType_Spec *)&PhysicsWorld_spec, NULL);
  if (!st->PhysicsWorldType) return -1;
  if (PyModule_AddObject(m, "PhysicsWorld", st->PhysicsWorldType) < 0) {
    Py_DECREF(st->PhysicsWorldType);
    return -1;
  }
  Py_INCREF(st->PhysicsWorldType);

  st->CharacterType = PyType_FromModuleAndSpec(m, (PyType_Spec *)&Character_spec, NULL);
  if (!st->CharacterType) return -1;
  if (PyModule_AddObject(m, "Character", st->CharacterType) < 0) {
    Py_DECREF(st->CharacterType);
    return -1;
  }
  Py_INCREF(st->CharacterType);

  PyModule_AddIntConstant(m, "SHAPE_BOX", 0);
  PyModule_AddIntConstant(m, "SHAPE_SPHERE", 1);
  PyModule_AddIntConstant(m, "SHAPE_CAPSULE", 2);
  PyModule_AddIntConstant(m, "SHAPE_CYLINDER", 3);
  PyModule_AddIntConstant(m, "SHAPE_PLANE", 4);
  PyModule_AddIntConstant(m, "SHAPE_MESH", 5);

  PyModule_AddIntConstant(m, "MOTION_STATIC", 0);
  PyModule_AddIntConstant(m, "MOTION_KINEMATIC", 1);
  PyModule_AddIntConstant(m, "MOTION_DYNAMIC", 2);

  return 0;
}

static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
  CulverinState *st = get_culverin_state(m);
  Py_VISIT(st->helper);
  Py_VISIT(st->PhysicsWorldType);
  Py_VISIT(st->CharacterType);
  return 0;
}

static int culverin_clear(PyObject *m) {
  CulverinState *st = get_culverin_state(m);
  Py_CLEAR(st->helper);
  Py_CLEAR(st->PhysicsWorldType);
  Py_CLEAR(st->CharacterType);
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