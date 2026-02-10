#include "culverin.h"

// Global lock for JPH callbacks
ShadowMutex g_jph_trampoline_lock; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

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

  // --- 3 LAYERS: 0=Static, 1=Dynamic, 2=VehicleRay ---
  self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(3, 3);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 0, 0);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 1, 1);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 2, 2);

  self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(3);
  
  // Matrix:
  // 0 (Static)  vs 1 (Dynamic) -> ON
  // 0 (Static)  vs 2 (Ray)     -> ON
  // 1 (Dynamic) vs 1 (Dynamic) -> ON
  // 1 (Dynamic) vs 2 (Ray)     -> OFF (Fixes self-collision)
  // 2 (Ray)     vs 2 (Ray)     -> OFF
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 0, 1);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 0, 2);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1);
  JPH_ObjectLayerPairFilterTable_DisableCollision(self->pair_filter, 1, 2);

  self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(
      self->bp_interface, 3, self->pair_filter, 3);

  JPH_PhysicsSystemSettings phys_settings = {
        .maxBodies = (uint32_t)limits.max_bodies,
        .maxBodyPairs = (uint32_t)limits.max_pairs, 
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

  self->contact_max_capacity = CONTACT_MAX_CAPACITY;
  self->contact_buffer = PyMem_RawMalloc(CONTACT_MAX_CAPACITY * sizeof(ContactEvent));
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

static PyObject *PhysicsWorld_apply_angular_impulse(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    uint64_t h;
    float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &h, &x, &y, &z)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    
    uint32_t slot = 0;
    if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid handle");
        return NULL;
    }

    JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
    JPH_Vec3 imp = {x, y, z};
    
    // Call Jolt's Angular Impulse API
    JPH_BodyInterface_AddAngularImpulse(self->body_interface, bid, &imp);
    JPH_BodyInterface_ActivateBody(self->body_interface, bid);

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_apply_force(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    uint64_t h; float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &h, &x, &y, &z)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    uint32_t slot = 0;
    if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return NULL;
    }
    JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
    JPH_Vec3 f = {x, y, z};
    JPH_BodyInterface_AddForce(self->body_interface, bid, &f); // Adds to accumulator
    JPH_BodyInterface_ActivateBody(self->body_interface, bid);
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_apply_torque(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    uint64_t h; float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &h, &x, &y, &z)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    uint32_t slot = 0;
    if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return NULL;
    }
    JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
    JPH_Vec3 t = {x, y, z};
    JPH_BodyInterface_AddTorque(self->body_interface, bid, &t);
    JPH_BodyInterface_ActivateBody(self->body_interface, bid);
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_gravity(PhysicsWorldObject *self, PyObject *args) {
    float x, y, z;
    if (!PyArg_ParseTuple(args, "(fff)", &x, &y, &z)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    
    JPH_Vec3 g = {x, y, z};
    JPH_PhysicsSystem_SetGravity(self->system, &g);
    
    // Wake up all bodies so they react to the new gravity direction immediately
    // (Optional, but usually expected)
    JPH_BodyInterface_ActivateBodies(self->body_interface, self->body_ids, self->count);

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_body_stats(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    uint64_t h;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    uint32_t slot = 0;
    if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }
    
    uint32_t i = self->slot_to_dense[slot];
    
    // Extract straight from shadow buffers (Fastest access)
    PyObject* pos = Py_BuildValue("(fff)", self->positions[(size_t)i*4], self->positions[i*4+1], self->positions[i*4+2]);
    PyObject* rot = Py_BuildValue("(ffff)", self->rotations[(size_t)i*4], self->rotations[i*4+1], self->rotations[i*4+2], self->rotations[i*4+3]);
    PyObject* vel = Py_BuildValue("(fff)", self->linear_velocities[(size_t)i*4], self->linear_velocities[i*4+1], self->linear_velocities[i*4+2]);
    
    SHADOW_UNLOCK(&self->shadow_lock);

    PyObject* ret = PyTuple_Pack(3, pos, rot, vel);
    Py_DECREF(pos); Py_DECREF(rot); Py_DECREF(vel);
    return ret; // Returns ((x,y,z), (x,y,z,w), (vx,vy,vz))
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

// Constraints

// Initialize defaults to avoid garbage data
static void params_init(ConstraintParams *p) {
  memset(p, 0, sizeof(ConstraintParams)); // Zero everything first
  p->ay = 1; // Default Up
  p->limit_min = -FLT_MAX;
  p->limit_max = FLT_MAX;
  p->frequency = 20.0f; // Default decent stiffness
  p->damping = 1.0f;
}

// --- Jolt Creator Helpers ---

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

static JPH_Constraint *create_hinge(const ConstraintParams *p, JPH_Body *b1, JPH_Body *b2) {
  JPH_HingeConstraintSettings s;
  JPH_HingeConstraintSettings_Init(&s);
  s.base.enabled = true;
  s.space = JPH_ConstraintSpace_WorldSpace;

  s.point1.x = p->px; s.point1.y = p->py; s.point1.z = p->pz;
  s.point2 = s.point1;

  JPH_Vec3 axis = {p->ax, p->ay, p->az};
  // ... (Keep existing normalization logic) ...
  float len_sq = axis.x * axis.x + axis.y * axis.y + axis.z * axis.z;
  if (len_sq > 1e-9f) JPH_Vec3_Normalize(&axis, &axis);
  else { axis.x = 0; axis.y = 1; axis.z = 0; }

  JPH_Vec3 norm;
  vec3_get_perpendicular(&axis, &norm);

  s.hingeAxis1 = axis; s.hingeAxis2 = axis;
  s.normalAxis1 = norm; s.normalAxis2 = norm;
  s.limitsMin = p->limit_min;
  s.limitsMax = p->limit_max;

  // --- MOTOR CONFIG ---
  if (p->has_motor) {
      s.motorSettings.springSettings.mode = (p->frequency > 0) ? JPH_SpringMode_FrequencyAndDamping : JPH_SpringMode_StiffnessAndDamping;
      s.motorSettings.springSettings.frequencyOrStiffness = p->frequency;
      s.motorSettings.springSettings.damping = p->damping;
      s.motorSettings.maxTorqueLimit = p->max_torque;
      s.motorSettings.minTorqueLimit = -p->max_torque;
  }

  JPH_HingeConstraint* c = JPH_HingeConstraint_Create(&s, b1, b2);
  
  // Apply Runtime State immediately
  if (c && p->has_motor && p->motor_type > 0) {
      JPH_HingeConstraint_SetMotorState(c, (JPH_MotorState)p->motor_type);
      if (p->motor_type == 1) JPH_HingeConstraint_SetTargetAngularVelocity(c, p->motor_target);
      if (p->motor_type == 2) JPH_HingeConstraint_SetTargetAngle(c, p->motor_target);
  }
  
  return (JPH_Constraint *)c;
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

static PyObject *PhysicsWorld_create_constraint(PhysicsWorldObject *self,
                                                PyObject *args,
                                                PyObject *kwds) {
  int type = 0;
  uint64_t h1 = 0;
  uint64_t h2 = 0;
  PyObject *params = NULL;
  PyObject *motor_dict = NULL; // NEW
  
  static char *kwlist[] = {"type", "body1", "body2", "params", "motor", NULL}; // Updated

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "iKK|OO", kwlist, &type, &h1, &h2,
                                   &params, &motor_dict)) {
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

  if (motor_dict) {
      parse_motor_config(motor_dict, &p);
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

static PyObject *PhysicsWorld_set_constraint_target(PhysicsWorldObject *self,
                                                    PyObject *args, PyObject *kwds) {
    uint64_t h = 0;
    float target = 0.0f;
    static char *kwlist[] = {"handle", "target", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kf", kwlist, &h, &target)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (!unpack_handle(self, h, &slot) || self->constraint_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid constraint handle");
        return NULL;
    }

    JPH_Constraint* c = self->constraints[slot];
    JPH_ConstraintType type = JPH_Constraint_GetType(c);
    JPH_ConstraintSubType sub = JPH_Constraint_GetSubType(c);

    // HINGE
    if (sub == JPH_ConstraintSubType_Hinge) {
        JPH_HingeConstraint* hc = (JPH_HingeConstraint*)c;
        JPH_MotorState state = JPH_HingeConstraint_GetMotorState(hc);
        if (state == JPH_MotorState_Velocity) {
            JPH_HingeConstraint_SetTargetAngularVelocity(hc, target);
        } else if (state == JPH_MotorState_Position) {
            JPH_HingeConstraint_SetTargetAngle(hc, target);
        }
    }
    // SLIDER
    else if (sub == JPH_ConstraintSubType_Slider) {
        JPH_SliderConstraint* sc = (JPH_SliderConstraint*)c;
        JPH_MotorState state = JPH_SliderConstraint_GetMotorState(sc);
        if (state == JPH_MotorState_Velocity) {
            JPH_SliderConstraint_SetTargetVelocity(sc, target);
        } else if (state == JPH_MotorState_Position) {
            JPH_SliderConstraint_SetTargetPosition(sc, target);
        }
    }

    // Wake up bodies!
    if (type == JPH_ConstraintType_TwoBodyConstraint) {
        JPH_Body* b1 = JPH_TwoBodyConstraint_GetBody1((JPH_TwoBodyConstraint*)c);
        JPH_Body* b2 = JPH_TwoBodyConstraint_GetBody2((JPH_TwoBodyConstraint*)c);
        JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b1));
        JPH_BodyInterface_ActivateBody(self->body_interface, JPH_Body_GetID(b2));
    }

    SHADOW_UNLOCK(&self->shadow_lock);
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
  double saved_time = 0.0f;

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
    self->contact_max_capacity = CONTACT_MAX_CAPACITY;
    self->contact_buffer =
        PyMem_RawMalloc(self->contact_max_capacity * sizeof(ContactEvent));
    if (UNLIKELY(!self->contact_buffer)) { // UNLIKELY OOM
      atomic_store_explicit(&self->is_stepping, false, memory_order_relaxed);
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
    }
  }

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
    .friction = mat.friction,
    .restitution = mat.restitution,
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
  size_t user_data = 0;
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

static PyObject *PhysicsWorld_set_position(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
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
  float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
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
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
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
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
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

static PyObject *PhysicsWorld_get_render_state(PhysicsWorldObject *self,
                                               PyObject *args) {
  float alpha = 0.0f;
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

static void RagdollSettings_dealloc(RagdollSettingsObject *self) {
  if (self->settings) {
    JPH_RagdollSettings_Destroy(self->settings);
  }
  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
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
    {"apply_angular_impulse", (PyCFunction)PhysicsWorld_apply_angular_impulse, 
     METH_VARARGS | METH_KEYWORDS, "Apply rotational momentum."},
    {"apply_impulse_at", (PyCFunction)PhysicsWorld_apply_impulse_at, 
     METH_VARARGS | METH_KEYWORDS, "Apply impulse at world position."},
    {"apply_force", (PyCFunction)PhysicsWorld_apply_force, 
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"apply_torque", (PyCFunction)PhysicsWorld_apply_torque, 
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_gravity", (PyCFunction)PhysicsWorld_set_gravity, 
     METH_VARARGS, NULL},
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
    {"set_constraint_target", (PyCFunction)PhysicsWorld_set_constraint_target, 
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
    {"get_body_stats", (PyCFunction)PhysicsWorld_get_body_stats, 
     METH_VARARGS | METH_KEYWORDS, NULL},

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