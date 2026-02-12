#include "culverin_physics_world_internal.h"

static void free_new_buffers(NewBuffers *nb) {
  PyMem_RawFree(nb->pos);
  PyMem_RawFree(nb->rot);
  PyMem_RawFree(nb->ppos);
  PyMem_RawFree(nb->prot);
  PyMem_RawFree(nb->lvel);
  PyMem_RawFree(nb->avel);
  PyMem_RawFree(nb->bids);
  PyMem_RawFree(nb->udat);
  PyMem_RawFree(nb->gens);
  PyMem_RawFree(nb->s2d);
  PyMem_RawFree(nb->d2s);
  PyMem_RawFree(nb->stat);
  PyMem_RawFree(nb->free);
  PyMem_RawFree(nb->cats);
  PyMem_RawFree(nb->masks);
  PyMem_RawFree(nb->mats);
}

static int alloc_new_buffers(NewBuffers *nb, size_t cap) {
  memset(nb, 0, sizeof(NewBuffers));
  size_t f4 = cap * 4 * sizeof(float);

  nb->pos = PyMem_RawMalloc(f4);
  nb->rot = PyMem_RawMalloc(f4);
  nb->ppos = PyMem_RawMalloc(f4);
  nb->prot = PyMem_RawMalloc(f4);
  nb->lvel = PyMem_RawMalloc(f4);
  nb->avel = PyMem_RawMalloc(f4);

  nb->bids = PyMem_RawMalloc(cap * sizeof(JPH_BodyID));
  nb->udat = PyMem_RawMalloc(cap * sizeof(uint64_t));
  nb->gens = PyMem_RawMalloc(cap * sizeof(uint32_t));
  nb->s2d = PyMem_RawMalloc(cap * sizeof(uint32_t));
  nb->d2s = PyMem_RawMalloc(cap * sizeof(uint32_t));
  nb->stat = PyMem_RawMalloc(cap * sizeof(uint8_t));
  nb->free = PyMem_RawMalloc(cap * sizeof(uint32_t));
  nb->cats = PyMem_RawMalloc(cap * sizeof(uint32_t));
  nb->masks = PyMem_RawMalloc(cap * sizeof(uint32_t));
  nb->mats = PyMem_RawMalloc(cap * sizeof(uint32_t));

  if (!nb->pos || !nb->rot || !nb->ppos || !nb->prot || !nb->lvel ||
      !nb->avel || !nb->bids || !nb->udat || !nb->gens || !nb->s2d ||
      !nb->d2s || !nb->stat || !nb->free || !nb->cats || !nb->masks ||
      !nb->mats) {
    free_new_buffers(nb);
    return -1;
  }
  return 0;
}

static void migrate_and_init(PhysicsWorldObject *self, NewBuffers *nb,
                             size_t new_cap) {
  size_t stride = 4 * sizeof(float);
  if (self->count > 0) {
    memcpy(nb->pos, self->positions, self->count * stride);
    memcpy(nb->rot, self->rotations, self->count * stride);
    memcpy(nb->ppos, self->prev_positions, self->count * stride);
    memcpy(nb->prot, self->prev_rotations, self->count * stride);
    memcpy(nb->lvel, self->linear_velocities, self->count * stride);
    memcpy(nb->avel, self->angular_velocities, self->count * stride);
    memcpy(nb->bids, self->body_ids, self->count * sizeof(JPH_BodyID));
    memcpy(nb->udat, self->user_data, self->count * sizeof(uint64_t));
    memcpy(nb->cats, self->categories, self->count * sizeof(uint32_t));
    memcpy(nb->masks, self->masks, self->count * sizeof(uint32_t));
    memcpy(nb->mats, self->material_ids, self->count * sizeof(uint32_t));
  }

  memcpy(nb->gens, self->generations, self->slot_capacity * sizeof(uint32_t));
  memcpy(nb->s2d, self->slot_to_dense, self->slot_capacity * sizeof(uint32_t));
  memcpy(nb->d2s, self->dense_to_slot, self->slot_capacity * sizeof(uint32_t));
  memcpy(nb->stat, self->slot_states, self->slot_capacity * sizeof(uint8_t));
  memcpy(nb->free, self->free_slots, self->free_count * sizeof(uint32_t));

  for (size_t i = self->slot_capacity; i < new_cap; i++) {
    nb->gens[i] = 1;
    nb->stat[i] = SLOT_EMPTY;
    nb->free[self->free_count++] = (uint32_t)i;
  }
}

int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
  // 1. Validation
  if (self->view_export_count > 0) {
    PyErr_SetString(PyExc_BufferError,
                    "Cannot resize while views are exported.");
    return -1;
  }
  BLOCK_UNTIL_NOT_QUERYING(self);
  if (new_capacity <= self->capacity) {
    return 0;
  }

  // 2. Transactional Allocation
  NewBuffers nb;
  if (alloc_new_buffers(&nb, new_capacity) < 0) {
    PyErr_NoMemory();
    return -1;
  }

  // 3. Data Migration
  migrate_and_init(self, &nb, new_capacity);

  // 4. Commit: Free OLD, assign NEW
  PyMem_RawFree(self->positions);
  self->positions = nb.pos;
  PyMem_RawFree(self->rotations);
  self->rotations = nb.rot;
  PyMem_RawFree(self->prev_positions);
  self->prev_positions = nb.ppos;
  PyMem_RawFree(self->prev_rotations);
  self->prev_rotations = nb.prot;
  PyMem_RawFree(self->linear_velocities);
  self->linear_velocities = nb.lvel;
  PyMem_RawFree(self->angular_velocities);
  self->angular_velocities = nb.avel;

  PyMem_RawFree(self->body_ids);
  self->body_ids = nb.bids;
  PyMem_RawFree(self->user_data);
  self->user_data = nb.udat;
  PyMem_RawFree(self->generations);
  self->generations = nb.gens;
  PyMem_RawFree(self->slot_to_dense);
  self->slot_to_dense = nb.s2d;
  PyMem_RawFree(self->dense_to_slot);
  self->dense_to_slot = nb.d2s;
  PyMem_RawFree(self->slot_states);
  self->slot_states = nb.stat;
  PyMem_RawFree(self->free_slots);
  self->free_slots = nb.free;
  PyMem_RawFree(self->categories);
  self->categories = nb.cats;
  PyMem_RawFree(self->masks);
  self->masks = nb.masks;
  PyMem_RawFree(self->material_ids);
  self->material_ids = nb.mats;

  self->capacity = new_capacity;
  self->slot_capacity = new_capacity;
  return 0;
}

void free_constraints(PhysicsWorldObject *self) {
  if (self->constraints) {
    for (size_t i = 0; i < self->constraint_capacity; i++) {
      if (!self->constraints[i]) {
        continue;
      }

      bool is_alive =
          !self->constraint_states || self->constraint_states[i] == SLOT_ALIVE;
      if (is_alive) {
        if (self->system) {
          JPH_PhysicsSystem_RemoveConstraint(self->system,
                                             self->constraints[i]);
        }
        JPH_Constraint_Destroy(self->constraints[i]);
      }
      self->constraints[i] = NULL;
    }
    PyMem_RawFree((void *)self->constraints);
    self->constraints = NULL;
  }
  PyMem_RawFree(self->constraint_generations);
  self->constraint_generations = NULL;
  PyMem_RawFree(self->free_constraint_slots);
  self->free_constraint_slots = NULL;
  PyMem_RawFree(self->constraint_states);
  self->constraint_states = NULL;
}

void free_shadow_buffers(PhysicsWorldObject *self) {
  PyMem_RawFree(self->positions);
  self->positions = NULL;
  PyMem_RawFree(self->rotations);
  self->rotations = NULL;
  PyMem_RawFree(self->prev_positions);
  self->prev_positions = NULL;
  PyMem_RawFree(self->prev_rotations);
  self->prev_rotations = NULL;
  PyMem_RawFree(self->linear_velocities);
  self->linear_velocities = NULL;
  PyMem_RawFree(self->angular_velocities);
  self->angular_velocities = NULL;
  PyMem_RawFree(self->body_ids);
  self->body_ids = NULL;
  PyMem_RawFree(self->generations);
  self->generations = NULL;
  PyMem_RawFree(self->slot_to_dense);
  self->slot_to_dense = NULL;
  PyMem_RawFree(self->dense_to_slot);
  self->dense_to_slot = NULL;
  PyMem_RawFree(self->free_slots);
  self->free_slots = NULL;
  PyMem_RawFree(self->slot_states);
  self->slot_states = NULL;
  PyMem_RawFree(self->command_queue);
  self->command_queue = NULL;
  PyMem_RawFree(self->user_data);
  self->user_data = NULL;
  PyMem_RawFree(self->categories);
  self->categories = NULL;
  PyMem_RawFree(self->masks);
  self->masks = NULL;
  PyMem_RawFree(self->material_ids);
  self->material_ids = NULL;
  PyMem_RawFree(self->materials);
  self->materials = NULL;
}

// --- Helper: Resource Cleanup (Idempotent) ---
// SAFETY:
// - Must not be called while PhysicsSystem is stepping
// - Must not be called from a Jolt callback
// - Must not race with Python memoryview access
void PhysicsWorld_free_members(PhysicsWorldObject *self) {
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

// helper: Initialize settings via Python helper
int init_settings(PhysicsWorldObject *self, PyObject *settings_dict, float *gx,
                  float *gy, float *gz, int *max_bodies, int *max_pairs) {
  PyObject *st_module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(st_module);
  PyObject *val_func = PyObject_GetAttrString(st->helper, "validate_settings");
  if (!val_func) {
    return -1;
  }

  PyObject *norm = PyObject_CallFunctionObjArgs(
      val_func, settings_dict ? settings_dict : Py_None, NULL);
  Py_DECREF(val_func);
  if (!norm) {
    return -1;
  }

  float slop;
  int ok = PyArg_ParseTuple(norm, "ffffii", gx, gy, gz, &slop, max_bodies,
                            max_pairs);
  Py_DECREF(norm);
  return ok ? 0 : -1;
}

// helper: Initialize Jolt Core Systems
int init_jolt_core(PhysicsWorldObject *self, WorldLimits limits,
                   GravityVector gravity) {
  JobSystemThreadPoolConfig job_cfg = {
      .maxJobs = 1024, .maxBarriers = 8, .numThreads = -1};
  self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);

  // --- 3 LAYERS: 0=Static, 1=Dynamic, 2=VehicleRay ---
  self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(3, 3);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 0, 0);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 1, 1);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 2, 2);

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
      .maxContactConstraints = 1024 * 1024,
      .broadPhaseLayerInterface = self->bp_interface,
      .objectLayerPairFilter = self->pair_filter,
      .objectVsBroadPhaseLayerFilter = self->bp_filter};

  self->system = JPH_PhysicsSystem_Create(&phys_settings);
  self->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
  JPH_PhysicsSystem_SetGravity(self->system,
                               &(JPH_Vec3){gravity.gx, gravity.gy, gravity.gz});
  self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);
  return 0;
}

// helper: Allocate shadow buffers and indirection maps
int allocate_buffers(PhysicsWorldObject *self, int max_bodies) {
  self->capacity = (size_t)max_bodies;
  if (self->capacity < self->count + 128) {
    self->capacity = self->count + 1024;
  }
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
      !self->slot_states) {
    return -1;
  }

  for (size_t i = 0; i < self->capacity; i++) {
    self->categories[i] = 0xFFFF;
    self->masks[i] = 0xFFFF;
  }
  return 0;
}

// helper: Iterate over baked Python data to create initial Jolt bodies
// helper: Iterate over baked Python data to create initial Jolt bodies
int load_baked_scene(PhysicsWorldObject *self, PyObject *baked) {
  // 1. EXTRACT POINTERS (GIL HELD)
  // We assume the caller (Python) ensures these tuples/bytes are valid and match 'self->count'.
  
  float *f_pos = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 1));
  float *f_rot = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 2));
  float *f_shape = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 3));
  unsigned char *u_mot =
      (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 4));
  unsigned char *u_layer =
      (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 5));
  uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, 6));

  if (!f_pos || !f_rot || !f_shape || !u_mot || !u_layer || !u_data) {
      PyErr_SetString(PyExc_ValueError, "Invalid baked data structure");
      return -1;
  }

  int result = 0;

  // 2. ENTER CRITICAL SECTION (Release GIL)
  Py_BEGIN_ALLOW_THREADS

  // Acquire locks in the Safe Order: Jolt (Outer) -> Shadow (Inner)
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  SHADOW_LOCK(&self->shadow_lock);

  JPH_BodyInterface *bi = self->body_interface;

  for (size_t i = 0; i < self->count; i++) {
    // A. Shape Lookup (Safe: Both locks held)
    // Structure: [Type, P1, P2, P3, P4] per shape entry
    float params[4] = {f_shape[i * 5 + 1], f_shape[i * 5 + 2],
                       f_shape[i * 5 + 3], f_shape[i * 5 + 4]};
    
    JPH_Shape *shape = find_or_create_shape_locked(self, (int)f_shape[i * 5], params);
    
    if (UNLIKELY(!shape)) {
      result = -1;
      break; // Exit loop, cleanup locks below
    }

    // B. Create Settings
    // Note: f_pos stride appears to be 4 in your data (likely aligned Vec4 or X,Y,Z,Pad)
    JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
        shape, 
        &(JPH_RVec3){f_pos[i * 4], f_pos[i * 4 + 1], f_pos[i * 4 + 2]},
        &(JPH_Quat){f_rot[i * 4], f_rot[i * 4 + 1], f_rot[i * 4 + 2], f_rot[i * 4 + 3]},
        (JPH_MotionType)u_mot[i], 
        (JPH_ObjectLayer)u_layer[i]
    );

    // C. Metadata Setup
    self->generations[i] = 1;
    JPH_BodyCreationSettings_SetUserData(creation, (uint64_t)make_handle((uint32_t)i, 1));
    
    if (u_mot[i] == 2) { // KINEMATIC/DYNAMIC check
      JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
    }

    // D. Jolt Creation
    self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(bi, creation, JPH_Activation_Activate);
    
    // E. Shadow Updates
    uint32_t j_idx = JPH_ID_TO_INDEX(self->body_ids[i]);
    if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
      self->id_to_handle_map[j_idx] = make_handle((uint32_t)i, 1);
    }

    self->slot_to_dense[i] = (uint32_t)i;
    self->dense_to_slot[i] = (uint32_t)i;
    self->slot_states[i] = SLOT_ALIVE;
    self->user_data[i] = u_data[i];

    JPH_BodyCreationSettings_Destroy(creation);
  }

  // 3. EXIT CRITICAL SECTION
  SHADOW_UNLOCK(&self->shadow_lock);
  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  
  Py_END_ALLOW_THREADS

  if (result == -1) {
      PyErr_SetString(PyExc_RuntimeError, "Failed to create shape during baked load");
  }

  return result;
}

int verify_abi_alignment(JPH_BodyInterface *bi) {
  JPH_BoxShapeSettings *bs =
      JPH_BoxShapeSettings_Create(&(JPH_Vec3){1, 1, 1}, 0.0f);
  JPH_Shape *shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(bs);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)bs);
  if (!shape) {
    return -1;
  }

  JPH_BodyCreationSettings *bcs = JPH_BodyCreationSettings_Create3(
      shape, &(JPH_RVec3){10.0, 20.0, 30.0}, &(JPH_Quat){0, 0, 0, 1},
      JPH_MotionType_Static, 0);
  JPH_Shape_Destroy(shape);
  if (!bcs) {
    return -1;
  }

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

// Buffer Release Slot
void PhysicsWorld_releasebuffer(PhysicsWorldObject *self, Py_buffer *view) {
  SHADOW_LOCK(&self->shadow_lock);
  if (self->view_export_count > 0) {
    self->view_export_count--;
  }
  SHADOW_UNLOCK(&self->shadow_lock);
}
