#include "culverin.h"
#include "culverin_character.h"
#include "culverin_constraint.h"
#include "culverin_contact_listener.h"
#include "culverin_getters.h"
#include "culverin_parsers.h"
#include "culverin_physics_world_internal.h"
#include "culverin_query_methods.h"
#include "culverin_ragdoll.h"
#include "culverin_shadow_sync.h"
#include "culverin_vehicle.h"

// Global lock for JPH callbacks
NativeMutex 
    g_jph_trampoline_lock; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// --- Lifecycle: Deallocation ---
static void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
  PhysicsWorld_free_members(self);
  FREE_NATIVE_MUTEX(self->step_sync.mutex);
  FREE_NATIVE_COND(self->step_sync.cond);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Lifecycle: Initialization ---

// Orchestrator function
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                             PyObject *kwds) {
  PyObject *settings_dict = NULL;
  PyObject *bodies_list = NULL;
  PyObject *baked = NULL;
  float gx;
  float gy;
  float gz;
  int max_bodies;
  int max_pairs;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO",
                                   (char *[]){"settings", "bodies", NULL},
                                   &settings_dict, &bodies_list)) {
    return -1;
  }

  // 1. Initial State
  memset(((char *)self) + offsetof(PhysicsWorldObject, system), 0,
         sizeof(PhysicsWorldObject) - offsetof(PhysicsWorldObject, system));
  INIT_LOCK(self->shadow_lock);
  self->debug_renderer = JPH_DebugRenderer_Create(self);
  JPH_DebugRenderer_SetProcs(&debug_procs);
  atomic_init(&self->is_stepping, false);

  INIT_NATIVE_MUTEX(self->step_sync.mutex);
  INIT_NATIVE_COND(self->step_sync.cond);

  // 2. Settings & Jolt Init
  if (init_settings(self, settings_dict, &gx, &gy, &gz, &max_bodies,
                    &max_pairs) < 0) {
    goto fail;
  }
  WorldLimits limits = {max_bodies, max_pairs};
  GravityVector gravity = {gx, gy, gz};
  if (init_jolt_core(self, limits, gravity) < 0) {
    goto fail;
  }

  if (verify_abi_alignment(self->body_interface) < 0) {
    goto fail;
  }

  self->contact_max_capacity = CONTACT_MAX_CAPACITY;
  self->contact_buffer =
      PyMem_RawMalloc(CONTACT_MAX_CAPACITY * sizeof(ContactEvent));
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
    if (!baked) {
      goto fail;
    }
    self->count = PyLong_AsSize_t(PyTuple_GetItem(baked, 0));
  }

  if (allocate_buffers(self, max_bodies) < 0) {
    goto fail;
  }

  // 4. Constraints & Data Loading
  self->constraint_capacity = 256;
  self->constraints =
      (JPH_Constraint **)PyMem_RawCalloc(256, sizeof(JPH_Constraint *));
  self->constraint_generations = PyMem_RawCalloc(256, sizeof(uint32_t));
  self->free_constraint_slots = PyMem_RawMalloc(256 * sizeof(uint32_t));
  self->constraint_states = PyMem_RawCalloc(256, sizeof(uint8_t));
  if (!self->constraints || !self->free_constraint_slots) {
    goto fail;
  }

  for (uint32_t i = 0; i < 256; i++) {
    self->constraint_generations[i] = 1;
    self->free_constraint_slots[i] = i;
  }
  self->free_constraint_count = 256;

  if (baked && load_baked_scene(self, baked) < 0) {
    goto fail;
  }
  Py_XDECREF(baked);

  for (auto i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity;
       i++) {
    self->generations[i] = 1;
    self->free_slots[self->free_count++] = i;
  }

  culverin_sync_shadow_buffers(self);
  return 0;

fail:
  Py_XDECREF(baked);
  FREE_NATIVE_MUTEX(self->step_sync.mutex);
  FREE_NATIVE_COND(self->step_sync.cond);
  PhysicsWorld_free_members(self);
  return -1;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static PyObject *PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                            PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  uint64_t h;
  float x, y, z;

  // --- 1. VECTOR PARSING (Ultra Fast Path) ---
  // No tuple creation by the interpreter. We read directly from the stack.
  if (LIKELY(kwnames == NULL && nargs == 4)) {
    h = PyLong_AsUnsignedLongLong(args[0]);
    x = (float)PyFloat_AsDouble(args[1]);
    y = (float)PyFloat_AsDouble(args[2]);
    z = (float)PyFloat_AsDouble(args[3]);

    if (UNLIKELY(PyErr_Occurred()))
      return NULL;
  } else {
    // --- 2. FALLBACK (Slow Path) ---
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};
    PyObject *temp_tuple = PyTuple_New(nargs);
    if (!temp_tuple)
      return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(temp_tuple, i, args[i]);
    }
    int ok = PyArg_ParseTupleAndKeywords(temp_tuple, kwnames, "Kfff", kwlist,
                                         &h, &x, &y, &z);
    Py_DECREF(temp_tuple);
    if (!ok)
      return NULL;
  }

  if (UNLIKELY(!isfinite(x) || !isfinite(y) || !isfinite(z))) {
    PyErr_SetString(PyExc_ValueError, "Impulse components must be finite");
    return NULL;
  }

  // --- 3. EXECUTION ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (UNLIKELY(!unpack_handle(self, h, &slot) ||
               self->slot_states[slot] != SLOT_ALIVE)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_BodyID bid = self->body_ids[dense_idx];

  // Release GIL for the JPH call. In Free-Threaded Python, this is a massive
  // win as it prevents this thread from blocking the GC or other logic threads.
  Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
  JPH_BodyInterface_AddImpulse(self->body_interface, bid, &imp);
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  Py_END_ALLOW_THREADS

      SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static PyObject *PhysicsWorld_apply_impulse_at(PhysicsWorldObject *self,
                                               PyObject *const *args,
                                               size_t nargsf,
                                               PyObject *kwnames) {
  // Get actual number of positional arguments
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  uint64_t h;
  float ix, iy, iz;
  JPH_Real px, py, pz;

  // --- 1. FAST PATH: Positional only (No keywords, exactly 7 args) ---
  if (LIKELY(kwnames == NULL && nargs == 7)) {
    h = PyLong_AsUnsignedLongLong(args[0]);
    ix = (float)PyFloat_AsDouble(args[1]);
    iy = (float)PyFloat_AsDouble(args[2]);
    iz = (float)PyFloat_AsDouble(args[3]);
    px = PyFloat_AsDouble(args[4]);
    py = PyFloat_AsDouble(args[5]);
    pz = PyFloat_AsDouble(args[6]);

    if (UNLIKELY(PyErr_Occurred()))
      return NULL;
  } else {
    // --- 2. SLOW PATH: Keywords or wrong count ---
    // We use the standard public API by creating a temporary tuple.
    // This is only called when the user uses keywords (rare in hot loops).
    static char *kwlist[] = {"handle", "ix", "iy", "iz",
                             "px",     "py", "pz", NULL};

    PyObject *temp_tuple = PyTuple_New(nargs);
    if (!temp_tuple)
      return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(temp_tuple, i, args[i]);
    }

    int ok = PyArg_ParseTupleAndKeywords(temp_tuple, kwnames, "Kfffddd", kwlist,
                                         &h, &ix, &iy, &iz, &px, &py, &pz);
    Py_DECREF(temp_tuple);
    if (!ok)
      return NULL;
  }

  // --- 3. CONCURRENCY & EXECUTION ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (UNLIKELY(!unpack_handle(self, h, &slot) ||
               self->slot_states[slot] != SLOT_ALIVE)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

  Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {ix, iy, iz};
  JPH_RVec3 v_pos = {px, py, pz};

  JPH_BodyInterface_AddImpulse2(self->body_interface, bid, &imp, &v_pos);
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  Py_END_ALLOW_THREADS

      SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_apply_angular_impulse(PhysicsWorldObject *self,
                                                    PyObject *const *args,
                                                    size_t nargsf,
                                                    PyObject *kwnames) {
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  uint64_t h;
  float x, y, z;

  // --- 1. FAST PATH (Zero Allocation) ---
  if (LIKELY(kwnames == NULL && nargs == 4)) {
    h = PyLong_AsUnsignedLongLong(args[0]);
    x = (float)PyFloat_AsDouble(args[1]);
    y = (float)PyFloat_AsDouble(args[2]);
    z = (float)PyFloat_AsDouble(args[3]);

    if (UNLIKELY(PyErr_Occurred()))
      return NULL;
  } else {
    // --- 2. FALLBACK (Keyword/Count Mismatch) ---
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};
    PyObject *temp_tuple = PyTuple_New(nargs);
    if (!temp_tuple)
      return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(temp_tuple, i, args[i]);
    }
    int ok = PyArg_ParseTupleAndKeywords(temp_tuple, kwnames, "Kfff", kwlist,
                                         &h, &x, &y, &z);
    Py_DECREF(temp_tuple);
    if (!ok)
      return NULL;
  }

  // Safety check for physics stability
  if (UNLIKELY(!isfinite(x) || !isfinite(y) || !isfinite(z))) {
    PyErr_SetString(PyExc_ValueError,
                    "Angular impulse components must be finite");
    return NULL;
  }

  // --- 3. CONCURRENCY & JOLT EXECUTION ---
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (UNLIKELY(!unpack_handle(self, h, &slot) ||
               self->slot_states[slot] != SLOT_ALIVE)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

  Py_BEGIN_ALLOW_THREADS JPH_Vec3 imp = {x, y, z};
  JPH_BodyInterface_AddAngularImpulse(self->body_interface, bid, &imp);
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  Py_END_ALLOW_THREADS

      SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_apply_force(PhysicsWorldObject *self,
                                          PyObject *const *args, size_t nargsf,
                                          PyObject *kwnames) {
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  uint64_t h;
  float x, y, z;

  // --- 1. FAST PATH (Zero Allocation) ---
  if (LIKELY(kwnames == NULL && nargs == 4)) {
    h = PyLong_AsUnsignedLongLong(args[0]);
    x = (float)PyFloat_AsDouble(args[1]);
    y = (float)PyFloat_AsDouble(args[2]);
    z = (float)PyFloat_AsDouble(args[3]);

    if (UNLIKELY(PyErr_Occurred()))
      return NULL;
  } else {
    // --- 2. FALLBACK (Slow Path) ---
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};
    PyObject *temp_tuple = PyTuple_New(nargs);
    if (!temp_tuple)
      return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(temp_tuple, i, args[i]);
    }
    int ok = PyArg_ParseTupleAndKeywords(temp_tuple, kwnames, "Kfff", kwlist,
                                         &h, &x, &y, &z);
    Py_DECREF(temp_tuple);
    if (!ok)
      return NULL;
  }

  // Safety: NaNs in forces can instantly delete the entire world broadphase
  if (UNLIKELY(!isfinite(x) || !isfinite(y) || !isfinite(z))) {
    PyErr_SetString(PyExc_ValueError, "Force components must be finite");
    return NULL;
  }

  // --- 3. CONCURRENCY & EXECUTION ---
  SHADOW_LOCK(&self->shadow_lock);

  // We use the priority-aware blocking we implemented for the 1M body test
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (UNLIKELY(!unpack_handle(self, h, &slot) ||
               self->slot_states[slot] != SLOT_ALIVE)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

  Py_BEGIN_ALLOW_THREADS JPH_Vec3 f = {x, y, z};
  // Adds to the accumulator (cleared after every world.step())
  JPH_BodyInterface_AddForce(self->body_interface, bid, &f);
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  Py_END_ALLOW_THREADS

      SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_apply_torque(PhysicsWorldObject *self,
                                           PyObject *const *args, size_t nargsf,
                                           PyObject *kwnames) {
  Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
  uint64_t h;
  float x, y, z;

  // --- 1. FAST PATH (Zero-Allocation Argument Extraction) ---
  if (LIKELY(kwnames == NULL && nargs == 4)) {
    h = PyLong_AsUnsignedLongLong(args[0]);
    x = (float)PyFloat_AsDouble(args[1]);
    y = (float)PyFloat_AsDouble(args[2]);
    z = (float)PyFloat_AsDouble(args[3]);

    if (UNLIKELY(PyErr_Occurred()))
      return NULL;
  } else {
    // --- 2. FALLBACK (Keyword/Count Mismatch) ---
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};
    PyObject *temp_tuple = PyTuple_New(nargs);
    if (!temp_tuple)
      return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
      Py_INCREF(args[i]);
      PyTuple_SET_ITEM(temp_tuple, i, args[i]);
    }
    int ok = PyArg_ParseTupleAndKeywords(temp_tuple, kwnames, "Kfff", kwlist,
                                         &h, &x, &y, &z);
    Py_DECREF(temp_tuple);
    if (!ok)
      return NULL;
  }

  // Safety: Prevent Jolt broadphase corruption from invalid math
  if (UNLIKELY(!isfinite(x) || !isfinite(y) || !isfinite(z))) {
    PyErr_SetString(PyExc_ValueError, "Torque components must be finite");
    return NULL;
  }

  // --- 3. CONCURRENCY & JPH EXECUTION ---
  SHADOW_LOCK(&self->shadow_lock);

  // Use the priority-aware blocking pattern
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_IF_STEP_PENDING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  uint32_t slot = 0;
  if (UNLIKELY(!unpack_handle(self, h, &slot) ||
               self->slot_states[slot] != SLOT_ALIVE)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  // Look up ID while under shadow_lock
  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];

  Py_BEGIN_ALLOW_THREADS JPH_Vec3 t = {x, y, z};
  // Adds to torque accumulator (cleared per world.step)
  JPH_BodyInterface_AddTorque(self->body_interface, bid, &t);
  // Ensure body is awake to process the new torque
  JPH_BodyInterface_ActivateBody(self->body_interface, bid);
  Py_END_ALLOW_THREADS

      SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_gravity(PhysicsWorldObject *self,
                                          PyObject *args) {
  float x;
  float y;
  float z;
  if (!PyArg_ParseTuple(args, "(fff)", &x, &y, &z)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  JPH_Vec3 g = {x, y, z};
  JPH_PhysicsSystem_SetGravity(self->system, &g);

  if (self->count > UINT32_MAX) {
    PyErr_SetString(PyExc_OverflowError, "Body count exceeds Jolt's 32-bit limit. Please make your world sane.");
    return NULL;
  }

  // Wake up all bodies so they react to the new gravity direction immediately
  // (Optional, but usually expected)
  JPH_BodyInterface_ActivateBodies(self->body_interface, self->body_ids,
                                   (uint32_t)self->count);

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_body_stats(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  uint64_t h;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  
  // Consistency Guard: Ensure we aren't reading while the world is swapping buffers
  BLOCK_UNTIL_NOT_STEPPING(self);

  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
  }

  uint32_t i = self->slot_to_dense[slot];

  // Cast to stride structs for safe, indexed access
  auto *shadow_pos = (PosStride *)self->positions;
  auto *shadow_rot = (AuxStride *)self->rotations;
  auto *shadow_vel = (AuxStride *)self->linear_velocities;

  // Snapshot the values while holding the lock
  PosStride p = shadow_pos[i];
  AuxStride r = shadow_rot[i];
  AuxStride v = shadow_vel[i];

  SHADOW_UNLOCK(&self->shadow_lock);

  // Build the Python objects
  // Note: We use "ddd" for position to support JPH_DOUBLE_PRECISION (double)
  PyObject *py_pos = Py_BuildValue("(ddd)", (double)p.x, (double)p.y, (double)p.z);
  PyObject *py_rot = Py_BuildValue("(dddd)", (double)r.x, (double)r.y, (double)r.z, (double)r.w);
  PyObject *py_vel = Py_BuildValue("(ddd)", (double)v.x, (double)v.y, (double)v.z);

  if (!py_pos || !py_rot || !py_vel) {
      Py_XDECREF(py_pos);
      Py_XDECREF(py_rot);
      Py_XDECREF(py_vel);
      return NULL;
  }

  PyObject *ret = PyTuple_Pack(3, py_pos, py_rot, py_vel);
  
  // Clean up local references
  Py_DECREF(py_pos);
  Py_DECREF(py_rot);
  Py_DECREF(py_vel);
  
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

  return PyBool_FromLong((int)submerged);
}

static PyObject *PhysicsWorld_apply_buoyancy_batch(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
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
      "handles",      "surface_y", "buoyancy",       "linear_drag",
      "angular_drag", "dt",        "fluid_velocity", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*|fffff(fff)", kwlist, &h_view,
                                   &surface_y, &buoyancy, &lin_drag, &ang_drag,
                                   &dt, &vx, &vy, &vz)) {
    return NULL;
  }

  // 1. Validation
  if (h_view.itemsize != 8) {
    PyBuffer_Release(&h_view);
    return PyErr_Format(PyExc_ValueError,
                        "Handle buffer must be uint64 (itemsize=8), got %zd",
                        h_view.itemsize);
  }

  auto count = (size_t)h_view.len / 8;
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
    JPH_BodyInterface_ApplyBuoyancyImpulse(bi, bid, surf_pos, surf_norm,
                                           buoyancy, lin_drag, ang_drag,
                                           fluid_vel, &gravity, dt);
  }

  PyMem_RawFree(ids);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_save_state(PhysicsWorldObject *self,
                                         PyObject *Py_UNUSED(unused)) {
  SHADOW_LOCK(&self->shadow_lock);

  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // 1. Unambiguous Size Calculation using Stride Structs
  size_t header_size = sizeof(size_t) /* count */ + 
                       sizeof(double) /* time */ +
                       sizeof(size_t) /* slot_capacity */;

  // Stride 3 for Positions, Stride 4 for Rot/Vel/AngVel
  size_t pos_size_total = self->count * sizeof(PosStride);
  size_t aux_size_total = self->count * sizeof(AuxStride);

  size_t mapping_size =
      self->slot_capacity *
      (sizeof(uint32_t) * 3 /* gen, s2d, d2s */ + sizeof(uint8_t) /* states */
      );

  // Total = Header + (1 * PosStride) + (3 * AuxStride) + Mapping
  size_t total_size = header_size + pos_size_total + (3 * aux_size_total) + mapping_size;

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

  // 3. Encode Dense Buffers (Stride Sensitive)
  // Position (Stride 3)
  memcpy(ptr, self->positions, pos_size_total);
  ptr += pos_size_total;
  
  // Rotation (Stride 4)
  memcpy(ptr, self->rotations, aux_size_total);
  ptr += aux_size_total;
  
  // Linear Velocity (Stride 4)
  memcpy(ptr, self->linear_velocities, aux_size_total);
  ptr += aux_size_total;
  
  // Angular Velocity (Stride 4)
  memcpy(ptr, self->angular_velocities, aux_size_total);
  ptr += aux_size_total;

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
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static PyObject *PhysicsWorld_load_state(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  Py_buffer view;
  static char *kwlist[] = {"state", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*", kwlist, &view)) {
    return NULL;
  }

  // 1. IMMEDIATE SNAPSHOT (GIL held)
  // Copy to local heap so we can release the Python buffer before yielding/waiting.
  void *local_state_copy = PyMem_RawMalloc(view.len);
  if (!local_state_copy) {
    PyBuffer_Release(&view);
    return PyErr_NoMemory();
  }
  memcpy(local_state_copy, view.buf, view.len);
  size_t total_len = (size_t)view.len;
  PyBuffer_Release(&view);

  SHADOW_LOCK(&self->shadow_lock);

  // 2. CONCURRENCY GUARD
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // 3. HEADER EXTRACTION
  auto *ptr = (char *)local_state_copy;
  if (total_len < (sizeof(size_t) * 2 + sizeof(double))) {
    goto size_fail;
  }

  size_t saved_count = 0;
  size_t saved_slot_cap = 0;
  double saved_time = 0.0;

  memcpy(&saved_count, ptr, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(&saved_time, ptr, sizeof(double));
  ptr += sizeof(double);
  memcpy(&saved_slot_cap, ptr, sizeof(size_t));
  ptr += sizeof(size_t);

  // CRITICAL: Slot capacity must match exactly
  if (saved_slot_cap != self->slot_capacity) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyMem_RawFree(local_state_copy);
    PyErr_Format(PyExc_ValueError,
                 "Capacity mismatch: World is %zu, Snapshot is %zu",
                 self->slot_capacity, saved_slot_cap);
    return NULL;
  }

  // 4. FULL SIZE VALIDATION (Stride Sensitive)
  size_t pos_bytes = saved_count * sizeof(PosStride);
  size_t aux_bytes = saved_count * sizeof(AuxStride);
  size_t mapping_bytes = saved_slot_cap * (sizeof(uint32_t) * 3 + 1);

  size_t expected = (sizeof(size_t) * 2 + sizeof(double)) +
                    pos_bytes + (aux_bytes * 3) + mapping_bytes;

  if (total_len != expected) {
    goto size_fail;
  }

  // 5. RESTORE SHADOW STATE
  self->count = saved_count;
  self->time = saved_time;
  self->view_shape[0] = (Py_ssize_t)self->count;

  // Multi-Stride copies
  memcpy(self->positions, ptr, pos_bytes);           ptr += pos_bytes;
  memcpy(self->rotations, ptr, aux_bytes);           ptr += aux_bytes;
  memcpy(self->linear_velocities, ptr, aux_bytes);   ptr += aux_bytes;
  memcpy(self->angular_velocities, ptr, aux_bytes);  ptr += aux_bytes;

  // Mapping Tables
  memcpy(self->generations, ptr, self->slot_capacity * sizeof(uint32_t));
  ptr += self->slot_capacity * sizeof(uint32_t);
  memcpy(self->slot_to_dense, ptr, self->slot_capacity * sizeof(uint32_t));
  ptr += self->slot_capacity * sizeof(uint32_t);
  memcpy(self->dense_to_slot, ptr, self->slot_capacity * sizeof(uint32_t));
  ptr += self->slot_capacity * sizeof(uint32_t);
  memcpy(self->slot_states, ptr, self->slot_capacity);

  // 6. HANDLE INVALIDATION
  // Increment generations so old Python handles become invalid.
  for (size_t i = 0; i < self->slot_capacity; i++) {
    self->generations[i]++;
  }

  // 7. REBUILD FREE LIST
  self->free_count = 0;
  for (uint32_t i = 0; i < (uint32_t)self->slot_capacity; i++) {
    if (self->slot_states[i] == SLOT_EMPTY) {
      self->free_slots[self->free_count++] = i;
    }
  }

  // Cast internal buffers to stride structs for the Sync Phase
  auto *shadow_pos = (PosStride *)self->positions;
  auto *shadow_rot = (AuxStride *)self->rotations;
  auto *shadow_lvel = (AuxStride *)self->linear_velocities;
  auto *shadow_avel = (AuxStride *)self->angular_velocities;

  // 8. JOLT SYNC (Unlocked to prevent deadlocks)
  // Snapshot the current BodyID table while locked
  JPH_BodyID *bids = self->body_ids;
  JPH_BodyInterface *bi = self->body_interface;

  SHADOW_UNLOCK(&self->shadow_lock);

  for (size_t i = 0; i < saved_count; i++) {
    JPH_BodyID bid = bids[i];
    if (bid == JPH_INVALID_BODY_ID) continue;

    // Use Stride Structs for safe coordinate extraction
    JPH_RVec3 p = { shadow_pos[i].x, shadow_pos[i].y, shadow_pos[i].z };
    JPH_Quat q = { shadow_rot[i].x, shadow_rot[i].y, shadow_rot[i].z, shadow_rot[i].w };
    JPH_Vec3 lv = { shadow_lvel[i].x, shadow_lvel[i].y, shadow_lvel[i].z };
    JPH_Vec3 av = { shadow_avel[i].x, shadow_avel[i].y, shadow_avel[i].z };

    JPH_BodyInterface_SetPositionAndRotation(bi, bid, &p, &q, JPH_Activation_Activate);
    JPH_BodyInterface_SetLinearVelocity(bi, bid, &lv);
    JPH_BodyInterface_SetAngularVelocity(bi, bid, &av);

    // Re-Sync UserData to the newly incremented generations
    uint32_t slot = self->dense_to_slot[i];
    BodyHandle new_h = make_handle(slot, self->generations[slot]);
    JPH_BodyInterface_SetUserData(bi, bid, (uint64_t)new_h);
    
    // Fast handle map update
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);
    if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
        self->id_to_handle_map[j_idx] = new_h;
    }
  }

  PyMem_RawFree(local_state_copy);
  Py_RETURN_NONE;

size_fail:
  SHADOW_UNLOCK(&self->shadow_lock);
  PyMem_RawFree(local_state_copy);
  PyErr_SetString(PyExc_ValueError, "Snapshot buffer truncated or stride mismatch");
  return NULL;
}

static PyObject *PhysicsWorld_step(PhysicsWorldObject *self, PyObject *args) {
    float dt = 1.0f / 60.0f;
    if (UNLIKELY(!PyArg_ParseTuple(args, "|f", &dt))) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    atomic_store_explicit(&self->step_requested, true, memory_order_relaxed);

    BLOCK_UNTIL_NOT_QUERYING(self);
    BLOCK_UNTIL_NOT_STEPPING(self);
    atomic_store_explicit(&self->is_stepping, true, memory_order_relaxed);

    // --- SNAPSHOT REMOVED ---
    // The 40MB memcpy is gone. The lock is now extremely "thin".

    // Double-buffer swap (Recycling logic)
    PhysicsCommand *captured_queue = self->command_queue;
    size_t captured_count = self->command_count;
    if (UNLIKELY(!self->command_queue_spare || self->command_capacity > self->spare_capacity)) {
        PyMem_RawFree(self->command_queue_spare);
        self->command_queue_spare = (PhysicsCommand*)PyMem_RawMalloc(self->command_capacity * sizeof(PhysicsCommand));
        self->spare_capacity = self->command_capacity;
    }
    self->command_queue = self->command_queue_spare;
    self->command_queue_spare = captured_queue; 
    self->command_count = 0;

    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- HEAVY LIFTING ---
    Py_BEGIN_ALLOW_THREADS 
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock); 

    if (captured_count > 0) {
        flush_commands_internal(self, captured_queue, captured_count);
    }

    JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);

    // The Snapshot + Sync now happen together in one GIL-free loop
    culverin_sync_shadow_buffers(self); 

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

    // --- FINALIZATION ---
    SHADOW_LOCK(&self->shadow_lock);
    size_t c_idx = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    self->contact_count = (c_idx > self->contact_max_capacity) ? self->contact_max_capacity : c_idx;

    NATIVE_MUTEX_LOCK(self->step_sync.mutex);
    atomic_store_explicit(&self->is_stepping, false, memory_order_release);
    atomic_store_explicit(&self->step_requested, false, memory_order_release);
    NATIVE_COND_BROADCAST(self->step_sync.cond);
    NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);

    self->time += (double)dt;
    SHADOW_UNLOCK(&self->shadow_lock);

    Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static PyObject *PhysicsWorld_create_convex_hull(PhysicsWorldObject *self,
                                                 PyObject *args,
                                                 PyObject *kwds) {
  JPH_Real px = 0.0, py = 0.0, pz = 0.0;
  float rx = 0.0f, ry = 0.0f, rz = 0.0f, rw = 1.0f;
  Py_buffer points_view = {0};
  uint64_t user_data = 0;
  int motion_type = 2; 
  float mass = -1.0f;
  uint32_t category = 0xFFFF, mask = 0xFFFF, material_id = 0;
  float friction = 0.2f, restitution = 0.0f;
  int use_ccd = 0;

  static char *kwlist[] = {"pos", "rot", "points", "motion", "mass", "user_data", 
                           "category", "mask", "material_id", "friction", 
                           "restitution", "ccd", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ddd)(ffff)y*|ifKIIffp", kwlist, 
                                   &px, &py, &pz, &rx, &ry, &rz, &rw, &points_view, 
                                   &motion_type, &mass, &user_data, &category,
                                   &mask, &material_id, &friction, &restitution, &use_ccd)) {
    return NULL;
  }

  // 1. Buffer Validation & Jolt Point Conversion
  if (points_view.len % (3 * sizeof(float)) != 0) {
    PyBuffer_Release(&points_view);
    return PyErr_Format(PyExc_ValueError, "Points buffer must be 3 * float32");
  }

  size_t num_points = points_view.len / (3 * sizeof(float));
  if (num_points < 3) {
    PyBuffer_Release(&points_view);
    return PyErr_Format(PyExc_ValueError, "Convex Hull requires at least 3 points");
  }

  JPH_Vec3 *jolt_points = PyMem_RawMalloc(num_points * sizeof(JPH_Vec3));
  float *raw_floats = (float *)points_view.buf;
  for (size_t i = 0; i < num_points; i++) {
    jolt_points[i].x = raw_floats[i * 3 + 0];
    jolt_points[i].y = raw_floats[i * 3 + 1];
    jolt_points[i].z = raw_floats[i * 3 + 2];
  }
  PyBuffer_Release(&points_view);

  // 2. Create Shape (Heavy math, outside Shadow Lock)
  JPH_ConvexHullShapeSettings *hull_settings = JPH_ConvexHullShapeSettings_Create(jolt_points, (uint32_t)num_points, 0.05f);
  PyMem_RawFree(jolt_points);

  if (!hull_settings) return PyErr_Format(PyExc_RuntimeError, "Failed to allocate Hull Settings");
  JPH_Shape *shape = (JPH_Shape *)JPH_ConvexHullShapeSettings_CreateShape(hull_settings);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hull_settings);

  if (!shape) return PyErr_Format(PyExc_RuntimeError, "Failed to build Convex Hull (Degenerate data?)");

  // 3. Jolt Settings Prep (Outside Shadow Lock)
  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, &(JPH_RVec3){(double)px, (double)py, (double)pz},
      &(JPH_Quat){rx, ry, rz, rw}, (JPH_MotionType)motion_type, (motion_type == 0) ? 0 : 1);

  // Apply properties
  if (mass > 0.0f) {
    JPH_MassProperties mp;
    JPH_Shape_GetMassProperties(shape, &mp);
    float scale = mass / fmaxf(mp.mass, 1e-6f);
    mp.mass = mass;
    for (int i = 0; i < 3; i++) {
      mp.inertia.column[i].x *= scale; mp.inertia.column[i].y *= scale; mp.inertia.column[i].z *= scale;
    }
    JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
    JPH_BodyCreationSettings_SetOverrideMassProperties(settings, JPH_OverrideMassProperties_CalculateInertia);
  }
  JPH_BodyCreationSettings_SetFriction(settings, friction);
  JPH_BodyCreationSettings_SetRestitution(settings, restitution);
  if (use_ccd) JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);

  // 4. COMMIT PHASE (Shadow Lock)
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // Ensure Capacity (Check both slots and dense memory)
  if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
    size_t needed = (self->capacity == 0) ? 1024 : self->capacity * 2;
    if (PhysicsWorld_resize(self, needed) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_BodyCreationSettings_Destroy(settings);
      JPH_Shape_Destroy(shape);
      return NULL;
    }
  }

  uint32_t slot = self->free_slots[--self->free_count];
  uint32_t dense = (uint32_t)self->count++;
  BodyHandle handle = make_handle(slot, self->generations[slot]);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  // --- IMMEDIATE SHADOW WRITE (Stride Sensitive) ---
  auto *shadow_pos = (PosStride *)self->positions;
  auto *shadow_ppos = (PosStride *)self->prev_positions;
  auto *shadow_rot = (AuxStride *)self->rotations;
  auto *shadow_prot = (AuxStride *)self->prev_rotations;
  auto *shadow_lvel = (AuxStride *)self->linear_velocities;
  auto *shadow_avel = (AuxStride *)self->angular_velocities;

  PosStride p = {px, py, pz};
  shadow_pos[dense] = p;
  shadow_ppos[dense] = p;

  AuxStride q = {rx, ry, rz, rw};
  shadow_rot[dense] = q;
  shadow_prot[dense] = q;

  AuxStride zero = {0, 0, 0, 0};
  shadow_lvel[dense] = zero;
  shadow_avel[dense] = zero;

  self->categories[dense] = category;
  self->masks[dense] = mask;
  self->material_ids[dense] = material_id;
  self->user_data[dense] = (uint64_t)user_data;
  self->body_ids[dense] = JPH_INVALID_BODY_ID;

  self->slot_to_dense[slot] = dense;
  self->dense_to_slot[dense] = slot;
  self->slot_states[slot] = SLOT_PENDING_CREATE;
  self->view_shape[0] = (Py_ssize_t)self->count;

  // 5. QUEUE COMMAND
  if (UNLIKELY(!ensure_command_capacity(self))) {
    // Rollback
    self->count--;
    self->free_slots[self->free_count++] = slot;
    self->slot_states[slot] = SLOT_EMPTY;
    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_BodyCreationSettings_Destroy(settings);
    JPH_Shape_Destroy(shape);
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
  JPH_Shape_Destroy(shape); // Release local ref, BodySettings now owns it
  return PyLong_FromUnsignedLongLong(handle);
}

// Helper 1: Build the Jolt Compound Shape from the Python parts list
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static JPH_Shape *init_compound_shape(PhysicsWorldObject *self, PyObject *parts) {
    if (!PyList_Check(parts)) {
        PyErr_SetString(PyExc_TypeError, "Compound parts must be a list");
        return NULL;
    }

    Py_ssize_t num_parts = PyList_Size(parts);
    if (num_parts == 0) {
        PyErr_SetString(PyExc_ValueError, "Compound shape must have at least one part");
        return NULL;
    }

    // --- 1. PARSE PHASE (GIL Held) ---
    // Allocate temp buffer to store parsed data so we can release GIL later
    CompoundPart *buffer = PyMem_RawMalloc(sizeof(CompoundPart) * num_parts);
    if (!buffer) {
      PyErr_NoMemory();
      return NULL;
    }

    for (Py_ssize_t i = 0; i < num_parts; i++) {
        PyObject *item = PyList_GetItem(parts, i);
        // Expecting tuple: (pos, rot, type, size_params)
        if (!PyTuple_Check(item) || PyTuple_Size(item) < 4) {
            PyMem_RawFree(buffer);
            PyErr_Format(PyExc_ValueError, "Part %zd must be a tuple(pos, rot, type, size)", i);
            return NULL;
        }

        PyObject *p_pos = PyTuple_GetItem(item, 0);
        PyObject *p_rot = PyTuple_GetItem(item, 1);
        long type_l = PyLong_AsLong(PyTuple_GetItem(item, 2));
        PyObject *p_size = PyTuple_GetItem(item, 3);
        
        if (PyErr_Occurred()) {
            PyMem_RawFree(buffer);
            return NULL;
        }

        buffer[i].type = (int)type_l;
        memset(buffer[i].params, 0, sizeof(float) * 4);

        // Parse Position
        if (PyTuple_Check(p_pos) && PyTuple_Size(p_pos) == 3) {
            buffer[i].local_p.x = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 0));
            buffer[i].local_p.y = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 1));
            buffer[i].local_p.z = (float)PyFloat_AsDouble(PyTuple_GetItem(p_pos, 2));
        } else {
            buffer[i].local_p = (JPH_Vec3){0,0,0};
        }

        // Parse Rotation
        if (PyTuple_Check(p_rot) && PyTuple_Size(p_rot) == 4) {
            buffer[i].local_q.x = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 0));
            buffer[i].local_q.y = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 1));
            buffer[i].local_q.z = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 2));
            buffer[i].local_q.w = (float)PyFloat_AsDouble(PyTuple_GetItem(p_rot, 3));
        } else {
            buffer[i].local_q = (JPH_Quat){0,0,0,1};
        }

        // Parse Size Params
        if (PyTuple_Check(p_size)) {
            Py_ssize_t sz = PyTuple_Size(p_size);
            for (int j = 0; j < 4 && j < sz; j++) {
                buffer[i].params[j] = (float)PyFloat_AsDouble(PyTuple_GetItem(p_size, j));
            }
        } else if (PyFloat_Check(p_size) || PyLong_Check(p_size)) {
            buffer[i].params[0] = (float)PyFloat_AsDouble(p_size);
        }
    }

    if (PyErr_Occurred()) {
        PyMem_RawFree(buffer);
        return NULL;
    }

    // --- 2. JOLT EXECUTION PHASE (Release GIL, Acquire Jolt Lock) ---
    JPH_Shape *final_shape = NULL;
    Py_BEGIN_ALLOW_THREADS

    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    JPH_StaticCompoundShapeSettings *compound_settings = JPH_StaticCompoundShapeSettings_Create();
    
    // We iterate through our C buffer, acquiring Shadow Lock briefly for each shape lookup
    for (Py_ssize_t i = 0; i < num_parts; i++) {
        SHADOW_LOCK(&self->shadow_lock);
        JPH_Shape *sub_shape = find_or_create_shape_locked(self, buffer[i].type, buffer[i].params);
        SHADOW_UNLOCK(&self->shadow_lock);

        if (sub_shape) {
            JPH_CompoundShapeSettings_AddShape2(
                (JPH_CompoundShapeSettings *)compound_settings, 
                &buffer[i].local_p, 
                &buffer[i].local_q,
                sub_shape, 
                0
            );
        }
    }

    final_shape = (JPH_Shape *)JPH_StaticCompoundShape_Create(compound_settings);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)compound_settings);

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

    // --- 3. CLEANUP ---
    PyMem_RawFree(buffer);

    if (!final_shape) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create compound shape");
        return NULL;
    }

    return final_shape;
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

  if (props.is_sensor) {
    JPH_BodyCreationSettings_SetIsSensor(settings, true);
  }

  if (props.use_ccd) {
    JPH_BodyCreationSettings_SetMotionQuality(settings,
                                              JPH_MotionQuality_LinearCast);
  }

  JPH_BodyCreationSettings_SetFriction(settings, props.friction);
  JPH_BodyCreationSettings_SetRestitution(settings, props.restitution);
}

// Orchestrator
static PyObject *PhysicsWorld_create_compound_body(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
  JPH_Real px = 0, py = 0, pz = 0;
  float rx = 0, ry = 0, rz = 0, rw = 1.0f;
  float mass = -1.0f, friction = 0.2f, restitution = 0.0f;
  int motion_type = 2, is_sensor = 0, use_ccd = 0;
  uint64_t user_data = 0;
  uint32_t category = 0xFFFF, mask = 0xFFFF, material_id = 0;
  PyObject *parts = NULL;

  static char *kwlist[] = {"pos",  "rot",         "parts",     "motion",
                           "mass", "user_data",   "is_sensor", "category",
                           "mask", "material_id", "friction",  "restitution",
                           "ccd",  NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "(ddd)(ffff)O|ifKpIIffp", kwlist, &px, &py, &pz, &rx, &ry,
          &rz, &rw, &parts, &motion_type, &mass, &user_data, &is_sensor,
          &category, &mask, &material_id, &friction, &restitution, &use_ccd)) {
    return NULL;
  }

  if (!PyList_Check(parts)) {
    return PyErr_Format(PyExc_TypeError, "Parts must be a list");
  }

  // 1. Build the Compound Shape (Heavy lifting, released GIL internally)
  // This helper already handles its own Jolt locks and internal Shadow Lock lookups.
  JPH_Shape *final_shape = init_compound_shape(self, parts);
  if (!final_shape) {
    return PyErr_Format(PyExc_RuntimeError, "Failed to create Compound Shape");
  }

  // 2. Prep Creation Settings
  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      final_shape, &(JPH_RVec3){px, py, pz},
      &(JPH_Quat){rx, ry, rz, rw}, (JPH_MotionType)motion_type,
      (motion_type == 0) ? 0 : 1);

  BodyCreationProps props = {.mass = mass, .friction = friction, 
                             .restitution = restitution, .is_sensor = is_sensor, 
                             .use_ccd = use_ccd};
  apply_body_creation_props(settings, final_shape, props);

  // 3. COMMIT PHASE (Shadow Lock)
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);
  BLOCK_UNTIL_NOT_QUERYING(self);

  // Ensure Capacity (Slots + Dense Memory)
  if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
    size_t needed = (self->capacity == 0) ? 1024 : self->capacity * 2;
    if (PhysicsWorld_resize(self, needed) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_BodyCreationSettings_Destroy(settings);
      JPH_Shape_Destroy(final_shape);
      return NULL;
    }
  }

  uint32_t slot = self->free_slots[--self->free_count];
  uint32_t dense = (uint32_t)self->count++;
  BodyHandle handle = make_handle(slot, self->generations[slot]);
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

  // --- IMMEDIATE SHADOW WRITE (Stride Sensitive) ---
  auto *shadow_pos = (PosStride *)self->positions;
  auto *shadow_ppos = (PosStride *)self->prev_positions;
  auto *shadow_rot = (AuxStride *)self->rotations;
  auto *shadow_prot = (AuxStride *)self->prev_rotations;
  auto *shadow_lvel = (AuxStride *)self->linear_velocities;
  auto *shadow_avel = (AuxStride *)self->angular_velocities;

  // Initialize Position/Rotation
  PosStride p = {px, py, pz};
  shadow_pos[dense] = p;
  shadow_ppos[dense] = p;

  AuxStride q = {rx, ry, rz, rw};
  shadow_rot[dense] = q;
  shadow_prot[dense] = q;

  // Zero Velocities
  AuxStride zero = {0, 0, 0, 0};
  shadow_lvel[dense] = zero;
  shadow_avel[dense] = zero;

  // Metadata
  self->categories[dense] = category;
  self->masks[dense] = mask;
  self->material_ids[dense] = material_id;
  self->user_data[dense] = (uint64_t)user_data;
  self->body_ids[dense] = JPH_INVALID_BODY_ID;

  // Indirection
  self->slot_to_dense[slot] = dense;
  self->dense_to_slot[dense] = slot;
  self->slot_states[slot] = SLOT_PENDING_CREATE;
  self->view_shape[0] = (Py_ssize_t)self->count;

  // 4. QUEUE COMMAND
  if (UNLIKELY(!ensure_command_capacity(self))) {
    self->count--;
    self->free_slots[self->free_count++] = slot;
    self->slot_states[slot] = SLOT_EMPTY;
    SHADOW_UNLOCK(&self->shadow_lock);
    JPH_BodyCreationSettings_Destroy(settings);
    JPH_Shape_Destroy(final_shape);
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
  
  // BodySettings and Jolt Body now own the shape, release local ref.
  JPH_Shape_Destroy(final_shape); 

  return PyLong_FromUnsignedLongLong(handle);
}

// Helper 1: Resolve material properties based on ID and explicit overrides
static MaterialSettings resolve_material_params(PhysicsWorldObject *self,
                                                uint32_t material_id,
                                                MaterialSettings input) {
  // 1. Start with Jolt Defaults
  float f = 0.2f;
  float r = 0.0f;

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

// Helper 3: Apply mass, sensor, CCD, and sleeping settings to the creation
// struct
static void configure_body_settings(JPH_BodyCreationSettings *settings,
                                    JPH_Shape *shape, BodyConfig cfg) {
  // Use the members of the struct instead of loose variables
  if (cfg.is_sensor) {
    JPH_BodyCreationSettings_SetIsSensor(settings, true);
  }

  if (cfg.use_ccd) {
    JPH_BodyCreationSettings_SetMotionQuality(settings,
                                              JPH_MotionQuality_LinearCast);
  }

  if (cfg.motion_type == 2) { // MOTION_DYNAMIC
    JPH_BodyCreationSettings_SetAllowSleeping(settings, true);
  }

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
                                          PyObject *const *args, 
                                          size_t nargsf, 
                                          PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    // 1. DEFAULT VALUES
    JPH_Real px = 0.0f, py = 0.0f, pz = 0.0f;
    float rx = 0.0f, ry = 0.0f, rz = 0.0f, rw = 1.0f;
    float mass = -1.0f, friction = -1.0f, restitution = -1.0f;
    int shape_type = 0, motion_type = 2, is_sensor = 0, use_ccd = 0;
    uint32_t category = 0xFFFF, mask = 0xFFFF, material_id = 0;
    unsigned long long user_data = 0;
    PyObject *py_size = NULL;

    static char *kwlist[] = {
        "pos", "rot", "size", "shape", "motion", "user_data", "is_sensor",
        "mass", "category", "mask", "friction", "restitution", "material_id", "ccd", NULL
    };

    // Argument parsing logic (Keep existing)
    PyObject *temp_tuple = PyTuple_New(nargs);
    if (UNLIKELY(!temp_tuple)) return NULL;
    for (Py_ssize_t i = 0; i < nargs; i++) {
        Py_INCREF(args[i]);
        PyTuple_SET_ITEM(temp_tuple, i, args[i]);
    }

    PyObject *temp_dict = NULL;
    if (kwnames) {
        temp_dict = PyDict_New();
        Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < nkw; i++) {
            PyDict_SetItem(temp_dict, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
        }
    }

    int ok = PyArg_ParseTupleAndKeywords(temp_tuple, temp_dict, "|(ddd)(ffff)OiiKpfIIffIp", kwlist,
                                         &px, &py, &pz, &rx, &ry, &rz, &rw, &py_size,
                                         &shape_type, &motion_type, &user_data, &is_sensor,
                                         &mass, &category, &mask, &friction, &restitution,
                                         &material_id, &use_ccd);
    
    Py_XDECREF(temp_dict);
    Py_DECREF(temp_tuple);
    if (!ok) return NULL;

    if (shape_type == 4 && motion_type != 0) {
        return PyErr_Format(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
    }

    MaterialSettings mat_in = {friction, restitution};
    MaterialSettings mat = resolve_material_params(self, material_id, mat_in);
    float s[4]; 
    parse_body_size(py_size, s);

    JPH_Shape *shape = NULL;
    JPH_BodyCreationSettings *settings = NULL;

    // --- CRITICAL SECTION: JOLT PREP ---
    Py_BEGIN_ALLOW_THREADS;
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    SHADOW_LOCK(&self->shadow_lock);
    
    shape = find_or_create_shape_locked(self, shape_type, s);
    
    SHADOW_UNLOCK(&self->shadow_lock);

    if (LIKELY(shape)) {
        settings = JPH_BodyCreationSettings_Create3(
            shape, 
            &(JPH_RVec3){px, py, pz},
            &(JPH_Quat){rx, ry, rz, rw}, 
            (JPH_MotionType)motion_type,
            (motion_type == 0) ? 0 : 1);

        if (settings) {
            BodyConfig config = {
                .mass = mass, .friction = mat.friction, .restitution = mat.restitution,
                .is_sensor = is_sensor, .use_ccd = use_ccd, .motion_type = motion_type
            };
            configure_body_settings(settings, shape, config);
        }
    }

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS;

    if (!shape) return PyErr_Format(PyExc_RuntimeError, "Failed to create/find Shape");
    if (!settings) return PyErr_Format(PyExc_RuntimeError, "Failed to create BodySettings");

    // --- COMMIT PHASE: SHADOW BUFFER UPDATE ---
    SHADOW_LOCK(&self->shadow_lock);
    
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // Ensure Capacity
    if (UNLIKELY(self->free_count == 0 || self->count + 1 > self->capacity)) {
        size_t new_cap = (self->capacity == 0) ? 1024 : self->capacity * 2;
        if (PhysicsWorld_resize(self, new_cap) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            JPH_BodyCreationSettings_Destroy(settings);
            return NULL;
        }
    }

    uint32_t slot = self->free_slots[--self->free_count];
    uint32_t dense = (uint32_t)self->count++; 
    BodyHandle handle = make_handle(slot, self->generations[slot]);

    JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)handle);

    // Typed Pointers for clean struct assignment
    PosStride *shadow_pos = (PosStride *)self->positions;
    PosStride *shadow_ppos = (PosStride *)self->prev_positions;
    AuxStride *shadow_rot = (AuxStride *)self->rotations;
    AuxStride *shadow_prot = (AuxStride *)self->prev_rotations;
    AuxStride *shadow_lvel = (AuxStride *)self->linear_velocities;
    AuxStride *shadow_avel = (AuxStride *)self->angular_velocities;

    // 1. Position Commit (Stride 3)
    PosStride p = {(JPH_Real)px, (JPH_Real)py, (JPH_Real)pz};
    shadow_pos[dense] = p;
    shadow_ppos[dense] = p;

    // 2. Rotation Commit (Stride 4)
    AuxStride q = {rx, ry, rz, rw};
    shadow_rot[dense] = q;
    shadow_prot[dense] = q;

    // 3. Aux Data Commit (Stride 4 / Stride 1)
    AuxStride zero = {0, 0, 0, 0};
    shadow_lvel[dense] = zero;
    shadow_avel[dense] = zero;

    self->categories[dense] = category;
    self->masks[dense] = mask;
    self->material_ids[dense] = material_id;
    self->user_data[dense] = (uint64_t)user_data;

    // 4. Indirection Commit
    self->slot_to_dense[slot] = dense;
    self->dense_to_slot[dense] = slot;
    self->slot_states[slot] = SLOT_PENDING_CREATE;
    
    self->view_shape[0] = (Py_ssize_t)self->count;

    // 5. Command Buffer Commit
    if (UNLIKELY(!ensure_command_capacity(self))) {
        // Rollback
        self->count--;
        self->free_slots[self->free_count++] = slot;
        self->slot_states[slot] = SLOT_EMPTY;
        SHADOW_UNLOCK(&self->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
    cmd->create.settings = settings;
    cmd->create.user_data = (uint64_t)user_data;
    cmd->create.category = category;
    cmd->create.mask = mask;
    cmd->create.material_id = material_id;

    SHADOW_UNLOCK(&self->shadow_lock);

    return PyLong_FromUnsignedLongLong(handle);
}

PyObject *PhysicsWorld_create_bodies_batch(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    PyObject *py_positions = NULL, *py_sizes = NULL;
    int shape_type = 0, motion_type = 2;
    static char *kwlist[] = {"positions", "sizes", "shape_type", "motion_type", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO|ii", kwlist, &py_positions, &py_sizes, &shape_type, &motion_type)) return NULL;
    
    Py_ssize_t batch_count = PyList_Size(py_positions);
    if (PyList_Size(py_sizes) != batch_count) return PyErr_Format(PyExc_ValueError, "List length mismatch");

    // 1. ALLOCATE TEMP BUFFERS
    PosStride *pos_buf = PyMem_RawMalloc(batch_count * sizeof(PosStride));
    ShapeParams *size_buf = PyMem_RawMalloc(batch_count * sizeof(ShapeParams));
    auto **settings_buf = (JPH_BodyCreationSettings **)PyMem_RawCalloc(batch_count, sizeof(JPH_BodyCreationSettings*));

    if (!pos_buf || !size_buf || !settings_buf) {
        PyMem_RawFree(pos_buf); PyMem_RawFree(size_buf); PyMem_RawFree((void *)settings_buf);
        return PyErr_NoMemory();
    }

    // 2. PARSE (GIL HELD)
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (!parse_py_vec3(PyList_GetItem(py_positions, i), &pos_buf[i])) pos_buf[i] = (PosStride){0,0,0};
        parse_body_size(PyList_GetItem(py_sizes, i), size_buf[i].p);
    }

    // 3. JOLT PREP (NO GIL)
    Py_BEGIN_ALLOW_THREADS
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    SHADOW_LOCK(&self->shadow_lock); 
    
    JPH_STACK_ALLOC(JPH_RVec3, j_pos);
    JPH_STACK_ALLOC(JPH_Quat, j_rot);
    j_rot->x = 0; j_rot->y = 0; j_rot->z = 0; j_rot->w = 1.0f;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        JPH_Shape *shape = find_or_create_shape_locked(self, shape_type, size_buf[i].p);
        if (shape) {
            j_pos->x = pos_buf[i].x; 
            j_pos->y = pos_buf[i].y; 
            j_pos->z = pos_buf[i].z;
            settings_buf[i] = JPH_BodyCreationSettings_Create3(shape, j_pos, j_rot, (JPH_MotionType)motion_type, (motion_type == 0 ? 0 : 1));
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS

    // --- 4. COMMIT PHASE (SHADOW LOCK) ---
    SHADOW_LOCK(&self->shadow_lock);
    
    // Resize Checks (Slots + Dense Capacity)
    if (self->free_count < (size_t)batch_count || (self->count + (size_t)batch_count) > self->capacity) {
        size_t needed = (self->count + (size_t)batch_count > self->capacity) ? (self->count + (size_t)batch_count + 1024) : self->capacity;
        if (PhysicsWorld_resize(self, needed) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            goto fail;
        }
    }

    // Command Queue Checks
    size_t needed_cmds = self->command_count + batch_count;
    if (self->command_capacity < needed_cmds) {
        void *new_q = PyMem_RawRealloc(self->command_queue, needed_cmds * sizeof(PhysicsCommand));
        if (!new_q) {
            SHADOW_UNLOCK(&self->shadow_lock);
            goto fail;
        }
        self->command_queue = (PhysicsCommand *)new_q;
        self->command_capacity = needed_cmds;
    }

    PyObject *result_list = PyList_New(batch_count);
    if (!result_list) {
        SHADOW_UNLOCK(&self->shadow_lock);
        goto fail;
    }

    // Typed Pointers for clean indexing
    auto *shadow_pos = (PosStride *)self->positions;
    auto *shadow_prev_pos = (PosStride *)self->prev_positions;
    auto *shadow_rot = (AuxStride *)self->rotations;
    auto *shadow_prev_rot = (AuxStride *)self->prev_rotations;
    auto *shadow_lvel = (AuxStride *)self->linear_velocities;
    auto *shadow_avel = (AuxStride *)self->angular_velocities;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (!settings_buf[i]) {
            Py_INCREF(Py_None);
            PyList_SetItem(result_list, i, Py_None);
            continue;
        }

        uint32_t slot = self->free_slots[--self->free_count];
        auto dense = (uint32_t)self->count++;
        BodyHandle handle = make_handle(slot, self->generations[slot]);
        JPH_BodyCreationSettings_SetUserData(settings_buf[i], (uint64_t)handle);

        // --- Clean Struct Assignment ---
        PosStride p = {pos_buf[i].x, pos_buf[i].y, pos_buf[i].z};
        shadow_pos[dense] = p;
        shadow_prev_pos[dense] = p;
        
        AuxStride r = {0.0f, 0.0f, 0.0f, 1.0f}; // Identity Quaternion
        shadow_rot[dense] = r;
        shadow_prev_rot[dense] = r;
        
        AuxStride zero_vel = {0.0f, 0.0f, 0.0f, 0.0f};
        shadow_lvel[dense] = zero_vel;
        shadow_avel[dense] = zero_vel;

        // Metadata
        self->body_ids[dense] = JPH_INVALID_BODY_ID; 
        self->slot_to_dense[slot] = dense;
        self->dense_to_slot[dense] = slot;
        self->slot_states[slot] = SLOT_PENDING_CREATE;

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header = CMD_HEADER(CMD_CREATE_BODY, slot);
        cmd->create.settings = settings_buf[i];

        PyList_SetItem(result_list, i, PyLong_FromUnsignedLongLong(handle));
    }
    
    self->view_shape[0] = (Py_ssize_t)self->count;
    SHADOW_UNLOCK(&self->shadow_lock);

    PyMem_RawFree(pos_buf); PyMem_RawFree(size_buf); PyMem_RawFree(settings_buf);
    return result_list;

fail:
    for(Py_ssize_t i=0; i<batch_count; i++) if(settings_buf[i]) JPH_BodyCreationSettings_Destroy(settings_buf[i]);
    PyMem_RawFree(pos_buf); PyMem_RawFree(size_buf); PyMem_RawFree(settings_buf);
    return NULL;
}

/**
 * Helper 1: Build the Jolt triangle array while verifying index bounds.
 */
static JPH_IndexedTriangle *build_mesh_triangles(const uint32_t *raw,
                                                 MeshBounds bounds) {
  auto *jolt_tris = (JPH_IndexedTriangle *)PyMem_RawMalloc(
      bounds.tri_count * sizeof(JPH_IndexedTriangle));
  if (!jolt_tris) {
    PyErr_NoMemory();
    return NULL;
  }

  for (uint32_t t = 0; t < bounds.tri_count; t++) {
    uint32_t i1 = raw[t * 3 + 0];
    uint32_t i2 = raw[t * 3 + 1];
    uint32_t i3 = raw[t * 3 + 2];

    if (i1 >= bounds.vertex_count || i2 >= bounds.vertex_count || i3 >= bounds.vertex_count) {
      PyMem_RawFree(jolt_tris);
      PyErr_Format(PyExc_ValueError, "Mesh index out of range: %u/%u/%u >= %u",
                   i1, i2, i3, bounds.vertex_count);
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
static JPH_Shape *build_mesh_shape(const void *v_data, MeshBounds bounds,
                                   JPH_IndexedTriangle *tris) {
  JPH_MeshShapeSettings *mss =
      JPH_MeshShapeSettings_Create2((JPH_Vec3 *)v_data, bounds.vertex_count, tris, bounds.tri_count);
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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static PyObject *PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  Py_buffer v_view = {0};
  Py_buffer i_view = {0};
  JPH_Real px = 0;
  JPH_Real py = 0;
  JPH_Real pz = 0;
  float rx = 0;
  float ry = 0;
  float rz = 0;
  float rw = 1.0f;
  size_t user_data = 0;
  uint32_t cat = 0xFFFF;
  uint32_t mask = 0xFFFF;
  static char *kwlist[] = {"pos",       "rot",      "vertices", "indices",
                           "user_data", "category", "mask",     NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ddd)(ffff)y*y*|KII", kwlist,
                                   &px, &py, &pz, &rx, &ry, &rz, &rw, &v_view,
                                   &i_view, &user_data, &cat, &mask)) {
    return NULL;
  }

  // 1. Validation
  if (v_view.len % 12 != 0 || i_view.len % 12 != 0) {
    PyErr_SetString(PyExc_ValueError,
                    "Buffer size mismatch: must be multiples of 12 bytes");
    goto cleanup;
  }

  MeshBounds bounds = {0};
  bounds.tri_count = (uint32_t)(i_view.len / 12);
  bounds.vertex_count = (uint32_t)(v_view.len / 12);

  // 2. Triangle processing
  JPH_IndexedTriangle *tris =
      build_mesh_triangles((uint32_t *)i_view.buf, bounds);
  if (!tris) {
    goto cleanup;
  }

  // 3. Jolt Shape Build
  JPH_Shape *shape = build_mesh_shape(v_view.buf, bounds, tris);
  PyMem_RawFree(tris);
  if (!shape) {
    goto cleanup;
  }

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
      shape, &(JPH_RVec3){px, py, pz},
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
  if (v_view.obj) {
    PyBuffer_Release(&v_view);
  }
  if (i_view.obj) {
    PyBuffer_Release(&i_view);
  }
  return NULL;
}

static PyObject *PhysicsWorld_destroy_body(PhysicsWorldObject *self,
                                           PyObject *const *args, 
                                           size_t nargsf, 
                                           PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint64_t handle_raw = 0;

    // --- 1. THE FAST PATH: Positional Only (No keywords, exactly 1 arg) ---
    if (LIKELY(kwnames == NULL && nargs == 1)) {
        handle_raw = PyLong_AsUnsignedLongLong(args[0]);
        if (UNLIKELY(PyErr_Occurred())) return NULL;
    } 
    else {
        // --- 2. THE SLOW PATH: Keywords Fallback ---
        static char *kwlist[] = {"handle", NULL};
        
        // Manual bridge to standard public API
        PyObject *temp_tuple = PyTuple_New(nargs);
        if (!temp_tuple) return NULL;
        for (Py_ssize_t i = 0; i < nargs; i++) {
            Py_INCREF(args[i]);
            PyTuple_SET_ITEM(temp_tuple, i, args[i]);
        }

        PyObject *temp_dict = NULL;
        if (kwnames) {
            temp_dict = PyDict_New();
            Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
            for (Py_ssize_t i = 0; i < nkw; i++) {
                PyDict_SetItem(temp_dict, PyTuple_GET_ITEM(kwnames, i), args[nargs + i]);
            }
        }

        int ok = PyArg_ParseTupleAndKeywords(temp_tuple, temp_dict, "K", kwlist, &handle_raw);
        Py_XDECREF(temp_dict);
        Py_DECREF(temp_tuple);
        if (!ok) return NULL;
    }

    // --- 3. EXECUTION ---
    SHADOW_LOCK(&self->shadow_lock);

    // We block for STEPPING because flush_commands modifies the same dense arrays.
    // We REMOVED the block for Queries as it is not needed for deferred commands.
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (UNLIKELY(!unpack_handle(self, (BodyHandle)handle_raw, &slot))) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // Mark for deferred destruction
    uint8_t state = self->slot_states[slot];
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {

        if (UNLIKELY(!ensure_command_capacity(self))) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        PhysicsCommand *cmd = &self->command_queue[self->command_count++];
        cmd->header = CMD_HEADER(CMD_DESTROY_BODY, slot);

        // Mark logically dead immediately so Python scripts treat it as gone.
        // Memory remains valid for current queries.
        self->slot_states[slot] = SLOT_PENDING_DESTROY;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_destroy_bodies_batch(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    PyObject *py_handles = NULL;
    static char *kwlist[] = {"handles", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &py_handles)) {
        return NULL;
    }

    if (!PySequence_Check(py_handles)) {
        return PyErr_Format(PyExc_TypeError, "Expected a sequence of handles");
    }

    Py_ssize_t batch_count = PySequence_Size(py_handles);
    if (batch_count == 0) Py_RETURN_NONE;

    SHADOW_LOCK(&self->shadow_lock);
    
    // We must ensure the engine isn't currently flushing commands
    BLOCK_UNTIL_NOT_STEPPING(self);

    // --- 1. PRE-ALLOCATE COMMAND QUEUE ---
    // Growing the queue once is significantly faster than doing it in a loop
    size_t needed = self->command_count + batch_count;
    if (needed > self->command_capacity) {
        size_t new_cap = (self->command_capacity * 2 > needed) ? self->command_capacity * 2 : needed;
        void *new_ptr = PyMem_RawRealloc(self->command_queue, new_cap * sizeof(PhysicsCommand));
        if (!new_ptr) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }
        self->command_queue = (PhysicsCommand *)new_ptr;
        self->command_capacity = new_cap;
    }

    int actual_destroyed = 0;

    // --- 2. FAST ITERATION ---
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        // Use Sequence API to support list, tuple, or even NumPy arrays
        PyObject *item = PySequence_GetItem(py_handles, i);
        if (UNLIKELY(!item)) continue;

        uint64_t h_raw = PyLong_AsUnsignedLongLong(item);
        Py_DECREF(item);

        if (UNLIKELY(PyErr_Occurred())) {
            PyErr_Clear();
            continue;
        }

        uint32_t slot = 0;
        if (unpack_handle(self, (BodyHandle)h_raw, &slot)) {
            uint8_t state = self->slot_states[slot];
            
            // Only destroy if it's currently alive or just created
            if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
                PhysicsCommand *cmd = &self->command_queue[self->command_count++];
                cmd->header = CMD_HEADER(CMD_DESTROY_BODY, slot);

                // Mark logically dead immediately.
                // This ensures is_alive(h) returns False right now.
                self->slot_states[slot] = SLOT_PENDING_DESTROY;
                actual_destroyed++;
            }
        }
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    
    DEBUG_LOG("Batch Destroy: Queued %d bodies for removal.", actual_destroyed);
    Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_position(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  JPH_Real x = 0.0;
  JPH_Real y = 0.0;
  JPH_Real z = 0.0;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kddd", kwlist, &handle_raw, &x,
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
  cmd->pos.x = x;
  cmd->pos.y = y;
  cmd->pos.z = z;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_rotation(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  float w = 0.0f;
  static char *kwlist[] = {"handle", "x", "y", "z", "w", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffff", kwlist, &handle_raw, &x,
                                   &y, &z, &w)) {
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
    PyErr_SetString(PyExc_ValueError,
                    "Handle is stale or body is being destroyed");
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
  cmd->quat.x = x;
  cmd->quat.y = y;
  cmd->quat.z = z;
  cmd->quat.w = w;

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
  cmd->vec3f.x = x;
  cmd->vec3f.y = y;
  cmd->vec3f.z = z;

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
  cmd->vec3f.x = x;
  cmd->vec3f.y = y;
  cmd->vec3f.z = z;

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
    PyErr_SetString(PyExc_ValueError,
                    "Handle is stale or body is being destroyed");
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
  JPH_Real px = 0;
  JPH_Real py = 0;
  JPH_Real pz = 0;
  float rx = 0;
  float ry = 0;
  float rz = 0;
  float rw = 1.0f;
  static char *kwlist[] = {"handle", "pos", "rot", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K(ddd)(ffff)", kwlist,
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

static PyObject *PhysicsWorld_set_ccd(PhysicsWorldObject *self, PyObject *args,
                                      PyObject *kwds) {
  uint64_t handle_raw = 0;
  int enabled = 0;
  static char *kwlist[] = {"handle", "enabled", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kp", kwlist, &handle_raw,
                                   &enabled)) {
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
  auto *id_scratch =
      (JPH_BodyID *)PyMem_RawMalloc(count * sizeof(JPH_BodyID));
  if (!id_scratch) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }
  memcpy(id_scratch, self->body_ids, count * sizeof(JPH_BodyID));
  SHADOW_UNLOCK(&self->shadow_lock);

  // 2. Query activity state WHILE UNLOCKED (Deadlock safe)
  auto *results = (uint32_t *)PyMem_RawMalloc(count * sizeof(uint32_t));
  size_t active_count = 0;
  JPH_BodyInterface *bi = self->body_interface;

  for (size_t i = 0; i < count; i++) {
    if (id_scratch[i] != JPH_INVALID_BODY_ID &&
        (int)JPH_BodyInterface_IsActive(bi, id_scratch[i])) {
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

  // Clamp alpha to [0, 1]
  alpha = fmaxf(0.0f, fminf(1.0f, alpha));
  auto d_alpha = (double)alpha; // Use double for position math

  SHADOW_LOCK(&self->shadow_lock);
  
  // Consistency: Ensure we aren't reading while a Step is finishing
  BLOCK_UNTIL_NOT_STEPPING(self);

  size_t count = self->count;
  size_t total_bytes = count * 7 * sizeof(float);

  PyObject *bytes_obj = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_bytes);
  if (!bytes_obj) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  float *out = (float *)PyBytes_AsString(bytes_obj);

  // Map shadow buffers to Stride Structs
  auto *curr_p = (PosStride *)self->positions;
  auto *prev_p = (PosStride *)self->prev_positions;
  auto *curr_r = (AuxStride *)self->rotations;
  auto *prev_r = (AuxStride *)self->prev_rotations;

  for (size_t i = 0; i < count; i++) {
    size_t dst = i * 7;

    // --- 1. Position Lerp (Performed in DOUBLE) ---
    // This prevents jittering when far from the world origin.
    JPH_Real px = prev_p[i].x + (curr_p[i].x - prev_p[i].x) * d_alpha;
    JPH_Real py = prev_p[i].y + (curr_p[i].y - prev_p[i].y) * d_alpha;
    JPH_Real pz = prev_p[i].z + (curr_p[i].z - prev_p[i].z) * d_alpha;

    out[dst + 0] = (float)px;
    out[dst + 1] = (float)py;
    out[dst + 2] = (float)pz;

    // --- 2. Rotation NLerp (Performed in FLOAT) ---
    // Rotations don't suffer from "large coordinate" precision loss, 
    // so float is perfect here.
    float q1x = prev_r[i].x;
    float q1y = prev_r[i].y;
    float q1z = prev_r[i].z;
    float q1w = prev_r[i].w;

    float q2x = curr_r[i].x;
    float q2y = curr_r[i].y;
    float q2z = curr_r[i].z;
    float q2w = curr_r[i].w;

    // Shortest path correction
    float dot = q1x * q2x + q1y * q2y + q1z * q2z + q1w * q2w;
    if (dot < 0.0f) {
      q2x = -q2x; q2y = -q2y; q2z = -q2z; q2w = -q2w;
    }

    float rx = q1x + (q2x - q1x) * alpha;
    float ry = q1y + (q2y - q1y) * alpha;
    float rz = q1z + (q2z - q1z) * alpha;
    float rw = q1w + (q2w - q1w) * alpha;

    // Re-normalize to ensure it's a valid quaternion
    float mag_sq = rx * rx + ry * ry + rz * rz + rw * rw;
    float inv_len = (mag_sq > 0.000001f) ? 1.0f / sqrtf(mag_sq) : 1.0f;
    
    out[dst + 3] = rx * inv_len;
    out[dst + 4] = ry * inv_len;
    out[dst + 5] = rz * inv_len;
    out[dst + 6] = rw * inv_len;
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  return bytes_obj;
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
  JPH_Real px = 0;
  JPH_Real py = 0;
  JPH_Real pz = 0;
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
          args, kwds, "(ddd)(ffff)(fff)y*i|KIIIff", kwlist, &px, &py, &pz, &rx,
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

  auto *shape =
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
  pos->x = px;
  pos->y = py;
  pos->z = pz;
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

static PyObject *PhysicsWorld_get_debug_data(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  int draw_shapes = 1;
  int draw_constraints = 1;
  int draw_bounding_box = 0;
  int draw_centers = 0;
  int wireframe = 1;

  static char *kwlist[] = {"shapes",  "constraints", "bbox",
                           "centers", "wireframe",   NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ppppp", kwlist, &draw_shapes,
                                   &draw_constraints, &draw_bounding_box,
                                   &draw_centers, &wireframe)) {
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
  settings.drawShape = (bool)draw_shapes;
  settings.drawShapeWireframe = (bool)wireframe;
  settings.drawBoundingBox = (bool)draw_bounding_box;
  settings.drawCenterOfMassTransform = (bool)draw_centers;

  // 3. Draw Bodies
  if (draw_shapes || draw_bounding_box || draw_centers) {
    JPH_PhysicsSystem_DrawBodies(self->system, &settings, self->debug_renderer,
                                 NULL);
  }

  // 4. Draw Constraints (Explicit Calls)
  if (draw_constraints) {
    JPH_PhysicsSystem_DrawConstraints(self->system, self->debug_renderer);
    JPH_PhysicsSystem_DrawConstraintLimits(self->system, self->debug_renderer);
  }

  // 5. Export to Python Bytes
  // We snapshot the C-arrays into Python immutable bytes objects.
  PyObject *lines_bytes = PyBytes_FromStringAndSize(
      (char *)self->debug_lines.data,
      (Py_ssize_t)(self->debug_lines.count * sizeof(DebugVertex)));
  PyObject *tris_bytes = PyBytes_FromStringAndSize(
      (char *)self->debug_triangles.data,
      (Py_ssize_t)(self->debug_triangles.count * sizeof(DebugVertex)));

  SHADOW_UNLOCK(&self->shadow_lock);

  if (!lines_bytes || !tris_bytes) {
    Py_XDECREF(lines_bytes);
    Py_XDECREF(tris_bytes);
    return PyErr_NoMemory();
  }

  PyObject *ret = PyTuple_Pack(2, lines_bytes, tris_bytes);
  Py_DECREF(lines_bytes);
  Py_DECREF(tris_bytes);
  return ret;
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
    {"shape_count", (getter)get_shape_count, NULL, "Number of unique shapes in cache", NULL},
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
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"create_bodies_batch", (PyCFunction)PhysicsWorld_create_bodies_batch, 
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)PhysicsWorld_destroy_body,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"destroy_bodies_batch", (PyCFunction)PhysicsWorld_destroy_bodies_batch, 
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
     "Create a body made of multiple primitives. parts=[((x,y,z), "
     "(rx,ry,rz,rw), type, size), ...]"},

    // --- Interaction ---
    {"apply_impulse", (PyCFunction)PhysicsWorld_apply_impulse,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"apply_angular_impulse", (PyCFunction)PhysicsWorld_apply_angular_impulse,
     METH_FASTCALL | METH_KEYWORDS, "Apply rotational momentum."},
    {"apply_impulse_at", (PyCFunction)PhysicsWorld_apply_impulse_at,
     METH_FASTCALL | METH_KEYWORDS, "Apply impulse at world position."},
    {"apply_force", (PyCFunction)PhysicsWorld_apply_force,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"apply_torque", (PyCFunction)PhysicsWorld_apply_torque,
     METH_FASTCALL | METH_KEYWORDS, NULL},
    {"set_gravity", (PyCFunction)PhysicsWorld_set_gravity, METH_VARARGS, NULL},
    {"apply_buoyancy", (PyCFunction)PhysicsWorld_apply_buoyancy,
     METH_VARARGS | METH_KEYWORDS, "Apply fluid forces to a body."},
    {"apply_buoyancy_batch", (PyCFunction)PhysicsWorld_apply_buoyancy_batch,
     METH_VARARGS | METH_KEYWORDS,
     "Apply buoyancy to a list of bodies. handles must be a buffer of uint64."},
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
    {"set_ccd", (PyCFunction)PhysicsWorld_set_ccd, METH_VARARGS | METH_KEYWORDS,
     "Enable/Disable Continuous Collision Detection."},

    // --- Queries ---
    {"raycast", (PyCFunction)PhysicsWorld_raycast, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"raycast_batch", (PyCFunction)PhysicsWorld_raycast_batch,
     METH_FASTCALL | METH_KEYWORDS, "Execute multiple raycasts efficiently."},
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
    {"get_debug_data", (PyCFunction)PhysicsWorld_get_debug_data,
     METH_VARARGS | METH_KEYWORDS,
     "Returns (lines_bytes, triangles_bytes). Each vertex is 16 bytes: [x, y, "
     "z, color_u32]."},
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
    {NULL, NULL, 0, NULL}};

static const PyMethodDef Vehicle_methods[] = {
    {"set_input", (PyCFunction)Vehicle_set_input, METH_VARARGS | METH_KEYWORDS,
     "Set driver inputs: forward [-1..1], right [-1..1], brake [0..1], "
     "handbrake [0..1]"},
    {"set_tank_input", (PyCFunction)Vehicle_set_tank_input,
     METH_VARARGS | METH_KEYWORDS,
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
    {NULL, NULL, 0, NULL}};

static const PyMethodDef Skeleton_methods[] = {
    {"add_joint", (PyCFunction)Skeleton_add_joint, METH_VARARGS,
     "Add joint(name, parent_idx=-1)"},
    {"get_joint_index", (PyCFunction)Skeleton_get_joint_index, METH_VARARGS,
     "Get index by name"},
    {"finalize", (PyCFunction)Skeleton_finalize, METH_NOARGS,
     "Bake skeleton hierarchy"},
    {NULL, NULL, 0, NULL}};

static const PyMethodDef Ragdoll_methods[] = {
    {"drive_to_pose", (PyCFunction)Ragdoll_drive_to_pose,
     METH_VARARGS | METH_KEYWORDS, "Drive motors to target pose"},
    {"get_body_handles", (PyCFunction)Ragdoll_get_body_ids, METH_NOARGS,
     "Get list of body handles"},
    {"get_debug_info", (PyCFunction)Ragdoll_get_debug_info, METH_NOARGS,
     "Returns list of dicts for each part"},
    {NULL, NULL, 0, NULL}};

static const PyMethodDef RagdollSettings_methods[] = {
    {"add_part", (PyCFunction)RagdollSettings_add_part,
     METH_VARARGS | METH_KEYWORDS, "Config part"},
    {"stabilize", (PyCFunction)RagdollSettings_stabilize, METH_NOARGS,
     "Auto-detect collisions"},
    {NULL, NULL, 0, NULL}};

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
  INIT_NATIVE_MUTEX(g_jph_trampoline_lock);
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
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
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
    {Py_mod_multiple_interpreters, Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED},
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