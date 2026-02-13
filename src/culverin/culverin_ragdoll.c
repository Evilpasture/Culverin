#include "culverin_ragdoll.h"
#include "culverin_math.h"
#include "culverin_physics_world_internal.h"

PyObject *Skeleton_add_joint(SkeletonObject *self, PyObject *args) {
  const char *name = NULL;
  int parent_idx = -1; // Default to root
  if (!PyArg_ParseTuple(args, "s|i", &name, &parent_idx)) {
    return NULL;
  }

  int idx = (int)JPH_Skeleton_AddJoint2(self->skeleton, name, parent_idx);
  return PyLong_FromLong(idx);
}

PyObject *Skeleton_get_joint_index(SkeletonObject *self, PyObject *args) {
  const char *name = NULL;
  if (!PyArg_ParseTuple(args, "s", &name)) {
    return NULL;
  }
  int idx = JPH_Skeleton_GetJointIndex(self->skeleton, name);
  return PyLong_FromLong(idx);
}

PyObject *Skeleton_finalize(SkeletonObject *self, PyObject *args) {
  JPH_Skeleton_CalculateParentJointIndices(self->skeleton);
  if (!JPH_Skeleton_AreJointsCorrectlyOrdered(self->skeleton)) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Skeleton joints are out of order (parent must be added before child)");
    return NULL;
  }
  Py_RETURN_NONE;
}

// --- Ragdoll Settings Implementation ---

PyObject *PhysicsWorld_create_ragdoll_settings(PhysicsWorldObject *self,
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

PyObject *RagdollSettings_add_part(RagdollSettingsObject *self, PyObject *args,
                                   PyObject *kwds) {
  // 1. ARGUMENT PARSING
  int joint_idx = 0;
  int parent_idx = -1; // Root parts have no parent
  int shape_type = 0;
  float mass = 10.0f;
  PyObject *py_size = NULL;
  PyObject *py_pos = NULL;

  float twist_min = -0.1f;
  float twist_max = 0.1f;
  float cone_angle = 0.0f;
  
  float cx = 1.0f, cy = 0.0f, cz = 0.0f; // Unpacked from axis tuple
  float nx = 0.0f, ny = 1.0f, nz = 0.0f; // Unpacked from normal tuple

  static char *kwlist[] = {
      "joint_index",  "shape_type", "size",      "mass",
      "parent_index", "twist_min",  "twist_max", "cone_angle",
      "axis",         "normal",     "pos",       NULL 
  };

  // Preserve the (fff) format for the tuples passed in Python
  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "iiOfi|fff(fff)(fff)O", kwlist, 
          &joint_idx, &shape_type, &py_size, &mass, &parent_idx, 
          &twist_min, &twist_max, &cone_angle,
          &cx, &cy, &cz, 
          &nx, &ny, &nz,
          &py_pos)) {
    return NULL;
  }

  // 2. SHAPE ACQUISITION (Thread-Safe)
  float s[4] = {1, 1, 1, 0};
  if (py_size && PyTuple_Check(py_size)) {
    for (Py_ssize_t i = 0; i < PyTuple_Size(py_size) && i < 4; i++) {
      s[i] = (float)PyFloat_AsDouble(PyTuple_GetItem(py_size, i));
    }
  } else if (py_size && PyNumber_Check(py_size)) {
    s[0] = (float)PyFloat_AsDouble(py_size);
  }

  JPH_Shape *shape = NULL;
  Py_BEGIN_ALLOW_THREADS

  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  SHADOW_LOCK(&self->world->shadow_lock);
  shape = find_or_create_shape_locked(self->world, shape_type, s);
  SHADOW_UNLOCK(&self->world->shadow_lock);
  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

  Py_END_ALLOW_THREADS;

  if (!shape) return PyErr_Format(PyExc_ValueError, "Invalid shape configuration");

  // 3. APPLY SETTINGS
  JPH_Skeleton *skel = (JPH_Skeleton *)JPH_RagdollSettings_GetSkeleton(self->settings);
  int skel_count = JPH_Skeleton_GetJointCount(skel);

  if (joint_idx < 0 || joint_idx >= skel_count) {
    return PyErr_Format(PyExc_IndexError, "Joint index %d out of bounds", joint_idx);
  }

  // Ensure capacity
  if (JPH_RagdollSettings_GetPartCount(self->settings) <= joint_idx) {
    JPH_RagdollSettings_ResizeParts(self->settings, skel_count);
  }

  JPH_RagdollSettings_SetPartShape(self->settings, joint_idx, shape);
  JPH_RagdollSettings_SetPartMassProperties(self->settings, joint_idx, mass);
  JPH_RagdollSettings_SetPartObjectLayer(self->settings, joint_idx, 1);
  JPH_RagdollSettings_SetPartMotionType(self->settings, joint_idx, JPH_MotionType_Dynamic);

  // --- CRITICAL FIX: POSITION (RVec3) ---
  if (py_pos && PyTuple_Check(py_pos) && PyTuple_Size(py_pos) == 3) {
    // Use the Stack Alloc macro to ensure correct precision (float vs double) 
    // and alignment for JPH_RVec3.
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = PyFloat_AsDouble(PyTuple_GetItem(py_pos, 0));
    p->y = PyFloat_AsDouble(PyTuple_GetItem(py_pos, 1));
    p->z = PyFloat_AsDouble(PyTuple_GetItem(py_pos, 2));
    JPH_RagdollSettings_SetPartPosition(self->settings, joint_idx, p);
  }

  // Handle Parent Constraint
  if (parent_idx >= 0) {
    JPH_SwingTwistConstraintSettings cs;
    JPH_SwingTwistConstraintSettings_Init(&cs);
    cs.base.enabled = true;
    
    // Explicitly zero out RVec3 positions
    cs.position1.x = 0; cs.position1.y = 0; cs.position1.z = 0;
    cs.position2.x = 0; cs.position2.y = 0; cs.position2.z = 0;

    cs.twistAxis1 = (JPH_Vec3){cx, cy, cz};
    cs.twistAxis2 = (JPH_Vec3){cx, cy, cz};
    cs.planeAxis1 = (JPH_Vec3){nx, ny, nz};
    cs.planeAxis2 = (JPH_Vec3){nx, ny, nz};

    cs.normalHalfConeAngle = cone_angle;
    cs.planeHalfConeAngle = cone_angle;
    cs.twistMinAngle = twist_min;
    cs.twistMaxAngle = twist_max;

    JPH_RagdollSettings_SetPartToParent(self->settings, joint_idx, &cs);
  }

  Py_RETURN_NONE;
}

PyObject *RagdollSettings_stabilize(RagdollSettingsObject *self,
                                    PyObject *args) {
  if (JPH_RagdollSettings_Stabilize(self->settings)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject *PhysicsWorld_create_ragdoll(PhysicsWorldObject *self, PyObject *args,
                                      PyObject *kwds) {
  RagdollSettingsObject *py_settings = NULL;
  JPH_Real px, py, pz;
  float rx = 0, ry = 0, rz = 0, rw = 1;
  uint64_t user_data = 0;
  uint32_t category = 0xFFFF, mask = 0xFFFF, material_id = 0;

  static char *kwlist[] = {"settings", "pos", "rot", "user_data",
                           "category", "mask", "material_id", NULL};

  if (!PyArg_ParseTupleAndKeywords(
          args, kwds, "O!(ddd)|(ffff)KIII", kwlist,
          get_culverin_state(PyType_GetModule(Py_TYPE(self)))->RagdollSettingsType,
          &py_settings, &px, &py, &pz, &rx, &ry, &rz, &rw, 
          &user_data, &category, &mask, &material_id)) {
    return NULL;
  }

  // 1. Jolt Preparation (Release GIL)
  PyThreadState *_save = NULL;
  Py_UNBLOCK_THREADS;
  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

  // Initialize internal Jolt mappings
  JPH_RagdollSettings_CalculateBodyIndexToConstraintIndex(py_settings->settings);
  JPH_RagdollSettings_CalculateConstraintIndexToBodyIdxPair(py_settings->settings);

  // Create the Ragdoll instance
  JPH_Ragdoll *j_rag = JPH_RagdollSettings_CreateRagdoll(py_settings->settings, self->system, 0, user_data);
  
  if (!j_rag) {
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_BLOCK_THREADS;
    return PyErr_Format(PyExc_RuntimeError, "Jolt failed to create Ragdoll instance");
  }

  // 1. Get Bind Pose (Model Space)
  auto joint_count = (size_t)JPH_Skeleton_GetJointCount(JPH_RagdollSettings_GetSkeleton(py_settings->settings));
  JPH_Mat4 *neutral_matrices = (JPH_Mat4 *)PyMem_RawMalloc(joint_count * sizeof(JPH_Mat4));
  
  JPH_RVec3 zero_root = {0, 0, 0};
  JPH_Ragdoll_GetPose2(j_rag, &zero_root, neutral_matrices, true);

  // 2. Apply Spawn Rotation to all joint matrices
  JPH_STACK_ALLOC(JPH_Quat, root_q);
  root_q->x = rx; root_q->y = ry; root_q->z = rz; root_q->w = rw;

  JPH_STACK_ALLOC(JPH_Mat4, rot_matrix);
  manual_mat4_from_quat(root_q, rot_matrix); // <--- CHANGED

  // Pre-multiply all joint matrices by the root rotation
  for(size_t i = 0; i < joint_count; i++) {
      JPH_STACK_ALLOC(JPH_Mat4, result);
      manual_mat4_multiply(rot_matrix, &neutral_matrices[i], result); // <--- CHANGED
      neutral_matrices[i] = *result;
  }

  // 3. Set World Pose
  JPH_RVec3 root_pos = {px, py, pz};
  JPH_Ragdoll_SetPose2(j_rag, &root_pos, neutral_matrices, true);

  // Add to system so we can query final positions
  JPH_Ragdoll_AddToPhysicsSystem(j_rag, JPH_Activation_Activate, true);

  auto body_count = (size_t)JPH_Ragdoll_GetBodyCount(j_rag);
  
  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_BLOCK_THREADS;

  // 2. Python Object Creation
  CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
  RagdollObject *obj = (RagdollObject *)PyObject_New(RagdollObject, (PyTypeObject *)st->RagdollType);
  if (!obj) {
    Py_BEGIN_ALLOW_THREADS
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    JPH_Ragdoll_Destroy(j_rag);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_END_ALLOW_THREADS
    PyMem_RawFree(neutral_matrices);
    return NULL;
  }

  obj->ragdoll = j_rag;
  obj->world = self;
  Py_INCREF(self);
  obj->body_count = body_count;
  obj->body_slots = (uint32_t *)PyMem_RawMalloc(body_count * sizeof(uint32_t));

  // 3. Shadow Buffer "Warm-up" (Sync Jolt -> CPU)
  SHADOW_LOCK(&self->shadow_lock);
  
  if (self->free_count < body_count) {
    if (PhysicsWorld_resize(self, self->capacity + body_count + 1024) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_Ragdoll_Destroy(j_rag);
      PyMem_RawFree(neutral_matrices);
      Py_DECREF(obj);
      return NULL;
    }
  }

  JPH_BodyInterface *bi = self->body_interface;
  
  // Cast pointers to Stride Structs for safe indexing
  auto *shadow_pos  = (PosStride *)self->positions;
  auto *shadow_ppos = (PosStride *)self->prev_positions;
  auto *shadow_rot  = (AuxStride *)self->rotations;
  auto *shadow_prot = (AuxStride *)self->prev_rotations;
  auto *shadow_lvel = (AuxStride *)self->linear_velocities;
  auto *shadow_avel = (AuxStride *)self->angular_velocities;

  for (size_t i = 0; i < body_count; i++) {
    JPH_BodyID bid = JPH_Ragdoll_GetBodyID(j_rag, (int)i);
    uint32_t slot = self->free_slots[--self->free_count];
    obj->body_slots[i] = slot;
    uint32_t dense = (uint32_t)self->count++; // Index of the new body

    // Query Jolt directly for the final world-space transform calculated by SetPose2
    JPH_RVec3 world_p;
    JPH_Quat world_q;
    JPH_BodyInterface_GetPosition(bi, bid, &world_p);
    JPH_BodyInterface_GetRotation(bi, bid, &world_q);

    // --- STRIDE-SAFE ASSIGNMENT ---
    shadow_pos[dense]  = (PosStride){world_p.x, world_p.y, world_p.z};
    shadow_ppos[dense] = shadow_pos[dense];

    shadow_rot[dense]  = (AuxStride){world_q.x, world_q.y, world_q.z, world_q.w};
    shadow_prot[dense] = shadow_rot[dense];

    // Initialize physics state
    shadow_lvel[dense] = (AuxStride){0, 0, 0, 0};
    shadow_avel[dense] = (AuxStride){0, 0, 0, 0};

    // Metadata
    self->body_ids[dense]     = bid;
    self->slot_to_dense[slot] = dense;
    self->dense_to_slot[dense] = slot;
    self->slot_states[slot]    = SLOT_ALIVE;
    self->user_data[dense]     = user_data;
    self->categories[dense]    = category;
    self->masks[dense]         = mask;
    self->material_ids[dense]  = material_id;

    // Fast handle map update (crucial for collisions)
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);
    if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
      self->id_to_handle_map[j_idx] = make_handle(slot, self->generations[slot]);
    }

    // Link Jolt Body back to our Handle
    JPH_BodyInterface_SetUserData(bi, bid, (uint64_t)make_handle(slot, self->generations[slot]));
  }

  self->view_shape[0] = (Py_ssize_t)self->count;
  SHADOW_UNLOCK(&self->shadow_lock);

  PyMem_RawFree(neutral_matrices);
  return (PyObject *)obj;
}

PyObject *Ragdoll_drive_to_pose(RagdollObject *self, PyObject *args,
                                PyObject *kwds) {
  float root_x = 0.0f;
  float root_y = 0.0f;
  float root_z = 0.0f;
  float rx = 0.0f;
  float ry = 0.0f;
  float rz = 0.0f;
  float rw = 0.0f;
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
    return PyErr_Format(
        PyExc_ValueError,
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

PyObject *Ragdoll_get_body_ids(RagdollObject *self, PyObject *args) {
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

PyObject *Ragdoll_get_debug_info(RagdollObject *self,
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
                         Py_BuildValue("(ddd)", (double)vel->x, (double)vel->y, (double)vel->z));

    PyList_SET_ITEM(list, i, dict);
  }
  SHADOW_UNLOCK(&self->world->shadow_lock);

  return list;
}

void Skeleton_dealloc(SkeletonObject *self) {
  if (self->skeleton) {
    JPH_Skeleton_Destroy(self->skeleton);
  }
  Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *Skeleton_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
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

void RagdollSettings_dealloc(RagdollSettingsObject *self) {
  if (self->settings) {
    JPH_RagdollSettings_Destroy(self->settings);
  }
  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Ragdoll Instance Implementation ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void Ragdoll_dealloc(RagdollObject *self) {
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
        if (slot >= self->world->slot_capacity) {
          continue;
        }

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