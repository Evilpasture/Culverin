#include "culverin_ragdoll.h"
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

PyObject *Skeleton_get_joint_index(SkeletonObject *self,
                                          PyObject *args) {
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

PyObject *RagdollSettings_add_part(RagdollSettingsObject *self,
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

PyObject *RagdollSettings_stabilize(RagdollSettingsObject *self,
                                           PyObject *args) {
  if (JPH_RagdollSettings_Stabilize(self->settings)) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

PyObject *PhysicsWorld_create_ragdoll(PhysicsWorldObject *self,
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
                         Py_BuildValue("(fff)", vel->x, vel->y, vel->z));

    PyList_SET_ITEM(list, i, dict);
  }
  SHADOW_UNLOCK(&self->world->shadow_lock);

  return list;
}