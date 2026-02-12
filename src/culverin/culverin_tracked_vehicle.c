#include "culverin_tracked_vehicle.h"
#include "culverin_parsers.h"

// --- Tracked Vehicle Implementation ---

static JPH_WheelSettings *create_track_wheel(PyObject *w_dict) {
  Vec3f pos;
  if (!parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
    return NULL;
  }

  float radius = get_py_float_attr(w_dict, "radius", 0.4f);
  float width = get_py_float_attr(w_dict, "width", 0.2f);
  float suspension_len = get_py_float_attr(w_dict, "suspension", 0.5f);
  float friction = get_py_float_attr(w_dict, "friction", 1.0f);

  // NEW: Suspension Spring Properties
  float freq = get_py_float_attr(w_dict, "spring_freq", 2.0f); // Stiffness
  float damp = get_py_float_attr(w_dict, "spring_damp", 0.5f); // Bounciness

  JPH_WheelSettingsTV *w = JPH_WheelSettingsTV_Create();

  JPH_WheelSettings_SetPosition((JPH_WheelSettings *)w,
                                &(JPH_Vec3){pos.x, pos.y, pos.z});
  JPH_WheelSettings_SetRadius((JPH_WheelSettings *)w, radius);
  JPH_WheelSettings_SetWidth((JPH_WheelSettings *)w, width);

  JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings *)w, 0.05f);
  JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings *)w,
                                           suspension_len);

  // NEW: Apply Spring
  JPH_SpringSettings spring = {JPH_SpringMode_FrequencyAndDamping, freq, damp};
  JPH_WheelSettings_SetSuspensionSpring((JPH_WheelSettings *)w, &spring);

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
  if (num_tracks > 2) {
    num_tracks = 2;
  }
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
    if (py_driven) {
      track_set.drivenWheel = (uint32_t)PyLong_AsUnsignedLong(py_driven);
    }

    JPH_TrackedVehicleControllerSettings_SetTrack(t_ctrl, (uint32_t)t,
                                                  &track_set);
  }
}

// Orchestrator
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyObject *PhysicsWorld_create_tracked_vehicle(PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds) {
  uint64_t chassis_h = 0;
  PyObject *py_wheels = NULL;
  PyObject *py_tracks = NULL;
  float max_rpm = 6000.0f, min_rpm = 500.0f, max_torque = 5000.0f;
  static char *kwlist[] = {"chassis", "wheels", "tracks", "max_torque", "max_rpm", NULL};

  PyThreadState *_save = NULL; 

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KOO|ff", kwlist, &chassis_h,
                                   &py_wheels, &py_tracks, &max_torque, &max_rpm)) {
    return NULL;
  }

  // --- 1. RESOLVE CHASSIS (Requires GIL & Shadow Lock) ---
  SHADOW_LOCK(&self->shadow_lock);
  sync_and_flush_internal(self);
  uint32_t slot = 0;
  if (!unpack_handle(self, chassis_h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_ValueError, "Invalid chassis handle");
  }
  JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
  SHADOW_UNLOCK(&self->shadow_lock);

  // --- 2. PRE-JOLT RESOURCE ALLOCATION (GIL HELD) ---
  VehicleResources r = {0};
  uint32_t num_wheels = (uint32_t)PyList_Size(py_wheels);
  
  r.f_curve = JPH_LinearCurve_Create();
  JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
  JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, 1.0f);

  r.w_settings = (JPH_WheelSettings **)PyMem_RawCalloc(num_wheels, sizeof(JPH_WheelSettings *));
  for (uint32_t i = 0; i < num_wheels; i++) {
    // CRITICAL: Call this while GIL is held
    r.w_settings[i] = create_track_wheel(PyList_GetItem(py_wheels, i));
    if (!r.w_settings[i]) goto python_fail;
  }

  // Parse Track Config into C structs while GIL is held
  TrackData tracks[2];
  int num_tracks = 0;
  parse_tracks_to_c(py_tracks, tracks, &num_tracks);

  // --- 3. JOLT COMMIT (No GIL) ---
  bool jolt_locked = false;
  Py_UNBLOCK_THREADS;

  NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
  jolt_locked = true;

  const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
  JPH_BodyLockWrite lock = {0};
  JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);

  if (UNLIKELY(!lock.body)) goto jolt_fail;

  // Setup Controller
  TrackedEngineConfig eng_cfg = {.torque = max_torque, .max_rpm = max_rpm, .min_rpm = min_rpm};
  JPH_VehicleTransmissionSettings *v_trans = NULL;
  JPH_TrackedVehicleControllerSettings *t_ctrl = init_tracked_controller_settings(eng_cfg, &v_trans);
  r.v_ctrl = (JPH_WheeledVehicleControllerSettings *)t_ctrl;
  r.v_trans_set = v_trans;

  // Apply Track Data to Jolt Controller
  for (int t = 0; t < num_tracks; t++) {
    JPH_VehicleTrackSettings track_set;
    JPH_VehicleTrackSettings_Init(&track_set);
    track_set.wheels = tracks[t].indices;
    track_set.wheelsCount = tracks[t].count;
    track_set.drivenWheel = tracks[t].driven_idx;
    JPH_TrackedVehicleControllerSettings_SetTrack(t_ctrl, (uint32_t)t, &track_set);
  }

  // Assembly
  JPH_VehicleConstraintSettings v_set;
  JPH_VehicleConstraintSettings_Init(&v_set);
  v_set.wheelsCount = num_wheels;
  v_set.wheels = r.w_settings;
  v_set.controller = (JPH_VehicleControllerSettings *)t_ctrl;

  r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
  if (!r.j_veh) goto jolt_fail;

  r.tester = JPH_VehicleCollisionTesterRay_Create(2, &(JPH_Vec3){0, 1.0f, 0}, 2.0f);
  if (!r.tester) goto jolt_fail;

  JPH_VehicleConstraint_SetVehicleCollisionTester(r.j_veh, (JPH_VehicleCollisionTester *)r.tester);
  JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint *)r.j_veh);
  JPH_PhysicsSystem_AddStepListener(self->system, JPH_VehicleConstraint_AsPhysicsStepListener(r.j_veh));
  r.is_added_to_world = true;

  JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
  NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  jolt_locked = false;
  Py_BLOCK_THREADS;

  // --- 4. CLEANUP & WRAP ---
  for (int t = 0; t < num_tracks; t++) PyMem_RawFree(tracks[t].indices);

  CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
  VehicleObject *obj = (VehicleObject *)PyObject_New(VehicleObject, (PyTypeObject *)st->VehicleType);
  if (!obj) {
    SHADOW_LOCK(&self->shadow_lock);
    cleanup_vehicle_resources(&r, num_wheels, self);
    SHADOW_UNLOCK(&self->shadow_lock);
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

jolt_fail:
  if (lock.body) JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
  if (jolt_locked) NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
  Py_BLOCK_THREADS;

python_fail:
  for (int t = 0; t < num_tracks; t++) if(tracks[t].indices) PyMem_RawFree(tracks[t].indices);
  SHADOW_LOCK(&self->shadow_lock);
  cleanup_vehicle_resources(&r, num_wheels, self);
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyErr_Format(PyExc_RuntimeError, "Tracked vehicle creation failed");
}

// Helper: Set Tank Input
PyObject *Vehicle_set_tank_input(VehicleObject *self, PyObject *args,
                                 PyObject *kwds) {
  float left = 0.0f;
  float right = 0.0f;
  float brake = 0.0f;
  static char *kwlist[] = {"left", "right", "brake", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "ff|f", kwlist, &left, &right,
                                   &brake)) {
    return NULL;
  }

  if (!self->vehicle || !self->world) {
    Py_RETURN_NONE;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  JPH_TrackedVehicleController *t_ctrl =
      (JPH_TrackedVehicleController *)JPH_VehicleConstraint_GetController(
          self->vehicle);
  JPH_BodyID bid =
      JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));
  JPH_BodyInterface_ActivateBody(self->world->body_interface, bid);

  JPH_VehicleTransmission *trans =
      (JPH_VehicleTransmission *)JPH_TrackedVehicleController_GetTransmission(
          t_ctrl);
  int gear = JPH_VehicleTransmission_GetCurrentGear(trans);

  // Throttle is the max power requested by either track
  float throttle = fmaxf(fabsf(left), fabsf(right));

  // Kickstart/Neutral logic
  if (throttle > 0.01f) {
    if (gear == 0) {
      JPH_VehicleTransmission_Set(trans, 1, 1.0f); // Shift to 1
    }
  } else {
    if (gear != 0) {
      JPH_VehicleTransmission_Set(trans, 0, 0.0f); // Force Neutral
    }
  }

  JPH_TrackedVehicleController_SetDriverInput(t_ctrl, throttle, left, right,
                                              brake);

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}