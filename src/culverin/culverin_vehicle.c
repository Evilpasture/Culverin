#include "culverin_vehicle.h"

// vroom vroom
// this is paperwork and i did surgery in the core
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
  float brake = get_py_float_attr(w_dict, "brake_torque", 1500.0f);
  float handbrake = get_py_float_attr(w_dict, "handbrake_torque", 4000.0f);
  float susp_max = get_py_float_attr(w_dict, "suspension", 0.3f);
  float freq = get_py_float_attr(w_dict, "spring_freq", 1.5f);
  float damp = get_py_float_attr(w_dict, "spring_damp", 0.5f);

  // 3. Jolt Object Setup
  JPH_WheelSettingsWV *w = JPH_WheelSettingsWV_Create();
  // A standard wheel has an inertia of about 0.1 to 0.5
  JPH_WheelSettingsWV_SetInertia(w, 0.5f);
  JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings *)w, 0.05f);
  JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings *)w, susp_max); 
  JPH_SpringSettings spring = {JPH_SpringMode_FrequencyAndDamping, freq, damp};
  JPH_WheelSettings_SetSuspensionSpring((JPH_WheelSettings *)w, &spring);
  // The axis the wheel pivots around for steering
  JPH_WheelSettings_SetSteeringAxis((JPH_WheelSettings *)w, &(JPH_Vec3){0, 1.0f, 0});
  
  // The 'Up' direction for the wheel geometry
  JPH_WheelSettings_SetWheelUp((JPH_WheelSettings *)w, &(JPH_Vec3){0, 1.0f, 0});
  
  // The 'Forward' direction (the way it rolls)
  JPH_WheelSettings_SetWheelForward((JPH_WheelSettings *)w, &(JPH_Vec3){0, 0, 1.0f});
  
  // Suspension direction (the way the shock absorber moves) - usually opposite to Up
  JPH_WheelSettings_SetSuspensionDirection((JPH_WheelSettings *)w, &(JPH_Vec3){0, -1.0f, 0});
  JPH_WheelSettingsWV_SetMaxBrakeTorque(w, brake);
  if (pos.z > 0.1f) {
      JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.5f);
      JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, 0.0f);
  } else {
      JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.0f);
      JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, handbrake);
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

// --- Internal Helpers for Vehicle Construction ---

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

void cleanup_vehicle_resources(VehicleResources *r, uint32_t num_wheels,
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
  if (py_trans && py_trans != Py_None) {
      PyObject *o_mode = PyObject_GetAttrString(py_trans, "mode");
      if (o_mode) {
          t_mode = (int)PyLong_AsLong(o_mode);
          Py_DECREF(o_mode);
      }
      PyErr_Clear();
  }

  JPH_VehicleTransmissionSettings_SetMode(v_trans_set,
                                          (JPH_TransmissionMode)t_mode);
  JPH_VehicleTransmissionSettings_SetClutchStrength(
      v_trans_set, get_py_float_attr(py_trans, "clutch_strength", 2000.0f));

  // Extract Gear Ratios from Python list
  if (py_trans && py_trans != Py_None) {
      PyObject *py_ratios = PyObject_GetAttrString(py_trans, "ratios");
      if (py_ratios && PyList_Check(py_ratios)) {
          Py_ssize_t n = PyList_Size(py_ratios);
          float* r = PyMem_RawMalloc(n * sizeof(float));
          if (r) {
              for(Py_ssize_t i=0; i<n; i++) r[i] = (float)PyFloat_AsDouble(PyList_GetItem(py_ratios, i));
              JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, r, (int)n);
              PyMem_RawFree(r);
          }
      }
      Py_XDECREF(py_ratios);
      PyErr_Clear();
  } else {
      // DEFAULT GEARS: If no transmission object provided, give it some standard gears
      // Otherwise, the car will have 0 gears and won't move!
      float default_ratios[] = { 2.66f, 1.78f, 1.30f, 1.00f, 0.74f, 0.50f };
      JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, default_ratios, 6);
  }
  
  // Apply Differential Ratio from Python Transmission object
  float diff_ratio = get_py_float_attr(py_trans, "differential_ratio", 3.42f);
  uint32_t num_diffs = JPH_WheeledVehicleControllerSettings_GetDifferentialsCount(v_ctrl);
  for (uint32_t d = 0; d < num_diffs; d++) {
    JPH_VehicleDifferentialSettings ds;
    
    // 1. Get the current settings (Pass pointer to 'ds' as 3rd arg)
    JPH_WheeledVehicleControllerSettings_GetDifferential(v_ctrl, d, &ds);
    
    // 2. Modify the local copy
    ds.differentialRatio = diff_ratio;
    
    // 3. Set it back (Most wrappers have a matching SetDifferential)
    JPH_WheeledVehicleControllerSettings_SetDifferential(v_ctrl, d, &ds);
  }
}

// --- Main coordinate function (Complexity: 1) ---
static void configure_drivetrain(VehicleResources *r, PyObject *py_engine,
                                 PyObject *py_trans, const char *drive_str,
                                 uint32_t num_wheels) {
  // 1. Setup Diffs FIRST so they exist when we want to set ratios
  setup_vehicle_differentials(r->v_ctrl, drive_str, num_wheels);

  // 2. Setup Engine
  setup_engine(r->v_ctrl, r->t_curve, py_engine);

  // 3. Setup Transmission (Now can find the diffs to apply the ratio)
  setup_transmission(r->v_ctrl, r->v_trans_set, py_trans);
}

// --- Main Function ---

PyObject *PhysicsWorld_create_vehicle(PhysicsWorldObject *self,
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
  JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);   // Full grip at stop
  JPH_LinearCurve_AddPoint(r.f_curve, 0.1f, 2.0f);   // Peak grip (10% slip)
  JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, 1.2f);   // Sliding grip
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
  // --- COLLISION TESTER SETUP ---
  // Use Layer 2 for rays so they ignore Layer 1 (Chassis/Dynamic Bodies)
  r.tester = JPH_VehicleCollisionTesterRay_Create(2, &(JPH_Vec3){0, 1.0f, 0}, 2.0f);
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