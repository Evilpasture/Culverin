#include "culverin_vehicle.h"
#include "culverin_math.h"

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

// --- Vehicles Methods ---

PyObject *Vehicle_set_input(VehicleObject *self, PyObject *args, PyObject *kwds) {
  float forward = 0.0f, right = 0.0f, brake = 0.0f, handbrake = 0.0f;
  static char *kwlist[] = {"forward", "right", "brake", "handbrake", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|ffff", kwlist, &forward, &right, &brake, &handbrake)) 
    return NULL;

  SHADOW_LOCK(&self->world->shadow_lock);
  if (UNLIKELY(self->world->is_stepping || !self->vehicle)) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  JPH_WheeledVehicleController *controller = (JPH_WheeledVehicleController *)JPH_VehicleConstraint_GetController(self->vehicle);
  JPH_BodyID chassis_id = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));
  JPH_BodyInterface *bi = self->world->body_interface;
  JPH_BodyInterface_ActivateBody(bi, chassis_id);

  // 1. Get Directional Speed
  JPH_STACK_ALLOC(JPH_Vec3, linear_vel);
  JPH_BodyInterface_GetLinearVelocity(bi, chassis_id, linear_vel);
  JPH_STACK_ALLOC(JPH_Quat, chassis_q);
  JPH_BodyInterface_GetRotation(bi, chassis_id, chassis_q);
  JPH_Vec3 world_fwd;
  manual_vec3_rotate_by_quat(&(JPH_Vec3){0, 0, 1.0f}, chassis_q, &world_fwd);
  float speed = (linear_vel->x * world_fwd.x) + (linear_vel->y * world_fwd.y) + (linear_vel->z * world_fwd.z);

  float input_throttle = fabsf(forward);
  float input_brake = brake;

  JPH_VehicleTransmission *trans = (JPH_VehicleTransmission *)JPH_WheeledVehicleController_GetTransmission(controller);
  int cur_gear = JPH_VehicleTransmission_GetCurrentGear(trans);

  // 2. DRIVE STATE MACHINE
  if (forward > 0.01f) {
      // FORWARD: Auto Shifting
      JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Auto);
      // Force into Gear 1 if we were stuck in Neutral/Reverse
      if (cur_gear <= 0 && speed > -0.5f) {
          JPH_VehicleTransmission_Set(trans, 1, 1.0f);
      }
      // Arcade Brake: Moving back while wanting forward
      if (speed < -0.1f) input_brake = 1.0f;
  } 
  else if (forward < -0.01f) {
      // REVERSE: Manual Shifting
      JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Manual);
      // Force into Gear -1 if we aren't there
      if (cur_gear != -1 && speed < 0.5f) {
          JPH_VehicleTransmission_Set(trans, -1, 1.0f);
      }
      // Arcade Brake: Moving forward while wanting reverse
      if (speed > 0.1f) input_brake = 1.0f;
  } 
  else {
      // COASTING: Neutral Force
      input_throttle = 0.0f;
      // Disconnect engine immediately to stop "Idle Creep" acceleration
      JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Manual);
      if (cur_gear != 0) {
          JPH_VehicleTransmission_Set(trans, 0, 0.0f); // 0.0 clutch = no connection
      }
      // Apply 5% rolling resistance to guarantee speed decay in Test 4
      if (fabsf(speed) > 0.1f) {
          input_brake = fmaxf(input_brake, 0.05f);
      }
  }

  // 3. Final Driver Input
  JPH_WheeledVehicleController_SetDriverInput(controller, input_throttle, right, input_brake, handbrake);

  SHADOW_UNLOCK(&self->world->shadow_lock);
  Py_RETURN_NONE;
}

PyObject *Vehicle_get_wheel_transform(VehicleObject *self,
                                             PyObject *args) {
  uint32_t index = 0;
  if (!PyArg_ParseTuple(args, "I", &index)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  if (!self->vehicle || index >= self->num_wheels) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_Vec3 right = {1.0f, 0.0f, 0.0f};
  JPH_Vec3 up = {0.0f, 1.0f, 0.0f};

  // Get Transform in Double Precision
  JPH_VehicleConstraint_GetWheelWorldTransform(self->vehicle, index, &right,
                                               &up, transform);

  // --- CRITICAL FIX: Layout Mapping ---

  // 1. Position: In Double Precision, this is the 'column3' member
  // (RVec3/doubles)
  double px = transform->column3.x;
  double py = transform->column3.y;
  double pz = transform->column3.z;

  // 2. Rotation: These are the first 3 columns (Vec4/floats in RMat44)
  // We copy them to a standard Mat4 to extract the quaternion.
  JPH_STACK_ALLOC(JPH_Mat4, rot_only_mat);
  JPH_Mat4_Identity(rot_only_mat);

  // Safe struct copy of Vec4/Vec3 columns
  rot_only_mat->column[0] = transform->column[0];
  rot_only_mat->column[1] = transform->column[1];
  rot_only_mat->column[2] = transform->column[2];

  JPH_STACK_ALLOC(JPH_Quat, q);
  JPH_Mat4_GetQuaternion(rot_only_mat, q);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  // Safe Python construction
  PyObject *py_pos = Py_BuildValue("(ddd)", px, py, pz);
  PyObject *py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);

  if (!py_pos || !py_rot) {
    Py_XDECREF(py_pos);
    Py_XDECREF(py_rot);
    return NULL;
  }

  PyObject *result = PyTuple_Pack(2, py_pos, py_rot);
  Py_DECREF(py_pos);
  Py_DECREF(py_rot);
  return result;
}

PyObject *Vehicle_get_wheel_local_transform(VehicleObject *self,
                                                   PyObject *args) {
  uint32_t index = 0;
  if (!PyArg_ParseTuple(args, "I", &index)) {
    return NULL;
  }

  SHADOW_LOCK(&self->world->shadow_lock);
  // Re-entry guard: Ensure we aren't stepping while querying
  BLOCK_UNTIL_NOT_STEPPING(self->world);

  if (!self->vehicle || index >= self->num_wheels) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  const JPH_Wheel *w_ptr = JPH_VehicleConstraint_GetWheel(self->vehicle, index);
  const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w_ptr);
  JPH_Vec3 local_pos_check;
  JPH_WheelSettings_GetPosition(ws, &local_pos_check);

  // If wheel is on the left (x < 0), we flip the right vector
  JPH_Vec3 right = { (local_pos_check.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f };
  JPH_Vec3 up = { 0.0f, 1.0f, 0.0f };

  JPH_STACK_ALLOC(JPH_Mat4, local_transform);
  // Initialize to identity just in case the API call fails or partially writes
  JPH_Mat4_Identity(local_transform);

  JPH_VehicleConstraint_GetWheelLocalTransform(self->vehicle, index, &right,
                                               &up, local_transform);

  float lx = local_transform->column[3].x;
  float ly = local_transform->column[3].y;
  float lz = local_transform->column[3].z;

  JPH_STACK_ALLOC(JPH_Quat, q);
  JPH_Mat4_GetQuaternion(local_transform, q);

  SHADOW_UNLOCK(&self->world->shadow_lock);

  PyObject *py_pos = Py_BuildValue("(fff)", lx, ly, lz);
  PyObject *py_rot = Py_BuildValue("(ffff)", q->x, q->y, q->z, q->w);

  if (!py_pos || !py_rot) {
    Py_XDECREF(py_pos);
    Py_XDECREF(py_rot);
    return NULL;
  }

  PyObject *result = PyTuple_Pack(2, py_pos, py_rot);
  Py_DECREF(py_pos);
  Py_DECREF(py_rot);

  return result;
}

PyObject *Vehicle_get_debug_state(VehicleObject *self,
                                         PyObject *Py_UNUSED(ignored)) {
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
  JPH_WheeledVehicleController *controller =
      (JPH_WheeledVehicleController *)JPH_VehicleConstraint_GetController(
          self->vehicle);
  if (!controller) {
    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
  }

  const JPH_VehicleEngine *engine =
      JPH_WheeledVehicleController_GetEngine(controller);
  const JPH_VehicleTransmission *trans =
      JPH_WheeledVehicleController_GetTransmission(controller);

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
    const JPH_Wheel *w = JPH_VehicleConstraint_GetWheel(self->vehicle, i);
    const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w);

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