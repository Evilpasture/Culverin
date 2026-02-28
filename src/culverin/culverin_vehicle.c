#include "culverin_vehicle.h"
#include "culverin_arg_indices.h"
#include "culverin_compiler_specifics.h"
#include "culverin_math.h"
#include "culverin_parsers.h"

// --- Wheel Configuration Defaults ---
static constexpr float WHEEL_RADIUS_DEFAULT           = 0.4f;
static constexpr float WHEEL_WIDTH_DEFAULT            = 0.2f;
static constexpr float WHEEL_BRAKE_TORQUE_DEFAULT     = 1500.0f;
static constexpr float WHEEL_HANDBRAKE_TORQUE_DEFAULT = 4000.0f;
static constexpr float WHEEL_SUSPENSION_MAX_DEFAULT   = 0.3f;
static constexpr float WHEEL_SUSPENSION_MIN_LENGTH    = 0.05f;
static constexpr float WHEEL_SPRING_FREQ_DEFAULT      = 1.5f;
static constexpr float WHEEL_SPRING_DAMP_DEFAULT      = 0.5f;
static constexpr float WHEEL_INERTIA_DEFAULT          = 0.5f;
static constexpr float WHEEL_STEER_THRESHOLD          = 0.1f;
static constexpr float WHEEL_MAX_STEER_ANGLE          = 0.5f;

// --- Engine Configuration Defaults ---
static constexpr float ENGINE_MAX_TORQUE_DEFAULT = 500.0f;
static constexpr float ENGINE_MAX_RPM_DEFAULT    = 7000.0f;
static constexpr float ENGINE_MIN_RPM_DEFAULT    = 1000.0f;
static constexpr float ENGINE_INERTIA_DEFAULT    = 0.5f;

// --- Transmission Configuration ---
static constexpr float TRANSMISSION_CLUTCH_STRENGTH_DEFAULT = 2000.0f;
static constexpr float TRANSMISSION_GEAR_RATIO_1            = 2.66f;
static constexpr float TRANSMISSION_GEAR_RATIO_2            = 1.78f;
static constexpr float TRANSMISSION_GEAR_RATIO_3            = 1.30f;
static constexpr float TRANSMISSION_GEAR_RATIO_DIRECT       = 1.00f;
static constexpr float TRANSMISSION_GEAR_RATIO_4            = 0.74f;
static constexpr float TRANSMISSION_GEAR_RATIO_5            = 0.50f;
static constexpr uint32_t TRANSMISSION_DEFAULT_GEAR_COUNT   = 6;
static constexpr float DIFFERENTIAL_RATIO_DEFAULT           = 3.42f;

// --- Friction Curve Points ---
static constexpr float FRICTION_CURVE_X1 = 0.1f;
static constexpr float FRICTION_CURVE_Y1 = 2.0f;
static constexpr float FRICTION_CURVE_Y2 = 1.2f;

// --- Vehicle Motion Thresholds ---
static constexpr float VEHICLE_COLLISION_TESTER_SCALE = 2.0f;
static constexpr float THROTTLE_INPUT_THRESHOLD       = 0.01f;
static constexpr float SPEED_DIRECTION_THRESHOLD      = 0.5f;
static constexpr float SPEED_MIN_THRESHOLD            = 0.1f;
static constexpr float ROLLING_RESISTANCE_COASTING    = 0.05f;

// --- Refactored Wheel Creation (Complexity: 2) ---
static JPH_WheelSettings *create_single_wheel(PyObject *w_dict, JPH_LinearCurve *f_curve) {
    PosStride pos;

    // 1. Parse Position using helper
    if (!parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
        PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 real numbers");
        return NULL;
    }

    // 2. Parse Float Attributes using consistent helper
    float radius    = get_py_float_attr(w_dict, "radius", WHEEL_RADIUS_DEFAULT);
    float width     = get_py_float_attr(w_dict, "width", WHEEL_WIDTH_DEFAULT);
    float brake     = get_py_float_attr(w_dict, "brake_torque", WHEEL_BRAKE_TORQUE_DEFAULT);
    float handbrake = get_py_float_attr(w_dict, "handbrake_torque", WHEEL_HANDBRAKE_TORQUE_DEFAULT);
    float susp_max  = get_py_float_attr(w_dict, "suspension", WHEEL_SUSPENSION_MAX_DEFAULT);
    float freq      = get_py_float_attr(w_dict, "spring_freq", WHEEL_SPRING_FREQ_DEFAULT);
    float damp      = get_py_float_attr(w_dict, "spring_damp", WHEEL_SPRING_DAMP_DEFAULT);

    // 3. Jolt Object Setup
    JPH_WheelSettingsWV *w = JPH_WheelSettingsWV_Create();
    // A standard wheel has an inertia of about 0.1 to 0.5
    JPH_WheelSettingsWV_SetInertia(w, WHEEL_INERTIA_DEFAULT);
    JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings *)w, WHEEL_SUSPENSION_MIN_LENGTH);
    JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings *)w, susp_max);
    JPH_SpringSettings spring = {JPH_SpringMode_FrequencyAndDamping, freq, damp};
    JPH_WheelSettings_SetSuspensionSpring((JPH_WheelSettings *)w, &spring);
    // The axis the wheel pivots around for steering
    JPH_WheelSettings_SetSteeringAxis((JPH_WheelSettings *)w, &(JPH_Vec3){0, 1.0f, 0});

    // The 'Up' direction for the wheel geometry
    JPH_WheelSettings_SetWheelUp((JPH_WheelSettings *)w, &(JPH_Vec3){0, 1.0f, 0});

    // The 'Forward' direction (the way it rolls)
    JPH_WheelSettings_SetWheelForward((JPH_WheelSettings *)w, &(JPH_Vec3){0, 0, 1.0f});

    // Suspension direction (the way the shock absorber moves) - usually opposite
    // to Up
    JPH_WheelSettings_SetSuspensionDirection((JPH_WheelSettings *)w, &(JPH_Vec3){0, -1.0f, 0});
    JPH_WheelSettingsWV_SetMaxBrakeTorque(w, brake);
    if (pos.z > WHEEL_STEER_THRESHOLD) {
        JPH_WheelSettingsWV_SetMaxSteerAngle(w, WHEEL_MAX_STEER_ANGLE);
        JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, 0.0f);
    } else {
        JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.0f);
        JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, handbrake);
    }
    JPH_WheelSettings_SetPosition((JPH_WheelSettings *)w,
                                  &(JPH_Vec3){(float)pos.x, (float)pos.y, (float)pos.z});
    JPH_WheelSettings_SetRadius((JPH_WheelSettings *)w, radius);
    JPH_WheelSettings_SetWidth((JPH_WheelSettings *)w, width);

    JPH_WheelSettingsWV_SetLongitudinalFriction(w, f_curve);
    JPH_WheelSettingsWV_SetLateralFriction(w, f_curve);

    // Steering logic (Simple branch)
    JPH_WheelSettingsWV_SetMaxSteerAngle(w, (pos.z > WHEEL_STEER_THRESHOLD) ? WHEEL_MAX_STEER_ANGLE
                                                                            : 0.0f);

    return (JPH_WheelSettings *)w;
}

// --- Internal Helpers for Vehicle Construction ---

static void setup_vehicle_differentials(JPH_WheeledVehicleControllerSettings *v_ctrl,
                                        const char *drive_str, uint32_t num_wheels) {
    if (strcmp(drive_str, "FWD") == 0) {
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
    } else if (strcmp(drive_str, "AWD") == 0 && num_wheels >= 4) {
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 0, 1);
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, 2, 3);
    } else { // RWD
        uint32_t i1 = (num_wheels >= 4) ? 2 : 0;
        uint32_t i2 = (num_wheels >= 4) ? 3 : 1;
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, (int)i1, (int)i2);
    }
}

void cleanup_vehicle_resources(VehicleResources *r, uint32_t num_wheels, PhysicsWorldObject *self) {
    if (r->j_veh) {
        // If it was already added to Jolt, we MUST remove it before destroying it
        if ((int)r->is_added_to_world && self && self->system) {
            JPH_PhysicsSystem_RemoveStepListener(
                self->system, JPH_VehicleConstraint_AsPhysicsStepListener(r->j_veh));
            JPH_PhysicsSystem_RemoveConstraint(self->system, (JPH_Constraint *)r->j_veh);
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
        JPH_VehicleControllerSettings_Destroy((JPH_VehicleControllerSettings *)r->v_ctrl);
    }

    if (r->w_settings) {
        for (auto i = 0u; i < num_wheels; i++) {
            if (r->w_settings[i]) {
                JPH_WheelSettings_Destroy(r->w_settings[i]);
            }
        }
        CULV_RAW_FREE((void *)r->w_settings);
    }

    if (r->f_curve) {
        JPH_LinearCurve_Destroy(r->f_curve);
    }
    if (r->t_curve) {
        JPH_LinearCurve_Destroy(r->t_curve);
    }
}

// --- Sub-helper: Engine Configuration ---
static void setup_engine(JPH_WheeledVehicleControllerSettings *v_ctrl, JPH_LinearCurve *t_curve,
                         PyObject *py_engine) {
    JPH_VehicleEngineSettings eng_set;
    JPH_VehicleEngineSettings_Init(&eng_set);

    // Flat execution: no nesting, no hidden macro branches
    eng_set.maxTorque = get_py_float_attr(py_engine, "max_torque", ENGINE_MAX_TORQUE_DEFAULT);
    eng_set.maxRPM    = get_py_float_attr(py_engine, "max_rpm", ENGINE_MAX_RPM_DEFAULT);
    eng_set.minRPM    = get_py_float_attr(py_engine, "min_rpm", ENGINE_MIN_RPM_DEFAULT);
    eng_set.inertia   = get_py_float_attr(py_engine, "inertia", ENGINE_INERTIA_DEFAULT);

    eng_set.normalizedTorque = t_curve;

    JPH_WheeledVehicleControllerSettings_SetEngine(v_ctrl, &eng_set);
}

// --- Sub-helper: Transmission Configuration ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void setup_transmission(JPH_WheeledVehicleControllerSettings *v_ctrl,
                               JPH_VehicleTransmissionSettings *v_trans_set, PyObject *py_trans) {
    // Determine mode
    auto t_mode = 1; // Default Manual
    if (py_trans && py_trans != Py_None) {
        PyObject *o_mode = PyObject_GetAttrString(py_trans, "mode");
        if (o_mode) {
            t_mode = PyLong_AsLong(o_mode);
            Py_DECREF(o_mode);
        }
        PyErr_Clear();
    }

    JPH_VehicleTransmissionSettings_SetMode(v_trans_set, (JPH_TransmissionMode)t_mode);
    JPH_VehicleTransmissionSettings_SetClutchStrength(
        v_trans_set,
        get_py_float_attr(py_trans, "clutch_strength", TRANSMISSION_CLUTCH_STRENGTH_DEFAULT));

    // Extract Gear Ratios from Python list
    if (py_trans && py_trans != Py_None) {
        PyObject *py_ratios = PyObject_GetAttrString(py_trans, "ratios");
        if (py_ratios && PyList_Check(py_ratios)) {
            Py_ssize_t n = PyList_Size(py_ratios);
            float *r     = CULV_RAW_MALLOC((size_t)n * sizeof(float));
            if (r) {
                for (Py_ssize_t i = 0; i < n; i++) {
                    r[i] = (float)PyFloat_AsDouble(PyList_GetItem(py_ratios, i));
                }
                JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, r, (uint32_t)n);
                CULV_RAW_FREE(r);
            }
        }
        Py_XDECREF(py_ratios);
        PyErr_Clear();
    } else {
        // DEFAULT GEARS: If no transmission object provided, give it some standard
        // gears Otherwise, the car will have 0 gears and won't move!
        float default_ratios[] = {TRANSMISSION_GEAR_RATIO_1, TRANSMISSION_GEAR_RATIO_2,
                                  TRANSMISSION_GEAR_RATIO_3, TRANSMISSION_GEAR_RATIO_DIRECT,
                                  TRANSMISSION_GEAR_RATIO_4, TRANSMISSION_GEAR_RATIO_5};
        JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, default_ratios,
                                                      TRANSMISSION_DEFAULT_GEAR_COUNT);
    }

    // Apply Differential Ratio from Python Transmission object
    float diff_ratio =
        get_py_float_attr(py_trans, "differential_ratio", DIFFERENTIAL_RATIO_DEFAULT);
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
static void configure_drivetrain(VehicleResources *r, PyObject *py_engine, PyObject *py_trans,
                                 const char *drive_str, uint32_t num_wheels) {
    // 1. Setup Diffs FIRST so they exist when we want to set ratios
    setup_vehicle_differentials(r->v_ctrl, drive_str, num_wheels);

    // 2. Setup Engine
    setup_engine(r->v_ctrl, r->t_curve, py_engine);

    // 3. Setup Transmission (Now can find the diffs to apply the ratio)
    setup_transmission(r->v_ctrl, r->v_trans_set, py_trans);
}

// --- Main Function ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_vehicle(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames) {
    // --- 1. FAST ARGUMENT PARSING ---
    uint64_t chassis_h    = 0;
    PyObject *py_wheels   = NULL;
    PyObject *py_drive    = NULL; // Wrapper for drive_str
    PyObject *py_engine   = NULL;
    PyObject *py_trans    = NULL;
    const char *drive_str = "RWD";

    void *targets[CreateVehicle_COUNT];
    targets[IDX_CV_CHASSIS] = &chassis_h;
    targets[IDX_CV_WHEELS]  = &py_wheels;
    targets[IDX_CV_DRIVE]   = &py_drive;
    targets[IDX_CV_ENGINE]  = &py_engine;
    targets[IDX_CV_TRANS]   = &py_trans;

    if (!FastParse_Unified(args, nargs, kwnames, &CreateVehicleParser, targets)) {
        return NULL;
    }

    // Handle string conversion for the drivetrain
    if (py_drive && PyUnicode_Check(py_drive)) {
        drive_str = PyUnicode_AsUTF8(py_drive);
    }

    // --- LOGIC PRESERVATION START ---
    
    if (!PyList_Check(py_wheels) || PyList_Size(py_wheels) < 2) {
        return PyErr_Format(PyExc_ValueError, "Wheels must be a list of at least 2 dictionaries");
    }
    uint32_t num_wheels = (uint32_t)PyList_Size(py_wheels);

    // Necessary for Py_UNBLOCK_THREADS / Py_BLOCK_THREADS
    PyThreadState *_save = NULL;

    // --- RESOLVE CHASSIS (Shadow Lock + Command Sync) ---
    SHADOW_LOCK(&self->shadow_lock);
    sync_and_flush_internal(self);

    uint32_t slot = 0;
    if (!unpack_handle(self, chassis_h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_ValueError, "Invalid or stale chassis handle");
    }
    JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- PRE-JOLT RESOURCE ALLOCATION (GIL Held) ---
    VehicleResources r = {0};
    r.f_curve          = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(r.f_curve, FRICTION_CURVE_X1, FRICTION_CURVE_Y1);
    JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, FRICTION_CURVE_Y2);

    r.t_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(r.t_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(r.t_curve, 1.0f, 1.0f);

    r.w_settings  = (JPH_WheelSettings **)CULV_RAW_CALLOC(num_wheels, sizeof(JPH_WheelSettings *));
    r.v_ctrl      = JPH_WheeledVehicleControllerSettings_Create();
    r.v_trans_set = JPH_VehicleTransmissionSettings_Create();

    for (auto i = 0u; i < num_wheels; i++) {
        r.w_settings[i] = create_single_wheel(PyList_GetItem(py_wheels, i), r.f_curve);
        if (!r.w_settings[i]) goto python_fail;
    }

    configure_drivetrain(&r, py_engine, py_trans, drive_str, num_wheels);

    // --- JOLT COMMIT (Release GIL, Lock Jolt) ---
    bool jolt_locked = false;
    Py_UNBLOCK_THREADS;

    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    jolt_locked = true;

    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock                  = {0};
    JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);

    if (UNLIKELY(!lock.body)) goto jolt_fail;

    JPH_VehicleConstraintSettings v_set;
    JPH_VehicleConstraintSettings_Init(&v_set);
    v_set.wheelsCount = num_wheels;
    v_set.wheels      = r.w_settings;
    v_set.controller  = (JPH_VehicleControllerSettings *)r.v_ctrl;

    r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
    if (!r.j_veh) goto jolt_fail;

    r.tester = JPH_VehicleCollisionTesterRay_Create(LAYER_DYNAMIC, &(JPH_Vec3){0, 1.0f, 0},
                                                    VEHICLE_COLLISION_TESTER_SCALE);
    if (!r.tester) goto jolt_fail;

    JPH_VehicleConstraint_SetVehicleCollisionTester(r.j_veh,
                                                    (JPH_VehicleCollisionTester *)r.tester);

    JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint *)r.j_veh);
    JPH_PhysicsSystem_AddStepListener(self->system,
                                      JPH_VehicleConstraint_AsPhysicsStepListener(r.j_veh));
    r.is_added_to_world = true;

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    jolt_locked = false;

    Py_BLOCK_THREADS;

    // --- PYTHON WRAPPER ---
    auto *st  = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    auto *obj = (VehicleObject *)PyObject_New(VehicleObject, (PyTypeObject *)st->VehicleType);

    if (!obj) {
        SHADOW_LOCK(&self->shadow_lock);
        cleanup_vehicle_resources(&r, num_wheels, self);
        SHADOW_UNLOCK(&self->shadow_lock);
        return NULL;
    }

    obj->vehicle               = r.j_veh;
    obj->tester                = (JPH_VehicleCollisionTester *)r.tester;
    obj->world                 = self;
    obj->num_wheels            = num_wheels;
    obj->current_gear          = 0;
    obj->wheel_settings        = r.w_settings;
    obj->controller_settings   = (JPH_VehicleControllerSettings *)r.v_ctrl;
    obj->transmission_settings = r.v_trans_set;
    obj->friction_curve        = r.f_curve;
    obj->torque_curve          = r.t_curve;

    Py_INCREF(self);
    return (PyObject *)obj;

jolt_fail:
    if (lock.body) JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    if (jolt_locked) NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    Py_BLOCK_THREADS;

python_fail:
    SHADOW_LOCK(&self->shadow_lock);
    cleanup_vehicle_resources(&r, num_wheels, self);
    SHADOW_UNLOCK(&self->shadow_lock);

    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Jolt vehicle creation failed");
    }
    return NULL;
}

// --- Vehicles Methods ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule Vehicle_set_input(VehicleObject *self, PyObject *const *args,
                                                      Py_ssize_t nargs, PyObject *kwnames) {
    // 1. FAST PARSE (Zero-Allocation)
    float forward   = 0.0f;
    float right     = 0.0f;
    float brake     = 0.0f;
    float handbrake = 0.0f;

    void *targets[VehicleInput_COUNT];
    targets[IDX_VI_FWD]   = &forward;
    targets[IDX_VI_RIGHT] = &right;
    targets[IDX_VI_BRAKE] = &brake;
    targets[IDX_VI_HAND]  = &handbrake;

    if (!FastParse_Unified(args, nargs, kwnames, &VehicleInputParser, targets)) {
        return NULL;
    }

    // 2. STATE MACHINE & JOLT SYNC
    SHADOW_LOCK(&self->world->shadow_lock);

    // Guard against mid-step access or destroyed vehicles
    if (UNLIKELY(self->world->is_stepping || !self->vehicle)) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    auto *controller =
        (JPH_WheeledVehicleController *)JPH_VehicleConstraint_GetController(self->vehicle);
    JPH_BodyID chassis_id = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));
    JPH_BodyInterface *bi = self->world->body_interface;

    // Ensure the vehicle is awake to respond to inputs
    JPH_BodyInterface_ActivateBody(bi, chassis_id);

    // Get current physics state for arcade drive logic
    JPH_STACK_ALLOC(JPH_Vec3, linear_vel);
    JPH_STACK_ALLOC(JPH_Quat, chassis_q);
    JPH_BodyInterface_GetLinearVelocity(bi, chassis_id, linear_vel);
    JPH_BodyInterface_GetRotation(bi, chassis_id, chassis_q);

    // Calculate forward speed via dot product (Project velocity onto chassis forward vector)
    JPH_Vec3 world_fwd;
    manual_vec3_rotate_by_quat(&(JPH_Vec3){0, 0, 1.0f}, chassis_q, &world_fwd);
    float speed = (linear_vel->x * world_fwd.x) + (linear_vel->y * world_fwd.y) +
                  (linear_vel->z * world_fwd.z);

    float input_throttle = fabsf(forward);
    float input_brake    = brake;

    JPH_VehicleTransmission *trans =
        (JPH_VehicleTransmission *)JPH_WheeledVehicleController_GetTransmission(controller);
    int cur_gear = JPH_VehicleTransmission_GetCurrentGear(trans);

    // 3. DRIVE LOGIC (Arcade Style State Machine)
    if (forward > THROTTLE_INPUT_THRESHOLD) {
        // FORWARD DRIVE
        JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Auto);
        if (cur_gear <= 0 && speed > -SPEED_DIRECTION_THRESHOLD) {
            JPH_VehicleTransmission_Set(trans, 1, 1.0f);
        }
        // Arcade Brake: Apply brakes if we are still moving backwards
        if (speed < -SPEED_MIN_THRESHOLD)
            input_brake = 1.0f;

    } else if (forward < -THROTTLE_INPUT_THRESHOLD) {
        // REVERSE DRIVE
        JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Manual);
        if (cur_gear != -1 && speed < SPEED_DIRECTION_THRESHOLD) {
            JPH_VehicleTransmission_Set(trans, -1, 1.0f);
        }
        // Arcade Brake: Apply brakes if we are still moving forwards
        if (speed > SPEED_MIN_THRESHOLD)
            input_brake = 1.0f;

    } else {
        // NEUTRAL / COASTING
        input_throttle = 0.0f;
        JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Manual);
        if (cur_gear != 0) {
            JPH_VehicleTransmission_Set(trans, 0, 0.0f); // Clutch out
        }
        // Rolling Resistance: Stop slow idle creep
        if (fabsf(speed) > SPEED_MIN_THRESHOLD) {
            input_brake = fmaxf(input_brake, ROLLING_RESISTANCE_COASTING);
        }
    }

    // 4. Final Application to Jolt
    JPH_WheeledVehicleController_SetDriverInput(controller, input_throttle, right, input_brake,
                                                handbrake);

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

// --- Helper for faster return value creation ---
static inline PyObject *pack_transform(double px, double py, double pz, float rx, float ry,
                                       float rz, float rw) {
    PyObject *pos =
        PyTuple_Pack(3, PyFloat_FromDouble(px), PyFloat_FromDouble(py), PyFloat_FromDouble(pz));
    PyObject *rot = PyTuple_Pack(4, PyFloat_FromDouble(rx), PyFloat_FromDouble(ry),
                                 PyFloat_FromDouble(rz), PyFloat_FromDouble(rw));
    PyObject *res = PyTuple_Pack(2, pos, rot);
    Py_DECREF(pos);
    Py_DECREF(rot);
    return res;
}

PyCFunction_DeclareMethodFromModule Vehicle_get_wheel_transform(VehicleObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames) {
    uint32_t index = 0;
    void *targets[WheelIdx_COUNT];
    targets[IDX_WH_INDEX] = &index;

    if (!FastParse_Unified(args, nargs, kwnames, &WheelIdxParser, targets))
        return NULL;

    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    if (!self->vehicle || index >= self->num_wheels) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    JPH_STACK_ALLOC(JPH_RMat4, transform);
    JPH_Vec3 right = {1.0f, 0.0f, 0.0f}, up = {0.0f, 1.0f, 0.0f};
    JPH_VehicleConstraint_GetWheelWorldTransform(self->vehicle, index, &right, &up, transform);

    // Extraction logic remains the same (Correct RMat44 mapping)
    auto px = transform->column3.x;
    auto py = transform->column3.y;
    auto pz = transform->column3.z;

    JPH_STACK_ALLOC(JPH_Mat4, rot_only_mat);
    rot_only_mat->column[0] = transform->column[0];
    rot_only_mat->column[1] = transform->column[1];
    rot_only_mat->column[2] = transform->column[2];
    rot_only_mat->column[3] = (JPH_Vec4){0, 0, 0, 1.0f};

    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_Mat4_GetQuaternion(rot_only_mat, q);
    SHADOW_UNLOCK(&self->world->shadow_lock);

    return pack_transform(px, py, pz, q->x, q->y, q->z, q->w);
}

PyCFunction_DeclareMethodFromModule Vehicle_get_wheel_local_transform(VehicleObject *self,
                                                                      PyObject *const *args,
                                                                      Py_ssize_t nargs,
                                                                      PyObject *kwnames) {
    uint32_t index = 0;
    void *targets[WheelIdx_COUNT];
    targets[IDX_WH_INDEX] = &index;

    if (!FastParse_Unified(args, nargs, kwnames, &WheelIdxParser, targets))
        return NULL;

    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    if (!self->vehicle || index >= self->num_wheels) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    const JPH_Wheel *w_ptr      = JPH_VehicleConstraint_GetWheel(self->vehicle, index);
    const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w_ptr);
    JPH_Vec3 local_pos_check;
    JPH_WheelSettings_GetPosition(ws, &local_pos_check);

    JPH_Vec3 right = {(local_pos_check.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f};
    JPH_Vec3 up    = {0.0f, 1.0f, 0.0f};

    JPH_STACK_ALLOC(JPH_Mat4, local_transform);
    JPH_VehicleConstraint_GetWheelLocalTransform(self->vehicle, index, &right, &up,
                                                 local_transform);

    float lx = local_transform->column[3].x;
    float ly = local_transform->column[3].y;
    float lz = local_transform->column[3].z;

    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_Mat4_GetQuaternion(local_transform, q);
    SHADOW_UNLOCK(&self->world->shadow_lock);

    return pack_transform(lx, ly, lz, q->x, q->y, q->z, q->w);
}

PyCFunction_DeclareMethodFromModule Vehicle_get_debug_state(VehicleObject *self,
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
    auto *controller =
        (JPH_WheeledVehicleController *)JPH_VehicleConstraint_GetController(self->vehicle);
    if (!controller) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        Py_RETURN_NONE;
    }

    const auto *engine = JPH_WheeledVehicleController_GetEngine(controller);
    const auto *trans  = JPH_WheeledVehicleController_GetTransmission(controller);

    // 3. CAPTURE INPUTS
    float in_fwd = JPH_WheeledVehicleController_GetForwardInput(controller);
    float in_brk = JPH_WheeledVehicleController_GetBrakeInput(controller);

    // 4. CAPTURE DRIVETRAIN
    float rpm           = JPH_VehicleEngine_GetCurrentRPM(engine);
    float engine_torque = JPH_VehicleEngine_GetTorque(engine, in_fwd);
    int gear            = JPH_VehicleTransmission_GetCurrentGear(trans);
    float clutch        = JPH_VehicleTransmission_GetClutchFriction(trans);

    DEBUG_LOG("=== VEHICLE DEBUG STATE ===");
    DEBUG_LOG("  Inputs: Fwd=%.2f | Brk=%.2f", in_fwd, in_brk);
    DEBUG_LOG("  Engine: %.2f RPM | Torque: %.2f Nm", rpm, engine_torque);
    DEBUG_LOG("  Trans : Gear %d | Clutch Friction: %.2f", gear, clutch);

    // 5. CAPTURE WHEEL STATE
    for (uint32_t i = 0; i < self->num_wheels; i++) {
        const JPH_Wheel *w          = JPH_VehicleConstraint_GetWheel(self->vehicle, i);
        const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w);

        bool contact   = JPH_Wheel_HasContact(w);
        float susp_len = JPH_Wheel_GetSuspensionLength(w);
        float ang_vel  = JPH_Wheel_GetAngularVelocity(w);
        float radius   = JPH_WheelSettings_GetRadius(ws);

        float tire_speed  = ang_vel * radius;
        float long_lambda = JPH_Wheel_GetLongitudinalLambda(w);
        float lat_lambda  = JPH_Wheel_GetLateralLambda(w);

        DEBUG_LOG("  Wheel %u: %s", i, contact ? "GROUND" : "AIR   ");
        DEBUG_LOG("    Susp: %.3fm | AngVel: %.2f rad/s | SurfSpd: %.2f m/s", susp_len, ang_vel,
                  tire_speed);
        DEBUG_LOG("    Trac: Long=%.2f | Lat=%.2f", long_lambda, lat_lambda);
    }
    DEBUG_LOG("===========================");

    // 6. UNLOCK
    SHADOW_UNLOCK(&self->world->shadow_lock);

    Py_RETURN_NONE;
}

// --- Vehicle GC Support ---
PyType_DeclareSlot_StatusFromModule Vehicle_traverse(VehicleObject *self, visitproc visit,
                                                     void *arg) {
    Py_VISIT(self->world);
    return 0;
}

PyType_DeclareSlot_StatusFromModule Vehicle_clear(VehicleObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

// --- Internal C Logic (No Python return value to ignore) ---
static void Vehicle_internal_cleanup(VehicleObject *self) {
    if (!self->world) {
        return;
    }

    SHADOW_LOCK(&self->world->shadow_lock);

    BLOCK_UNTIL_NOT_STEPPING(self->world);
    BLOCK_UNTIL_NOT_QUERYING(self->world);

    if (!self->vehicle) {
        SHADOW_UNLOCK(&self->world->shadow_lock);
        return;
    }

    // Capture pointers and NULL the struct members immediately
    JPH_VehicleConstraint *j_veh             = self->vehicle;
    JPH_VehicleCollisionTester *tester       = self->tester;
    JPH_VehicleControllerSettings *v_ctrl    = self->controller_settings;
    JPH_VehicleTransmissionSettings *v_trans = self->transmission_settings;
    JPH_WheelSettings **wheels               = self->wheel_settings;
    JPH_LinearCurve *f_curve                 = self->friction_curve;
    JPH_LinearCurve *t_curve                 = self->torque_curve;
    auto wheel_count                         = self->num_wheels;

    self->vehicle               = NULL;
    self->tester                = NULL;
    self->controller_settings   = NULL;
    self->transmission_settings = NULL;
    self->wheel_settings        = NULL;
    self->friction_curve        = NULL;
    self->torque_curve          = NULL;

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // Safely destroy Jolt objects outside the lock
    if (j_veh) {
        JPH_PhysicsStepListener *step_listener = JPH_VehicleConstraint_AsPhysicsStepListener(j_veh);
        JPH_PhysicsSystem_RemoveStepListener(self->world->system, step_listener);
        JPH_PhysicsSystem_RemoveConstraint(self->world->system, (JPH_Constraint *)j_veh);
        JPH_Constraint_Destroy((JPH_Constraint *)j_veh);
    }

    if (tester) {
        JPH_VehicleCollisionTester_Destroy(tester);
    }
    if (v_ctrl) {
        JPH_VehicleControllerSettings_Destroy(v_ctrl);
    }
    if (v_trans) {
        JPH_VehicleTransmissionSettings_Destroy(v_trans);
    }

    if (wheels) {
        for (auto i = 0u; i < wheel_count; i++) {
            if (wheels[i]) {
                JPH_WheelSettings_Destroy(wheels[i]);
            }
        }
        CULV_RAW_FREE((void *)wheels);
    }

    if (f_curve) {
        JPH_LinearCurve_Destroy(f_curve);
    }
    if (t_curve) {
        JPH_LinearCurve_Destroy(t_curve);
    }
}

// --- Python Wrapper ---
// Using [[nodiscard]] on this will now only affect Python callers
PyCFunction_DeclareMethodFromModule Vehicle_destroy(VehicleObject *self,
                                                    PyObject *Py_UNUSED(ignored)) {
    Vehicle_internal_cleanup(self);
    Py_RETURN_NONE;
}

// --- Dealloc Slot ---
PyType_DeclareSlot_VoidFromModule Vehicle_dealloc(VehicleObject *self) {
    PyObject_GC_UnTrack(self);

    // Call the internal C logic directly
    // This avoids [[nodiscard]] warnings because the return is void
    Vehicle_internal_cleanup(self);

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}