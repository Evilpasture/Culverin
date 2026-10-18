#include "vehicles/Vehicle.h"
#include "bindings/BindingCommon.h"
#include "bindings/ModuleState.h"
#include "bindings/PythonSupport.h"
#include "bindings/parsing/ParserRegistry.hpp"
#include "bindings/parsing/Parsers.h"
#include "bindings/physics/PhysicsSync.h"
#include "core/CompilerSpecifics.h"
#include <math.h>

// --- Wheel Configuration Defaults ---
static constexpr float kWheelRadiusDefault          = 0.4f;
static constexpr float kWheelWidthDefault           = 0.2f;
static constexpr float kWheelBrakeTorqueDefault     = 1500.0f;
static constexpr float kWheelHandbrakeTorqueDefault = 4000.0f;
static constexpr float kWheelSuspensionMaxDefault   = 0.3f;
static constexpr float kWheelSuspensionMinLength    = 0.05f;
static constexpr float kWheelSpringFreqDefault      = 1.5f;
static constexpr float kWheelSpringDampDefault      = 0.5f;
static constexpr float kWheelInertiaDefault         = 0.5f;
static constexpr float kWheelSteerThreshold         = 0.1f;
static constexpr float kWheelMaxSteerAngle          = 0.5f;

// --- Engine Configuration Defaults ---
static constexpr float kEngineMaxTorqueDefault = 500.0f;
static constexpr float kEngineMaxRpmDefault    = 7000.0f;
static constexpr float kEngineMinRpmDefault    = 1000.0f;
static constexpr float kEngineInertiaDefault   = 0.5f;

// --- Transmission Configuration ---
static constexpr float kTransmissionClutchStrengthDefault = 2000.0f;
static constexpr float kTransmissionGearRatio1            = 2.66f;
static constexpr float kTransmissionGearRatio2            = 1.78f;
static constexpr float kTransmissionGearRatio3            = 1.30f;
static constexpr float kTransmissionGearRatioDirect       = 1.00f;
static constexpr float kTransmissionGearRatio4            = 0.74f;
static constexpr float kTransmissionGearRatio5            = 0.50f;
static constexpr uint32_t kTransmissionDefaultGearCount   = 6;
static constexpr float kDifferentialRatioDefault          = 3.42f;

// --- Friction Curve Points ---
static constexpr float kFrictionCurveX1 = 0.1f;
static constexpr float kFrictionCurveY1 = 2.0f;
static constexpr float kFrictionCurveY2 = 1.2f;

// --- Vehicle Motion Thresholds ---
static constexpr float kVehicleCollisionTesterScale = 2.0f;
static constexpr float kThrottleInputThreshold      = 0.01f;
static constexpr float kSpeedDirectionThreshold     = 0.5f;
static constexpr float kSpeedMinThreshold           = 0.1f;
static constexpr float kRollingResistanceCoasting   = 0.05f;

// --- Refactored Wheel Creation (Complexity: 2) ---
static JPH_WheelSettings *create_single_wheel(PyObject *w_dict, JPH_LinearCurve *f_curve) {
    CPH_PosStride pos;

    // 1. Parse Position using helper
    if (!CPH_parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
        PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 real numbers");
        return nullptr;
    }

    // 2. Parse Float Attributes using consistent helper
    float radius    = CPH_get_py_attr(w_dict, "radius", kWheelRadiusDefault);
    float width     = CPH_get_py_attr(w_dict, "width", kWheelWidthDefault);
    float brake     = CPH_get_py_attr(w_dict, "brake_torque", kWheelBrakeTorqueDefault);
    float handbrake = CPH_get_py_attr(w_dict, "handbrake_torque", kWheelHandbrakeTorqueDefault);
    float susp_max  = CPH_get_py_attr(w_dict, "suspension", kWheelSuspensionMaxDefault);
    float freq      = CPH_get_py_attr(w_dict, "spring_freq", kWheelSpringFreqDefault);
    float damp      = CPH_get_py_attr(w_dict, "spring_damp", kWheelSpringDampDefault);

    // 3. Jolt Object Setup
    JPH_WheelSettingsWV *w = JPH_WheelSettingsWV_Create();
    // A standard wheel has an inertia of about 0.1 to 0.5
    JPH_WheelSettingsWV_SetInertia(w, kWheelInertiaDefault);
    JPH_WheelSettings_SetSuspensionMinLength(reinterpret_cast<JPH_WheelSettings *>(w),
                                             kWheelSuspensionMinLength);
    JPH_WheelSettings_SetSuspensionMaxLength(reinterpret_cast<JPH_WheelSettings *>(w), susp_max);
    JPH_SpringSettings spring = {JPH_SpringMode_FrequencyAndDamping, freq, damp};
    JPH_WheelSettings_SetSuspensionSpring(reinterpret_cast<JPH_WheelSettings *>(w), &spring);
    // The axis the wheel pivots around for steering
    JPH_Vec3 steering_axis{0, 1.0f, 0};
    JPH_WheelSettings_SetSteeringAxis(reinterpret_cast<JPH_WheelSettings *>(w), &steering_axis);

    // The 'Up' direction for the wheel geometry
    JPH_Vec3 wheel_up{0, 1.0f, 0};
    JPH_WheelSettings_SetWheelUp(reinterpret_cast<JPH_WheelSettings *>(w), &wheel_up);

    // The 'Forward' direction (the way it rolls)
    JPH_Vec3 wheel_forward{0, 0, 1.0f};
    JPH_WheelSettings_SetWheelForward(reinterpret_cast<JPH_WheelSettings *>(w), &wheel_forward);

    // Suspension direction (the way the shock absorber moves) - usually opposite
    // to Up
    JPH_Vec3 suspension_direction{0, -1.0f, 0};
    JPH_WheelSettings_SetSuspensionDirection(reinterpret_cast<JPH_WheelSettings *>(w),
                                             &suspension_direction);
    JPH_WheelSettingsWV_SetMaxBrakeTorque(w, brake);
    if (pos.z > kWheelSteerThreshold) {
        JPH_WheelSettingsWV_SetMaxSteerAngle(w, kWheelMaxSteerAngle);
        JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, 0.0f);
    } else {
        JPH_WheelSettingsWV_SetMaxSteerAngle(w, 0.0f);
        JPH_WheelSettingsWV_SetMaxHandBrakeTorque(w, handbrake);
    }
    JPH_Vec3 wheel_position{static_cast<float>(pos.x), static_cast<float>(pos.y),
                            static_cast<float>(pos.z)};
    JPH_WheelSettings_SetPosition(reinterpret_cast<JPH_WheelSettings *>(w), &wheel_position);
    JPH_WheelSettings_SetRadius(reinterpret_cast<JPH_WheelSettings *>(w), radius);
    JPH_WheelSettings_SetWidth(reinterpret_cast<JPH_WheelSettings *>(w), width);

    JPH_WheelSettingsWV_SetLongitudinalFriction(w, f_curve);
    JPH_WheelSettingsWV_SetLateralFriction(w, f_curve);

    // Steering logic (Simple branch)
    JPH_WheelSettingsWV_SetMaxSteerAngle(w, (pos.z > kWheelSteerThreshold) ? kWheelMaxSteerAngle
                                                                           : 0.0f);

    return reinterpret_cast<JPH_WheelSettings *>(w);
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
        JPH_WheeledVehicleControllerSettings_AddDifferential(v_ctrl, static_cast<int>(i1),
                                                             static_cast<int>(i2));
    }
}

void CPH_CleanupVehicleResources(CPH_VehicleResources *r, uint32_t num_wheels,
                                 CPH_PhysicsWorldObject *self) {
    if (r->j_veh) {
        // If it was already added to Jolt, we MUST remove it before destroying it
        if (static_cast<int>(r->is_added_to_world) && self && self->physics.system) {
            JPH_PhysicsSystem_RemoveStepListener(
                self->physics.system, JPH_VehicleConstraint_AsPhysicsStepListener(r->j_veh));
            JPH_PhysicsSystem_RemoveConstraint(self->physics.system,
                                               reinterpret_cast<JPH_Constraint *>(r->j_veh));
        }
        JPH_Constraint_Destroy(reinterpret_cast<JPH_Constraint *>(r->j_veh));
    }

    if (r->tester) {
        JPH_VehicleCollisionTester_Destroy(
            reinterpret_cast<JPH_VehicleCollisionTester *>(r->tester));
    }
    if (r->v_trans_set) {
        JPH_VehicleTransmissionSettings_Destroy(r->v_trans_set);
    }
    if (r->v_ctrl) {
        JPH_VehicleControllerSettings_Destroy(
            reinterpret_cast<JPH_VehicleControllerSettings *>(r->v_ctrl));
    }

    if (r->w_settings) {
        for (auto i = 0u; i < num_wheels; i++) {
            if (r->w_settings[i]) {
                JPH_WheelSettings_Destroy(r->w_settings[i]);
            }
        }
        CPH_PyFree(r->w_settings);
    }

    if (r->f_curve) {
        JPH_LinearCurve_Destroy(r->f_curve);
    }
    if (r->t_curve) {
        JPH_LinearCurve_Destroy(r->t_curve);
    }
}

int CPH_InitVehicleParsers(CPH_VehicleObject *obj) {
    obj->parsers = CPH::NewParserGroup<CPH_VehicleParsers>();
    if (!obj->parsers) {
        return -1;
    }
    obj->parsers->Init();
    return 0;
}

// --- Sub-helper: Engine Configuration ---
static void setup_engine(JPH_WheeledVehicleControllerSettings *v_ctrl, JPH_LinearCurve *t_curve,
                         PyObject *py_engine) {
    JPH_VehicleEngineSettings eng_set;
    JPH_VehicleEngineSettings_Init(&eng_set);

    // Flat execution: no nesting, no hidden macro branches
    eng_set.maxTorque = CPH_get_py_attr(py_engine, "max_torque", kEngineMaxTorqueDefault);
    eng_set.maxRPM    = CPH_get_py_attr(py_engine, "max_rpm", kEngineMaxRpmDefault);
    eng_set.minRPM    = CPH_get_py_attr(py_engine, "min_rpm", kEngineMinRpmDefault);
    eng_set.inertia   = CPH_get_py_attr(py_engine, "inertia", kEngineInertiaDefault);

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
            t_mode = static_cast<int>(PyLong_AsLong(o_mode));
            Py_DECREF(o_mode);
        }
        PyErr_Clear();
    }

    JPH_VehicleTransmissionSettings_SetMode(v_trans_set, static_cast<JPH_TransmissionMode>(t_mode));
    JPH_VehicleTransmissionSettings_SetClutchStrength(
        v_trans_set,
        CPH_get_py_attr(py_trans, "clutch_strength", kTransmissionClutchStrengthDefault));

    // Extract Gear Ratios from Python list
    if (py_trans && py_trans != Py_None) {
        PyObject *py_ratios = PyObject_GetAttrString(py_trans, "ratios");
        if (py_ratios && PyList_Check(py_ratios)) {
            Py_ssize_t n = PyList_Size(py_ratios);
            float *r = static_cast<float *>(CPH_PyAlloc(static_cast<size_t>(n) * sizeof(float)));
            if (r) {
                for (Py_ssize_t i = 0; i < n; i++) {
                    r[i] = static_cast<float>(PyFloat_AsDouble(PyList_GetItem(py_ratios, i)));
                }
                JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, r,
                                                              static_cast<uint32_t>(n));
                CPH_PyFree(r);
            }
        }
        Py_XDECREF(py_ratios);
        PyErr_Clear();
    } else {
        // DEFAULT GEARS: If no transmission object provided, give it some standard
        // gears Otherwise, the car will have 0 gears and won't move!
        float default_ratios[] = {kTransmissionGearRatio1, kTransmissionGearRatio2,
                                  kTransmissionGearRatio3, kTransmissionGearRatioDirect,
                                  kTransmissionGearRatio4, kTransmissionGearRatio5};
        JPH_VehicleTransmissionSettings_SetGearRatios(v_trans_set, default_ratios,
                                                      kTransmissionDefaultGearCount);
    }

    // Apply Differential Ratio from Python Transmission object
    float diff_ratio   = CPH_get_py_attr(py_trans, "differential_ratio", kDifferentialRatioDefault);
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
static void configure_drivetrain(CPH_VehicleResources *r, PyObject *py_engine, PyObject *py_trans,
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
PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateVehicle(CPH_PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    CPH_ModuleState *st    = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    // --- 1. FAST ARGUMENT PARSING (Unchanged) ---
    uint64_t chassis_h_raw = 0;
    PyObject *py_wheels    = nullptr;
    PyObject *py_drive     = nullptr;
    PyObject *py_engine    = nullptr;
    PyObject *py_trans     = nullptr;
    const char *drive_str  = "RWD";

    if (!self->parsers->CreateVehicle.Parse(args, nargs, kwnames, chassis_h_raw, py_wheels,
                                            py_drive, py_engine, py_trans)) {
        return nullptr;
    }

    if (py_drive && PyUnicode_Check(py_drive)) {
        drive_str = PyUnicode_AsUTF8(py_drive);
    }

    if (!PyList_Check(py_wheels) || PyList_Size(py_wheels) < 2) {
        return PyErr_Format(PyExc_ValueError, "Wheels must be a list of at least 2 dictionaries");
    }
    uint32_t num_wheels = static_cast<uint32_t>(PyList_Size(py_wheels));

    PyThreadState *_save = nullptr;

    // --- 2. RESOLVE CHASSIS (ATOMIC REFACTOR) ---
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    // Ensure all body creation commands are processed before linking
    CPH_SyncAndFlushInternal(self);

    uint32_t slot = 0;
    // TSan Fix: Cast to atomic type for handle verification
    bool handle_valid =
        unpack_handle(&self->physics, static_cast<BodyHandle>(chassis_h_raw), &slot);

    // TSan Fix: Atomic load of state (Acquire ensures visibility of the chassis initialization)
    uint8_t state =
        static_cast<int>(handle_valid)
            ? atomic_load_explicit(&self->physics.slot_states[slot], memory_order_acquire)
            : SLOT_EMPTY;

    if (!handle_valid || state != SLOT_ALIVE) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        return PyErr_Format(PyExc_ValueError, "Invalid chassis handle (Body is dead or pending)");
    }

    JPH_BodyID chassis_bid = self->physics.body_ids[self->physics.slot_to_dense[slot]];
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    // --- 3. PRE-JOLT RESOURCE ALLOCATION (Unchanged) ---
    CPH_VehicleResources r{};
    bool jolt_locked                        = false;
    const JPH_BodyLockInterface *lock_iface = nullptr;
    JPH_BodyLockWrite lock{};
    JPH_VehicleConstraintSettings v_set{};
    CPH_VehicleObject *obj = nullptr;
    JPH_Vec3 tester_up{0, 1.0f, 0};

    r.f_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(r.f_curve, kFrictionCurveX1, kFrictionCurveY1);
    JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, kFrictionCurveY2);

    r.t_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(r.t_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(r.t_curve, 1.0f, 1.0f);

    r.w_settings =
        static_cast<JPH_WheelSettings **>(CPH_PyCalloc(num_wheels, sizeof(JPH_WheelSettings *)));
    r.v_ctrl      = JPH_WheeledVehicleControllerSettings_Create();
    r.v_trans_set = JPH_VehicleTransmissionSettings_Create();

    for (auto i = 0u; i < num_wheels; i++) {
        r.w_settings[i] = create_single_wheel(PyList_GetItem(py_wheels, i), r.f_curve);
        if (!r.w_settings[i]) {
            goto python_fail;
        }
    }

    configure_drivetrain(&r, py_engine, py_trans, drive_str, num_wheels);

    // --- 4. JOLT COMMIT (No GIL - Unchanged) ---
    Py_UNBLOCK_THREADS;

    CPH_NATIVE_MUTEX_LOCK(self->physics.jph_trampoline_lock);
    jolt_locked = true;

    lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->physics.system);
    JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);

    if (CPH_UNLIKELY(!lock.body)) {
        goto jolt_fail;
    }

    JPH_VehicleConstraintSettings_Init(&v_set);
    v_set.wheelsCount = num_wheels;
    v_set.wheels      = r.w_settings;
    v_set.controller  = reinterpret_cast<JPH_VehicleControllerSettings *>(r.v_ctrl);

    r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
    if (!r.j_veh) {
        goto jolt_fail;
    }

    r.tester = JPH_VehicleCollisionTesterRay_Create(kLayerDynamic, &tester_up,
                                                    kVehicleCollisionTesterScale);
    if (!r.tester) {
        goto jolt_fail;
    }

    JPH_VehicleConstraint_SetVehicleCollisionTester(
        r.j_veh, reinterpret_cast<JPH_VehicleCollisionTester *>(r.tester));
    JPH_PhysicsSystem_AddConstraint(self->physics.system,
                                    reinterpret_cast<JPH_Constraint *>(r.j_veh));
    JPH_PhysicsSystem_AddStepListener(self->physics.system,
                                      JPH_VehicleConstraint_AsPhysicsStepListener(r.j_veh));
    r.is_added_to_world = true;

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    CPH_NATIVE_MUTEX_UNLOCK(self->physics.jph_trampoline_lock);
    jolt_locked = false;
    Py_BLOCK_THREADS;

    // --- 5. PYTHON WRAPPER (Unchanged) ---
    obj = PyObject_GC_New(CPH_VehicleObject, reinterpret_cast<PyTypeObject *>(st->VehicleType));

    if (!obj) {
        CPH_SHADOW_LOCK(&self->physics.shadow_lock);
        CPH_CleanupVehicleResources(&r, num_wheels, self);
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        return nullptr;
    }
    if (CPH_InitVehicleParsers(obj) < 0) {
        CPH_SHADOW_LOCK(&self->physics.shadow_lock);
        CPH_CleanupVehicleResources(&r, num_wheels, self);
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        Py_DECREF(obj);
        return nullptr;
    }

    obj->vehicle               = r.j_veh;
    obj->tester                = reinterpret_cast<JPH_VehicleCollisionTester *>(r.tester);
    obj->world                 = reinterpret_cast<CPH_PhysicsWorldObject *>(Py_NewRef(self));
    obj->num_wheels            = num_wheels;
    obj->current_gear          = 0;
    obj->wheel_settings        = r.w_settings;
    obj->controller_settings   = reinterpret_cast<JPH_VehicleControllerSettings *>(r.v_ctrl);
    obj->transmission_settings = r.v_trans_set;
    obj->friction_curve        = r.f_curve;
    obj->torque_curve          = r.t_curve;

    Py_INCREF(self);
    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

jolt_fail:
    if (lock.body) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    }
    if (jolt_locked) {
        CPH_NATIVE_MUTEX_UNLOCK(self->physics.jph_trampoline_lock);
    }
    Py_BLOCK_THREADS;

python_fail:
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    CPH_CleanupVehicleResources(&r, num_wheels, self);
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Jolt vehicle creation failed");
    }
    return nullptr;
}

// --- Vehicles Methods ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethodFromModule CPH_PyVehicle_SetInput(CPH_VehicleObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // 1. FAST PARSE (Unchanged)
    float forward   = 0.0f;
    float right     = 0.0f;
    float brake     = 0.0f;
    float handbrake = 0.0f;

    if (!self->parsers->VehicleInput.Parse(args, nargs, kwnames, forward, right, brake,
                                           handbrake)) {
        return nullptr;
    }

    // 2. STATE MACHINE & JOLT SYNC
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);

    // TSan Fix: Explicit relaxed load of is_stepping flag.
    // Since we hold the SHADOW_LOCK, we only need to verify if the Stepper
    // thread has raised the flag to begin a buffer swap.
    bool stepping = atomic_load_explicit(&self->world->physics.is_stepping, memory_order_relaxed);

    if (CPH_UNLIKELY(stepping || !self->vehicle)) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
        Py_RETURN_NONE;
    }

    auto controller = reinterpret_cast<JPH_WheeledVehicleController *>(
        JPH_VehicleConstraint_GetController(self->vehicle));
    JPH_BodyID chassis_id = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));
    JPH_BodyInterface *bi = self->world->physics.body_interface;

    // Ensure the vehicle is awake to respond to inputs
    JPH_BodyInterface_ActivateBody(bi, chassis_id);

    // Get current physics state for arcade drive logic
    JPH_STACK_ALLOC(JPH_Vec3, linear_vel);
    JPH_STACK_ALLOC(JPH_Quat, chassis_q);
    JPH_BodyInterface_GetLinearVelocity(bi, chassis_id, linear_vel);
    JPH_BodyInterface_GetRotation(bi, chassis_id, chassis_q);

    // Calculate forward speed via dot product
    JPH_Vec3 world_fwd{};
    JPH_Vec3 local_forward{0, 0, 1.0f};
    JPH_Quat_Rotate(chassis_q, &local_forward, &world_fwd);
    float speed = (linear_vel->x * world_fwd.x) + (linear_vel->y * world_fwd.y) +
                  (linear_vel->z * world_fwd.z);

    float input_throttle = fabsf(forward);
    float input_brake    = brake;

    JPH_VehicleTransmission *trans = const_cast<JPH_VehicleTransmission *>(
        JPH_WheeledVehicleController_GetTransmission(controller));
    int cur_gear = JPH_VehicleTransmission_GetCurrentGear(trans);

    // 3. DRIVE LOGIC (Arcade Style State Machine)
    if (forward > kThrottleInputThreshold) {
        JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Auto);
        if (cur_gear <= 0 && speed > -kSpeedDirectionThreshold) {
            JPH_VehicleTransmission_Set(trans, 1, 1.0f);
        }
        if (speed < -kSpeedMinThreshold) {
            input_brake = 1.0f;
        }

    } else if (forward < -kThrottleInputThreshold) {
        JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Manual);
        if (cur_gear != -1 && speed < kSpeedDirectionThreshold) {
            JPH_VehicleTransmission_Set(trans, -1, 1.0f);
        }
        if (speed > kSpeedMinThreshold) {
            input_brake = 1.0f;
        }

    } else {
        input_throttle = 0.0f;
        JPH_VehicleTransmission_SetMode(trans, JPH_TransmissionMode_Manual);
        if (cur_gear != 0) {
            JPH_VehicleTransmission_Set(trans, 0, 0.0f);
        }
        if (fabsf(speed) > kSpeedMinThreshold) {
            input_brake = fmaxf(input_brake, kRollingResistanceCoasting);
        }
    }

    // 4. Final Application to Jolt
    JPH_WheeledVehicleController_SetDriverInput(controller, input_throttle, right, input_brake,
                                                handbrake);

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
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

PyCFunction_DeclareMethodFromModule CPH_PyVehicle_GetWheelTransform(CPH_VehicleObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint32_t index         = 0;

    if (!self->world->parsers->WheelIdx.Parse(args, nargs, kwnames, index)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);

    if (!self->vehicle || index >= self->num_wheels) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
        Py_RETURN_NONE;
    }

    JPH_STACK_ALLOC(JPH_RMat4, transform);
    JPH_Vec3 right = {1.0f, 0.0f, 0.0f};
    JPH_Vec3 up    = {0.0f, 1.0f, 0.0f};
    JPH_VehicleConstraint_GetWheelWorldTransform(self->vehicle, index, &right, &up, transform);

    // Extraction logic remains the same (Correct RMat44 mapping)
    auto px = transform->column3.x;
    auto py = transform->column3.y;
    auto pz = transform->column3.z;

    JPH_STACK_ALLOC(JPH_Mat4, rot_only_mat);
    rot_only_mat->column[0] = transform->column[0];
    rot_only_mat->column[1] = transform->column[1];
    rot_only_mat->column[2] = transform->column[2];
    rot_only_mat->column[3] = JPH_Vec4{0, 0, 0, 1.0f};

    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_Mat4_GetQuaternion(rot_only_mat, q);
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    return pack_transform(px, py, pz, q->x, q->y, q->z, q->w);
}

PyCFunction_DeclareMethodFromModule CPH_PyVehicle_GetWheelLocalTransform(CPH_VehicleObject *self,
                                                                         PyObject *const *args,
                                                                         size_t nargsf,
                                                                         PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint32_t index         = 0;

    if (!self->world->parsers->WheelIdx.Parse(args, nargs, kwnames, index)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);

    if (!self->vehicle || index >= self->num_wheels) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
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
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    return pack_transform(lx, ly, lz, q->x, q->y, q->z, q->w);
}

PyCFunction_DeclareMethodFromModule
CPH_PyVehicle_GetDebugState([[maybe_unused]] CPH_VehicleObject *self,
                            PyObject *ignored [[maybe_unused]]) {
#ifndef CULVERIN_DEBUG
    Py_RETURN_NONE;
#else
    // 1. LOCK AND GUARD
    // We need the world lock to ensure the vehicle pointer is stable
    // and the physics step isn't currently mutating these values.
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);

    if (self->world->physics.is_stepping) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
        // Debug prints usually shouldn't raise Python errors,
        // so we just log a warning and return.
        CPH_DEBUG_LOG("Warning: Cannot get debug state while physics is stepping.");
        Py_RETURN_NONE;
    }

    if (!self->vehicle) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
        CPH_DEBUG_LOG("Warning: Vehicle has been destroyed.");
        Py_RETURN_NONE;
    }

    // 2. RESOLVE JOLT COMPONENTS
    auto controller = reinterpret_cast<JPH_WheeledVehicleController *>(
        JPH_VehicleConstraint_GetController(self->vehicle));
    if (!controller) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
        Py_RETURN_NONE;
    }

    const auto engine = JPH_WheeledVehicleController_GetEngine(controller);
    const auto trans  = JPH_WheeledVehicleController_GetTransmission(controller);

    // 3. CAPTURE INPUTS
    [[maybe_unused]] float in_fwd = JPH_WheeledVehicleController_GetForwardInput(controller);
    [[maybe_unused]] float in_brk = JPH_WheeledVehicleController_GetBrakeInput(controller);

    // 4. CAPTURE DRIVETRAIN
    [[maybe_unused]] float rpm           = JPH_VehicleEngine_GetCurrentRPM(engine);
    [[maybe_unused]] float engine_torque = JPH_VehicleEngine_GetTorque(engine, in_fwd);
    [[maybe_unused]] int gear            = JPH_VehicleTransmission_GetCurrentGear(trans);
    [[maybe_unused]] float clutch        = JPH_VehicleTransmission_GetClutchFriction(trans);

    CPH_DEBUG_LOG("=== VEHICLE DEBUG STATE ===");
    CPH_DEBUG_LOG("  Inputs: Fwd=%.2f | Brk=%.2f", in_fwd, in_brk);
    CPH_DEBUG_LOG("  Engine: %.2f RPM | Torque: %.2f Nm", rpm, engine_torque);
    CPH_DEBUG_LOG("  Trans : Gear %d | Clutch Friction: %.2f", gear, clutch);

    // 5. CAPTURE WHEEL STATE
    for (uint32_t i = 0; i < self->num_wheels; i++) {
        const JPH_Wheel *w          = JPH_VehicleConstraint_GetWheel(self->vehicle, i);
        const JPH_WheelSettings *ws = JPH_Wheel_GetSettings(w);

        [[maybe_unused]] bool contact   = JPH_Wheel_HasContact(w);
        [[maybe_unused]] float susp_len = JPH_Wheel_GetSuspensionLength(w);
        [[maybe_unused]] float ang_vel  = JPH_Wheel_GetAngularVelocity(w);
        [[maybe_unused]] float radius   = JPH_WheelSettings_GetRadius(ws);

        [[maybe_unused]] float tire_speed  = ang_vel * radius;
        [[maybe_unused]] float long_lambda = JPH_Wheel_GetLongitudinalLambda(w);
        [[maybe_unused]] float lat_lambda  = JPH_Wheel_GetLateralLambda(w);

        CPH_DEBUG_LOG("  Wheel %u: %s", i, contact ? "GROUND" : "AIR   ");
        CPH_DEBUG_LOG("    Susp: %.3fm | AngVel: %.2f rad/s | SurfSpd: %.2f m/s", susp_len, ang_vel,
                      tire_speed);
        CPH_DEBUG_LOG("    Trac: Long=%.2f | Lat=%.2f", long_lambda, lat_lambda);
    }
    CPH_DEBUG_LOG("===========================");

    // 6. UNLOCK
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    Py_RETURN_NONE;
#endif
}

// --- Vehicle GC Support ---
PyType_DeclareSlot_StatusFromModule CPH_PyVehicle_Traverse(CPH_VehicleObject *self, visitproc visit,
                                                           void *arg) {
    Py_VISIT(self->world);
    return 0;
}

PyType_DeclareSlot_StatusFromModule CPH_PyVehicle_Clear(CPH_VehicleObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

// --- Internal C Logic (No Python return value to ignore) ---
static void CPH_PyVehicle_InternalCleanup(CPH_VehicleObject *self) {
    if (!self->world) {
        return;
    }

    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);

    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);
    CPH_BLOCK_UNTIL_NOT_QUERYING(self->world);

    if (!self->vehicle) {
        CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
        return;
    }

    // Capture pointers and nullptr the struct members immediately
    JPH_VehicleConstraint *j_veh             = self->vehicle;
    JPH_VehicleCollisionTester *tester       = self->tester;
    JPH_VehicleControllerSettings *v_ctrl    = self->controller_settings;
    JPH_VehicleTransmissionSettings *v_trans = self->transmission_settings;
    JPH_WheelSettings **wheels               = self->wheel_settings;
    JPH_LinearCurve *f_curve                 = self->friction_curve;
    JPH_LinearCurve *t_curve                 = self->torque_curve;
    auto wheel_count                         = self->num_wheels;

    self->vehicle               = nullptr;
    self->tester                = nullptr;
    self->controller_settings   = nullptr;
    self->transmission_settings = nullptr;
    self->wheel_settings        = nullptr;
    self->friction_curve        = nullptr;
    self->torque_curve          = nullptr;

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    // Safely destroy Jolt objects outside the lock
    if (j_veh) {
        JPH_PhysicsStepListener *step_listener = JPH_VehicleConstraint_AsPhysicsStepListener(j_veh);
        JPH_PhysicsSystem_RemoveStepListener(self->world->physics.system, step_listener);
        JPH_PhysicsSystem_RemoveConstraint(self->world->physics.system,
                                           reinterpret_cast<JPH_Constraint *>(j_veh));
        JPH_Constraint_Destroy(reinterpret_cast<JPH_Constraint *>(j_veh));
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
        CPH_PyFree(wheels);
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
PyCFunction_DeclareMethodFromModule CPH_PyVehicle_Destroy(CPH_VehicleObject *self,
                                                          PyObject *ignored [[maybe_unused]]) {
    CPH_PyVehicle_InternalCleanup(self);
    if (self->parsers) {

        delete self->parsers;
    }
    Py_RETURN_NONE;
}

// --- Dealloc Slot ---
PyType_DeclareSlot_VoidFromModule CPH_PyVehicle_Dealloc(CPH_VehicleObject *self) {
    PyObject_GC_UnTrack(self);

    // Call the internal C logic directly
    // This avoids [[nodiscard]] warnings because the return is void
    CPH_PyVehicle_InternalCleanup(self);

    if (self->parsers) {

        delete self->parsers;
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

PyGetSet_DeclareGetter CPH_PyVehicle_GetWheelCount(CPH_VehicleObject *self,
                                                   [[maybe_unused]] void *closure) {
    // num_wheels is set at creation and never changes
    return PyLong_FromUnsignedLong(self->num_wheels);
}

#define CPH_VEH_FASTCALL(fn, name) CPH_FEAT(fn, name, METH_FASTCALL | METH_KEYWORDS)
#define CPH_VEH_NOARGS(fn, name) CPH_FEAT(fn, name, METH_NOARGS)

PyCFunction_DeclareMethodFromModule CPH_PyVehicle_SetTankInput(CPH_VehicleObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

static PyMethodDef kVehicleMethods[] = {

    CPH_VEH_FASTCALL(CPH_PyVehicle_SetInput, set_input),
    CPH_VEH_FASTCALL(CPH_PyVehicle_SetTankInput, set_tank_input),
    CPH_VEH_FASTCALL(CPH_PyVehicle_GetWheelTransform, get_wheel_transform),
    CPH_VEH_FASTCALL(CPH_PyVehicle_GetWheelLocalTransform, get_wheel_local_transform),
    CPH_VEH_NOARGS(CPH_PyVehicle_Destroy, destroy),
    CPH_VEH_NOARGS(CPH_PyVehicle_GetDebugState, get_debug_state),
    {nullptr, nullptr, 0, nullptr}};

PyType_Spec Vehicle_spec = {
    .name      = "culverin._culverin_c.Vehicle",
    .basicsize = sizeof(CPH_VehicleObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(CPH_PyVehicle_Dealloc)},
            {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(CPH_PyVehicle_Traverse)},
            {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(CPH_PyVehicle_Clear)},
            {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kVehicleMethods)},
            {.slot = Py_tp_getset,
             .pfunc =
                 (PyGetSetDef[]){

                     CPH_GETSET("wheel_count", CPH_PyVehicle_GetWheelCount), {}

                 }},
            {},

        },
};