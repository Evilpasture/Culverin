#include "vehicles/TrackedVehicle.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsSync.hpp"
#include <math.h>

namespace CPH::Python {

// --- Tracked Vehicle Constants ---

// Wheel configuration defaults
static constexpr float kTrackedWheelRadiusDefault     = 0.4f;
static constexpr float kTrackedWheelWidthDefault      = 0.2f;
static constexpr float kTrackedWheelSuspensionDefault = 0.5f;

// Suspension parameters
static constexpr float kTrackedSuspensionMinLength = 0.05f;

// Spring properties
static constexpr float kTrackedSpringFreqDefault = 2.0f;
static constexpr float kTrackedSpringDampDefault = 0.5f;

// Transmission gear ratios
static constexpr float kTrackedGearRatio1       = 2.0f;
static constexpr float kTrackedGearRatio2       = 1.4f;
static constexpr float kTrackedGearRatio3       = 1.0f;
static constexpr float kTrackedGearRatio4       = 0.7f;
static constexpr float kTrackedReverseGearRatio = -1.5f;

// Engine parameters (RPM and torque)
static constexpr float kTrackedEngineMaxRpmDefault    = 6000.0f;
static constexpr float kTrackedEngineMinRpmDefault    = 500.0f;
static constexpr float kTrackedEngineMaxTorqueDefault = 5000.0f;

// Collision tester parameters
static constexpr float kTrackedCollisionTesterScale = 2.0f;

// Throttle activation threshold
static constexpr float kTrackedThrottleKickstartThreshold = 0.01f;

// --- Tracked Vehicle Implementation ---

static JPH_WheelSettings *create_track_wheel(PyObject *w_dict) {
    Vec3f pos;
    if (!parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
        return nullptr;
    }

    float radius = CPH::Python::GetAttribute(w_dict, "radius", kTrackedWheelRadiusDefault);
    float width  = CPH::Python::GetAttribute(w_dict, "width", kTrackedWheelWidthDefault);
    float suspension_len =
        CPH::Python::GetAttribute(w_dict, "suspension", kTrackedWheelSuspensionDefault);
    float friction = CPH::Python::GetAttribute(w_dict, "friction", 1.0f);

    // Suspension Spring Properties
    float freq = CPH::Python::GetAttribute(w_dict, "spring_freq", kTrackedSpringFreqDefault);
    float damp = CPH::Python::GetAttribute(w_dict, "spring_damp", kTrackedSpringDampDefault);

    JPH_WheelSettingsTV *w = JPH_WheelSettingsTV_Create();

    JPH_Vec3 wheel_position{pos.x, pos.y, pos.z};
    JPH_WheelSettings_SetPosition(reinterpret_cast<JPH_WheelSettings *>(w), &wheel_position);
    JPH_WheelSettings_SetRadius(reinterpret_cast<JPH_WheelSettings *>(w), radius);
    JPH_WheelSettings_SetWidth(reinterpret_cast<JPH_WheelSettings *>(w), width);

    JPH_WheelSettings_SetSuspensionMinLength(reinterpret_cast<JPH_WheelSettings *>(w),
                                             kTrackedSuspensionMinLength);
    JPH_WheelSettings_SetSuspensionMaxLength(reinterpret_cast<JPH_WheelSettings *>(w),
                                             suspension_len);

    JPH_SpringSettings spring = {JPH_SpringMode_FrequencyAndDamping, freq, damp};
    JPH_WheelSettings_SetSuspensionSpring(reinterpret_cast<JPH_WheelSettings *>(w), &spring);

    JPH_WheelSettingsTV_SetLongitudinalFriction(w, friction);
    JPH_WheelSettingsTV_SetLateralFriction(w, friction);

    return reinterpret_cast<JPH_WheelSettings *>(w);
}

// Helper 1: Setup Engine, Transmission, and Controller settings
static JPH_TrackedVehicleControllerSettings *
init_tracked_controller_settings(TrackedEngineConfig config,
                                 JPH_VehicleTransmissionSettings **out_trans) {

    auto t_ctrl = JPH_TrackedVehicleControllerSettings_Create();

    JPH_VehicleEngineSettings eng;
    JPH_VehicleEngineSettings_Init(&eng);

    eng.maxTorque = config.torque;
    eng.maxRPM    = config.max_rpm;
    eng.minRPM    = config.min_rpm;

    JPH_TrackedVehicleControllerSettings_SetEngine(t_ctrl, &eng);

    auto trans = JPH_VehicleTransmissionSettings_Create();
    JPH_VehicleTransmissionSettings_SetMode(trans, JPH_TransmissionMode_Auto);

    float gears[] = {kTrackedGearRatio1, kTrackedGearRatio2, kTrackedGearRatio3,
                     kTrackedGearRatio4};
    JPH_VehicleTransmissionSettings_SetGearRatios(trans, gears, 4);
    float reverse[] = {kTrackedReverseGearRatio};
    JPH_VehicleTransmissionSettings_SetReverseGearRatios(trans, reverse, 1);

    JPH_TrackedVehicleControllerSettings_SetTransmission(t_ctrl, trans);
    *out_trans = trans;
    return t_ctrl;
}
int InitVehicleParsers(VehicleObject *obj);

extern void CleanupVehicleResources(VehicleResources *r, uint32_t num_wheels,
                                        PhysicsWorldObject *self);

// Orchestrator
[[nodiscard]] PyObject * WorldCreateTrackedVehicle(PhysicsWorldObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    ModuleState *st    = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    // --- 1. FAST ARGUMENT PARSING ---
    uint64_t chassis_h_raw = 0;
    PyObject *py_wheels    = nullptr;
    PyObject *py_tracks    = nullptr;
    float max_torque       = kTrackedEngineMaxTorqueDefault;
    float max_rpm          = kTrackedEngineMaxRpmDefault;
    float min_rpm          = kTrackedEngineMinRpmDefault;

    if (!self->parsers->CreateTracked.Parse(args, nargs, kwnames, chassis_h_raw, py_wheels,
                                            py_tracks, max_torque, max_rpm)) {
        return nullptr;
    }

    // --- 2. RESOLVE CHASSIS (ATOMIC REFACTOR) ---
    (self->physics.shadow_lock).lock();

    // Ensure all pending creation commands are flushed before we try to link a constraint
    SyncAndFlushInternal(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    bool handle_valid =
        unpack_handle(&self->physics, static_cast<BodyHandle>(chassis_h_raw), &slot);

    // TSan Fix: Atomic load of the slot state (Acquire ensures chassis data is visible)
    uint8_t state =
        static_cast<int>(handle_valid)
            ? (self->physics.bodies.slot_states[slot]).load(std::memory_order::acquire)
            : std::to_underlying(SlotState::Empty);

    if (!handle_valid || state != std::to_underlying(SlotState::Alive)) {
        (self->physics.shadow_lock).unlock();
        return PyErr_Format(PyExc_ValueError, "Invalid chassis handle (Body is dead or pending)");
    }

    // Access non-atomic dense buffer while holding shadow_lock
    JPH_BodyID chassis_bid = self->physics.bodies.body_ids[self->physics.bodies.slot_to_dense[slot]];
    (self->physics.shadow_lock).unlock();

    // --- 3. PRE-JOLT RESOURCE ALLOCATION (Unchanged) ---
    VehicleResources r{};
    auto num_wheels = static_cast<uint32_t>(PyList_Size(py_wheels));
    TrackData tracks[2];
    memset(tracks, 0, sizeof(tracks));
    int num_tracks                          = 0;
    bool jolt_locked                        = false;
    PyThreadState *_save                    = nullptr;
    const JPH_BodyLockInterface *lock_iface = nullptr;
    JPH_BodyLockWrite lock{};
    TrackedEngineConfig eng_cfg{};
    JPH_VehicleTransmissionSettings *v_trans     = nullptr;
    JPH_TrackedVehicleControllerSettings *t_ctrl = nullptr;
    JPH_VehicleConstraintSettings v_set{};
    VehicleObject *obj = nullptr;
    JPH_Vec3 tester_up{0, 1.0f, 0};

    r.f_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, 1.0f);

    r.w_settings =
        static_cast<JPH_WheelSettings **>(AllocateZeroed(num_wheels, sizeof(JPH_WheelSettings *)));
    for (uint32_t i = 0; i < num_wheels; i++) {
        r.w_settings[i] = create_track_wheel(PyList_GetItem(py_wheels, i));
        if (!r.w_settings[i]) {
            goto python_fail;
        }
    }

    if (!ParseTracksToC(py_tracks, tracks, &num_tracks)) {
        goto python_fail;
    }

    // --- 4. JOLT COMMIT (No GIL - Unchanged) ---
    Py_UNBLOCK_THREADS;

    (self->physics.jph_trampoline_lock).lock();
    jolt_locked = true;

    lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->physics.system);
    JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);

    if (CPH_UNLIKELY(!lock.body)) {
        goto jolt_fail;
    }

    eng_cfg       = {.torque = max_torque, .max_rpm = max_rpm, .min_rpm = min_rpm};
    t_ctrl        = init_tracked_controller_settings(eng_cfg, &v_trans);
    r.v_ctrl      = reinterpret_cast<JPH_WheeledVehicleControllerSettings *>(t_ctrl);
    r.v_trans_set = v_trans;

    for (int t = 0; t < num_tracks; t++) {
        JPH_VehicleTrackSettings track_set;
        JPH_VehicleTrackSettings_Init(&track_set);
        track_set.wheels      = tracks[t].indices;
        track_set.wheelsCount = tracks[t].count;
        track_set.drivenWheel = tracks[t].driven_idx;
        JPH_TrackedVehicleControllerSettings_SetTrack(t_ctrl, static_cast<uint32_t>(t), &track_set);
    }

    JPH_VehicleConstraintSettings_Init(&v_set);
    v_set.wheelsCount = num_wheels;
    v_set.wheels      = r.w_settings;
    v_set.controller  = reinterpret_cast<JPH_VehicleControllerSettings *>(t_ctrl);

    r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
    if (!r.j_veh) {
        goto jolt_fail;
    }

    r.tester = JPH_VehicleCollisionTesterRay_Create(kTrackedLayerDrivable, &tester_up,
                                                    kTrackedCollisionTesterScale);
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
    (self->physics.jph_trampoline_lock).unlock();
    jolt_locked = false;
    Py_BLOCK_THREADS;

    // --- 5. CLEANUP & WRAP (Unchanged) ---
    for (int t = 0; t < num_tracks; t++) {
        Deallocate(tracks[t].indices);
    }

    obj = PyObject_GC_New(VehicleObject, reinterpret_cast<PyTypeObject *>(st->VehicleType));
    if (!obj) {
        (self->physics.shadow_lock).lock();
        CleanupVehicleResources(&r, num_wheels, self);
        (self->physics.shadow_lock).unlock();
        return nullptr;
    }

    if (InitVehicleParsers(obj) < 0) {
        (self->physics.shadow_lock).lock();
        CleanupVehicleResources(&r, num_wheels, self);
        (self->physics.shadow_lock).unlock();
        Py_DECREF(obj);
        return nullptr;
    }

    obj->vehicle               = r.j_veh;
    obj->tester                = reinterpret_cast<JPH_VehicleCollisionTester *>(r.tester);
    obj->world                 = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->num_wheels            = num_wheels;
    obj->wheel_settings        = r.w_settings;
    obj->controller_settings   = reinterpret_cast<JPH_VehicleControllerSettings *>(t_ctrl);
    obj->transmission_settings = r.v_trans_set;
    obj->friction_curve        = r.f_curve;
    obj->torque_curve          = nullptr;

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

jolt_fail:
    if (lock.body) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    }
    if (jolt_locked) {
        (self->physics.jph_trampoline_lock).unlock();
    }
    Py_BLOCK_THREADS;

python_fail:
    for (int t = 0; t < num_tracks; t++) {
        if (tracks[t].indices) {
            Deallocate(tracks[t].indices);
        }
    }
    (self->physics.shadow_lock).lock();
    CleanupVehicleResources(&r, num_wheels, self);
    (self->physics.shadow_lock).unlock();
    return PyErr_Format(PyExc_RuntimeError, "Tracked vehicle creation failed");
}

// Helper: Set Tank Input
[[nodiscard]] PyObject * VehicleSetTankInput(VehicleObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // 1. FAST PARSE (Zero-Allocation)
    float left  = 0.0f;
    float right = 0.0f;
    float brake = 0.0f;

    if (!self->parsers->TankInput.Parse(args, nargs, kwnames, left, right, brake)) {
        return nullptr;
    }

    // Safety check outside lock
    if (CPH_UNLIKELY(!self->vehicle || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. STATE SNAPSHOT & JOLT ACTIVATION
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    auto t_ctrl = reinterpret_cast<JPH_TrackedVehicleController *>(
        JPH_VehicleConstraint_GetController(self->vehicle));
    JPH_BodyID bid = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));

    // Wake up the tank to process inputs
    JPH_BodyInterface_ActivateBody(self->world->physics.body_interface, bid);

    auto trans =
        const_cast<JPH_VehicleTransmission *>(JPH_TrackedVehicleController_GetTransmission(t_ctrl));
    int gear = JPH_VehicleTransmission_GetCurrentGear(trans);

    // 3. TANK DRIVE LOGIC
    // Throttle for a tracked vehicle is typically the max absolute power requested from either side
    float throttle = fmaxf(fabsf(left), fabsf(right));

    // Simple Kickstart Logic: Auto-shift from Neutral to Gear 1 when throttle is applied
    if (throttle > kTrackedThrottleKickstartThreshold) {
        if (gear == 0) {
            JPH_VehicleTransmission_Set(trans, 1, 1.0f);
        }
    } else {
        // Force Neutral when no input is detected to prevent "crawling"
        if (gear != 0) {
            JPH_VehicleTransmission_Set(trans, 0, 0.0f);
        }
    }

    // 4. Final Application
    JPH_TrackedVehicleController_SetDriverInput(t_ctrl, throttle, left, right, brake);

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

} // namespace CPH::Python
