#include "vehicles/TrackedVehicle.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsSync.hpp"
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Vehicle/TrackedVehicleController.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Vehicle/VehicleTrack.h>
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

static JPH::Ref<JPH::WheelSettingsTV> CreateTrackWheel(PyObject *w_dict) {
    JPH::Float3 pos;
    if (ParseResult parsed = ParseVec3Into(PyDict_GetItemString(w_dict, "pos"), pos);
        !parsed) {
        RaiseParseError(parsed.error());
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

    JPH::Ref<JPH::WheelSettingsTV> w = new JPH::WheelSettingsTV;

    w->mPosition            = JPH::Vec3(pos.x, pos.y, pos.z);
    w->mRadius              = radius;
    w->mWidth               = width;
    w->mSuspensionMinLength = kTrackedSuspensionMinLength;
    w->mSuspensionMaxLength = suspension_len;
    w->mSuspensionSpring =
        JPH::SpringSettings(JPH::ESpringMode::FrequencyAndDamping, freq, damp);

    w->mLongitudinalFriction = friction;
    w->mLateralFriction      = friction;

    return w;
}

// Helper 1: Setup Engine, Transmission, and Controller settings
static JPH::Ref<JPH::TrackedVehicleControllerSettings>
InitTrackedControllerSettings(TrackedEngineConfig config) {

    JPH::Ref<JPH::TrackedVehicleControllerSettings> t_ctrl =
        new JPH::TrackedVehicleControllerSettings;

    t_ctrl->mEngine.mMaxTorque = config.torque;
    t_ctrl->mEngine.mMaxRPM    = config.max_rpm;
    t_ctrl->mEngine.mMinRPM    = config.min_rpm;

    // VehicleTransmissionSettings is a member of the controller settings, so this
    // is written straight through -- unlike the wheeled path, a tracked vehicle's
    // transmission really does reach the controller.
    t_ctrl->mTransmission.mMode = JPH::ETransmissionMode::Auto;

    const float gears[] = {kTrackedGearRatio1, kTrackedGearRatio2, kTrackedGearRatio3,
                           kTrackedGearRatio4};
    t_ctrl->mTransmission.mGearRatios.assign(gears, gears + 4);
    const float reverse[] = {kTrackedReverseGearRatio};
    t_ctrl->mTransmission.mReverseGearRatios.assign(reverse, reverse + 1);

    return t_ctrl;
}
int InitVehicleParsers(VehicleObject *obj);

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

    const ParseResult parsed =
        self->parsers->CreateTracked.Parse(args, nargs, kwnames, chassis_h_raw, py_wheels,
                                            py_tracks, max_torque, max_rpm);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // --- 2. RESOLVE CHASSIS (ATOMIC REFACTOR) ---
    (self->physics.shadow_lock).lock();

    // Ensure all pending creation commands are flushed before we try to link a constraint
    SyncAndFlushInternal(self);

    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    const std::optional<std::uint32_t> resolved =
        self->physics.Resolve(static_cast<BodyHandle>(chassis_h_raw));
    const uint32_t slot        = resolved.value_or(0);
    const bool handle_valid    = resolved.has_value();

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
    JPH::BodyID chassis_bid = self->physics.bodies.body_ids[self->physics.bodies.slot_to_dense[slot]];
    (self->physics.shadow_lock).unlock();

    // --- 3. PRE-JOLT RESOURCE ALLOCATION ---
    // All RefCounted, and declared before any goto: crossing the initialization
    // of a Ref<> on a forward jump is ill-formed.
    auto num_wheels = static_cast<uint32_t>(PyList_Size(py_wheels));
    TrackData tracks[2];
    memset(tracks, 0, sizeof(tracks));
    int num_tracks       = 0;
    PyThreadState *_save = nullptr;
    TrackedEngineConfig eng_cfg{};
    JPH::Ref<JPH::TrackedVehicleControllerSettings> t_ctrl;
    JPH::VehicleConstraintSettings v_set;
    JPH::Ref<JPH::VehicleConstraint> j_veh;
    JPH::Ref<JPH::VehicleCollisionTester> tester;
    VehicleObject *obj = nullptr;

    // WheelSettingsTV carries friction as a plain float, so there is no
    // LinearCurve to build here.
    v_set.mWheels.reserve(num_wheels);
    for (uint32_t i = 0; i < num_wheels; i++) {
        JPH::Ref<JPH::WheelSettingsTV> wheel = CreateTrackWheel(PyList_GetItem(py_wheels, i));
        if (!wheel) {
            goto python_fail;
        }
        v_set.mWheels.push_back(wheel.GetPtr());
    }

    {
        const std::expected<ParsedTracks, ParseError> parsed_tracks =
            ParseTracksToC(py_tracks, tracks);
        if (!parsed_tracks) {
            RaiseParseError(parsed_tracks.error());
            goto python_fail;
        }
        num_tracks = parsed_tracks->count;
    }

    // --- 4. JOLT COMMIT (No GIL) ---
    Py_UNBLOCK_THREADS;

    (self->physics.jph_trampoline_lock).lock();
    {
        const JPH::BodyLockInterface &lock_iface =
            self->physics.physics_system.GetBodyLockInterface();
        const JPH::BodyLockWrite lock(lock_iface, chassis_bid);

        if (!lock.Succeeded()) [[unlikely]] {
            (self->physics.jph_trampoline_lock).unlock();
            Py_BLOCK_THREADS;
            goto python_fail;
        }

        eng_cfg = {.torque = max_torque, .max_rpm = max_rpm, .min_rpm = min_rpm};
        t_ctrl  = InitTrackedControllerSettings(eng_cfg);

        // VehicleTrackSettings::mWheels is an Array, so the parsed index buffers
        // are copied in and the temporaries can go straight back.
        for (int t = 0; t < num_tracks; t++) {
            JPH::VehicleTrackSettings &track_set = t_ctrl->mTracks[static_cast<size_t>(t)];
            track_set.mWheels.assign(tracks[t].indices, tracks[t].indices + tracks[t].count);
            track_set.mDrivenWheel = tracks[t].driven_idx;
        }

        v_set.mController = t_ctrl;
        j_veh             = new JPH::VehicleConstraint(lock.GetBody(), v_set);

        tester = new JPH::VehicleCollisionTesterRay(
            kTrackedLayerDrivable, JPH::Vec3(0, 1.0f, 0), kTrackedCollisionTesterScale);

        j_veh->SetVehicleCollisionTester(tester);
        self->physics.physics_system.AddConstraint(j_veh);
        // VehicleConstraint derives from PhysicsStepListener, so no
        // AsPhysicsStepListener() trampoline is needed.
        self->physics.physics_system.AddStepListener(j_veh);
    }
    (self->physics.jph_trampoline_lock).unlock();
    Py_BLOCK_THREADS;

    // --- 5. CLEANUP & WRAP ---
    for (int t = 0; t < num_tracks; t++) {
        Deallocate(tracks[t].indices);
    }
    num_tracks = 0;

    obj = PyObject_GC_New(VehicleObject, reinterpret_cast<PyTypeObject *>(st->VehicleType));
    if (!obj) {
        goto jolt_created_fail;
    }

    if (InitVehicleParsers(obj) < 0) {
        Py_DECREF(obj);
        goto jolt_created_fail;
    }

    obj->vehicle    = j_veh.GetPtr();
    obj->tester     = tester.GetPtr();
    obj->world      = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->num_wheels = num_wheels;

    // Hand the creation references over to the wrapper; it releases them.
    j_veh  = nullptr;
    tester = nullptr;

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

jolt_created_fail:
    (self->physics.shadow_lock).lock();
    self->physics.physics_system.RemoveStepListener(j_veh);
    self->physics.physics_system.RemoveConstraint(j_veh);
    (self->physics.shadow_lock).unlock();

python_fail:
    for (int t = 0; t < num_tracks; t++) {
        Deallocate(tracks[t].indices);
    }
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

    const ParseResult parsed =
        self->parsers->TankInput.Parse(args, nargs, kwnames, left, right, brake);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // Safety check outside lock
    if (!self->vehicle || !self->world) [[unlikely]] {
        Py_RETURN_NONE;
    }

    // 2. STATE SNAPSHOT & JOLT ACTIVATION
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    auto *t_ctrl = static_cast<JPH::TrackedVehicleController *>(self->vehicle->GetController());
    const JPH::BodyID bid = self->vehicle->GetVehicleBody()->GetID();

    // Wake up the tank to process inputs
    self->world->physics.physics_system.GetBodyInterface().ActivateBody(bid);

    // GetController() has a non-const overload, so no const_cast is needed.
    JPH::VehicleTransmission &trans = t_ctrl->GetTransmission();
    const int gear                  = trans.GetCurrentGear();

    // 3. TANK DRIVE LOGIC
    // Throttle for a tracked vehicle is typically the max absolute power requested from either side
    float throttle = fmaxf(fabsf(left), fabsf(right));

    // Simple Kickstart Logic: Auto-shift from Neutral to Gear 1 when throttle is applied
    if (throttle > kTrackedThrottleKickstartThreshold) {
        if (gear == 0) {
            trans.Set(1, 1.0f);
        }
    } else {
        // Force Neutral when no input is detected to prevent "crawling"
        if (gear != 0) {
            trans.Set(0, 0.0f);
        }
    }

    // 4. Final Application
    // TODO(smell): TrackedVehicleController::SetDriverInput asserts
    // inLeftRatio != 0 && inRightRatio != 0, but set_tank_input(0, 0, 0) is a
    // perfectly reasonable Python call and reaches this line. Release builds
    // compile the assert out, so it has never fired; preserved as-is rather than
    // silently clamping.
    t_ctrl->SetDriverInput(throttle, left, right, brake);

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

} // namespace CPH::Python
