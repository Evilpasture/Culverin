#include "vehicles/Vehicle.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Compiler.hpp"
#include "physics/PhysicsSync.hpp"
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/Vehicle/VehicleCollisionTester.h>
#include <Jolt/Physics/Vehicle/VehicleDifferential.h>
#include <Jolt/Physics/Vehicle/WheeledVehicleController.h>
#include <math.h>

namespace CPH::Python {

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

// Engine defaults
static constexpr float kEngineMaxTorqueDefault = 500.0f;
static constexpr float kEngineMaxRpmDefault    = 7000.0f;
static constexpr float kEngineMinRpmDefault    = 1000.0f;
static constexpr float kEngineInertiaDefault   = 0.5f;

// Transmission defaults
static constexpr float kTransmissionClutchStrengthDefault = 2000.0f;
static constexpr float kTransmissionGearRatio1            = 2.66f;
static constexpr float kTransmissionGearRatio2            = 1.78f;
static constexpr float kTransmissionGearRatio3            = 1.30f;
static constexpr float kTransmissionGearRatioDirect       = 1.00f;
static constexpr float kTransmissionGearRatio4            = 0.74f;
static constexpr float kTransmissionGearRatio5            = 0.50f;
static constexpr uint32_t kTransmissionDefaultGearCount   = 6;
static constexpr float kDifferentialRatioDefault          = 3.42f;

// Friction curve
static constexpr float kFrictionCurveX1 = 0.1f;
static constexpr float kFrictionCurveY1 = 2.0f;
static constexpr float kFrictionCurveY2 = 1.2f;

// TODO(smell): the name says "scale" but this value is passed to
// VehicleCollisionTesterRay as its maximum slope angle in radians, i.e.
// ~114.6 degrees -- past vertical, so the tester accepts essentially any
// surface normal. Preserved as-is; rename and pick a real angle separately.
static constexpr float kVehicleCollisionTesterScale = 2.0f;
static constexpr float kThrottleInputThreshold      = 0.01f;
static constexpr float kSpeedDirectionThreshold     = 0.5f;
static constexpr float kSpeedMinThreshold           = 0.1f;
static constexpr float kRollingResistanceCoasting   = 0.05f;

static JPH::Ref<JPH::WheelSettingsWV> create_single_wheel(PyObject *w_dict,
                                                         const JPH::LinearCurve &f_curve) {
    PosStride pos;

    // 1. Parse Position using helper
    if (ParseResult parsed =
            parse_vec3_into(PyDict_GetItemString(w_dict, "pos"), pos);
        !parsed) {
        // Deliberate message override: the wheel dict is user data, and this
        // diagnostic is what callers see for a malformed wheel entry.
        PyErr_SetString(PyExc_ValueError, "Wheel 'pos' must be a sequence of 3 real numbers");
        return nullptr;
    }

    // 2. Parse Float Attributes using consistent helper
    float radius = CPH::Python::GetAttribute(w_dict, "radius", kWheelRadiusDefault);
    float width  = CPH::Python::GetAttribute(w_dict, "width", kWheelWidthDefault);
    float brake  = CPH::Python::GetAttribute(w_dict, "brake_torque", kWheelBrakeTorqueDefault);
    float handbrake =
        CPH::Python::GetAttribute(w_dict, "handbrake_torque", kWheelHandbrakeTorqueDefault);
    float susp_max = CPH::Python::GetAttribute(w_dict, "suspension", kWheelSuspensionMaxDefault);
    float freq     = CPH::Python::GetAttribute(w_dict, "spring_freq", kWheelSpringFreqDefault);
    float damp     = CPH::Python::GetAttribute(w_dict, "spring_damp", kWheelSpringDampDefault);

    // 3. Jolt Object Setup
    JPH::Ref<JPH::WheelSettingsWV> w = new JPH::WheelSettingsWV;
    // A standard wheel has an inertia of about 0.1 to 0.5
    w->mInertia             = kWheelInertiaDefault;
    w->mSuspensionMinLength = kWheelSuspensionMinLength;
    w->mSuspensionMaxLength = susp_max;
    w->mSuspensionSpring    = JPH::SpringSettings(JPH::ESpringMode::FrequencyAndDamping, freq, damp);

    // The axis the wheel pivots around for steering
    w->mSteeringAxis = JPH::Vec3(0, 1.0f, 0);

    // The 'Up' direction for the wheel geometry
    w->mWheelUp = JPH::Vec3(0, 1.0f, 0);

    // The 'Forward' direction (the way it rolls)
    w->mWheelForward = JPH::Vec3(0, 0, 1.0f);

    // Suspension direction (the way the shock absorber moves) - usually opposite
    // to Up
    w->mSuspensionDirection = JPH::Vec3(0, -1.0f, 0);
    w->mMaxBrakeTorque      = brake;

    // TODO(smell): "does this wheel steer" is inferred from the sign of the
    // wheel's local Z position, so a vehicle laid out along -Z gets no steering
    // wheels at all. Preserved as-is; the wheel dict should say so explicitly.
    if (pos.z > kWheelSteerThreshold) {
        w->mMaxSteerAngle      = kWheelMaxSteerAngle;
        w->mMaxHandBrakeTorque = 0.0f;
    } else {
        w->mMaxSteerAngle      = 0.0f;
        w->mMaxHandBrakeTorque = handbrake;
    }

    w->mPosition = JPH::Vec3(static_cast<float>(pos.x), static_cast<float>(pos.y),
                             static_cast<float>(pos.z));
    w->mRadius   = radius;
    w->mWidth    = width;

    // LinearCurve is a value type, so each wheel gets its own copy; the caller's
    // curve is only a template.
    w->mLongitudinalFriction = f_curve;
    w->mLateralFriction      = f_curve;

    // TODO(smell): this rewrites mMaxSteerAngle a second time with the same
    // value the branch above already chose, but it does NOT re-touch
    // mMaxHandBrakeTorque. Harmless today, kept so the diff stays reviewable.
    // Steering logic (Simple branch)
    w->mMaxSteerAngle = (pos.z > kWheelSteerThreshold) ? kWheelMaxSteerAngle : 0.0f;

    return w;
}

// --- Internal Helpers for Vehicle Construction ---

static void setup_vehicle_differentials(JPH::WheeledVehicleControllerSettings &v_ctrl,
                                        const char *drive_str, uint32_t num_wheels) {
    if (strcmp(drive_str, "FWD") == 0) {
        v_ctrl.mDifferentials.push_back(JPH::VehicleDifferentialSettings(0, 1));
    } else if (strcmp(drive_str, "AWD") == 0 && num_wheels >= 4) {
        v_ctrl.mDifferentials.push_back(JPH::VehicleDifferentialSettings(0, 1));
        v_ctrl.mDifferentials.push_back(JPH::VehicleDifferentialSettings(2, 3));
    } else { // RWD
        uint32_t i1 = (num_wheels >= 4) ? 2 : 0;
        uint32_t i2 = (num_wheels >= 4) ? 3 : 1;
        v_ctrl.mDifferentials.push_back(
            JPH::VehicleDifferentialSettings(static_cast<int>(i1), static_cast<int>(i2)));
    }
}

int InitVehicleParsers(VehicleObject *obj) {
    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::VehicleParsers>();
    if (!obj->parsers) {
        return -1;
    }
    return 0;
}

// --- Sub-helper: Engine Configuration ---
static void setup_engine(JPH::WheeledVehicleControllerSettings &v_ctrl,
                         const JPH::LinearCurve &t_curve, PyObject *py_engine) {
    JPH::VehicleEngineSettings eng_set;

    // Flat execution: no nesting, no hidden macro branches
    eng_set.mMaxTorque = CPH::Python::GetAttribute(py_engine, "max_torque", kEngineMaxTorqueDefault);
    eng_set.mMaxRPM    = CPH::Python::GetAttribute(py_engine, "max_rpm", kEngineMaxRpmDefault);
    eng_set.mMinRPM    = CPH::Python::GetAttribute(py_engine, "min_rpm", kEngineMinRpmDefault);
    eng_set.mInertia   = CPH::Python::GetAttribute(py_engine, "inertia", kEngineInertiaDefault);

    eng_set.mNormalizedTorque = t_curve;

    v_ctrl.mEngine = eng_set;
}

// --- Sub-helper: Transmission Configuration ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void setup_transmission(JPH::WheeledVehicleControllerSettings &v_ctrl,
                               JPH::VehicleTransmissionSettings &v_trans_set, PyObject *py_trans) {
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

    v_trans_set.mMode = static_cast<JPH::ETransmissionMode>(t_mode);
    v_trans_set.mClutchStrength =
        CPH::Python::GetAttribute(py_trans, "clutch_strength", kTransmissionClutchStrengthDefault);

    // Extract Gear Ratios from Python list
    if (py_trans && py_trans != Py_None) {
        PyObject *py_ratios = PyObject_GetAttrString(py_trans, "ratios");
        if (py_ratios && PyList_Check(py_ratios)) {
            Py_ssize_t n = PyList_Size(py_ratios);
            v_trans_set.mGearRatios.clear();
            v_trans_set.mGearRatios.reserve(static_cast<JPH::Array<float>::size_type>(n));
            for (Py_ssize_t i = 0; i < n; i++) {
                v_trans_set.mGearRatios.push_back(
                    static_cast<float>(PyFloat_AsDouble(PyList_GetItem(py_ratios, i))));
            }
        }
        Py_XDECREF(py_ratios);
        PyErr_Clear();
    } else {
        // DEFAULT GEARS: If no transmission object provided, give it some standard
        // gears Otherwise, the car will have 0 gears and won't move!
        const float default_ratios[] = {kTransmissionGearRatio1, kTransmissionGearRatio2,
                                        kTransmissionGearRatio3, kTransmissionGearRatioDirect,
                                        kTransmissionGearRatio4, kTransmissionGearRatio5};
        v_trans_set.mGearRatios.assign(default_ratios,
                                       default_ratios + kTransmissionDefaultGearCount);
    }

    // Apply Differential Ratio from Python Transmission object
    float diff_ratio =
        CPH::Python::GetAttribute(py_trans, "differential_ratio", kDifferentialRatioDefault);
    for (JPH::VehicleDifferentialSettings &ds : v_ctrl.mDifferentials) {
        ds.mDifferentialRatio = diff_ratio;
    }
}

// --- Main coordinate function (Complexity: 1) ---
static void configure_drivetrain(JPH::WheeledVehicleControllerSettings &v_ctrl,
                                 JPH::VehicleTransmissionSettings &v_trans_set,
                                 const JPH::LinearCurve &t_curve, PyObject *py_engine,
                                 PyObject *py_trans, const char *drive_str, uint32_t num_wheels) {
    // 1. Setup Diffs FIRST so they exist when we want to set ratios
    setup_vehicle_differentials(v_ctrl, drive_str, num_wheels);

    // 2. Setup Engine
    setup_engine(v_ctrl, t_curve, py_engine);

    // 3. Setup Transmission (Now can find the diffs to apply the ratio)
    setup_transmission(v_ctrl, v_trans_set, py_trans);
}

// --- Main Function ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] PyObject * WorldCreateVehicle(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    ModuleState *st    = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    // --- 1. FAST ARGUMENT PARSING (Unchanged) ---
    uint64_t chassis_h_raw = 0;
    PyObject *py_wheels    = nullptr;
    PyObject *py_drive     = nullptr;
    PyObject *py_engine    = nullptr;
    PyObject *py_trans     = nullptr;
    const char *drive_str  = "RWD";

    const ParseResult parsed =
        self->parsers->CreateVehicle.Parse(args, nargs, kwnames, chassis_h_raw, py_wheels,
                                            py_drive, py_engine, py_trans);
    if (!parsed) {
        RaiseParseError(parsed.error());
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
    (self->physics.shadow_lock).lock();

    // Ensure all body creation commands are processed before linking
    SyncAndFlushInternal(self);

    // TSan Fix: Cast to atomic type for handle verification
    const std::optional<std::uint32_t> resolved =
        self->physics.UnpackHandle(static_cast<BodyHandle>(chassis_h_raw));
    const uint32_t slot        = resolved.value_or(0);
    const bool handle_valid    = resolved.has_value();

    // TSan Fix: Atomic load of state (Acquire ensures visibility of the chassis initialization)
    uint8_t state =
        static_cast<int>(handle_valid)
            ? (self->physics.bodies.slot_states[slot]).load(std::memory_order::acquire)
            : std::to_underlying(SlotState::Empty);

    if (!handle_valid || state != std::to_underlying(SlotState::Alive)) {
        (self->physics.shadow_lock).unlock();
        return PyErr_Format(PyExc_ValueError, "Invalid chassis handle (Body is dead or pending)");
    }

    JPH::BodyID chassis_bid = self->physics.bodies.body_ids[self->physics.bodies.slot_to_dense[slot]];
    (self->physics.shadow_lock).unlock();

    // --- 3. PRE-JOLT RESOURCE ALLOCATION ---
    // Everything below is RefCounted and declared before any goto: crossing the
    // initialization of a Ref<> on a forward jump is ill-formed, and the failure
    // paths jump past this point.
    JPH::LinearCurve friction_curve;
    friction_curve.AddPoint(0.0f, 1.0f);
    friction_curve.AddPoint(kFrictionCurveX1, kFrictionCurveY1);
    friction_curve.AddPoint(1.0f, kFrictionCurveY2);

    JPH::LinearCurve torque_curve;
    torque_curve.AddPoint(0.0f, 1.0f);
    torque_curve.AddPoint(1.0f, 1.0f);

    JPH::Ref<JPH::WheeledVehicleControllerSettings> v_ctrl = new JPH::WheeledVehicleControllerSettings;
    JPH::VehicleTransmissionSettings v_trans_set;
    JPH::VehicleConstraintSettings v_set;
    JPH::Ref<JPH::VehicleConstraint> j_veh;
    JPH::Ref<JPH::VehicleCollisionTester> tester;
    VehicleObject *obj = nullptr;

    v_set.mWheels.reserve(num_wheels);
    for (auto i = 0u; i < num_wheels; i++) {
        JPH::Ref<JPH::WheelSettingsWV> wheel =
            create_single_wheel(PyList_GetItem(py_wheels, i), friction_curve);
        if (!wheel) {
            goto python_fail;
        }
        v_set.mWheels.push_back(wheel.GetPtr());
    }

    configure_drivetrain(*v_ctrl, v_trans_set, torque_curve, py_engine, py_trans, drive_str,
                         num_wheels);

    // TODO(bug): v_trans_set is configured from the caller's transmission object
    // above and then never copied into v_ctrl->mTransmission -- the old code had
    // the same shape (it built a standalone JPH_VehicleTransmissionSettings and
    // only ever freed it), so create_vehicle(transmission=...) has never affected
    // a wheeled vehicle. The controller therefore runs on Jolt's defaults: Auto
    // mode and gear ratios {2.66, 1.78, 1.3, 1.0, 0.74}, not the six ratios built
    // here. Deliberately preserved so legacy tests keep passing; wiring it up is
    // `v_ctrl->mTransmission = v_trans_set;` and is a behaviour change.
    (void)v_trans_set;

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

        v_set.mController = v_ctrl;
        j_veh             = new JPH::VehicleConstraint(lock.GetBody(), v_set);

        tester = new JPH::VehicleCollisionTesterRay(
            kLayerDynamic, JPH::Vec3(0, 1.0f, 0), kVehicleCollisionTesterScale);

        j_veh->SetVehicleCollisionTester(tester);
        self->physics.physics_system.AddConstraint(j_veh);
        // VehicleConstraint derives from PhysicsStepListener, so no
        // AsPhysicsStepListener() trampoline is needed.
        self->physics.physics_system.AddStepListener(j_veh);
    }
    (self->physics.jph_trampoline_lock).unlock();
    Py_BLOCK_THREADS;

    // --- 5. PYTHON WRAPPER ---
    obj = PyObject_GC_New(VehicleObject, reinterpret_cast<PyTypeObject *>(st->VehicleType));

    if (!obj) {
        goto jolt_created_fail;
    }
    if (InitVehicleParsers(obj) < 0) {
        Py_DECREF(obj);
        goto jolt_created_fail;
    }

    obj->vehicle      = j_veh.GetPtr();
    obj->tester       = tester.GetPtr();
    obj->world        = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->num_wheels   = num_wheels;
    obj->current_gear = 0;

    // Hand the creation references over to the wrapper; it releases them.
    j_veh   = nullptr;
    tester  = nullptr;

    // TODO(bug): obj->world already took a strong reference via Py_NewRef above,
    // and VehicleDealloc releases exactly one, so this extra INCREF leaks a world
    // reference per vehicle and keeps the world alive for as long as any vehicle
    // object exists. WorldCreateTrackedVehicle does not do this. Preserved as-is;
    // dropping the line is the fix and is a lifecycle change.
    Py_INCREF(self);
    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

jolt_created_fail:
    // The constraint made it into the physics system, so it has to come back out
    // before its references drop.
    (self->physics.shadow_lock).lock();
    self->physics.physics_system.RemoveStepListener(j_veh);
    self->physics.physics_system.RemoveConstraint(j_veh);
    (self->physics.shadow_lock).unlock();

python_fail:
    if (!PyErr_Occurred()) {
        PyErr_SetString(PyExc_RuntimeError, "Jolt vehicle creation failed");
    }
    return nullptr;
}

// --- Vehicles Methods ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] PyObject * VehicleSetInput(VehicleObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // 1. FAST PARSE (Unchanged)
    float forward   = 0.0f;
    float right     = 0.0f;
    float brake     = 0.0f;
    float handbrake = 0.0f;

    const ParseResult parsed =
        self->parsers->VehicleInput.Parse(args, nargs, kwnames, forward, right, brake,
                                           handbrake);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 2. STATE MACHINE & JOLT SYNC
    (self->world->physics.shadow_lock).lock();

    // TSan Fix: Explicit relaxed load of is_stepping flag.
    // Since we hold the SHADOW_LOCK, we only need to verify if the Stepper
    // thread has raised the flag to begin a buffer swap.
    bool stepping = (self->world->physics.is_stepping).load(std::memory_order::relaxed);

    if (stepping || !self->vehicle) [[unlikely]] {
        (self->world->physics.shadow_lock).unlock();
        Py_RETURN_NONE;
    }

    auto *controller = static_cast<JPH::WheeledVehicleController *>(self->vehicle->GetController());
    const JPH::BodyID chassis_id  = self->vehicle->GetVehicleBody()->GetID();
    JPH::BodyInterface &bi        = self->world->physics.physics_system.GetBodyInterface();

    // Ensure the vehicle is awake to respond to inputs
    bi.ActivateBody(chassis_id);

    // Get current physics state for arcade drive logic
    const JPH::Vec3 linear_vel = bi.GetLinearVelocity(chassis_id);
    const JPH::Quat chassis_q  = bi.GetRotation(chassis_id);

    // Calculate forward speed via dot product
    const JPH::Vec3 world_fwd = chassis_q * JPH::Vec3(0, 0, 1.0f);
    const float speed         = linear_vel.Dot(world_fwd);

    float input_throttle = fabsf(forward);
    float input_brake    = brake;

    // GetController() has a non-const overload, so the transmission is mutable
    // without a const_cast.
    JPH::VehicleTransmission &trans = controller->GetTransmission();
    const int cur_gear              = trans.GetCurrentGear();

    // 3. DRIVE LOGIC (Arcade Style State Machine)
    if (forward > kThrottleInputThreshold) {
        // VehicleTransmission derives from VehicleTransmissionSettings, so the mode
    // lives on the runtime object itself.
    trans.mMode = JPH::ETransmissionMode::Auto;
        if (cur_gear <= 0 && speed > -kSpeedDirectionThreshold) {
            trans.Set(1, 1.0f);
        }
        if (speed < -kSpeedMinThreshold) {
            input_brake = 1.0f;
        }

    } else if (forward < -kThrottleInputThreshold) {
        trans.mMode = JPH::ETransmissionMode::Manual;
        if (cur_gear != -1 && speed < kSpeedDirectionThreshold) {
            trans.Set(-1, 1.0f);
        }
        if (speed > kSpeedMinThreshold) {
            input_brake = 1.0f;
        }

    } else {
        input_throttle = 0.0f;
        trans.mMode = JPH::ETransmissionMode::Manual;
        if (cur_gear != 0) {
            trans.Set(0, 0.0f);
        }
        if (fabsf(speed) > kSpeedMinThreshold) {
            input_brake = fmaxf(input_brake, kRollingResistanceCoasting);
        }
    }

    // 4. Final Application to Jolt
    controller->SetDriverInput(input_throttle, right, input_brake, handbrake);

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

// --- Helper for faster return value creation ---
// TODO(smell): PyTuple_Pack already increfs each element, so the seven fresh
// references from PyFloat_FromDouble are leaked on every call. Switching to
// PyTuple_SET_ITEM (which steals) fixes it, but that changes observable
// refcounts, so it is left for a dedicated fix.
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

[[nodiscard]] PyObject * VehicleGetWheelTransform(VehicleObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint32_t index         = 0;

    const ParseResult parsed =
        self->world->parsers->WheelIdx.Parse(args, nargs, kwnames, index);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    if (!self->vehicle || index >= self->num_wheels) {
        (self->world->physics.shadow_lock).unlock();
        Py_RETURN_NONE;
    }

    // RMat44::GetQuaternion() rotates the 3x3 part only, which is what the old
    // code did by hand: copy three columns into a Mat44, zero the translation,
    // then convert.
    const JPH::RMat44 transform = self->vehicle->GetWheelWorldTransform(
        index, JPH::Vec3(1.0f, 0.0f, 0.0f), JPH::Vec3(0.0f, 1.0f, 0.0f));

    const JPH::DVec3 p = transform.GetTranslation();
    const JPH::Quat q  = transform.GetQuaternion();
    (self->world->physics.shadow_lock).unlock();

    return pack_transform(p.GetX(), p.GetY(), p.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW());
}

[[nodiscard]] PyObject * VehicleGetWheelLocalTransform(VehicleObject *self,
                                                                         PyObject *const *args,
                                                                         size_t nargsf,
                                                                         PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint32_t index         = 0;

    const ParseResult parsed =
        self->world->parsers->WheelIdx.Parse(args, nargs, kwnames, index);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    if (!self->vehicle || index >= self->num_wheels) {
        (self->world->physics.shadow_lock).unlock();
        Py_RETURN_NONE;
    }

    // The wheel keeps a RefConst to its settings, so the attachment point is
    // readable at runtime without keeping a separate settings array alive.
    const JPH::Wheel *w_ptr          = self->vehicle->GetWheel(index);
    const JPH::WheelSettings *ws     = w_ptr->GetSettings();
    const JPH::Vec3 local_pos_check  = ws->mPosition;

    const JPH::Vec3 right((local_pos_check.GetX() >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
    const JPH::Vec3 up(0.0f, 1.0f, 0.0f);

    const JPH::Mat44 local_transform = self->vehicle->GetWheelLocalTransform(index, right, up);

    const JPH::Vec3 t = local_transform.GetTranslation();
    const JPH::Quat q = local_transform.GetQuaternion();
    (self->world->physics.shadow_lock).unlock();

    return pack_transform(t.GetX(), t.GetY(), t.GetZ(), q.GetX(), q.GetY(), q.GetZ(), q.GetW());
}

[[nodiscard]] PyObject *
VehicleGetDebugState([[maybe_unused]] VehicleObject *self,
                            PyObject *ignored [[maybe_unused]]) {
#ifndef CULVERIN_DEBUG
    Py_RETURN_NONE;
#else
    // 1. LOCK AND GUARD
    // We need the world lock to ensure the vehicle pointer is stable
    // and the physics step isn't currently mutating these values.
    (self->world->physics.shadow_lock).lock();

    if (self->world->physics.is_stepping) {
        (self->world->physics.shadow_lock).unlock();
        // Debug prints usually shouldn't raise Python errors,
        // so we just log a warning and return.
        DebugLog("Warning: Cannot get debug state while physics is stepping.");
        Py_RETURN_NONE;
    }

    if (!self->vehicle) {
        (self->world->physics.shadow_lock).unlock();
        DebugLog("Warning: Vehicle has been destroyed.");
        Py_RETURN_NONE;
    }

    // 2. RESOLVE JOLT COMPONENTS
    auto *controller = static_cast<JPH::WheeledVehicleController *>(self->vehicle->GetController());
    if (!controller) {
        (self->world->physics.shadow_lock).unlock();
        Py_RETURN_NONE;
    }

    const JPH::VehicleEngine &engine           = controller->GetEngine();
    const JPH::VehicleTransmission &trans      = controller->GetTransmission();

    // 3. CAPTURE INPUTS
    [[maybe_unused]] float in_fwd = controller->GetForwardInput();
    [[maybe_unused]] float in_brk = controller->GetBrakeInput();

    // 4. CAPTURE DRIVETRAIN
    [[maybe_unused]] float rpm           = engine.GetCurrentRPM();
    [[maybe_unused]] float engine_torque = engine.GetTorque(in_fwd);
    [[maybe_unused]] int gear            = trans.GetCurrentGear();
    [[maybe_unused]] float clutch        = trans.GetClutchFriction();

    DebugLog("=== VEHICLE DEBUG STATE ===");
    DebugLog("  Inputs: Fwd=%.2f | Brk=%.2f", in_fwd, in_brk);
    DebugLog("  Engine: %.2f RPM | Torque: %.2f Nm", rpm, engine_torque);
    DebugLog("  Trans : Gear %d | Clutch Friction: %.2f", gear, clutch);

    // 5. CAPTURE WHEEL STATE
    for (uint32_t i = 0; i < self->num_wheels; i++) {
        const JPH::Wheel *w          = self->vehicle->GetWheel(i);
        const JPH::WheelSettings *ws = w->GetSettings();

        [[maybe_unused]] bool contact   = w->HasContact();
        [[maybe_unused]] float susp_len = w->GetSuspensionLength();
        [[maybe_unused]] float ang_vel  = w->GetAngularVelocity();
        [[maybe_unused]] float radius   = ws->mRadius;

        [[maybe_unused]] float tire_speed  = ang_vel * radius;
        [[maybe_unused]] float long_lambda = w->GetLongitudinalLambda();
        [[maybe_unused]] float lat_lambda  = w->GetLateralLambda();

        DebugLog("  Wheel %u: %s", i, contact ? "GROUND" : "AIR   ");
        DebugLog("    Susp: %.3fm | AngVel: %.2f rad/s | SurfSpd: %.2f m/s", susp_len, ang_vel,
                      tire_speed);
        DebugLog("    Trac: Long=%.2f | Lat=%.2f", long_lambda, lat_lambda);
    }
    DebugLog("===========================");

    // 6. UNLOCK
    (self->world->physics.shadow_lock).unlock();

    Py_RETURN_NONE;
#endif
}

// --- Vehicle GC Support ---
[[nodiscard]] int VehicleTraverse(VehicleObject *self, visitproc visit,
                                                           void *arg) {
    Py_VISIT(self->world);
    return 0;
}

[[nodiscard]] int VehicleClear(VehicleObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

// --- Internal C Logic (No Python return value to ignore) ---
static void VehicleInternalCleanup(VehicleObject *self) {
    if (!self->world) {
        return;
    }

    (self->world->physics.shadow_lock).lock();

    BlockUntilNotStepping(self->world);
    BlockUntilNotQuerying(self->world);

    if (!self->vehicle) {
        (self->world->physics.shadow_lock).unlock();
        return;
    }

    // Capture pointers and nullptr the struct members immediately
    JPH::VehicleConstraint *j_veh       = self->vehicle;
    JPH::VehicleCollisionTester *tester = self->tester;

    self->vehicle = nullptr;
    self->tester  = nullptr;

    (self->world->physics.shadow_lock).unlock();

    // Safely destroy Jolt objects outside the lock. Wheel settings are held by
    // the constraint's wheels and the controller copied the engine/transmission
    // settings by value, so releasing the constraint retires all of them.
    if (j_veh) {
        self->world->physics.physics_system.RemoveStepListener(j_veh);
        self->world->physics.physics_system.RemoveConstraint(j_veh);
        j_veh->Release();
    }

    if (tester) {
        tester->Release();
    }
}

// --- Python Wrapper ---
// Using [[nodiscard]] on this will now only affect Python callers
[[nodiscard]] PyObject * VehicleDestroy(VehicleObject *self,
                                                          PyObject *ignored [[maybe_unused]]) {
    VehicleInternalCleanup(self);
    if (self->parsers) {

        delete self->parsers;
    }
    Py_RETURN_NONE;
}

// --- Dealloc Slot ---
void VehicleDealloc(VehicleObject *self) {
    PyObject_GC_UnTrack(self);

    // Call the internal C logic directly
    // This avoids [[nodiscard]] warnings because the return is void
    VehicleInternalCleanup(self);

    if (self->parsers) {

        delete self->parsers;
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] PyObject * VehicleGetWheelCount(VehicleObject *self,
                                                   [[maybe_unused]] void *closure) {
    // num_wheels is set at creation and never changes
    return PyLong_FromUnsignedLong(self->num_wheels);
}

[[nodiscard]] PyObject * VehicleSetTankInput(VehicleObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

static PyMethodDef kVehicleMethods[] = {

    CPH::Python::FastCallMethod("set_input", VehicleSetInput),
    CPH::Python::FastCallMethod("set_tank_input", VehicleSetTankInput),
    CPH::Python::FastCallMethod("get_wheel_transform", VehicleGetWheelTransform),
    CPH::Python::FastCallMethod("get_wheel_local_transform", VehicleGetWheelLocalTransform),
    CPH::Python::NoArgsMethod("destroy", VehicleDestroy),
    CPH::Python::NoArgsMethod("get_debug_state", VehicleGetDebugState),
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef VehicleGetSets[] = {
    CPH::Python::GetSet("wheel_count", VehicleGetWheelCount),
    {},
};

static PyType_Slot VehicleSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(VehicleDealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(VehicleTraverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(VehicleClear)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kVehicleMethods)},
    {.slot = Py_tp_getset, .pfunc = reinterpret_cast<void *>(VehicleGetSets)},
    {},
};

PyType_Spec VehicleSpec = {
    .name      = "culverin._culverin_c.Vehicle",
    .basicsize = sizeof(VehicleObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = VehicleSlots,
};

} // namespace CPH::Python
