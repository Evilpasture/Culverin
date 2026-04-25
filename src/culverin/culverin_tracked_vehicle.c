#include "culverin_tracked_vehicle.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_compiler_specifics.h"
#include "culverin_module.h"
#include "culverin_parsers.h"
#include "culverin_physics_sync.h"
#include "culverin_python.h"

// --- Tracked Vehicle Constants ---

// Wheel configuration defaults
static constexpr float TRACKED_WHEEL_RADIUS_DEFAULT     = 0.4f;
static constexpr float TRACKED_WHEEL_WIDTH_DEFAULT      = 0.2f;
static constexpr float TRACKED_WHEEL_SUSPENSION_DEFAULT = 0.5f;

// Suspension parameters
static constexpr float TRACKED_SUSPENSION_MIN_LENGTH = 0.05f;

// Spring properties
static constexpr float TRACKED_SPRING_FREQ_DEFAULT = 2.0f;
static constexpr float TRACKED_SPRING_DAMP_DEFAULT = 0.5f;

// Transmission gear ratios
static constexpr float TRACKED_GEAR_RATIO_1       = 2.0f;
static constexpr float TRACKED_GEAR_RATIO_2       = 1.4f;
static constexpr float TRACKED_GEAR_RATIO_3       = 1.0f;
static constexpr float TRACKED_GEAR_RATIO_4       = 0.7f;
static constexpr float TRACKED_REVERSE_GEAR_RATIO = -1.5f;

// Engine parameters (RPM and torque)
static constexpr float TRACKED_ENGINE_MAX_RPM_DEFAULT    = 6000.0f;
static constexpr float TRACKED_ENGINE_MIN_RPM_DEFAULT    = 500.0f;
static constexpr float TRACKED_ENGINE_MAX_TORQUE_DEFAULT = 5000.0f;

// Collision tester parameters
static constexpr float TRACKED_COLLISION_TESTER_SCALE = 2.0f;

// Throttle activation threshold
static constexpr float TRACKED_THROTTLE_KICKSTART_THRESHOLD = 0.01f;

// --- Tracked Vehicle Implementation ---

static JPH_WheelSettings *create_track_wheel(PyObject *w_dict) {
    Vec3f pos;
    if (!parse_py_vec3(PyDict_GetItemString(w_dict, "pos"), &pos)) {
        return nullptr;
    }

    float radius         = get_py_attr(w_dict, "radius", TRACKED_WHEEL_RADIUS_DEFAULT);
    float width          = get_py_attr(w_dict, "width", TRACKED_WHEEL_WIDTH_DEFAULT);
    float suspension_len = get_py_attr(w_dict, "suspension", TRACKED_WHEEL_SUSPENSION_DEFAULT);
    float friction       = get_py_attr(w_dict, "friction", 1.0f);

    // Suspension Spring Properties
    float freq = get_py_attr(w_dict, "spring_freq", TRACKED_SPRING_FREQ_DEFAULT);
    float damp = get_py_attr(w_dict, "spring_damp", TRACKED_SPRING_DAMP_DEFAULT);

    JPH_WheelSettingsTV *w = JPH_WheelSettingsTV_Create();

    JPH_WheelSettings_SetPosition((JPH_WheelSettings *)w, &(JPH_Vec3){pos.x, pos.y, pos.z});
    JPH_WheelSettings_SetRadius((JPH_WheelSettings *)w, radius);
    JPH_WheelSettings_SetWidth((JPH_WheelSettings *)w, width);

    JPH_WheelSettings_SetSuspensionMinLength((JPH_WheelSettings *)w, TRACKED_SUSPENSION_MIN_LENGTH);
    JPH_WheelSettings_SetSuspensionMaxLength((JPH_WheelSettings *)w, suspension_len);

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

    auto t_ctrl = JPH_TrackedVehicleControllerSettings_Create();

    JPH_VehicleEngineSettings eng;
    JPH_VehicleEngineSettings_Init(&eng);

    eng.maxTorque = config.torque;
    eng.maxRPM    = config.max_rpm;
    eng.minRPM    = config.min_rpm;

    JPH_TrackedVehicleControllerSettings_SetEngine(t_ctrl, &eng);

    auto trans = JPH_VehicleTransmissionSettings_Create();
    JPH_VehicleTransmissionSettings_SetMode(trans, JPH_TransmissionMode_Auto);

    float gears[] = {TRACKED_GEAR_RATIO_1, TRACKED_GEAR_RATIO_2, TRACKED_GEAR_RATIO_3,
                     TRACKED_GEAR_RATIO_4};
    JPH_VehicleTransmissionSettings_SetGearRatios(trans, gears, 4);
    float reverse[] = {TRACKED_REVERSE_GEAR_RATIO};
    JPH_VehicleTransmissionSettings_SetReverseGearRatios(trans, reverse, 1);

    JPH_TrackedVehicleControllerSettings_SetTransmission(t_ctrl, trans);
    *out_trans = trans;
    return t_ctrl;
}

extern void cleanup_vehicle_resources(VehicleResources *r, uint32_t num_wheels,
                                      PhysicsWorldObject *self);

// Orchestrator
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_tracked_vehicle(PhysicsWorldObject *self,
                                                                        PyObject *const *args,
                                                                        Py_ssize_t nargs,
                                                                        PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // --- 1. FAST ARGUMENT PARSING (Unchanged) ---
    uint64_t chassis_h_raw = 0;
    PyObject *py_wheels    = nullptr;
    PyObject *py_tracks    = nullptr;
    float max_torque       = TRACKED_ENGINE_MAX_TORQUE_DEFAULT;
    float max_rpm          = TRACKED_ENGINE_MAX_RPM_DEFAULT;
    float min_rpm          = TRACKED_ENGINE_MIN_RPM_DEFAULT;

    void *targets[CreateTracked_COUNT] = {
        [IDX_CT_CHASSIS] = (void *)&chassis_h_raw, [IDX_CT_WHEELS] = (void *)&py_wheels,
        [IDX_CT_TRACKS] = (void *)&py_tracks,      [IDX_CT_TORQUE] = (void *)&max_torque,
        [IDX_CT_RPM] = (void *)&max_rpm,
    };

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CreateTrackedParser, targets)) {
        return nullptr;
    }

    // --- 2. RESOLVE CHASSIS (ATOMIC REFACTOR) ---
    SHADOW_LOCK(&self->shadow_lock);

    // Ensure all pending creation commands are flushed before we try to link a constraint
    sync_and_flush_internal(self);

    uint32_t slot = 0;
    // TSan Fix: Cast raw uint64 to atomic BodyHandle for verification
    bool handle_valid = unpack_handle(self, (BodyHandle)chassis_h_raw, &slot);

    // TSan Fix: Atomic load of the slot state (Acquire ensures chassis data is visible)
    uint8_t state = (int)handle_valid
                        ? atomic_load_explicit(&self->slot_states[slot], memory_order_acquire)
                        : SLOT_EMPTY;

    if (!handle_valid || state != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_ValueError, "Invalid chassis handle (Body is dead or pending)");
    }

    // Access non-atomic dense buffer while holding shadow_lock
    JPH_BodyID chassis_bid = self->body_ids[self->slot_to_dense[slot]];
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 3. PRE-JOLT RESOURCE ALLOCATION (Unchanged) ---
    VehicleResources r = {0};
    auto num_wheels    = (uint32_t)PyList_Size(py_wheels);
    TrackData tracks[2];
    memset(tracks, 0, sizeof(tracks));
    int num_tracks = 0;

    r.f_curve = JPH_LinearCurve_Create();
    JPH_LinearCurve_AddPoint(r.f_curve, 0.0f, 1.0f);
    JPH_LinearCurve_AddPoint(r.f_curve, 1.0f, 1.0f);

    r.w_settings = (JPH_WheelSettings **)CULV_RAW_CALLOC(num_wheels, sizeof(JPH_WheelSettings *));
    for (uint32_t i = 0; i < num_wheels; i++) {
        r.w_settings[i] = create_track_wheel(PyList_GetItem(py_wheels, i));
        if (!r.w_settings[i]) {
            goto python_fail;
        }
    }

    if (!parse_tracks_to_c(py_tracks, tracks, &num_tracks)) {
        goto python_fail;
    }

    // --- 4. JOLT COMMIT (No GIL - Unchanged) ---
    bool jolt_locked     = false;
    PyThreadState *_save = nullptr;
    Py_UNBLOCK_THREADS;

    NATIVE_MUTEX_LOCK(self->jph_trampoline_lock);
    jolt_locked = true;

    const JPH_BodyLockInterface *lock_iface = JPH_PhysicsSystem_GetBodyLockInterface(self->system);
    JPH_BodyLockWrite lock                  = {0};
    JPH_BodyLockInterface_LockWrite(lock_iface, chassis_bid, &lock);

    if (UNLIKELY(!lock.body)) {
        goto jolt_fail;
    }

    TrackedEngineConfig eng_cfg = {.torque = max_torque, .max_rpm = max_rpm, .min_rpm = min_rpm};
    JPH_VehicleTransmissionSettings *v_trans = nullptr;
    JPH_TrackedVehicleControllerSettings *t_ctrl =
        init_tracked_controller_settings(eng_cfg, &v_trans);
    r.v_ctrl      = (JPH_WheeledVehicleControllerSettings *)t_ctrl;
    r.v_trans_set = v_trans;

    for (int t = 0; t < num_tracks; t++) {
        JPH_VehicleTrackSettings track_set;
        JPH_VehicleTrackSettings_Init(&track_set);
        track_set.wheels      = tracks[t].indices;
        track_set.wheelsCount = tracks[t].count;
        track_set.drivenWheel = tracks[t].driven_idx;
        JPH_TrackedVehicleControllerSettings_SetTrack(t_ctrl, (uint32_t)t, &track_set);
    }

    JPH_VehicleConstraintSettings v_set;
    JPH_VehicleConstraintSettings_Init(&v_set);
    v_set.wheelsCount = num_wheels;
    v_set.wheels      = r.w_settings;
    v_set.controller  = (JPH_VehicleControllerSettings *)t_ctrl;

    r.j_veh = JPH_VehicleConstraint_Create(lock.body, &v_set);
    if (!r.j_veh) {
        goto jolt_fail;
    }

    r.tester = JPH_VehicleCollisionTesterRay_Create(TRACKED_LAYER_DRIVABLE, &(JPH_Vec3){0, 1.0f, 0},
                                                    TRACKED_COLLISION_TESTER_SCALE);
    if (!r.tester) {
        goto jolt_fail;
    }

    JPH_VehicleConstraint_SetVehicleCollisionTester(r.j_veh,
                                                    (JPH_VehicleCollisionTester *)r.tester);
    JPH_PhysicsSystem_AddConstraint(self->system, (JPH_Constraint *)r.j_veh);
    JPH_PhysicsSystem_AddStepListener(self->system,
                                      JPH_VehicleConstraint_AsPhysicsStepListener(r.j_veh));
    r.is_added_to_world = true;

    JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    NATIVE_MUTEX_UNLOCK(self->jph_trampoline_lock);
    jolt_locked = false;
    Py_BLOCK_THREADS;

    // --- 5. CLEANUP & WRAP (Unchanged) ---
    for (int t = 0; t < num_tracks; t++) {
        CULV_RAW_FREE(tracks[t].indices);
    }

    auto obj = (VehicleObject *)PyObject_GC_New(VehicleObject, (PyTypeObject *)st->VehicleType);
    if (!obj) {
        SHADOW_LOCK(&self->shadow_lock);
        cleanup_vehicle_resources(&r, num_wheels, self);
        SHADOW_UNLOCK(&self->shadow_lock);
        return nullptr;
    }

    obj->vehicle               = r.j_veh;
    obj->tester                = (JPH_VehicleCollisionTester *)r.tester;
    obj->world                 = (PhysicsWorldObject *)Py_NewRef(self);
    obj->num_wheels            = num_wheels;
    obj->wheel_settings        = r.w_settings;
    obj->controller_settings   = (JPH_VehicleControllerSettings *)t_ctrl;
    obj->transmission_settings = r.v_trans_set;
    obj->friction_curve        = r.f_curve;
    obj->torque_curve          = nullptr;

    PyObject_GC_Track((PyObject *)obj);
    return (PyObject *)obj;

jolt_fail:
    if (lock.body) {
        JPH_BodyLockInterface_UnlockWrite(lock_iface, &lock);
    }
    if (jolt_locked) {
        NATIVE_MUTEX_UNLOCK(self->jph_trampoline_lock);
    }
    Py_BLOCK_THREADS;

python_fail:
    for (int t = 0; t < num_tracks; t++) {
        if (tracks[t].indices) {
            CULV_RAW_FREE(tracks[t].indices);
        }
    }
    SHADOW_LOCK(&self->shadow_lock);
    cleanup_vehicle_resources(&r, num_wheels, self);
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_Format(PyExc_RuntimeError, "Tracked vehicle creation failed");
}

// Helper: Set Tank Input
PyCFunction_DeclareMethodFromModule Vehicle_set_tank_input(VehicleObject *self,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. FAST PARSE (Zero-Allocation)
    float left  = 0.0f;
    float right = 0.0f;
    float brake = 0.0f;

    void *targets[TankInput_COUNT] = {
        [IDX_TI_LEFT]  = (void *)&left,
        [IDX_TI_RIGHT] = (void *)&right,
        [IDX_TI_BRAKE] = (void *)&brake,
    };

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.TankInputParser, targets)) {
        return nullptr;
    }

    // Safety check outside lock
    if (UNLIKELY(!self->vehicle || !self->world)) {
        Py_RETURN_NONE;
    }

    // 2. STATE SNAPSHOT & JOLT ACTIVATION
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    auto t_ctrl =
        (JPH_TrackedVehicleController *)JPH_VehicleConstraint_GetController(self->vehicle);
    JPH_BodyID bid = JPH_Body_GetID(JPH_VehicleConstraint_GetVehicleBody(self->vehicle));

    // Wake up the tank to process inputs
    JPH_BodyInterface_ActivateBody(self->world->body_interface, bid);

    auto trans = (JPH_VehicleTransmission *)JPH_TrackedVehicleController_GetTransmission(t_ctrl);
    int gear   = JPH_VehicleTransmission_GetCurrentGear(trans);

    // 3. TANK DRIVE LOGIC
    // Throttle for a tracked vehicle is typically the max absolute power requested from either side
    float throttle = fmaxf(fabsf(left), fabsf(right));

    // Simple Kickstart Logic: Auto-shift from Neutral to Gear 1 when throttle is applied
    if (throttle > TRACKED_THROTTLE_KICKSTART_THRESHOLD) {
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

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}