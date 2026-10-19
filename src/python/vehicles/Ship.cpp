#include "vehicles/Ship.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include <math.h>

namespace CPH::Python {

// --- THE HOT PATH (No Locks, No GIL) ---

static void JPH_API_CALL Ship_OnStep(void *userData,
                                     const JPH_PhysicsStepListenerContext *inContext) {
    ShipObject *self  = static_cast<ShipObject *>(userData);
    JPH_BodyInterface *bi = JPH_PhysicsSystem_GetBodyInterfaceNoLock(inContext->physicsSystem);

    if (self->sled_bid == kJphInvalidBodyId) {
        return;
    }

    // 1. Get current state
    JPH_Vec3 ang_vel;
    JPH_Quat rot;
    JPH_Vec3 lin_vel;
    JPH_BodyInterface_GetAngularVelocity(bi, self->sled_bid, &ang_vel);
    JPH_BodyInterface_GetRotation(bi, self->sled_bid, &rot);
    JPH_BodyInterface_GetLinearVelocity(bi, self->sled_bid, &lin_vel);

    float fwd_input   = atomic_load_explicit(&self->input_fwd, memory_order_relaxed);
    float steer_input = atomic_load_explicit(&self->input_right, memory_order_relaxed);

    // 2. PD Stabilizer & Banking
    JPH_Vec3 up_dir = {0, 1.0f, 0};
    JPH_Vec3 current_up;
    JPH_Quat_Rotate(&rot, &up_dir, &current_up);

    float target_roll_z = -steer_input * self->banking_strength;

    // Torque X corrects Pitch, Torque Z corrects Roll (plus banking target)
    float tx = (-current_up.z * self->kp) - (ang_vel.x * self->kd);
    float tz = ((current_up.x - target_roll_z) * self->kp) - (ang_vel.z * self->kd);

    // 3. Torque Steering (Yaw)
    float current_yaw_vel = ang_vel.y;
    float target_yaw_vel  = steer_input * self->steer_speed;
    float ty              = (target_yaw_vel - current_yaw_vel) * (self->kp * 0.5f);

    JPH_Vec3 torque = {tx, ty, tz};
    JPH_BodyInterface_AddTorque(bi, self->sled_bid, &torque);

    // 4. Forward Movement (Throttle)
    if (fabsf(fwd_input) > 0.01f) {
        JPH_Vec3 fwd_dir = {0, 0, 1.0f};
        JPH_Vec3 current_fwd;
        JPH_Quat_Rotate(&rot, &fwd_dir, &current_fwd);

        JPH_Vec3 force = {current_fwd.x * fwd_input * self->throttle_force, 0.0f,
                          current_fwd.z * fwd_input * self->throttle_force};
        JPH_BodyInterface_AddForce(bi, self->sled_bid, &force);
    }

    // 5. Lateral Friction (Anti-Drift / Keel effect)
    JPH_Vec3 right_dir = {1.0f, 0, 0};
    JPH_Vec3 current_right;
    JPH_Quat_Rotate(&rot, &right_dir, &current_right);

    float lateral_speed = (lin_vel.x * current_right.x) + (lin_vel.y * current_right.y) +
                          (lin_vel.z * current_right.z);
    float lateral_force_mag = -lateral_speed * self->lateral_grip;

    if (fabsf(lateral_force_mag) > 0.01f) {
        JPH_Vec3 side_force = {current_right.x * lateral_force_mag,
                               current_right.y * lateral_force_mag,
                               current_right.z * lateral_force_mag};
        JPH_BodyInterface_AddForce(bi, self->sled_bid, &side_force);
    }

    // 6. Quadratic Drag (Speed Limiter)
    float speed_sq = (lin_vel.x * lin_vel.x) + (lin_vel.y * lin_vel.y) + (lin_vel.z * lin_vel.z);
    if (speed_sq > 0.01f) {
        float speed         = sqrtf(speed_sq);
        float drag_mag      = speed_sq * self->linear_drag;
        JPH_Vec3 drag_force = {-(lin_vel.x / speed) * drag_mag, -(lin_vel.y / speed) * drag_mag,
                               -(lin_vel.z / speed) * drag_mag};
        JPH_BodyInterface_AddForce(bi, self->sled_bid, &drag_force);
    }
}

static const JPH_PhysicsStepListener_Procs kShipListenerProcs = {.OnStep = Ship_OnStep};

// --- CREATION ---
[[nodiscard]] PyObject * WorldCreateShip(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint64_t sled_raw      = 0;
    float kp               = 0.0f;
    float kd               = 0.0f;
    float throttle         = 0.0f;
    float steer            = 0.0f;
    float banking          = 0.15f;
    float grip             = 500.0f;
    float drag             = 10.0f;

    if (!self->parsers->CreateShip.Parse(args, nargs, kwnames, sled_raw, kp, kd, throttle, steer,
                                         banking, grip, drag)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    SyncAndFlushInternal(self);

    uint32_t slot;
    if (!unpack_handle(&self->physics, static_cast<BodyHandle>(sled_raw), &slot)) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        return PyErr_Format(PyExc_ValueError, "Invalid sled handle.");
    }
    JPH_BodyID bid = self->physics.body_ids[self->physics.slot_to_dense[slot]];
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
    ModuleState *st = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    auto obj = PyObject_GC_New(ShipObject, reinterpret_cast<PyTypeObject *>(st->ShipType));
    if (!obj) {
        return nullptr;
    }
    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::ShipParsers>();
    if (!obj->parsers) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return nullptr;
    }

    obj->world      = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->sled_bid   = bid;
    obj->sled_h_raw = sled_raw;
    atomic_init(&obj->input_fwd, 0.0f);
    atomic_init(&obj->input_right, 0.0f);
    obj->kp               = kp;
    obj->kd               = kd;
    obj->throttle_force   = throttle;
    obj->steer_speed      = steer;
    obj->banking_strength = banking;
    obj->lateral_grip     = grip;
    obj->linear_drag      = drag;

    // Use global trampoline lock to register listener
    CPH_NATIVE_MUTEX_LOCK(self->physics.jph_trampoline_lock);
    JPH_PhysicsStepListener_SetProcs(&kShipListenerProcs);
    obj->listener = JPH_PhysicsStepListener_Create(obj);
    JPH_PhysicsSystem_AddStepListener(self->physics.system, obj->listener);
    CPH_NATIVE_MUTEX_UNLOCK(self->physics.jph_trampoline_lock);

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);
}

// --- INPUT (ULTRA FAST) ---

[[nodiscard]] PyObject * ShipSetInput(ShipObject *self, PyObject *const *args,
                                                        size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float fwd              = 0.0f;
    float right            = 0.0f;

    if (!self->parsers->ShipInput.Parse(args, nargs, kwnames, fwd, right)) {
        return nullptr;
    }

    // No locks! Just update atomics.
    // The C listener running in the Jolt thread will pick these up automatically.
    atomic_store_explicit(&self->input_fwd, fwd, memory_order_relaxed);
    atomic_store_explicit(&self->input_right, right, memory_order_relaxed);

    Py_RETURN_NONE;
}

// --- CLEANUP ---
void ShipDealloc(ShipObject *self) {
    PyObject_GC_UnTrack(self);

    if (self->world && self->listener) {
        // Protect Jolt system call with trampoline lock
        CPH_NATIVE_MUTEX_LOCK(self->world->physics.jph_trampoline_lock);
        JPH_PhysicsSystem_RemoveStepListener(self->world->physics.system, self->listener);
        JPH_PhysicsStepListener_Destroy(self->listener);
        CPH_NATIVE_MUTEX_UNLOCK(self->world->physics.jph_trampoline_lock);
    }

    if (self->parsers) {

        delete self->parsers;
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

[[nodiscard]] int ShipTraverse(ShipObject *self, visitproc visit,
                                                        void *arg) {
    Py_VISIT(self->world);
    return 0;
}

[[nodiscard]] int ShipClear(ShipObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

static PyMethodDef kShipMethods[] = {CPH::Python::FastCallMethod("set_input", ShipSetInput),
                                     {nullptr, nullptr, 0, nullptr}};

static PyType_Slot ShipSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(ShipDealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(ShipTraverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(ShipClear)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kShipMethods)},
    {},
};

PyType_Spec ShipSpec = {
    .name      = "culverin._culverin_c.Ship",
    .basicsize = sizeof(ShipObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = ShipSlots,
};

} // namespace CPH::Python
