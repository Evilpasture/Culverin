#include "culverin_ship.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_module.h"
#include "culverin_physics_world.h"
#include "culverin_physics_world_internal.h"
#include "culverin_python.h"
#include <math.h>

// --- THE HOT PATH (No Locks, No GIL) ---

static void JPH_API_CALL Ship_OnStep(void *userData,
                                     const JPH_PhysicsStepListenerContext *inContext) {
    ShipObject *self      = (ShipObject *)userData;
    JPH_BodyInterface *bi = JPH_PhysicsSystem_GetBodyInterfaceNoLock(inContext->physicsSystem);

    if (self->sled_bid == JPH_INVALID_BODY_ID) {
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

    float lateral_speed     = (lin_vel.x * current_right.x) + (lin_vel.y * current_right.y) +
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

static const JPH_PhysicsStepListener_Procs ship_listener_procs = {.OnStep = Ship_OnStep};

// --- CREATION ---

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ship(PhysicsWorldObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    uint64_t sled_raw = 0;
    float kp          = 0.0f;
    float kd          = 0.0f;
    float throttle    = 0.0f;
    float steer       = 0.0f;
    float banking     = 0.15f;
    float grip        = 500.0f;
    float drag        = 10.0f;

    void *targets[CreateShip_COUNT] = {
        [IDX_CS_SLED] = (void *)&sled_raw, [IDX_CS_KP] = (void *)&kp,
        [IDX_CS_KD] = (void *)&kd,         [IDX_CS_THROTTLE] = (void *)&throttle,
        [IDX_CS_STEER] = (void *)&steer,   [IDX_CS_BANKING] = (void *)&banking,
        [IDX_CS_GRIP] = (void *)&grip,     [IDX_CS_DRAG] = (void *)&drag};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CreateShipParser, targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->shadow_lock);
    sync_and_flush_internal(self);

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)sled_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_Format(PyExc_ValueError, "Invalid sled handle.");
    }
    JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
    SHADOW_UNLOCK(&self->shadow_lock);

    auto obj = (ShipObject *)PyObject_GC_New(ShipObject, (PyTypeObject *)st->ShipType);
    if (!obj) {
        return nullptr;
    }

    obj->world      = (PhysicsWorldObject *)Py_NewRef(self);
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
    NATIVE_MUTEX_LOCK(self->jph_trampoline_lock);
    JPH_PhysicsStepListener_SetProcs(&ship_listener_procs);
    obj->listener = JPH_PhysicsStepListener_Create(obj);
    JPH_PhysicsSystem_AddStepListener(self->system, obj->listener);
    NATIVE_MUTEX_UNLOCK(self->jph_trampoline_lock);

    PyObject_GC_Track((PyObject *)obj);
    return (PyObject *)obj;
}

// --- INPUT (ULTRA FAST) ---

PyCFunction_DeclareMethodFromModule Ship_set_input(ShipObject *self, PyObject *const *args,
                                                   Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st              = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    float fwd                      = 0.0f;
    float right                    = 0.0f;
    void *targets[ShipInput_COUNT] = {[IDX_SI_FWD] = &fwd, [IDX_SI_RIGHT] = &right};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.ShipInputParser, targets)) {
        return nullptr;
    }

    // No locks! Just update atomics.
    // The C listener running in the Jolt thread will pick these up automatically.
    atomic_store_explicit(&self->input_fwd, fwd, memory_order_relaxed);
    atomic_store_explicit(&self->input_right, right, memory_order_relaxed);

    Py_RETURN_NONE;
}

// --- CLEANUP ---

PyType_DeclareSlot_VoidFromModule Ship_dealloc(ShipObject *self) {
    PyObject_GC_UnTrack(self);

    if (self->world && self->listener) {
        // Protect Jolt system call with trampoline lock
        NATIVE_MUTEX_LOCK(self->world->jph_trampoline_lock);
        JPH_PhysicsSystem_RemoveStepListener(self->world->system, self->listener);
        JPH_PhysicsStepListener_Destroy(self->listener);
        NATIVE_MUTEX_UNLOCK(self->world->jph_trampoline_lock);
    }

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyType_DeclareSlot_StatusFromModule Ship_traverse(ShipObject *self, visitproc visit, void *arg) {
    Py_VISIT(self->world);
    return 0;
}

PyType_DeclareSlot_StatusFromModule Ship_clear(ShipObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

#define SHIP_FASTCALL(name) CULV_FEAT(Ship, name, METH_FASTCALL | METH_KEYWORDS)
PyType_Spec Ship_spec = {
    .name      = "culverin._culverin_c.Ship",
    .basicsize = sizeof(ShipObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = (PyType_Slot[]){{.slot = Py_tp_dealloc, .pfunc = Ship_dealloc},
                                 {.slot = Py_tp_traverse, .pfunc = Ship_traverse},
                                 {.slot = Py_tp_clear, .pfunc = Ship_clear},
                                 {.slot  = Py_tp_methods,
                                  .pfunc = (PyMethodDef[]){SHIP_FASTCALL(set_input), {}}},
                                 {}},
};