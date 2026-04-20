#include "culverin_ship.h"
#include "culverin_arg_indices.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"
#include "culverin_python.h"
#include <math.h>

// --- THE HOT PATH (No Locks, No GIL) ---

static void JPH_API_CALL Ship_OnStep(void *userData, const JPH_PhysicsStepListenerContext *inContext) {
    ShipObject *self = (ShipObject *)userData;
    JPH_BodyInterface *bi = JPH_PhysicsSystem_GetBodyInterfaceNoLock(inContext->physicsSystem);
    
    if (self->sled_bid == JPH_INVALID_BODY_ID) return;

    // 1. Get current state
    JPH_Vec3 ang_vel;
    JPH_Quat rot;
    JPH_BodyInterface_GetAngularVelocity(bi, self->sled_bid, &ang_vel);
    JPH_BodyInterface_GetRotation(bi, self->sled_bid, &rot);

    // 2. PD Stabilizer (Keep upright)
    JPH_Vec3 up_dir = {0, 1.0f, 0};
    JPH_Vec3 current_up;
    JPH_Quat_Rotate(&rot, &up_dir, &current_up);

    // --- CORRECTED PD MATH ---
    // Torque around X axis corrects tilt in Z (Pitch)
    float tx = (-current_up.z * self->kp) - (ang_vel.x * self->kd);
    // Torque around Z axis corrects tilt in X (Roll)
    // Note the sign flip here: if ux is negative (tilted left), 
    // we need negative torque (clockwise) to roll back right.
    float tz = (current_up.x * self->kp) - (ang_vel.z * self->kd);
    
    JPH_Vec3 torque = {tx, 0.0f, tz};
    JPH_BodyInterface_AddTorque(bi, self->sled_bid, &torque);

    // 3. Forward Movement (Throttle)
    float fwd_input = atomic_load_explicit(&self->input_fwd, memory_order_relaxed);
    if (fabsf(fwd_input) > 0.01f) {
        JPH_Vec3 fwd_dir = {0, 0, 1.0f};
        JPH_Vec3 current_fwd;
        JPH_Quat_Rotate(&rot, &fwd_dir, &current_fwd);

        JPH_Vec3 force = {
            current_fwd.x * fwd_input * self->throttle_force,
            0.0f,
            current_fwd.z * fwd_input * self->throttle_force
        };
        JPH_BodyInterface_AddForce(bi, self->sled_bid, &force);
    }

    // 4. Steering (High-speed angular velocity clamp)
    float steer_input = atomic_load_explicit(&self->input_right, memory_order_relaxed);
    // Always apply steering velocity to the Y axis to override drift, 
    // while preserving the stabilizer's damping on X and Z.
    JPH_Vec3 new_avel = {ang_vel.x, steer_input * self->steer_speed, ang_vel.z};
    JPH_BodyInterface_SetAngularVelocity(bi, self->sled_bid, &new_avel);
}

static const JPH_PhysicsStepListener_Procs ship_listener_procs = {
    .OnStep = Ship_OnStep
};

// --- CREATION ---

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ship(PhysicsWorldObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs,
                                                             PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    uint64_t sled_raw = 0;
    float kp = 0.0f, kd = 0.0f, throttle = 0.0f, steer = 0.0f;

    void *targets[CreateShip_COUNT] = {
        [IDX_CS_SLED]     = (void *)&sled_raw,
        [IDX_CS_KP]       = (void *)&kp,
        [IDX_CS_KD]       = (void *)&kd,
        [IDX_CS_THROTTLE] = (void *)&throttle,
        [IDX_CS_STEER]    = (void *)&steer
    };

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
    if (!obj) return nullptr;

    obj->world = (PhysicsWorldObject *)Py_NewRef(self);
    obj->sled_bid = bid;
    obj->sled_h_raw = sled_raw;
    atomic_init(&obj->input_fwd, 0.0f);
    atomic_init(&obj->input_right, 0.0f);
    obj->kp = kp;
    obj->kd = kd;
    obj->throttle_force = throttle;
    obj->steer_speed = steer;

    // Use global trampoline lock to register listener
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    JPH_PhysicsStepListener_SetProcs(&ship_listener_procs);
    obj->listener = JPH_PhysicsStepListener_Create(obj);
    JPH_PhysicsSystem_AddStepListener(self->system, obj->listener);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    PyObject_GC_Track((PyObject *)obj);
    return (PyObject *)obj;
}

// --- INPUT (ULTRA FAST) ---

PyCFunction_DeclareMethodFromModule Ship_set_input(ShipObject *self, PyObject *const *args,
                                                   Py_ssize_t nargs, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    float fwd = 0.0f, right = 0.0f;
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
        NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
        JPH_PhysicsSystem_RemoveStepListener(self->world->system, self->listener);
        JPH_PhysicsStepListener_Destroy(self->listener);
        NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
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
    .slots = (PyType_Slot[]){
        {.slot = Py_tp_dealloc, .pfunc = Ship_dealloc},
        {.slot = Py_tp_traverse, .pfunc = Ship_traverse},
        {.slot = Py_tp_clear, .pfunc = Ship_clear},
        {.slot = Py_tp_methods, .pfunc = (PyMethodDef[]){
            SHIP_FASTCALL(set_input),
            {}
        }},
        {}
    },
};