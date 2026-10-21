#include "vehicles/Ship.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include <Jolt/Physics/Body/BodyInterface.h>
#include <math.h>

namespace CPH::Python {

// --- THE HOT PATH (No Locks, No GIL) ---

namespace {

/// Per-ship physics step listener.
///
/// This replaces the last global proc table in the tree. JoltC's
/// JPH_PhysicsStepListener_SetProcs installed one process-wide s_Procs pointer
/// shared by every ship, with OnStep recovering its owner from a void
/// *userData; each ship now owns its listener, so there is no shared state and
/// no C trampoline.
class ShipStepListener final : public JPH::PhysicsStepListener {
  public:
    explicit ShipStepListener(ShipObject *owner) : owner_(owner) {}

    void OnStep([[maybe_unused]] const JPH::PhysicsStepListenerContext &context) override {
        ShipObject *self = owner_;
        if (self->sled_bid.IsInvalid()) {
            return;
        }

        const JPH::BodyID &sled_id = self->sled_bid;
        // OnStep runs while every body mutex is held, so this has to be the
        // no-lock interface; the locking one would deadlock.
        JPH::BodyInterface &bi = self->world->physics.physics_system.GetBodyInterfaceNoLock();

        // 1. Get current state
        const JPH::Vec3 ang_vel = bi.GetAngularVelocity(sled_id);
        const JPH::Quat rot     = bi.GetRotation(sled_id);
        const JPH::Vec3 lin_vel = bi.GetLinearVelocity(sled_id);

        const float fwd_input   = (self->input_fwd).load(std::memory_order::relaxed);
        const float steer_input = (self->input_right).load(std::memory_order::relaxed);

        // 2. PD Stabilizer & Banking
        const JPH::Vec3 current_up = rot * JPH::Vec3(0, 1.0f, 0);

        const float target_roll_z = -steer_input * self->banking_strength;

        // Torque X corrects Pitch, Torque Z corrects Roll (plus banking target)
        const float tx = (-current_up.GetZ() * self->kp) - (ang_vel.GetX() * self->kd);
        const float tz =
            ((current_up.GetX() - target_roll_z) * self->kp) - (ang_vel.GetZ() * self->kd);

        // 3. Torque Steering (Yaw)
        const float current_yaw_vel = ang_vel.GetY();
        const float target_yaw_vel  = steer_input * self->steer_speed;
        const float ty              = (target_yaw_vel - current_yaw_vel) * (self->kp * 0.5f);

        bi.AddTorque(sled_id, JPH::Vec3(tx, ty, tz));

        // 4. Forward Movement (Throttle)
        if (fabsf(fwd_input) > 0.01f) {
            const JPH::Vec3 current_fwd = rot * JPH::Vec3(0, 0, 1.0f);

            const JPH::Vec3 force(current_fwd.GetX() * fwd_input * self->throttle_force, 0.0f,
                                  current_fwd.GetZ() * fwd_input * self->throttle_force);
            bi.AddForce(sled_id, force);
        }

        // 5. Lateral Friction (Anti-Drift / Keel effect)
        const JPH::Vec3 current_right = rot * JPH::Vec3(1.0f, 0, 0);

        // Dots and the drag division are kept as explicit scalar arithmetic
        // rather than Vec3::Dot / operator/: this runs every step, Jolt's Dot
        // reorders the adds, and its scalar division multiplies by a reciprocal.
        // Either would drift the recorded-scene replays.
        const float lateral_speed = (lin_vel.GetX() * current_right.GetX()) +
                                    (lin_vel.GetY() * current_right.GetY()) +
                                    (lin_vel.GetZ() * current_right.GetZ());
        const float lateral_force_mag = -lateral_speed * self->lateral_grip;

        if (fabsf(lateral_force_mag) > 0.01f) {
            const JPH::Vec3 side_force(current_right.GetX() * lateral_force_mag,
                                       current_right.GetY() * lateral_force_mag,
                                       current_right.GetZ() * lateral_force_mag);
            bi.AddForce(sled_id, side_force);
        }

        // 6. Quadratic Drag (Speed Limiter)
        const float speed_sq = (lin_vel.GetX() * lin_vel.GetX()) +
                               (lin_vel.GetY() * lin_vel.GetY()) +
                               (lin_vel.GetZ() * lin_vel.GetZ());
        if (speed_sq > 0.01f) {
            const float speed    = sqrtf(speed_sq);
            const float drag_mag = speed_sq * self->linear_drag;
            const JPH::Vec3 drag_force(-(lin_vel.GetX() / speed) * drag_mag,
                                       -(lin_vel.GetY() / speed) * drag_mag,
                                       -(lin_vel.GetZ() / speed) * drag_mag);
            bi.AddForce(sled_id, drag_force);
        }
    }

  private:
    ShipObject *owner_;
};

} // namespace

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

    const ParseResult parsed =
        self->parsers->CreateShip.Parse(args, nargs, kwnames, sled_raw, kp, kd, throttle, steer,
                                         banking, grip, drag);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    (self->physics.shadow_lock).lock();
    SyncAndFlushInternal(self);

    const std::optional<std::uint32_t> resolved =
        self->physics.Resolve(static_cast<BodyHandle>(sled_raw));
    if (!resolved) {
        (self->physics.shadow_lock).unlock();
        return PyErr_Format(PyExc_ValueError, "Invalid sled handle.");
    }
    const uint32_t slot = *resolved;
    JPH::BodyID bid = self->physics.bodies.body_ids[self->physics.bodies.slot_to_dense[slot]];
    (self->physics.shadow_lock).unlock();
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
    (obj->input_fwd).store(0.0f, std::memory_order::relaxed);
    (obj->input_right).store(0.0f, std::memory_order::relaxed);
    obj->kp               = kp;
    obj->kd               = kd;
    obj->throttle_force   = throttle;
    obj->steer_speed      = steer;
    obj->banking_strength = banking;
    obj->lateral_grip     = grip;
    obj->linear_drag      = drag;

    // The trampoline lock stays: it is part of the documented acquisition order
    // (jph_trampoline_lock -> shadow_lock -> step_sync), not just protection for
    // the proc table that is now gone.
    (self->physics.jph_trampoline_lock).lock();
    obj->listener = new ShipStepListener(obj);
    self->physics.physics_system.AddStepListener(obj->listener);
    (self->physics.jph_trampoline_lock).unlock();

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);
}

// --- INPUT (ULTRA FAST) ---

[[nodiscard]] PyObject * ShipSetInput(ShipObject *self, PyObject *const *args,
                                                        size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float fwd              = 0.0f;
    float right            = 0.0f;

    const ParseResult parsed =
        self->parsers->ShipInput.Parse(args, nargs, kwnames, fwd, right);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // No locks! Just update atomics.
    // The C listener running in the Jolt thread will pick these up automatically.
    (self->input_fwd).store(fwd, std::memory_order::relaxed);
    (self->input_right).store(right, std::memory_order::relaxed);

    Py_RETURN_NONE;
}

// --- CLEANUP ---
void ShipDealloc(ShipObject *self) {
    PyObject_GC_UnTrack(self);

    if (self->world && self->listener) {
        // Protect Jolt system call with trampoline lock
        (self->world->physics.jph_trampoline_lock).lock();
        self->world->physics.physics_system.RemoveStepListener(self->listener);
        delete self->listener;
        self->listener = nullptr;
        (self->world->physics.jph_trampoline_lock).unlock();
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
