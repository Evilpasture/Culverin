#pragma once
#include "bindings/ThreadState.hpp"
#include "core/Threading.hpp"
#include "physics/PhysicsWorldBinding.hpp"

#include <utility>

namespace CPH::Python {

// Cold path for the Block* helpers. Caller holds shadow_lock; it is
// released while parked on the step-sync condition (with the GIL dropped)
// and re-acquired before returning, mirroring the native wait loops.
//
// The wait itself only ever touches native state, so it is written against
// `PhysicsWorld` and the PyObject forms below forward to it. That is what lets
// a buffer proxy -- which is native-first by design -- park here without
// knowing it is being driven from Python.
template <typename Predicate>
static inline void WaitForPhysicsSync(CPH::PhysicsWorld &physics, Predicate &&blocked) {
    physics.shadowLock.unlock();

    {
        const AllowThreads allow;
        std::unique_lock sync(physics.stepSync.mutex);
        physics.stepSync.cond.wait(sync, [&] { return !blocked(); });
    }

    physics.shadowLock.lock();
}

[[gnu::always_inline]]
static inline void BlockUntilNotStepping(CPH::PhysicsWorld &physics) {
    if (physics.isStepping.load(std::memory_order::relaxed)) {
        physics.waitingThreads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(physics, [&] { return physics.isStepping.load(std::memory_order::relaxed); });
        physics.waitingThreads.fetch_sub(1, std::memory_order::relaxed);
    }
}

template <typename Predicate>
static inline void WaitForPhysicsSync(PhysicsWorldObject *self, Predicate &&blocked) {
    WaitForPhysicsSync(self->physics, std::forward<Predicate>(blocked));
}

[[gnu::always_inline]]
static inline void BlockUntilNotStepping(PhysicsWorldObject *self) {
    BlockUntilNotStepping(self->physics);
}

[[gnu::always_inline]]
static inline void BlockUntilNotQuerying(PhysicsWorldObject *self) {
    if (self->physics.activeQueries.load(std::memory_order::acquire) > 0) {
        WaitForPhysicsSync(self, [&] {
            return self->physics.activeQueries.load(std::memory_order::relaxed) > 0;
        });
    }
}

static inline void BlockIfStepPending(PhysicsWorldObject *self) {
    if (self->physics.stepRequested.load(std::memory_order::relaxed)) {
        self->physics.waitingThreads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(self, [&] {
            return self->physics.stepRequested.load(std::memory_order::relaxed);
        });
        self->physics.waitingThreads.fetch_sub(1, std::memory_order::relaxed);
    }
}

[[gnu::always_inline]]
static inline void BlockUntilCanQuery(PhysicsWorldObject *self) {
    const auto blocked = [&] {
        return self->physics.isStepping.load(std::memory_order::relaxed) ||
               self->physics.stepRequested.load(std::memory_order::relaxed);
    };
    if (blocked()) {
        self->physics.waitingThreads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(self, blocked);
        self->physics.waitingThreads.fetch_sub(1, std::memory_order::relaxed);
    }
}

} // namespace CPH::Python
