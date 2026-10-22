#pragma once
#include "core/Threading.hpp"
#include "physics/PhysicsWorldBinding.hpp"

namespace CPH::Python {

// Cold path for the Block* helpers. Caller holds shadow_lock; it is
// released while parked on the step-sync condition (with the GIL dropped)
// and re-acquired before returning, mirroring the native wait loops.
template <typename Predicate>
static inline void WaitForPhysicsSync(PhysicsWorldObject *self, Predicate &&blocked) {
    self->physics.shadowLock.unlock();

    Py_BEGIN_ALLOW_THREADS {
        std::unique_lock sync(self->physics.stepSync.mutex);
        self->physics.stepSync.cond.wait(sync, [&] { return !blocked(); });
    }
    Py_END_ALLOW_THREADS

    self->physics.shadowLock.lock();
}

[[gnu::always_inline]]
static inline void BlockUntilNotStepping(PhysicsWorldObject *self) {
    if (self->physics.isStepping.load(std::memory_order::relaxed)) {
        self->physics.waitingThreads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(self, [&] {
            return self->physics.isStepping.load(std::memory_order::relaxed);
        });
        self->physics.waitingThreads.fetch_sub(1, std::memory_order::relaxed);
    }
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
