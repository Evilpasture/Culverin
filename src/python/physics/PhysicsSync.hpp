#pragma once
#include "core/Threading.hpp"
#include "physics/PhysicsWorldBinding.hpp"

namespace CPH::Python {

// Cold path for the Block* helpers. Caller holds shadow_lock; it is
// released while parked on the step-sync condition (with the GIL dropped)
// and re-acquired before returning, mirroring the native wait loops.
template <typename Predicate>
static inline void WaitForPhysicsSync(PhysicsWorldObject *self, Predicate &&blocked) {
    self->physics.shadow_lock.unlock();

    Py_BEGIN_ALLOW_THREADS {
        std::unique_lock sync(self->physics.step_sync.mutex);
        self->physics.step_sync.cond.wait(sync, [&] { return !blocked(); });
    }
    Py_END_ALLOW_THREADS

    self->physics.shadow_lock.lock();
}

[[gnu::always_inline]]
static inline void BlockUntilNotStepping(PhysicsWorldObject *self) {
    if (self->physics.is_stepping.load(std::memory_order::relaxed)) {
        self->physics.waiting_threads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(self, [&] {
            return self->physics.is_stepping.load(std::memory_order::relaxed);
        });
        self->physics.waiting_threads.fetch_sub(1, std::memory_order::relaxed);
    }
}

[[gnu::always_inline]]
static inline void BlockUntilNotQuerying(PhysicsWorldObject *self) {
    if (self->physics.active_queries.load(std::memory_order::acquire) > 0) {
        WaitForPhysicsSync(self, [&] {
            return self->physics.active_queries.load(std::memory_order::relaxed) > 0;
        });
    }
}

static inline void BlockIfStepPending(PhysicsWorldObject *self) {
    if (self->physics.step_requested.load(std::memory_order::relaxed)) {
        self->physics.waiting_threads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(self, [&] {
            return self->physics.step_requested.load(std::memory_order::relaxed);
        });
        self->physics.waiting_threads.fetch_sub(1, std::memory_order::relaxed);
    }
}

[[gnu::always_inline]]
static inline void BlockUntilCanQuery(PhysicsWorldObject *self) {
    const auto blocked = [&] {
        return self->physics.is_stepping.load(std::memory_order::relaxed) ||
               self->physics.step_requested.load(std::memory_order::relaxed);
    };
    if (blocked()) {
        self->physics.waiting_threads.fetch_add(1, std::memory_order::relaxed);
        WaitForPhysicsSync(self, blocked);
        self->physics.waiting_threads.fetch_sub(1, std::memory_order::relaxed);
    }
}

} // namespace CPH::Python
