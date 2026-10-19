#pragma once
#include "core/Threading.h"
#include "physics/PhysicsWorldBinding.hpp"

namespace CPH::Python {
/**
 * INTERNAL HELPER: WaitForPhysicsSync
 * Encapsulates the heavy lifting of waiting on the condition variable.
 * This is the "Cold Path" logic that allows GIL release.
 */
static inline void WaitForPhysicsSync(PhysicsWorldObject *self,
                                           const CPH_ATOMIC(bool) * bool_cond,
                                           const CPH_ATOMIC(int) * int_cond, bool is_bool) {
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    Py_BEGIN_ALLOW_THREADS CPH_NATIVE_MUTEX_LOCK(self->physics.step_sync.mutex);

    if (is_bool) {
        while (atomic_load_explicit(bool_cond, memory_order_relaxed)) {
            CPH_NATIVE_COND_WAIT(self->physics.step_sync.cond, self->physics.step_sync.mutex);
        }
    } else {
        while (atomic_load_explicit(int_cond, memory_order_relaxed) > 0) {
            CPH_NATIVE_COND_WAIT(self->physics.step_sync.cond, self->physics.step_sync.mutex);
        }
    }

    CPH_NATIVE_MUTEX_UNLOCK(self->physics.step_sync.mutex);
    Py_END_ALLOW_THREADS

        CPH_SHADOW_LOCK(&self->physics.shadow_lock);
}

/* ============================================================================
 * PUBLIC SYNCHRONIZATION API
 * ============================================================================ */
[[gnu::always_inline]]
static inline void BlockUntilNotStepping(PhysicsWorldObject *self) {
    if (atomic_load_explicit(&self->physics.is_stepping, memory_order_relaxed)) {
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_add_explicit(&self->physics.waiting_threads, 1, memory_order_relaxed);
        // #endif

        while (atomic_load_explicit(&self->physics.is_stepping, memory_order_relaxed)) {
            WaitForPhysicsSync(self, &self->physics.is_stepping, nullptr, true);
        }
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_sub_explicit(&self->physics.waiting_threads, 1, memory_order_relaxed);
        // #endif
    }
}
[[gnu::always_inline]]
static inline void BlockUntilNotQuerying(PhysicsWorldObject *self) {
    // Note: Query check uses 'acquire' to ensure visibility of query completion
    while (atomic_load_explicit(&self->physics.active_queries, memory_order_acquire) > 0) {
        WaitForPhysicsSync(self, nullptr, &self->physics.active_queries, false);
    }
}

static inline void BlockIfStepPending(PhysicsWorldObject *self) {
    if (atomic_load_explicit(&self->physics.step_requested, memory_order_relaxed)) {
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_add_explicit(&self->physics.waiting_threads, 1, memory_order_relaxed);
        // #endif
        while (atomic_load_explicit(&self->physics.step_requested, memory_order_relaxed)) {
            WaitForPhysicsSync(self, &self->physics.step_requested, nullptr, true);
        }
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_sub_explicit(&self->physics.waiting_threads, 1, memory_order_relaxed);
        // #endif
    }
}
[[gnu::always_inline]]
static inline void BlockUntilCanQuery(PhysicsWorldObject *self) {
    if (atomic_load_explicit(&self->physics.is_stepping, memory_order_relaxed) ||
        atomic_load_explicit(&self->physics.step_requested, memory_order_relaxed)) {

        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_add_explicit(&self->physics.waiting_threads, 1, memory_order_relaxed);
        // #endif

        while (atomic_load_explicit(&self->physics.is_stepping, memory_order_relaxed) ||
               atomic_load_explicit(&self->physics.step_requested, memory_order_relaxed)) {

            // This is a special composite case; we don't use the simple helper loop
            CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
            Py_BEGIN_ALLOW_THREADS CPH_NATIVE_MUTEX_LOCK(self->physics.step_sync.mutex);
            while (atomic_load_explicit(&self->physics.is_stepping, memory_order_relaxed) ||
                   atomic_load_explicit(&self->physics.step_requested, memory_order_relaxed)) {
                CPH_NATIVE_COND_WAIT(self->physics.step_sync.cond, self->physics.step_sync.mutex);
            }
            CPH_NATIVE_MUTEX_UNLOCK(self->physics.step_sync.mutex);
            Py_END_ALLOW_THREADS CPH_SHADOW_LOCK(&self->physics.shadow_lock);
        }
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_sub_explicit(&self->physics.waiting_threads, 1, memory_order_relaxed);
        // #endif
    }
}


} // namespace CPH::Python
