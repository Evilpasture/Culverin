#pragma once
#include "culverin_physics_world.h"
#include "culverin_threading.h"
/**
 * INTERNAL HELPER: internal_sync_wait_loop
 * Encapsulates the heavy lifting of waiting on the condition variable.
 * This is the "Cold Path" logic that allows GIL release.
 */
static inline void internal_sync_wait_loop(PhysicsWorldObject *self,
                                           const CULV_ATOMIC(bool) * bool_cond,
                                           const CULV_ATOMIC(int) * int_cond, bool is_bool) {
    SHADOW_UNLOCK(&self->shadow_lock);

    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(self->step_sync.mutex);

    if (is_bool) {
        while (atomic_load_explicit(bool_cond, memory_order_relaxed)) {
            NATIVE_COND_WAIT(self->step_sync.cond, self->step_sync.mutex);
        }
    } else {
        while (atomic_load_explicit(int_cond, memory_order_relaxed) > 0) {
            NATIVE_COND_WAIT(self->step_sync.cond, self->step_sync.mutex);
        }
    }

    NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
    Py_END_ALLOW_THREADS

        SHADOW_LOCK(&self->shadow_lock);
}

/* ============================================================================
 * PUBLIC SYNCHRONIZATION API
 * ============================================================================ */

static inline void internal_block_until_not_stepping(PhysicsWorldObject *self) {
    if (atomic_load_explicit(&self->is_stepping, memory_order_relaxed)) {
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_add_explicit(&self->waiting_threads, 1, memory_order_relaxed);
        // #endif

        while (atomic_load_explicit(&self->is_stepping, memory_order_relaxed)) {
            internal_sync_wait_loop(self, &self->is_stepping, NULL, true);
        }
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_sub_explicit(&self->waiting_threads, 1, memory_order_relaxed);
        // #endif
    }
}

static inline void internal_block_until_not_querying(PhysicsWorldObject *self) {
    // Note: Query check uses 'acquire' to ensure visibility of query completion
    while (atomic_load_explicit(&self->active_queries, memory_order_acquire) > 0) {
        internal_sync_wait_loop(self, NULL, &self->active_queries, false);
    }
}

static inline void internal_block_if_step_pending(PhysicsWorldObject *self) {
    if (atomic_load_explicit(&self->step_requested, memory_order_relaxed)) {
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_add_explicit(&self->waiting_threads, 1, memory_order_relaxed);
        // #endif
        while (atomic_load_explicit(&self->step_requested, memory_order_relaxed)) {
            internal_sync_wait_loop(self, &self->step_requested, NULL, true);
        }
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_sub_explicit(&self->waiting_threads, 1, memory_order_relaxed);
        // #endif
    }
}

static inline void internal_block_until_can_query(PhysicsWorldObject *self) {
    if (atomic_load_explicit(&self->is_stepping, memory_order_relaxed) ||
        atomic_load_explicit(&self->step_requested, memory_order_relaxed)) {

        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_add_explicit(&self->waiting_threads, 1, memory_order_relaxed);
        // #endif

        while (atomic_load_explicit(&self->is_stepping, memory_order_relaxed) ||
               atomic_load_explicit(&self->step_requested, memory_order_relaxed)) {

            // This is a special composite case; we don't use the simple helper loop
            SHADOW_UNLOCK(&self->shadow_lock);
            Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(self->step_sync.mutex);
            while (atomic_load_explicit(&self->is_stepping, memory_order_relaxed) ||
                   atomic_load_explicit(&self->step_requested, memory_order_relaxed)) {
                NATIVE_COND_WAIT(self->step_sync.cond, self->step_sync.mutex);
            }
            NATIVE_MUTEX_UNLOCK(self->step_sync.mutex);
            Py_END_ALLOW_THREADS SHADOW_LOCK(&self->shadow_lock);
        }
        // #if !defined(Py_GIL_DISABLED)
        atomic_fetch_sub_explicit(&self->waiting_threads, 1, memory_order_relaxed);
        // #endif
    }
}

/* API MACROS: Forward to the inlined functions */
#define BLOCK_UNTIL_NOT_STEPPING(self) internal_block_until_not_stepping(self)
#define BLOCK_UNTIL_NOT_QUERYING(self) internal_block_until_not_querying(self)
#define BLOCK_IF_STEP_PENDING(self) internal_block_if_step_pending(self)
#define BLOCK_UNTIL_CAN_QUERY(self) internal_block_until_can_query(self)