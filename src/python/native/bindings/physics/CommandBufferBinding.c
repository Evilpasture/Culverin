#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdatomic.h>
#include <stddef.h>

#include "bindings/physics/PhysicsWorld.h"
#include "physics/CommandBuffer.h"
#include "bindings/physics/PhysicsSync.h"
#include "core/Threading.h"

void sync_and_flush_internal(PhysicsWorldObject *self) {
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    if (self->physics.command_count == 0) {
        return;
    }

    atomic_store_explicit(&self->physics.is_stepping, true, memory_order_relaxed);

    PhysicsCommand *captured_queue = self->physics.command_queue;
    size_t captured_count          = self->physics.command_count;

    self->physics.command_queue       = self->physics.command_queue_spare;
    self->physics.command_queue_spare = captured_queue;
    self->physics.command_count       = 0;

    SHADOW_UNLOCK(&self->physics.shadow_lock);

    Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK(self->physics.jph_trampoline_lock);

    flush_commands_internal(&self->physics, captured_queue, captured_count);

    NATIVE_MUTEX_UNLOCK(self->physics.jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        SHADOW_LOCK(&self->physics.shadow_lock);

    atomic_store_explicit(&self->physics.is_stepping, false, memory_order_release);
    NATIVE_MUTEX_LOCK(self->physics.step_sync.mutex);
    NATIVE_COND_BROADCAST(self->physics.step_sync.cond);
    NATIVE_MUTEX_UNLOCK(self->physics.step_sync.mutex);
}
