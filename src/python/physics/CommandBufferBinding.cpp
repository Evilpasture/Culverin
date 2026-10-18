#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdatomic.h>
#include <stddef.h>

#include "core/Threading.h"
#include "physics/CommandBuffer.h"
#include "physics/PhysicsSync.h"
#include "physics/PhysicsWorldBinding.h"

void CPH_SyncAndFlushInternal(CPH_PhysicsWorldObject *self) {
    CPH_BLOCK_UNTIL_NOT_STEPPING(self);
    CPH_BLOCK_UNTIL_NOT_QUERYING(self);

    if (self->physics.command_count == 0) {
        return;
    }

    atomic_store_explicit(&self->physics.is_stepping, true, memory_order_relaxed);

    PhysicsCommand *captured_queue = self->physics.command_queue;
    size_t captured_count          = self->physics.command_count;

    self->physics.command_queue       = self->physics.command_queue_spare;
    self->physics.command_queue_spare = captured_queue;
    self->physics.command_count       = 0;

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    Py_BEGIN_ALLOW_THREADS CPH_NATIVE_MUTEX_LOCK(self->physics.jph_trampoline_lock);

    CPH_FlushCommandsInternal(&self->physics, captured_queue, captured_count);

    CPH_NATIVE_MUTEX_UNLOCK(self->physics.jph_trampoline_lock);
    Py_END_ALLOW_THREADS

        CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    atomic_store_explicit(&self->physics.is_stepping, false, memory_order_release);
    CPH_NATIVE_MUTEX_LOCK(self->physics.step_sync.mutex);
    CPH_NATIVE_COND_BROADCAST(self->physics.step_sync.cond);
    CPH_NATIVE_MUTEX_UNLOCK(self->physics.step_sync.mutex);
}
