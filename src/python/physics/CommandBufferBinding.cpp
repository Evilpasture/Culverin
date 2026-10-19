#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <atomic>
#include <cstddef>

#include "core/Threading.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/PhysicsWorld.hpp"
#include "physics/PhysicsSync.hpp"
#include "physics/PhysicsWorldBinding.hpp"

namespace CPH::Python {

void SyncAndFlushInternal(PhysicsWorldObject *self) {
    BlockUntilNotStepping(self);
    BlockUntilNotQuerying(self);

    if (self->physics.command_count == 0) {
        return;
    }

    self->physics.is_stepping.store(true, std::memory_order::relaxed);
    const auto detached = CPH_PhysicsWorldDetachCommands(&self->physics);
    self->physics.shadow_lock.unlock();

    Py_BEGIN_ALLOW_THREADS {
        const std::scoped_lock trampoline(self->physics.jph_trampoline_lock);
        CPH_FlushCommandsInternal(&self->physics, detached);
    }
    Py_END_ALLOW_THREADS

    self->physics.shadow_lock.lock();
    self->physics.is_stepping.store(false, std::memory_order::release);
    CPH_NotifyStepSync(&self->physics);
}

} // namespace CPH::Python
