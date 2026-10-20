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
    const auto detached = self->physics.DetachCommands();
    self->physics.shadow_lock.unlock();

    Py_BEGIN_ALLOW_THREADS {
        const std::scoped_lock trampoline(self->physics.jph_trampoline_lock);
        self->physics.FlushCommands(detached);
    }
    Py_END_ALLOW_THREADS

    self->physics.shadow_lock.lock();
    self->physics.is_stepping.store(false, std::memory_order::release);
    self->physics.NotifyStepSync();
}

} // namespace CPH::Python
