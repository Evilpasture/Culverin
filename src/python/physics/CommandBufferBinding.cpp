#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <atomic>
#include <cstddef>

#include "bindings/ThreadState.hpp"
#include "core/Threading.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/PhysicsWorld.hpp"
#include "physics/PhysicsSync.hpp"
#include "physics/PhysicsWorldBinding.hpp"

namespace CPH::Python {

void SyncAndFlushInternal(PhysicsWorldObject *self) {
    BlockUntilNotStepping(self);
    BlockUntilNotQuerying(self);

    if (self->physics.mailbox.Empty()) {
        return;
    }

    self->physics.isStepping.store(true, std::memory_order::relaxed);
    const auto detached = self->physics.DetachCommands();
    self->physics.shadowLock.unlock();

    {
        const AllowThreads allow;
        const std::scoped_lock trampoline(self->physics.jphTrampolineLock);
        self->physics.FlushCommands(detached);
    }

    self->physics.shadowLock.lock();
    self->physics.isStepping.store(false, std::memory_order::release);
    self->physics.NotifyStepSync();
}

} // namespace CPH::Python
