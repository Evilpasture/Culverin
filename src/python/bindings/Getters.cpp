#include "bindings/Getters.hpp"
#include "bindings/ModuleState.hpp"
#include "ecs/Ecs.hpp"
#include "physics/PhysicsSync.hpp"
#include <Python.h>
#include <utility>

namespace CPH::Python {

static Py_ssize_t BufferProxy_length(BufferProxyObject *self) {
    // Return the cached shape. Since we update shape[0] in the
    // constructor and in getbuffer, this is always accurate
    // to the count at the time of the last sync/query.
    return self->shape[0];
}

[[nodiscard]] static int BufferProxy_traverse(BufferProxyObject *self, visitproc visit,
                                               void *arg) {
    Py_VISIT(self->owner);
    return 0;
}

// Inside BufferProxy_clear
[[nodiscard]] static int BufferProxy_clear(BufferProxyObject *self) {
    ModuleState *st = GetModuleState(PyType_GetModule(Py_TYPE(self)));
    if (self->owner) {
        // We know that both PhysicsWorld and Registry have the export count at a fixed offset
        // or we can switch based on type.
        if (PyObject_TypeCheck(self->owner,
                               reinterpret_cast<PyTypeObject *>(st->physicsWorldType))) {
            ((reinterpret_cast<PhysicsWorldObject *>(self->owner))
                                           ->physics.viewExportCount).fetch_sub(1, std::memory_order::relaxed);
        } else {
            reinterpret_cast<RegistryObject *>(self->owner)->registry->OnViewReleased();
        }
        Py_CLEAR(self->owner);
    }
    return 0;
}

static void BufferProxy_dealloc(BufferProxyObject *self) {
    PyObject_GC_UnTrack(self);
    [[maybe_unused]] auto cleared = BufferProxy_clear(self);
    // For heap types, we must release our reference to the type itself
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free(reinterpret_cast<PyObject *>(self));
    Py_DECREF(tp);
}

[[nodiscard]] static int BufferProxy_getbuffer(BufferProxyObject *self, Py_buffer *view,
                                                [[maybe_unused]] int flags) {
    // Determine the lock and target based on the buffer type category
    void *target_ptr = nullptr;

    if (self->bufType < ProxyBufferType::EcsData) {
        /* --- CATEGORY: Physics World Buffers --- */
        PhysicsWorldObject *world = reinterpret_cast<PhysicsWorldObject *>(self->owner);

        (world->physics.shadowLock).lock();
        BlockUntilNotStepping(world);

        switch (self->bufType) {
        case ProxyBufferType::Positions:
            target_ptr = world->physics.bodies.positions.data();
            break;
        case ProxyBufferType::Rotations:
            target_ptr = world->physics.bodies.rotations.data();
            break;
        case ProxyBufferType::LinearVelocities:
            target_ptr = world->physics.bodies.linearVelocities.data();
            break;
        case ProxyBufferType::AngularVelocities:
            target_ptr = world->physics.bodies.angularVelocities.data();
            break;
        case ProxyBufferType::UserData:
            target_ptr = world->physics.bodies.userData.data();
            break;
        case ProxyBufferType::Dynamic:
            target_ptr = self->dynamicPtr;
            break; // Soft Body Verts
        default:
            std::unreachable();
        }

        if (!target_ptr) {
            (world->physics.shadowLock).unlock();
            PyErr_SetString(PyExc_RuntimeError, "Physics buffer not allocated");
            return -1;
        }

        // Auto-update shape based on current body count for non-dynamic buffers
        if (self->bufType != ProxyBufferType::Dynamic) {
            size_t count   = (world->physics.count).load(std::memory_order::acquire);
            self->shape[0] = static_cast<Py_ssize_t>(count * self->stride);
        }

        (world->physics.shadowLock).unlock();

    } else {
        /* --- CATEGORY: ECS Registry Buffers --- */
        // For ECS, the pointer was stored in dynamic_ptr when the proxy was
        // built. Nothing mutable is read here: the registry refuses to grow a
        // component set while a view is exported, so the address and the shape
        // stay valid for the proxy's whole life.
        target_ptr = self->dynamicPtr;

        if (!target_ptr) {
            // A zero-length view still needs a non-null address for the buffer
            // protocol; its size is zero, so this sentinel is never dereferenced.
            static char empty_buffer = '\0';
            target_ptr               = &empty_buffer;
        }

        // ECS shape is pinned at creation because we block reallocs while views
        // are held, so self->shape[0] needs no refresh here.
    }

    // --- Standard Buffer Protocol Population ---
    self->strides[0] = static_cast<Py_ssize_t>(self->itemsize);

    view->buf = target_ptr;
    view->obj = reinterpret_cast<PyObject *>(self);
    Py_INCREF(self);

    view->len        = self->shape[0] * self->strides[0];
    view->readonly   = (self->bufType == ProxyBufferType::EcsEntities) ? 1 : 0; // Entities are read-only
    view->itemsize   = static_cast<Py_ssize_t>(self->itemsize);
    view->format     = const_cast<char *>(self->format);
    view->ndim       = 1;
    view->shape      = self->shape;
    view->strides    = self->strides;
    view->suboffsets = nullptr;
    view->internal   = nullptr;

    return 0;
}

static void BufferProxy_releasebuffer([[maybe_unused]] BufferProxyObject *self,
                                                  [[maybe_unused]] Py_buffer *view) {
// Let NumPy hold the world buffer permanently.
#if defined(STRICT_THREAD_SAFETY)
    PhysicsWorldObject *world = self->world;

    // 4. Mark query finished
    // Use 'release' barrier to ensure Numpy reads are finished before Stepper writes
    if ((world->physics.activeQueries).fetch_sub(1, std::memory_order::release) == 1) {

        // 5. If we were the last ones, wake up the Stepper!
        // We use the internal sync mutex/cond here
        (world->physics.stepSync.mutex).lock();
        (world->physics.stepSync.cond).notify_all();
        (world->physics.stepSync.mutex).unlock();
    }
#endif
}

// --- Heap Type Specification ---
PyType_Slot BufferProxySlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(BufferProxy_dealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(BufferProxy_traverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(BufferProxy_clear)},
    {.slot = Py_bf_getbuffer, .pfunc = reinterpret_cast<void *>(BufferProxy_getbuffer)},
    {.slot = Py_bf_releasebuffer, .pfunc = reinterpret_cast<void *>(BufferProxy_releasebuffer)},
    {.slot = Py_sq_length, .pfunc = reinterpret_cast<void *>(BufferProxy_length)},
    {}};

PyType_Spec BufferProxySpec = {.name      = "culverin.BufferProxy",
                                .basicsize = sizeof(BufferProxyObject),
                                .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
                                .slots     = BufferProxySlots};

// --- Updated MakeProxy helper ---
static PyObject *MakeProxy(PhysicsWorldObject *self, ProxyBufferType type, const char *format,
                            size_t itemsize, int stride) {
    ModuleState *st = GetModuleState(PyType_GetModule(Py_TYPE(self)));

    // Allocation
    BufferProxyObject *proxy = PyObject_GC_New(
        BufferProxyObject, reinterpret_cast<PyTypeObject *>(st->bufferProxyType));
    if (!proxy) {
        return nullptr;
    }

    // Initialization
    proxy->owner = reinterpret_cast<PyObject *>(self);
    Py_INCREF(self);
    proxy->bufType = type;
    proxy->format   = format;
    proxy->itemsize = itemsize;
    proxy->stride   = stride;

    // INITIALIZE SHAPE IMMEDIATELY
    // This allows len(proxy) to work even before a buffer is requested
    size_t count      = (self->physics.count).load(std::memory_order::acquire);
    proxy->shape[0]   = static_cast<Py_ssize_t>(count * stride);
    proxy->strides[0] = static_cast<Py_ssize_t>(itemsize);

    (self->physics.viewExportCount).fetch_add(1, std::memory_order::relaxed);

    PyObject_GC_Track(proxy);
    return reinterpret_cast<PyObject *>(proxy);
}

// --- Advanced getters for lock and thread management ---
[[nodiscard]] PyObject * WorldGetIsStepPending(PhysicsWorldObject *self,
                                                    [[maybe_unused]] void *closure) {
    if ((self->physics.stepRequested).load(std::memory_order::acquire)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/* --- Shadow Buffer Getters --- */

[[nodiscard]] PyObject * WorldGetPositions(PhysicsWorldObject *self,
                                                [[maybe_unused]] void *c) {
    // Positions are Stride 4 (X, Y, Z, W)
    return MakeProxy(self, ProxyBufferType::Positions, CPH::kRealFormatString, sizeof(JPH::Real), 4);
}

[[nodiscard]] PyObject * WorldGetRotations(PhysicsWorldObject *self,
                                                [[maybe_unused]] void *c) {
    // Rotations are Stride 4 (X, Y, Z, W)
    return MakeProxy(self, ProxyBufferType::Rotations, "f", sizeof(float), 4);
}

[[nodiscard]] PyObject * WorldGetVelocities(PhysicsWorldObject *self,
                                                 [[maybe_unused]] void *c) {
    // Velocities are Stride 4 (X, Y, Z, Pad)
    return MakeProxy(self, ProxyBufferType::LinearVelocities, "f", sizeof(float), 4);
}

[[nodiscard]] PyObject * WorldGetAngularVelocities(PhysicsWorldObject *self,
                                                        [[maybe_unused]] void *c) {
    return MakeProxy(self, ProxyBufferType::AngularVelocities, "f", sizeof(float), 4);
}

/* --- Mutable Metadata Getters (Hardened with Locks) --- */

[[nodiscard]] PyObject * WorldGetCount(PhysicsWorldObject *self,
                                            [[maybe_unused]] void *c) {
    // TSan Fix: Lock-free atomic read.
    // Acquire ensures we see the final state of any bodies recently added/removed.
    size_t val = (self->physics.count).load(std::memory_order::acquire);
    return PyLong_FromSize_t(val);
}

[[nodiscard]] PyObject * WorldGetTime(PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    (self->physics.shadowLock).lock();
    double val = self->physics.time;
    (self->physics.shadowLock).unlock();
    return PyFloat_FromDouble(val);
}

[[nodiscard]] PyObject * WorldGetUserDataBuffer(PhysicsWorldObject *self,
                                                     [[maybe_unused]] void *c) {
    // User data is Stride 1 (One uint64 per body)
    return MakeProxy(self, ProxyBufferType::UserData, "Q", sizeof(uint64_t), 1);
}

[[nodiscard]] PyObject * WorldGetShapeCount(PhysicsWorldObject *self,
                                                 [[maybe_unused]] void *closure) {
    // Protected because resize() could move shape_cache
    (self->physics.shadowLock).lock();
    size_t count = self->physics.shapeCacheCount;
    (self->physics.shadowLock).unlock();
    return PyLong_FromSize_t(count);
}

[[nodiscard]] PyObject * WorldGetMaxBodies(PhysicsWorldObject *self,
                                                [[maybe_unused]] void *closure) {
    return PyLong_FromUnsignedLong(self->physics.maxJoltBodies);
}

[[nodiscard]] PyObject * WorldGetRemainingCapacity(PhysicsWorldObject *self,
                                                        [[maybe_unused]] void *closure) {
    // TSan Fix: Lock-free calculation using atomic count.
    size_t current = (self->physics.count).load(std::memory_order::acquire);

    // max_jolt_bodies is immutable after world initialization
    size_t limit = self->physics.maxJoltBodies;
    size_t rem   = (current >= limit) ? 0 : (limit - current);

    return PyLong_FromSize_t(rem);
}

} // namespace CPH::Python
