#include "bindings/Getters.hpp"
#include "bindings/ModuleState.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include <Python.h>
#include <expected>
#include <utility>

namespace CPH::Python {

static Py_ssize_t BufferProxy_length(BufferProxyObject *self) {
    // The element count the proxy last saw: the body count it was built with,
    // refreshed by every successful acquisition.
    return static_cast<Py_ssize_t>(self->Proxy().Length());
}

[[nodiscard]] static int BufferProxy_traverse(BufferProxyObject *self, visitproc visit,
                                               void *arg) {
    Py_VISIT(self->owner);
    return 0;
}

[[nodiscard]] static int BufferProxy_clear(BufferProxyObject *self) {
    if (self->owner) {
        // Give the export back to whichever domain issued it. The adapter used
        // to answer that with a PyObject_TypeCheck against the world type and a
        // fall-through to the registry; the proxy knows its own domain.
        self->Proxy().ReleaseExport();
        Py_CLEAR(self->owner);
    }
    return 0;
}

static void BufferProxy_dealloc(BufferProxyObject *self) {
    PyObject_GC_UnTrack(self);
    [[maybe_unused]] auto cleared = BufferProxy_clear(self);
    // PyObject_GC_New handed us raw storage; close the C++ object explicitly.
    self->Proxy().~BufferProxy();
    // For heap types, we must release our reference to the type itself
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free(reinterpret_cast<PyObject *>(self));
    Py_DECREF(tp);
}

[[nodiscard]] static int BufferProxy_getbuffer(BufferProxyObject *self, Py_buffer *view,
                                                [[maybe_unused]] int flags) {
    // Everything domain-shaped is the proxy's: which array, whether the address
    // is re-read, what lock it needs, how wide an item is, whether the consumer
    // may write. This function's whole job is to move those facts into the
    // struct CPython asked for.
    const std::expected<BufferDescriptor, const char *> described = self->Proxy().Describe();
    if (!described) {
        PyErr_SetString(PyExc_RuntimeError, described.error());
        return -1;
    }
    const BufferDescriptor &buffer = *described;

    // An empty view still needs an address the buffer protocol considers valid;
    // nothing is ever read through it, because the length is zero.
    static char empty_buffer = '\0';
    void       *target_ptr   = buffer.data != nullptr ? buffer.data : &empty_buffer;

    self->shape[0]   = static_cast<Py_ssize_t>(self->Proxy().Length());
    self->strides[0] = static_cast<Py_ssize_t>(buffer.itemSize);

    view->buf        = target_ptr;
    view->obj        = reinterpret_cast<PyObject *>(self);
    Py_INCREF(self);
    view->len        = static_cast<Py_ssize_t>(buffer.bytes);
    view->readonly   = buffer.readonly ? 1 : 0;
    view->itemsize   = static_cast<Py_ssize_t>(buffer.itemSize);
    view->format     = const_cast<char *>(buffer.format);
    view->ndim       = 1;
    view->shape      = self->shape;
    view->strides    = self->strides;
    view->suboffsets = nullptr;
    view->internal   = nullptr;

    return 0;
}

static void BufferProxy_releasebuffer(BufferProxyObject *self, [[maybe_unused]] Py_buffer *view) {
    // Let NumPy hold the world buffer permanently: the release hook is the
    // proxy's too, and for the registry it is a no-op.
    self->Proxy().OnBufferReleased();
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

// --- Proxy construction ---

PyObject *NewBufferProxy(PyObject *owner, BufferProxyPayload payload) {
    ModuleState *st = GetModuleState(PyType_GetModule(Py_TYPE(owner)));

    BufferProxyObject *proxy =
        PyObject_GC_New(BufferProxyObject, reinterpret_cast<PyTypeObject *>(st->bufferProxyType));
    if (!proxy) {
        return nullptr;
    }

    // The storage PyObject_GC_New returned is zeroed but has never held a
    // C++ object, so this is where the variant comes into existence.
    new (proxy->proxyStorage) BufferProxy(std::move(payload));
    proxy->owner = owner;
    Py_INCREF(owner);

    // The length is already cached, so len(proxy) works before anyone asks for
    // a buffer; the export is counted from here until clear(). shape/strides
    // are provisional -- Py_sq_length reads Length(), and the exported Py_buffer
    // can only point at them after getbuffer has filled them in.
    proxy->shape[0]   = static_cast<Py_ssize_t>(proxy->Proxy().Length());
    proxy->strides[0] = 1;
    proxy->Proxy().AcquireExport();

    PyObject_GC_Track(proxy);
    return reinterpret_cast<PyObject *>(proxy);
}

static PyObject *MakeProxy(PhysicsWorldObject *self, WorldChannel channel) {
    return NewBufferProxy(reinterpret_cast<PyObject *>(self), WorldBuffer(self->physics, channel));
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
    return MakeProxy(self, WorldChannel::Positions);
}

[[nodiscard]] PyObject * WorldGetRotations(PhysicsWorldObject *self,
                                                [[maybe_unused]] void *c) {
    // Rotations are Stride 4 (X, Y, Z, W)
    return MakeProxy(self, WorldChannel::Rotations);
}

[[nodiscard]] PyObject * WorldGetVelocities(PhysicsWorldObject *self,
                                                 [[maybe_unused]] void *c) {
    // Velocities are Stride 4 (X, Y, Z, Pad)
    return MakeProxy(self, WorldChannel::LinearVelocities);
}

[[nodiscard]] PyObject * WorldGetAngularVelocities(PhysicsWorldObject *self,
                                                        [[maybe_unused]] void *c) {
    return MakeProxy(self, WorldChannel::AngularVelocities);
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
    return MakeProxy(self, WorldChannel::UserData);
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
