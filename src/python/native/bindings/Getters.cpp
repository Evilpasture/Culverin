#include "bindings/Getters.h"
#include "ecs/Ecs.h"
#include "bindings/ModuleState.h"
#include "bindings/physics/PhysicsSync.h"
#include <Python.h>

static Py_ssize_t BufferProxy_length(CPH_BufferProxyObject *self) {
    // Return the cached shape. Since we update shape[0] in the
    // constructor and in getbuffer, this is always accurate
    // to the count at the time of the last sync/query.
    return self->shape[0];
}

PyType_DeclareSlot_Status BufferProxy_traverse(CPH_BufferProxyObject *self, visitproc visit,
                                               void *arg) {
    Py_VISIT(self->owner);
    return 0;
}

// Inside BufferProxy_clear
PyType_DeclareSlot_Status BufferProxy_clear(CPH_BufferProxyObject *self) {
    CPH_ModuleState *st = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    if (self->owner) {
        // We know that both PhysicsWorld and Registry have the export count at a fixed offset
        // or we can switch based on type.
        if (PyObject_TypeCheck(self->owner, (PyTypeObject *)st->PhysicsWorldType)) {
            atomic_fetch_sub_explicit(&((CPH_PhysicsWorldObject *)self->owner)->physics.view_export_count, 1,
                                      memory_order_relaxed);
        } else {
            atomic_fetch_sub_explicit(&((CPH_RegistryObject *)self->owner)->view_export_count, 1,
                                      memory_order_relaxed);
        }
        Py_CLEAR(self->owner);
    }
    return 0;
}

PyType_DeclareSlot_Void BufferProxy_dealloc(CPH_BufferProxyObject *self) {
    PyObject_GC_UnTrack(self);
    [[maybe_unused]] auto cleared = BufferProxy_clear(self);
    // For heap types, we must release our reference to the type itself
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free((PyObject *)self);
    Py_DECREF(tp);
}

PyType_DeclareSlot_Status BufferProxy_getbuffer(CPH_BufferProxyObject *self, Py_buffer *view,
                                                [[maybe_unused]] int flags) {
    // Determine the lock and target based on the buffer type category
    void *target_ptr = nullptr;

    if (self->buf_type < PROXY_ECS_DATA) {
        /* --- CATEGORY: Physics World Buffers --- */
        CPH_PhysicsWorldObject *world = (CPH_PhysicsWorldObject *)self->owner;

        CPH_SHADOW_LOCK(&world->physics.shadow_lock);
        CPH_BLOCK_UNTIL_NOT_STEPPING(world);

        switch (self->buf_type) {
        case PROXY_POSITIONS:
            target_ptr = world->physics.positions;
            break;
        case PROXY_ROTATIONS:
            target_ptr = world->physics.rotations;
            break;
        case PROXY_LINEAR_VELOCITIES:
            target_ptr = world->physics.linear_velocities;
            break;
        case PROXY_ANGULAR_VELOCITIES:
            target_ptr = world->physics.angular_velocities;
            break;
        case PROXY_USER_DATA:
            target_ptr = world->physics.user_data;
            break;
        case PROXY_DYNAMIC:
            target_ptr = self->dynamic_ptr;
            break; // Soft Body Verts
        default:
            CPH_Unreachable();
        }

        if (!target_ptr) {
            CPH_SHADOW_UNLOCK(&world->physics.shadow_lock);
            PyErr_SetString(PyExc_RuntimeError, "Physics buffer not allocated");
            return -1;
        }

        // Auto-update shape based on current body count for non-dynamic buffers
        if (self->buf_type != PROXY_DYNAMIC) {
            size_t count   = atomic_load_explicit(&world->physics.count, memory_order_acquire);
            self->shape[0] = (Py_ssize_t)(count * self->stride);
        }

        CPH_SHADOW_UNLOCK(&world->physics.shadow_lock);

    } else {
        /* --- CATEGORY: ECS Registry Buffers --- */
        CPH_RegistryObject *reg = (CPH_RegistryObject *)self->owner;

        CPH_SHADOW_LOCK(&reg->ecs_lock);

        // For ECS, the pointer was stored in dynamic_ptr during make_ecs_proxy
        target_ptr = self->dynamic_ptr;

        if (!target_ptr) {
            // This can happen if get_view was called on an empty component
            // We allow it to return an empty buffer rather than erroring
            target_ptr = (void *)"";
        }

        // ECS shape is pinned at creation because we block reallocs while views are held
        // No need to update self->shape[0] here.

        CPH_SHADOW_UNLOCK(&reg->ecs_lock);
    }

    // --- Standard Buffer Protocol Population ---
    self->strides[0] = (Py_ssize_t)self->itemsize;

    view->buf = target_ptr;
    view->obj = (PyObject *)self;
    Py_INCREF(self);

    view->len        = self->shape[0] * self->strides[0];
    view->readonly   = (self->buf_type == PROXY_ECS_ENTITIES) ? 1 : 0; // Entities are read-only
    view->itemsize   = (Py_ssize_t)self->itemsize;
    view->format     = (char *)self->format;
    view->ndim       = 1;
    view->shape      = self->shape;
    view->strides    = self->strides;
    view->suboffsets = nullptr;
    view->internal   = nullptr;

    return 0;
}

PyType_DeclareSlot_Void BufferProxy_releasebuffer([[maybe_unused]] CPH_BufferProxyObject *self,
                                                  [[maybe_unused]] Py_buffer *view) {
// Let NumPy hold the world buffer permanently.
#if defined(STRICT_THREAD_SAFETY)
    CPH_PhysicsWorldObject *world = self->world;

    // 4. Mark query finished
    // Use 'release' barrier to ensure Numpy reads are finished before Stepper writes
    if (atomic_fetch_sub_explicit(&world->physics.active_queries, 1, memory_order_release) == 1) {

        // 5. If we were the last ones, wake up the Stepper!
        // We use the internal sync mutex/cond here
        CPH_NATIVE_MUTEX_LOCK(world->physics.step_sync.mutex);
        CPH_NATIVE_COND_BROADCAST(world->physics.step_sync.cond);
        CPH_NATIVE_MUTEX_UNLOCK(world->physics.step_sync.mutex);
    }
#endif
}

// --- Heap Type Specification ---
PyType_Slot BufferProxy_slots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(BufferProxy_dealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(BufferProxy_traverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(BufferProxy_clear)},
    {.slot = Py_bf_getbuffer, .pfunc = reinterpret_cast<void *>(BufferProxy_getbuffer)},
    {.slot = Py_bf_releasebuffer, .pfunc = reinterpret_cast<void *>(BufferProxy_releasebuffer)},
    {.slot = Py_sq_length, .pfunc = reinterpret_cast<void *>(BufferProxy_length)},
    {}};

PyType_Spec BufferProxy_spec = {.name      = "culverin.BufferProxy",
                                .basicsize = sizeof(CPH_BufferProxyObject),
                                .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
                                .slots     = BufferProxy_slots};

// --- Updated make_proxy helper ---
static PyObject *make_proxy(CPH_PhysicsWorldObject *self, ProxyBufferType type, const char *format,
                            size_t itemsize, int stride) {
    CPH_ModuleState *st = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));

    // Allocation
    CPH_BufferProxyObject *proxy =
        PyObject_GC_New(CPH_BufferProxyObject, (PyTypeObject *)st->BufferProxyType);
    if (!proxy) {
        return nullptr;
    }

    // Initialization
    proxy->owner = (PyObject *)self;
    Py_INCREF(self);
    proxy->buf_type = type;
    proxy->format   = format;
    proxy->itemsize = itemsize;
    proxy->stride   = stride;

    // INITIALIZE SHAPE IMMEDIATELY
    // This allows len(proxy) to work even before a buffer is requested
    size_t count      = atomic_load_explicit(&self->physics.count, memory_order_acquire);
    proxy->shape[0]   = (Py_ssize_t)(count * stride);
    proxy->strides[0] = (Py_ssize_t)itemsize;

    atomic_fetch_add_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);

    PyObject_GC_Track(proxy);
    return (PyObject *)proxy;
}

// --- Advanced getters for lock and thread management ---
PyGetSet_DeclareGetter CPH_PyWorld_GetIsStepPending(CPH_PhysicsWorldObject *self,
                                           [[maybe_unused]] void *closure) {
    if (atomic_load_explicit(&self->physics.step_requested, memory_order_acquire)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/* --- Shadow Buffer Getters --- */

PyGetSet_DeclareGetter CPH_PyWorld_GetPositions(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    // Positions are Stride 4 (X, Y, Z, W)
    return make_proxy(self, PROXY_POSITIONS, JPH_REAL_STRING, sizeof(JPH_Real), 4);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetRotations(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    // Rotations are Stride 4 (X, Y, Z, W)
    return make_proxy(self, PROXY_ROTATIONS, "f", sizeof(float), 4);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetVelocities(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    // Velocities are Stride 4 (X, Y, Z, Pad)
    return make_proxy(self, PROXY_LINEAR_VELOCITIES, "f", sizeof(float), 4);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetAngularVelocities(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    return make_proxy(self, PROXY_ANGULAR_VELOCITIES, "f", sizeof(float), 4);
}

/* --- Mutable Metadata Getters (Hardened with Locks) --- */

PyGetSet_DeclareGetter CPH_PyWorld_GetCount(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    // TSan Fix: Lock-free atomic read.
    // Acquire ensures we see the final state of any bodies recently added/removed.
    size_t val = atomic_load_explicit(&self->physics.count, memory_order_acquire);
    return PyLong_FromSize_t(val);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetTime(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    double val = self->physics.time;
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
    return PyFloat_FromDouble(val);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetUserDataBuffer(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c) {
    // User data is Stride 1 (One uint64 per body)
    return make_proxy(self, PROXY_USER_DATA, "Q", sizeof(uint64_t), 1);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetShapeCount(CPH_PhysicsWorldObject *self, [[maybe_unused]] void *closure) {
    // Protected because resize() could move shape_cache
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    size_t count = self->physics.shape_cache_count;
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
    return PyLong_FromSize_t(count);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetMaxBodies(CPH_PhysicsWorldObject *self,
                                                   [[maybe_unused]] void *closure) {
    return PyLong_FromUnsignedLong(self->physics.max_jolt_bodies);
}

PyGetSet_DeclareGetter CPH_PyWorld_GetRemainingCapacity(CPH_PhysicsWorldObject *self,
                                                           [[maybe_unused]] void *closure) {
    // TSan Fix: Lock-free calculation using atomic count.
    size_t current = atomic_load_explicit(&self->physics.count, memory_order_acquire);

    // max_jolt_bodies is immutable after world initialization
    size_t limit = self->physics.max_jolt_bodies;
    size_t rem   = (current >= limit) ? 0 : (limit - current);

    return PyLong_FromSize_t(rem);
}