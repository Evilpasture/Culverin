#include "culverin_getters.h"
#include "culverin.h"
#include "culverin_character.h"
#include "culverin_ecs.h"
#include "culverin_fast_build.h"
#include "culverin_physics_sync.h"
#include <Python.h>

static Py_ssize_t BufferProxy_length(BufferProxyObject *self) {
    // Return the cached shape. Since we update shape[0] in the
    // constructor and in getbuffer, this is always accurate
    // to the count at the time of the last sync/query.
    return self->shape[0];
}

PyType_DeclareSlot_Status BufferProxy_traverse(BufferProxyObject *self, visitproc visit,
                                               void *arg) {
    Py_VISIT(self->owner);
    return 0;
}

// Inside BufferProxy_clear
PyType_DeclareSlot_Status BufferProxy_clear(BufferProxyObject *self) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    if (self->owner) {
        // We know that both PhysicsWorld and Registry have the export count at a fixed offset
        // or we can switch based on type.
        if (PyObject_TypeCheck(self->owner, (PyTypeObject *)st->PhysicsWorldType)) {
            atomic_fetch_sub_explicit(&((PhysicsWorldObject *)self->owner)->view_export_count, 1,
                                      memory_order_relaxed);
        } else {
            atomic_fetch_sub_explicit(&((RegistryObject *)self->owner)->view_export_count, 1,
                                      memory_order_relaxed);
        }
        Py_CLEAR(self->owner);
    }
    return 0;
}

PyType_DeclareSlot_Void BufferProxy_dealloc(BufferProxyObject *self) {
    PyObject_GC_UnTrack(self);
    CULV_MAYBE_UNUSED auto cleared = BufferProxy_clear(self);
    // For heap types, we must release our reference to the type itself
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free((PyObject *)self);
    Py_DECREF(tp);
}

PyType_DeclareSlot_Status BufferProxy_getbuffer(BufferProxyObject *self, Py_buffer *view,
                                                CULV_MAYBE_UNUSED int flags) {
    // Determine the lock and target based on the buffer type category
    void *target_ptr = nullptr;

    if (self->buf_type < PROXY_ECS_DATA) {
        /* --- CATEGORY: Physics World Buffers --- */
        PhysicsWorldObject *world = (PhysicsWorldObject *)self->owner;

        SHADOW_LOCK(&world->shadow_lock);
        BLOCK_UNTIL_NOT_STEPPING(world);

        switch (self->buf_type) {
        case PROXY_POSITIONS:
            target_ptr = world->positions;
            break;
        case PROXY_ROTATIONS:
            target_ptr = world->rotations;
            break;
        case PROXY_LINEAR_VELOCITIES:
            target_ptr = world->linear_velocities;
            break;
        case PROXY_ANGULAR_VELOCITIES:
            target_ptr = world->angular_velocities;
            break;
        case PROXY_USER_DATA:
            target_ptr = world->user_data;
            break;
        case PROXY_DYNAMIC:
            target_ptr = self->dynamic_ptr;
            break; // Soft Body Verts
        default:
            culv_unreachable();
        }

        if (!target_ptr) {
            SHADOW_UNLOCK(&world->shadow_lock);
            PyErr_SetString(PyExc_RuntimeError, "Physics buffer not allocated");
            return -1;
        }

        // Auto-update shape based on current body count for non-dynamic buffers
        if (self->buf_type != PROXY_DYNAMIC) {
            size_t count   = atomic_load_explicit(&world->count, memory_order_acquire);
            self->shape[0] = (Py_ssize_t)(count * self->stride);
        }

        SHADOW_UNLOCK(&world->shadow_lock);

    } else {
        /* --- CATEGORY: ECS Registry Buffers --- */
        RegistryObject *reg = (RegistryObject *)self->owner;

        SHADOW_LOCK(&reg->ecs_lock);

        // For ECS, the pointer was stored in dynamic_ptr during make_ecs_proxy
        target_ptr = self->dynamic_ptr;

        if (!target_ptr) {
            // This can happen if get_view was called on an empty component
            // We allow it to return an empty buffer rather than erroring
            target_ptr = (void *)"";
        }

        // ECS shape is pinned at creation because we block reallocs while views are held
        // No need to update self->shape[0] here.

        SHADOW_UNLOCK(&reg->ecs_lock);
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

PyType_DeclareSlot_Void BufferProxy_releasebuffer(CULV_MAYBE_UNUSED BufferProxyObject *self,
                                                  CULV_MAYBE_UNUSED Py_buffer *view) {
// Let NumPy hold the world buffer permanently.
#if defined(STRICT_THREAD_SAFETY)
    PhysicsWorldObject *world = self->world;

    // 4. Mark query finished
    // Use 'release' barrier to ensure Numpy reads are finished before Stepper writes
    if (atomic_fetch_sub_explicit(&world->active_queries, 1, memory_order_release) == 1) {

        // 5. If we were the last ones, wake up the Stepper!
        // We use the internal sync mutex/cond here
        NATIVE_MUTEX_LOCK(world->step_sync.mutex);
        NATIVE_COND_BROADCAST(world->step_sync.cond);
        NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
    }
#endif
}

// --- Heap Type Specification ---
PyType_Slot BufferProxy_slots[] = {
    {.slot = Py_tp_dealloc, .pfunc = BufferProxy_dealloc},
    {.slot = Py_tp_traverse, .pfunc = BufferProxy_traverse},
    {.slot = Py_tp_clear, .pfunc = BufferProxy_clear},
    {.slot = Py_bf_getbuffer, .pfunc = BufferProxy_getbuffer},
    {.slot = Py_bf_releasebuffer, .pfunc = BufferProxy_releasebuffer},
    {.slot = Py_sq_length, .pfunc = BufferProxy_length},
    {}};

PyType_Spec BufferProxy_spec = {.name      = "culverin.BufferProxy",
                                .basicsize = sizeof(BufferProxyObject),
                                .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
                                .slots     = BufferProxy_slots};

// --- Updated make_proxy helper ---
static PyObject *make_proxy(PhysicsWorldObject *self, ProxyBufferType type, const char *format,
                            size_t itemsize, int stride) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));

    // Allocation
    BufferProxyObject *proxy =
        PyObject_GC_New(BufferProxyObject, (PyTypeObject *)st->BufferProxyType);
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
    size_t count      = atomic_load_explicit(&self->count, memory_order_acquire);
    proxy->shape[0]   = (Py_ssize_t)(count * stride);
    proxy->strides[0] = (Py_ssize_t)itemsize;

    atomic_fetch_add_explicit(&self->view_export_count, 1, memory_order_relaxed);

    PyObject_GC_Track(proxy);
    return (PyObject *)proxy;
}

// --- Advanced getters for lock and thread management ---
PyGetSet_DeclareGetter get_is_step_pending(PhysicsWorldObject *self,
                                           CULV_MAYBE_UNUSED void *closure) {
    if (atomic_load_explicit(&self->step_requested, memory_order_acquire)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/* --- Shadow Buffer Getters --- */

PyGetSet_DeclareGetter get_positions(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // Positions are Stride 4 (X, Y, Z, W)
    return make_proxy(self, PROXY_POSITIONS, JPH_REAL_STRING, sizeof(JPH_Real), 4);
}

PyGetSet_DeclareGetter get_rotations(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // Rotations are Stride 4 (X, Y, Z, W)
    return make_proxy(self, PROXY_ROTATIONS, "f", sizeof(float), 4);
}

PyGetSet_DeclareGetter get_velocities(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // Velocities are Stride 4 (X, Y, Z, Pad)
    return make_proxy(self, PROXY_LINEAR_VELOCITIES, "f", sizeof(float), 4);
}

PyGetSet_DeclareGetter get_angular_velocities(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    return make_proxy(self, PROXY_ANGULAR_VELOCITIES, "f", sizeof(float), 4);
}

/* --- Mutable Metadata Getters (Hardened with Locks) --- */

PyGetSet_DeclareGetter get_count(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // TSan Fix: Lock-free atomic read.
    // Acquire ensures we see the final state of any bodies recently added/removed.
    size_t val = atomic_load_explicit(&self->count, memory_order_acquire);
    return PyLong_FromSize_t(val);
}

PyGetSet_DeclareGetter get_time(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    SHADOW_LOCK(&self->shadow_lock);
    double val = self->time;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyFloat_FromDouble(val);
}

PyGetSet_DeclareGetter get_user_data_buffer(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // User data is Stride 1 (One uint64 per body)
    return make_proxy(self, PROXY_USER_DATA, "Q", sizeof(uint64_t), 1);
}

PyGetSet_DeclareGetter get_shape_count(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *closure) {
    // Protected because resize() could move shape_cache
    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->shape_cache_count;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromSize_t(count);
}

PyGetSet_DeclareGetter PhysicsWorld_get_max_bodies(PhysicsWorldObject *self,
                                                   CULV_MAYBE_UNUSED void *closure) {
    return PyLong_FromUnsignedLong(self->max_jolt_bodies);
}

PyGetSet_DeclareGetter PhysicsWorld_get_remaining_capacity(PhysicsWorldObject *self,
                                                           CULV_MAYBE_UNUSED void *closure) {
    // TSan Fix: Lock-free calculation using atomic count.
    size_t current = atomic_load_explicit(&self->count, memory_order_acquire);

    // max_jolt_bodies is immutable after world initialization
    size_t limit = self->max_jolt_bodies;
    size_t rem   = (current >= limit) ? 0 : (limit - current);

    return PyLong_FromSize_t(rem);
}