#include "culverin_getters.h"
#include "culverin.h"
#include "culverin_character.h"
#include <Python.h>

// --- Advanced getters for lock and thread management ---
PyGetSet_DeclareGetter get_is_step_pending(PhysicsWorldObject *self,
                                           CULV_MAYBE_UNUSED void *closure) {
    if (atomic_load_explicit(&self->step_requested, memory_order_acquire)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// Helper to create MemoryViews with specific types and sizes
static PyObject *make_view(PhysicsWorldObject *self, void *ptr, const char *format, size_t itemsize,
                           int stride) {
    if (!ptr) {
        Py_RETURN_NONE;
    }

    SHADOW_LOCK(&self->shadow_lock);
    
    // TSan Fix: Read the atomic count safely. 
    // Acquire ensures we see the finished state of any preceding world_remove_body_slot.
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    
    // view_export_count is a standard int protected by shadow_lock
    self->view_export_count++;

    // Update persistent storage in the object (used by Py_buffer pointers)
    self->view_shape[0]   = (Py_ssize_t)(current_count * stride);
    self->view_strides[0] = (Py_ssize_t)itemsize;
    
    SHADOW_UNLOCK(&self->shadow_lock);

    Py_buffer buf;
    memset(&buf, 0, sizeof(Py_buffer));
    buf.buf = ptr;
    buf.obj = (PyObject *)self;
    Py_INCREF(self);

    // Calculate total length based on the snapshot we took inside the lock
    buf.len      = self->view_shape[0] * self->view_strides[0];
    buf.readonly = 1;
    buf.itemsize = (Py_ssize_t)itemsize;
    buf.format   = (char *)format;
    buf.ndim     = 1;

    // These pointers point to persistent fields in the PhysicsWorldObject struct,
    // which remain valid even after this stack frame is destroyed.
    buf.shape   = self->view_shape;
    buf.strides = self->view_strides;

    PyObject *mv = PyMemoryView_FromBuffer(&buf);
    if (!mv) {
        // Cleanup if memoryview allocation fails
        SHADOW_LOCK(&self->shadow_lock);
        self->view_export_count--;
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_DECREF(self);
        return nullptr;
    }
    return mv;
}
/* --- Immutable Getters (Safe without locks) --- */

PyGetSet_DeclareGetter Vehicle_get_wheel_count(VehicleObject *self,
                                               CULV_MAYBE_UNUSED void *closure) {
    // num_wheels is set at creation and never changes
    return PyLong_FromUnsignedLong(self->num_wheels);
}

PyGetSet_DeclareGetter Character_get_handle(CharacterObject *self,
                                            CULV_MAYBE_UNUSED void *closure) {
    // TSan Fix: Atomic load of the character handle.
    // Relaxed is sufficient because the handle is set during creation and never changes.
    uint64_t raw_h = atomic_load_explicit(&self->handle, memory_order_relaxed);
    return PyLong_FromUnsignedLongLong(raw_h);
}

/* --- Shadow Buffer Getters (Safe via hardened make_view) --- */

PyGetSet_DeclareGetter get_positions(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // Positions are Stride 4 (X, Y, Z, W)
    return make_view(self, self->positions, JPH_REAL_STRING, sizeof(JPH_Real), 4);
}

PyGetSet_DeclareGetter get_rotations(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // Rotations are Stride 4 (X, Y, Z, W)
    return make_view(self, self->rotations, "f", sizeof(float), 4);
}

PyGetSet_DeclareGetter get_velocities(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    // Velocities are Stride 4 (X, Y, Z, Pad)
    return make_view(self, self->linear_velocities, "f", sizeof(float), 4);
}

PyGetSet_DeclareGetter get_angular_velocities(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
    return make_view(self, self->angular_velocities, "f", sizeof(float), 4);
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
    return make_view(self, self->user_data, "Q", sizeof(uint64_t), 1);
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
    size_t rem = (current >= limit) ? 0 : (limit - current);
    
    return PyLong_FromSize_t(rem);
}