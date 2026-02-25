#include "culverin_getters.h"
#include "culverin.h"
#include "culverin_character.h"
#include <Python.h>

// --- Advanced getters for lock and thread management ---
PyObject *get_is_step_pending(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *closure) {
    if (atomic_load_explicit(&self->step_requested, memory_order_acquire)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// Helper to create MemoryViews with specific types and sizes
static PyObject *make_view(PhysicsWorldObject *self, void *ptr, const char *format, size_t itemsize, int stride) {
  if (!ptr) Py_RETURN_NONE;

  SHADOW_LOCK(&self->shadow_lock);
  size_t current_count = self->count;
  self->view_export_count++;
  
  // Update persistent storage in the object
  self->view_shape[0] = (Py_ssize_t)(current_count * stride);
  self->view_strides[0] = (Py_ssize_t)itemsize;
  SHADOW_UNLOCK(&self->shadow_lock);

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = ptr;
  buf.obj = (PyObject *)self; 
  Py_INCREF(self);

  buf.len = self->view_shape[0] * self->view_strides[0];
  buf.readonly = 1;
  buf.itemsize = (Py_ssize_t)itemsize;
  buf.format = (char *)format; 
  buf.ndim = 1;
  
  // CRITICAL: Ensure these pointers are not to the stack!
  buf.shape = self->view_shape;
  buf.strides = self->view_strides;

  PyObject *mv = PyMemoryView_FromBuffer(&buf);
  if (!mv) {
    Py_DECREF(self);
    return NULL;
  }
  return mv;
}
/* --- Immutable Getters (Safe without locks) --- */

PyObject *Vehicle_get_wheel_count(VehicleObject *self, CULV_MAYBE_UNUSED  void *closure) {
  // num_wheels is set at creation and never changes
  return PyLong_FromUnsignedLong(self->num_wheels);
}

PyObject *Character_get_handle(CharacterObject *self, CULV_MAYBE_UNUSED  void *closure) {
  // handle is immutable for the life of this Character instance
  return PyLong_FromUnsignedLongLong(self->handle);
}

/* --- Shadow Buffer Getters (Safe via hardened make_view) --- */

PyObject *get_positions(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  // Positions are Stride 4 (X, Y, Z, W)
  return make_view(self, self->positions, JPH_REAL_STRING, sizeof(JPH_Real), 4);
}

PyObject *get_rotations(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  // Rotations are Stride 4 (X, Y, Z, W)
  return make_view(self, self->rotations, "f", sizeof(float), 4);
}

PyObject *get_velocities(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  // Velocities are Stride 4 (X, Y, Z, Pad)
  return make_view(self, self->linear_velocities, "f", sizeof(float), 4);
}

PyObject *get_angular_velocities(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  return make_view(self, self->angular_velocities, "f", sizeof(float), 4);
}


/* --- Mutable Metadata Getters (Hardened with Locks) --- */

PyObject *get_count(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  SHADOW_LOCK(&self->shadow_lock);
  size_t val = self->count;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromSize_t(val);
}

PyObject *get_time(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  SHADOW_LOCK(&self->shadow_lock);
  double val = self->time;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyFloat_FromDouble(val);
}

PyObject *get_user_data_buffer(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c) {
  // User data is Stride 1 (One uint64 per body)
  return make_view(self, self->user_data, "Q", sizeof(uint64_t), 1);
}

PyObject *get_shape_count(PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *closure) {
    // Protected because resize() could move shape_cache
    SHADOW_LOCK(&self->shadow_lock);
    size_t count = self->shape_cache_count;
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromSize_t(count);
}