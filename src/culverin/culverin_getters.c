#include <Python.h>
#include "culverin.h"
#include "culverin_getters.h"

static PyObject *make_view(PhysicsWorldObject *self, void *ptr) {
  if (!ptr) {
    Py_RETURN_NONE;
  }

  // 1. Capture State Under Lock
  SHADOW_LOCK(&self->shadow_lock);

  // We capture the current count at the moment the view is exported
  size_t current_count = self->count;

  // Increment export count to prevent resize() from moving this pointer
  self->view_export_count++;

  SHADOW_UNLOCK(&self->shadow_lock);

  // 2. Setup Local Buffer Metadata
  // These are copied by Python into the memoryview object.
  // We use 4 floats per body (stride is 16 bytes).
  Py_ssize_t local_shape[1] = {(Py_ssize_t)(current_count * 4)};
  Py_ssize_t local_strides[1] = {(Py_ssize_t)sizeof(float)};

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = ptr;
  buf.obj = (PyObject *)self; // Ownership link
  Py_INCREF(self);

  buf.len = local_shape[0] * (Py_ssize_t)sizeof(float);
  buf.readonly = 1;
  buf.itemsize = sizeof(float);
  buf.format = "f";
  buf.ndim = 1;
  buf.shape = local_shape;
  buf.strides = local_strides;

  // 3. Create MemoryView
  PyObject *mv = PyMemoryView_FromBuffer(&buf);

  if (!mv) {
    // Clean up on failure
    SHADOW_LOCK(&self->shadow_lock);
    if (self->view_export_count > 0) {
      self->view_export_count--;
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    Py_DECREF(self); // Drop the ownership link ref
    return NULL;
  }

  return mv;
}

/* --- Immutable Getters (Safe without locks) --- */

PyObject *Vehicle_get_wheel_count(VehicleObject *self, void *closure) {
  // num_wheels is set at creation and never changes
  return PyLong_FromUnsignedLong(self->num_wheels);
}

PyObject *Character_get_handle(CharacterObject *self, void *closure) {
  // handle is immutable for the life of this Character instance
  return PyLong_FromUnsignedLongLong(self->handle);
}

/* --- Shadow Buffer Getters (Safe via hardened make_view) --- */

PyObject *get_positions(PhysicsWorldObject *self, void *c) {
  // make_view internally acquires SHADOW_LOCK and snapshots count
  return make_view(self, self->positions);
}

PyObject *get_rotations(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->rotations);
}

PyObject *get_velocities(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->linear_velocities);
}

PyObject *get_angular_velocities(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->angular_velocities);
}

/* --- Mutable Metadata Getters (Hardened with Locks) --- */

PyObject *get_count(PhysicsWorldObject *self, void *c) {
  SHADOW_LOCK(&self->shadow_lock);
  size_t val = self->count;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromSize_t(val);
}

PyObject *get_time(PhysicsWorldObject *self, void *c) {
  SHADOW_LOCK(&self->shadow_lock);
  double val = self->time;
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyFloat_FromDouble(val);
}

PyObject *get_user_data_buffer(PhysicsWorldObject *self, void *c) {
  if (!self->user_data) {
    Py_RETURN_NONE;
  }

  SHADOW_LOCK(&self->shadow_lock);
  size_t current_count = self->count;
  self->view_export_count++;
  SHADOW_UNLOCK(&self->shadow_lock);

  // Use stack-allocated metadata to prevent cross-thread corruption
  Py_ssize_t local_shape[1] = {(Py_ssize_t)current_count};
  Py_ssize_t local_stride = sizeof(uint64_t);

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = self->user_data;
  buf.obj = (PyObject *)self;
  Py_INCREF(self);

  buf.len = (Py_ssize_t)(current_count * sizeof(uint64_t));
  buf.readonly = 1;
  buf.itemsize = sizeof(uint64_t);
  buf.format = "Q";
  buf.ndim = 1;
  buf.shape = local_shape;
  buf.strides = &local_stride;

  PyObject *mv = PyMemoryView_FromBuffer(&buf);
  if (!mv) {
    SHADOW_LOCK(&self->shadow_lock);
    if (self->view_export_count > 0) {
      self->view_export_count--;
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_DECREF(self);
    return NULL;
  }
  return mv;
}