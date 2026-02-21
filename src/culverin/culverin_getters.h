#pragma once
#include <Python.h>

// Forward declarations: "Trust me, these exist"
struct PhysicsWorldObject;
struct CharacterObject;
struct VehicleObject;

PyObject *get_positions(struct PhysicsWorldObject *self, void *Py_UNUSED(closure));
PyObject *get_rotations(struct PhysicsWorldObject *self, void *Py_UNUSED(closure));
PyObject *get_velocities(struct PhysicsWorldObject *self, void *Py_UNUSED(c));
PyObject *get_angular_velocities(struct PhysicsWorldObject *self, void *Py_UNUSED(c));
PyObject *get_count(struct PhysicsWorldObject *self, void *Py_UNUSED(c));
PyObject *get_time(struct PhysicsWorldObject *self, void *Py_UNUSED(c));
PyObject *get_user_data_buffer(struct PhysicsWorldObject *self, void *Py_UNUSED(c));

PyObject *Character_get_handle(struct CharacterObject *self, void *Py_UNUSED(closure));

PyObject *Vehicle_get_wheel_count(struct VehicleObject *self, void *Py_UNUSED(closure));

PyObject *get_shape_count(struct PhysicsWorldObject *self, void *Py_UNUSED(closure));