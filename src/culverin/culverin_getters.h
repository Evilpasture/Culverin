#pragma once
#include <Python.h>
#include "culverin_compiler_specifics.h"

// Forward declarations: "Trust me, these exist"
struct PhysicsWorldObject;
struct CharacterObject;
struct VehicleObject;

PyObject *get_is_step_pending(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *closure);
PyObject *get_positions(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyObject *get_rotations(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyObject *get_velocities(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyObject *get_angular_velocities(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyObject *get_count(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyObject *get_time(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyObject *get_user_data_buffer(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);

PyObject *Character_get_handle(struct CharacterObject *self, CULV_MAYBE_UNUSED void *closure);

PyObject *Vehicle_get_wheel_count(struct VehicleObject *self, CULV_MAYBE_UNUSED void *closure);

PyObject *get_shape_count(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *closure);