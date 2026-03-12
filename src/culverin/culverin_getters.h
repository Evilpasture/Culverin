#pragma once
#include "culverin_compiler_specifics.h"
#include <Python.h>

// Forward declarations: "Trust me, these exist"
struct PhysicsWorldObject;
struct CharacterObject;
struct VehicleObject;

PyGetSet_DeclareGetter get_is_step_pending(struct PhysicsWorldObject *self,
                                           CULV_MAYBE_UNUSED void *closure);
PyGetSet_DeclareGetter get_positions(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_rotations(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_velocities(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_angular_velocities(struct PhysicsWorldObject *self,
                                              CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_count(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_time(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_user_data_buffer(struct PhysicsWorldObject *self,
                                            CULV_MAYBE_UNUSED void *c);

PyGetSet_DeclareGetter Character_get_handle(struct CharacterObject *self,
                                            CULV_MAYBE_UNUSED void *closure);

PyGetSet_DeclareGetter Vehicle_get_wheel_count(struct VehicleObject *self,
                                               CULV_MAYBE_UNUSED void *closure);

PyGetSet_DeclareGetter get_shape_count(struct PhysicsWorldObject *self,
                                       CULV_MAYBE_UNUSED void *closure);

PyGetSet_DeclareGetter PhysicsWorld_get_max_bodies(struct PhysicsWorldObject *self,
                                                   CULV_MAYBE_UNUSED void *closure);
PyGetSet_DeclareGetter PhysicsWorld_get_remaining_capacity(struct PhysicsWorldObject *self,
                                                           CULV_MAYBE_UNUSED void *closure);
