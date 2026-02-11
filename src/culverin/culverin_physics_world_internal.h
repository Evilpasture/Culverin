#pragma once
#include "culverin.h"

int PhysicsWorld_resize(struct PhysicsWorldObject *self, size_t new_capacity);

void free_constraints(PhysicsWorldObject *self);

void free_shape_cache(PhysicsWorldObject *self);

void free_shadow_buffers(PhysicsWorldObject *self);

void PhysicsWorld_free_members(PhysicsWorldObject *self);

int init_settings(PhysicsWorldObject *self, PyObject *settings_dict, float *gx,
                  float *gy, float *gz, int *max_bodies, int *max_pairs);

int init_jolt_core(PhysicsWorldObject *self, WorldLimits limits,
                   GravityVector gravity);

int allocate_buffers(PhysicsWorldObject *self, int max_bodies);

int load_baked_scene(PhysicsWorldObject *self, PyObject *baked);

int verify_abi_alignment(JPH_BodyInterface *bi);

void PhysicsWorld_releasebuffer(PhysicsWorldObject *self, Py_buffer *view);