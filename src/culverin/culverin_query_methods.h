#pragma once
#include "culverin.h"

// Helper for Overlap Callbacks
typedef struct {
    PhysicsWorldObject *world;
    uint64_t *hits; // C array to store baked handles
    size_t count;
    size_t capacity;
    bool is_on_stack;
} OverlapContext;

constexpr int STACK_ALLOCATE_HITS = 64;

void end_query_scope(PhysicsWorldObject *self);

PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_raycast_batch(PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_shapecast(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
