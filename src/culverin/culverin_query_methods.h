#pragma once
#include "culverin_compiler_specifics.h"
#include "culverin_physics_world.h"
#include <Python.h>

// Helper for Overlap Callbacks
typedef struct {
    PhysicsWorldObject *world;
    uint64_t *hits; // C array to store baked handles
    size_t count;
    size_t capacity;
    bool is_on_stack;
} OverlapContext;

// --- Raycast Batch Result (Aligned to 16-bytes, Total 48-bytes) ---
#ifdef _MSC_VER
#    pragma pack(push, 1)
#endif
typedef struct
#ifndef _MSC_VER
    __attribute__((packed))
#endif
{
    uint64_t handle;      // 8 bytes
    float fraction;       // 4 bytes
    float nx, ny, nz;     // 12 bytes
    float px, py, pz;     // 12 bytes
    uint32_t subShapeID;  // 4 bytes
    uint32_t material_id; // 4 bytes
    uint32_t _pad;
} RayCastBatchResult;
#ifdef _MSC_VER
#    pragma pack(pop)
#endif

static constexpr size_t RAYCAST_RESULT_SIZE = 48;

static_assert(sizeof(RayCastBatchResult) == RAYCAST_RESULT_SIZE);

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
