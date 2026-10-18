#pragma once

#include "core/Types.h"
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>

struct PhysicsWorld;

static constexpr float PHYSICS_QUERY_DEFAULT_MAX_DISTANCE = 1000.0f;
static constexpr size_t PHYSICS_QUERY_INLINE_HIT_CAPACITY = 64;

typedef enum PhysicsQueryStatus : int {
    PHYSICS_QUERY_STATUS_OK               = 0,
    PHYSICS_QUERY_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_QUERY_STATUS_INVALID_ARGUMENT = -2,
    PHYSICS_QUERY_STATUS_SHAPE_FAILURE    = -3,
} PhysicsQueryStatus;

typedef struct PhysicsQueryHitBuffer {
    BodyHandle inline_hits[PHYSICS_QUERY_INLINE_HIT_CAPACITY];
    BodyHandle *hits;
    size_t count;
    size_t capacity;
} PhysicsQueryHitBuffer;

// Packed output for the Python raycast-batch byte buffer. Keep this wire layout
// stable: consumers decode each record as exactly 48 bytes.
#ifdef _MSC_VER
#    pragma pack(push, 1)
#endif
typedef struct
#ifndef _MSC_VER
    __attribute__((packed))
#endif
{
    uint64_t handle;
    float fraction;
    float nx, ny, nz;
    float px, py, pz;
    uint32_t subShapeID;
    uint32_t material_id;
    uint32_t _pad;
} RayCastBatchResult;
#ifdef _MSC_VER
#    pragma pack(pop)
#endif

static constexpr size_t RAYCAST_RESULT_SIZE = 48;
static_assert(sizeof(RayCastBatchResult) == RAYCAST_RESULT_SIZE);

typedef struct PhysicsRaycastHit {
    BodyHandle handle;
    float fraction;
    JPH_Vec3 normal;
    bool has_hit;
} PhysicsRaycastHit;

typedef struct PhysicsShapecastHit {
    BodyHandle handle;
    float fraction;
    JPH_Vec3 point;
    JPH_Vec3 normal;
    bool has_hit;
} PhysicsShapecastHit;

void physics_query_hit_buffer_init(PhysicsQueryHitBuffer *buffer);
void physics_query_hit_buffer_destroy(PhysicsQueryHitBuffer *buffer);

// Native query lifetime/synchronization helpers shared with other native
// physics operations that increment active_queries while running outside the
// shadow lock.
void physics_world_query_end(struct PhysicsWorld *world);

PhysicsQueryStatus physics_world_overlap_sphere(struct PhysicsWorld *world, JPH_RVec3 center,
                                                float radius, PhysicsQueryHitBuffer *hits);
PhysicsQueryStatus physics_world_overlap_aabb(struct PhysicsWorld *world, JPH_RVec3 min,
                                              JPH_RVec3 max, PhysicsQueryHitBuffer *hits);
PhysicsQueryStatus physics_world_raycast(struct PhysicsWorld *world, JPH_RVec3 origin,
                                         JPH_Vec3 direction, float max_distance,
                                         BodyHandle ignore_handle, PhysicsRaycastHit *hit);
PhysicsQueryStatus physics_world_raycast_batch(struct PhysicsWorld *world, const float *starts,
                                               const float *directions, size_t count,
                                               float max_distance, RayCastBatchResult *results);
PhysicsQueryStatus physics_world_shapecast(struct PhysicsWorld *world, int shape_type,
                                           JPH_RVec3 position, JPH_Quat rotation,
                                           JPH_Vec3 sweep_direction, const float size[4],
                                           BodyHandle ignore_handle, PhysicsShapecastHit *hit);
