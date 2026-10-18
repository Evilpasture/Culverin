#pragma once

#include "core/Types.h"
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>

struct CPH_PhysicsWorld;

static constexpr float kPhysicsQueryDefaultMaxDistance = 1000.0f;
static constexpr size_t kPhysicsQueryInlineHitCapacity = 64;

typedef enum PhysicsQueryStatus : int {
    PHYSICS_QUERY_STATUS_OK               = 0,
    PHYSICS_QUERY_STATUS_OUT_OF_MEMORY    = -1,
    PHYSICS_QUERY_STATUS_INVALID_ARGUMENT = -2,
    PHYSICS_QUERY_STATUS_SHAPE_FAILURE    = -3,
} PhysicsQueryStatus;

typedef struct CPH_PhysicsQueryHitBuffer {
    BodyHandle inline_hits[kPhysicsQueryInlineHitCapacity];
    BodyHandle *hits;
    size_t count;
    size_t capacity;
} CPH_PhysicsQueryHitBuffer;

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

static constexpr size_t kRaycastResultSize = 48;
static_assert(sizeof(RayCastBatchResult) == kRaycastResultSize);

typedef struct CPH_PhysicsRaycastHit {
    BodyHandle handle;
    float fraction;
    JPH_Vec3 normal;
    bool has_hit;
} CPH_PhysicsRaycastHit;

typedef struct CPH_PhysicsShapecastHit {
    BodyHandle handle;
    float fraction;
    JPH_Vec3 point;
    JPH_Vec3 normal;
    bool has_hit;
} CPH_PhysicsShapecastHit;

void CPH_PhysicsQueryHitBufferInit(CPH_PhysicsQueryHitBuffer *buffer);
void CPH_PhysicsQueryHitBufferDestroy(CPH_PhysicsQueryHitBuffer *buffer);

// Native query lifetime/synchronization helpers shared with other native
// physics operations that increment active_queries while running outside the
// shadow lock.
void CPH_PhysicsWorldQueryEnd(struct CPH_PhysicsWorld *world);

PhysicsQueryStatus CPH_PhysicsWorldOverlapSphere(struct CPH_PhysicsWorld *world, JPH_RVec3 center,
                                                float radius, CPH_PhysicsQueryHitBuffer *hits);
PhysicsQueryStatus CPH_PhysicsWorldOverlapAabb(struct CPH_PhysicsWorld *world, JPH_RVec3 min,
                                              JPH_RVec3 max, CPH_PhysicsQueryHitBuffer *hits);
PhysicsQueryStatus CPH_PhysicsWorldRaycast(struct CPH_PhysicsWorld *world, JPH_RVec3 origin,
                                         JPH_Vec3 direction, float max_distance,
                                         BodyHandle ignore_handle, CPH_PhysicsRaycastHit *hit);
PhysicsQueryStatus CPH_PhysicsWorldRaycastBatch(struct CPH_PhysicsWorld *world, const float *starts,
                                               const float *directions, size_t count,
                                               float max_distance, RayCastBatchResult *results);
PhysicsQueryStatus CPH_PhysicsWorldShapecast(struct CPH_PhysicsWorld *world, int shape_type,
                                           JPH_RVec3 position, JPH_Quat rotation,
                                           JPH_Vec3 sweep_direction, const float size[4],
                                           BodyHandle ignore_handle, CPH_PhysicsShapecastHit *hit);
