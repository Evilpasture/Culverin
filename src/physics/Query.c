#include "physics/Query.h"
#include "core/NativeAlloc.h"
#include "math/MathCore.h"
#include "physics/InternalQuery.h"
#include "physics/PhysicsWorldState.h"
#include <math.h>
#include <string.h>

static constexpr float kQueryDirectionEpsilonSquared = 1e-12f;
static constexpr float kQueryDirectionUnitTolerance  = 1e-4f;

static void wait_until_can_query_locked(CPH_PhysicsWorld *world) {
    if (atomic_load_explicit(&world->is_stepping, memory_order_relaxed) ||
        atomic_load_explicit(&world->step_requested, memory_order_relaxed)) {
        atomic_fetch_add_explicit(&world->waiting_threads, 1, memory_order_relaxed);

        while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed) ||
               atomic_load_explicit(&world->step_requested, memory_order_relaxed)) {
            // Keep the step flags and active-query count ordered by shadow_lock,
            // but never hold that lock while parking on the condition variable.
            CPH_SHADOW_UNLOCK(&world->shadow_lock);
            CPH_NATIVE_MUTEX_LOCK(world->step_sync.mutex);
            while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed) ||
                   atomic_load_explicit(&world->step_requested, memory_order_relaxed)) {
                CPH_NATIVE_COND_WAIT(world->step_sync.cond, world->step_sync.mutex);
            }
            CPH_NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
            CPH_SHADOW_LOCK(&world->shadow_lock);
        }

        atomic_fetch_sub_explicit(&world->waiting_threads, 1, memory_order_relaxed);
    }
}

static void query_begin(CPH_PhysicsWorld *world) {
    CPH_SHADOW_LOCK(&world->shadow_lock);
    wait_until_can_query_locked(world);

    // This increment is protected by shadow_lock, so CPH_PyWorld_Step cannot
    // set the step flags between the gate check and query registration.
    atomic_fetch_add_explicit(&world->active_queries, 1, memory_order_acquire);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
}

void CPH_PhysicsWorldQueryEnd(CPH_PhysicsWorld *world) {
    if (!world) {
        return;
    }

    CPH_NATIVE_MUTEX_LOCK(world->step_sync.mutex);
    const int previous = atomic_fetch_sub_explicit(&world->active_queries, 1, memory_order_acq_rel);
    if (previous == 1) {
        CPH_NATIVE_COND_BROADCAST(world->step_sync.cond);
    }
    CPH_NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
}

void CPH_PhysicsQueryHitBufferInit(CPH_PhysicsQueryHitBuffer *buffer) {
    if (!buffer) {
        return;
    }

    buffer->hits     = buffer->inline_hits;
    buffer->count    = 0;
    buffer->capacity = kPhysicsQueryInlineHitCapacity;
}

void CPH_PhysicsQueryHitBufferDestroy(CPH_PhysicsQueryHitBuffer *buffer) {
    if (!buffer) {
        return;
    }

    if (buffer->hits && buffer->hits != buffer->inline_hits) {
        CPH_Free(buffer->hits);
    }
    buffer->hits     = buffer->inline_hits;
    buffer->count    = 0;
    buffer->capacity = kPhysicsQueryInlineHitCapacity;
}

static bool query_hit_buffer_append(CPH_PhysicsQueryHitBuffer *buffer, BodyHandle handle) {
    if (buffer->count >= buffer->capacity) {
        size_t new_capacity =
            buffer->capacity ? buffer->capacity * 2 : kPhysicsQueryInlineHitCapacity;
        if (new_capacity < buffer->capacity || new_capacity > SIZE_MAX / sizeof(*buffer->hits)) {
            return false;
        }

        BodyHandle *new_hits;
        if (buffer->hits == buffer->inline_hits) {
            new_hits = CPH_Alloc(new_capacity * sizeof(*new_hits));
            if (new_hits) {
                memcpy(new_hits, buffer->hits, buffer->count * sizeof(*new_hits));
            }
        } else {
            new_hits = CPH_Realloc(buffer->hits, new_capacity * sizeof(*new_hits));
        }

        if (!new_hits) {
            return false;
        }
        buffer->hits     = new_hits;
        buffer->capacity = new_capacity;
    }

    buffer->hits[buffer->count++] = handle;
    return true;
}

static bool query_handle_is_valid_locked(CPH_PhysicsWorld *world, BodyHandle handle,
                                         uint32_t *slot_out) {
    if (!handle) {
        return false;
    }

    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return false;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return false;
    }

    if (slot_out) {
        *slot_out = slot;
    }
    return true;
}

static void filter_query_hits(CPH_PhysicsWorld *world, CPH_PhysicsQueryHitBuffer *hits) {
    CPH_SHADOW_LOCK(&world->shadow_lock);
    size_t write_index = 0;
    for (size_t i = 0; i < hits->count; ++i) {
        if (query_handle_is_valid_locked(world, hits->hits[i], nullptr)) {
            hits->hits[write_index++] = hits->hits[i];
        }
    }
    hits->count = write_index;
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
}

static JPH_Shape *query_begin_with_shape(CPH_PhysicsWorld *world, int shape_type, const float params[4],
                                         JPH_BodyID *ignore_body, BodyHandle ignore_handle) {
    CPH_SHADOW_LOCK(&world->shadow_lock);
    wait_until_can_query_locked(world);

    // The shadow lock serializes cache access. Avoid taking jph_trampoline_lock
    // here: other native paths may hold it while waiting for shadow_lock.
    JPH_Shape *shape = CPH_FindOrCreateShapeLocked(world, shape_type, params);

    if (shape && ignore_body) {
        *ignore_body         = kJphInvalidBodyId;
        uint32_t ignore_slot = 0;
        if (query_handle_is_valid_locked(world, ignore_handle, &ignore_slot)) {
            const size_t body_count = atomic_load_explicit(&world->count, memory_order_relaxed);
            const uint32_t dense    = world->slot_to_dense[ignore_slot];
            if (dense < body_count) {
                *ignore_body = world->body_ids[dense];
            }
        }
    }

    // Match the binding's former lock scope: shape/ignore resolution and query
    // registration are atomic with respect to step flags and shadow mutations.
    if (shape) {
        atomic_fetch_add_explicit(&world->active_queries, 1, memory_order_acquire);
    }
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return shape;
}

static JPH_BodyID query_begin_with_ignore(CPH_PhysicsWorld *world, BodyHandle ignore_handle) {
    CPH_SHADOW_LOCK(&world->shadow_lock);
    wait_until_can_query_locked(world);

    JPH_BodyID ignore_body = kJphInvalidBodyId;
    uint32_t ignore_slot   = 0;
    if (query_handle_is_valid_locked(world, ignore_handle, &ignore_slot)) {
        const size_t body_count = atomic_load_explicit(&world->count, memory_order_relaxed);
        const uint32_t dense    = world->slot_to_dense[ignore_slot];
        if (dense < body_count) {
            ignore_body = world->body_ids[dense];
        }
    }

    atomic_fetch_add_explicit(&world->active_queries, 1, memory_order_acquire);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return ignore_body;
}

typedef struct CPH_QueryFilters {
    JPH_BroadPhaseLayerFilter *broad_phase;
    JPH_ObjectLayerFilter *object_layer;
    JPH_BodyFilter *body;
} CPH_QueryFilters;

static void destroy_query_filters(CPH_QueryFilters *filters) {
    if (filters->body) {
        JPH_BodyFilter_Destroy(filters->body);
        filters->body = nullptr;
    }
    if (filters->object_layer) {
        JPH_ObjectLayerFilter_Destroy(filters->object_layer);
        filters->object_layer = nullptr;
    }
    if (filters->broad_phase) {
        JPH_BroadPhaseLayerFilter_Destroy(filters->broad_phase);
        filters->broad_phase = nullptr;
    }
}

static PhysicsQueryStatus create_query_filters(CPH_QueryFilters *filters, bool with_body_filter,
                                               void *body_user_data) {
    *filters              = (CPH_QueryFilters){};
    filters->broad_phase  = JPH_BroadPhaseLayerFilter_Create(nullptr);
    filters->object_layer = JPH_ObjectLayerFilter_Create(nullptr);
    if (with_body_filter) {
        filters->body = JPH_BodyFilter_Create(body_user_data);
    }

    if (!filters->broad_phase || !filters->object_layer || (with_body_filter && !filters->body)) {
        destroy_query_filters(filters);
        return PHYSICS_QUERY_STATUS_OUT_OF_MEMORY;
    }
    return PHYSICS_QUERY_STATUS_OK;
}

typedef struct CPH_OverlapContext {
    CPH_PhysicsWorld *world;
    CPH_PhysicsQueryHitBuffer *hits;
    PhysicsQueryStatus status;
} CPH_OverlapContext;

static void overlap_record_hit(CPH_OverlapContext *context, JPH_BodyID body_id) {
    if (context->status != PHYSICS_QUERY_STATUS_OK) {
        return;
    }

    const BodyHandle handle =
        JPH_BodyInterface_GetUserData(context->world->body_interface, body_id);
    if (handle && !query_hit_buffer_append(context->hits, handle)) {
        context->status = PHYSICS_QUERY_STATUS_OUT_OF_MEMORY;
    }
}

static float overlap_callback_narrow(void *user_data, const JPH_CollideShapeResult *result) {
    overlap_record_hit((CPH_OverlapContext *)user_data, result->bodyID2);
    return 1.0f;
}

static float overlap_callback_broad(void *user_data, JPH_BodyID body_id) {
    overlap_record_hit((CPH_OverlapContext *)user_data, body_id);
    return 1.0f;
}

PhysicsQueryStatus CPH_PhysicsWorldOverlapSphere(CPH_PhysicsWorld *world, JPH_RVec3 center, float radius,
                                                CPH_PhysicsQueryHitBuffer *hits) {
    if (!world || !hits || !hits->hits) {
        return PHYSICS_QUERY_STATUS_INVALID_ARGUMENT;
    }

    const float shape_params[4] = {radius, 0.0f, 0.0f, 0.0f};
    JPH_Shape *shape = query_begin_with_shape(world, CULV_SHAPE_SPHERE, shape_params, nullptr, 0);
    if (!shape) {
        return PHYSICS_QUERY_STATUS_SHAPE_FAILURE;
    }

    CPH_QueryFilters filters      = {};
    PhysicsQueryStatus status = create_query_filters(&filters, true, nullptr);
    if (status != PHYSICS_QUERY_STATUS_OK) {
        CPH_PhysicsWorldQueryEnd(world);
        return status;
    }

    JPH_RMat4 transform;
    JPH_Quat rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    JPH_RMat4_RotationTranslation(&transform, &rotation, &center);

    const JPH_Vec3 scale  = {1.0f, 1.0f, 1.0f};
    JPH_RVec3 base_offset = {0.0, 0.0, 0.0};
    JPH_CollideShapeSettings settings;
    JPH_CollideShapeSettings_Init(&settings);

    CPH_OverlapContext context = {.world = world, .hits = hits, .status = PHYSICS_QUERY_STATUS_OK};
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(world->system);
    JPH_NarrowPhaseQuery_CollideShape(query, shape, &scale, &transform, &settings, &base_offset,
                                      overlap_callback_narrow, &context, filters.broad_phase,
                                      filters.object_layer, filters.body, nullptr);

    destroy_query_filters(&filters);
    if (context.status == PHYSICS_QUERY_STATUS_OK) {
        filter_query_hits(world, hits);
    }
    CPH_PhysicsWorldQueryEnd(world);
    return context.status;
}

PhysicsQueryStatus CPH_PhysicsWorldOverlapAabb(CPH_PhysicsWorld *world, JPH_RVec3 min, JPH_RVec3 max,
                                              CPH_PhysicsQueryHitBuffer *hits) {
    if (!world || !hits || !hits->hits) {
        return PHYSICS_QUERY_STATUS_INVALID_ARGUMENT;
    }

    query_begin(world);

    CPH_QueryFilters filters      = {};
    PhysicsQueryStatus status = create_query_filters(&filters, false, nullptr);
    if (status != PHYSICS_QUERY_STATUS_OK) {
        CPH_PhysicsWorldQueryEnd(world);
        return status;
    }

    JPH_AABox box;
    box.min = (JPH_Vec3){(float)min.x, (float)min.y, (float)min.z};
    box.max = (JPH_Vec3){(float)max.x, (float)max.y, (float)max.z};

    CPH_OverlapContext context = {.world = world, .hits = hits, .status = PHYSICS_QUERY_STATUS_OK};
    const JPH_BroadPhaseQuery *query = JPH_PhysicsSystem_GetBroadPhaseQuery(world->system);
    JPH_BroadPhaseQuery_CollideAABox(query, &box, overlap_callback_broad, &context,
                                     filters.broad_phase, filters.object_layer);

    destroy_query_filters(&filters);
    if (context.status == PHYSICS_QUERY_STATUS_OK) {
        filter_query_hits(world, hits);
    }
    CPH_PhysicsWorldQueryEnd(world);
    return context.status;
}

static float ray_direction_scale(JPH_Vec3 direction, float max_distance, float *magnitude_sq) {
    const float length_sq =
        direction.x * direction.x + direction.y * direction.y + direction.z * direction.z;
    *magnitude_sq = length_sq;
    if (length_sq < kQueryDirectionEpsilonSquared) {
        return 0.0f;
    }
    if (fabsf(length_sq - 1.0f) < kQueryDirectionUnitTolerance) {
        return max_distance;
    }
    return max_distance * CPH_FastRsqrt(length_sq);
}

PhysicsQueryStatus CPH_PhysicsWorldRaycast(CPH_PhysicsWorld *world, JPH_RVec3 origin, JPH_Vec3 direction,
                                         float max_distance, BodyHandle ignore_handle,
                                         CPH_PhysicsRaycastHit *out_hit) {
    if (!world || !out_hit) {
        return PHYSICS_QUERY_STATUS_INVALID_ARGUMENT;
    }
    *out_hit = (CPH_PhysicsRaycastHit){};

    float magnitude_sq = 0.0f;
    const float scale  = ray_direction_scale(direction, max_distance, &magnitude_sq);
    if (magnitude_sq < kQueryDirectionEpsilonSquared) {
        return PHYSICS_QUERY_STATUS_OK;
    }

    // A missing/stale ignored handle intentionally means "ignore nothing".
    const JPH_BodyID ignore_body   = query_begin_with_ignore(world, ignore_handle);
    CPH_CastShapeFilter filter_context = {.ignore_id = ignore_body};
    CPH_QueryFilters filters           = {};
    PhysicsQueryStatus status      = create_query_filters(&filters, true, &filter_context);
    if (status != PHYSICS_QUERY_STATUS_OK) {
        CPH_PhysicsWorldQueryEnd(world);
        return status;
    }

    const JPH_Vec3 ray_direction = {direction.x * scale, direction.y * scale, direction.z * scale};
    JPH_RayCastResult ray_hit = {.bodyID = kJphInvalidBodyId, .fraction = 1.0f, .subShapeID2 = 0};
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(world->system);
    const bool has_hit =
        JPH_NarrowPhaseQuery_CastRay(query, &origin, &ray_direction, &ray_hit, filters.broad_phase,
                                     filters.object_layer, filters.body);

    if (has_hit) {
        const BodyHandle handle =
            JPH_BodyInterface_GetUserData(world->body_interface, ray_hit.bodyID);
        bool valid_handle = false;
        if (handle) {
            CPH_SHADOW_LOCK(&world->shadow_lock);
            valid_handle = query_handle_is_valid_locked(world, handle, nullptr);
            CPH_SHADOW_UNLOCK(&world->shadow_lock);
        }

        if (valid_handle) {
            JPH_Vec3 normal = {};
            const JPH_BodyLockInterface *lock_interface =
                JPH_PhysicsSystem_GetBodyLockInterfaceNoLock(world->system);
            JPH_BodyLockRead body_lock;
            JPH_BodyLockInterface_LockRead(lock_interface, ray_hit.bodyID, &body_lock);
            if (body_lock.body) {
                const JPH_RVec3 hit_position = {
                    origin.x + (JPH_Real)ray_direction.x * ray_hit.fraction,
                    origin.y + (JPH_Real)ray_direction.y * ray_hit.fraction,
                    origin.z + (JPH_Real)ray_direction.z * ray_hit.fraction};
                JPH_Body_GetWorldSpaceSurfaceNormal(body_lock.body, ray_hit.subShapeID2,
                                                    &hit_position, &normal);
            }
            JPH_BodyLockInterface_UnlockRead(lock_interface, &body_lock);

            out_hit->handle   = handle;
            out_hit->fraction = ray_hit.fraction;
            out_hit->normal   = normal;
            out_hit->has_hit  = true;
        }
    }

    destroy_query_filters(&filters);
    CPH_PhysicsWorldQueryEnd(world);
    return PHYSICS_QUERY_STATUS_OK;
}

PhysicsQueryStatus CPH_PhysicsWorldRaycastBatch(CPH_PhysicsWorld *world, const float *starts,
                                               const float *directions, size_t count,
                                               float max_distance, RayCastBatchResult *results) {
    if (!world || (count && (!starts || !directions || !results)) ||
        count > SIZE_MAX / sizeof(*results)) {
        return PHYSICS_QUERY_STATUS_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return PHYSICS_QUERY_STATUS_OK;
    }

    memset(results, 0, count * sizeof(*results));
    query_begin(world);

    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(world->system);
    const JPH_BodyLockInterface *lock_interface =
        JPH_PhysicsSystem_GetBodyLockInterfaceNoLock(world->system);
    JPH_BodyInterface *body_interface = world->body_interface;

    for (size_t i = 0; i < count; ++i) {
        const size_t offset      = i * 3;
        const JPH_Vec3 direction = {directions[offset], directions[offset + 1],
                                    directions[offset + 2]};
        float magnitude_sq       = 0.0f;
        const float scale        = ray_direction_scale(direction, max_distance, &magnitude_sq);
        if (magnitude_sq < kQueryDirectionEpsilonSquared) {
            continue;
        }

        const JPH_Vec3 ray_direction = {direction.x * scale, direction.y * scale,
                                        direction.z * scale};
        const JPH_RVec3 origin       = {(JPH_Real)starts[offset], (JPH_Real)starts[offset + 1],
                                        (JPH_Real)starts[offset + 2]};
        JPH_RayCastResult ray_hit    = {
            .bodyID = kJphInvalidBodyId, .fraction = 1.0f, .subShapeID2 = 0};

        if (!JPH_NarrowPhaseQuery_CastRay(query, &origin, &ray_direction, &ray_hit, nullptr,
                                          nullptr, nullptr)) {
            continue;
        }

        const BodyHandle handle = JPH_BodyInterface_GetUserData(body_interface, ray_hit.bodyID);
        if (!handle) {
            continue;
        }

        RayCastBatchResult *result = &results[i];
        result->handle             = handle;
        result->fraction           = ray_hit.fraction;
        result->subShapeID         = ray_hit.subShapeID2;

        JPH_BodyLockRead body_lock;
        JPH_BodyLockInterface_LockRead(lock_interface, ray_hit.bodyID, &body_lock);
        if (body_lock.body) {
            const JPH_RVec3 hit_position = {origin.x + (JPH_Real)ray_direction.x * ray_hit.fraction,
                                            origin.y + (JPH_Real)ray_direction.y * ray_hit.fraction,
                                            origin.z +
                                                (JPH_Real)ray_direction.z * ray_hit.fraction};
            JPH_Vec3 normal;
            JPH_Body_GetWorldSpaceSurfaceNormal(body_lock.body, ray_hit.subShapeID2, &hit_position,
                                                &normal);
            result->nx = normal.x;
            result->ny = normal.y;
            result->nz = normal.z;
            result->px = (float)hit_position.x;
            result->py = (float)hit_position.y;
            result->pz = (float)hit_position.z;
        }
        JPH_BodyLockInterface_UnlockRead(lock_interface, &body_lock);
    }

    // Validate all collected body handles and resolve their material IDs while
    // the slot map is protected. active_queries prevents step/resize from
    // invalidating this batch while the Jolt queries are in flight.
    CPH_SHADOW_LOCK(&world->shadow_lock);
    const size_t body_count = atomic_load_explicit(&world->count, memory_order_relaxed);
    for (size_t i = 0; i < count; ++i) {
        RayCastBatchResult *result = &results[i];
        uint32_t slot              = 0;
        if (!query_handle_is_valid_locked(world, result->handle, &slot)) {
            memset(result, 0, sizeof(*result));
            continue;
        }

        const uint32_t dense = world->slot_to_dense[slot];
        if (dense < body_count && world->material_ids) {
            result->material_id = world->material_ids[dense];
        }
    }
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    CPH_PhysicsWorldQueryEnd(world);
    return PHYSICS_QUERY_STATUS_OK;
}

PhysicsQueryStatus CPH_PhysicsWorldShapecast(CPH_PhysicsWorld *world, int shape_type, JPH_RVec3 position,
                                           JPH_Quat rotation, JPH_Vec3 sweep_direction,
                                           const float size[4], BodyHandle ignore_handle,
                                           CPH_PhysicsShapecastHit *out_hit) {
    if (!world || !size || !out_hit) {
        return PHYSICS_QUERY_STATUS_INVALID_ARGUMENT;
    }
    *out_hit = (CPH_PhysicsShapecastHit){};

    const float magnitude_sq = sweep_direction.x * sweep_direction.x +
                               sweep_direction.y * sweep_direction.y +
                               sweep_direction.z * sweep_direction.z;
    if (magnitude_sq < kQueryDirectionEpsilonSquared) {
        return PHYSICS_QUERY_STATUS_OK;
    }

    const float shape_params[4] = {size[0], size[1], size[2], size[3]};
    JPH_BodyID ignore_body      = kJphInvalidBodyId;
    JPH_Shape *shape =
        query_begin_with_shape(world, shape_type, shape_params, &ignore_body, ignore_handle);
    if (!shape) {
        return PHYSICS_QUERY_STATUS_SHAPE_FAILURE;
    }

    JPH_RMat4 transform;
    JPH_RMat4_RotationTranslation(&transform, &rotation, &position);
    CPH_CastShapeContext context = {.has_hit = false};
    context.hit.fraction     = 1.0f;

    if (!CPH_ShapecastExecuteInternal(world, shape, &transform, &sweep_direction, ignore_body,
                                    &context)) {
        CPH_PhysicsWorldQueryEnd(world);
        return PHYSICS_QUERY_STATUS_OUT_OF_MEMORY;
    }

    if (context.has_hit) {
        const BodyHandle handle =
            JPH_BodyInterface_GetUserData(world->body_interface, context.hit.bodyID2);
        bool valid_handle = false;
        if (handle) {
            CPH_SHADOW_LOCK(&world->shadow_lock);
            valid_handle = query_handle_is_valid_locked(world, handle, nullptr);
            CPH_SHADOW_UNLOCK(&world->shadow_lock);
        }

        if (valid_handle) {
            float nx                     = -context.hit.penetrationAxis.x;
            float ny                     = -context.hit.penetrationAxis.y;
            float nz                     = -context.hit.penetrationAxis.z;
            const float normal_length_sq = nx * nx + ny * ny + nz * nz;
            if (normal_length_sq > kQueryDirectionEpsilonSquared) {
                const float inverse_length = CPH_FastRsqrt(normal_length_sq);
                nx *= inverse_length;
                ny *= inverse_length;
                nz *= inverse_length;
            }

            out_hit->handle   = handle;
            out_hit->fraction = context.hit.fraction;
            out_hit->point    = context.hit.contactPointOn2;
            out_hit->normal   = (JPH_Vec3){nx, ny, nz};
            out_hit->has_hit  = true;
        }
    }

    CPH_PhysicsWorldQueryEnd(world);
    return PHYSICS_QUERY_STATUS_OK;
}
