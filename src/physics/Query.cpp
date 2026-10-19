#include "physics/Query.hpp"
#include "core/Buffer.hpp"
#include "math/MathCore.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/SlotState.hpp"
#include <cmath>
#include <cstring>

static constexpr float kQueryDirectionEpsilonSquared = 1e-12f;
static constexpr float kQueryDirectionUnitTolerance  = 1e-4f;

static void wait_until_can_query(CPH_PhysicsWorld *world, std::unique_lock<std::mutex> &shadow) {
    const auto blocked = [&] {
        return world->is_stepping.load(std::memory_order::relaxed) ||
               world->step_requested.load(std::memory_order::relaxed);
    };
    if (blocked()) {
        world->waiting_threads.fetch_add(1, std::memory_order::relaxed);
        while (blocked()) {
            // Never hold shadow_lock while parked on the condition.
            shadow.unlock();
            {
                std::unique_lock sync(world->step_sync.mutex);
                world->step_sync.cond.wait(sync, [&] { return !blocked(); });
            }
            shadow.lock();
        }
        world->waiting_threads.fetch_sub(1, std::memory_order::relaxed);
    }
}

static void query_begin(CPH_PhysicsWorld *world) {
    std::unique_lock shadow(world->shadow_lock);
    wait_until_can_query(world, shadow);
    // The gate check and this increment are one critical section, so a
    // step cannot start between them.
    world->active_queries.fetch_add(1, std::memory_order::acquire);
}

void CPH_PhysicsWorldQueryEnd(CPH_PhysicsWorld *world) {
    if (!world) {
        return;
    }
    const std::scoped_lock sync(world->step_sync.mutex);
    if (world->active_queries.fetch_sub(1, std::memory_order::acq_rel) == 1) {
        world->step_sync.cond.notify_all();
    }
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

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return false;
    }

    if (slot_out) {
        *slot_out = slot;
    }
    return true;
}

static void filter_query_hits(CPH_PhysicsWorld *world, CPH_PhysicsQueryHitBuffer *hits) {
    const std::scoped_lock shadow(world->shadow_lock);
    size_t write_index = 0;
    for (size_t i = 0; i < hits->count; ++i) {
        if (query_handle_is_valid_locked(world, hits->hits[i], nullptr)) {
            hits->hits[write_index++] = hits->hits[i];
        }
    }
    hits->count = write_index;
}

static JPH_Shape *query_begin_with_shape(CPH_PhysicsWorld *world, int shape_type, const float params[4],
                                         JPH_BodyID *ignore_body, BodyHandle ignore_handle) {
    std::unique_lock shadow(world->shadow_lock);
    wait_until_can_query(world, shadow);

    // The shadow lock serializes cache access. Avoid taking jph_trampoline_lock
    // here: other native paths may hold it while waiting for shadow_lock.
    JPH_Shape *shape = CPH_FindOrCreateShapeLocked(world, shape_type, params);

    if (shape && ignore_body) {
        *ignore_body         = kJphInvalidBodyId;
        uint32_t ignore_slot = 0;
        if (query_handle_is_valid_locked(world, ignore_handle, &ignore_slot)) {
            const size_t body_count = (world->count).load(std::memory_order::relaxed);
            const uint32_t dense    = world->bodies.slot_to_dense[ignore_slot];
            if (dense < body_count) {
                *ignore_body = world->bodies.body_ids[dense];
            }
        }
    }

    // Match the binding's former lock scope: shape/ignore resolution and query
    // registration are atomic with respect to step flags and shadow mutations.
    if (shape) {
        (world->active_queries).fetch_add(1, std::memory_order::acquire);
    }
    return shape;
}

static JPH_BodyID query_begin_with_ignore(CPH_PhysicsWorld *world, BodyHandle ignore_handle) {
    std::unique_lock shadow(world->shadow_lock);
    wait_until_can_query(world, shadow);

    JPH_BodyID ignore_body = kJphInvalidBodyId;
    uint32_t ignore_slot   = 0;
    if (query_handle_is_valid_locked(world, ignore_handle, &ignore_slot)) {
        const size_t body_count = (world->count).load(std::memory_order::relaxed);
        const uint32_t dense    = world->bodies.slot_to_dense[ignore_slot];
        if (dense < body_count) {
            ignore_body = world->bodies.body_ids[dense];
        }
    }

    (world->active_queries).fetch_add(1, std::memory_order::acquire);
    return ignore_body;
}

struct CPH_QueryFilters {
    JoltPtr<JPH_BroadPhaseLayerFilter, JPH_BroadPhaseLayerFilter_Destroy> broad_phase;
    JoltPtr<JPH_ObjectLayerFilter, JPH_ObjectLayerFilter_Destroy> object_layer;
    JoltPtr<JPH_BodyFilter, JPH_BodyFilter_Destroy> body;
};

static PhysicsQueryStatus create_query_filters(CPH_QueryFilters &filters, bool with_body_filter,
                                               void *body_user_data) {
    filters.broad_phase.reset(JPH_BroadPhaseLayerFilter_Create(nullptr));
    filters.object_layer.reset(JPH_ObjectLayerFilter_Create(nullptr));
    if (with_body_filter) {
        filters.body.reset(JPH_BodyFilter_Create(body_user_data));
    }
    if (!filters.broad_phase || !filters.object_layer || (with_body_filter && !filters.body)) {
        return PHYSICS_QUERY_STATUS_OUT_OF_MEMORY;
    }
    return PHYSICS_QUERY_STATUS_OK;
}

struct CPH_OverlapContext {
    CPH_PhysicsWorld *world;
    CPH_PhysicsQueryHitBuffer *hits;
    PhysicsQueryStatus status;
};

static void overlap_record_hit(CPH_OverlapContext *context, JPH_BodyID body_id) {
    if (context->status != PHYSICS_QUERY_STATUS_OK) {
        return;
    }

    const BodyHandle handle =
        JPH_BodyInterface_GetUserData(context->world->body_interface, body_id);
    if (handle && !context->hits->append(handle)) {
        context->status = PHYSICS_QUERY_STATUS_OUT_OF_MEMORY;
    }
}

static float overlap_callback_narrow(void *user_data, const JPH_CollideShapeResult *result) {
    overlap_record_hit(static_cast<CPH_OverlapContext *>(user_data), result->bodyID2);
    return 1.0f;
}

static float overlap_callback_broad(void *user_data, JPH_BodyID body_id) {
    overlap_record_hit(static_cast<CPH_OverlapContext *>(user_data), body_id);
    return 1.0f;
}

PhysicsQueryStatus CPH_PhysicsWorldOverlapSphere(CPH_PhysicsWorld *world, JPH_RVec3 center, float radius,
                                                CPH_PhysicsQueryHitBuffer *hits) {
    if (!world || !hits || !hits->hits) {
        return PHYSICS_QUERY_STATUS_INVALID_ARGUMENT;
    }

    const float shape_params[4] = {radius, 0.0f, 0.0f, 0.0f};
    JPH_Shape *shape = query_begin_with_shape(world, CPH_SHAPE_SPHERE, shape_params, nullptr, 0);
    if (!shape) {
        return PHYSICS_QUERY_STATUS_SHAPE_FAILURE;
    }

    CPH_QueryFilters filters      = {};
    PhysicsQueryStatus status = create_query_filters(filters, true, nullptr);
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
                                      overlap_callback_narrow, &context, filters.broad_phase.get(),
                                      filters.object_layer.get(), filters.body.get(), nullptr);
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
    PhysicsQueryStatus status = create_query_filters(filters, false, nullptr);
    if (status != PHYSICS_QUERY_STATUS_OK) {
        CPH_PhysicsWorldQueryEnd(world);
        return status;
    }

    JPH_AABox box;
    box.min = JPH_Vec3{(float)min.x, (float)min.y, (float)min.z};
    box.max = JPH_Vec3{(float)max.x, (float)max.y, (float)max.z};

    CPH_OverlapContext context = {.world = world, .hits = hits, .status = PHYSICS_QUERY_STATUS_OK};
    const JPH_BroadPhaseQuery *query = JPH_PhysicsSystem_GetBroadPhaseQuery(world->system);
    JPH_BroadPhaseQuery_CollideAABox(query, &box, overlap_callback_broad, &context,
                                     filters.broad_phase.get(), filters.object_layer.get());
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
    if (std::fabs(length_sq - 1.0f) < kQueryDirectionUnitTolerance) {
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
    *out_hit = CPH_PhysicsRaycastHit{};

    float magnitude_sq = 0.0f;
    const float scale  = ray_direction_scale(direction, max_distance, &magnitude_sq);
    if (magnitude_sq < kQueryDirectionEpsilonSquared) {
        return PHYSICS_QUERY_STATUS_OK;
    }

    // A missing/stale ignored handle intentionally means "ignore nothing".
    const JPH_BodyID ignore_body   = query_begin_with_ignore(world, ignore_handle);
    CPH_CastShapeFilter filter_context = {ignore_body};
    CPH_QueryFilters filters           = {};
    PhysicsQueryStatus status      = create_query_filters(filters, true, &filter_context);
    if (status != PHYSICS_QUERY_STATUS_OK) {
        CPH_PhysicsWorldQueryEnd(world);
        return status;
    }

    const JPH_Vec3 ray_direction = {direction.x * scale, direction.y * scale, direction.z * scale};
    JPH_RayCastResult ray_hit = {.bodyID = kJphInvalidBodyId, .fraction = 1.0f, .subShapeID2 = 0};
    const JPH_NarrowPhaseQuery *query = JPH_PhysicsSystem_GetNarrowPhaseQuery(world->system);
    const bool has_hit =
        JPH_NarrowPhaseQuery_CastRay(query, &origin, &ray_direction, &ray_hit, filters.broad_phase.get(),
                                     filters.object_layer.get(), filters.body.get());

    if (has_hit) {
        const BodyHandle handle =
            JPH_BodyInterface_GetUserData(world->body_interface, ray_hit.bodyID);
        bool valid_handle = false;
        if (handle) {
            const std::scoped_lock shadow(world->shadow_lock);
            valid_handle = query_handle_is_valid_locked(world, handle, nullptr);
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

    std::memset(results, 0, count * sizeof(*results));
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
    const std::scoped_lock shadow(world->shadow_lock);
    const size_t body_count = (world->count).load(std::memory_order::relaxed);
    for (size_t i = 0; i < count; ++i) {
        RayCastBatchResult *result = &results[i];
        uint32_t slot              = 0;
        if (!query_handle_is_valid_locked(world, result->handle, &slot)) {
            std::memset(result, 0, sizeof(*result));
            continue;
        }

        const uint32_t dense = world->bodies.slot_to_dense[slot];
        if (dense < body_count && world->bodies.material_ids) {
            result->material_id = world->bodies.material_ids[dense];
        }
    }

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
    *out_hit = CPH_PhysicsShapecastHit{};

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
    CPH_CastShapeContext context{}; // has_hit = false
    context.hit.fraction = 1.0f;

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
            const std::scoped_lock shadow(world->shadow_lock);
            valid_handle = query_handle_is_valid_locked(world, handle, nullptr);
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
            out_hit->normal   = JPH_Vec3{nx, ny, nz};
            out_hit->has_hit  = true;
        }
    }

    CPH_PhysicsWorldQueryEnd(world);
    return PHYSICS_QUERY_STATUS_OK;
}
