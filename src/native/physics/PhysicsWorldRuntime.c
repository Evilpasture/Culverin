#include "physics/PhysicsWorldRuntime.h"
#include "core/NativeAlloc.h"
#include "math/MathCore.h"
#include "physics/ContactEventDefinitions.h"
#include "physics/Query.h"
#include "physics/SlotState.h"
#include <string.h>

static constexpr size_t kInitialMaterialCapacity = 16;

bool CPH_PhysicsWorldAttachContactListener(CPH_PhysicsWorld *world) {
    if (!world || !world->system) {
        return false;
    }
    world->contact_listener = JPH_ContactListener_Create(world);
    if (!world->contact_listener) {
        return false;
    }
    JPH_PhysicsSystem_SetContactListener(world->system, world->contact_listener);
    return true;
}

void CPH_PhysicsWorldSyncLoadedState(CPH_PhysicsWorld *world, size_t saved_count) {
    if (!world) {
        return;
    }

    JPH_BodyID *CPH_RESTRICT body_ids              = world->body_ids;
    JPH_BodyInterface *CPH_RESTRICT body_interface = world->body_interface;
    CPH_PosStride *CPH_RESTRICT positions              = (CPH_PosStride *)world->positions;
    CPH_AuxStride *CPH_RESTRICT rotations              = (CPH_AuxStride *)world->rotations;
    CPH_AuxStride *CPH_RESTRICT linear_velocities      = (CPH_AuxStride *)world->linear_velocities;
    CPH_AuxStride *CPH_RESTRICT angular_velocities     = (CPH_AuxStride *)world->angular_velocities;

    for (size_t i = 0; i < saved_count; i++) {
        const JPH_BodyID body_id = body_ids[i];
        if (body_id == kJphInvalidBodyId) {
            continue;
        }

        JPH_RVec3 position       = {positions[i].x, positions[i].y, positions[i].z};
        JPH_Quat rotation        = {rotations[i].x, rotations[i].y, rotations[i].z, rotations[i].w};
        JPH_Vec3 linear_velocity = {linear_velocities[i].x, linear_velocities[i].y,
                                    linear_velocities[i].z};
        JPH_Vec3 angular_velocity = {angular_velocities[i].x, angular_velocities[i].y,
                                     angular_velocities[i].z};
        JPH_BodyInterface_SetPositionAndRotation(body_interface, body_id, &position, &rotation,
                                                 JPH_Activation_Activate);
        JPH_BodyInterface_SetLinearVelocity(body_interface, body_id, &linear_velocity);
        JPH_BodyInterface_SetAngularVelocity(body_interface, body_id, &angular_velocity);

        const uint32_t slot = world->dense_to_slot[i];
        const uint32_t generation =
            atomic_load_explicit(&world->generations[slot], memory_order_relaxed);
        const uint64_t raw_handle = make_handle(slot, generation);
        JPH_BodyInterface_SetUserData(body_interface, body_id, raw_handle);

        const uint32_t jolt_index = JPH_ID_TO_INDEX(body_id);
        if (world->id_to_handle_map && jolt_index <= world->max_jolt_bodies) {
            atomic_store_explicit(&world->id_to_handle_map[jolt_index], raw_handle,
                                  memory_order_relaxed);
        }
    }
}

static void lock_until_not_stepping(CPH_PhysicsWorld *world) {
    CPH_SHADOW_LOCK(&world->shadow_lock);
    if (!atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
        return;
    }

    atomic_fetch_add_explicit(&world->waiting_threads, 1, memory_order_relaxed);
    while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        CPH_NATIVE_MUTEX_LOCK(world->step_sync.mutex);
        while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
            CPH_NATIVE_COND_WAIT(world->step_sync.cond, world->step_sync.mutex);
        }
        CPH_NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
        CPH_SHADOW_LOCK(&world->shadow_lock);
    }
    atomic_fetch_sub_explicit(&world->waiting_threads, 1, memory_order_relaxed);
}

static void wait_until_not_querying_locked(CPH_PhysicsWorld *world) {
    while (atomic_load_explicit(&world->active_queries, memory_order_acquire) > 0) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        CPH_NATIVE_MUTEX_LOCK(world->step_sync.mutex);
        while (atomic_load_explicit(&world->active_queries, memory_order_relaxed) > 0) {
            CPH_NATIVE_COND_WAIT(world->step_sync.cond, world->step_sync.mutex);
        }
        CPH_NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
        CPH_SHADOW_LOCK(&world->shadow_lock);
    }
}

static bool calculate_snapshot_size(size_t body_count, size_t slot_capacity, size_t *out_size,
                                    size_t *out_positions_size, size_t *out_auxiliary_size,
                                    size_t *out_mapping_size) {
    const size_t header_size    = sizeof(size_t) + sizeof(double) + sizeof(size_t);
    size_t positions_size       = 0;
    size_t auxiliary_size       = 0;
    size_t mapping_size         = 0;
    size_t three_auxiliary_size = 0;
    size_t total_size           = header_size;
    if (__builtin_mul_overflow(body_count, sizeof(CPH_PosStride), &positions_size) ||
        __builtin_mul_overflow(body_count, sizeof(CPH_AuxStride), &auxiliary_size) ||
        __builtin_mul_overflow(slot_capacity, sizeof(uint32_t) * 3 + sizeof(uint8_t),
                               &mapping_size) ||
        __builtin_mul_overflow(auxiliary_size, (size_t)3, &three_auxiliary_size) ||
        __builtin_add_overflow(total_size, positions_size, &total_size) ||
        __builtin_add_overflow(total_size, three_auxiliary_size, &total_size) ||
        __builtin_add_overflow(total_size, mapping_size, &total_size)) {
        return false;
    }
    if (out_size) {
        *out_size = total_size;
    }
    if (out_positions_size) {
        *out_positions_size = positions_size;
    }
    if (out_auxiliary_size) {
        *out_auxiliary_size = auxiliary_size;
    }
    if (out_mapping_size) {
        *out_mapping_size = mapping_size;
    }
    return true;
}

PhysicsWorldSnapshotStatus CPH_PhysicsWorldSaveSnapshot(CPH_PhysicsWorld *world, uint8_t **out_data,
                                                       size_t *out_size) {
    if (!world || !out_data || !out_size) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    *out_data = nullptr;
    *out_size = 0;

    lock_until_not_stepping(world);
    wait_until_not_querying_locked(world);
    const size_t body_count    = atomic_load_explicit(&world->count, memory_order_acquire);
    const size_t slot_capacity = world->slot_capacity;
    const double saved_time    = world->time;
    size_t total_size          = 0;
    size_t position_size       = 0;
    size_t auxiliary_size      = 0;
    size_t mapping_size        = 0;
    if (!calculate_snapshot_size(body_count, slot_capacity, &total_size, &position_size,
                                 &auxiliary_size, &mapping_size)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_OVERFLOW;
    }
    uint8_t *snapshot = CulvNative_Malloc(total_size);
    if (!snapshot) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_OUT_OF_MEMORY;
    }

    uint8_t *cursor = snapshot;
    memcpy(cursor, &body_count, sizeof(body_count));
    cursor += sizeof(body_count);
    memcpy(cursor, &saved_time, sizeof(saved_time));
    cursor += sizeof(saved_time);
    memcpy(cursor, &slot_capacity, sizeof(slot_capacity));
    cursor += sizeof(slot_capacity);
    memcpy(cursor, world->positions, position_size);
    cursor += position_size;
    memcpy(cursor, world->rotations, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(cursor, world->linear_velocities, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(cursor, world->angular_velocities, auxiliary_size);
    cursor += auxiliary_size;

    uint32_t *out_generations = (uint32_t *)cursor;
    for (size_t i = 0; i < slot_capacity; i++) {
        out_generations[i] = atomic_load_explicit(&world->generations[i], memory_order_relaxed);
    }
    cursor += slot_capacity * sizeof(uint32_t);
    memcpy(cursor, world->slot_to_dense, slot_capacity * sizeof(uint32_t));
    cursor += slot_capacity * sizeof(uint32_t);
    memcpy(cursor, world->dense_to_slot, slot_capacity * sizeof(uint32_t));
    cursor += slot_capacity * sizeof(uint32_t);
    uint8_t *out_states = cursor;
    for (size_t i = 0; i < slot_capacity; i++) {
        out_states[i] = atomic_load_explicit(&world->slot_states[i], memory_order_relaxed);
    }

    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    *out_data = snapshot;
    *out_size = total_size;
    return PHYSICS_WORLD_SNAPSHOT_STATUS_OK;
}

PhysicsWorldSnapshotStatus CPH_PhysicsWorldLoadSnapshot(CPH_PhysicsWorld *world, const uint8_t *data,
                                                       size_t data_size, size_t *out_saved_count,
                                                       size_t *out_world_capacity,
                                                       size_t *out_saved_capacity) {
    if (!world || !data || !out_saved_count || !out_world_capacity || !out_saved_capacity) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    *out_saved_count    = 0;
    *out_world_capacity = 0;
    *out_saved_capacity = 0;

    lock_until_not_stepping(world);
    wait_until_not_querying_locked(world);
    const size_t header_size = sizeof(size_t) + sizeof(double) + sizeof(size_t);
    if (data_size < header_size) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }

    const uint8_t *cursor = data;
    size_t saved_count    = 0;
    double saved_time     = 0.0;
    size_t saved_capacity = 0;
    memcpy(&saved_count, cursor, sizeof(saved_count));
    cursor += sizeof(saved_count);
    memcpy(&saved_time, cursor, sizeof(saved_time));
    cursor += sizeof(saved_time);
    memcpy(&saved_capacity, cursor, sizeof(saved_capacity));
    cursor += sizeof(saved_capacity);

    *out_world_capacity = world->slot_capacity;
    *out_saved_capacity = saved_capacity;
    if (saved_capacity != world->slot_capacity) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_CAPACITY_MISMATCH;
    }
    if (saved_count > saved_capacity || saved_count > world->max_jolt_bodies) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }

    size_t expected_size  = 0;
    size_t position_size  = 0;
    size_t auxiliary_size = 0;
    size_t mapping_size   = 0;
    if (!calculate_snapshot_size(saved_count, saved_capacity, &expected_size, &position_size,
                                 &auxiliary_size, &mapping_size)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }
    if (data_size != expected_size) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }

    atomic_store_explicit(&world->count, saved_count, memory_order_relaxed);
    world->time = saved_time;
    memcpy(world->positions, cursor, position_size);
    cursor += position_size;
    memcpy(world->rotations, cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(world->linear_velocities, cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(world->angular_velocities, cursor, auxiliary_size);
    cursor += auxiliary_size;

    const uint32_t *in_generations = (const uint32_t *)cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        atomic_store_explicit(&world->generations[i], in_generations[i], memory_order_relaxed);
    }
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(world->slot_to_dense, cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(world->dense_to_slot, cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    const uint8_t *in_states = cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        atomic_store_explicit(&world->slot_states[i], in_states[i], memory_order_relaxed);
    }

    size_t free_count = 0;
    for (uint32_t i = 0; i < saved_capacity; i++) {
        if (atomic_load_explicit(&world->slot_states[i], memory_order_relaxed) == SLOT_EMPTY) {
            world->free_slots[free_count++] = i;
        }
    }
    atomic_store_explicit(&world->free_count, free_count, memory_order_release);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    CPH_PhysicsWorldSyncLoadedState(world, saved_count);
    *out_saved_count = saved_count;
    return PHYSICS_WORLD_SNAPSHOT_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldTakeContactEvents(CPH_PhysicsWorld *world,
                                                            struct CPH_ContactEvent **out_events,
                                                            size_t *out_count) {
    if (!world || !out_events || !out_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    *out_events = nullptr;
    *out_count  = 0;

    lock_until_not_stepping(world);
    size_t count = atomic_load_explicit(&world->contact_atomic_idx, memory_order_acquire);
    if (count > world->contact_max_capacity) {
        count = world->contact_max_capacity;
    }
    if (count == 0) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }
    if (count > SIZE_MAX / sizeof(struct CPH_ContactEvent)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }

    struct CPH_ContactEvent *events =
        CulvNative_AlignedMalloc(count * sizeof(struct CPH_ContactEvent), kMemoryAlignmentSize);
    if (!events) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    memcpy(events, world->contact_buffer, count * sizeof(struct CPH_ContactEvent));
    atomic_store_explicit(&world->contact_atomic_idx, 0, memory_order_relaxed);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    *out_events = events;
    *out_count  = count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldSetGravity(CPH_PhysicsWorld *world, JPH_Vec3 gravity) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    const size_t current_count = atomic_load_explicit(&world->count, memory_order_acquire);
    JPH_PhysicsSystem_SetGravity(world->system, &gravity);

    if (CPH_UNLIKELY(current_count > UINT32_MAX)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }

    if (current_count > 0) {
        JPH_BodyInterface_ActivateBodies(world->body_interface, world->body_ids,
                                         (uint32_t)current_count);
    }

    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetGravity(CPH_PhysicsWorld *world, JPH_Vec3 *out_gravity) {
    if (!world || !out_gravity) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    CPH_SHADOW_LOCK(&world->shadow_lock);
    JPH_PhysicsSystem_GetGravity(world->system, out_gravity);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetBodyStats(CPH_PhysicsWorld *world, BodyHandle handle,
                                                       CPH_PhysicsBodyStats *out_stats) {
    if (!world || !out_stats) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint32_t dense       = world->slot_to_dense[slot];
    out_stats->position        = ((CPH_PosStride *)world->positions)[dense];
    out_stats->rotation        = ((CPH_AuxStride *)world->rotations)[dense];
    out_stats->linear_velocity = ((CPH_AuxStride *)world->linear_velocities)[dense];
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

static PhysicsWorldRuntimeStatus
lock_soft_body_for_read(CPH_PhysicsWorld *world, BodyHandle handle, JPH_BodyLockRead *out_lock,
                        const JPH_BodyLockInterface **out_interface, uint32_t *out_count) {
    if (!world || !out_lock || !out_interface || !out_count || !world->system) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }
    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStrict).is_immediate) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }
    const uint32_t dense     = world->slot_to_dense[slot];
    const JPH_BodyID body_id = world->body_ids[dense];
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    const JPH_BodyLockInterface *lock_interface =
        JPH_PhysicsSystem_GetBodyLockInterface(world->system);
    JPH_BodyLockInterface_LockRead(lock_interface, body_id, out_lock);
    if (!out_lock->body) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }
    if (!JPH_Body_IsSoftBody(out_lock->body)) {
        JPH_BodyLockInterface_UnlockRead(lock_interface, out_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_WRONG_BODY_TYPE;
    }

    *out_count     = JPH_Body_GetSoftBodyVertexCount(out_lock->body);
    *out_interface = lock_interface;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetSoftBodyVertexCount(CPH_PhysicsWorld *world,
                                                                   BodyHandle handle,
                                                                   uint32_t *out_count) {
    if (!out_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    JPH_BodyLockRead lock;
    const JPH_BodyLockInterface *lock_interface = nullptr;
    PhysicsWorldRuntimeStatus status =
        lock_soft_body_for_read(world, handle, &lock, &lock_interface, out_count);
    if (status == PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
    }
    return status;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetSoftBodyVertexPosition(CPH_PhysicsWorld *world,
                                                                      BodyHandle handle,
                                                                      uint32_t vertex_index,
                                                                      JPH_Vec3 *out_position,
                                                                      uint32_t *out_count) {
    if (!out_position || !out_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    JPH_BodyLockRead lock;
    const JPH_BodyLockInterface *lock_interface = nullptr;
    const PhysicsWorldRuntimeStatus status =
        lock_soft_body_for_read(world, handle, &lock, &lock_interface, out_count);
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return status;
    }
    if (vertex_index >= *out_count) {
        JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_RANGE;
    }
    JPH_Body_GetSoftBodyVertexPosition(lock.body, vertex_index, out_position);
    JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetSoftBodyLocalVertices(CPH_PhysicsWorld *world,
                                                                     BodyHandle handle,
                                                                     JPH_Vec3 **out_positions,
                                                                     uint32_t *out_count) {
    if (!out_positions || !out_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    *out_positions = nullptr;
    *out_count     = 0;
    JPH_BodyLockRead lock;
    const JPH_BodyLockInterface *lock_interface = nullptr;
    const PhysicsWorldRuntimeStatus status =
        lock_soft_body_for_read(world, handle, &lock, &lock_interface, out_count);
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return status;
    }

    size_t allocation_size = 0;
    if (__builtin_mul_overflow((size_t)*out_count, sizeof(JPH_Vec3), &allocation_size)) {
        JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }
    JPH_Vec3 *positions = CulvNative_Malloc(allocation_size);
    if (!positions && *out_count != 0) {
        JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    uint32_t result_count = 0;
    JPH_Body_GetSoftBodyVertexPositions(lock.body, positions, *out_count, &result_count);
    JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
    *out_positions = positions;
    *out_count     = result_count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldApplyBuoyancy(CPH_PhysicsWorld *world, BodyHandle handle,
                                                       JPH_Real surface_y, float buoyancy,
                                                       float linear_drag, float angular_drag,
                                                       JPH_Vec3 fluid_velocity, float dt,
                                                       bool *out_submerged) {
    if (!world || !out_submerged) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    *out_submerged = false;

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStrict).is_immediate) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    const uint32_t dense     = world->slot_to_dense[slot];
    const JPH_BodyID body_id = world->body_ids[dense];
    atomic_fetch_add_explicit(&world->active_queries, 1, memory_order_acquire);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    JPH_BodyInterface_ActivateBody(world->body_interface, body_id);
    JPH_Vec3 gravity;
    JPH_PhysicsSystem_GetGravity(world->system, &gravity);
    JPH_RVec3 surface_position = {0, surface_y, 0};
    JPH_Vec3 surface_normal    = {0, 1.0f, 0};
    *out_submerged             = JPH_BodyInterface_ApplyBuoyancyImpulse(
        world->body_interface, body_id, &surface_position, &surface_normal, buoyancy, linear_drag,
        angular_drag, &fluid_velocity, &gravity, dt);
    CPH_PhysicsWorldQueryEnd(world);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldApplyBuoyancyBatch(
    CPH_PhysicsWorld *world, const BodyHandle *handles, size_t handle_count, JPH_Real surface_y,
    float buoyancy, float linear_drag, float angular_drag, JPH_Vec3 fluid_velocity, float dt) {
    if (!world || (handle_count > 0 && !handles)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    if (handle_count == 0) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }
    if (CPH_UNLIKELY(handle_count > SIZE_MAX / sizeof(JPH_BodyID))) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }

    JPH_BodyID *body_ids = (JPH_BodyID *)CulvNative_Malloc(handle_count * sizeof(JPH_BodyID));
    if (!body_ids) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    size_t valid_count = 0;
    lock_until_not_stepping(world);
    for (size_t i = 0; i < handle_count; i++) {
        uint32_t slot = 0;
        if (!unpack_handle(world, handles[i], &slot)) {
            continue;
        }
        const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
        if (get_slot_predicate(state, kMaskImmStrict).is_immediate) {
            body_ids[valid_count++] = world->body_ids[world->slot_to_dense[slot]];
        }
    }

    if (valid_count > 0) {
        atomic_fetch_add_explicit(&world->active_queries, 1, memory_order_acquire);
    }
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    if (valid_count > 0) {
        JPH_Vec3 gravity;
        JPH_PhysicsSystem_GetGravity(world->system, &gravity);
        JPH_RVec3 surface_position = {0, surface_y, 0};
        JPH_Vec3 surface_normal    = {0, 1.0f, 0};
        for (size_t i = 0; i < valid_count; i++) {
            JPH_BodyInterface_ActivateBody(world->body_interface, body_ids[i]);
            JPH_BodyInterface_ApplyBuoyancyImpulse(
                world->body_interface, body_ids[i], &surface_position, &surface_normal, buoyancy,
                linear_drag, angular_drag, &fluid_velocity, &gravity, dt);
        }
        CPH_PhysicsWorldQueryEnd(world);
    }

    CulvNative_Free(body_ids);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetMotionType(CPH_PhysicsWorld *world, BodyHandle handle,
                                                        JPH_MotionType *out_motion_type) {
    if (!world || !out_motion_type) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        const uint32_t dense     = world->slot_to_dense[slot];
        const JPH_BodyID body_id = world->body_ids[dense];
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        *out_motion_type = JPH_BodyInterface_GetMotionType(world->body_interface, body_id);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    if (state == SLOT_PENDING_CREATE) {
        return PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE;
    }
    return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetUserData(CPH_PhysicsWorld *world, BodyHandle handle,
                                                      uint64_t *out_user_data) {
    if (!world || !out_user_data) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_executable) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint32_t dense = world->slot_to_dense[slot];
    *out_user_data       = world->user_data[dense];
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetIndex(CPH_PhysicsWorld *world, BodyHandle handle,
                                                  uint32_t *out_index) {
    if (!world || !out_index) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    *out_index = world->slot_to_dense[slot];
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

bool CPH_PhysicsWorldIsAlive(CPH_PhysicsWorld *world, BodyHandle handle) {
    if (!world) {
        return false;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    bool alive    = false;
    if (unpack_handle(world, handle, &slot)) {
        const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
        alive               = get_slot_predicate(state, kMaskImmStandard).is_executable;
    }
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return alive;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldIsActive(CPH_PhysicsWorld *world, BodyHandle handle,
                                                  bool *out_active) {
    if (!world || !out_active) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        *out_active = false;
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    const uint32_t dense     = world->slot_to_dense[slot];
    const JPH_BodyID body_id = world->body_ids[dense];
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    *out_active = JPH_BodyInterface_IsActive(world->body_interface, body_id);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus
CPH_PhysicsWorldGetActiveIndices(CPH_PhysicsWorld *world, uint32_t **out_indices, size_t *out_count) {
    if (!world || !out_indices || !out_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    *out_indices = nullptr;
    *out_count   = 0;
    lock_until_not_stepping(world);
    const size_t count = atomic_load_explicit(&world->count, memory_order_acquire);
    if (count == 0) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    JPH_BodyID *body_ids = (JPH_BodyID *)CulvNative_Malloc(count * sizeof(JPH_BodyID));
    if (!body_ids) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    memcpy(body_ids, world->body_ids, count * sizeof(JPH_BodyID));
    CPH_SHADOW_UNLOCK(&world->shadow_lock);

    uint32_t *indices = (uint32_t *)CulvNative_Malloc(count * sizeof(uint32_t));
    if (!indices) {
        CulvNative_Free(body_ids);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    size_t active_count = 0;
    for (size_t i = 0; i < count; i++) {
        if (body_ids[i] != kJphInvalidBodyId &&
            JPH_BodyInterface_IsActive(world->body_interface, body_ids[i])) {
            indices[active_count++] = (uint32_t)i;
        }
    }
    CulvNative_Free(body_ids);
    *out_indices = indices;
    *out_count   = active_count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldGetRenderState(CPH_PhysicsWorld *world, float alpha,
                                                         float **out_state, size_t *out_count) {
    if (!world || !out_state || !out_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    *out_state = nullptr;
    *out_count = 0;
    lock_until_not_stepping(world);
    const size_t count = atomic_load_explicit(&world->count, memory_order_acquire);
    if (count == 0) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    if (CPH_UNLIKELY(count > SIZE_MAX / (7 * sizeof(float)))) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }
    float *state = (float *)CulvNative_Malloc(count * 7 * sizeof(float));
    if (!state) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    culverin_compute_interpolation_loop(
        (CPH_PosStride *)world->positions, (CPH_PosStride *)world->prev_positions,
        (CPH_AuxStride *)world->rotations, (CPH_AuxStride *)world->prev_rotations, alpha, state, count);
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    *out_state = state;
    *out_count = count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldSetCollisionFilter(CPH_PhysicsWorld *world, BodyHandle handle,
                                                             uint32_t category, uint32_t mask) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    wait_until_not_querying_locked(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = atomic_load_explicit(&world->slot_states[slot], memory_order_acquire);
    if (get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        const uint32_t dense     = world->slot_to_dense[slot];
        world->categories[dense] = category;
        world->masks[dense]      = mask;
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    if (state == SLOT_PENDING_CREATE) {
        return PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE;
    }
    return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
}

PhysicsWorldRuntimeStatus CPH_PhysicsWorldRegisterMaterial(CPH_PhysicsWorld *world, uint32_t id,
                                                          float friction, float restitution) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    lock_until_not_stepping(world);
    wait_until_not_querying_locked(world);

    for (size_t i = 0; i < world->material_count; i++) {
        if (world->materials[i].id == id) {
            world->materials[i].friction    = friction;
            world->materials[i].restitution = restitution;
            CPH_SHADOW_UNLOCK(&world->shadow_lock);
            return PHYSICS_WORLD_RUNTIME_STATUS_OK;
        }
    }

    if (world->material_count >= world->material_capacity) {
        const size_t new_capacity = world->material_capacity == 0 ? kInitialMaterialCapacity
                                                                  : world->material_capacity * 2;
        CPH_MaterialData *materials   = (CPH_MaterialData *)CulvNative_Realloc(
            world->materials, new_capacity * sizeof(CPH_MaterialData));
        if (CPH_UNLIKELY(!materials)) {
            CPH_SHADOW_UNLOCK(&world->shadow_lock);
            return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
        }
        world->materials         = materials;
        world->material_capacity = new_capacity;
    }

    world->materials[world->material_count++] = (CPH_MaterialData){id, friction, restitution};
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus
CPH_PhysicsWorldBuildDebugData(CPH_PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                               bool draw_bounding_box, bool draw_centers, bool wireframe,
                               CPH_DebugVertex **out_lines, size_t *out_line_count,
                               CPH_DebugVertex **out_triangles, size_t *out_triangle_count) {
    if (!world || !out_lines || !out_line_count || !out_triangles || !out_triangle_count) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    *out_lines          = nullptr;
    *out_line_count     = 0;
    *out_triangles      = nullptr;
    *out_triangle_count = 0;

    lock_until_not_stepping(world);
    world->debug_lines.count     = 0;
    world->debug_triangles.count = 0;

    JPH_DrawSettings settings;
    JPH_DrawSettings_InitDefault(&settings);
    settings.drawShape                 = draw_shapes;
    settings.drawShapeWireframe        = wireframe;
    settings.drawBoundingBox           = draw_bounding_box;
    settings.drawCenterOfMassTransform = draw_centers;

    if (draw_shapes || draw_bounding_box || draw_centers) {
        JPH_PhysicsSystem_DrawBodies(world->system, &settings, world->debug_renderer, nullptr);
    }
    if (draw_constraints) {
        JPH_PhysicsSystem_DrawConstraints(world->system, world->debug_renderer);
        JPH_PhysicsSystem_DrawConstraintLimits(world->system, world->debug_renderer);
    }

    const size_t line_bytes     = world->debug_lines.count * sizeof(CPH_DebugVertex);
    const size_t triangle_bytes = world->debug_triangles.count * sizeof(CPH_DebugVertex);
    CPH_DebugVertex *lines = line_bytes ? (CPH_DebugVertex *)CulvNative_Malloc(line_bytes) : nullptr;
    CPH_DebugVertex *triangles =
        triangle_bytes ? (CPH_DebugVertex *)CulvNative_Malloc(triangle_bytes) : nullptr;
    if ((line_bytes && !lines) || (triangle_bytes && !triangles)) {
        CulvNative_Free(lines);
        CulvNative_Free(triangles);
        CPH_SHADOW_UNLOCK(&world->shadow_lock);
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    if (line_bytes) {
        memcpy(lines, world->debug_lines.data, line_bytes);
    }
    if (triangle_bytes) {
        memcpy(triangles, world->debug_triangles.data, triangle_bytes);
    }

    *out_lines          = lines;
    *out_line_count     = world->debug_lines.count;
    *out_triangles      = triangles;
    *out_triangle_count = world->debug_triangles.count;
    CPH_SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}
