#include "physics/PhysicsWorldRuntime.hpp"
#include "core/Buffer.hpp"
#include "math/MathCore.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/Query.hpp"
#include "physics/SlotState.hpp"
#include <cstring>
#include <limits>


namespace CPH {

static constexpr size_t kInitialMaterialCapacity = 16;

bool PhysicsWorldAttachContactListener(PhysicsWorld *world) {
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

void PhysicsWorldSyncLoadedState(PhysicsWorld *world, size_t saved_count) {
    if (!world) {
        return;
    }

    JPH_BodyID *CPH_RESTRICT body_ids              = world->bodies.body_ids.data();
    JPH_BodyInterface *CPH_RESTRICT body_interface = world->body_interface;
    PosStride *CPH_RESTRICT positions              = world->bodies.positions.data();
    AuxStride *CPH_RESTRICT rotations              = world->bodies.rotations.data();
    AuxStride *CPH_RESTRICT linear_velocities      = world->bodies.linear_velocities.data();
    AuxStride *CPH_RESTRICT angular_velocities     = world->bodies.angular_velocities.data();

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

        const uint32_t slot = world->bodies.dense_to_slot[i];
        const uint32_t generation =
            (world->bodies.generations[slot]).load(std::memory_order::relaxed);
        const uint64_t raw_handle = make_handle(slot, generation);
        JPH_BodyInterface_SetUserData(body_interface, body_id, raw_handle);

        const uint32_t jolt_index = JPH_ID_TO_INDEX(body_id);
        if (world->id_to_handle_map && jolt_index <= world->max_jolt_bodies) {
            (world->id_to_handle_map[jolt_index]).store(raw_handle, std::memory_order::relaxed);
        }
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
    if (__builtin_mul_overflow(body_count, sizeof(PosStride), &positions_size) ||
        __builtin_mul_overflow(body_count, sizeof(AuxStride), &auxiliary_size) ||
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

PhysicsWorldSnapshotStatus PhysicsWorldSaveSnapshot(PhysicsWorld *world,
                                                       Buffer<std::uint8_t> &out_data) {
    if (!world) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    out_data.reset();

    auto shadow = LockShadowNotStepping(world);
    WaitUntilNotQuerying(world, shadow);
    const std::size_t body_count    = world->count.load(std::memory_order::acquire);
    const std::size_t slot_capacity = world->slot_capacity;
    const double saved_time         = world->time;
    std::size_t total_size          = 0;
    std::size_t position_size       = 0;
    std::size_t auxiliary_size      = 0;
    std::size_t mapping_size        = 0;
    if (!calculate_snapshot_size(body_count, slot_capacity, &total_size, &position_size,
                                 &auxiliary_size, &mapping_size)) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_OVERFLOW;
    }
    if (!out_data.allocate(total_size)) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_OUT_OF_MEMORY;
    }

    std::uint8_t *cursor = out_data.data();
    std::memcpy(cursor, &body_count, sizeof(body_count));
    cursor += sizeof(body_count);
    std::memcpy(cursor, &saved_time, sizeof(saved_time));
    cursor += sizeof(saved_time);
    std::memcpy(cursor, &slot_capacity, sizeof(slot_capacity));
    cursor += sizeof(slot_capacity);
    std::memcpy(cursor, world->bodies.positions.data(), position_size);
    cursor += position_size;
    std::memcpy(cursor, world->bodies.rotations.data(), auxiliary_size);
    cursor += auxiliary_size;
    std::memcpy(cursor, world->bodies.linear_velocities.data(), auxiliary_size);
    cursor += auxiliary_size;
    std::memcpy(cursor, world->bodies.angular_velocities.data(), auxiliary_size);
    cursor += auxiliary_size;

    auto *out_generations = reinterpret_cast<std::uint32_t *>(cursor);
    for (std::size_t i = 0; i < slot_capacity; i++) {
        out_generations[i] = world->bodies.generations[i].load(std::memory_order::relaxed);
    }
    cursor += slot_capacity * sizeof(std::uint32_t);
    std::memcpy(cursor, world->bodies.slot_to_dense.data(), slot_capacity * sizeof(std::uint32_t));
    cursor += slot_capacity * sizeof(std::uint32_t);
    std::memcpy(cursor, world->bodies.dense_to_slot.data(), slot_capacity * sizeof(std::uint32_t));
    cursor += slot_capacity * sizeof(std::uint32_t);
    for (std::size_t i = 0; i < slot_capacity; i++) {
        cursor[i] = world->bodies.slot_states[i].load(std::memory_order::relaxed);
    }
    return PHYSICS_WORLD_SNAPSHOT_STATUS_OK;
}


PhysicsWorldSnapshotStatus PhysicsWorldLoadSnapshot(PhysicsWorld *world, const uint8_t *data,
                                                       size_t data_size, size_t *out_saved_count,
                                                       size_t *out_world_capacity,
                                                       size_t *out_saved_capacity) {
    if (!world || !data || !out_saved_count || !out_world_capacity || !out_saved_capacity) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT;
    }
    *out_saved_count    = 0;
    *out_world_capacity = 0;
    *out_saved_capacity = 0;

    auto shadow = LockShadowNotStepping(world);
    WaitUntilNotQuerying(world, shadow);
    const size_t header_size = sizeof(size_t) + sizeof(double) + sizeof(size_t);
    if (data_size < header_size) {
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
        return PHYSICS_WORLD_SNAPSHOT_STATUS_CAPACITY_MISMATCH;
    }
    if (saved_count > saved_capacity || saved_count > world->max_jolt_bodies) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }

    size_t expected_size  = 0;
    size_t position_size  = 0;
    size_t auxiliary_size = 0;
    size_t mapping_size   = 0;
    if (!calculate_snapshot_size(saved_count, saved_capacity, &expected_size, &position_size,
                                 &auxiliary_size, &mapping_size)) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }
    if (data_size != expected_size) {
        return PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED;
    }

    (world->count).store(saved_count, std::memory_order::relaxed);
    world->time = saved_time;
    memcpy(world->bodies.positions.data(), cursor, position_size);
    cursor += position_size;
    memcpy(world->bodies.rotations.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(world->bodies.linear_velocities.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(world->bodies.angular_velocities.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;

    const uint32_t *in_generations = (const uint32_t *)cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        (world->bodies.generations[i]).store(in_generations[i], std::memory_order::relaxed);
    }
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(world->bodies.slot_to_dense.data(), cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(world->bodies.dense_to_slot.data(), cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    const uint8_t *in_states = cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        (world->bodies.slot_states[i]).store(in_states[i], std::memory_order::relaxed);
    }

    size_t free_count = 0;
    for (uint32_t i = 0; i < saved_capacity; i++) {
        if ((world->bodies.slot_states[i]).load(std::memory_order::relaxed) == std::to_underlying(SlotState::Empty)) {
            world->bodies.free_slots[free_count++] = i;
        }
    }
    (world->free_count).store(free_count, std::memory_order::release);

    PhysicsWorldSyncLoadedState(world, saved_count);
    *out_saved_count = saved_count;
    return PHYSICS_WORLD_SNAPSHOT_STATUS_OK;
}

PhysicsWorldRuntimeStatus
PhysicsWorldTakeContactEvents(PhysicsWorld *world, Buffer<ContactEvent> &out_events) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    out_events.reset();

    const auto shadow = LockShadowNotStepping(world);
    std::size_t count = world->contact_atomic_idx.load(std::memory_order::acquire);
    if (count > world->contact_max_capacity) {
        count = world->contact_max_capacity;
    }
    if (count == 0) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }
    if (!out_events.allocate(count)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    // Contact events contain atomics; the wire copy is a byte transfer,
    // same as the legacy implementation.
    std::memcpy(out_events.data(), world->contact_buffer.data(),
                count * sizeof(ContactEvent));
    world->contact_atomic_idx.store(0, std::memory_order::relaxed);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}


PhysicsWorldRuntimeStatus PhysicsWorldSetGravity(PhysicsWorld *world, JPH_Vec3 gravity) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    const size_t current_count = (world->count).load(std::memory_order::acquire);
    JPH_PhysicsSystem_SetGravity(world->system, &gravity);

    if (CPH_UNLIKELY(current_count > UINT32_MAX)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }

    if (current_count > 0) {
        JPH_BodyInterface_ActivateBodies(world->body_interface, world->bodies.body_ids.data(),
                                         (uint32_t)current_count);
    }
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus PhysicsWorldGetGravity(PhysicsWorld *world, JPH_Vec3 *out_gravity) {
    if (!world || !out_gravity) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    const std::scoped_lock shadow(world->shadow_lock);
    JPH_PhysicsSystem_GetGravity(world->system, out_gravity);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus PhysicsWorldGetBodyStats(PhysicsWorld *world, BodyHandle handle,
                                                       PhysicsBodyStats *out_stats) {
    if (!world || !out_stats) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint32_t dense       = world->bodies.slot_to_dense[slot];
    out_stats->position        = (world->bodies.positions.data())[dense];
    out_stats->rotation        = (world->bodies.rotations.data())[dense];
    out_stats->linear_velocity = (world->bodies.linear_velocities.data())[dense];
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

static PhysicsWorldRuntimeStatus
lock_soft_body_for_read(PhysicsWorld *world, BodyHandle handle, JPH_BodyLockRead *out_lock,
                        const JPH_BodyLockInterface **out_interface, uint32_t *out_count) {
    if (!world || !out_lock || !out_interface || !out_count || !world->system) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }
    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStrict).is_immediate) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }
    const uint32_t dense     = world->bodies.slot_to_dense[slot];
    const JPH_BodyID body_id = world->bodies.body_ids[dense];

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

PhysicsWorldRuntimeStatus PhysicsWorldGetSoftBodyVertexCount(PhysicsWorld *world,
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

PhysicsWorldRuntimeStatus PhysicsWorldGetSoftBodyVertexPosition(PhysicsWorld *world,
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

PhysicsWorldRuntimeStatus PhysicsWorldGetSoftBodyLocalVertices(
    PhysicsWorld *world, BodyHandle handle, Buffer<JPH_Vec3> &out_positions,
    std::uint32_t &out_count) {
    out_positions.reset();
    out_count = 0;
    JPH_BodyLockRead lock;
    const JPH_BodyLockInterface *lock_interface = nullptr;
    const PhysicsWorldRuntimeStatus status =
        lock_soft_body_for_read(world, handle, &lock, &lock_interface, &out_count);
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return status;
    }
    if (out_count != 0 && !out_positions.allocate(out_count)) {
        JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
        out_count = 0;
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    std::uint32_t result_count = 0;
    JPH_Body_GetSoftBodyVertexPositions(lock.body, out_positions.data(), out_count, &result_count);
    JPH_BodyLockInterface_UnlockRead(lock_interface, &lock);
    out_count = result_count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}


PhysicsWorldRuntimeStatus PhysicsWorldApplyBuoyancy(PhysicsWorld *world, BodyHandle handle,
                                                       JPH_Real surface_y, float buoyancy,
                                                       float linear_drag, float angular_drag,
                                                       JPH_Vec3 fluid_velocity, float dt,
                                                       bool *out_submerged) {
    if (!world || !out_submerged) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    *out_submerged = false;

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStrict).is_immediate) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    const uint32_t dense     = world->bodies.slot_to_dense[slot];
    const JPH_BodyID body_id = world->bodies.body_ids[dense];
    (world->active_queries).fetch_add(1, std::memory_order::acquire);

    JPH_BodyInterface_ActivateBody(world->body_interface, body_id);
    JPH_Vec3 gravity;
    JPH_PhysicsSystem_GetGravity(world->system, &gravity);
    JPH_RVec3 surface_position = {0, surface_y, 0};
    JPH_Vec3 surface_normal    = {0, 1.0f, 0};
    *out_submerged             = JPH_BodyInterface_ApplyBuoyancyImpulse(
        world->body_interface, body_id, &surface_position, &surface_normal, buoyancy, linear_drag,
        angular_drag, &fluid_velocity, &gravity, dt);
    PhysicsWorldQueryEnd(world);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus PhysicsWorldApplyBuoyancyBatch(
    PhysicsWorld *world, const BodyHandle *handles, size_t handle_count, JPH_Real surface_y,
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

    Buffer<JPH_BodyID> body_ids;
    if (!body_ids.allocate(handle_count)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }

    size_t valid_count = 0;
    auto shadow = LockShadowNotStepping(world);
    for (size_t i = 0; i < handle_count; i++) {
        uint32_t slot = 0;
        if (!unpack_handle(world, handles[i], &slot)) {
            continue;
        }
        const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
        if (get_slot_predicate(state, kMaskImmStrict).is_immediate) {
            body_ids[valid_count++] = world->bodies.body_ids[world->bodies.slot_to_dense[slot]];
        }
    }

    if (valid_count > 0) {
        (world->active_queries).fetch_add(1, std::memory_order::acquire);
    }

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
        PhysicsWorldQueryEnd(world);
    }

    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus PhysicsWorldGetMotionType(PhysicsWorld *world, BodyHandle handle,
                                                        JPH_MotionType *out_motion_type) {
    if (!world || !out_motion_type) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        const uint32_t dense     = world->bodies.slot_to_dense[slot];
        const JPH_BodyID body_id = world->bodies.body_ids[dense];
        *out_motion_type = JPH_BodyInterface_GetMotionType(world->body_interface, body_id);
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }
    if (state == std::to_underlying(SlotState::PendingCreate)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE;
    }
    return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
}

PhysicsWorldRuntimeStatus PhysicsWorldGetUserData(PhysicsWorld *world, BodyHandle handle,
                                                      uint64_t *out_user_data) {
    if (!world || !out_user_data) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_executable()) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint32_t dense = world->bodies.slot_to_dense[slot];
    *out_user_data       = world->bodies.user_data[dense];
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus PhysicsWorldGetIndex(PhysicsWorld *world, BodyHandle handle,
                                                  uint32_t *out_index) {
    if (!world || !out_index) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    *out_index = world->bodies.slot_to_dense[slot];
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

bool PhysicsWorldIsAlive(PhysicsWorld *world, BodyHandle handle) {
    if (!world) {
        return false;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    bool alive    = false;
    if (unpack_handle(world, handle, &slot)) {
        const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
        alive               = get_slot_predicate(state, kMaskImmStandard).is_executable();
    }
    return alive;
}

PhysicsWorldRuntimeStatus PhysicsWorldIsActive(PhysicsWorld *world, BodyHandle handle,
                                                  bool *out_active) {
    if (!world || !out_active) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        *out_active = false;
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }

    const uint32_t dense     = world->bodies.slot_to_dense[slot];
    const JPH_BodyID body_id = world->bodies.body_ids[dense];
    *out_active = JPH_BodyInterface_IsActive(world->body_interface, body_id);
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus
PhysicsWorldGetActiveIndices(PhysicsWorld *world, Buffer<std::uint32_t> &out_indices,
                                  std::size_t &out_count) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    out_indices.reset();
    out_count = 0;

    Buffer<JPH_BodyID> body_ids;
    std::size_t count = 0;
    {
        const auto shadow = LockShadowNotStepping(world);
        count             = world->count.load(std::memory_order::acquire);
        if (count == 0) {
            return PHYSICS_WORLD_RUNTIME_STATUS_OK;
        }
        if (!body_ids.allocate(count)) {
            return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
        }
        std::memcpy(body_ids.data(), world->bodies.body_ids.data(), count * sizeof(JPH_BodyID));
    }

    if (!out_indices.allocate(count)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    std::size_t active_count = 0;
    for (std::size_t i = 0; i < count; i++) {
        if (body_ids[i] != kJphInvalidBodyId &&
            JPH_BodyInterface_IsActive(world->body_interface, body_ids[i])) {
            out_indices[active_count++] = static_cast<std::uint32_t>(i);
        }
    }
    out_count = active_count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}


PhysicsWorldRuntimeStatus PhysicsWorldGetRenderState(PhysicsWorld *world, float alpha,
                                                          Buffer<float> &out_state,
                                                          std::size_t &out_count) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    out_state.reset();
    out_count = 0;

    const auto shadow  = LockShadowNotStepping(world);
    const std::size_t count = world->count.load(std::memory_order::acquire);
    if (count == 0) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }
    if (CPH_UNLIKELY(count > std::numeric_limits<std::size_t>::max() / 7)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW;
    }
    if (!out_state.allocate(count * 7)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    MathComputeInterpolationLoop(world->bodies.positions.data(),
                                     world->bodies.prev_positions.data(),
                                     world->bodies.rotations.data(),
                                     world->bodies.prev_rotations.data(), alpha,
                                     out_state.data(), count);
    out_count = count;
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}


PhysicsWorldRuntimeStatus PhysicsWorldSetCollisionFilter(PhysicsWorld *world, BodyHandle handle,
                                                             uint32_t category, uint32_t mask) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    WaitUntilNotQuerying(world, shadow);
    uint32_t slot = 0;
    if (!unpack_handle(world, handle, &slot)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
    }

    const uint8_t state = (world->bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        const uint32_t dense     = world->bodies.slot_to_dense[slot];
        world->bodies.categories[dense] = category;
        world->bodies.masks[dense]      = mask;
        return PHYSICS_WORLD_RUNTIME_STATUS_OK;
    }
    if (state == std::to_underlying(SlotState::PendingCreate)) {
        return PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE;
    }
    return PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE;
}

PhysicsWorldRuntimeStatus PhysicsWorldRegisterMaterial(PhysicsWorld *world, uint32_t id,
                                                          float friction, float restitution) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }

    auto shadow = LockShadowNotStepping(world);
    WaitUntilNotQuerying(world, shadow);

    for (size_t i = 0; i < world->material_count; i++) {
        if (world->materials[i].id == id) {
            world->materials[i].friction    = friction;
            world->materials[i].restitution = restitution;
            return PHYSICS_WORLD_RUNTIME_STATUS_OK;
        }
    }

    if (world->material_count >= world->materials.size()) {
        const size_t new_capacity = world->materials.size() == 0 ? kInitialMaterialCapacity
                                                                  : world->materials.size() * 2;
        if (CPH_UNLIKELY(!world->materials.reallocate(new_capacity))) {
            return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
        }
    }

    world->materials[world->material_count++] = MaterialData{id, friction, restitution};
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

PhysicsWorldRuntimeStatus
PhysicsWorldBuildDebugData(PhysicsWorld *world, bool draw_shapes, bool draw_constraints,
                                bool draw_bounding_box, bool draw_centers, bool wireframe,
                                Buffer<DebugVertex> &out_lines,
                                Buffer<DebugVertex> &out_triangles) {
    if (!world) {
        return PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    out_lines.reset();
    out_triangles.reset();

    const auto shadow = LockShadowNotStepping(world);
    world->debug_lines.clear();
    world->debug_triangles.clear();

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

    const auto copy_out = [](const DebugBuffer &source, Buffer<DebugVertex> &out) {
        if (source.count == 0) {
            return true;
        }
        if (!out.allocate(source.count)) {
            return false;
        }
        std::memcpy(out.data(), source.data.data(), source.count * sizeof(DebugVertex));
        return true;
    };
    if (!copy_out(world->debug_lines, out_lines) ||
        !copy_out(world->debug_triangles, out_triangles)) {
        out_lines.reset();
        out_triangles.reset();
        return PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY;
    }
    return PHYSICS_WORLD_RUNTIME_STATUS_OK;
}

} // namespace CPH
