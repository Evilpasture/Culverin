#include "physics/PhysicsWorldRuntime.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "core/Buffer.hpp"
#include "math/MathCore.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/Query.hpp"
#include "physics/SlotState.hpp"
#include <cstring>
#include <limits>


namespace CPH {

static constexpr size_t kInitialMaterialCapacity = 16;

static JPH_Vec3 get_gravity(const PhysicsWorld &world) {
    const JPH::Vec3 gravity = world.physics_system.GetGravity();
    return {gravity.GetX(), gravity.GetY(), gravity.GetZ()};
}

bool PhysicsWorld::AttachContactListener() {
    if (!jolt_initialized) {
        return false;
    }
    contact_listener = JPH_ContactListener_Create(this);
    if (!contact_listener) {
        return false;
    }
    physics_system.SetContactListener(
        reinterpret_cast<JPH::ContactListener *>(contact_listener));
    return true;
}

void PhysicsWorld::SyncLoadedState(size_t saved_count) {
    JPH_BodyID *CPH_RESTRICT body_ids              = bodies.body_ids.data();
    JPH_BodyInterface *CPH_RESTRICT body_interface = this->body_interface;
    PosStride *CPH_RESTRICT positions              = bodies.positions.data();
    AuxStride *CPH_RESTRICT rotations              = bodies.rotations.data();
    AuxStride *CPH_RESTRICT linear_velocities      = bodies.linear_velocities.data();
    AuxStride *CPH_RESTRICT angular_velocities     = bodies.angular_velocities.data();

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

        const uint32_t slot = bodies.dense_to_slot[i];
        const uint32_t generation =
            (bodies.generations[slot]).load(std::memory_order::relaxed);
        const uint64_t raw_handle = make_handle(slot, generation);
        JPH_BodyInterface_SetUserData(body_interface, body_id, raw_handle);

        const uint32_t jolt_index = JPH_ID_TO_INDEX(body_id);
        if (id_to_handle_map && jolt_index <= max_jolt_bodies) {
            (id_to_handle_map[jolt_index]).store(raw_handle, std::memory_order::relaxed);
        }
    }
}

struct SnapshotSizes {
    size_t total;
    size_t positions;
    size_t auxiliary;
    size_t mapping;
};

static std::optional<SnapshotSizes> calculate_snapshot_size(size_t body_count,
                                                            size_t slot_capacity) {
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
        return std::nullopt;
    }
    return SnapshotSizes{.total     = total_size,
                         .positions = positions_size,
                         .auxiliary = auxiliary_size,
                         .mapping   = mapping_size};
}

std::optional<SnapshotHeader> ReadSnapshotHeader(std::span<const std::uint8_t> data) {
    if (data.size() < sizeof(std::size_t) + sizeof(double) + sizeof(std::size_t)) {
        return std::nullopt;
    }
    SnapshotHeader header;
    const std::uint8_t *cursor = data.data();
    std::memcpy(&header.body_count, cursor, sizeof(header.body_count));
    cursor += sizeof(header.body_count);
    std::memcpy(&header.time, cursor, sizeof(header.time));
    cursor += sizeof(header.time);
    std::memcpy(&header.slot_capacity, cursor, sizeof(header.slot_capacity));
    return header;
}

std::expected<Buffer<std::uint8_t>, Error> PhysicsWorld::SaveSnapshot() {
    Buffer<std::uint8_t> out_data;

    auto shadow = LockShadowNotStepping();
    WaitUntilNotQuerying(shadow);
    const std::size_t body_count          = count.load(std::memory_order::acquire);
    const std::size_t saved_slot_capacity = slot_capacity;
    const double saved_time               = time;
    const std::optional<SnapshotSizes> sizes =
        calculate_snapshot_size(body_count, saved_slot_capacity);
    if (!sizes) {
        return std::unexpected(WorldSnapshotError::Overflow);
    }
    const std::size_t position_size  = sizes->positions;
    const std::size_t auxiliary_size = sizes->auxiliary;
    if (!out_data.allocate(sizes->total)) {
        return std::unexpected(WorldSnapshotError::OutOfMemory);
    }

    std::uint8_t *cursor = out_data.data();
    std::memcpy(cursor, &body_count, sizeof(body_count));
    cursor += sizeof(body_count);
    std::memcpy(cursor, &saved_time, sizeof(saved_time));
    cursor += sizeof(saved_time);
    std::memcpy(cursor, &saved_slot_capacity, sizeof(saved_slot_capacity));
    cursor += sizeof(saved_slot_capacity);
    std::memcpy(cursor, bodies.positions.data(), position_size);
    cursor += position_size;
    std::memcpy(cursor, bodies.rotations.data(), auxiliary_size);
    cursor += auxiliary_size;
    std::memcpy(cursor, bodies.linear_velocities.data(), auxiliary_size);
    cursor += auxiliary_size;
    std::memcpy(cursor, bodies.angular_velocities.data(), auxiliary_size);
    cursor += auxiliary_size;

    auto *out_generations = reinterpret_cast<std::uint32_t *>(cursor);
    for (std::size_t i = 0; i < saved_slot_capacity; i++) {
        out_generations[i] = bodies.generations[i].load(std::memory_order::relaxed);
    }
    cursor += saved_slot_capacity * sizeof(std::uint32_t);
    std::memcpy(cursor, bodies.slot_to_dense.data(), saved_slot_capacity * sizeof(std::uint32_t));
    cursor += saved_slot_capacity * sizeof(std::uint32_t);
    std::memcpy(cursor, bodies.dense_to_slot.data(), saved_slot_capacity * sizeof(std::uint32_t));
    cursor += saved_slot_capacity * sizeof(std::uint32_t);
    for (std::size_t i = 0; i < saved_slot_capacity; i++) {
        cursor[i] = bodies.slot_states[i].load(std::memory_order::relaxed);
    }
    return out_data;
}

std::expected<std::size_t, Error> PhysicsWorld::LoadSnapshot(std::span<const std::uint8_t> data) {
    auto shadow = LockShadowNotStepping();
    WaitUntilNotQuerying(shadow);
    const std::optional<SnapshotHeader> header = ReadSnapshotHeader(data);
    if (!header) {
        return std::unexpected(WorldSnapshotError::Truncated);
    }
    const size_t saved_count    = header->body_count;
    const double saved_time     = header->time;
    const size_t saved_capacity = header->slot_capacity;
    const uint8_t *cursor =
        data.data() + sizeof(size_t) + sizeof(double) + sizeof(size_t);

    if (saved_capacity != slot_capacity) {
        return std::unexpected(WorldSnapshotError::CapacityMismatch);
    }
    if (saved_count > saved_capacity || saved_count > max_jolt_bodies) {
        return std::unexpected(WorldSnapshotError::Truncated);
    }

    const std::optional<SnapshotSizes> sizes =
        calculate_snapshot_size(saved_count, saved_capacity);
    if (!sizes) {
        return std::unexpected(WorldSnapshotError::Truncated);
    }
    const size_t position_size  = sizes->positions;
    const size_t auxiliary_size = sizes->auxiliary;
    if (data.size() != sizes->total) {
        return std::unexpected(WorldSnapshotError::Truncated);
    }

    (count).store(saved_count, std::memory_order::relaxed);
    time = saved_time;
    memcpy(bodies.positions.data(), cursor, position_size);
    cursor += position_size;
    memcpy(bodies.rotations.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(bodies.linear_velocities.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(bodies.angular_velocities.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;

    const uint32_t *in_generations = (const uint32_t *)cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        (bodies.generations[i]).store(in_generations[i], std::memory_order::relaxed);
    }
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(bodies.slot_to_dense.data(), cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(bodies.dense_to_slot.data(), cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    const uint8_t *in_states = cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        (bodies.slot_states[i]).store(in_states[i], std::memory_order::relaxed);
    }

    size_t rebuilt_free_count = 0;
    for (uint32_t i = 0; i < saved_capacity; i++) {
        if ((bodies.slot_states[i]).load(std::memory_order::relaxed) == std::to_underlying(SlotState::Empty)) {
            bodies.free_slots[rebuilt_free_count++] = i;
        }
    }
    free_count.store(rebuilt_free_count, std::memory_order::release);

    SyncLoadedState(saved_count);
    return saved_count;
}

std::expected<Buffer<ContactEvent>, Error> PhysicsWorld::TakeContactEvents() {
    Buffer<ContactEvent> out_events;

    const auto shadow = LockShadowNotStepping();
    std::size_t count = contact_atomic_idx.load(std::memory_order::acquire);
    if (count > contact_max_capacity) {
        count = contact_max_capacity;
    }
    if (count == 0) {
        return out_events;
    }
    if (!out_events.allocate(count)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    // Contact events contain atomics; the wire copy is a byte transfer,
    // same as the legacy implementation.
    std::memcpy(out_events.data(), contact_buffer.data(),
                count * sizeof(ContactEvent));
    contact_atomic_idx.store(0, std::memory_order::relaxed);
    return out_events;
}


std::expected<void, Error> PhysicsWorld::SetGravity(JPH::Vec3 gravity) {
    auto shadow = LockShadowNotStepping();
    const size_t current_count = (count).load(std::memory_order::acquire);
    physics_system.SetGravity(gravity);

    if (CPH_UNLIKELY(current_count > UINT32_MAX)) {
        return std::unexpected(WorldRuntimeError::Overflow);
    }

    if (current_count > 0) {
        JPH_BodyInterface_ActivateBodies(body_interface, bodies.body_ids.data(),
                                         (uint32_t)current_count);
    }
    return {};
}

JPH::Vec3 PhysicsWorld::GetGravity() {
    const std::scoped_lock shadow(shadow_lock);
    return physics_system.GetGravity();
}

std::expected<PhysicsBodyStats, Error> PhysicsWorld::GetBodyStats(BodyHandle handle) {
    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }

    const uint32_t dense = bodies.slot_to_dense[slot];
    return PhysicsBodyStats{.position        = (bodies.positions.data())[dense],
                            .rotation        = (bodies.rotations.data())[dense],
                            .linear_velocity = (bodies.linear_velocities.data())[dense]};
}

struct SoftBodyReadLock {
    JPH_BodyLockRead lock;
    const JPH_BodyLockInterface *lock_interface;
    std::uint32_t vertex_count;
};

static std::expected<SoftBodyReadLock, Error>
lock_soft_body_for_read(PhysicsWorld &world, BodyHandle handle) {
    if (!world.jolt_initialized) {
        return std::unexpected(WorldRuntimeError::InvalidArgument);
    }

    auto shadow = world.LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = world.UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;
    const uint8_t state = (world.bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStrict).is_immediate) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const uint32_t dense     = world.bodies.slot_to_dense[slot];
    const JPH_BodyID body_id = world.bodies.body_ids[dense];

    SoftBodyReadLock result;
    result.lock_interface = reinterpret_cast<const JPH_BodyLockInterface *>(
        &world.physics_system.GetBodyLockInterface());
    JPH_BodyLockInterface_LockRead(result.lock_interface, body_id, &result.lock);
    if (!result.lock.body) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    if (!JPH_Body_IsSoftBody(result.lock.body)) {
        JPH_BodyLockInterface_UnlockRead(result.lock_interface, &result.lock);
        return std::unexpected(WorldRuntimeError::WrongBodyType);
    }
    result.vertex_count = JPH_Body_GetSoftBodyVertexCount(result.lock.body);
    return result;
}

std::expected<std::uint32_t, Error>
PhysicsWorld::GetSoftBodyVertexCount(BodyHandle handle) {
    std::expected<SoftBodyReadLock, Error> locked = lock_soft_body_for_read(*this, handle);
    if (!locked) {
        return std::unexpected(locked.error());
    }
    const std::uint32_t count = locked->vertex_count;
    JPH_BodyLockInterface_UnlockRead(locked->lock_interface, &locked->lock);
    return count;
}

std::expected<SoftBodyVertexPosition, Error>
PhysicsWorld::GetSoftBodyVertexPosition(BodyHandle handle, uint32_t vertex_index) {
    std::expected<SoftBodyReadLock, Error> locked = lock_soft_body_for_read(*this, handle);
    if (!locked) {
        return std::unexpected(locked.error());
    }
    if (vertex_index >= locked->vertex_count) {
        JPH_BodyLockInterface_UnlockRead(locked->lock_interface, &locked->lock);
        return std::unexpected(WorldRuntimeError::OutOfRange);
    }
    SoftBodyVertexPosition result{.position = {}, .vertex_count = locked->vertex_count};
    JPH_Body_GetSoftBodyVertexPosition(locked->lock.body, vertex_index, &result.position);
    JPH_BodyLockInterface_UnlockRead(locked->lock_interface, &locked->lock);
    return result;
}

std::expected<Buffer<JPH_Vec3>, Error>
PhysicsWorld::GetSoftBodyLocalVertices(BodyHandle handle) {
    Buffer<JPH_Vec3> out_positions;
    std::expected<SoftBodyReadLock, Error> locked = lock_soft_body_for_read(*this, handle);
    if (!locked) {
        return std::unexpected(locked.error());
    }
    const std::uint32_t count = locked->vertex_count;
    if (count != 0 && !out_positions.allocate(count)) {
        JPH_BodyLockInterface_UnlockRead(locked->lock_interface, &locked->lock);
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    // The body is read-locked, so the count Jolt reports here is the count
    // lock_soft_body_for_read just validated; the buffer is exact-size.
    std::uint32_t result_count = 0;
    JPH_Body_GetSoftBodyVertexPositions(locked->lock.body, out_positions.data(), count,
                                        &result_count);
    JPH_BodyLockInterface_UnlockRead(locked->lock_interface, &locked->lock);
    return out_positions;
}

std::expected<bool, Error>
PhysicsWorld::ApplyBuoyancy(BodyHandle handle, JPH_Real surface_y, float buoyancy,
                              float linear_drag, float angular_drag, JPH::Vec3 fluid_velocity,
                              float dt) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return false;
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStrict).is_immediate) {
        return false;
    }

    const uint32_t dense     = bodies.slot_to_dense[slot];
    const JPH_BodyID body_id = bodies.body_ids[dense];
    (active_queries).fetch_add(1, std::memory_order::acquire);

    JPH_BodyInterface_ActivateBody(body_interface, body_id);
    JPH_Vec3 gravity = get_gravity(*this);
    JPH_RVec3 surface_position = {0, surface_y, 0};
    JPH_Vec3 surface_normal    = {0, 1.0f, 0};
    JPH_Vec3 fluid             = {fluid_velocity.GetX(), fluid_velocity.GetY(),
                      fluid_velocity.GetZ()};
    const bool submerged       = JPH_BodyInterface_ApplyBuoyancyImpulse(
        body_interface, body_id, &surface_position, &surface_normal, buoyancy, linear_drag,
        angular_drag, &fluid, &gravity, dt);
    QueryEnd();
    return submerged;
}

std::expected<void, Error> PhysicsWorld::ApplyBuoyancyBatch(
    std::span<const BodyHandle> handles, JPH_Real surface_y, float buoyancy, float linear_drag,
    float angular_drag, JPH::Vec3 fluid_velocity, float dt) {
    const size_t handle_count = handles.size();
    if (handle_count == 0) {
        return {};
    }
    if (CPH_UNLIKELY(handle_count > SIZE_MAX / sizeof(JPH_BodyID))) {
        return std::unexpected(WorldRuntimeError::Overflow);
    }

    Buffer<JPH_BodyID> body_ids;
    if (!body_ids.allocate(handle_count)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }

    size_t valid_count = 0;
    auto shadow = LockShadowNotStepping();
    for (size_t i = 0; i < handle_count; i++) {
        const std::optional<std::uint32_t> resolved = UnpackHandle(handles[i]);
        if (!resolved) {
            continue;
        }
        const std::uint32_t slot = *resolved;
        const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
        if (get_slot_predicate(state, kMaskImmStrict).is_immediate) {
            body_ids[valid_count++] = bodies.body_ids[bodies.slot_to_dense[slot]];
        }
    }

    if (valid_count > 0) {
        (active_queries).fetch_add(1, std::memory_order::acquire);
    }

    if (valid_count > 0) {
        JPH_Vec3 gravity = get_gravity(*this);
        JPH_RVec3 surface_position = {0, surface_y, 0};
        JPH_Vec3 surface_normal    = {0, 1.0f, 0};
        JPH_Vec3 fluid             = {fluid_velocity.GetX(), fluid_velocity.GetY(),
                          fluid_velocity.GetZ()};
        for (size_t i = 0; i < valid_count; i++) {
            JPH_BodyInterface_ActivateBody(body_interface, body_ids[i]);
            JPH_BodyInterface_ApplyBuoyancyImpulse(
                body_interface, body_ids[i], &surface_position, &surface_normal, buoyancy,
                linear_drag, angular_drag, &fluid, &gravity, dt);
        }
        QueryEnd();
    }

    return {};
}

std::expected<MotionType, Error> PhysicsWorld::GetMotionType(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        const uint32_t dense     = bodies.slot_to_dense[slot];
        const JPH_BodyID body_id = bodies.body_ids[dense];
        return static_cast<MotionType>(
            JPH_BodyInterface_GetMotionType(body_interface, body_id));
    }
    if (state == std::to_underlying(SlotState::PendingCreate)) {
        return std::unexpected(WorldRuntimeError::PendingCreate);
    }
    return std::unexpected(WorldRuntimeError::StaleHandle);
}

std::expected<std::uint64_t, Error> PhysicsWorld::GetUserData(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_executable()) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }

    const uint32_t dense = bodies.slot_to_dense[slot];
    return bodies.user_data[dense];
}

std::expected<std::uint32_t, Error> PhysicsWorld::GetIndex(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }

    return bodies.slot_to_dense[slot];
}

bool PhysicsWorld::IsAlive(BodyHandle handle) {
    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return false;
    }
    const uint8_t state = (bodies.slot_states[*resolved]).load(std::memory_order::acquire);
    return get_slot_predicate(state, kMaskImmStandard).is_executable();
}

std::expected<bool, Error> PhysicsWorld::IsActive(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (!get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        return false;
    }

    const uint32_t dense     = bodies.slot_to_dense[slot];
    const JPH_BodyID body_id = bodies.body_ids[dense];
    return JPH_BodyInterface_IsActive(body_interface, body_id);
}

std::expected<std::size_t, Error>
PhysicsWorld::GetActiveIndices(Buffer<std::uint32_t> &out_indices) {
    out_indices.reset();

    Buffer<JPH_BodyID> body_ids;
    std::size_t count = 0;
    {
        const auto shadow = LockShadowNotStepping();
        count             = this->count.load(std::memory_order::acquire);
        if (count == 0) {
            return std::size_t{0};
        }
        if (!body_ids.allocate(count)) {
            return std::unexpected(WorldRuntimeError::OutOfMemory);
        }
        std::memcpy(body_ids.data(), bodies.body_ids.data(), count * sizeof(JPH_BodyID));
    }

    if (!out_indices.allocate(count)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    std::size_t active_count = 0;
    for (std::size_t i = 0; i < count; i++) {
        if (body_ids[i] != kJphInvalidBodyId &&
            JPH_BodyInterface_IsActive(body_interface, body_ids[i])) {
            out_indices[active_count++] = static_cast<std::uint32_t>(i);
        }
    }
    return active_count;
}


std::expected<Buffer<float>, Error> PhysicsWorld::GetRenderState(float alpha) {
    Buffer<float> out_state;

    const auto shadow       = LockShadowNotStepping();
    const std::size_t count = this->count.load(std::memory_order::acquire);
    if (count == 0) {
        return out_state;
    }
    if (CPH_UNLIKELY(count > std::numeric_limits<std::size_t>::max() / 7)) {
        return std::unexpected(WorldRuntimeError::Overflow);
    }
    if (!out_state.allocate(count * 7)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    MathComputeInterpolationLoop(bodies.positions.data(),
                                     bodies.prev_positions.data(),
                                     bodies.rotations.data(),
                                     bodies.prev_rotations.data(), alpha,
                                     out_state.data(), count);
    return out_state;
}


std::expected<void, Error> PhysicsWorld::SetCollisionFilter(BodyHandle handle,
                                                             uint32_t category, uint32_t mask) {

    auto shadow = LockShadowNotStepping();
    WaitUntilNotQuerying(shadow);
    const std::optional<std::uint32_t> resolved = UnpackHandle(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slot_states[slot]).load(std::memory_order::acquire);
    if (get_slot_predicate(state, kMaskImmStandard).is_immediate) {
        const uint32_t dense     = bodies.slot_to_dense[slot];
        bodies.categories[dense] = category;
        bodies.masks[dense]      = mask;
        return {};
    }
    if (state == std::to_underlying(SlotState::PendingCreate)) {
        return std::unexpected(WorldRuntimeError::PendingCreate);
    }
    return std::unexpected(WorldRuntimeError::StaleHandle);
}

std::expected<void, Error> PhysicsWorld::RegisterMaterial(uint32_t id, float friction,
                                                          float restitution) {

    auto shadow = LockShadowNotStepping();
    WaitUntilNotQuerying(shadow);

    for (size_t i = 0; i < material_count; i++) {
        if (materials[i].id == id) {
            materials[i].friction    = friction;
            materials[i].restitution = restitution;
            return {};
        }
    }

    if (material_count >= materials.size()) {
        const size_t new_capacity = materials.size() == 0 ? kInitialMaterialCapacity
                                                                  : materials.size() * 2;
        if (CPH_UNLIKELY(!materials.reallocate(new_capacity))) {
            return std::unexpected(WorldRuntimeError::OutOfMemory);
        }
    }

    materials[material_count++] = MaterialData{id, friction, restitution};
    return {};
}

std::expected<DebugData, Error>
PhysicsWorld::BuildDebugData(bool draw_shapes, bool draw_constraints, bool draw_bounding_box,
                                bool draw_centers, bool wireframe) {
    DebugData out;
    Buffer<DebugVertex> &out_lines     = out.lines;
    Buffer<DebugVertex> &out_triangles = out.triangles;

    const auto shadow = LockShadowNotStepping();
    debug_lines.clear();
    debug_triangles.clear();

    JPH::BodyManager::DrawSettings settings;
    settings.mDrawShape                 = draw_shapes;
    settings.mDrawShapeWireframe        = wireframe;
    settings.mDrawBoundingBox           = draw_bounding_box;
    settings.mDrawCenterOfMassTransform = draw_centers;

    auto *renderer = reinterpret_cast<JPH::DebugRenderer *>(debug_renderer);
    if (draw_shapes || draw_bounding_box || draw_centers) {
        physics_system.DrawBodies(settings, renderer, nullptr);
    }
    if (draw_constraints) {
        physics_system.DrawConstraints(renderer);
        physics_system.DrawConstraintLimits(renderer);
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
    if (!copy_out(debug_lines, out_lines) || !copy_out(debug_triangles, out_triangles)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    return out;
}

} // namespace CPH
