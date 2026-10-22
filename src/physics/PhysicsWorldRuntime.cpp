#include "physics/PhysicsWorldRuntime.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "core/Buffer.hpp"
#include "math/MathCore.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/Query.hpp"
#include "physics/SlotState.hpp"
#include <Jolt/Physics/Body/BodyLock.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodyVertex.h>
#include <cstring>
#include <limits>
#include <type_traits>


namespace CPH {

static constexpr size_t kInitialMaterialCapacity = 16;

bool PhysicsWorld::AttachContactListener() {
    if (!joltInitialized) {
        return false;
    }
    physicsSystem.SetContactListener(&contactListener);
    return true;
}

void PhysicsWorld::SyncLoadedState(size_t saved_count) {
    JPH::BodyID *CPH_RESTRICT body_ids             = bodies.bodyIds.data();
    JPH::BodyInterface &bi                         = physicsSystem.GetBodyInterface();
    PosStride *CPH_RESTRICT positions              = bodies.positions.data();
    AuxStride *CPH_RESTRICT rotations              = bodies.rotations.data();
    AuxStride *CPH_RESTRICT linear_velocities      = bodies.linearVelocities.data();
    AuxStride *CPH_RESTRICT angular_velocities     = bodies.angularVelocities.data();

    for (size_t i = 0; i < saved_count; i++) {
        if (body_ids[i].IsInvalid()) {
            continue;
        }
        const JPH::BodyID body_id = body_ids[i];

        const JPH::RVec3 position(positions[i].x, positions[i].y, positions[i].z);
        const JPH::Quat rotation(rotations[i].x, rotations[i].y, rotations[i].z, rotations[i].w);
        const JPH::Vec3 linear_velocity(linear_velocities[i].x, linear_velocities[i].y,
                                        linear_velocities[i].z);
        const JPH::Vec3 angular_velocity(angular_velocities[i].x, angular_velocities[i].y,
                                         angular_velocities[i].z);
        bi.SetPositionAndRotation(body_id, position, rotation, JPH::EActivation::Activate);
        bi.SetLinearVelocity(body_id, linear_velocity);
        bi.SetAngularVelocity(body_id, angular_velocity);

        const uint32_t slot = bodies.denseToSlot[i];
        const uint32_t generation =
            (bodies.generations[slot]).load(std::memory_order::relaxed);
        const BodyHandle handle = BodyHandle::FromParts(slot, generation);
        bi.SetUserData(body_id, handle.Raw());

        const uint32_t jolt_index = body_id.GetIndex();
        if (idToHandleMap && jolt_index <= maxJoltBodies) {
            (idToHandleMap[jolt_index]).store(handle, std::memory_order::relaxed);
        }
    }
}

struct SnapshotSizes {
    size_t total;
    size_t positions;
    size_t auxiliary;
    size_t mapping;
};

static std::optional<SnapshotSizes> CalculateSnapshotSize(size_t body_count,
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
    std::memcpy(&header.bodyCount, cursor, sizeof(header.bodyCount));
    cursor += sizeof(header.bodyCount);
    std::memcpy(&header.time, cursor, sizeof(header.time));
    cursor += sizeof(header.time);
    std::memcpy(&header.slotCapacity, cursor, sizeof(header.slotCapacity));
    return header;
}

std::expected<Buffer<std::uint8_t>, Error> PhysicsWorld::SaveSnapshot() {
    Buffer<std::uint8_t> out_data;

    auto shadow = LockShadowNotStepping();
    WaitUntilNotQuerying(shadow);
    const std::size_t body_count          = count.load(std::memory_order::acquire);
    const std::size_t saved_slot_capacity = slotCapacity;
    const double saved_time               = time;
    const std::optional<SnapshotSizes> sizes =
        CalculateSnapshotSize(body_count, saved_slot_capacity);
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
    std::memcpy(cursor, bodies.linearVelocities.data(), auxiliary_size);
    cursor += auxiliary_size;
    std::memcpy(cursor, bodies.angularVelocities.data(), auxiliary_size);
    cursor += auxiliary_size;

    auto *out_generations = reinterpret_cast<std::uint32_t *>(cursor);
    for (std::size_t i = 0; i < saved_slot_capacity; i++) {
        out_generations[i] = bodies.generations[i].load(std::memory_order::relaxed);
    }
    cursor += saved_slot_capacity * sizeof(std::uint32_t);
    std::memcpy(cursor, bodies.slotToDense.data(), saved_slot_capacity * sizeof(std::uint32_t));
    cursor += saved_slot_capacity * sizeof(std::uint32_t);
    std::memcpy(cursor, bodies.denseToSlot.data(), saved_slot_capacity * sizeof(std::uint32_t));
    cursor += saved_slot_capacity * sizeof(std::uint32_t);
    for (std::size_t i = 0; i < saved_slot_capacity; i++) {
        cursor[i] = bodies.slotStates[i].load(std::memory_order::relaxed);
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
    const size_t saved_count    = header->bodyCount;
    const double saved_time     = header->time;
    const size_t saved_capacity = header->slotCapacity;
    const uint8_t *cursor =
        data.data() + sizeof(size_t) + sizeof(double) + sizeof(size_t);

    if (saved_capacity != slotCapacity) {
        return std::unexpected(WorldSnapshotError::CapacityMismatch);
    }
    if (saved_count > saved_capacity || saved_count > maxJoltBodies) {
        return std::unexpected(WorldSnapshotError::Truncated);
    }

    const std::optional<SnapshotSizes> sizes =
        CalculateSnapshotSize(saved_count, saved_capacity);
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
    memcpy(bodies.linearVelocities.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;
    memcpy(bodies.angularVelocities.data(), cursor, auxiliary_size);
    cursor += auxiliary_size;

    const uint32_t *in_generations = (const uint32_t *)cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        (bodies.generations[i]).store(in_generations[i], std::memory_order::relaxed);
    }
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(bodies.slotToDense.data(), cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    memcpy(bodies.denseToSlot.data(), cursor, saved_capacity * sizeof(uint32_t));
    cursor += saved_capacity * sizeof(uint32_t);
    const uint8_t *in_states = cursor;
    for (size_t i = 0; i < saved_capacity; i++) {
        (bodies.slotStates[i]).store(in_states[i], std::memory_order::relaxed);
    }

    size_t rebuilt_free_count = 0;
    for (uint32_t i = 0; i < saved_capacity; i++) {
        if ((bodies.slotStates[i]).load(std::memory_order::relaxed) == std::to_underlying(SlotState::Empty)) {
            bodies.freeSlots[rebuilt_free_count++] = i;
        }
    }
    freeCount.store(rebuilt_free_count, std::memory_order::release);

    SyncLoadedState(saved_count);
    return saved_count;
}

std::expected<Buffer<ContactEvent>, Error> PhysicsWorld::TakeContactEvents() {
    Buffer<ContactEvent> out_events;

    const auto shadow = LockShadowNotStepping();
    std::size_t count = contactAtomicIdx.load(std::memory_order::acquire);
    if (count > contactMaxCapacity) {
        count = contactMaxCapacity;
    }
    if (count == 0) {
        return out_events;
    }
    if (!out_events.allocate(count)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    // Contact events contain atomics; the wire copy is a byte transfer,
    // same as the legacy implementation.
    std::memcpy(out_events.data(), contactBuffer.data(),
                count * sizeof(ContactEvent));
    contactAtomicIdx.store(0, std::memory_order::relaxed);
    return out_events;
}


std::expected<void, Error> PhysicsWorld::SetGravity(JPH::Vec3 gravity) {
    auto shadow = LockShadowNotStepping();
    const size_t current_count = (count).load(std::memory_order::acquire);
    physicsSystem.SetGravity(gravity);

    if (current_count > UINT32_MAX) [[unlikely]] {
        return std::unexpected(WorldRuntimeError::Overflow);
    }

    if (current_count > 0) {
        // body_ids is JPH::BodyID now, so this is a plain typed pointer --
        // JoltC used to copy the whole array into a fresh Array<BodyID> here.
        physicsSystem.GetBodyInterface().ActivateBodies(bodies.bodyIds.data(),
                                                        (uint32_t)current_count);
    }
    return {};
}

JPH::Vec3 PhysicsWorld::GetGravity() {
    const std::scoped_lock shadow(shadowLock);
    return physicsSystem.GetGravity();
}

std::expected<PhysicsBodyStats, Error> PhysicsWorld::GetBodyStats(BodyHandle handle) {
    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (!GetSlotPredicate(state, kMaskImmStandard).isImmediate) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }

    const uint32_t dense = bodies.slotToDense[slot];
    return PhysicsBodyStats{.position        = (bodies.positions.data())[dense],
                            .rotation        = (bodies.rotations.data())[dense],
                            .linearVelocity = (bodies.linearVelocities.data())[dense]};
}

// Runs `fn(body, vertex_count)` while holding a Jolt read lock on a soft
// body, then releases it and returns fn's result.
//
// JPH::BodyLockRead is NonCopyable with a user-declared destructor, so it is
// neither movable nor default-constructible (verified with
// is_move_constructible / is_default_constructible). It therefore cannot be
// returned out of a helper the way JoltC's JPH_BodyLockRead byte-blob could
// be -- JoltC placement-news the native object into caller-owned storage
// precisely to get around this. The lock has to be scoped, so the work that
// needs it is passed in as a callback. A side benefit: the six hand-written
// JPH_BodyLockInterface_UnlockRead calls, each an opportunity to leak the
// lock from a newly added early return, are gone.
//
// The shadow lock is deliberately released before the body lock is taken,
// matching the previous two-step behaviour where lock_soft_body_for_read
// returned (dropping the shadow lock) and only the body lock stayed held.
template <class Fn>
static auto WithSoftBodyRead(PhysicsWorld &world, BodyHandle handle, Fn &&fn)
    -> std::invoke_result_t<Fn, const JPH::Body &, std::uint32_t> {
    if (!world.joltInitialized) {
        return std::unexpected(WorldRuntimeError::InvalidArgument);
    }

    JPH::BodyID body_id;
    {
        auto shadow = world.LockShadowNotStepping();
        const std::optional<std::uint32_t> resolved = world.Resolve(handle);
        if (!resolved) {
            return std::unexpected(WorldRuntimeError::StaleHandle);
        }
        const std::uint32_t slot = *resolved;
        const uint8_t state =
            (world.bodies.slotStates[slot]).load(std::memory_order::acquire);
        if (!GetSlotPredicate(state, kMaskImmStrict).isImmediate) {
            return std::unexpected(WorldRuntimeError::StaleHandle);
        }
        const uint32_t dense = world.bodies.slotToDense[slot];
        body_id              = world.bodies.bodyIds[dense];
    }

    const JPH::BodyLockRead lock(world.physicsSystem.GetBodyLockInterface(), body_id);
    if (!lock.Succeeded()) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const JPH::Body &body = lock.GetBody();
    if (!body.IsSoftBody()) {
        return std::unexpected(WorldRuntimeError::WrongBodyType);
    }
    const auto *mp =
        static_cast<const JPH::SoftBodyMotionProperties *>(body.GetMotionProperties());
    const std::uint32_t vertex_count = static_cast<std::uint32_t>(mp->GetVertices().size());
    return std::forward<Fn>(fn)(body, vertex_count);
}

std::expected<std::uint32_t, Error>
PhysicsWorld::GetSoftBodyVertexCount(BodyHandle handle) {
    return WithSoftBodyRead(*this, handle,
                            [](const JPH::Body &, std::uint32_t vertex_count)
                                -> std::expected<std::uint32_t, Error> { return vertex_count; });
}

std::expected<SoftBodyVertexPosition, Error>
PhysicsWorld::GetSoftBodyVertexPosition(BodyHandle handle, uint32_t vertex_index) {
    return WithSoftBodyRead(
        *this, handle,
        [vertex_index](const JPH::Body &body,
                       std::uint32_t vertex_count) -> std::expected<SoftBodyVertexPosition, Error> {
            if (vertex_index >= vertex_count) {
                return std::unexpected(WorldRuntimeError::OutOfRange);
            }
            SoftBodyVertexPosition result{.position = {}, .vertexCount = vertex_count};
            const auto *mp =
                static_cast<const JPH::SoftBodyMotionProperties *>(body.GetMotionProperties());
            // Matches JoltC's JPH_Body_GetSoftBodyVertexPosition: the local
            // vertex position transformed by the centre-of-mass transform,
            // then narrowed to float for the caller.
            const JPH::RVec3 world_pos =
                body.GetCenterOfMassTransform() * mp->GetVertices()[vertex_index].mPosition;
            result.position.SetX(static_cast<float>(world_pos.GetX()));
            result.position.SetY(static_cast<float>(world_pos.GetY()));
            result.position.SetZ(static_cast<float>(world_pos.GetZ()));
            return result;
        });
}

// This buffer IS a wire format: the binding hands its bytes straight to Python
// via PyBytes_FromStringAndSize over the whole allocation, and the caller reads
// them back with np.frombuffer(dtype=float32).reshape(-1, 3). JPH::Float3 is
// the right element type precisely because it stays a tight, unpadded triple --
// JPH::Vec3 would be 16 bytes with padding and would silently corrupt every
// vertex. Pin it so a future Jolt change cannot shift the layout unnoticed.
static_assert(sizeof(JPH::Float3) == 3 * sizeof(float),
              "soft-body local vertices are exposed to Python as packed float32 triples");
static_assert(alignof(JPH::Float3) == alignof(float),
              "soft-body local vertices must not gain padding between elements");

std::expected<Buffer<JPH::Float3>, Error>
PhysicsWorld::GetSoftBodyLocalVertices(BodyHandle handle) {
    return WithSoftBodyRead(
        *this, handle,
        [](const JPH::Body &body,
           std::uint32_t vertex_count) -> std::expected<Buffer<JPH::Float3>, Error> {
            Buffer<JPH::Float3> out_positions;
            if (vertex_count != 0 && !out_positions.allocate(vertex_count)) {
                return std::unexpected(WorldRuntimeError::OutOfMemory);
            }
            const auto *mp =
                static_cast<const JPH::SoftBodyMotionProperties *>(body.GetMotionProperties());
            const JPH::Array<JPH::SoftBodyVertex> &vertices = mp->GetVertices();
            // The body is read-locked, so this is the count the lock helper
            // just validated and the buffer is exact-size: JoltC's capacity
            // clamp can never bite here. Its outCount reported the true count
            // but the caller never read it, so there is nothing to reproduce.
            const JPH::RMat44 com_transform = body.GetCenterOfMassTransform();
            for (std::uint32_t i = 0; i < vertex_count; ++i) {
                const JPH::RVec3 world_pos = com_transform * vertices[i].mPosition;
                out_positions[i].x         = static_cast<float>(world_pos.GetX());
                out_positions[i].y         = static_cast<float>(world_pos.GetY());
                out_positions[i].z         = static_cast<float>(world_pos.GetZ());
            }
            return out_positions;
        });
}

std::expected<bool, Error>
PhysicsWorld::ApplyBuoyancy(BodyHandle handle, JPH::Real surface_y, float buoyancy,
                              float linear_drag, float angular_drag, JPH::Vec3 fluid_velocity,
                              float dt) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return false;
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (!GetSlotPredicate(state, kMaskImmStrict).isImmediate) {
        return false;
    }

    const uint32_t dense     = bodies.slotToDense[slot];
    const JPH::BodyID body_id(bodies.bodyIds[dense]);
    (activeQueries).fetch_add(1, std::memory_order::acquire);

    JPH::BodyInterface &bi        = physicsSystem.GetBodyInterface();
    const JPH::Vec3 gravity       = physicsSystem.GetGravity();
    const JPH::RVec3 surface_position(0, surface_y, 0);
    const JPH::Vec3 surface_normal(0, 1.0f, 0);
    bi.ActivateBody(body_id);
    const bool submerged =
        bi.ApplyBuoyancyImpulse(body_id, surface_position, surface_normal, buoyancy, linear_drag,
                                angular_drag, fluid_velocity, gravity, dt);
    QueryEnd();
    return submerged;
}

std::expected<void, Error> PhysicsWorld::ApplyBuoyancyBatch(
    std::span<const BodyHandle> handles, JPH::Real surface_y, float buoyancy, float linear_drag,
    float angular_drag, JPH::Vec3 fluid_velocity, float dt) {
    const size_t handle_count = handles.size();
    if (handle_count == 0) {
        return {};
    }
    if (handle_count > SIZE_MAX / sizeof(JPH::BodyID)) [[unlikely]] {
        return std::unexpected(WorldRuntimeError::Overflow);
    }

    Buffer<JPH::BodyID> body_ids;
    if (!body_ids.allocate(handle_count)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }

    size_t valid_count = 0;
    auto shadow = LockShadowNotStepping();
    for (size_t i = 0; i < handle_count; i++) {
        const std::optional<std::uint32_t> resolved = Resolve(handles[i]);
        if (!resolved) {
            continue;
        }
        const std::uint32_t slot = *resolved;
        const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
        if (GetSlotPredicate(state, kMaskImmStrict).isImmediate) {
            body_ids[valid_count++] = bodies.bodyIds[bodies.slotToDense[slot]];
        }
    }

    if (valid_count > 0) {
        (activeQueries).fetch_add(1, std::memory_order::acquire);
    }

    if (valid_count > 0) {
        JPH::BodyInterface &bi        = physicsSystem.GetBodyInterface();
        const JPH::Vec3 gravity       = physicsSystem.GetGravity();
        const JPH::RVec3 surface_position(0, surface_y, 0);
        const JPH::Vec3 surface_normal(0, 1.0f, 0);
        for (size_t i = 0; i < valid_count; i++) {
            const JPH::BodyID body_id = body_ids[i];
            bi.ActivateBody(body_id);
            bi.ApplyBuoyancyImpulse(body_id, surface_position, surface_normal, buoyancy,
                                    linear_drag, angular_drag, fluid_velocity, gravity, dt);
        }
        QueryEnd();
    }

    return {};
}

std::expected<MotionType, Error> PhysicsWorld::GetMotionType(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (GetSlotPredicate(state, kMaskImmStandard).isImmediate) {
        const uint32_t dense     = bodies.slotToDense[slot];
        const JPH::BodyID body_id(bodies.bodyIds[dense]);
        return static_cast<MotionType>(
            physicsSystem.GetBodyInterface().GetMotionType(body_id));
    }
    if (state == std::to_underlying(SlotState::PendingCreate)) {
        return std::unexpected(WorldRuntimeError::PendingCreate);
    }
    return std::unexpected(WorldRuntimeError::StaleHandle);
}

std::expected<std::uint64_t, Error> PhysicsWorld::GetUserData(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (!GetSlotPredicate(state, kMaskImmStandard).is_executable()) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }

    const uint32_t dense = bodies.slotToDense[slot];
    return bodies.userData[dense];
}

std::expected<std::uint32_t, Error> PhysicsWorld::GetIndex(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (!GetSlotPredicate(state, kMaskImmStandard).isImmediate) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }

    return bodies.slotToDense[slot];
}

bool PhysicsWorld::IsAlive(BodyHandle handle) {
    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return false;
    }
    const uint8_t state = (bodies.slotStates[*resolved]).load(std::memory_order::acquire);
    return GetSlotPredicate(state, kMaskImmStandard).is_executable();
}

std::expected<bool, Error> PhysicsWorld::IsActive(BodyHandle handle) {

    auto shadow = LockShadowNotStepping();
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (!GetSlotPredicate(state, kMaskImmStandard).isImmediate) {
        return false;
    }

    const uint32_t dense     = bodies.slotToDense[slot];
    const JPH::BodyID body_id(bodies.bodyIds[dense]);
    return physicsSystem.GetBodyInterface().IsActive(body_id);
}

std::expected<std::size_t, Error>
PhysicsWorld::GetActiveIndices(Buffer<std::uint32_t> &out_indices) {
    out_indices.reset();

    Buffer<JPH::BodyID> body_ids;
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
        std::memcpy(body_ids.data(), bodies.bodyIds.data(), count * sizeof(JPH::BodyID));
    }

    if (!out_indices.allocate(count)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    JPH::BodyInterface &bi = physicsSystem.GetBodyInterface();
    std::size_t active_count = 0;
    for (std::size_t i = 0; i < count; i++) {
        if (!body_ids[i].IsInvalid() && bi.IsActive(body_ids[i])) {
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
    if (count > std::numeric_limits<std::size_t>::max() / 7) [[unlikely]] {
        return std::unexpected(WorldRuntimeError::Overflow);
    }
    if (!out_state.allocate(count * 7)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    MathComputeInterpolationLoop(bodies.positions.data(),
                                     bodies.prevPositions.data(),
                                     bodies.rotations.data(),
                                     bodies.prevRotations.data(), alpha,
                                     out_state.data(), count);
    return out_state;
}


std::expected<void, Error> PhysicsWorld::SetCollisionFilter(BodyHandle handle,
                                                             uint32_t category, uint32_t mask) {

    auto shadow = LockShadowNotStepping();
    WaitUntilNotQuerying(shadow);
    const std::optional<std::uint32_t> resolved = Resolve(handle);
    if (!resolved) {
        return std::unexpected(WorldRuntimeError::StaleHandle);
    }
    const std::uint32_t slot = *resolved;

    const uint8_t state = (bodies.slotStates[slot]).load(std::memory_order::acquire);
    if (GetSlotPredicate(state, kMaskImmStandard).isImmediate) {
        const uint32_t dense     = bodies.slotToDense[slot];
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

    for (size_t i = 0; i < materialCount; i++) {
        if (materials[i].id == id) {
            materials[i].friction    = friction;
            materials[i].restitution = restitution;
            return {};
        }
    }

    if (materialCount >= materials.size()) {
        const size_t new_capacity = materials.size() == 0 ? kInitialMaterialCapacity
                                                                  : materials.size() * 2;
        if (!materials.reallocate(new_capacity)) [[unlikely]] {
            return std::unexpected(WorldRuntimeError::OutOfMemory);
        }
    }

    materials[materialCount++] = MaterialData{id, friction, restitution};
    return {};
}

std::expected<DebugData, Error>
PhysicsWorld::BuildDebugData(bool draw_shapes, bool draw_constraints, bool draw_bounding_box,
                                bool draw_centers, bool wireframe) {
    DebugData out;
    Buffer<DebugVertex> &out_lines     = out.lines;
    Buffer<DebugVertex> &out_triangles = out.triangles;

    const auto shadow = LockShadowNotStepping();
    debugLines.clear();
    debugTriangles.clear();

    JPH::BodyManager::DrawSettings settings;
    settings.mDrawShape                 = draw_shapes;
    settings.mDrawShapeWireframe        = wireframe;
    settings.mDrawBoundingBox           = draw_bounding_box;
    settings.mDrawCenterOfMassTransform = draw_centers;

    JPH::DebugRenderer *const renderer = debugRenderer.get();
    if (draw_shapes || draw_bounding_box || draw_centers) {
        physicsSystem.DrawBodies(settings, renderer, nullptr);
    }
    if (draw_constraints) {
        physicsSystem.DrawConstraints(renderer);
        physicsSystem.DrawConstraintLimits(renderer);
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
    if (!copy_out(debugLines, out_lines) || !copy_out(debugTriangles, out_triangles)) {
        return std::unexpected(WorldRuntimeError::OutOfMemory);
    }
    return out;
}

} // namespace CPH
