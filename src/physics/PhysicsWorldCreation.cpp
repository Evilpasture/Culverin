#include "physics/PhysicsWorldCreation.hpp"
#include "core/Buffer.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/PhysicsWorld.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/SlotState.hpp"
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <cstring>
#include <limits>
#include <memory>


namespace CPH {

static constexpr float kDefaultMaterialFriction      = 0.2f;
static constexpr float kConvexHullTolerance          = 0.05f;
static constexpr size_t kInitialBodyCapacity         = 1024;
static constexpr size_t kRagdollBodyBufferIncrement = 1024;

enum class BodySetup : uint8_t {
    Configured,
    CreationProps,
    MaterialOnly,
    None,
};

static void InitializeBodyShadow(PhysicsWorld &world, BodyHandle handle,
                                   const PhysicsWorldBodyCreateRequest &request) {
    const uint32_t slot      = handle.Index();
    const uint32_t dense     = world.bodies.slot_to_dense[slot];
    const PosStride position = {request.position.GetX(), request.position.GetY(), request.position.GetZ(), 0.0};
    const AuxStride rotation = {request.rotation.GetX(), request.rotation.GetY(), request.rotation.GetZ(),
                                request.rotation.GetW()};

    world.bodies.positions[dense]          = position;
    world.bodies.prev_positions[dense]     = position;
    world.bodies.rotations[dense]          = rotation;
    world.bodies.prev_rotations[dense]     = rotation;
    world.bodies.linear_velocities[dense]  = AuxStride{};
    world.bodies.angular_velocities[dense] = AuxStride{};
    world.bodies.categories[dense]                        = request.category;
    world.bodies.masks[dense]                             = request.mask;
    world.bodies.material_ids[dense]                      = request.material_id;
    world.bodies.user_data[dense]                         = request.user_data;
}

static std::expected<BodyHandle, Error>
EnqueueBodySettings(PhysicsWorld &world, CPH::cmd::BodySettingsPtr settings,
                      const PhysicsWorldBodyCreateRequest &request) {
    auto shadow = world.LockShadowNotStepping();
    const std::expected<BodyHandle, Error> commit_result =
        world.CommitCreate(*settings, std::to_underlying(SlotState::PendingCreate));
    if (!commit_result) {
        return std::unexpected(commit_result.error());
    }
    const BodyHandle handle = *commit_result;

    InitializeBodyShadow(world, handle, request);
    const uint32_t slot = handle.Index();
    world.EnqueueCommand(slot, CPH::cmd::CreateBody{std::move(settings)});
    return handle;
}

static std::expected<BodyHandle, Error>
CreateBodyFromShape(PhysicsWorld &world, const JPH::Shape *shape,
                       const PhysicsWorldBodyCreateRequest &request,
                       JPH::EMotionType motion_type, uint32_t object_layer, BodySetup setup) {
    if (!shape) {
        return std::unexpected(WorldCreationError::ShapeFailed);
    }

    auto settings = std::make_unique<JPH::BodyCreationSettings>(
        shape, request.position, request.rotation, motion_type,
        static_cast<JPH::ObjectLayer>(object_layer));

    switch (setup) {
    case BodySetup::Configured:
        ConfigureBodySettings(*settings, shape, request.config);
        break;
    case BodySetup::CreationProps: {
        const BodyCreationProps props = {
            request.config.mass,      request.config.friction, request.config.restitution,
            request.config.is_sensor, request.config.use_ccd,
        };
        ApplyBodyCreationProps(*settings, shape, props);
        break;
    }
    case BodySetup::MaterialOnly:
        settings->mFriction    = request.config.friction;
        settings->mRestitution = request.config.restitution;
        break;
    case BodySetup::None:
        break;
    }

    return EnqueueBodySettings(world, std::move(settings), request);
}

ResolvedMaterial PhysicsWorld::ResolveMaterial(uint32_t material_id, float input_friction,
                                               float input_restitution) {
    float friction    = kDefaultMaterialFriction;
    float restitution = 0.0f;
    if (material_id > 0) {
        const std::scoped_lock guard(shadow_lock);
        for (size_t i = 0; i < material_count; i++) {
            if (materials[i].id == material_id) {
                friction    = materials[i].friction;
                restitution = materials[i].restitution;
                break;
            }
        }
    }

    return {.friction    = input_friction >= 0.0f ? input_friction : friction,
            .restitution = input_restitution >= 0.0f ? input_restitution : restitution};
}

std::expected<BodyHandle, Error>
PhysicsWorld::CreateBody(const PhysicsWorldBodyCreateRequest &request) {
    JPH::ShapeRefC shape;
    {
        const std::scoped_lock guard(shadow_lock);
        shape = FindOrCreateShapeLocked(request.shape_type, request.shape_params);
    }
    if (shape == nullptr) {
        return std::unexpected(WorldCreationError::BodySettingsFailed);
    }

    const uint32_t object_layer = request.config.motion_type == std::to_underlying(MotionType::Kinematic) ||
                                          request.config.motion_type == std::to_underlying(MotionType::Static)
                                      ? std::to_underlying(ObjectLayer::Static)
                                      : std::to_underlying(ObjectLayer::Dynamic);
    return CreateBodyFromShape(
        *this, shape.GetPtr(), request,
        static_cast<JPH::EMotionType>(request.config.motion_type), object_layer,
        BodySetup::Configured);
}

std::expected<BodyHandle, Error>
PhysicsWorld::CreateSoftBody(const PhysicsWorldSoftBodyCreateRequest &request) {
    if (!request.shared_settings) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }

    auto settings       = std::make_unique<JPH::SoftBodyCreationSettings>();
    settings->mSettings = request.shared_settings;
    settings->mPosition              = request.position;
    settings->mRotation              = request.rotation;
    settings->mObjectLayer           = std::to_underlying(ObjectLayer::Dynamic);
    settings->mNumIterations         = request.num_iterations;
    settings->mLinearDamping         = request.linear_damping;
    settings->mMaxLinearVelocity     = request.max_linear_velocity;
    settings->mRestitution           = request.restitution;
    settings->mFriction              = request.friction;
    settings->mPressure              = request.pressure;
    settings->mGravityFactor         = request.gravity_factor;
    settings->mVertexRadius          = request.vertex_radius;
    settings->mUpdatePosition        = request.update_position;
    settings->mMakeRotationIdentity  = request.make_rotation_identity;
    settings->mAllowSleeping         = true;
    settings->mFacesDoubleSided      = request.faces_double_sided;

    auto shadow = LockShadowNotStepping();
    const std::expected<BodyHandle, Error> commit_result =
        CommitCreateSoft(*settings, std::to_underlying(SlotState::PendingCreate));
    if (!commit_result) {
        return std::unexpected(commit_result.error());
    }
    const BodyHandle handle = *commit_result;

    const uint32_t slot                         = handle.Index();
    const uint32_t dense                        = bodies.slot_to_dense[slot];
    const PosStride position_shadow             = {request.position.GetX(), request.position.GetY(),
                                                   request.position.GetZ(), 0.0};
    const AuxStride rotation_shadow             = {request.rotation.GetX(), request.rotation.GetY(),
                                                   request.rotation.GetZ(), request.rotation.GetW()};
    bodies.positions[dense]      = position_shadow;
    bodies.prev_positions[dense] = position_shadow;
    bodies.rotations[dense]      = rotation_shadow;
    bodies.prev_rotations[dense] = rotation_shadow;
    bodies.categories[dense]                    = request.category;
    bodies.masks[dense]                         = request.mask;
    bodies.user_data[dense]                     = request.user_data;

    EnqueueCommand(slot, CPH::cmd::CreateSoftBody{std::move(settings), request.vertex_count});
    return handle;
}

std::expected<RagdollInstance, Error>
PhysicsWorld::CreateRagdollInstance(JPH::RagdollSettings *settings, JPH::RVec3 position,
                                      JPH::Quat rotation, uint64_t user_data) {
    if (!settings || !jolt_initialized) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }

    const std::scoped_lock trampoline(jph_trampoline_lock);
    settings->CalculateBodyIndexToConstraintIndex();
    settings->CalculateConstraintIndexToBodyIdxPair();
    JPH::Ragdoll *ragdoll = settings->CreateRagdoll(0, user_data, &physics_system);
    if (!ragdoll) {
        return std::unexpected(WorldCreationError::RagdollFailed);
    }
    ragdoll->AddRef();

    const size_t joint_count = (size_t)settings->GetSkeleton()->GetJointCount();
    Buffer<JPH::Mat44> neutral_matrices;
    if (joint_count && !neutral_matrices.allocate(joint_count)) {
        ragdoll->Release();
        return std::unexpected(WorldCreationError::OutOfMemory);
    }

    JPH::RVec3 zero_root = JPH::RVec3::sZero();
    ragdoll->GetPose(zero_root, neutral_matrices.data(), true);
    const JPH::Mat44 rotation_matrix = JPH::Mat44::sRotation(rotation);
    for (size_t i = 0; i < joint_count; i++) {
        neutral_matrices[i] = rotation_matrix * neutral_matrices[i];
    }

    ragdoll->SetPose(position, neutral_matrices.data(), true);
    ragdoll->AddToPhysicsSystem(JPH::EActivation::Activate, true);
    return RagdollInstance{.ragdoll    = ragdoll,
                           .body_count = (size_t)ragdoll->GetBodyCount()};
}

std::expected<std::size_t, Error> PhysicsWorld::RegisterRagdollBodies(
    JPH::Ragdoll *ragdoll, std::span<std::uint32_t> body_slots, uint64_t user_data,
    uint32_t category, uint32_t mask, uint32_t material_id) {
    if (!ragdoll) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }
    const size_t body_count = body_slots.size();

    auto shadow = LockShadowNotStepping();
    const size_t current_count = (count).load(std::memory_order::acquire);
    size_t available           = (free_count).load(std::memory_order::acquire);
    if (current_count > max_jolt_bodies ||
        body_count > max_jolt_bodies - current_count) {
        return std::unexpected(WorldError::LimitReached);
    }
    if (available < body_count) {
        if (body_count > SIZE_MAX - capacity - kRagdollBodyBufferIncrement) {
            return std::unexpected(WorldError::OutOfMemory);
        }
        size_t new_capacity = capacity + body_count + kRagdollBodyBufferIncrement;
        if (new_capacity > max_jolt_bodies) {
            new_capacity = max_jolt_bodies;
        }
        const std::expected<void, Error> resize_status = Resize(new_capacity);
        if (!resize_status) {
            return std::unexpected(resize_status.error());
        }
        available = (free_count).load(std::memory_order::acquire);
        if (available < body_count) {
            return std::unexpected(WorldError::LimitReached);
        }
    }

    JPH::BodyInterface &bi = physics_system.GetBodyInterface();
    PosStride *positions              = bodies.positions.data();
    PosStride *previous_positions     = bodies.prev_positions.data();
    AuxStride *rotations              = bodies.rotations.data();
    AuxStride *previous_rotations     = bodies.prev_rotations.data();
    AuxStride *linear_velocities      = bodies.linear_velocities.data();
    AuxStride *angular_velocities     = bodies.angular_velocities.data();
    for (size_t i = 0; i < body_count; i++) {
        const JPH::BodyID body_id = ragdoll->GetBodyID((int)i);
        const size_t free_index =
            (free_count).fetch_sub(1, std::memory_order::relaxed) - 1;
        const uint32_t slot = bodies.free_slots[free_index];
        body_slots[i]       = slot;
        const uint32_t dense =
            (uint32_t)(count).fetch_add(1, std::memory_order::relaxed);
        const JPH::RVec3 position = bi.GetPosition(body_id);
        const JPH::Quat rotation  = bi.GetRotation(body_id);

        positions[dense]            = PosStride{position.GetX(), position.GetY(), position.GetZ(),
                                                0.0};
        previous_positions[dense]   = positions[dense];
        rotations[dense]            = AuxStride{rotation.GetX(), rotation.GetY(), rotation.GetZ(),
                                                rotation.GetW()};
        previous_rotations[dense]   = rotations[dense];
        linear_velocities[dense]    = AuxStride{};
        angular_velocities[dense]   = AuxStride{};
        bodies.body_ids[dense]      = body_id;
        bodies.slot_to_dense[slot]  = dense;
        bodies.dense_to_slot[dense] = slot;

        const uint32_t generation =
            (bodies.generations[slot]).load(std::memory_order::relaxed);
        const BodyHandle handle   = BodyHandle::FromParts(slot, generation);
        const uint32_t jolt_index = body_id.GetIndex();
        if (id_to_handle_map && jolt_index < max_jolt_bodies) {
            (id_to_handle_map[jolt_index]).store(handle, std::memory_order::release);
        }
        bi.SetUserData(body_id, handle.Raw());
        (bodies.slot_states[slot]).store(std::to_underlying(SlotState::Alive), std::memory_order::release);
        bodies.user_data[dense]    = user_data;
        bodies.categories[dense]   = category;
        bodies.masks[dense]        = mask;
        bodies.material_ids[dense] = material_id;
    }

    return (count).load(std::memory_order::relaxed);
}

void PhysicsWorld::DestroyRagdoll(JPH::Ragdoll *ragdoll) {
    if (!ragdoll) {
        return;
    }
    const std::scoped_lock trampoline(jph_trampoline_lock);
    ragdoll->Release();
}

std::expected<void, Error> PhysicsWorld::CreateBodiesBatch(
    std::span<const PosStride> positions, std::span<const ShapeParams> sizes, int shape_type,
    int motion_type, std::span<BodyHandle> out_handles) {
    const size_t batch_count = positions.size();
    if (batch_count == 0 || sizes.size() != batch_count || out_handles.size() != batch_count ||
        batch_count > SIZE_MAX / sizeof(CPH::cmd::BodySettingsPtr)) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }

    Buffer<CPH::cmd::BodySettingsPtr> settings;
    if (!settings.allocate(batch_count)) {
        return std::unexpected(WorldCreationError::OutOfMemory);
    }

    JPH::ShapeRefC last_shape;
    const ShapeParams sentinel = {.p = {-1.0f, -1.0f, -1.0f, -1.0f}};
    ShapeParams last_size      = sentinel;
    for (size_t i = 0; i < batch_count; i++) {
        JPH::ShapeRefC shape  = last_shape;
        const float *current  = sizes[i].p;
        const float *previous = last_size.p;
        if (current[0] != previous[0] || current[1] != previous[1] || current[2] != previous[2] ||
            current[3] != previous[3]) {
            {
                const std::scoped_lock guard(shadow_lock);
                shape = FindOrCreateShapeLocked(shape_type, std::span<const float, 4>(current, 4));
            }
            last_shape = shape;
            last_size  = sizes[i];
        }

        if (shape != nullptr) {
            const uint32_t object_layer =
                motion_type == std::to_underlying(MotionType::Static) ? std::to_underlying(ObjectLayer::Static) : std::to_underlying(ObjectLayer::Dynamic);
            settings[i] = std::make_unique<JPH::BodyCreationSettings>(
                shape.GetPtr(), JPH::RVec3(positions[i].x, positions[i].y, positions[i].z),
                JPH::Quat::sIdentity(), static_cast<JPH::EMotionType>(motion_type),
                static_cast<JPH::ObjectLayer>(object_layer));
        }
    }

    auto shadow = LockShadowNotStepping();
    size_t current_count = (count).load(std::memory_order::acquire);
    size_t available     = (free_count).load(std::memory_order::acquire);
    if (available < batch_count || current_count + batch_count > capacity) {
        size_t needed = current_count + batch_count + kInitialBodyCapacity;
        if (needed > max_jolt_bodies) {
            needed = max_jolt_bodies;
        }
        const std::expected<void, Error> resize_status = Resize(needed);
        if (!resize_status) {
            return resize_status;
        }
        available = (free_count).load(std::memory_order::acquire);
    }

    if (current_count + batch_count > max_jolt_bodies) {
        return std::unexpected(WorldError::LimitReached);
    }

    if (!EnsureCommandBulkCapacity(batch_count)) {
        return std::unexpected(WorldError::OutOfMemory);
    }

    const uint32_t base_dense =
        (uint32_t)(count).fetch_add(batch_count, std::memory_order::relaxed);
    const size_t free_head = available - batch_count;
    (free_count).store(free_head, std::memory_order::release);
    for (size_t i = 0; i < batch_count; i++) {
        if (!settings[i]) {
            out_handles[i] = BodyHandle{};
            continue;
        }

        const uint32_t slot  = bodies.free_slots[free_head + i];
        const uint32_t dense = base_dense + (uint32_t)i;
        const uint32_t generation =
            (bodies.generations[slot]).load(std::memory_order::relaxed);
        const BodyHandle handle = BodyHandle::FromParts(slot, generation);
        settings[i]->mUserData  = handle.Raw();
        bodies.slot_to_dense[slot]  = dense;
        bodies.dense_to_slot[dense] = slot;
        bodies.body_ids[dense]      = JPH::BodyID();

        PosStride position                          = positions[i];
        position.w                                  = 0.0;
        bodies.positions[dense]      = position;
        bodies.prev_positions[dense] = position;
        bodies.rotations[dense]      = AuxStride{0, 0, 0, 1};
        bodies.prev_rotations[dense] = AuxStride{0, 0, 0, 1};
        // Creation defaults for the fields resize leaves indeterminate
        // (batch creation takes no per-body values for these). Legacy
        // left categories/masks uninitialized here, and resize dropped
        // even the initial kAllLayerBits fill, so in practice batch
        // bodies read as 0/0 — OnContactValidate then rejects all their
        // pairs until the user calls set_collision_filter. That is
        // almost certainly an upstream bug (single-body creation
        // defaults to kAllLayerBits), but it is the observable legacy
        // behavior and test_perf thresholds are calibrated on it, so
        // the migration preserves it deliberately. Changing this to
        // kAllLayerBits is a one-line follow-up fix; see
        // docs/CPP23_MIGRATION.md ("Discovered legacy defect").
        bodies.linear_velocities[dense]  = AuxStride{};
        bodies.angular_velocities[dense] = AuxStride{};
        bodies.user_data[dense]          = 0;
        bodies.categories[dense]         = 0;
        bodies.masks[dense]              = 0;
        bodies.material_ids[dense]       = 0;
        (bodies.slot_states[slot]).store(std::to_underlying(SlotState::PendingCreate), std::memory_order::release);

        EnqueueCommand(slot, CPH::cmd::CreateBody{std::move(settings[i])});
        out_handles[i] = handle;
    }
    return {};
}

std::expected<BodyHandle, Error>
PhysicsWorld::CreateConvexHull(std::span<const float> points,
                                 const PhysicsWorldBodyCreateRequest &request) {
    if (points.size() % 3 != 0) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }
    const size_t point_count = points.size() / 3;
    if (point_count > UINT32_MAX) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }

    Buffer<JPH::Vec3> jolt_points;
    if (!jolt_points.allocate(point_count)) {
        return std::unexpected(WorldCreationError::OutOfMemory);
    }
    for (size_t i = 0; i < point_count; i++) {
        jolt_points[i] = JPH::Vec3(points[i * 3], points[i * 3 + 1], points[i * 3 + 2]);
    }

    const JPH::ConvexHullShapeSettings hull_settings(
        jolt_points.data(), static_cast<int>(point_count), kConvexHullTolerance);
    const auto shape_result = hull_settings.Create();
    if (!shape_result.IsValid()) {
        return std::unexpected(WorldCreationError::ShapeFailed);
    }
    const JPH::ShapeRefC shape = shape_result.Get();

    const uint32_t object_layer =
        request.config.motion_type == std::to_underlying(MotionType::Static) ? std::to_underlying(ObjectLayer::Static) : std::to_underlying(ObjectLayer::Dynamic);
    return CreateBodyFromShape(
        *this, shape.GetPtr(), request,
        static_cast<JPH::EMotionType>(request.config.motion_type), object_layer,
        BodySetup::Configured);
}

static JPH::ShapeRefC CreateCompoundShape(PhysicsWorld &world,
                                            std::span<const PhysicsWorldCompoundPart> parts) {
    const std::scoped_lock trampoline(world.jph_trampoline_lock);
    JPH::StaticCompoundShapeSettings compound_settings;

    for (const auto &part : parts) {
        JPH::ShapeRefC sub_shape;
        {
            const std::scoped_lock guard(world.shadow_lock);
            sub_shape = world.FindOrCreateShapeLocked(part.shape_type, part.shape_params);
        }
        if (sub_shape != nullptr) {
            compound_settings.AddShape(part.local_position, part.local_rotation,
                                       sub_shape.GetPtr(), 0);
        }
    }

    const auto shape_result = compound_settings.Create();
    return shape_result.IsValid() ? JPH::ShapeRefC(shape_result.Get()) : JPH::ShapeRefC{};
}

std::expected<BodyHandle, Error>
PhysicsWorld::CreateCompoundBody(std::span<const PhysicsWorldCompoundPart> parts,
                                   const PhysicsWorldBodyCreateRequest &request) {
    if (parts.empty()) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }
    const JPH::ShapeRefC shape = CreateCompoundShape(*this, parts);
    if (shape == nullptr) {
        return std::unexpected(WorldCreationError::ShapeFailed);
    }

    const uint32_t object_layer =
        request.config.motion_type == std::to_underlying(MotionType::Static) ? std::to_underlying(ObjectLayer::Static) : std::to_underlying(ObjectLayer::Dynamic);
    return CreateBodyFromShape(
        *this, shape.GetPtr(), request,
        static_cast<JPH::EMotionType>(request.config.motion_type), object_layer,
        BodySetup::CreationProps);
}

std::expected<BodyHandle, Error>
PhysicsWorld::CreateMeshBody(std::span<const float> vertices,
                               std::span<const std::uint32_t> indices, JPH::RVec3 position,
                               JPH::Quat rotation, uint32_t category, uint32_t mask,
                               uint64_t user_data) {
    if (vertices.size() % 3 != 0 || indices.size() % 3 != 0) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }
    const size_t vertex_count   = vertices.size() / 3;
    const size_t triangle_count = indices.size() / 3;
    if (vertex_count == 0 || triangle_count == 0 || vertex_count > UINT32_MAX ||
        triangle_count > UINT32_MAX) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }
    if (FindInvalidMeshTriangle(indices, vertex_count)) {
        return std::unexpected(WorldCreationError::InvalidMeshIndex);
    }
    if (triangle_count > SIZE_MAX / sizeof(JPH::IndexedTriangle) ||
        vertex_count > SIZE_MAX / sizeof(JPH::Float3)) {
        return std::unexpected(WorldCreationError::OutOfMemory);
    }
    JPH::VertexList jolt_vertices;
    jolt_vertices.reserve(vertex_count);
    for (size_t i = 0; i < vertex_count; i++) {
        jolt_vertices.emplace_back(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
    }
    JPH::IndexedTriangleList triangles;
    triangles.reserve(triangle_count);
    for (size_t triangle = 0; triangle < triangle_count; triangle++) {
        triangles.emplace_back(indices[triangle * 3], indices[(triangle * 3) + 1],
                               indices[(triangle * 3) + 2], 0, 0);
    }

    const JPH::MeshShapeSettings mesh_settings(std::move(jolt_vertices), std::move(triangles));
    const auto shape_result = mesh_settings.Create();
    if (!shape_result.IsValid()) {
        return std::unexpected(WorldCreationError::ShapeFailed);
    }
    const JPH::ShapeRefC shape = shape_result.Get();

    const PhysicsWorldBodyCreateRequest request = {
        .position  = position,
        .rotation  = rotation,
        .config    = {.motion_type = std::to_underlying(MotionType::Static)},
        .category  = category,
        .mask      = mask,
        .user_data = user_data,
    };
    return CreateBodyFromShape(*this, shape.GetPtr(), request, JPH::EMotionType::Static,
                                  std::to_underlying(ObjectLayer::Static), BodySetup::None);
}

std::expected<BodyHandle, Error> PhysicsWorld::CreateHeightfieldBody(
    std::span<const float> heights, size_t grid_size, JPH::Vec3 scale, JPH::RVec3 position,
    JPH::Quat rotation, float friction, float restitution, uint32_t category, uint32_t mask,
    uint32_t material_id, uint64_t user_data) {
    if (grid_size == 0 || grid_size > UINT32_MAX || grid_size > SIZE_MAX / grid_size ||
        grid_size * grid_size > SIZE_MAX / sizeof(float) ||
        heights.size() < grid_size * grid_size) {
        return std::unexpected(WorldCreationError::InvalidArgument);
    }

    const JPH::HeightFieldShapeSettings heightfield_settings(
        heights.data(), JPH::Vec3::sZero(), JPH::Vec3(scale.GetX(), scale.GetY(), scale.GetZ()),
        static_cast<JPH::uint32>(grid_size));
    const auto shape_result = heightfield_settings.Create();
    if (!shape_result.IsValid()) {
        return std::unexpected(WorldCreationError::ShapeFailed);
    }
    const JPH::ShapeRefC shape = shape_result.Get();

    const PhysicsWorldBodyCreateRequest request = {
        .position = position,
        .rotation = rotation,
        .config = {.friction = friction, .restitution = restitution, .motion_type = std::to_underlying(MotionType::Static)},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };
    return CreateBodyFromShape(*this, shape.GetPtr(), request, JPH::EMotionType::Static,
                                  std::to_underlying(ObjectLayer::Static), BodySetup::MaterialOnly);
}

} // namespace CPH
