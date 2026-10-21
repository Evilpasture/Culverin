#pragma once

// Plain native data shared by the parser adapters and the physics core.

#include "core/Types.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Math/Mat44.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>

#include <cstddef>
#include <cstdint>
#include <limits>


namespace CPH {

inline constexpr std::size_t kBakedPositionStride = 4;
inline constexpr std::size_t kBakedRotationStride = 4;
inline constexpr std::size_t kBakedShapeStride    = 5;

inline constexpr std::uint32_t kAllLayerBits = 0xFFFFU;

struct BakedSceneData {
    std::size_t count;
    const JPH::Real *positions;
    const float *rotations;
    const float *shapes;
    const std::uint8_t *motion_types;
    const std::uint8_t *object_layers;
    const std::uint64_t *user_data;
};

struct ShapeParams {
    float p[4];
};

struct TrackData {
    std::uint32_t *indices;
    std::uint32_t count;
    std::uint32_t driven_idx;
};

// Constraint parameters. Default member initializers carry the sane
// defaults the old CPH_ParamsInit() applied (20 Hz / critically damped
// motors, unlimited limits), so a value-initialized struct is ready to use.
struct ConstraintParams {
    float px = 0.0F, py = 0.0F, pz = 0.0F;
    float ax = 0.0F, ay = 1.0F, az = 0.0F;
    float limit_min       = -std::numeric_limits<float>::max();
    float limit_max       = std::numeric_limits<float>::max();
    float half_cone_angle = 0.0F;
    bool has_motor        = false;
    int motor_type        = 0;
    float motor_target    = 0.0F;
    float max_torque      = 1e6F;
    float frequency       = 20.0F;
    float damping         = 1.0F;
    float max_force       = 1e6F;
};

// --- Body creation requests ---

struct PhysicsWorldBodyCreateRequest {
    // Shape-cache key; unused (0) when the caller creates the shape itself
    // (mesh/heightfield paths).
    int shape_type          = 0;
    float shape_params[4]   = {};
    JPH::RVec3 position     = JPH::RVec3::sZero();
    JPH::Quat rotation      = JPH::Quat::sIdentity();
    BodyConfig config;
    uint32_t category       = kAllLayerBits;
    uint32_t mask           = kAllLayerBits;
    uint32_t material_id    = 0;
    uint64_t user_data      = 0;
};

struct PhysicsWorldCompoundPart {
    int shape_type          = 0;
    float shape_params[4]   = {};
    JPH::Vec3 local_position = JPH::Vec3::sZero();
    JPH::Quat local_rotation = JPH::Quat::sIdentity();
};

struct PhysicsWorldSoftBodyCreateRequest {
    // Const: SoftBodyCreationSettings::mSettings holds it read-only, which is
    // why the old code needed a reinterpret_cast here.
    const JPH::SoftBodySharedSettings *shared_settings = nullptr;
    uint32_t vertex_count                       = 0;
    JPH::RVec3 position                         = JPH::RVec3::sZero();
    JPH::Quat rotation                          = JPH::Quat::sIdentity();
    uint64_t user_data                          = 0;
    uint32_t category                           = kAllLayerBits;
    uint32_t mask                               = kAllLayerBits;
    float pressure                              = 0.0f;
    float vertex_radius                         = 0.0f;
    float linear_damping                        = 0.0f;
    uint32_t num_iterations                     = 0;
    float max_linear_velocity                   = 0.0f;
    float gravity_factor                        = 0.0f;
    float friction                              = 0.0f;
    float restitution                           = 0.0f;
    bool make_rotation_identity                 = false;
    bool update_position                        = false;
    bool faces_double_sided                     = false;
};

// Result of material resolution: the caller's explicit values win over
// the registered material (negative input means "use the material").
struct ResolvedMaterial {
    float friction;
    float restitution;
};

// Result of ragdoll instantiation: the Jolt ragdoll and how many bodies
// it added.
struct RagdollInstance {
    // Non-owning: the caller AddRef'd it via CreateRagdollInstance and releases
    // it through PhysicsWorld::DestroyRagdoll.
    JPH::Ragdoll *ragdoll;
    std::size_t body_count;
};

// --- Body operations ---

enum class PhysicsBodyOperationType : std::uint8_t {
    ApplyImpulse,
    ApplyImpulseAt,
    ApplyAngularImpulse,
    ApplyForce,
    ApplyTorque,
    SetPosition,
    SetRotation,
    SetLinearVelocity,
    SetAngularVelocity,
    SetMotionType,
    SetUserData,
    Activate,
    Deactivate,
    SetTransform,
    SetCcd,
};

struct PhysicsBodyOperation {
    PhysicsBodyOperationType type;
    BodyHandle handle;
    JPH::RVec3 position;
    JPH::Vec3 vector;
    JPH::Quat rotation;
    std::uint64_t user_data;
    int value;
};

// --- Runtime query results ---

struct PhysicsBodyStats {
    PosStride position;
    AuxStride rotation;
    AuxStride linear_velocity;
};

// One soft-body vertex read: the position, plus the body's vertex count
// so callers can range-check without a second call.
struct SoftBodyVertexPosition {
    JPH::Vec3 position;
    std::uint32_t vertex_count;
};

// Fixed snapshot prefix written by PhysicsWorld::SaveSnapshot.
struct SnapshotHeader {
    std::size_t body_count;
    double time;
    std::size_t slot_capacity;
};

// --- Ragdoll data exchange ---

struct PhysicsRagdollPartRequest {
    int joint_index;
    int parent_index;
    float mass;
    float twist_min;
    float twist_max;
    float cone_angle;
    JPH::Vec3 axis;
    JPH::Vec3 normal;
    bool has_position;
    JPH::RVec3 position;
};

struct PhysicsRagdollBodyHandleInfo {
    BodyHandle handle;
    bool alive;
};

struct PhysicsRagdollBodyDebugInfo {
    JPH::RVec3 position;
    JPH::Quat rotation;
    JPH::Vec3 linear_velocity;
};

} // namespace CPH
