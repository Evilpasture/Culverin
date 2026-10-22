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
    const std::uint8_t *motionTypes;
    const std::uint8_t *objectLayers;
    const std::uint64_t *userData;
};

struct ShapeParams {
    float p[4];
};

struct TrackData {
    std::uint32_t *indices;
    std::uint32_t count;
    std::uint32_t drivenIdx;
};

// Constraint parameters. Default member initializers carry the sane
// defaults the old CPH_ParamsInit() applied (20 Hz / critically damped
// motors, unlimited limits), so a value-initialized struct is ready to use.
struct ConstraintParams {
    float px = 0.0F, py = 0.0F, pz = 0.0F;
    float ax = 0.0F, ay = 1.0F, az = 0.0F;
    float limitMin       = -std::numeric_limits<float>::max();
    float limitMax       = std::numeric_limits<float>::max();
    float halfConeAngle = 0.0F;
    bool hasMotor        = false;
    int motorType        = 0;
    float motorTarget    = 0.0F;
    float maxTorque      = 1e6F;
    float frequency       = 20.0F;
    float damping         = 1.0F;
    float maxForce       = 1e6F;
};

// --- Body creation requests ---

struct PhysicsWorldBodyCreateRequest {
    // Shape-cache key; unused (0) when the caller creates the shape itself
    // (mesh/heightfield paths).
    int shapeType          = 0;
    float shapeParams[4]   = {};
    JPH::RVec3 position     = JPH::RVec3::sZero();
    JPH::Quat rotation      = JPH::Quat::sIdentity();
    BodyConfig config;
    uint32_t category       = kAllLayerBits;
    uint32_t mask           = kAllLayerBits;
    uint32_t materialId    = 0;
    uint64_t userData      = 0;
};

struct PhysicsWorldCompoundPart {
    int shapeType          = 0;
    float shapeParams[4]   = {};
    JPH::Vec3 localPosition = JPH::Vec3::sZero();
    JPH::Quat localRotation = JPH::Quat::sIdentity();
};

struct PhysicsWorldSoftBodyCreateRequest {
    // Const: SoftBodyCreationSettings::mSettings holds it read-only, which is
    // why the old code needed a reinterpret_cast here.
    const JPH::SoftBodySharedSettings *sharedSettings = nullptr;
    uint32_t vertexCount                       = 0;
    JPH::RVec3 position                         = JPH::RVec3::sZero();
    JPH::Quat rotation                          = JPH::Quat::sIdentity();
    uint64_t userData                          = 0;
    uint32_t category                           = kAllLayerBits;
    uint32_t mask                               = kAllLayerBits;
    float pressure                              = 0.0f;
    float vertexRadius                         = 0.0f;
    float linearDamping                        = 0.0f;
    uint32_t numIterations                     = 0;
    float maxLinearVelocity                   = 0.0f;
    float gravityFactor                        = 0.0f;
    float friction                              = 0.0f;
    float restitution                           = 0.0f;
    bool makeRotationIdentity                 = false;
    bool updatePosition                        = false;
    bool facesDoubleSided                     = false;
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
    std::size_t bodyCount;
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
    std::uint64_t userData;
    int value;
};

// --- Runtime query results ---

struct PhysicsBodyStats {
    PosStride position;
    AuxStride rotation;
    AuxStride linearVelocity;
};

// One soft-body vertex read: the position, plus the body's vertex count
// so callers can range-check without a second call.
struct SoftBodyVertexPosition {
    JPH::Vec3 position;
    std::uint32_t vertexCount;
};

// Fixed snapshot prefix written by PhysicsWorld::SaveSnapshot.
struct SnapshotHeader {
    std::size_t bodyCount;
    double time;
    std::size_t slotCapacity;
};

// --- Ragdoll data exchange ---

struct PhysicsRagdollPartRequest {
    int jointIndex;
    int parentIndex;
    float mass;
    float twistMin;
    float twistMax;
    float coneAngle;
    JPH::Vec3 axis;
    JPH::Vec3 normal;
    bool hasPosition;
    JPH::RVec3 position;
};

struct PhysicsRagdollBodyHandleInfo {
    BodyHandle handle;
    bool alive;
};

struct PhysicsRagdollBodyDebugInfo {
    JPH::RVec3 position;
    JPH::Quat rotation;
    JPH::Vec3 linearVelocity;
};

} // namespace CPH
