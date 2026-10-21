#include "ragdolls/Ragdoll.hpp"
#include "physics/PhysicsParams.hpp"
#include "physics/PhysicsWorldCreation.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/SlotState.hpp"

#include <Jolt/Core/Reference.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Skeleton/SkeletonPose.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <utility>

namespace CPH {

// --- Skeleton ----------------------------------------------------------------

std::unique_ptr<Skeleton> Skeleton::Create() {
    std::unique_ptr<Skeleton> skeleton(new Skeleton());
    skeleton->skeleton_ = new JPH::Skeleton();
    // joltc's JPH_Skeleton_Create returned an already-AddRef'd object; match it
    // so the ownership accounting stays identical.
    skeleton->skeleton_->AddRef();
    return skeleton;
}

Skeleton::~Skeleton() {
    if (skeleton_ != nullptr) {
        skeleton_->Release();
    }
}

int Skeleton::AddJoint(const char *name, int parent_index) {
    if (skeleton_ == nullptr || name == nullptr) {
        return -1;
    }
    return static_cast<int>(skeleton_->AddJoint(name, parent_index));
}

int Skeleton::GetJointIndex(const char *name) const {
    if (skeleton_ == nullptr || name == nullptr) {
        return -1;
    }
    return skeleton_->GetJointIndex(name);
}

bool Skeleton::Finalize() {
    if (skeleton_ == nullptr) {
        return false;
    }
    skeleton_->CalculateParentJointIndices();
    return skeleton_->AreJointsCorrectlyOrdered();
}

// --- RagdollSettings ---------------------------------------------------------

std::unique_ptr<RagdollSettings> RagdollSettings::Create(JPH::Skeleton *skeleton) {
    if (skeleton == nullptr) {
        return nullptr;
    }

    std::unique_ptr<RagdollSettings> settings(new RagdollSettings());
    settings->settings_ = new JPH::RagdollSettings();
    // joltc's JPH_RagdollSettings_Create returned an already-AddRef'd object.
    settings->settings_->AddRef();
    settings->settings_->mSkeleton = skeleton; // Ref<> assignment takes its own reference
    return settings;
}

RagdollSettings::~RagdollSettings() {
    if (settings_ != nullptr) {
        settings_->Release();
    }
}

std::expected<int, Error> RagdollSettings::AddPart(const JPH::Shape *shape,
                                                   const PhysicsRagdollPartRequest &request) {
    if (settings_ == nullptr) {
        return std::unexpected(RagdollError::InvalidArgument);
    }
    if (shape == nullptr) {
        return std::unexpected(RagdollError::InvalidShape);
    }

    const JPH::Skeleton *skeleton = settings_->GetSkeleton();
    if (skeleton == nullptr) {
        return std::unexpected(RagdollError::InvalidArgument);
    }
    const int joint_count = skeleton->GetJointCount();
    if (request.joint_index < 0 || request.joint_index >= joint_count) {
        return std::unexpected(RagdollError::JointOutOfRange);
    }

    if (static_cast<int>(settings_->mParts.size()) <= request.joint_index) {
        settings_->mParts.resize(joint_count);
    }

    JPH::RagdollSettings::Part &part = settings_->mParts[request.joint_index];
    part.SetShape(shape);
    // joltc set CalculateInertia, not MassAndInertia: the mass is overridden but
    // the inertia tensor is still derived from the shape.
    part.mMassPropertiesOverride.mMass = request.mass;
    part.mOverrideMassProperties       = JPH::EOverrideMassProperties::CalculateInertia;
    part.mObjectLayer                  = 1;
    part.mMotionType                   = JPH::EMotionType::Dynamic;

    if (request.has_position) {
        part.mPosition = request.position;
    }

    if (request.parent_index >= 0) {
        // A fresh SwingTwistConstraintSettings carries Jolt's defaults, which is
        // exactly what the JoltC path produced: JPH_SwingTwistConstraintSettings_Init
        // memset the struct and then copied the defaults back out of a
        // default-constructed native settings object, and FromJolt mirrored every
        // field SetPartToParent later read. So default-constructing here and
        // applying only the overrides below is equivalent field for field.
        auto *constraint = new JPH::SwingTwistConstraintSettings();
        // TODO(bug): the JoltC path wrote constraint.base.enabled = true, but
        // JPH_RagdollSettings_SetPartToParent never copied `base` across (nor
        // constraintPriority, the step-count overrides, drawConstraintSize or
        // userData), so that write was dead. It was harmless only because
        // ConstraintSettings::mEnabled defaults to true (Constraint.h:15). Left
        // unset here to match observable behaviour; setting it would be a no-op
        // today and a behaviour change if that default ever moved.
        constraint->mPosition1           = JPH::RVec3::sZero();
        constraint->mPosition2           = JPH::RVec3::sZero();
        constraint->mTwistAxis1          = request.axis;
        constraint->mTwistAxis2          = request.axis;
        constraint->mPlaneAxis1          = request.normal;
        constraint->mPlaneAxis2          = request.normal;
        constraint->mNormalHalfConeAngle = request.cone_angle;
        constraint->mPlaneHalfConeAngle  = request.cone_angle;
        constraint->mTwistMinAngle       = request.twist_min;
        constraint->mTwistMaxAngle       = request.twist_max;
        part.mToParent                   = constraint;
    }

    return joint_count;
}

bool RagdollSettings::Stabilize() {
    return settings_ != nullptr && settings_->Stabilize();
}

// --- Ragdoll -----------------------------------------------------------------

Ragdoll::Ragdoll(PhysicsWorld &world, JPH::Ragdoll *ragdoll, Buffer<std::uint32_t> body_slots)
    : world_(&world), ragdoll_(ragdoll), body_slots_(std::move(body_slots)) {}

std::expected<std::unique_ptr<Ragdoll>, Error>
Ragdoll::Create(PhysicsWorld &world, RagdollSettings &settings, JPH::RVec3 position,
                JPH::Quat rotation, std::uint64_t user_data, std::uint32_t category,
                std::uint32_t mask, std::uint32_t material_id) {
    const std::expected<RagdollInstance, Error> instance =
        world.CreateRagdollInstance(settings.raw(), position, rotation, user_data);
    if (!instance) {
        return std::unexpected(instance.error());
    }

    Buffer<std::uint32_t> body_slots;
    if (instance->body_count != 0 && !body_slots.allocate(instance->body_count)) {
        world.DestroyRagdoll(instance->ragdoll);
        return std::unexpected(WorldCreationError::OutOfMemory);
    }

    const std::expected<std::size_t, Error> registered = world.RegisterRagdollBodies(
        instance->ragdoll, {body_slots.data(), instance->body_count}, user_data, category, mask,
        material_id);
    if (!registered) {
        // DestroyRagdoll takes the instance back out of the system and releases
        // it; the slots were never registered, so nothing else refers to them.
        world.DestroyRagdoll(instance->ragdoll);
        return std::unexpected(registered.error());
    }

    std::unique_ptr<Ragdoll> ragdoll(
        new Ragdoll(world, instance->ragdoll, std::move(body_slots)));
    ragdoll->registered_body_count_ = *registered;
    return ragdoll;
}

void Ragdoll::Detach() {
    if (ragdoll_ == nullptr) {
        return;
    }

    JPH::Ragdoll *ragdoll = ragdoll_;
    ragdoll_              = nullptr;

    if (world_ == nullptr) {
        ragdoll->Release();
        return;
    }

    {
        std::unique_lock shadow(world_->LockShadowNotStepping());
        world_->WaitUntilNotQuerying(shadow);

        ragdoll->RemoveFromPhysicsSystem(true);

        if (world_->bodies.slot_states.data() != nullptr) {
            for (std::size_t i = 0; i < body_slots_.size(); ++i) {
                const std::uint32_t slot = body_slots_[i];
                if (slot >= world_->slot_capacity) {
                    continue;
                }
                const std::uint8_t state =
                    world_->bodies.slot_states[slot].load(std::memory_order::acquire);
                if (state == std::to_underlying(SlotState::Alive)) {
                    world_->RemoveBodySlot(slot);
                }
            }
        }
    }

    // A Jolt-system mutation, so it takes the trampoline lock alone: holding
    // the shadow lock here would invert the documented order.
    const std::scoped_lock trampoline(world_->jph_trampoline_lock);
    ragdoll->Release();
}

Ragdoll::~Ragdoll() { Detach(); }

int Ragdoll::JointCount() const {
    if (ragdoll_ == nullptr) {
        return 0;
    }
    const JPH::RagdollSettings *settings = ragdoll_->GetRagdollSettings();
    if (settings == nullptr) {
        return 0;
    }
    const JPH::Skeleton *skeleton = settings->GetSkeleton();
    return skeleton != nullptr ? skeleton->GetJointCount() : 0;
}

int Ragdoll::BodyCount() const {
    return ragdoll_ != nullptr ? static_cast<int>(ragdoll_->GetBodyCount()) : 0;
}

bool Ragdoll::DriveToPose(JPH::RVec3 root_position, std::span<const JPH::Mat44> matrices) {
    if (ragdoll_ == nullptr || world_ == nullptr ||
        matrices.size() > static_cast<std::size_t>(INT_MAX)) {
        return false;
    }

    const JPH::RagdollSettings *settings = ragdoll_->GetRagdollSettings();
    if (settings == nullptr) {
        return false;
    }
    const JPH::Skeleton *skeleton = settings->GetSkeleton();
    if (skeleton == nullptr) {
        return false;
    }

    JPH::SkeletonPose pose;
    pose.SetSkeleton(skeleton);
    pose.SetRootOffset(root_position);

    {
        std::unique_lock shadow(world_->LockShadowNotStepping());
        ragdoll_->SetPose(root_position, matrices.data(), true);

        // joltc clamped the copy to the pose's own joint count rather than
        // resizing or rejecting a mismatch; keep that, since SetSkeleton has
        // already sized mJointMatrices from the skeleton.
        JPH::SkeletonPose::Mat44Vector &pose_matrices = pose.GetJointMatrices();
        const int total = static_cast<int>(pose_matrices.size());
        const int count = std::min(total, static_cast<int>(matrices.size()));
        for (int i = 0; i < count; i++) {
            pose_matrices[i] = matrices[i];
        }
        ragdoll_->Activate(true);
        ragdoll_->DriveToPoseUsingMotors(pose);
    }
    return true;
}

void Ragdoll::CopyBodyHandles(std::span<PhysicsRagdollBodyHandleInfo> out) const {
    // Mismatched sizes used to leave the caller's buffer untouched; keep that.
    if (body_slots_.size() != out.size()) {
        return;
    }

    if (world_ == nullptr) {
        // No world left to resolve slots against: report every part as dead.
        for (PhysicsRagdollBodyHandleInfo &info : out) {
            info = PhysicsRagdollBodyHandleInfo{.handle = BodyHandle{}, .alive = false};
        }
        return;
    }

    const std::scoped_lock shadow(world_->shadow_lock);
    for (std::size_t i = 0; i < body_slots_.size(); ++i) {
        out[i]              = PhysicsRagdollBodyHandleInfo{.handle = BodyHandle{}, .alive = false};
        const std::uint32_t slot = body_slots_[i];
        if (slot >= world_->slot_capacity || world_->bodies.slot_states.data() == nullptr ||
            world_->bodies.generations.data() == nullptr) {
            continue;
        }
        const std::uint8_t state =
            world_->bodies.slot_states[slot].load(std::memory_order::acquire);
        if (state != std::to_underlying(SlotState::Alive)) {
            continue;
        }
        const std::uint32_t generation =
            world_->bodies.generations[slot].load(std::memory_order::relaxed);
        out[i] = PhysicsRagdollBodyHandleInfo{
            .handle = BodyHandle::FromParts(slot, generation),
            .alive  = true,
        };
    }
}

bool Ragdoll::CopyDebugInfo(std::span<PhysicsRagdollBodyDebugInfo> out) const {
    if (ragdoll_ == nullptr || world_ == nullptr || world_->body_interface == nullptr) {
        return false;
    }

    const int ragdoll_body_count = static_cast<int>(ragdoll_->GetBodyCount());
    if (ragdoll_body_count < 0 ||
        static_cast<std::size_t>(ragdoll_body_count) != out.size()) {
        return false;
    }

    const std::scoped_lock shadow(world_->shadow_lock);
    JPH::BodyInterface &bodies = *world_->body_interface;
    for (int i = 0; i < ragdoll_body_count; ++i) {
        const JPH::BodyID body_id = ragdoll_->GetBodyID(i);
        out[i].position           = bodies.GetPosition(body_id);
        // TODO(smell): rotation is fetched and stored, but no caller has ever
        // surfaced it; preserved verbatim from the C path.
        out[i].rotation        = bodies.GetRotation(body_id);
        out[i].linear_velocity = bodies.GetLinearVelocity(body_id);
    }
    return true;
}

} // namespace CPH
