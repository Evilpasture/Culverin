#pragma once

// Ragdolls: the skeleton, the settings that describe the parts, and the
// runtime instance that owns the Jolt ragdoll and its world body slots.
//
// This used to be a set of free functions in src/physics/ with joltc's
// ownership model (raw pointers, an AddRef on create matched by a Release on
// destroy, and int status codes). The three types are ordinary RAII classes
// now, and the world-side locking that used to live in the binding moved into
// Ragdoll itself.

#include "core/Buffer.hpp"
#include "core/Error.hpp"
#include "jolt/Jolt.hpp"
#include "physics/PhysicsParams.hpp"

#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Skeleton/Skeleton.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

namespace CPH {

struct PhysicsWorld;

enum class RagdollError : std::uint32_t {
    InvalidArgument = 1,
    InvalidShape    = 2,
    JointOutOfRange = 3,
};

// A Jolt Skeleton, owned. `Create` returns an object that holds one reference,
// which is exactly what joltc's JPH_Skeleton_Create left behind; nothing else
// refers to a skeleton except the settings that take their own Ref.
class Skeleton {
  public:
    [[nodiscard]] static std::unique_ptr<Skeleton> Create();

    ~Skeleton();

    Skeleton(const Skeleton &)            = delete;
    Skeleton &operator=(const Skeleton &) = delete;
    Skeleton(Skeleton &&)                 = delete;
    Skeleton &operator=(Skeleton &&)      = delete;

    // Returns the new joint's index, or -1 when the name is null.
    [[nodiscard]] int AddJoint(const char *name, int parent_index);
    [[nodiscard]] int GetJointIndex(const char *name) const;

    // Calculates the parent indices and reports whether every joint was added
    // after its parent.
    [[nodiscard]] bool Finalize();

    [[nodiscard]] JPH::Skeleton *raw() const noexcept { return skeleton_; }

  private:
    Skeleton() = default;

    JPH::Skeleton *skeleton_ = nullptr;
};

// RagdollSettings plus the part layout behind it.
class RagdollSettings {
  public:
    [[nodiscard]] static std::unique_ptr<RagdollSettings> Create(JPH::Skeleton *skeleton);

    ~RagdollSettings();

    RagdollSettings(const RagdollSettings &)            = delete;
    RagdollSettings &operator=(const RagdollSettings &) = delete;
    RagdollSettings(RagdollSettings &&)                 = delete;
    RagdollSettings &operator=(RagdollSettings &&)      = delete;

    // Adds (or overwrites) the part for the request's joint. Returns the joint
    // count on success.
    [[nodiscard]] std::expected<int, Error> AddPart(const JPH::Shape *shape,
                                                    const PhysicsRagdollPartRequest &request);

    [[nodiscard]] bool Stabilize();

    [[nodiscard]] JPH::RagdollSettings *raw() const noexcept { return settings_; }

  private:
    RagdollSettings() = default;

    JPH::RagdollSettings *settings_ = nullptr;
};

// A live ragdoll: the Jolt instance, its world body slots and the world it
// lives in. Destroying it takes it out of the physics system and releases the
// slots, which is what the binding's teardown used to do by hand.
class Ragdoll {
  public:
    [[nodiscard]] static std::expected<std::unique_ptr<Ragdoll>, Error>
    Create(PhysicsWorld &world, RagdollSettings &settings, JPH::RVec3 position,
           JPH::Quat rotation, std::uint64_t user_data, std::uint32_t category,
           std::uint32_t mask, std::uint32_t material_id);

    ~Ragdoll();

    Ragdoll(const Ragdoll &)            = delete;
    Ragdoll &operator=(const Ragdoll &) = delete;
    Ragdoll(Ragdoll &&)                 = delete;
    Ragdoll &operator=(Ragdoll &&)      = delete;

    [[nodiscard]] int JointCount() const;
    [[nodiscard]] int BodyCount() const;

    // The world body count after the parts were registered. The binding
    // publishes it to the shape view, as it always has.
    [[nodiscard]] std::size_t registered_body_count() const noexcept {
        return registered_body_count_;
    }

    // Drives the ragdoll towards a pose described by one matrix per joint.
    // Parks until the world is not stepping; false means Jolt was in no state
    // to accept the pose.
    [[nodiscard]] bool DriveToPose(JPH::RVec3 root_position,
                                   std::span<const JPH::Mat44> matrices);

    // Fills `out` with the current handle of every part's body slot.
    void CopyBodyHandles(std::span<PhysicsRagdollBodyHandleInfo> out) const;

    // Fills `out` with every part's transform; false when the sizes disagree
    // or the world has no body interface.
    [[nodiscard]] bool CopyDebugInfo(std::span<PhysicsRagdollBodyDebugInfo> out) const;

    [[nodiscard]] JPH::Ragdoll *raw() const noexcept { return ragdoll_; }

  private:
    Ragdoll(PhysicsWorld &world, JPH::Ragdoll *ragdoll, Buffer<std::uint32_t> body_slots);

    void Detach();

    PhysicsWorld         *world_ = nullptr;
    JPH::Ragdoll         *ragdoll_ = nullptr;
    Buffer<std::uint32_t> body_slots_;
    std::size_t           registered_body_count_ = 0;
};

} // namespace CPH
