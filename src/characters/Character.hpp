#pragma once

// A character: a JPH::CharacterVirtual plus the contact listener that turns
// its collisions into Culverin contact events and pushes dynamic bodies aside.
//
// This is engine code, not binding glue. It used to live in the Python
// binding, which meant the platform-interaction maths -- inheriting the
// velocity of whatever the character stands on, scaled by that surface's
// friction -- could only be run from Python. A headless build gets exactly the
// same behaviour now.
//
// One instance owns one listener, so characters share no contact state.

#include "core/Handle.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <atomic>
#include <cstdint>
#include <memory>

namespace CPH {

struct PhysicsWorld;

struct CharacterParams {
    float height    = 1.8f;
    float radius    = 0.4f;
    float maxSlope = 45.0f; // degrees
};

class Character {
  public:
    // Creates the Jolt character, claims a body slot in the world, publishes
    // its transform into the shadow buffers and installs the contact
    // listener. Returns nullptr if the capsule shape cannot be created or the
    // world cannot grow.
    [[nodiscard]] static std::unique_ptr<Character> Create(PhysicsWorld &world,
                                                           const CharacterParams &params,
                                                           const JPH::RVec3 &position);

    ~Character();

    Character(const Character &)            = delete;
    Character &operator=(const Character &) = delete;
    Character(Character &&)                 = delete;
    Character &operator=(Character &&)      = delete;

    [[nodiscard]] JPH::CharacterVirtual *Virtual() const noexcept { return virtual_; }
    [[nodiscard]] BodyHandle Handle() const noexcept {
        return handle_.load(std::memory_order::relaxed);
    }

    // Stronger characters push dynamic bodies harder; this also feeds the
    // impulse the contact listener applies.
    void SetStrength(float strength);

    // Drives one step: inherits the ground platform's velocity, runs Jolt's
    // extended update and republishes the transform into the shadow buffers.
    void Move(float vx, float vy, float vz, float dt);

    [[nodiscard]] JPH::RVec3 GetPosition() const;
    [[nodiscard]] JPH::Quat  GetRotation() const;
    [[nodiscard]] JPH::Vec3  GetLinearVelocity() const;

    void SetPosition(const JPH::RVec3 &position);
    void SetRotation(const JPH::Quat &rotation);

    [[nodiscard]] bool IsGrounded() const;

  private:
    class Listener;

    Character(PhysicsWorld &world, JPH::CharacterVirtual *virtual_character);

    // Republishes the current transform into the world's shadow buffers.
    // Caller holds shadow_lock.
    void PublishTransformLocked() const;

    [[nodiscard]] std::uint32_t DenseIndexLocked() const noexcept;

    // True when both bodies are live and each one's mask admits the other's
    // category. Anything unverifiable collides, matching the pre-migration
    // behaviour of falling through to `return true`.
    [[nodiscard]] bool LayersAllow(BodyHandle self, BodyHandle other) const noexcept;

    PhysicsWorld           *world_;
    JPH::CharacterVirtual  *virtual_ = nullptr;
    std::unique_ptr<Listener> listener_;

    // Read by the contact listener on Jolt worker threads, written by the
    // owning thread, so they are atomics and never locked.
    std::atomic<BodyHandle> handle_{};
    std::atomic<float>      pushStrength_{0.0f};
    std::atomic<float>      lastVx_{0.0f};
    std::atomic<float>      lastVy_{0.0f};
    std::atomic<float>      lastVz_{0.0f};
};

} // namespace CPH
