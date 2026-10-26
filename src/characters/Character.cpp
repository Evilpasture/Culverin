#include "characters/Character.hpp"
#include "core/Types.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/PhysicsWorldState.hpp"
#include "physics/RotationSanitize.hpp"
#include "physics/SlotState.hpp"

#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Body/BodyFilter.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace CPH {

namespace {

constexpr float kMinimumHalfHeight = 0.1f;

// The contact ring buffer is written from Jolt worker threads. Every path here
// is the same shape: bump the atomic cursor, and only touch the slot if the
// bump landed inside the buffer.
ContactEvent *ClaimContactEvent(PhysicsWorld &world, size_t &out_index) {
    out_index = world.contactAtomicIdx.fetch_add(1, std::memory_order::relaxed);
    if (out_index >= world.contactMaxCapacity) {
        return nullptr;
    }
    return &world.contactBuffer[out_index];
}

// Canonical body ordering: the smaller handle first, so a pair is recorded
// identically no matter which side noticed the contact.
void StoreOrdered(ContactEventSlim *slim, BodyHandle a, BodyHandle b) {
    if (a < b) {
        slim->body1.store(a, std::memory_order::relaxed);
        slim->body2.store(b, std::memory_order::relaxed);
    } else {
        slim->body1.store(b, std::memory_order::relaxed);
        slim->body2.store(a, std::memory_order::relaxed);
    }
}

} // namespace

/// Turns Jolt's CharacterVirtual callbacks into Culverin contact events and
/// the body-push impulse. Owns nothing: the Character it points at owns it.
class Character::Listener final : public JPH::CharacterContactListener {
  public:
    explicit Listener(Character &owner) : owner_(owner) {}

    // --- platform interaction ------------------------------------------------

    // Inherit the velocity of whatever the character stands on, scaled by that
    // surface's friction.
    void OnAdjustBodyVelocity([[maybe_unused]] const JPH::CharacterVirtual *character,
                              const JPH::Body &body2, JPH::Vec3 &io_linear_velocity,
                              JPH::Vec3 &io_angular_velocity) override {
        PhysicsWorld &world = *owner_.world_;

        // 1. Platform material -> friction.
        const BodyHandle platform_handle(body2.GetUserData());
        float friction = 1.0f;
        const auto slot = platform_handle.Index();
        if (slot < world.slotCapacity) {
            const std::uint32_t dense = world.bodies.slotToDense[slot];
            const std::uint32_t material = world.bodies.materialIds[dense];
            for (std::size_t i = 0; i < world.materialCount; i++) {
                if (world.materials[i].id == material) {
                    friction = world.materials[i].friction;
                    break;
                }
            }
        }

        // 2. Tangential velocity at the character's offset from the platform.
        const JPH::Vec3 omega    = body2.GetAngularVelocity();
        const JPH::RVec3 char_pos = character->GetPosition();
        const JPH::RVec3 plat_pos = body2.GetPosition();

        const float rx = static_cast<float>(char_pos.GetX() - plat_pos.GetX());
        const float rz = static_cast<float>(char_pos.GetZ() - plat_pos.GetZ());

        // v = omega x r
        const float target_vt_x = omega.GetY() * rz;
        const float target_vt_z = -omega.GetY() * rx;

        // 3. Override the inherited velocity with the friction-scaled version.
        // Full grip above 0.2 friction, proportionally less below it.
        const float factor = (friction > 0.2f) ? 1.0f : (friction / 0.2f);

        io_linear_velocity.SetX(target_vt_x * factor);
        io_linear_velocity.SetZ(target_vt_z * factor);
        io_angular_velocity.SetY(omega.GetY() * factor);
    }

    // --- collision filtering -------------------------------------------------

    bool OnContactValidate([[maybe_unused]] const JPH::CharacterVirtual *character,
                           const JPH::CharacterContact &contact) override {
        PhysicsWorld &world = *owner_.world_;

        // The other body's handle, from user data, then the body interface,
        // then the lock-free id -> handle map.
        BodyHandle other(contact.mUserData);
        if (!other.IsValid() && world.bodyInterface) {
            other = BodyHandle(world.physicsSystem.GetBodyInterface().GetUserData(contact.mBodyB));
        }
        if (!other.IsValid() && world.idToHandleMap) {
            const std::uint32_t j_idx = contact.mBodyB.GetIndex();
            if (j_idx <= world.maxJoltBodies) {
                other = world.idToHandleMap[j_idx].load(std::memory_order::acquire);
            }
        }
        if (!other.IsValid()) {
            return true; // static environment
        }
        return owner_.LayersAllow(owner_.Handle(), other);
    }

    bool OnCharacterContactValidate([[maybe_unused]] const JPH::CharacterVirtual *character,
                                    const JPH::CharacterContact &contact) override {
        BodyHandle other(contact.mUserData);
        if (!other.IsValid() && contact.mCharacterB) {
            other = BodyHandle(contact.mCharacterB->GetUserData());
        }
        if (!other.IsValid()) {
            return true;
        }
        return owner_.LayersAllow(owner_.Handle(), other);
    }

    // --- contact events ------------------------------------------------------

    void OnContactAdded([[maybe_unused]] const JPH::CharacterVirtual *character,
                        const JPH::CharacterContact &contact,
                        JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;
        RecordBodyContact(contact.mBodyB, contact.mPosition, contact.mContactNormal,
                          ContactEventType::Added);
        ApplyImpulse(contact.mBodyB, contact.mContactNormal);
    }

    void OnContactPersisted([[maybe_unused]] const JPH::CharacterVirtual *character,
                            const JPH::CharacterContact &contact,
                            JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;
        RecordBodyContact(contact.mBodyB, contact.mPosition, contact.mContactNormal,
                          ContactEventType::Persisted);
        ApplyImpulse(contact.mBodyB, contact.mContactNormal);
    }

    void OnContactRemoved([[maybe_unused]] const JPH::CharacterVirtual *character,
                          const JPH::BodyID &body_id2,
                          [[maybe_unused]] const JPH::SubShapeID &sub_shape_id2) override {
        PhysicsWorld &world = *owner_.world_;
        const BodyHandle self = owner_.Handle();
        const std::uint32_t j_idx = body_id2.GetIndex();

        BodyHandle other;
        if (world.idToHandleMap && j_idx <= world.maxJoltBodies) {
            other = world.idToHandleMap[j_idx].load(std::memory_order::acquire);
        }
        if (!other.IsValid()) {
            return;
        }

        size_t index = 0;
        ContactEvent *event = ClaimContactEvent(world, index);
        if (event == nullptr) {
            return;
        }
        ContactEventSlim *slim = &event->slim;
        slim->flags            = std::to_underlying(ContactEventType::Removed);
        StoreOrdered(slim, self, other);
        slim->px            = 0.0;
        slim->py            = 0.0;
        slim->pz            = 0.0;
        slim->nx            = 0.0f;
        slim->ny            = 0.0f;
        slim->nz            = 0.0f;
        slim->impulse       = 0.0f;
        slim->slidingSpeed = 0.0f;
        std::atomic_thread_fence(std::memory_order::release);
    }

    void OnCharacterContactAdded([[maybe_unused]] const JPH::CharacterVirtual *character,
                                 const JPH::CharacterContact &contact,
                                 JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;
        RecordCharacterContact(contact.mCharacterB, contact.mPosition, contact.mContactNormal,
                               ContactEventType::Added);
    }

    void OnCharacterContactPersisted([[maybe_unused]] const JPH::CharacterVirtual *character,
                                     const JPH::CharacterContact &contact,
                                     JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;
        RecordCharacterContact(contact.mCharacterB, contact.mPosition, contact.mContactNormal,
                               ContactEventType::Persisted);
    }

    void OnCharacterContactRemoved([[maybe_unused]] const JPH::CharacterVirtual *character,
                                   const JPH::CharacterID &other_character_id,
                                   [[maybe_unused]] const JPH::SubShapeID &sub_shape_id2) override {
        PhysicsWorld &world = *owner_.world_;
        const BodyHandle self = owner_.Handle();

        const std::uint32_t j_idx = (other_character_id.GetValue() & JPH::BodyID::cMaxBodyIndex);
        BodyHandle other;
        if (world.idToHandleMap && j_idx <= world.maxJoltBodies) {
            other = world.idToHandleMap[j_idx].load(std::memory_order::acquire);
        }
        if (!other.IsValid()) {
            return;
        }

        size_t index = 0;
        ContactEvent *event = ClaimContactEvent(world, index);
        if (event == nullptr) {
            return;
        }
        ContactEventSlim *slim = &event->slim;
        slim->flags            = std::to_underlying(ContactEventType::Removed);
        StoreOrdered(slim, self, other);
        slim->px            = 0.0;
        slim->py            = 0.0;
        slim->pz            = 0.0;
        slim->nx            = 0.0f;
        slim->ny            = 0.0f;
        slim->nz            = 0.0f;
        slim->impulse       = 0.0f;
        slim->slidingSpeed = 0.0f;
        std::atomic_thread_fence(std::memory_order::release);
    }

    // OnContactSolve / OnCharacterContactSolve are deliberately not overridden:
    // the base implementations are no-ops.

  private:
    // Records a character-vs-body contact. Runs on Jolt worker threads: no
    // allocation, no locking.
    void RecordBodyContact(JPH::BodyID body_id2, JPH::RVec3Arg position, JPH::Vec3Arg normal,
                           ContactEventType type) {
        PhysicsWorld &world = *owner_.world_;
        const std::uint32_t j_idx = body_id2.GetIndex();

        BodyHandle other;
        if (world.idToHandleMap && j_idx <= world.maxJoltBodies) {
            other = world.idToHandleMap[j_idx].load(std::memory_order::acquire);
        }
        if (!other.IsValid()) {
            return;
        }

        size_t index = 0;
        ContactEvent *event = ClaimContactEvent(world, index);
        if (event == nullptr) {
            return;
        }

        const BodyHandle self  = owner_.Handle();
        ContactEventSlim *slim = &event->slim;
        ContactEventFatExt *fat = &event->fat;

        slim->flags = static_cast<std::uint32_t>(type);
        StoreOrdered(slim, self, other);

        slim->px            = static_cast<JPH::Real>(position.GetX());
        slim->py            = static_cast<JPH::Real>(position.GetY());
        slim->pz            = static_cast<JPH::Real>(position.GetZ());
        slim->nx            = normal.GetX();
        slim->ny            = normal.GetY();
        slim->nz            = normal.GetZ();
        slim->impulse       = 1.0f;
        slim->slidingSpeed = 0.0f;

        const std::uint32_t dense = world.bodies.slotToDense[other.Index()];
        fat->mat1                 = 0;
        fat->mat2                 = world.bodies.materialIds[dense];

        std::atomic_thread_fence(std::memory_order::release);
    }

    // Records a character-vs-character contact. The peer's handle comes
    // straight off its user data, which is authoritative.
    void RecordCharacterContact(const JPH::CharacterVirtual *other, JPH::RVec3Arg position,
                                JPH::Vec3Arg normal, ContactEventType type) {
        if (other == nullptr) {
            return;
        }
        const BodyHandle peer(other->GetUserData());
        if (!peer.IsValid()) {
            return; // a Jolt-internal body
        }

        PhysicsWorld &world = *owner_.world_;
        size_t index = 0;
        ContactEvent *event = ClaimContactEvent(world, index);
        if (event == nullptr) {
            return;
        }

        const BodyHandle self   = owner_.Handle();
        ContactEventSlim *slim  = &event->slim;
        ContactEventFatExt *fat = &event->fat;

        slim->flags = static_cast<std::uint32_t>(type);
        StoreOrdered(slim, self, peer);

        slim->px            = static_cast<JPH::Real>(position.GetX());
        slim->py            = static_cast<JPH::Real>(position.GetY());
        slim->pz            = static_cast<JPH::Real>(position.GetZ());
        slim->nx            = normal.GetX();
        slim->ny            = normal.GetY();
        slim->nz            = normal.GetZ();
        slim->impulse       = 1.0f;
        slim->slidingSpeed = 0.0f;
        fat->mat1           = 0;
        fat->mat2           = 0;

        std::atomic_thread_fence(std::memory_order::release);
    }

    // Pushes a dynamic, non-sensor body away from the character along the
    // contact normal, scaled by how fast the character is moving into it.
    void ApplyImpulse(JPH::BodyID body_id2, JPH::Vec3Arg contact_normal) {
        PhysicsWorld &world = *owner_.world_;
        JPH::BodyInterface &bodies = world.physicsSystem.GetBodyInterface();

        if (bodies.IsSensor(body_id2) ||
            bodies.GetMotionType(body_id2) != JPH::EMotionType::Dynamic) {
            return;
        }

        const float vx = owner_.lastVx_.load(std::memory_order::relaxed);
        const float vy = owner_.lastVy_.load(std::memory_order::relaxed);
        const float vz = owner_.lastVz_.load(std::memory_order::relaxed);
        const float strength = owner_.pushStrength_.load(std::memory_order::relaxed);

        const float dot =
            vx * contact_normal.GetX() + vy * contact_normal.GetY() + vz * contact_normal.GetZ();

        // The normal points towards the character, so a negative dot means the
        // character is moving into the body.
        if (dot >= -0.01f) {
            return;
        }

        float factor = -dot * strength;
        constexpr float kMaxImpulse = 50000.0f;
        factor = std::min(factor, kMaxImpulse);

        // Flip the normal to push the body away; never push it down.
        const float y_push = -contact_normal.GetY() * factor;
        const JPH::Vec3 impulse(-contact_normal.GetX() * factor, (y_push > 0.0f) ? y_push : 0.0f,
                                -contact_normal.GetZ() * factor);

        bodies.AddImpulse(body_id2, impulse);
        bodies.ActivateBody(body_id2);
    }

    Character &owner_;
};

// --- Character ---------------------------------------------------------------

Character::Character(PhysicsWorld &world, JPH::CharacterVirtual *virtual_character)
    : world_(&world), virtual_(virtual_character) {}

std::unique_ptr<Character> Character::Create(PhysicsWorld &world, const CharacterParams &params,
                                             const JPH::RVec3 &position) {
    // --- 1. Claim a body slot --------------------------------------------
    std::uint32_t slot = 0;
    {
        std::unique_lock shadow = world.LockShadowNotStepping();
        world.WaitUntilNotQuerying(shadow);

        std::size_t available = world.freeCount.load(std::memory_order::acquire);
        if (available == 0) {
            const std::expected<void, Error> grown = world.Resize(world.capacity * 2);
            if (!grown) {
                return nullptr;
            }
            available = world.freeCount.load(std::memory_order::acquire);
        }
        if (available == 0) {
            return nullptr;
        }

        slot = world.bodies.freeSlots[available - 1];
        world.freeCount.store(available - 1, std::memory_order::release);
        world.bodies.slotStates[slot].store(std::to_underlying(SlotState::PendingCreate),
                                             std::memory_order::release);
    }

    // From here on, any failure has to hand the slot back.
    const auto release_slot = [&world, slot]() {
        const std::scoped_lock shadow(world.shadowLock);
        world.bodies.slotStates[slot].store(std::to_underlying(SlotState::Empty),
                                             std::memory_order::relaxed);
        const std::size_t free_index = world.freeCount.fetch_add(1, std::memory_order::relaxed);
        world.bodies.freeSlots[free_index] = slot;
    };

    // --- 2. Build the Jolt character -------------------------------------
    const float half_height =
        std::fmax((params.height - 2.0f * params.radius) * 0.5f, kMinimumHalfHeight);
    const JPH::ShapeSettings::ShapeResult shape_result =
        JPH::CapsuleShapeSettings(half_height, params.radius).Create();
    if (!shape_result.IsValid()) {
        release_slot();
        return nullptr;
    }

    JPH::CharacterVirtualSettings settings;
    settings.mShape         = shape_result.Get();
    settings.mMaxSlopeAngle = JPH::DegreesToRadians(params.maxSlope);

    std::unique_ptr<Character> character(
        new Character(world, new JPH::CharacterVirtual(&settings, position, JPH::Quat::sIdentity(),
                                                       1, &world.physicsSystem)));

    world.characterVsCharacter.Add(character->virtual_);
    character->virtual_->SetCharacterVsCharacterCollision(&world.characterVsCharacter);

    // --- 3. Publish into the world's shadow state -------------------------
    {
        const std::scoped_lock shadow(world.shadowLock);

        const std::uint32_t generation =
            world.bodies.generations[slot].load(std::memory_order::relaxed);
        const BodyHandle handle = BodyHandle::FromParts(slot, generation);
        character->handle_.store(handle, std::memory_order::relaxed);

        const auto dense = static_cast<std::uint32_t>(world.count.load(std::memory_order::relaxed));
        const JPH::BodyID inner = character->virtual_->GetInnerBodyID();
        const std::uint32_t j_idx = inner.GetIndex();
        if (j_idx <= world.maxJoltBodies) {
            world.idToHandleMap[j_idx].store(handle, std::memory_order::release);
        }

        world.bodies.bodyIds[dense]      = inner;
        world.bodies.slotToDense[slot]  = dense;
        world.bodies.denseToSlot[dense] = slot;
        world.bodies.userData[dense]     = 0;

        constexpr auto kCollisionFilterAllCategories = 0xFFFF;
        constexpr auto kCollisionFilterAllMasks      = 0xFFFF;
        world.bodies.categories[dense]   = kCollisionFilterAllCategories;
        world.bodies.masks[dense]        = kCollisionFilterAllMasks;
        world.bodies.materialIds[dense] = 0;

        const JPH::RVec3 p = character->virtual_->GetPosition();
        const JPH::Quat q  = character->virtual_->GetRotation();
        const PosStride p_val{p.GetX(), p.GetY(), p.GetZ(), 0.0};
        const AuxStride r_val{q.GetX(), q.GetY(), q.GetZ(), q.GetW()};

        world.bodies.positions[dense]            = p_val;
        world.bodies.prevPositions[dense]       = p_val;
        world.bodies.rotations[dense]            = r_val;
        world.bodies.prevRotations[dense]       = r_val;
        world.bodies.linearVelocities[dense]    = AuxStride{};
        world.bodies.angularVelocities[dense]   = AuxStride{};
        world.PublishTransformSlot(dense);

        if (world.bodies.softShadows) {
            world.bodies.softShadows[dense].vertices.reset();
        }

        world.bodies.slotStates[slot].store(std::to_underlying(SlotState::Character),
                                             std::memory_order::release);
        world.count.fetch_add(1, std::memory_order::release);

        world.physicsSystem.GetBodyInterface().SetUserData(inner, handle.Raw());
        character->virtual_->SetUserData(handle.Raw());
        character->CacheState();
    }

    // --- 4. Install the listener ------------------------------------------
    character->listener_ = std::make_unique<Listener>(*character);
    character->virtual_->SetListener(character->listener_.get());

    return character;
}

Character::~Character() {
    if (world_ == nullptr) {
        return;
    }

    // 1. Give the slot back, with the same locks the old dealloc took.
    {
        std::unique_lock shadow = world_->LockShadowNotStepping();
        world_->WaitUntilNotQuerying(shadow);

        const auto slot = handle_.load(std::memory_order::relaxed).Index();
        world_->RemoveBodySlot(slot);
    }

    // 2. Destroy the Jolt character under the trampoline lock.
    const std::scoped_lock trampoline(world_->jphTrampolineLock);
    if (virtual_ != nullptr) {
        world_->characterVsCharacter.Remove(virtual_);
        // The listener holds a raw back-pointer to this Character, so it must
        // stop being called before it is destroyed.
        virtual_->SetListener(nullptr);
        virtual_->Release();
        virtual_ = nullptr;
    }
    listener_.reset();
}

void Character::CacheState() {
    CharacterState &snapshot = state_.Back();
    const JPH::RVec3 p       = virtual_->GetPosition();
    const JPH::Quat q        = virtual_->GetRotation();
    const JPH::Vec3 v        = virtual_->GetLinearVelocity();
    snapshot.px              = p.GetX();
    snapshot.py              = p.GetY();
    snapshot.pz              = p.GetZ();
    snapshot.qx              = q.GetX();
    snapshot.qy              = q.GetY();
    snapshot.qz              = q.GetZ();
    snapshot.qw              = q.GetW();
    snapshot.vx              = v.GetX();
    snapshot.vy              = v.GetY();
    snapshot.vz              = v.GetZ();
    snapshot.grounded = virtual_->GetGroundState() == JPH::CharacterBase::EGroundState::OnGround;
    state_.Publish();
}

void Character::PublishTransformLocked() const {
    const std::uint32_t dense = DenseIndexLocked();
    const JPH::RVec3 p = virtual_->GetPosition();
    const JPH::Quat  q = virtual_->GetRotation();
    world_->bodies.positions[dense]      = PosStride{p.GetX(), p.GetY(), p.GetZ(), 0.0};
    world_->bodies.rotations[dense]      = AuxStride{q.GetX(), q.GetY(), q.GetZ(), q.GetW()};
    world_->PublishTransformSlot(dense);
}

std::uint32_t Character::DenseIndexLocked() const noexcept {
    const std::uint32_t slot = handle_.load(std::memory_order::relaxed).Index();
    return world_->bodies.slotToDense[slot];
}

bool Character::LayersAllow(BodyHandle self, BodyHandle other) const noexcept {
    PhysicsWorld &world = *world_;

    const std::optional<std::uint32_t> resolved_self  = world.Resolve(self);
    const std::optional<std::uint32_t> resolved_other = world.Resolve(other);
    if (!resolved_self || !resolved_other) {
        return true;
    }
    const std::uint32_t slot_self  = *resolved_self;
    const std::uint32_t slot_other = *resolved_other;
    if (slot_self >= world.slotCapacity || slot_other >= world.slotCapacity) {
        return true;
    }

    const std::uint8_t state_self =
        world.bodies.slotStates[slot_self].load(std::memory_order::relaxed);
    const std::uint8_t state_other =
        world.bodies.slotStates[slot_other].load(std::memory_order::relaxed);
    if (state_self == std::to_underlying(SlotState::Empty) ||
        state_other == std::to_underlying(SlotState::Empty)) {
        return true;
    }

    const std::uint32_t dense_self  = world.bodies.slotToDense[slot_self];
    const std::uint32_t dense_other = world.bodies.slotToDense[slot_other];
    const std::size_t active = world.count.load(std::memory_order::relaxed);
    if (dense_self >= active || dense_other >= active) {
        return true;
    }

    const std::uint32_t cat_self   = world.bodies.categories[dense_self];
    const std::uint32_t mask_self  = world.bodies.masks[dense_self];
    const std::uint32_t cat_other  = world.bodies.categories[dense_other];
    const std::uint32_t mask_other = world.bodies.masks[dense_other];

    // Reject if either mask blocks the other's category.
    return (cat_self & mask_other) != 0 && (cat_other & mask_self) != 0;
}

void Character::SetStrength(float strength) {
    std::unique_lock shadow = world_->LockShadowNotStepping();
    world_->WaitUntilNotQuerying(shadow);

    pushStrength_.store(strength, std::memory_order::relaxed);
    virtual_->SetMaxStrength(strength);
}

void Character::Move(float vx, float vy, float vz, float dt) {
    // 1. Snapshot for interpolation, and publish the velocity the contact
    //    listener will scale its impulses by.
    {
        std::unique_lock shadow = world_->LockShadowNotStepping();

        lastVx_.store(vx, std::memory_order::relaxed);
        lastVy_.store(vy, std::memory_order::relaxed);
        lastVz_.store(vz, std::memory_order::relaxed);

        const std::uint32_t dense = DenseIndexLocked();
        world_->bodies.prevPositions[dense] = world_->bodies.positions[dense];
        world_->bodies.prevRotations[dense] = world_->bodies.rotations[dense];
    }

    // 2. Inherit the ground platform's horizontal velocity.
    JPH::Vec3 velocity(vx, vy, vz);
    if (!virtual_->GetGroundBodyID().IsInvalid()) {
        const JPH::Vec3 ground_velocity = virtual_->GetGroundVelocity();
        velocity.SetX(velocity.GetX() + ground_velocity.GetX());
        velocity.SetZ(velocity.GetZ() + ground_velocity.GetZ());
    }
    // The y component is deliberately not inherited: the old binding left it
    // commented out, and matching that keeps recorded replays stable.
    virtual_->SetLinearVelocity(velocity);

    const JPH::CharacterVirtual::ExtendedUpdateSettings update_settings{
        .mStickToFloorStepDown             = JPH::Vec3(0.0f, -0.5f, 0.0f),
        .mWalkStairsStepUp                 = JPH::Vec3(0.0f, 0.4f, 0.0f),
        .mWalkStairsMinStepForward         = 0.02f,
        .mWalkStairsStepForwardTest        = 0.15f,
        .mWalkStairsCosAngleForwardContact = 0.996f,
        .mWalkStairsStepDownExtra          = JPH::Vec3::sZero()};

    constexpr auto character_layer =
        static_cast<JPH::ObjectLayer>(std::to_underlying(ObjectLayer::Dynamic));
    // Default-constructed filters accept everything, which is what the heap
    // filters this replaces were built with.
    const JPH::BodyFilter  body_filter;
    const JPH::ShapeFilter shape_filter;

    {
        const std::scoped_lock trampoline(world_->jphTrampolineLock);
        virtual_->ExtendedUpdate(dt, world_->physicsSystem.GetGravity(), update_settings,
                                 world_->physicsSystem.GetDefaultBroadPhaseLayerFilter(character_layer),
                                 world_->physicsSystem.GetDefaultLayerFilter(character_layer),
                                 body_filter, shape_filter, *world_->tempAllocator.get());
    }

    // 3. Republish.
    const std::scoped_lock shadow(world_->shadowLock);
    PublishTransformLocked();
    CacheState();
}

JPH::RVec3 Character::GetPosition() const {
    // Zero-lock read of the character's own published state (Step 4).
    const CharacterState &state = state_.Front();
    return JPH::RVec3(state.px, state.py, state.pz);
}

JPH::Quat Character::GetRotation() const {
    const CharacterState &state = state_.Front();
    return JPH::Quat(state.qx, state.qy, state.qz, state.qw);
}

JPH::Vec3 Character::GetLinearVelocity() const {
    const CharacterState &state = state_.Front();
    return JPH::Vec3(state.vx, state.vy, state.vz);
}

void Character::SetPosition(const JPH::RVec3 &position) {
    std::unique_lock shadow = world_->LockShadowNotStepping();

    virtual_->SetPosition(position);

    // TODO(smell): the snap this represents is invisible to the interpolator
    // because prev and current are set to the same value. Splitting them would
    // change the rendered motion, so it is left as it was.
    const std::uint32_t dense = DenseIndexLocked();
    const PosStride p_val{position.GetX(), position.GetY(), position.GetZ(), 0.0};
    world_->bodies.positions[dense]      = p_val;
    world_->bodies.prevPositions[dense] = p_val;
    world_->PublishTransformSlot(dense);
    CacheState();
}

void Character::SetRotation(const JPH::Quat &rotation) {
    std::unique_lock shadow = world_->LockShadowNotStepping();

    // CharacterVirtual::SetRotation only stores the value and forwards it to the
    // inner body through SetPositionAndRotation, which is exactly where Jolt
    // requires a unit quaternion (see RotationSanitize.hpp). Normalize once and
    // use the same value for the shadow buffers.
    const JPH::Quat sanitized = SanitizeRotation(rotation);
    virtual_->SetRotation(sanitized);

    // AuxStride is already four floats, so the dense index indexes it directly.
    // The pre-migration binding carried a stale `dense * 4` here from when this
    // storage was a flat float array; with Buffer<AuxStride> that wrote 16
    // bytes at four times the intended offset, corrupting a neighbouring
    // body's rotation for any character past the first.
    const std::uint32_t dense = DenseIndexLocked();
    const AuxStride r_val{sanitized.GetX(), sanitized.GetY(), sanitized.GetZ(), sanitized.GetW()};
    world_->bodies.rotations[dense]      = r_val;
    world_->bodies.prevRotations[dense] = r_val;
    world_->PublishTransformSlot(dense);
    CacheState();
}

bool Character::IsGrounded() const {
    // Zero-lock read (smoking gun #2): querying grounded state no longer
    // contends with a step for the world's global shadowLock.
    return state_.Front().grounded;
}

} // namespace CPH
