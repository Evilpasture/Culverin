#include "characters/Character.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "core/Compiler.hpp"
#include "core/Types.hpp"
#include "math/MathCore.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/PhysicsSync.hpp"
#include "physics/WorldAdapters.hpp"
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyInterface.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>
#include <Jolt/Math/Math.h>
#include <math.h>
#include <cstring>

namespace CPH::Python {

// Character helpers

/// Records a character-vs-body contact into the world's contact ring buffer.
/// Runs on Jolt worker threads: no allocation, no locking.
static void RecordCharacterContact(CharacterObject *self, JPH::BodyID body_id2, JPH::RVec3Arg pos,
                                   JPH::Vec3Arg norm, ContactEventType type) {
    auto *world          = self->world;
    const uint32_t j_idx = body_id2.GetIndex();
    BodyHandle h2        = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        // TSan Fix: Atomic load from the shared handle map.
        // Acquire ensures we see the body initialization if it was just created.
        h2 = (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
    }

    // Extract raw value for liveness check
    const uint64_t h2_raw = h2;
    if (h2_raw == 0) {
        return;
    }

    // TSan Fix: Atomic load of the character's own handle
    const uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

    const size_t idx =
        (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);

    if (idx < world->physics.contact_max_capacity) {
        // ContactEvent_ Refactor
        ContactEvent &event     = world->physics.contact_buffer[idx];
        std::memset(&event, 0, sizeof(event)); // buffer is allocate_for_overwrite
        ContactEventSlim *slim  = &event.slim;
        ContactEventFatExt *fat = &event.fat;

        slim->flags = static_cast<uint32_t>(type);

        // TSan Fix: Use raw values for canonical handle ordering
        if (h1_raw < h2_raw) {
            (slim->body1).store(h1_raw, std::memory_order::relaxed);
            (slim->body2).store(h2_raw, std::memory_order::relaxed);
        } else {
            (slim->body1).store(h2_raw, std::memory_order::relaxed);
            (slim->body2).store(h1_raw, std::memory_order::relaxed);
        }

        // Geometry
        slim->px            = static_cast<JPH_Real>(pos.GetX());
        slim->py            = static_cast<JPH_Real>(pos.GetY());
        slim->pz            = static_cast<JPH_Real>(pos.GetZ());
        slim->nx            = norm.GetX();
        slim->ny            = norm.GetY();
        slim->nz            = norm.GetZ();
        slim->impulse       = 1.0f;
        slim->sliding_speed = 0.0f;

        // Metadata lookups
        auto slot2      = static_cast<uint32_t>(h2_raw & kHandleIndexMask);
        uint32_t dense2 = world->physics.bodies.slot_to_dense[slot2];
        fat->mat1       = 0;
        fat->mat2       = world->physics.bodies.material_ids[dense2];

        // Release fence ensures all previous stores are visible to Python when it reads
        // contact_atomic_idx
        std::atomic_thread_fence(std::memory_order::release);
    }
}

/// Records a character-vs-character contact. The peer's handle comes straight
/// off its user data, which is authoritative and needs no map lookup.
static void ReportCharVsChar(CharacterObject *self, const JPH::CharacterVirtual *other,
                             JPH::Vec3Arg normal, JPH::RVec3Arg pos, ContactEventType type) {
    auto *world             = self->world;
    const uint64_t h1_raw   = (self->handle).load(std::memory_order::relaxed);

    // Retrieve the handle directly from the other character's UserData.
    // This is 100% reliable and avoids map boundary checks.
    const uint64_t h2_raw = other->GetUserData();

    if (h2_raw == 0) {
        return; // Still 0? Probably a Jolt internal body.
    }

    const size_t idx =
        (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);
    if (idx < world->physics.contact_max_capacity) {
        // ContactEvent_ Refactor
        ContactEvent &event     = world->physics.contact_buffer[idx];
        std::memset(&event, 0, sizeof(event)); // buffer is allocate_for_overwrite
        ContactEventSlim *slim  = &event.slim;
        ContactEventFatExt *fat = &event.fat;

        slim->flags = static_cast<uint32_t>(type);

        // TSan Fix: Canonicalize ordering
        if (h1_raw < h2_raw) {
            (slim->body1).store(h1_raw, std::memory_order::relaxed);
            (slim->body2).store(h2_raw, std::memory_order::relaxed);
        } else {
            (slim->body1).store(h2_raw, std::memory_order::relaxed);
            (slim->body2).store(h1_raw, std::memory_order::relaxed);
        }

        slim->sliding_speed = 0.0f;
        slim->nx            = normal.GetX();
        slim->ny            = normal.GetY();
        slim->nz            = normal.GetZ();
        slim->px            = static_cast<JPH_Real>(pos.GetX());
        slim->py            = static_cast<JPH_Real>(pos.GetY());
        slim->pz            = static_cast<JPH_Real>(pos.GetZ());
        slim->impulse       = 1.0f;
        fat->mat1           = 0;
        fat->mat2           = 0;

        // Release visibility of the whole event to Python
        std::atomic_thread_fence(std::memory_order::release);
    }
}

/// Pushes a dynamic, non-sensor body away from the character along the contact
/// normal, scaled by how fast the character is moving into it.
static void ApplyCharacterImpulse(CharacterObject *self, JPH::BodyID body_id2,
                                  JPH::Vec3Arg contact_normal) {
    // 1. Thread-Safe Member Access
    const float vx       = self->last_vx.load(std::memory_order::relaxed);
    const float vy       = self->last_vy.load(std::memory_order::relaxed);
    const float vz       = self->last_vz.load(std::memory_order::relaxed);
    const float strength = self->push_strength.load(std::memory_order::relaxed);

    JPH::BodyInterface &bi = self->world->physics.physics_system.GetBodyInterface();

    // 2. Ignore Sensors & Non-Dynamic Bodies
    if (bi.IsSensor(body_id2) || bi.GetMotionType(body_id2) != JPH::EMotionType::Dynamic) {
        return;
    }

    // 3. Calculate Pushing Force
    const float dot =
        vx * contact_normal.GetX() + vy * contact_normal.GetY() + vz * contact_normal.GetZ();

    // Normal points TOWARDS character, so dot is negative when colliding
    if (dot < -0.01f) {
        float factor                = -dot * strength; // Negate to get positive push force
        constexpr float max_impulse = 50000.0f;
        if (factor > max_impulse) {
            factor = max_impulse;
        }

        // Flip the normal to push the OBJECT away from the character; never push down.
        const float y_push = -contact_normal.GetY() * factor;
        const JPH::Vec3 impulse(-contact_normal.GetX() * factor,
                                (y_push > 0.0f) ? y_push : 0.0f,
                                -contact_normal.GetZ() * factor);

        bi.AddImpulse(body_id2, impulse);
        bi.ActivateBody(body_id2);
    }
}

/// Bridges Jolt's CharacterContactListener to a CharacterObject.
///
/// This replaces the JoltC proc table: previously every character shared one
/// global `s_Procs` pointer and recovered its owner from a `void *userData`,
/// which is why listener creation had to be serialized behind the trampoline
/// lock. Each character now owns a listener, so there is no shared state left
/// to serialize.
class CharacterListener final : public JPH::CharacterContactListener {
  public:
    explicit CharacterListener(CharacterObject *owner) : owner_(owner) {}

    // Callback: inherit the velocity of whatever the character stands on,
    // scaled by that surface's friction.
    void OnAdjustBodyVelocity(const JPH::CharacterVirtual *character, const JPH::Body &body2,
                              JPH::Vec3 &io_linear_velocity,
                              JPH::Vec3 &io_angular_velocity) override {
        CharacterObject *self = owner_;
        if (!self || !self->world) {
            return;
        }

        // 1. Get Platform/Friction Data
        const uint64_t h2_raw = body2.GetUserData();
        float friction        = 1.0f;
        auto slot2            = static_cast<uint32_t>(h2_raw & kHandleIndexMask);
        if (slot2 < self->world->physics.slot_capacity) {
            uint32_t dense2 = self->world->physics.bodies.slot_to_dense[slot2];
            uint32_t mat_id = self->world->physics.bodies.material_ids[dense2];
            for (size_t i = 0; i < self->world->physics.material_count; i++) {
                if (self->world->physics.materials[i].id == mat_id) {
                    friction = self->world->physics.materials[i].friction;
                    break;
                }
            }
        }

        // 2. Calculate Tangential Velocity (v = omega x r)
        const JPH::Vec3 omega    = body2.GetAngularVelocity();
        const JPH::RVec3 char_pos = character->GetPosition();
        const JPH::RVec3 plat_pos = body2.GetPosition();

        // Relative offset
        const float rx = static_cast<float>(char_pos.GetX() - plat_pos.GetX());
        const float rz = static_cast<float>(char_pos.GetZ() - plat_pos.GetZ());

        // v = omega x r
        const float target_vt_x = omega.GetY() * rz;
        const float target_vt_z = -omega.GetY() * rx;

        // 3. APPLY: Override inheritance with friction scaling.
        // We use a factor of 1.0 for friction >= 0.2. Only the components the
        // character model cares about are written; Jolt fills in the rest.
        const float factor = (friction > 0.2f) ? 1.0f : (friction / 0.2f);

        io_linear_velocity.SetX(target_vt_x * factor);
        io_linear_velocity.SetZ(target_vt_z * factor);

        // Set the inherited angular velocity (rotation)
        io_angular_velocity.SetY(omega.GetY() * factor);
    }

    // Callback: Can the character collide with this object?
    bool OnContactValidate([[maybe_unused]] const JPH::CharacterVirtual *character,
                           const JPH::CharacterContact &contact) override {
        CharacterObject *self = owner_;
        if (!self || !self->world) {
            return true;
        }
        PhysicsWorldObject *world = self->world;

        // 1. Get Culverin Handles
        const uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

        // Direct thread-safe lookup of the other body's UserData from contact
        uint64_t h2_raw = contact.mUserData;
        if (h2_raw == 0 && world->physics.body_interface) {
            h2_raw = world->physics.physics_system.GetBodyInterface().GetUserData(contact.mBodyB);
        }
        // Fallback to safe lock-free id-to-handle map lookup
        if (h2_raw == 0 && world->physics.id_to_handle_map) {
            const uint32_t j_idx = contact.mBodyB.GetIndex();
            if (j_idx <= world->physics.max_jolt_bodies) {
                h2_raw = (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
            }
        }

        if (h2_raw == 0) {
            return true; // Collide by default if handle is missing (e.g. static environment)
        }

        // 2. Safely Resolve Slots with Generation Checking
        const std::optional<std::uint32_t> resolved1 =
            world->physics.UnpackHandle(static_cast<BodyHandle>(h1_raw));
        const std::optional<std::uint32_t> resolved2 =
            world->physics.UnpackHandle(static_cast<BodyHandle>(h2_raw));
        if (!resolved1 || !resolved2) {
            return true; // Collide on handle verification mismatches
        }
        const uint32_t slot1 = *resolved1;
        const uint32_t slot2 = *resolved2;

        // Safety checks for array bounds
        if (slot1 >= world->physics.slot_capacity || slot2 >= world->physics.slot_capacity) {
            return true;
        }

        // Check states to ensure both slots are in use
        uint8_t state1 = (world->physics.bodies.slot_states[slot1]).load(std::memory_order::relaxed);
        uint8_t state2 = (world->physics.bodies.slot_states[slot2]).load(std::memory_order::relaxed);
        if (state1 == std::to_underlying(SlotState::Empty) ||
            state2 == std::to_underlying(SlotState::Empty)) {
            return true;
        }

        const uint32_t idx1 = world->physics.bodies.slot_to_dense[slot1];
        const uint32_t idx2 = world->physics.bodies.slot_to_dense[slot2];

        const size_t active_count = (world->physics.count).load(std::memory_order::relaxed);
        if (idx1 >= active_count || idx2 >= active_count) {
            return true;
        }

        // 3. Perform Bitmask Filtering
        const uint32_t cat1  = world->physics.bodies.categories[idx1];
        const uint32_t mask1 = world->physics.bodies.masks[idx1];
        const uint32_t cat2  = world->physics.bodies.categories[idx2];
        const uint32_t mask2 = world->physics.bodies.masks[idx2];

        // Reject if either mask blocks the other's category
        return (cat1 & mask2) != 0 && (cat2 & mask1) != 0;
    }

    bool OnCharacterContactValidate([[maybe_unused]] const JPH::CharacterVirtual *character,
                                    const JPH::CharacterContact &contact) override {
        CharacterObject *self = owner_;
        if (!self || !self->world) {
            return true;
        }
        PhysicsWorldObject *world = self->world;

        // 1. Get Culverin Handles
        const uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

        // In CharacterContact the other character's handle is stored in mUserData,
        // or falls back to the peer character's own user data.
        uint64_t h2_raw = contact.mUserData;
        if (h2_raw == 0 && contact.mCharacterB) {
            h2_raw = contact.mCharacterB->GetUserData();
        }

        if (h2_raw == 0) {
            return true; // Collide by default if handle is missing
        }

        // 2. Resolve Slots
        const std::optional<std::uint32_t> resolved1 =
            world->physics.UnpackHandle(static_cast<BodyHandle>(h1_raw));
        const std::optional<std::uint32_t> resolved2 =
            world->physics.UnpackHandle(static_cast<BodyHandle>(h2_raw));
        if (!resolved1 || !resolved2) {
            return true;
        }
        const uint32_t slot1 = *resolved1;
        const uint32_t slot2 = *resolved2;

        if (slot1 >= world->physics.slot_capacity || slot2 >= world->physics.slot_capacity) {
            return true;
        }

        const uint32_t idx1 = world->physics.bodies.slot_to_dense[slot1];
        const uint32_t idx2 = world->physics.bodies.slot_to_dense[slot2];

        const size_t active_count = (world->physics.count).load(std::memory_order::relaxed);
        if (idx1 >= active_count || idx2 >= active_count) {
            return true;
        }

        // 3. Perform Bitmask Filtering
        const uint32_t cat1  = world->physics.bodies.categories[idx1];
        const uint32_t mask1 = world->physics.bodies.masks[idx1];
        const uint32_t cat2  = world->physics.bodies.categories[idx2];
        const uint32_t mask2 = world->physics.bodies.masks[idx2];

        // Reject if either mask blocks the other's category
        return ((cat1 & mask2) && (cat2 & mask1)) != 0;
    }

    void OnContactAdded([[maybe_unused]] const JPH::CharacterVirtual *character,
                        const JPH::CharacterContact &contact,
                        JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;

        CharacterObject *self = owner_;
        if (!self) {
            return;
        }

        // Record Event
        RecordCharacterContact(self, contact.mBodyB, contact.mPosition, contact.mContactNormal,
                               ContactEventType::Added);

        // Apply Impulse
        ApplyCharacterImpulse(self, contact.mBodyB, contact.mContactNormal);
    }

    void OnContactPersisted([[maybe_unused]] const JPH::CharacterVirtual *character,
                            const JPH::CharacterContact &contact,
                            JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;

        CharacterObject *self = owner_;
        if (!self) {
            return;
        }

        // Record Event
        RecordCharacterContact(self, contact.mBodyB, contact.mPosition, contact.mContactNormal,
                               ContactEventType::Persisted);

        // Apply Impulse
        ApplyCharacterImpulse(self, contact.mBodyB, contact.mContactNormal);
    }

    void OnContactRemoved([[maybe_unused]] const JPH::CharacterVirtual *character,
                          const JPH::BodyID &body_id2,
                          [[maybe_unused]] const JPH::SubShapeID &sub_shape_id2) override {
        CharacterObject *self = owner_;
        if (!self || !self->world) {
            return;
        }

        PhysicsWorldObject *world   = self->world;
        const uint32_t j_idx        = body_id2.GetIndex();

        // TSan Fix: Explicitly load raw values to avoid seq_cst overhead
        const uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);
        uint64_t h2_raw       = 0;

        if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
            h2_raw = (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
        }

        if (h2_raw == 0) {
            return;
        }

        const size_t idx =
            (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);
        if (idx < world->physics.contact_max_capacity) {
            // Buffer is allocate_for_overwrite: define the whole slot
            // (incl. fat fields) before filling slim.
            std::memset(&world->physics.contact_buffer[idx], 0, sizeof(ContactEvent));
            ContactEventSlim *slim = &world->physics.contact_buffer[idx].slim;

            slim->flags = std::to_underlying(ContactEventType::Removed);

            // TSan Fix: Canonicalize using raw registers
            if (h1_raw < h2_raw) {
                (slim->body1).store(h1_raw, std::memory_order::relaxed);
                (slim->body2).store(h2_raw, std::memory_order::relaxed);
            } else {
                (slim->body1).store(h2_raw, std::memory_order::relaxed);
                (slim->body2).store(h1_raw, std::memory_order::relaxed);
            }

            // Explicit zeroing (Safer than memset due to strict aliasing/layout padding)
            slim->px            = 0.0;
            slim->py            = 0.0;
            slim->pz            = 0.0;
            slim->nx            = 0.0f;
            slim->ny            = 0.0f;
            slim->nz            = 0.0f;
            slim->impulse       = 0.0f;
            slim->sliding_speed = 0.0f;

            // Finalize visibility for Python readers
            std::atomic_thread_fence(std::memory_order::release);
        }
    }

    void OnCharacterContactAdded([[maybe_unused]] const JPH::CharacterVirtual *character,
                                 const JPH::CharacterContact &contact,
                                 JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;

        CharacterObject *self = owner_;
        if (!self) {
            return;
        }

        ReportCharVsChar(self, contact.mCharacterB, contact.mContactNormal, contact.mPosition,
                         ContactEventType::Added);
    }

    void OnCharacterContactPersisted([[maybe_unused]] const JPH::CharacterVirtual *character,
                                     const JPH::CharacterContact &contact,
                                     JPH::CharacterContactSettings &io_settings) override {
        io_settings.mCanPushCharacter   = true;
        io_settings.mCanReceiveImpulses = true;

        CharacterObject *self = owner_;
        if (!self) {
            return;
        }

        ReportCharVsChar(self, contact.mCharacterB, contact.mContactNormal, contact.mPosition,
                         ContactEventType::Persisted);
    }

    void OnCharacterContactRemoved([[maybe_unused]] const JPH::CharacterVirtual *character,
                                   const JPH::CharacterID &other_character_id,
                                   [[maybe_unused]] const JPH::SubShapeID &sub_shape_id2) override {
        CharacterObject *self = owner_;
        auto *world           = self->world;

        // TSan Fix: Explicit relaxed load of own handle
        const uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

        // We have to use the CharacterID to find the handle.
        const uint32_t j_idx = (other_character_id.GetValue() & JPH::BodyID::cMaxBodyIndex);
        uint64_t h2_raw      = 0;

        if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
            h2_raw = (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
        }

        if (h2_raw == 0) {
            return;
        }

        const size_t idx =
            (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);
        if (idx < world->physics.contact_max_capacity) {
            // Buffer is allocate_for_overwrite: define the whole slot
            // (incl. fat fields) before filling slim.
            std::memset(&world->physics.contact_buffer[idx], 0, sizeof(ContactEvent));
            ContactEventSlim *slim = &world->physics.contact_buffer[idx].slim;

            slim->flags = std::to_underlying(ContactEventType::Removed);

            // TSan Fix: Canonicalize using standard integer registers
            if (h1_raw < h2_raw) {
                (slim->body1).store(h1_raw, std::memory_order::relaxed);
                (slim->body2).store(h2_raw, std::memory_order::relaxed);
            } else {
                (slim->body1).store(h2_raw, std::memory_order::relaxed);
                (slim->body2).store(h1_raw, std::memory_order::relaxed);
            }

            // Explicit zeroing (Safer than memset)
            slim->px            = 0.0;
            slim->py            = 0.0;
            slim->pz            = 0.0;
            slim->nx            = 0.0f;
            slim->ny            = 0.0f;
            slim->nz            = 0.0f;
            slim->impulse       = 0.0f;
            slim->sliding_speed = 0.0f;

            // Finalize event publication
            std::atomic_thread_fence(std::memory_order::release);
        }
    }

    // OnContactSolve / OnCharacterContactSolve are deliberately not overridden:
    // the proc table wired them to nullptr, and the base implementations are
    // no-ops.

  private:
    CharacterObject *owner_;
};

[[nodiscard]] PyObject * CharacterMove(CharacterObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {
    // 1. INTEGRATED FAST PARSE (Unchanged)
    Vec3f v_in = {};
    float dt       = 0.0f;

    auto nargs = PyVectorcall_NARGS(nargsf);
    const ParseResult parsed =
        self->parsers->CharMove.Parse(args, nargs, kwnames, v_in, dt);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (!ValidateFinite("Character velocity", v_in.x, v_in.y, v_in.z)) {
        return nullptr;
    }

    // 2. SNAPSHOT (Shadow Lock)
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    // TSan Fix: Explicit relaxed stores for worker-thread inputs
    (self->last_vx).store(v_in.x, std::memory_order::relaxed);
    (self->last_vy).store(v_in.y, std::memory_order::relaxed);
    (self->last_vz).store(v_in.z, std::memory_order::relaxed);

    // TSan Fix: Load handle atomically to resolve indices
    uint64_t h_raw = (self->handle).load(std::memory_order::relaxed);
    uint32_t slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t dense = self->world->physics.bodies.slot_to_dense[slot];

    // Snapshot current state to prev for interpolation
    auto shadow_pos  = self->world->physics.bodies.positions.data();
    auto shadow_ppos = self->world->physics.bodies.prev_positions.data();
    auto shadow_rot  = self->world->physics.bodies.rotations.data();
    auto shadow_prot = self->world->physics.bodies.prev_rotations.data();

    shadow_ppos[dense] = shadow_pos[dense];
    shadow_prot[dense] = shadow_rot[dense];

    (self->world->physics.shadow_lock).unlock();

    // 3. JOLT EXECUTION
    JPH::Vec3 j_v(v_in.x, v_in.y, v_in.z);

    // Inherit velocity from ground platform if grounded
    if (!self->character->GetGroundBodyID().IsInvalid()) {
        const JPH::Vec3 ground_vel = self->character->GetGroundVelocity();
        j_v.SetX(j_v.GetX() + ground_vel.GetX());
        // j_v.SetY(j_v.GetY() + ground_vel.GetY());
        j_v.SetZ(j_v.GetZ() + ground_vel.GetZ());
    }

    self->character->SetLinearVelocity(j_v);

    const JPH::CharacterVirtual::ExtendedUpdateSettings update_settings{
        .mStickToFloorStepDown             = JPH::Vec3(0.0f, -0.5f, 0.0f),
        .mWalkStairsStepUp                 = JPH::Vec3(0.0f, 0.4f, 0.0f),
        .mWalkStairsMinStepForward         = 0.02f,
        .mWalkStairsStepForwardTest        = 0.15f,
        .mWalkStairsCosAngleForwardContact = 0.996f,
        .mWalkStairsStepDownExtra          = JPH::Vec3::sZero()};

    auto *native_character = self->character;
    auto &physics_system   = self->world->physics.physics_system;
    constexpr auto character_layer =
        static_cast<JPH::ObjectLayer>(std::to_underlying(ObjectLayer::Dynamic));
    // Default-constructed filters accept everything, which is what the heap
    // filters this replaces were built with: JoltC's ManagedBodyFilter falls
    // through to `return true` when no procs are installed.
    const JPH::BodyFilter body_filter;
    const JPH::ShapeFilter shape_filter;
    auto *temp_allocator = self->world->physics.temp_allocator.get();

    (self->world->physics.jph_trampoline_lock).lock();
    Py_BEGIN_ALLOW_THREADS native_character->ExtendedUpdate(
        dt, physics_system.GetGravity(), update_settings,
        physics_system.GetDefaultBroadPhaseLayerFilter(character_layer),
        physics_system.GetDefaultLayerFilter(character_layer), body_filter, shape_filter,
        *temp_allocator);
    Py_END_ALLOW_THREADS (self->world->physics.jph_trampoline_lock).unlock();

    // 4. POST-MOVE SYNC
    (self->world->physics.shadow_lock).lock();

    const JPH::RVec3 current_p = self->character->GetPosition();
    const JPH::Quat current_r  = self->character->GetRotation();

    // Sync non-atomic shadow buffers
    shadow_pos[dense] = PosStride{current_p.GetX(), current_p.GetY(), current_p.GetZ(), 0.0};
    shadow_rot[dense] =
        AuxStride{current_r.GetX(), current_r.GetY(), current_r.GetZ(), current_r.GetW()};

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * CharacterGetPosition(CharacterObject *self,
                                                           PyObject *ignored [[maybe_unused]]) {
    // Lock for consistency (ensure we aren't reading mid-step)
    (self->world->physics.shadow_lock).lock();
    const JPH::RVec3 pos = self->character->GetPosition();
    (self->world->physics.shadow_lock).unlock();

    PyObject *ret = PyTuple_New(3);
    if (!ret) {
        return nullptr;
    }

    // Use the double precision provided by RVec3
    PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble(pos.GetX()));
    PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble(pos.GetY()));
    PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(pos.GetZ()));

    return ret;
}

[[nodiscard]] PyObject *
CharacterGetLinearVelocity(CharacterObject *self, PyObject *ignored [[maybe_unused]]) {
    (self->world->physics.shadow_lock).lock();
    const JPH::Vec3 vel = self->character->GetLinearVelocity();
    (self->world->physics.shadow_lock).unlock();

    return BuildTuple(vel.GetX(), vel.GetY(), vel.GetZ());
}

[[nodiscard]] PyObject * CharacterSetPosition(CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PosStride pos      = {};

    const ParseResult parsed =
        self->parsers->SetPosChar.Parse(args, nargs, kwnames, pos);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    self->character->SetPosition(JPH::RVec3(pos.x, pos.y, pos.z));

    // Update Shadow Buffers
    uint64_t h_raw = (self->handle).load(std::memory_order::relaxed);
    uint32_t slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t dense = self->world->physics.bodies.slot_to_dense[slot];

    auto shadow_pos  = self->world->physics.bodies.positions.data();
    auto shadow_ppos = self->world->physics.bodies.prev_positions.data();

    PosStride p_val = {pos.x, pos.y, pos.z, 0.0};
    shadow_pos[dense]   = p_val;
    shadow_ppos[dense]  = p_val;

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * CharacterSetRotation(CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    AuxStride rot      = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    const ParseResult parsed =
        self->parsers->SetRotChar.Parse(args, nargs, kwnames, rot);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    const JPH::Quat q(rot.x, rot.y, rot.z, rot.w);
    self->character->SetRotation(q);

    uint64_t raw_h = (self->handle).load(std::memory_order::relaxed);

    auto slot          = static_cast<uint32_t>(raw_h & kHandleIndexMask);
    uint32_t dense_idx = self->world->physics.bodies.slot_to_dense[slot];
    size_t off         = static_cast<size_t>(dense_idx) * 4;

    memcpy(&self->world->physics.bodies.rotations[off], &q, 16);
    memcpy(&self->world->physics.bodies.prev_rotations[off], &q, 16);

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * CharacterSetStrength(CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float strength         = 0.0f;

    const ParseResult parsed =
        self->parsers->SetStrengthChar.Parse(args, nargs, kwnames, strength);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);
    BlockUntilNotQuerying(self->world);

    (self->push_strength).store(strength, std::memory_order::relaxed);
    self->character->SetMaxStrength(strength);

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * CharacterGetRenderTransform(CharacterObject *self,
                                                                  PyObject *arg) {
    const auto alpha_raw = PyFloat_AsDouble(arg);
    if (alpha_raw == -1.0 && PyErr_Occurred()) {
        return nullptr;
    }

    constexpr float min_v = 0.0f;
    constexpr float max_v = 1.0f;
    const auto alpha      = static_cast<float>(fmax(min_v, fmin(max_v, alpha_raw)));

    alignas(16) float res_p[3];
    alignas(16) float res_r[4];

    (self->world->physics.shadow_lock).lock();

    BlockUntilNotStepping(self->world);

    const auto h_raw = (self->handle).load(std::memory_order::relaxed);
    const auto slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    const auto dense = self->world->physics.bodies.slot_to_dense[slot];

    const auto shadow_ppos = self->world->physics.bodies.prev_positions.data();
    const auto shadow_prot = self->world->physics.bodies.prev_rotations.data();

    const JPH::RVec3 end_p = self->character->GetPosition();
    const JPH::Quat end_r  = self->character->GetRotation();

    MathInterpolateCharacterTransform(&shadow_ppos[dense], &shadow_prot[dense], end_p, end_r,
                                      alpha, res_p, res_r);

    (self->world->physics.shadow_lock).unlock();

    return BuildTuple(BuildTuple(res_p[0], res_p[1], res_p[2]),
                              BuildTuple(res_r[0], res_r[1], res_r[2], res_r[3]));
}

[[nodiscard]] PyObject * CharacterIsGrounded(CharacterObject *self,
                                                          PyObject *ignored [[maybe_unused]]) {
    (self->world->physics.shadow_lock).lock();
    const JPH::CharacterBase::EGroundState state = self->character->GetGroundState();
    (self->world->physics.shadow_lock).unlock();

    if (state == JPH::CharacterBase::EGroundState::OnGround) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// NEW: GC Traverse/Clear for Character
[[nodiscard]] int CharacterTraverse(CharacterObject *self, visitproc visit,
                                                        void *arg) {
    Py_VISIT(self->world);
    return 0;
}
[[nodiscard]] int CharacterClear(CharacterObject *self) {
    Py_CLEAR(self->world);
    return 0;
}
void CharacterDealloc(CharacterObject *self) {
    PyObject_GC_UnTrack(self);

    uint64_t h_raw = 0;
    uint32_t slot  = 0;
    if (!self->world) {
        goto finalize;
    }

    // --- 1. WORLD DRAIN (Locked) ---
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);
    BlockUntilNotQuerying(self->world);

    h_raw = (self->handle).load(std::memory_order::relaxed);
    slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);

    self->world->physics.RemoveBodySlot(slot);

    (self->world->physics.shadow_lock).unlock();

    // --- 2. JOLT DESTRUCTION (Hard Serialized) ---
    (self->world->physics.jph_trampoline_lock).lock();

    if (self->character) {
        self->world->physics.character_vs_character.Remove(self->character);
        // The listener holds a raw back-pointer to this CharacterObject, so it
        // must go with the character rather than after it.
        self->character->SetListener(nullptr);
        self->character->Release();
        self->character = nullptr;
    }
    delete self->listener;
    self->listener = nullptr;

    (self->world->physics.jph_trampoline_lock).unlock();

finalize:
    if (self->parsers) {

        delete self->parsers;
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// Helper 1: Jolt-side allocation and Collision Manager linking
static inline JPH::CharacterVirtual *
alloc_j_char(PhysicsWorldObject *self, PositionVector pos, CharacterParams params) {

    const float half_h = fmaxf((params.height - 2.0f * params.radius) * 0.5f, 0.1f);
    const JPH::ShapeSettings::ShapeResult shape_result = JPH::CapsuleShapeSettings(half_h, params.radius).Create();
    if (!shape_result.IsValid()) {
        return nullptr;
    }

    JPH::CharacterVirtualSettings settings;
    settings.mShape         = shape_result.Get();
    settings.mMaxSlopeAngle = JPH::DegreesToRadians(params.max_slope);

    const JPH::RVec3 initial_position(pos.x, pos.y, pos.z);
    auto *native_character =
        new JPH::CharacterVirtual(&settings, initial_position, JPH::Quat::sIdentity(), 1,
                                  &self->physics.physics_system);

    self->physics.character_vs_character.Add(native_character);
    native_character->SetCharacterVsCharacterCollision(&self->physics.character_vs_character);
    return native_character;
}

// Helper 2: Shadow Buffer Registration (Atomic Commit)
static inline void register_char(PhysicsWorldObject *self, CharacterObject *obj,
                                 JPH::CharacterVirtual *j_char, uint32_t slot) {
    (self->physics.shadow_lock).lock();

    uint32_t gen = (self->physics.bodies.generations[slot]).load(std::memory_order::relaxed);
    BodyHandle h = make_handle(slot, gen);

    uint64_t raw_h = h;
    (obj->handle).store(raw_h, std::memory_order::relaxed);

    auto dense_idx =
        static_cast<uint32_t>((self->physics.count).load(std::memory_order::relaxed));
    const JPH::BodyID bid = j_char->GetInnerBodyID();
    uint32_t j_idx        = bid.GetIndex();

    if (j_idx <= self->physics.max_jolt_bodies) {
        (self->physics.id_to_handle_map[j_idx]).store(raw_h, std::memory_order::release);
    }

    self->physics.bodies.body_ids[dense_idx]      = bid;
    self->physics.bodies.slot_to_dense[slot]      = dense_idx;
    self->physics.bodies.dense_to_slot[dense_idx] = slot;
    self->physics.bodies.user_data[dense_idx]     = 0;

    constexpr auto kCollisionFilterAllCategories = 0xFFFF;
    constexpr auto kCollisionFilterAllMasks      = 0xFFFF;

    self->physics.bodies.categories[dense_idx]   = kCollisionFilterAllCategories;
    self->physics.bodies.masks[dense_idx]        = kCollisionFilterAllMasks;
    self->physics.bodies.material_ids[dense_idx] = 0;

    const JPH::RVec3 p = j_char->GetPosition();
    const JPH::Quat q  = j_char->GetRotation();

    PosStride p_val = {p.GetX(), p.GetY(), p.GetZ(), 0.0};
    AuxStride r_val = {q.GetX(), q.GetY(), q.GetZ(), q.GetW()};

    (self->physics.bodies.positions.data())[dense_idx]      = p_val;
    (self->physics.bodies.prev_positions.data())[dense_idx] = p_val;
    (self->physics.bodies.rotations.data())[dense_idx]      = r_val;
    (self->physics.bodies.prev_rotations.data())[dense_idx] = r_val;
    (self->physics.bodies.linear_velocities.data())[dense_idx] =
        AuxStride{};
    (self->physics.bodies.angular_velocities.data())[dense_idx] =
        AuxStride{};

    if (self->physics.bodies.soft_shadows) {
        self->physics.bodies.soft_shadows[dense_idx].vertices.reset();
    }

    (self->physics.bodies.slot_states[slot]).store(std::to_underlying(SlotState::Character), std::memory_order::release);
    (self->physics.count).fetch_add(1, std::memory_order::release);

    size_t final_count  = (self->physics.count).load(std::memory_order::relaxed);
    self->view_shape[0] = static_cast<Py_ssize_t>(final_count);

    self->physics.physics_system.GetBodyInterface().SetUserData(bid, raw_h);
    j_char->SetUserData(raw_h);

    (self->physics.shadow_lock).unlock();
}

// Helper 3: Install the contact listener.
//
// No trampoline lock: the JoltC version installed one process-wide procs table
// (JPH_CharacterContactListener_SetProcs) that every character shared, so
// creating a listener mutated global state. The listener is per-character now.
static inline void setup_char_listener(CharacterObject *obj) {
    obj->listener = new CharacterListener(obj);
    obj->character->SetListener(obj->listener);
}
[[nodiscard]]
static inline int setup_char_parsers(CharacterObject *obj) {
    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::CharacterParsers>();
    if (!obj->parsers) {
        return -1;
    }
    return 0;
}

// Main Orchestrator
[[nodiscard]] PyObject * WorldCreateCharacter(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs        = PyVectorcall_NARGS(nargsf);
    PosStride pos             = {};
    constexpr auto DEFAULT_HEIGHT = 1.8f;
    constexpr auto DEFAULT_RADIUS = 0.4f;
    constexpr auto DEFUALT_STEP_H = 0.4f;
    constexpr auto DEFAULT_SLOPE  = 45.0f;
    float height                  = DEFAULT_HEIGHT;
    float radius                  = DEFAULT_RADIUS;
    float step_h                  = DEFUALT_STEP_H;
    float slope                   = DEFAULT_SLOPE;

    const ParseResult parsed =
        self->parsers->CreateChar.Parse(args, nargs, kwnames, pos, height, radius, step_h,
                                         slope);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    (self->physics.shadow_lock).lock();
    BlockUntilNotStepping(self);
    BlockUntilNotQuerying(self);

    size_t available = (self->physics.free_count).load(std::memory_order::acquire);

    if (available == 0) {
        if (WorldResize(self, self->physics.capacity * 2) < 0) {
            (self->physics.shadow_lock).unlock();
            return nullptr;
        }
        available = (self->physics.free_count).load(std::memory_order::acquire);
    }

    uint32_t char_slot = self->physics.bodies.free_slots[available - 1];
    (self->physics.free_count).store(available - 1, std::memory_order::release);

    (self->physics.bodies.slot_states[char_slot]).store(std::to_underlying(SlotState::PendingCreate), std::memory_order::release);

    (self->physics.shadow_lock).unlock();

    PositionVector pos_vec      = {pos.x, pos.y, pos.z};
    CharacterParams char_params = {height, radius, slope};

    CharacterObject *obj = nullptr;
    JPH::CharacterVirtual *j_char = alloc_j_char(self, pos_vec, char_params);
    if (!j_char) {
        goto fail_jolt;
    }

    obj = PyObject_GC_New(CharacterObject,
                          reinterpret_cast<PyTypeObject *>(
                              GetModuleState(PyType_GetModule(Py_TYPE(self)))->CharacterType));
    if (!obj) {
        goto fail_py;
    }

    if (setup_char_parsers(obj) < 0) {
        goto fail_py;
    }

    obj->world     = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->character = j_char;
    // PyObject_GC_New does not run member initializers, and the dealloc frees
    // the listener unconditionally, so it must never be left indeterminate.
    obj->listener  = nullptr;
    obj->prev_px   = pos.x;
    obj->prev_py   = pos.y;
    obj->prev_pz   = pos.z;

    obj->prev_rx = 0.0f;
    obj->prev_ry = 0.0f;
    obj->prev_rz = 0.0f;
    obj->prev_rw = 1.0f;

    (obj->push_strength).store(0.0f, std::memory_order::relaxed);
    (obj->last_vx).store(0.0f, std::memory_order::relaxed);
    (obj->last_vy).store(0.0f, std::memory_order::relaxed);
    (obj->last_vz).store(0.0f, std::memory_order::relaxed);

    register_char(self, obj, j_char, char_slot);
    setup_char_listener(obj);

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

fail_py:
    j_char->Release();
fail_jolt:
    (self->physics.shadow_lock).lock();

    (self->physics.bodies.slot_states[char_slot]).store(std::to_underlying(SlotState::Empty), std::memory_order::relaxed);

    size_t f_idx = (self->physics.free_count).fetch_add(1, std::memory_order::relaxed);
    self->physics.bodies.free_slots[f_idx] = char_slot;

    (self->physics.shadow_lock).unlock();
    return nullptr;
}

[[nodiscard]] PyObject * CharacterGetHandle(CharacterObject *self,
                                            [[maybe_unused]] void *closure) {
    uint64_t raw_h = (self->handle).load(std::memory_order::relaxed);
    return BuildValue(raw_h);
}

static PyMethodDef kCharacterMethods[] = {

    CPH::Python::FastCallMethod("move", CharacterMove),
    CPH::Python::NoArgsMethod("get_position", CharacterGetPosition),
    CPH::Python::NoArgsMethod("get_linear_velocity", CharacterGetLinearVelocity),
    CPH::Python::FastCallMethod("set_position", CharacterSetPosition),
    CPH::Python::FastCallMethod("set_rotation", CharacterSetRotation),
    CPH::Python::NoArgsMethod("is_grounded", CharacterIsGrounded),
    CPH::Python::FastCallMethod("set_strength", CharacterSetStrength),
    CPH::Python::OneArgMethod("get_render_transform", CharacterGetRenderTransform),
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef CharacterGetSets[] = {
    CPH::Python::GetSet("handle", CharacterGetHandle),
    {},
};

static PyType_Slot CharacterSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(CharacterDealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(CharacterTraverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(CharacterClear)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kCharacterMethods)},
    {.slot = Py_tp_getset, .pfunc = reinterpret_cast<void *>(CharacterGetSets)},
    {},
};

PyType_Spec CharacterSpec = {
    .name      = "culverin._culverin_c.Character",
    .basicsize = sizeof(CharacterObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = CharacterSlots,
};

} // namespace CPH::Python
