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
#include <math.h>
#include <cstring>

namespace CPH::Python {

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL
char_on_contact_validate(void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
                         const JPH_CharacterContact *contact) {

    CharacterObject *self = static_cast<CharacterObject *>(userData);
    if (!self || !self->world || !contact) {
        return true;
    }
    PhysicsWorldObject *world = self->world;

    // 1. Get Culverin Handles
    uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

    // Direct thread-safe lookup of the other body's UserData from contact
    uint64_t h2_raw = contact->userData;
    if (h2_raw == 0 && world->physics.body_interface) {
        h2_raw = JPH_BodyInterface_GetUserData(world->physics.body_interface, contact->bodyB);
    }
    // Fallback to safe lock-free id-to-handle map lookup
    if (h2_raw == 0 && world->physics.id_to_handle_map) {
        uint32_t j_idx = JPH_ID_TO_INDEX(contact->bodyB);
        if (j_idx <= world->physics.max_jolt_bodies) {
            h2_raw =
                (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
        }
    }

    if (h2_raw == 0) {
        return true; // Collide by default if handle is missing (e.g. static environment)
    }

    // 2. Safely Resolve Slots with Generation Checking
    uint32_t slot1 = 0;
    uint32_t slot2 = 0;
    if (!unpack_handle(&world->physics, static_cast<BodyHandle>(h1_raw), &slot1) ||
        !unpack_handle(&world->physics, static_cast<BodyHandle>(h2_raw), &slot2)) {
        return true; // Collide on handle verification mismatches
    }

    // Safety checks for array bounds
    if (slot1 >= world->physics.slot_capacity || slot2 >= world->physics.slot_capacity) {
        return true;
    }

    // Check states to ensure both slots are in use
    uint8_t state1 = (world->physics.bodies.slot_states[slot1]).load(std::memory_order::relaxed);
    uint8_t state2 = (world->physics.bodies.slot_states[slot2]).load(std::memory_order::relaxed);
    if (state1 == SLOT_EMPTY || state2 == SLOT_EMPTY) {
        return true;
    }

    uint32_t idx1 = world->physics.bodies.slot_to_dense[slot1];
    uint32_t idx2 = world->physics.bodies.slot_to_dense[slot2];

    size_t active_count = (world->physics.count).load(std::memory_order::relaxed);
    if (idx1 >= active_count || idx2 >= active_count) {
        return true;
    }

    // 3. Perform Bitmask Filtering
    uint32_t cat1  = world->physics.bodies.categories[idx1];
    uint32_t mask1 = world->physics.bodies.masks[idx1];
    uint32_t cat2  = world->physics.bodies.categories[idx2];
    uint32_t mask2 = world->physics.bodies.masks[idx2];

    // Reject if either mask blocks the other's category
    return (cat1 & mask2) != 0 && (cat2 & mask1) != 0;
}

static void record_character_contact(CharacterObject *self, JPH_BodyID bodyID2,
                                     const JPH_RVec3 *pos, const JPH_Vec3 *norm,
                                     ContactEventType type) {
    auto world     = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    BodyHandle h2  = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        // TSan Fix: Atomic load from the shared handle map.
        // Acquire ensures we see the body initialization if it was just created.
        h2 = (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
    }

    // Extract raw value for liveness check
    uint64_t h2_raw = h2;
    if (h2_raw == 0) {
        return;
    }

    // TSan Fix: Atomic load of the character's own handle
    uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

    size_t idx =
        (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);

    if (idx < world->physics.contact_max_capacity) {
        // ContactEvent_ Refactor
        CPH_ContactEvent &event     = world->physics.contact_buffer[idx];
        std::memset(&event, 0, sizeof(event)); // buffer is allocate_for_overwrite
        CPH_ContactEventSlim *slim  = &event.slim;
        CPH_ContactEventFatExt *fat = &event.fat;

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
        slim->px            = static_cast<JPH_Real>(pos->x);
        slim->py            = static_cast<JPH_Real>(pos->y);
        slim->pz            = static_cast<JPH_Real>(pos->z);
        slim->nx            = norm->x;
        slim->ny            = norm->y;
        slim->nz            = norm->z;
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

static void report_char_vs_char(CharacterObject *self, const JPH_CharacterVirtual *other,
                                const JPH_Vec3 *normal, const JPH_RVec3 *pos,
                                ContactEventType type) {
    auto world      = self->world;
    uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

    // FIX: Retrieve the handle directly from the other character's UserData.
    // This is 100% reliable and avoids map boundary checks.
    uint64_t h2_raw = JPH_CharacterVirtual_GetUserData(other);

    if (h2_raw == 0) {
        return; // Still 0? Probably a Jolt internal body.
    }

    size_t idx =
        (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);
    if (idx < world->physics.contact_max_capacity) {
        // ContactEvent_ Refactor
        CPH_ContactEvent &event     = world->physics.contact_buffer[idx];
        std::memset(&event, 0, sizeof(event)); // buffer is allocate_for_overwrite
        CPH_ContactEventSlim *slim  = &event.slim;
        CPH_ContactEventFatExt *fat = &event.fat;

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
        slim->nx            = normal->x;
        slim->ny            = normal->y;
        slim->nz            = normal->z;
        slim->px            = static_cast<JPH_Real>(pos->x);
        slim->py            = static_cast<JPH_Real>(pos->y);
        slim->pz            = static_cast<JPH_Real>(pos->z);
        slim->impulse       = 1.0f;
        fat->mat1           = 0;
        fat->mat2           = 0;

        // Release visibility of the whole event to Python
        std::atomic_thread_fence(std::memory_order::release);
    }
}

static void JPH_API_CALL char_on_character_contact_added(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = static_cast<CharacterObject *>(userData);
    if (!self || !contact) {
        return;
    }

    report_char_vs_char(self, contact->characterB, &contact->contactNormal, &contact->position,
                        EVENT_ADDED);
}

static void apply_character_impulse(CharacterObject *self, JPH_BodyID bodyID2,
                                    const JPH_Vec3 *contactNormal) {
    // 1. Thread-Safe Member Access
    float vx       = self->last_vx.load(std::memory_order::relaxed);
    float vy       = self->last_vy.load(std::memory_order::relaxed);
    float vz       = self->last_vz.load(std::memory_order::relaxed);
    float strength = self->push_strength.load(std::memory_order::relaxed);

    JPH_BodyInterface *bi = self->world->physics.body_interface;

    // 2. Ignore Sensors & Non-Dynamic Bodies
    if (static_cast<int>(JPH_BodyInterface_IsSensor(bi, bodyID2)) ||
        JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
        return;
    }

    // 3. Calculate Pushing Force
    float dot = vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

    // Normal points TOWARDS character, so dot is negative when colliding
    if (dot < -0.01f) {
        float factor                = -dot * strength; // Negate to get positive push force
        constexpr float max_impulse = 50000.0f;
        if (factor > max_impulse) {
            factor = max_impulse;
        }

        JPH_Vec3 impulse;
        // Flip the normal to push the OBJECT away from the character
        impulse.x    = -contactNormal->x * factor;
        float y_push = -contactNormal->y * factor;
        impulse.y    = (y_push > 0.0f) ? y_push : 0.0f;
        impulse.z    = -contactNormal->z * factor;

        JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
        JPH_BodyInterface_ActivateBody(bi, bodyID2);
    }
}

static void JPH_API_CALL char_on_contact_added(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = static_cast<CharacterObject *>(userData);
    if (!self || !contact) {
        return;
    }

    // Record Event
    record_character_contact(self, contact->bodyB, &contact->position, &contact->contactNormal,
                             EVENT_ADDED);

    // Apply Impulse
    apply_character_impulse(self, contact->bodyB, &contact->contactNormal);
}

static void JPH_API_CALL char_on_contact_persisted(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = static_cast<CharacterObject *>(userData);
    if (!self || !contact) {
        return;
    }

    // Record Event
    record_character_contact(self, contact->bodyB, &contact->position, &contact->contactNormal,
                             EVENT_PERSISTED);

    // Apply Impulse
    apply_character_impulse(self, contact->bodyB, &contact->contactNormal);
}

static void JPH_API_CALL char_on_contact_removed(void *userData,
                                                 const JPH_CharacterVirtual *character
                                                 [[maybe_unused]],
                                                 JPH_BodyID bodyID2,
                                                 JPH_SubShapeID subShapeID2 [[maybe_unused]]) {
    auto self = static_cast<CharacterObject *>(userData);
    if (!self || !self->world) {
        return;
    }

    PhysicsWorldObject *world = self->world;
    uint32_t j_idx                = JPH_ID_TO_INDEX(bodyID2);

    // TSan Fix: Explicitly load raw values to avoid seq_cst overhead
    uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);
    uint64_t h2_raw = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        h2_raw =
            (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
    }

    if (h2_raw == 0) {
        return;
    }

    size_t idx =
        (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);
    if (idx < world->physics.contact_max_capacity) {
        // Buffer is allocate_for_overwrite: define the whole slot
        // (incl. fat fields) before filling slim.
        std::memset(&world->physics.contact_buffer[idx], 0, sizeof(CPH_ContactEvent));
        CPH_ContactEventSlim *slim = &world->physics.contact_buffer[idx].slim;

        slim->flags = EVENT_REMOVED;

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

static void JPH_API_CALL char_on_character_contact_persisted(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    CharacterObject *self = static_cast<CharacterObject *>(userData);
    if (!self || !contact) {
        return;
    }

    report_char_vs_char(self, contact->characterB, &contact->contactNormal, &contact->position,
                        EVENT_PERSISTED);
}

static void JPH_API_CALL char_on_character_contact_removed(
    void *userData, const JPH_CharacterVirtual *character [[maybe_unused]],
    const JPH_CharacterID otherCharacterID, JPH_SubShapeID subShapeID2 [[maybe_unused]]) {
    auto self  = static_cast<CharacterObject *>(userData);
    auto world = self->world;

    // TSan Fix: Explicit relaxed load of own handle
    uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

    // We have to use the CharacterID to find the handle.
    uint32_t j_idx  = JPH_ID_TO_INDEX(otherCharacterID);
    uint64_t h2_raw = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        h2_raw =
            (world->physics.id_to_handle_map[j_idx]).load(std::memory_order::acquire);
    }

    if (h2_raw == 0) {
        return;
    }

    size_t idx =
        (world->physics.contact_atomic_idx).fetch_add(1, std::memory_order::relaxed);
    if (idx < world->physics.contact_max_capacity) {
        // Buffer is allocate_for_overwrite: define the whole slot
        // (incl. fat fields) before filling slim.
        std::memset(&world->physics.contact_buffer[idx], 0, sizeof(CPH_ContactEvent));
        CPH_ContactEventSlim *slim = &world->physics.contact_buffer[idx].slim;

        slim->flags = EVENT_REMOVED;

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

static void JPH_API_CALL char_on_adjust_velocity(void *userData,
                                                 const JPH_CharacterVirtual *character,
                                                 const JPH_Body *body2, JPH_Vec3 *ioLinearVelocity,
                                                 JPH_Vec3 *ioAngularVelocity) {

    CharacterObject *self = static_cast<CharacterObject *>(userData);
    if (!self || !self->world) {
        return;
    }

    // 1. Get Platform/Friction Data
    uint64_t h2_raw = JPH_Body_GetUserData(const_cast<JPH_Body *>(body2));
    float friction  = 1.0f;
    uint32_t slot2  = static_cast<uint32_t>(h2_raw & kHandleIndexMask);
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
    JPH_Vec3 omega;
    JPH_Body_GetAngularVelocity(const_cast<JPH_Body *>(body2), &omega);

    JPH_RVec3 char_pos;
    JPH_RVec3 plat_pos;
    JPH_CharacterVirtual_GetPosition(const_cast<JPH_CharacterVirtual *>(character), &char_pos);
    JPH_Body_GetPosition(const_cast<JPH_Body *>(body2), &plat_pos);

    // Relative offset
    float rx = static_cast<float>(char_pos.x - plat_pos.x);
    float rz = static_cast<float>(char_pos.z - plat_pos.z);

    // v = omega x r
    float target_vt_x = omega.y * rz;
    float target_vt_z = -omega.y * rx;

    // 3. APPLY: Override inheritance with friction scaling
    // We use a factor of 1.0 for friction >= 0.2
    float factor = (friction > 0.2f) ? 1.0f : (friction / 0.2f);

    // Set the inherited linear velocity
    ioLinearVelocity->x = target_vt_x * factor;
    ioLinearVelocity->z = target_vt_z * factor;

    // Set the inherited angular velocity (rotation)
    ioAngularVelocity->y = omega.y * factor;
}

static bool JPH_API_CALL char_on_character_contact_validate(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact) {

    CharacterObject *self = static_cast<CharacterObject *>(userData);
    if (!self || !self->world || !contact) {
        return true;
    }

    PhysicsWorldObject *world = self->world;

    // 1. Get Culverin Handles
    uint64_t h1_raw = (self->handle).load(std::memory_order::relaxed);

    // In JPH_CharacterContact, other character handle is stored in contact->userData
    // or falls back to JPH_CharacterVirtual_GetUserData(contact->characterB)
    uint64_t h2_raw = contact->userData;
    if (h2_raw == 0 && contact->characterB) {
        h2_raw = JPH_CharacterVirtual_GetUserData(contact->characterB);
    }

    if (h2_raw == 0) {
        return true; // Collide by default if handle is missing
    }

    // 2. Resolve Slots
    uint32_t slot1 = 0;
    uint32_t slot2 = 0;
    if (!unpack_handle(&world->physics, static_cast<BodyHandle>(h1_raw), &slot1) ||
        !unpack_handle(&world->physics, static_cast<BodyHandle>(h2_raw), &slot2)) {
        return true;
    }

    if (slot1 >= world->physics.slot_capacity || slot2 >= world->physics.slot_capacity) {
        return true;
    }

    uint32_t idx1 = world->physics.bodies.slot_to_dense[slot1];
    uint32_t idx2 = world->physics.bodies.slot_to_dense[slot2];

    size_t active_count = (world->physics.count).load(std::memory_order::relaxed);
    if (idx1 >= active_count || idx2 >= active_count) {
        return true;
    }

    // 3. Perform Bitmask Filtering
    uint32_t cat1  = world->physics.bodies.categories[idx1];
    uint32_t mask1 = world->physics.bodies.masks[idx1];
    uint32_t cat2  = world->physics.bodies.categories[idx2];
    uint32_t mask2 = world->physics.bodies.masks[idx2];

    // Reject if either mask blocks the other's category
    return ((cat1 & mask2) && (cat2 & mask1)) != 0;
}

// High-frequency callback: DO NOT allocate memory or lock.
[[maybe_unused]] static void JPH_API_CALL char_on_contact_solve(
    [[maybe_unused]] void *userData, [[maybe_unused]] const JPH_Body *body1,
    [[maybe_unused]] const JPH_Body *body2, [[maybe_unused]] const JPH_ContactManifold *manifold,
    [[maybe_unused]] JPH_ContactSettings *settings) CPH_NO_TSAN {
    // Advanced solver overrides here if needed
}

// Map the procs
const JPH_CharacterContactListener_Procs kCharListenerProcs = {
    .OnAdjustBodyVelocity        = char_on_adjust_velocity,
    .OnContactValidate           = char_on_contact_validate,
    .OnCharacterContactValidate  = char_on_character_contact_validate,
    .OnContactAdded              = char_on_contact_added,
    .OnContactPersisted          = char_on_contact_persisted,
    .OnContactRemoved            = char_on_contact_removed,
    .OnCharacterContactAdded     = char_on_character_contact_added,
    .OnCharacterContactPersisted = char_on_character_contact_persisted,
    .OnCharacterContactRemoved   = char_on_character_contact_removed,
    .OnContactSolve              = nullptr,
    .OnCharacterContactSolve     = nullptr};

[[nodiscard]] PyObject * CharacterMove(CharacterObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {
    // 1. INTEGRATED FAST PARSE (Unchanged)
    CPH_Vec3f v_in = {};
    float dt       = 0.0f;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!self->parsers->CharMove.Parse(args, nargs, kwnames, v_in, dt)) {
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
    JPH_Vec3 j_v = {v_in.x, v_in.y, v_in.z};

    // Inherit velocity from ground platform if grounded
    JPH_BodyID ground_id =
        JPH_CharacterBase_GetGroundBodyId(reinterpret_cast<JPH_CharacterBase *>(self->character));

    if (ground_id != kJphInvalidBodyId) {
        JPH_Vec3 ground_vel;
        JPH_CharacterBase_GetGroundVelocity(reinterpret_cast<JPH_CharacterBase *>(self->character),
                                            &ground_vel);
        j_v.x += ground_vel.x;
        // j_v.y += ground_vel.y;
        j_v.z += ground_vel.z;
    }

    JPH_CharacterVirtual_SetLinearVelocity(self->character, &j_v);

    JPH_ExtendedUpdateSettings update_settings       = {};
    update_settings.stickToFloorStepDown             = JPH_Vec3{0.0f, -0.5f, 0.0f};
    update_settings.walkStairsStepUp                 = JPH_Vec3{0.0f, 0.4f, 0.0f};
    update_settings.walkStairsMinStepForward         = 0.02f;
    update_settings.walkStairsStepForwardTest        = 0.15f;
    update_settings.walkStairsCosAngleForwardContact = 0.996f;

    (self->world->physics.jph_trampoline_lock).lock();
    Py_BEGIN_ALLOW_THREADS JPH_CharacterVirtual_ExtendedUpdate(
        self->character, dt, &update_settings, OBJECT_LAYER_DYNAMIC, self->world->physics.system,
        self->body_filter, self->shape_filter);
    Py_END_ALLOW_THREADS (self->world->physics.jph_trampoline_lock).unlock();

    // 4. POST-MOVE SYNC
    (self->world->physics.shadow_lock).lock();

    alignas(alignof(JPH_RVec3)) JPH_RVec3 current_p_storage{};
    JPH_RVec3 *current_p = &current_p_storage;
    alignas(alignof(JPH_Quat)) JPH_Quat current_r_storage{};
    JPH_Quat *current_r = &current_r_storage;
    JPH_CharacterVirtual_GetPosition(self->character, current_p);
    JPH_CharacterVirtual_GetRotation(self->character, current_r);

    // Sync non-atomic shadow buffers
    shadow_pos[dense] = CPH_PosStride{current_p->x, current_p->y, current_p->z, 0.0};
    shadow_rot[dense] = CPH_AuxStride{current_r->x, current_r->y, current_r->z, current_r->w};

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * CharacterGetPosition(CharacterObject *self,
                                                           PyObject *ignored [[maybe_unused]]) {
    // 1. Aligned stack storage for SIMD
    alignas(alignof(JPH_RVec3)) JPH_RVec3 pos_storage{};
    JPH_RVec3 *pos = &pos_storage;

    // 2. Lock for consistency (ensure we aren't reading mid-step)
    (self->world->physics.shadow_lock).lock();
    JPH_CharacterVirtual_GetPosition(self->character, pos);
    (self->world->physics.shadow_lock).unlock();

    PyObject *ret = PyTuple_New(3);
    if (!ret) {
        return nullptr;
    }

    // Use the double precision provided by RVec3
    PyTuple_SET_ITEM(ret, 0, PyFloat_FromDouble(pos->x));
    PyTuple_SET_ITEM(ret, 1, PyFloat_FromDouble(pos->y));
    PyTuple_SET_ITEM(ret, 2, PyFloat_FromDouble(pos->z));

    return ret;
}

[[nodiscard]] PyObject *
CharacterGetLinearVelocity(CharacterObject *self, PyObject *ignored [[maybe_unused]]) {
    alignas(alignof(JPH_Vec3)) JPH_Vec3 vel_storage{};
    JPH_Vec3 *vel = &vel_storage;

    (self->world->physics.shadow_lock).lock();
    JPH_CharacterVirtual_GetLinearVelocity(self->character, vel);
    (self->world->physics.shadow_lock).unlock();

    return BuildTuple(vel->x, vel->y, vel->z);
}

[[nodiscard]] PyObject * CharacterSetPosition(CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    CPH_PosStride pos      = {};

    if (!self->parsers->SetPosChar.Parse(args, nargs, kwnames, pos)) {
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    JPH_RVec3 j_pos = {pos.x, pos.y, pos.z};
    JPH_CharacterVirtual_SetPosition(self->character, &j_pos);

    // Update Shadow Buffers
    uint64_t h_raw = (self->handle).load(std::memory_order::relaxed);
    uint32_t slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t dense = self->world->physics.bodies.slot_to_dense[slot];

    auto shadow_pos  = self->world->physics.bodies.positions.data();
    auto shadow_ppos = self->world->physics.bodies.prev_positions.data();

    CPH_PosStride p_val = {pos.x, pos.y, pos.z, 0.0};
    shadow_pos[dense]   = p_val;
    shadow_ppos[dense]  = p_val;

    (self->world->physics.shadow_lock).unlock();
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * CharacterSetRotation(CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    CPH_AuxStride rot      = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    if (!self->parsers->SetRotChar.Parse(args, nargs, kwnames, rot)) {
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);

    JPH_Quat q = {.x = rot.x, .y = rot.y, .z = rot.z, .w = rot.w};
    JPH_CharacterVirtual_SetRotation(self->character, &q);

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

    if (!self->parsers->SetStrengthChar.Parse(args, nargs, kwnames, strength)) {
        return nullptr;
    }

    (self->world->physics.shadow_lock).lock();
    BlockUntilNotStepping(self->world);
    BlockUntilNotQuerying(self->world);

    (self->push_strength).store(strength, std::memory_order::relaxed);
    JPH_CharacterVirtual_SetMaxStrength(self->character, strength);

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

    alignas(alignof(JPH_RVec3)) JPH_RVec3 end_p_storage{};
    JPH_RVec3 *end_p = &end_p_storage;
    alignas(alignof(JPH_Quat)) JPH_Quat end_r_storage{};
    JPH_Quat *end_r = &end_r_storage;
    JPH_CharacterVirtual_GetPosition(self->character, end_p);
    JPH_CharacterVirtual_GetRotation(self->character, end_r);

    CPH_MathInterpolateCharacterTransform(&shadow_ppos[dense], &shadow_prot[dense], end_p, end_r,
                                          alpha, res_p, res_r);

    (self->world->physics.shadow_lock).unlock();

    return BuildTuple(BuildTuple(res_p[0], res_p[1], res_p[2]),
                              BuildTuple(res_r[0], res_r[1], res_r[2], res_r[3]));
}

[[nodiscard]] PyObject * CharacterIsGrounded(CharacterObject *self,
                                                          PyObject *ignored [[maybe_unused]]) {
    (self->world->physics.shadow_lock).lock();
    JPH_GroundState state =
        JPH_CharacterBase_GetGroundState(reinterpret_cast<JPH_CharacterBase *>(self->character));
    (self->world->physics.shadow_lock).unlock();

    if (state == JPH_GroundState_OnGround) {
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

    CPH_WorldRemoveBodySlot(&self->world->physics, slot);

    (self->world->physics.shadow_lock).unlock();

    // --- 2. JOLT DESTRUCTION (Hard Serialized) ---
    (self->world->physics.jph_trampoline_lock).lock();

    if (self->world->physics.char_vs_char_manager && self->character) {
        JPH_CharacterVsCharacterCollisionSimple_RemoveCharacter(
            self->world->physics.char_vs_char_manager, self->character);
    }

    if (self->character) {
        JPH_CharacterBase_Destroy(reinterpret_cast<JPH_CharacterBase *>(self->character));
    }
    if (self->listener) {
        JPH_CharacterContactListener_Destroy(self->listener);
    }
    if (self->body_filter) {
        JPH_BodyFilter_Destroy(self->body_filter);
    }
    if (self->shape_filter) {
        JPH_ShapeFilter_Destroy(self->shape_filter);
    }
    if (self->bp_filter) {
        JPH_BroadPhaseLayerFilter_Destroy(self->bp_filter);
    }
    if (self->obj_filter) {
        JPH_ObjectLayerFilter_Destroy(self->obj_filter);
    }

    (self->world->physics.jph_trampoline_lock).unlock();

finalize:
    if (self->parsers) {

        delete self->parsers;
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// Helper 1: Jolt-side allocation and Collision Manager linking
static inline JPH_CharacterVirtual *
alloc_j_char(PhysicsWorldObject *self, PositionVector pos, CharacterParams params) {

    float half_h                 = fmaxf((params.height - 2.0f * params.radius) * 0.5f, 0.1f);
    JPH_CapsuleShapeSettings *ss = JPH_CapsuleShapeSettings_Create(half_h, params.radius);
    auto shape = reinterpret_cast<JPH_Shape *>(JPH_CapsuleShapeSettings_CreateShape(ss));
    JPH_ShapeSettings_Destroy(reinterpret_cast<JPH_ShapeSettings *>(ss));
    if (!shape) {
        return nullptr;
    }

    JPH_CharacterVirtualSettings settings;
    JPH_CharacterVirtualSettings_Init(&settings);
    settings.base.shape                         = shape;
    static constexpr float DegreesPerSemiCircle = 180.0f;
    settings.base.maxSlopeAngle = params.max_slope * (JPH_M_PI / DegreesPerSemiCircle);

    JPH_RVec3 initial_position{static_cast<double>(pos.x), static_cast<double>(pos.y),
                               static_cast<double>(pos.z)};
    JPH_Quat initial_rotation{0, 0, 0, 1};
    JPH_CharacterVirtual *j_char = JPH_CharacterVirtual_Create(
        &settings, &initial_position, &initial_rotation, 1, self->physics.system);

    JPH_Shape_Destroy(shape);
    if (!j_char) {
        return nullptr;
    }

    if (self->physics.char_vs_char_manager) {
        JPH_CharacterVsCharacterCollisionSimple_AddCharacter(self->physics.char_vs_char_manager,
                                                             j_char);
        JPH_CharacterVirtual_SetCharacterVsCharacterCollision(j_char,
                                                              self->physics.char_vs_char_manager);
    }
    return j_char;
}

// Helper 2: Shadow Buffer Registration (Atomic Commit)
static inline void register_char(PhysicsWorldObject *self, CharacterObject *obj,
                                 JPH_CharacterVirtual *j_char, uint32_t slot) {
    (self->physics.shadow_lock).lock();

    uint32_t gen = (self->physics.bodies.generations[slot]).load(std::memory_order::relaxed);
    BodyHandle h = make_handle(slot, gen);

    uint64_t raw_h = h;
    (obj->handle).store(raw_h, std::memory_order::relaxed);

    auto dense_idx =
        static_cast<uint32_t>((self->physics.count).load(std::memory_order::relaxed));
    JPH_BodyID bid = JPH_CharacterVirtual_GetInnerBodyID(j_char);
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);

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

    alignas(alignof(JPH_RVec3)) JPH_RVec3 p_storage{};
    JPH_RVec3 *p = &p_storage;
    alignas(alignof(JPH_Quat)) JPH_Quat q_storage{};
    JPH_Quat *q = &q_storage;
    JPH_CharacterVirtual_GetPosition(j_char, p);
    JPH_CharacterVirtual_GetRotation(j_char, q);

    CPH_PosStride p_val = {p->x, p->y, p->z, 0.0};
    CPH_AuxStride r_val = {q->x, q->y, q->z, q->w};

    (self->physics.bodies.positions.data())[dense_idx]      = p_val;
    (self->physics.bodies.prev_positions.data())[dense_idx] = p_val;
    (self->physics.bodies.rotations.data())[dense_idx]      = r_val;
    (self->physics.bodies.prev_rotations.data())[dense_idx] = r_val;
    (self->physics.bodies.linear_velocities.data())[dense_idx] =
        CPH_AuxStride{};
    (self->physics.bodies.angular_velocities.data())[dense_idx] =
        CPH_AuxStride{};

    if (self->physics.bodies.soft_shadows) {
        self->physics.bodies.soft_shadows[dense_idx].vertices.reset();
    }

    (self->physics.bodies.slot_states[slot]).store(SLOT_CHARACTER, std::memory_order::release);
    (self->physics.count).fetch_add(1, std::memory_order::release);

    size_t final_count  = (self->physics.count).load(std::memory_order::relaxed);
    self->view_shape[0] = static_cast<Py_ssize_t>(final_count);

    JPH_BodyInterface_SetUserData(self->physics.body_interface, bid, raw_h);
    JPH_CharacterVirtual_SetUserData(j_char, raw_h);

    (self->physics.shadow_lock).unlock();
}

// Helper 3: Filter and Listener serialization (Trampoline Lock)
static inline void setup_char_filters(CharacterObject *obj) {
    (obj->world->physics.jph_trampoline_lock).lock();
    obj->listener     = JPH_CharacterContactListener_Create(obj);
    obj->body_filter  = JPH_BodyFilter_Create(nullptr);
    obj->shape_filter = JPH_ShapeFilter_Create(nullptr);
    obj->bp_filter    = JPH_BroadPhaseLayerFilter_Create(nullptr);
    obj->obj_filter   = JPH_ObjectLayerFilter_Create(nullptr);
    (obj->world->physics.jph_trampoline_lock).unlock();
    JPH_CharacterVirtual_SetListener(obj->character, obj->listener);
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
    CPH_PosStride pos             = {};
    constexpr auto DEFAULT_HEIGHT = 1.8f;
    constexpr auto DEFAULT_RADIUS = 0.4f;
    constexpr auto DEFUALT_STEP_H = 0.4f;
    constexpr auto DEFAULT_SLOPE  = 45.0f;
    float height                  = DEFAULT_HEIGHT;
    float radius                  = DEFAULT_RADIUS;
    float step_h                  = DEFUALT_STEP_H;
    float slope                   = DEFAULT_SLOPE;

    if (!self->parsers->CreateChar.Parse(args, nargs, kwnames, pos, height, radius, step_h,
                                         slope)) {
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

    (self->physics.bodies.slot_states[char_slot]).store(SLOT_PENDING_CREATE, std::memory_order::release);

    (self->physics.shadow_lock).unlock();

    PositionVector pos_vec      = {pos.x, pos.y, pos.z};
    CharacterParams char_params = {height, radius, slope};

    CharacterObject *obj     = nullptr;
    JPH_CharacterVirtual *j_char = alloc_j_char(self, pos_vec, char_params);
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
    setup_char_filters(obj);

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

fail_py:
    JPH_CharacterBase_Destroy(reinterpret_cast<JPH_CharacterBase *>(j_char));
fail_jolt:
    (self->physics.shadow_lock).lock();

    (self->physics.bodies.slot_states[char_slot]).store(SLOT_EMPTY, std::memory_order::relaxed);

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
