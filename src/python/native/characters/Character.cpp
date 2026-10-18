#include "characters/Character.h"
#include "bindings/BindingCommon.h"
#include "bindings/FastBuild.h"
#include "bindings/ModuleState.h"
#include "bindings/PythonSupport.h"
#include "bindings/parsing/ParserRegistry.hpp"
#include "bindings/physics/PhysicsSync.h"
#include "bindings/physics/WorldAdapters.h"
#include "core/CompilerSpecifics.h"
#include "core/Types.h"
#include "math/MathCore.h"
#include "physics/ContactEventDefinitions.h"
#include <math.h>

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL
char_on_contact_validate(void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
                         const JPH_CharacterContact *contact) {

    CPH_CharacterObject *self = static_cast<CPH_CharacterObject *>(userData);
    if (!self || !self->world || !contact) {
        return true;
    }
    CPH_PhysicsWorldObject *world = self->world;

    // 1. Get Culverin Handles
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

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
                atomic_load_explicit(&world->physics.id_to_handle_map[j_idx], memory_order_acquire);
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
    uint8_t state1 = atomic_load_explicit(&world->physics.slot_states[slot1], memory_order_relaxed);
    uint8_t state2 = atomic_load_explicit(&world->physics.slot_states[slot2], memory_order_relaxed);
    if (state1 == SLOT_EMPTY || state2 == SLOT_EMPTY) {
        return true;
    }

    uint32_t idx1 = world->physics.slot_to_dense[slot1];
    uint32_t idx2 = world->physics.slot_to_dense[slot2];

    size_t active_count = atomic_load_explicit(&world->physics.count, memory_order_relaxed);
    if (idx1 >= active_count || idx2 >= active_count) {
        return true;
    }

    // 3. Perform Bitmask Filtering
    uint32_t cat1  = world->physics.categories[idx1];
    uint32_t mask1 = world->physics.masks[idx1];
    uint32_t cat2  = world->physics.categories[idx2];
    uint32_t mask2 = world->physics.masks[idx2];

    // Reject if either mask blocks the other's category
    return (cat1 & mask2) != 0 && (cat2 & mask1) != 0;
}

static void record_character_contact(CPH_CharacterObject *self, JPH_BodyID bodyID2,
                                     const JPH_RVec3 *pos, const JPH_Vec3 *norm,
                                     ContactEventType type) {
    auto world     = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    BodyHandle h2  = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        // TSan Fix: Atomic load from the shared handle map.
        // Acquire ensures we see the body initialization if it was just created.
        h2 = atomic_load_explicit(&world->physics.id_to_handle_map[j_idx], memory_order_acquire);
    }

    // Extract raw value for liveness check
    uint64_t h2_raw = h2;
    if (h2_raw == 0) {
        return;
    }

    // TSan Fix: Atomic load of the character's own handle
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

    size_t idx =
        atomic_fetch_add_explicit(&world->physics.contact_atomic_idx, 1, memory_order_relaxed);

    if (idx < world->physics.contact_max_capacity) {
        // ContactEvent_ Refactor
        CPH_ContactEvent *ev_base   = GetEventAt(world->physics.contact_buffer, idx);
        CPH_ContactEventSlim *slim  = GetSlimHeader(ev_base);
        CPH_ContactEventFatExt *fat = GetFatExtension(ev_base);

        slim->flags = static_cast<uint32_t>(type);

        // TSan Fix: Use raw values for canonical handle ordering
        if (h1_raw < h2_raw) {
            atomic_store_explicit(&slim->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&slim->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h1_raw, memory_order_relaxed);
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
        uint32_t dense2 = world->physics.slot_to_dense[slot2];
        fat->mat1       = 0;
        fat->mat2       = world->physics.material_ids[dense2];

        // Release fence ensures all previous stores are visible to Python when it reads
        // contact_atomic_idx
        atomic_thread_fence(memory_order_release);
    }
}

static void report_char_vs_char(CPH_CharacterObject *self, const JPH_CharacterVirtual *other,
                                const JPH_Vec3 *normal, const JPH_RVec3 *pos,
                                ContactEventType type) {
    auto world      = self->world;
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

    // FIX: Retrieve the handle directly from the other character's UserData.
    // This is 100% reliable and avoids map boundary checks.
    uint64_t h2_raw = JPH_CharacterVirtual_GetUserData(other);

    if (h2_raw == 0) {
        return; // Still 0? Probably a Jolt internal body.
    }

    size_t idx =
        atomic_fetch_add_explicit(&world->physics.contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->physics.contact_max_capacity) {
        // ContactEvent_ Refactor
        CPH_ContactEvent *ev_base   = GetEventAt(world->physics.contact_buffer, idx);
        CPH_ContactEventSlim *slim  = GetSlimHeader(ev_base);
        CPH_ContactEventFatExt *fat = GetFatExtension(ev_base);

        slim->flags = static_cast<uint32_t>(type);

        // TSan Fix: Canonicalize ordering
        if (h1_raw < h2_raw) {
            atomic_store_explicit(&slim->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&slim->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h1_raw, memory_order_relaxed);
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
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_character_contact_added(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = static_cast<CPH_CharacterObject *>(userData);
    if (!self || !contact) {
        return;
    }

    report_char_vs_char(self, contact->characterB, &contact->contactNormal, &contact->position,
                        EVENT_ADDED);
}

static void apply_character_impulse(CPH_CharacterObject *self, JPH_BodyID bodyID2,
                                    const JPH_Vec3 *contactNormal) {
    // 1. Thread-Safe Member Access
    float vx       = atomic_load_explicit((&self->last_vx), memory_order_relaxed);
    float vy       = atomic_load_explicit((&self->last_vy), memory_order_relaxed);
    float vz       = atomic_load_explicit((&self->last_vz), memory_order_relaxed);
    float strength = atomic_load_explicit((&self->push_strength), memory_order_relaxed);

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

    auto self = static_cast<CPH_CharacterObject *>(userData);
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

    auto self = static_cast<CPH_CharacterObject *>(userData);
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
    auto self = static_cast<CPH_CharacterObject *>(userData);
    if (!self || !self->world) {
        return;
    }

    CPH_PhysicsWorldObject *world = self->world;
    uint32_t j_idx                = JPH_ID_TO_INDEX(bodyID2);

    // TSan Fix: Explicitly load raw values to avoid seq_cst overhead
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint64_t h2_raw = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        h2_raw =
            atomic_load_explicit(&world->physics.id_to_handle_map[j_idx], memory_order_acquire);
    }

    if (h2_raw == 0) {
        return;
    }

    size_t idx =
        atomic_fetch_add_explicit(&world->physics.contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->physics.contact_max_capacity) {
        CPH_ContactEvent *ev_base  = GetEventAt(world->physics.contact_buffer, idx);
        CPH_ContactEventSlim *slim = GetSlimHeader(ev_base);

        slim->flags = EVENT_REMOVED;

        // TSan Fix: Canonicalize using raw registers
        if (h1_raw < h2_raw) {
            atomic_store_explicit(&slim->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&slim->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h1_raw, memory_order_relaxed);
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
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_character_contact_persisted(
    void *userData, [[maybe_unused]] const JPH_CharacterVirtual *character,
    const JPH_CharacterContact *contact, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    CPH_CharacterObject *self = static_cast<CPH_CharacterObject *>(userData);
    if (!self || !contact) {
        return;
    }

    report_char_vs_char(self, contact->characterB, &contact->contactNormal, &contact->position,
                        EVENT_PERSISTED);
}

static void JPH_API_CALL char_on_character_contact_removed(
    void *userData, const JPH_CharacterVirtual *character [[maybe_unused]],
    const JPH_CharacterID otherCharacterID, JPH_SubShapeID subShapeID2 [[maybe_unused]]) {
    auto self  = static_cast<CPH_CharacterObject *>(userData);
    auto world = self->world;

    // TSan Fix: Explicit relaxed load of own handle
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

    // We have to use the CharacterID to find the handle.
    uint32_t j_idx  = JPH_ID_TO_INDEX(otherCharacterID);
    uint64_t h2_raw = 0;

    if (world->physics.id_to_handle_map && j_idx <= world->physics.max_jolt_bodies) {
        h2_raw =
            atomic_load_explicit(&world->physics.id_to_handle_map[j_idx], memory_order_acquire);
    }

    if (h2_raw == 0) {
        return;
    }

    size_t idx =
        atomic_fetch_add_explicit(&world->physics.contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->physics.contact_max_capacity) {
        CPH_ContactEvent *ev_base  = GetEventAt(world->physics.contact_buffer, idx);
        CPH_ContactEventSlim *slim = GetSlimHeader(ev_base);

        slim->flags = EVENT_REMOVED;

        // TSan Fix: Canonicalize using standard integer registers
        if (h1_raw < h2_raw) {
            atomic_store_explicit(&slim->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&slim->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&slim->body2, h1_raw, memory_order_relaxed);
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
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_adjust_velocity(void *userData,
                                                 const JPH_CharacterVirtual *character,
                                                 const JPH_Body *body2, JPH_Vec3 *ioLinearVelocity,
                                                 JPH_Vec3 *ioAngularVelocity) {

    CPH_CharacterObject *self = static_cast<CPH_CharacterObject *>(userData);
    if (!self || !self->world) {
        return;
    }

    // 1. Get Platform/Friction Data
    uint64_t h2_raw = JPH_Body_GetUserData(const_cast<JPH_Body *>(body2));
    float friction  = 1.0f;
    uint32_t slot2  = static_cast<uint32_t>(h2_raw & kHandleIndexMask);
    if (slot2 < self->world->physics.slot_capacity) {
        uint32_t dense2 = self->world->physics.slot_to_dense[slot2];
        uint32_t mat_id = self->world->physics.material_ids[dense2];
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

    CPH_CharacterObject *self = static_cast<CPH_CharacterObject *>(userData);
    if (!self || !self->world || !contact) {
        return true;
    }

    CPH_PhysicsWorldObject *world = self->world;

    // 1. Get Culverin Handles
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

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

    uint32_t idx1 = world->physics.slot_to_dense[slot1];
    uint32_t idx2 = world->physics.slot_to_dense[slot2];

    size_t active_count = atomic_load_explicit(&world->physics.count, memory_order_relaxed);
    if (idx1 >= active_count || idx2 >= active_count) {
        return true;
    }

    // 3. Perform Bitmask Filtering
    uint32_t cat1  = world->physics.categories[idx1];
    uint32_t mask1 = world->physics.masks[idx1];
    uint32_t cat2  = world->physics.categories[idx2];
    uint32_t mask2 = world->physics.masks[idx2];

    // Reject if either mask blocks the other's category
    return ((cat1 & mask2) && (cat2 & mask1)) != 0;
}

// High-frequency callback: DO NOT allocate memory or lock.
CPH_NO_TSAN [[maybe_unused]] static void JPH_API_CALL char_on_contact_solve(
    [[maybe_unused]] void *userData, [[maybe_unused]] const JPH_Body *body1,
    [[maybe_unused]] const JPH_Body *body2, [[maybe_unused]] const JPH_ContactManifold *manifold,
    [[maybe_unused]] JPH_ContactSettings *settings) {
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

PyCFunction_DeclareMethodFromModule CPH_PyChar_Move(CPH_CharacterObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {
    // 1. INTEGRATED FAST PARSE (Unchanged)
    CPH_Vec3f v_in = {};
    float dt       = 0.0f;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!self->parsers->CharMove.Parse(args, nargs, kwnames, v_in, dt)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_VEC3(v_in.x, v_in.y, v_in.z, "Character velocity");

    // 2. SNAPSHOT (Shadow Lock)
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);

    // TSan Fix: Explicit relaxed stores for worker-thread inputs
    atomic_store_explicit(&self->last_vx, v_in.x, memory_order_relaxed);
    atomic_store_explicit(&self->last_vy, v_in.y, memory_order_relaxed);
    atomic_store_explicit(&self->last_vz, v_in.z, memory_order_relaxed);

    // TSan Fix: Load handle atomically to resolve indices
    uint64_t h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint32_t slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t dense = self->world->physics.slot_to_dense[slot];

    // Snapshot current state to prev for interpolation
    auto shadow_pos  = reinterpret_cast<CPH_PosStride *>(self->world->physics.positions);
    auto shadow_ppos = reinterpret_cast<CPH_PosStride *>(self->world->physics.prev_positions);
    auto shadow_rot  = reinterpret_cast<CPH_AuxStride *>(self->world->physics.rotations);
    auto shadow_prot = reinterpret_cast<CPH_AuxStride *>(self->world->physics.prev_rotations);

    shadow_ppos[dense] = shadow_pos[dense];
    shadow_prot[dense] = shadow_rot[dense];

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

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

    CPH_NATIVE_MUTEX_LOCK(self->world->physics.jph_trampoline_lock);
    Py_BEGIN_ALLOW_THREADS JPH_CharacterVirtual_ExtendedUpdate(
        self->character, dt, &update_settings, OBJECT_LAYER_DYNAMIC, self->world->physics.system,
        self->body_filter, self->shape_filter);
    Py_END_ALLOW_THREADS CPH_NATIVE_MUTEX_UNLOCK(self->world->physics.jph_trampoline_lock);

    // 4. POST-MOVE SYNC
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);

    JPH_STACK_ALLOC(JPH_RVec3, current_p);
    JPH_STACK_ALLOC(JPH_Quat, current_r);
    JPH_CharacterVirtual_GetPosition(self->character, current_p);
    JPH_CharacterVirtual_GetRotation(self->character, current_r);

    // Sync non-atomic shadow buffers
    shadow_pos[dense] = CPH_PosStride{current_p->x, current_p->y, current_p->z, 0.0};
    shadow_rot[dense] = CPH_AuxStride{current_r->x, current_r->y, current_r->z, current_r->w};

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyChar_GetPosition(CPH_CharacterObject *self,
                                                           PyObject *ignored [[maybe_unused]]) {
    // 1. Aligned stack storage for SIMD
    JPH_STACK_ALLOC(JPH_RVec3, pos);

    // 2. Lock for consistency (ensure we aren't reading mid-step)
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    JPH_CharacterVirtual_GetPosition(self->character, pos);
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

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

PyCFunction_DeclareMethodFromModule
CPH_PyChar_GetLinearVelocity(CPH_CharacterObject *self, PyObject *ignored [[maybe_unused]]) {
    JPH_STACK_ALLOC(JPH_Vec3, vel);

    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    JPH_CharacterVirtual_GetLinearVelocity(self->character, vel);
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    return CPH_PY_BUILD_TUPLE(vel->x, vel->y, vel->z);
}

PyCFunction_DeclareMethodFromModule CPH_PyChar_SetPosition(CPH_CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    CPH_PosStride pos      = {};

    if (!self->parsers->SetPosChar.Parse(args, nargs, kwnames, pos)) {
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);

    JPH_RVec3 j_pos = {pos.x, pos.y, pos.z};
    JPH_CharacterVirtual_SetPosition(self->character, &j_pos);

    // Update Shadow Buffers
    uint64_t h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint32_t slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    uint32_t dense = self->world->physics.slot_to_dense[slot];

    auto shadow_pos  = reinterpret_cast<CPH_PosStride *>(self->world->physics.positions);
    auto shadow_ppos = reinterpret_cast<CPH_PosStride *>(self->world->physics.prev_positions);

    CPH_PosStride p_val = {pos.x, pos.y, pos.z, 0.0};
    shadow_pos[dense]   = p_val;
    shadow_ppos[dense]  = p_val;

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyChar_SetRotation(CPH_CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    CPH_AuxStride rot      = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    if (!self->parsers->SetRotChar.Parse(args, nargs, kwnames, rot)) {
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);

    JPH_Quat q = {.x = rot.x, .y = rot.y, .z = rot.z, .w = rot.w};
    JPH_CharacterVirtual_SetRotation(self->character, &q);

    uint64_t raw_h = atomic_load_explicit(&self->handle, memory_order_relaxed);

    auto slot          = static_cast<uint32_t>(raw_h & kHandleIndexMask);
    uint32_t dense_idx = self->world->physics.slot_to_dense[slot];
    size_t off         = static_cast<size_t>(dense_idx) * 4;

    memcpy(&self->world->physics.rotations[off], &q, 16);
    memcpy(&self->world->physics.prev_rotations[off], &q, 16);

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyChar_SetStrength(CPH_CharacterObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float strength         = 0.0f;

    if (!self->parsers->SetStrengthChar.Parse(args, nargs, kwnames, strength)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);
    CPH_BLOCK_UNTIL_NOT_QUERYING(self->world);

    atomic_store_explicit(&self->push_strength, strength, memory_order_relaxed);
    JPH_CharacterVirtual_SetMaxStrength(self->character, strength);

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule CPH_PyChar_GetRenderTransform(CPH_CharacterObject *self,
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

    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);

    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);

    const auto h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    const auto slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);
    const auto dense = self->world->physics.slot_to_dense[slot];

    const auto shadow_ppos = reinterpret_cast<CPH_PosStride *>(self->world->physics.prev_positions);
    const auto shadow_prot = reinterpret_cast<CPH_AuxStride *>(self->world->physics.prev_rotations);

    JPH_STACK_ALLOC(JPH_RVec3, end_p);
    JPH_STACK_ALLOC(JPH_Quat, end_r);
    JPH_CharacterVirtual_GetPosition(self->character, end_p);
    JPH_CharacterVirtual_GetRotation(self->character, end_r);

    CPH_MathInterpolateCharacterTransform(&shadow_ppos[dense], &shadow_prot[dense], end_p, end_r,
                                          alpha, res_p, res_r);

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    return CPH_PY_BUILD_TUPLE(CPH_PY_BUILD_TUPLE(res_p[0], res_p[1], res_p[2]),
                              CPH_PY_BUILD_TUPLE(res_r[0], res_r[1], res_r[2], res_r[3]));
}

PyCFunction_DeclareMethodFromModule CPH_PyChar_IsGrounded(CPH_CharacterObject *self,
                                                          PyObject *ignored [[maybe_unused]]) {
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    JPH_GroundState state =
        JPH_CharacterBase_GetGroundState(reinterpret_cast<JPH_CharacterBase *>(self->character));
    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    if (state == JPH_GroundState_OnGround) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// NEW: GC Traverse/Clear for Character
PyType_DeclareSlot_StatusFromModule CPH_PyChar_Traverse(CPH_CharacterObject *self, visitproc visit,
                                                        void *arg) {
    Py_VISIT(self->world);
    return 0;
}
PyType_DeclareSlot_StatusFromModule CPH_PyChar_Clear(CPH_CharacterObject *self) {
    Py_CLEAR(self->world);
    return 0;
}
PyType_DeclareSlot_VoidFromModule CPH_PyChar_Dealloc(CPH_CharacterObject *self) {
    PyObject_GC_UnTrack(self);

    uint64_t h_raw = 0;
    uint32_t slot  = 0;
    if (!self->world) {
        goto finalize;
    }

    // --- 1. WORLD DRAIN (Locked) ---
    CPH_SHADOW_LOCK(&self->world->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self->world);
    CPH_BLOCK_UNTIL_NOT_QUERYING(self->world);

    h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    slot  = static_cast<uint32_t>(h_raw & kHandleIndexMask);

    CPH_WorldRemoveBodySlot(&self->world->physics, slot);

    CPH_SHADOW_UNLOCK(&self->world->physics.shadow_lock);

    // --- 2. JOLT DESTRUCTION (Hard Serialized) ---
    CPH_NATIVE_MUTEX_LOCK(self->world->physics.jph_trampoline_lock);

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

    CPH_NATIVE_MUTEX_UNLOCK(self->world->physics.jph_trampoline_lock);

finalize:
    if (self->parsers) {

        delete self->parsers;
    }
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// Helper 1: Jolt-side allocation and Collision Manager linking
static inline JPH_CharacterVirtual *
alloc_j_char(CPH_PhysicsWorldObject *self, CPH_PositionVector pos, CPH_CharacterParams params) {

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

    JPH_RVec3 initial_position{static_cast<double>(pos.px), static_cast<double>(pos.py),
                               static_cast<double>(pos.pz)};
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
static inline void register_char(CPH_PhysicsWorldObject *self, CPH_CharacterObject *obj,
                                 JPH_CharacterVirtual *j_char, uint32_t slot) {
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    uint32_t gen = atomic_load_explicit(&self->physics.generations[slot], memory_order_relaxed);
    BodyHandle h = make_handle(slot, gen);

    uint64_t raw_h = h;
    atomic_store_explicit(&obj->handle, raw_h, memory_order_relaxed);

    auto dense_idx =
        static_cast<uint32_t>(atomic_load_explicit(&self->physics.count, memory_order_relaxed));
    JPH_BodyID bid = JPH_CharacterVirtual_GetInnerBodyID(j_char);
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);

    if (j_idx <= self->physics.max_jolt_bodies) {
        atomic_store_explicit(&self->physics.id_to_handle_map[j_idx], raw_h, memory_order_release);
    }

    self->physics.body_ids[dense_idx]      = bid;
    self->physics.slot_to_dense[slot]      = dense_idx;
    self->physics.dense_to_slot[dense_idx] = slot;
    self->physics.user_data[dense_idx]     = 0;

    constexpr auto kCollisionFilterAllCategories = 0xFFFF;
    constexpr auto kCollisionFilterAllMasks      = 0xFFFF;

    self->physics.categories[dense_idx]   = kCollisionFilterAllCategories;
    self->physics.masks[dense_idx]        = kCollisionFilterAllMasks;
    self->physics.material_ids[dense_idx] = 0;

    JPH_STACK_ALLOC(JPH_RVec3, p);
    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_CharacterVirtual_GetPosition(j_char, p);
    JPH_CharacterVirtual_GetRotation(j_char, q);

    CPH_PosStride p_val = {p->x, p->y, p->z, 0.0};
    CPH_AuxStride r_val = {q->x, q->y, q->z, q->w};

    (reinterpret_cast<CPH_PosStride *>(self->physics.positions))[dense_idx]      = p_val;
    (reinterpret_cast<CPH_PosStride *>(self->physics.prev_positions))[dense_idx] = p_val;
    (reinterpret_cast<CPH_AuxStride *>(self->physics.rotations))[dense_idx]      = r_val;
    (reinterpret_cast<CPH_AuxStride *>(self->physics.prev_rotations))[dense_idx] = r_val;
    (reinterpret_cast<CPH_AuxStride *>(self->physics.linear_velocities))[dense_idx] =
        CPH_AuxStride{};
    (reinterpret_cast<CPH_AuxStride *>(self->physics.angular_velocities))[dense_idx] =
        CPH_AuxStride{};

    if (self->physics.soft_shadows) {
        self->physics.soft_shadows[dense_idx].vertices = nullptr;
    }

    atomic_store_explicit(&self->physics.slot_states[slot], SLOT_CHARACTER, memory_order_release);
    atomic_fetch_add_explicit(&self->physics.count, 1, memory_order_release);

    size_t final_count  = atomic_load_explicit(&self->physics.count, memory_order_relaxed);
    self->view_shape[0] = static_cast<Py_ssize_t>(final_count);

    JPH_BodyInterface_SetUserData(self->physics.body_interface, bid, raw_h);
    JPH_CharacterVirtual_SetUserData(j_char, raw_h);

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
}

// Helper 3: Filter and Listener serialization (Trampoline Lock)
static inline void setup_char_filters(CPH_CharacterObject *obj) {
    CPH_NATIVE_MUTEX_LOCK(obj->world->physics.jph_trampoline_lock);
    obj->listener     = JPH_CharacterContactListener_Create(obj);
    obj->body_filter  = JPH_BodyFilter_Create(nullptr);
    obj->shape_filter = JPH_ShapeFilter_Create(nullptr);
    obj->bp_filter    = JPH_BroadPhaseLayerFilter_Create(nullptr);
    obj->obj_filter   = JPH_ObjectLayerFilter_Create(nullptr);
    CPH_NATIVE_MUTEX_UNLOCK(obj->world->physics.jph_trampoline_lock);
    JPH_CharacterVirtual_SetListener(obj->character, obj->listener);
}
[[nodiscard]]
static inline int setup_char_parsers(CPH_CharacterObject *obj) {
    obj->parsers = CPH::NewParserGroup<CPH_CharacterParsers>();
    if (!obj->parsers) {
        return -1;
    }
    obj->parsers->Init();
    return 0;
}

// Main Orchestrator
PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateCharacter(CPH_PhysicsWorldObject *self,
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

    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self);
    CPH_BLOCK_UNTIL_NOT_QUERYING(self);

    size_t available = atomic_load_explicit(&self->physics.free_count, memory_order_acquire);

    if (available == 0) {
        if (CPH_PyWorld_Resize(self, self->physics.capacity * 2) < 0) {
            CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
            return nullptr;
        }
        available = atomic_load_explicit(&self->physics.free_count, memory_order_acquire);
    }

    uint32_t char_slot = self->physics.free_slots[available - 1];
    atomic_store_explicit(&self->physics.free_count, available - 1, memory_order_release);

    atomic_store_explicit(&self->physics.slot_states[char_slot], SLOT_PENDING_CREATE,
                          memory_order_release);

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    CPH_PositionVector pos_vec      = {pos.x, pos.y, pos.z};
    CPH_CharacterParams char_params = {height, radius, slope};

    CPH_CharacterObject *obj     = nullptr;
    JPH_CharacterVirtual *j_char = alloc_j_char(self, pos_vec, char_params);
    if (!j_char) {
        goto fail_jolt;
    }

    obj = PyObject_GC_New(CPH_CharacterObject,
                          reinterpret_cast<PyTypeObject *>(
                              CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)))->CharacterType));
    if (!obj) {
        goto fail_py;
    }

    if (setup_char_parsers(obj) < 0) {
        goto fail_py;
    }

    obj->world     = reinterpret_cast<CPH_PhysicsWorldObject *>(Py_NewRef(self));
    obj->character = j_char;
    obj->prev_px   = pos.x;
    obj->prev_py   = pos.y;
    obj->prev_pz   = pos.z;

    obj->prev_rx = 0.0f;
    obj->prev_ry = 0.0f;
    obj->prev_rz = 0.0f;
    obj->prev_rw = 1.0f;

    atomic_init(&obj->push_strength, 0.0f);
    atomic_init(&obj->last_vx, 0.0f);
    atomic_init(&obj->last_vy, 0.0f);
    atomic_init(&obj->last_vz, 0.0f);

    register_char(self, obj, j_char, char_slot);
    setup_char_filters(obj);

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);

fail_py:
    JPH_CharacterBase_Destroy(reinterpret_cast<JPH_CharacterBase *>(j_char));
fail_jolt:
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    atomic_store_explicit(&self->physics.slot_states[char_slot], SLOT_EMPTY, memory_order_relaxed);

    size_t f_idx = atomic_fetch_add_explicit(&self->physics.free_count, 1, memory_order_relaxed);
    self->physics.free_slots[f_idx] = char_slot;

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
    return nullptr;
}

PyGetSet_DeclareGetter CPH_PyChar_GetHandle(CPH_CharacterObject *self,
                                            [[maybe_unused]] void *closure) {
    uint64_t raw_h = atomic_load_explicit(&self->handle, memory_order_relaxed);
    return CPH_PY_BUILD_VALUE(raw_h);
}

#define CPH_CHAR_FASTCALL(fn, name) CPH_FEAT(fn, name, METH_FASTCALL | METH_KEYWORDS)
#define CPH_CHAR_NOARGS(fn, name) CPH_FEAT(fn, name, METH_NOARGS)
#define CPH_CHAR_O(fn, name) CPH_FEAT(fn, name, METH_O)

static PyMethodDef kCharacterMethods[] = {

    CPH_CHAR_FASTCALL(CPH_PyChar_Move, move),
    CPH_CHAR_NOARGS(CPH_PyChar_GetPosition, get_position),
    CPH_CHAR_NOARGS(CPH_PyChar_GetLinearVelocity, get_linear_velocity),
    CPH_CHAR_FASTCALL(CPH_PyChar_SetPosition, set_position),
    CPH_CHAR_FASTCALL(CPH_PyChar_SetRotation, set_rotation),
    CPH_CHAR_NOARGS(CPH_PyChar_IsGrounded, is_grounded),
    CPH_CHAR_FASTCALL(CPH_PyChar_SetStrength, set_strength),
    CPH_CHAR_O(CPH_PyChar_GetRenderTransform, get_render_transform),
    {nullptr, nullptr, 0, nullptr}};

PyType_Spec Character_spec = {
    .name      = "culverin._culverin_c.Character",
    .basicsize = sizeof(CPH_CharacterObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(CPH_PyChar_Dealloc)},
            {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(CPH_PyChar_Traverse)},
            {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(CPH_PyChar_Clear)},
            {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kCharacterMethods)},
            {.slot = Py_tp_getset,
             .pfunc =
                 (PyGetSetDef[]){

                     CPH_GETSET("handle", CPH_PyChar_GetHandle), {}

                 }},
            {}

        },
};
