#include "culverin_character.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_fast_build.h"
#include "culverin_physics_sync.h"
#include "culverin_physics_world_internal.h"
#include "culverin_types.h"

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL char_on_contact_validate(
    CULV_MAYBE_UNUSED void *userData, CULV_MAYBE_UNUSED const JPH_CharacterVirtual *character,
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    CULV_MAYBE_UNUSED JPH_BodyID bodyID2, CULV_MAYBE_UNUSED JPH_SubShapeID subShapeID2) {
    return true; // Usually true, unless you want to walk through certain bodies
}

static void record_character_contact(CharacterObject *self, JPH_BodyID bodyID2,
                                     const JPH_RVec3 *pos, const JPH_Vec3 *norm,
                                     ContactEventType type) {
    auto world    = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    BodyHandle h2  = 0;

    if (world->id_to_handle_map && j_idx <= world->max_jolt_bodies) {
        // TSan Fix: Atomic load from the shared handle map.
        // Acquire ensures we see the body initialization if it was just created.
        h2 = atomic_load_explicit(&world->id_to_handle_map[j_idx], memory_order_acquire);
    }

    // Extract raw value for liveness check
    uint64_t h2_raw = h2;
    if (h2_raw == 0) {
        return;
    }

    // TSan Fix: Atomic load of the character's own handle
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);

    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = (uint32_t)type;

        // TSan Fix: Use raw values for canonical handle ordering
        // canonicalize and store using the raw values we already loaded
        if (h1_raw < h2_raw) {
            // TSan Fix: Store base uint64_t values to avoid implicit seq_cst loads
            atomic_store_explicit(&ev->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&ev->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h1_raw, memory_order_relaxed);
        }

        // Geometry (Standard floats are safe to store here as world.step() owns the buffer)
        ev->px               = (float)pos->x;
        ev->py               = (float)pos->y;
        ev->pz               = (float)pos->z;
        ev->nx               = norm->x;
        ev->ny               = norm->y;
        ev->nz               = norm->z;
        ev->impulse          = 1.0f;
        ev->sliding_speed_sq = 0.0f;

        // Metadata lookups
        auto slot2      = (uint32_t)(h2_raw & HANDLE_INDEX_MASK);
        uint32_t dense2 = world->slot_to_dense[slot2];
        ev->mat1        = 0;
        ev->mat2        = world->material_ids[dense2];

        // Release fence ensures all previous stores are visible to Python when it reads
        // contact_atomic_idx
        atomic_thread_fence(memory_order_release);
    }
}

static void report_char_vs_char(CharacterObject *self, const JPH_CharacterVirtual *other,
                                const JPH_Vec3 *normal, const JPH_RVec3 *pos,
                                ContactEventType type) {
    auto world     = self->world;
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

    // FIX: Retrieve the handle directly from the other character's UserData.
    // This is 100% reliable and avoids map boundary checks.
    uint64_t h2_raw = JPH_CharacterVirtual_GetUserData(other);

    if (h2_raw == 0) {
        return; // Still 0? Probably a Jolt internal body.
    }

    // Initialize atomic local handle for the other character
    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = (uint32_t)type;

        // TSan Fix: Use raw values for canonical ordering check
        // canonicalize and store using the raw values we already loaded
        if (h1_raw < h2_raw) {
            // TSan Fix: Store base uint64_t values to avoid implicit seq_cst loads
            atomic_store_explicit(&ev->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&ev->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h1_raw, memory_order_relaxed);
        }

        ev->sliding_speed_sq = 0.0f;
        ev->nx               = normal->x;
        ev->ny               = normal->y;
        ev->nz               = normal->z;
        ev->px               = (float)pos->x;
        ev->py               = (float)pos->y;
        ev->pz               = (float)pos->z;
        ev->impulse          = 1.0f;
        ev->mat1             = 0;
        ev->mat2             = 0;

        // Release visibility of the whole event to Python
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_character_contact_added(
    void *userData, CULV_MAYBE_UNUSED const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, CULV_MAYBE_UNUSED JPH_SubShapeID subShapeID2,
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = (CharacterObject *)userData;
    if (!self || !self->world) {
        return;
    }

    report_char_vs_char(self, otherCharacter, contactNormal, contactPosition, EVENT_ADDED);
}

static void apply_character_impulse(CharacterObject *self, JPH_BodyID bodyID2,
                                    const JPH_Vec3 *contactNormal) {
    // 1. Thread-Safe Member Access
    float vx       = atomic_load_explicit((&self->last_vx), memory_order_relaxed);
    float vy       = atomic_load_explicit((&self->last_vy), memory_order_relaxed);
    float vz       = atomic_load_explicit((&self->last_vz), memory_order_relaxed);
    float strength = atomic_load_explicit((&self->push_strength), memory_order_relaxed);

    JPH_BodyInterface *bi = self->world->body_interface;

    // 2. Ignore Sensors & Non-Dynamic Bodies
    if ((int)JPH_BodyInterface_IsSensor(bi, bodyID2) ||
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

// --- Updated Added Callback ---
static void JPH_API_CALL char_on_contact_added(
    void *userData, CULV_MAYBE_UNUSED const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    CULV_MAYBE_UNUSED JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = (CharacterObject *)userData;
    if (!self) {
        return;
    }

    // Record Event
    record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_ADDED);

    // Apply Impulse
    apply_character_impulse(self, bodyID2, contactNormal);
}

static void JPH_API_CALL char_on_contact_persisted(
    void *userData, CULV_MAYBE_UNUSED const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    CULV_MAYBE_UNUSED JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto self = (CharacterObject *)userData;
    if (!self) {
        return;
    }

    // Record Event
    record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_PERSISTED);

    // Apply Impulse (CRITICAL FIX)
    apply_character_impulse(self, bodyID2, contactNormal);
}

static void JPH_API_CALL char_on_contact_removed(void *userData,
                                                 const JPH_CharacterVirtual *Py_UNUSED(character),
                                                 JPH_BodyID bodyID2,
                                                 JPH_SubShapeID Py_UNUSED(subShapeID2)) {
    auto self = (CharacterObject *)userData;
    if (!self || !self->world) {
        return;
    }

    PhysicsWorldObject *world = self->world;
    uint32_t j_idx            = JPH_ID_TO_INDEX(bodyID2);

    // TSan Fix: Explicitly load raw values to avoid seq_cst overhead
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint64_t h2_raw = 0;

    if (world->id_to_handle_map && j_idx <= world->max_jolt_bodies) {
        // TSan Fix: Use acquire to see initialization data if body was just created
        h2_raw = atomic_load_explicit(&world->id_to_handle_map[j_idx], memory_order_acquire);
    }

    if (h2_raw == 0) {
        return;
    }

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = EVENT_REMOVED;

        // TSan Fix: Canonicalize using raw registers
        if (h1_raw < h2_raw) {
            atomic_store_explicit(&ev->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&ev->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h1_raw, memory_order_relaxed);
        }

        // Geometry is zeroed for removal (Standard floats)
        memset(&ev->px, 0, sizeof(float) * 8);

        // Finalize visibility for Python readers
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_character_contact_persisted(
    void *userData, const JPH_CharacterVirtual *Py_UNUSED(character),
    const JPH_CharacterVirtual *otherCharacter, JPH_SubShapeID Py_UNUSED(subShapeID2),
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) {
        return;
    }

    report_char_vs_char(self, otherCharacter, contactNormal, contactPosition, EVENT_PERSISTED);
}

static void JPH_API_CALL char_on_character_contact_removed(
    void *userData, const JPH_CharacterVirtual *Py_UNUSED(character),
    const JPH_CharacterID otherCharacterID, JPH_SubShapeID Py_UNUSED(subShapeID2)) {
    auto self  = (CharacterObject *)userData;
    auto world = self->world;

    // TSan Fix: Explicit relaxed load of own handle
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);

    // We have to use the CharacterID to find the handle.
    uint32_t j_idx  = JPH_ID_TO_INDEX(otherCharacterID);
    uint64_t h2_raw = 0;

    if (world->id_to_handle_map && j_idx <= world->max_jolt_bodies) {
        // TSan Fix: Atomic load from map with acquire semantics
        h2_raw = atomic_load_explicit(&world->id_to_handle_map[j_idx], memory_order_acquire);
    }

    if (h2_raw == 0) {
        return;
    }

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = EVENT_REMOVED;

        // TSan Fix: Canonicalize using standard integer registers
        if (h1_raw < h2_raw) {
            atomic_store_explicit(&ev->body1, h1_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h2_raw, memory_order_relaxed);
        } else {
            atomic_store_explicit(&ev->body1, h2_raw, memory_order_relaxed);
            atomic_store_explicit(&ev->body2, h1_raw, memory_order_relaxed);
        }

        // Finalize event publication
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_adjust_velocity(void *userData,
                                                 const JPH_CharacterVirtual *character,
                                                 const JPH_Body *body2, JPH_Vec3 *ioLinearVelocity,
                                                 JPH_Vec3 *ioAngularVelocity) {

    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) {
        return;
    }

    // 1. Get Platform/Friction Data
    uint64_t h2_raw = JPH_Body_GetUserData((JPH_Body *)body2);
    float friction  = 1.0f;
    uint32_t slot2  = (uint32_t)(h2_raw & HANDLE_INDEX_MASK);
    if (slot2 < self->world->slot_capacity) {
        uint32_t dense2 = self->world->slot_to_dense[slot2];
        uint32_t mat_id = self->world->material_ids[dense2];
        for (size_t i = 0; i < self->world->material_count; i++) {
            if (self->world->materials[i].id == mat_id) {
                friction = self->world->materials[i].friction;
                break;
            }
        }
    }

    // 2. Calculate Tangential Velocity (v = omega x r)
    JPH_Vec3 omega;
    JPH_Body_GetAngularVelocity((JPH_Body *)body2, &omega);

    JPH_RVec3 char_pos;
    JPH_RVec3 plat_pos;
    JPH_CharacterVirtual_GetPosition((JPH_CharacterVirtual *)character, &char_pos);
    JPH_Body_GetPosition((JPH_Body *)body2, &plat_pos);

    // Relative offset
    float rx = (float)(char_pos.x - plat_pos.x);
    float rz = (float)(char_pos.z - plat_pos.z);

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
    void *userData, CULV_MAYBE_UNUSED const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, CULV_MAYBE_UNUSED JPH_SubShapeID subShapeID2) {

    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) {
        return true;
    }

    PhysicsWorldObject *world = self->world;

    // 1. Get Culverin Handles
    // h1: self (stored on the CharacterObject)
    // h2: other (stored in JPH UserData, which we set in register_char)
    uint64_t h1_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint64_t h2_raw = JPH_CharacterVirtual_GetUserData(otherCharacter);

    if (h2_raw == 0) {
        return true; // Collide by default if handle is missing
    }

    // 2. Resolve Dense Indices for filter lookup
    uint32_t slot1 = (uint32_t)(h1_raw & HANDLE_INDEX_MASK);
    uint32_t slot2 = (uint32_t)(h2_raw & HANDLE_INDEX_MASK);

    // Safety check for array bounds
    if (slot1 >= world->slot_capacity || slot2 >= world->slot_capacity) {
        return true;
    }

    uint32_t idx1 = world->slot_to_dense[slot1];
    uint32_t idx2 = world->slot_to_dense[slot2];

    // 3. Perform Bitmask Filtering
    uint32_t cat1  = world->categories[idx1];
    uint32_t mask1 = world->masks[idx1];
    uint32_t cat2  = world->categories[idx2];
    uint32_t mask2 = world->masks[idx2];

    // Reject if either mask blocks the other's category
    return ((cat1 & mask2) && (cat2 & mask1)) != 0;
}

// High-frequency callback: DO NOT allocate memory or lock.
CULV_NO_TSAN CULV_MAYBE_UNUSED static void JPH_API_CALL char_on_contact_solve(
    CULV_MAYBE_UNUSED void *userData, CULV_MAYBE_UNUSED const JPH_Body *body1,
    CULV_MAYBE_UNUSED const JPH_Body *body2, CULV_MAYBE_UNUSED const JPH_ContactManifold *manifold,
    CULV_MAYBE_UNUSED JPH_ContactSettings *settings) {
    // 1. You can access userData (PhysicsWorldObject*) here if you need
    // to look up global world parameters.

    // 2. You can override friction/restitution dynamically.
    // Example: If you wanted to make a contact "slippery" based on its depth
    // float depth = JPH_ContactManifold_GetContactPointOn1(manifold, 0, ...) -> Z
    // if (depth < -0.1f) settings->mFriction = 0.0f;

    // 3. For now, leave it empty or add logging/metrics if needed.
}

// Map the procs
const JPH_CharacterContactListener_Procs char_listener_procs = {
    .OnAdjustBodyVelocity        = char_on_adjust_velocity,
    .OnContactValidate           = char_on_contact_validate,
    .OnCharacterContactValidate  = char_on_character_contact_validate,
    .OnContactAdded              = char_on_contact_added,
    .OnContactPersisted          = char_on_contact_persisted,
    .OnContactRemoved            = char_on_contact_removed,
    .OnCharacterContactAdded     = char_on_character_contact_added,
    .OnCharacterContactPersisted = char_on_character_contact_persisted,
    .OnCharacterContactRemoved   = char_on_character_contact_removed,
    .OnContactSolve              = nullptr, // Advanced solver override
    .OnCharacterContactSolve     = nullptr  // Advanced solver override
};

PyCFunction_DeclareMethodFromModule Character_move(CharacterObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. INTEGRATED FAST PARSE (Unchanged)
    Vec3f v_in = {.x = 0.0f, .y = 0.0f, .z = 0.0f};
    float dt   = 0.0f;

    void *targets[CharMove_COUNT] = {[IDX_CM_VEL] = &v_in, [IDX_CM_DT] = &dt};

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CharMoveParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(v_in.x, v_in.y, v_in.z, "Character velocity");

    // 2. SNAPSHOT (Shadow Lock)
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // TSan Fix: Explicit relaxed stores for worker-thread inputs
    atomic_store_explicit(&self->last_vx, v_in.x, memory_order_relaxed);
    atomic_store_explicit(&self->last_vy, v_in.y, memory_order_relaxed);
    atomic_store_explicit(&self->last_vz, v_in.z, memory_order_relaxed);

    // TSan Fix: Load handle atomically to resolve indices
    uint64_t h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint32_t slot  = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t dense = self->world->slot_to_dense[slot];

    // Snapshot current state to prev for interpolation
    auto shadow_pos  = (PosStride *)self->world->positions;
    auto shadow_ppos = (PosStride *)self->world->prev_positions;
    auto shadow_rot  = (AuxStride *)self->world->rotations;
    auto shadow_prot = (AuxStride *)self->world->prev_rotations;

    shadow_ppos[dense] = shadow_pos[dense];
    shadow_prot[dense] = shadow_rot[dense];

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // 3. JOLT EXECUTION
    JPH_Vec3 j_v = {v_in.x, v_in.y, v_in.z};

    // Inherit velocity from ground platform if grounded
    JPH_BodyID ground_id = JPH_CharacterBase_GetGroundBodyId((JPH_CharacterBase *)self->character);

    if (ground_id != JPH_INVALID_BODY_ID) {
        JPH_Vec3 ground_vel;
        JPH_CharacterBase_GetGroundVelocity((JPH_CharacterBase *)self->character, &ground_vel);
        j_v.x += ground_vel.x;
        // j_v.y += ground_vel.y;
        j_v.z += ground_vel.z;
    }

    JPH_CharacterVirtual_SetLinearVelocity(self->character, &j_v);

    JPH_ExtendedUpdateSettings update_settings       = {}; // Guaranteed zero-init
    update_settings.stickToFloorStepDown             = (JPH_Vec3){0.0f, -0.5f, 0.0f};
    update_settings.walkStairsStepUp                 = (JPH_Vec3){0.0f, 0.4f, 0.0f};
    update_settings.walkStairsMinStepForward         = 0.02f;
    update_settings.walkStairsStepForwardTest        = 0.15f;
    update_settings.walkStairsCosAngleForwardContact = 0.996f;

    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    Py_BEGIN_ALLOW_THREADS JPH_CharacterVirtual_ExtendedUpdate(
        self->character, dt, &update_settings, OBJECT_LAYER_DYNAMIC, self->world->system,
        self->body_filter, self->shape_filter);
    Py_END_ALLOW_THREADS NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // 4. POST-MOVE SYNC
    SHADOW_LOCK(&self->world->shadow_lock);

    JPH_STACK_ALLOC(JPH_RVec3, current_p);
    JPH_STACK_ALLOC(JPH_Quat, current_r);
    JPH_CharacterVirtual_GetPosition(self->character, current_p);
    JPH_CharacterVirtual_GetRotation(self->character, current_r);

    // Sync non-atomic shadow buffers
    shadow_pos[dense] = (PosStride){current_p->x, current_p->y, current_p->z, 0.0};
    shadow_rot[dense] = (AuxStride){current_r->x, current_r->y, current_r->z, current_r->w};

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Character_get_position(CharacterObject *self,
                                                           PyObject *Py_UNUSED(ignored)) {
    // 1. Aligned stack storage for SIMD
    JPH_STACK_ALLOC(JPH_RVec3, pos);

    // 2. Lock for consistency (ensure we aren't reading mid-step)
    SHADOW_LOCK(&self->world->shadow_lock);
    JPH_CharacterVirtual_GetPosition(self->character, pos);
    SHADOW_UNLOCK(&self->world->shadow_lock);

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

PyCFunction_DeclareMethodFromModule Character_set_position(CharacterObject *self,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Stack Allocation and Parser Targets (Unchanged)
    PosStride pos                   = {};
    void *targets[SetPosChar_COUNT] = {
        [IDX_SPC_POS] = (void *)&pos,
    };

    // 2. High Speed Parse (Unchanged)
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetPosCharParser, targets)) {
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // 3. Update Jolt
    JPH_RVec3 j_pos = {pos.x, pos.y, pos.z};
    JPH_CharacterVirtual_SetPosition(self->character, &j_pos);

    // 4. Update Shadow Buffers (ATOMIC REFACTOR)

    // TSan Fix: Load handle atomically to resolve mapping
    uint64_t h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint32_t slot  = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t dense = self->world->slot_to_dense[slot];

    // Map world buffers using Stride types for better alignment and readability
    auto shadow_pos  = (PosStride *)self->world->positions;
    auto shadow_ppos = (PosStride *)self->world->prev_positions;

    // Reset both current and previous to the new position.
    // This prevents "teleport streaks" by forcing LERP(prev, curr, alpha) to return exactly 'pos'.
    PosStride p_val    = {pos.x, pos.y, pos.z, 0.0};
    shadow_pos[dense]  = p_val;
    shadow_ppos[dense] = p_val;

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Character_set_rotation(CharacterObject *self,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Explicitly initialized stack target
    AuxStride rot = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    void *targets[SetRotChar_COUNT] = {
        [IDX_SRC_ROT] = (void *)&rot,
    };

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetRotCharParser, targets)) {
        return nullptr;
    }

    // --- PHYSICS LOGIC ---
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // 3. Update Jolt
    // We can cast our AuxStride directly to JPH_Quat if they are layout-compatible
    // or just initialize a temporary.
    JPH_Quat q = {.x = rot.x, .y = rot.y, .z = rot.z, .w = rot.w};
    JPH_CharacterVirtual_SetRotation(self->character, &q);

    uint64_t raw_h = atomic_load_explicit(&self->handle, memory_order_relaxed);

    // 4. Update Shadow Buffers (Zero-Streak Reset)
    auto slot          = (uint32_t)(raw_h & HANDLE_INDEX_MASK);
    uint32_t dense_idx = self->world->slot_to_dense[slot];
    size_t off         = (size_t)dense_idx * 4;

    // Direct 16-byte copies to update current and previous states
    memcpy(&self->world->rotations[off], &q, 16);
    memcpy(&self->world->prev_rotations[off], &q, 16);

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Character_set_strength(CharacterObject *self,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Explicitly initialized target
    float strength = 0.0f;

    void *targets[SetStrengthChar_COUNT] = {
        [IDX_SSC_STRENGTH] = (void *)&strength,
    };

    // 2. High Speed Parse (Handles both world.set_strength(200.0) and
    // world.set_strength(strength=200.0))
    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.SetStrengthCharParser, targets)) {
        return nullptr;
    }

    // --- CONCURRENCY & LOCKING ---
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);
    BLOCK_UNTIL_NOT_QUERYING(self->world);

    // 3. Update Atomic for Jolt worker threads (Physics side)
    atomic_store_explicit(&self->push_strength, strength, memory_order_relaxed);

    // 4. Update Jolt internal state (Direct state change)
    JPH_CharacterVirtual_SetMaxStrength(self->character, strength);

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

// Change signature to take PyObject* arg directly
PyCFunction_DeclareMethodFromModule Character_get_render_transform(CharacterObject *self,
                                                                   PyObject *arg) {
    // --- 1. Fast Argument Parsing ---
    double alpha_dbl = PyFloat_AsDouble(arg);
    if (alpha_dbl == -1.0 && PyErr_Occurred()) {
        return nullptr;
    }

    // Clamp alpha to [0.0, 1.0]
    JPH_Real alpha = fmax(0.0, fmin(1.0, (JPH_Real)alpha_dbl));

    // --- 2. Snapshot State (Locked) ---
    SHADOW_LOCK(&self->world->shadow_lock);

    // Consistency Guard: Ensure we aren't reading while the world is stepping/swapping
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // TSan Fix: Explicitly load raw handle value to resolve slot
    uint64_t h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint32_t slot  = (uint32_t)(h_raw & HANDLE_INDEX_MASK);
    uint32_t dense = self->world->slot_to_dense[slot];

    // Map world buffers using Strides
    auto shadow_ppos = (PosStride *)self->world->prev_positions;
    auto shadow_prot = (AuxStride *)self->world->prev_rotations;

    // Capture "Start" state (Previous frame)
    PosStride start_p = shadow_ppos[dense];
    AuxStride start_r = shadow_prot[dense];

    // Capture "End" state (Current frame from Jolt)
    JPH_STACK_ALLOC(JPH_RVec3, end_p);
    JPH_STACK_ALLOC(JPH_Quat, end_r);
    JPH_CharacterVirtual_GetPosition(self->character, end_p);
    JPH_CharacterVirtual_GetRotation(self->character, end_r);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // --- 3. MATH (Unlocked) ---

    // Position LERP (Performed in DOUBLE to prevent jitter far from origin)
    JPH_Real px = start_p.x + (end_p->x - start_p.x) * alpha;
    JPH_Real py = start_p.y + (end_p->y - start_p.y) * alpha;
    JPH_Real pz = start_p.z + (end_p->z - start_p.z) * alpha;

    // Rotation NLERP (Performed in FLOAT)
    float p_rx = start_r.x;
    float p_ry = start_r.y;
    float p_rz = start_r.z;
    float p_rw = start_r.w;

    float dot = p_rx * end_r->x + p_ry * end_r->y + p_rz * end_r->z + p_rw * end_r->w;
    float q2x = end_r->x;
    float q2y = end_r->y;
    float q2z = end_r->z;
    float q2w = end_r->w;

    // Correct for quaternion double-cover (shortest path)
    if (dot < 0.0f) {
        q2x = -q2x;
        q2y = -q2y;
        q2z = -q2z;
        q2w = -q2w;
    }

    float rx = p_rx + (q2x - p_rx) * (float)alpha;
    float ry = p_ry + (q2y - p_ry) * (float)alpha;
    float rz = p_rz + (q2z - p_rz) * (float)alpha;
    float rw = p_rw + (q2w - p_rw) * (float)alpha;

    // Re-normalize to ensure the result is a unit quaternion
    float mag_sq  = rx * rx + ry * ry + rz * rz + rw * rw;
    float inv_len = (mag_sq > 1e-9f) ? 1.0f / sqrtf(mag_sq) : 1.0f;
    rx *= inv_len;
    ry *= inv_len;
    rz *= inv_len;
    rw *= inv_len;

    // --- 4. Optimized Return (Using FastBuild) ---
    return FastBuild_Tuple(FastBuild_Tuple(px, py, pz), FastBuild_Tuple(rx, ry, rz, rw));
}

PyCFunction_DeclareMethodFromModule Character_is_grounded(CharacterObject *self,
                                                          PyObject *Py_UNUSED(ignored)) {
    SHADOW_LOCK(&self->world->shadow_lock);

    /*
       Jolt C-API: CharacterVirtual is a subclass of CharacterBase.
       We cast the virtual instance to the base pointer to check the ground state.
    */
    JPH_GroundState state = JPH_CharacterBase_GetGroundState((JPH_CharacterBase *)self->character);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // Return True only if supported by walkable ground
    if (state == JPH_GroundState_OnGround) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// NEW: GC Traverse/Clear for Character
PyType_DeclareSlot_StatusFromModule Character_traverse(CharacterObject *self, visitproc visit,
                                                       void *arg) {
    Py_VISIT(self->world);
    return 0;
}
PyType_DeclareSlot_StatusFromModule Character_clear(CharacterObject *self) {
    Py_CLEAR(self->world);
    return 0;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_VoidFromModule Character_dealloc(CharacterObject *self) {
    PyObject_GC_UnTrack(self);

    if (!self->world) {
        goto finalize;
    }

    // --- 1. WORLD DRAIN (Locked) ---
    SHADOW_LOCK(&self->world->shadow_lock);

    // Guard: Ensure we aren't mid-step or mid-query
    BLOCK_UNTIL_NOT_STEPPING(self->world);
    BLOCK_UNTIL_NOT_QUERYING(self->world);

    // TSan Fix: Explicitly load raw handle to extract the slot index
    uint64_t h_raw = atomic_load_explicit(&self->handle, memory_order_relaxed);
    uint32_t slot  = (uint32_t)(h_raw & HANDLE_INDEX_MASK);

    // Note: world_remove_body_slot handles atomic count/state updates internally
    world_remove_body_slot(self->world, slot);

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // --- 2. JOLT DESTRUCTION (Hard Serialized) ---
    // Protect Jolt's callback registry during destruction
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);

    if (self->world->char_vs_char_manager && self->character) {
        JPH_CharacterVsCharacterCollisionSimple_RemoveCharacter(self->world->char_vs_char_manager,
                                                                self->character);
    }

    if (self->character) {
        JPH_CharacterBase_Destroy((JPH_CharacterBase *)self->character);
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

    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

finalize:
    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// Helper 1: Jolt-side allocation and Collision Manager linking
static JPH_CharacterVirtual *
alloc_j_char(PhysicsWorldObject *self, PositionVector pos,
             CharacterParams params) { // Reduced to 2 conceptual arguments

    // Position parameters are now accessed via pos.px, pos.py, pos.pz
    // Size parameters are now accessed via params.height, params.radius, etc.

    float half_h                 = fmaxf((params.height - 2.0f * params.radius) * 0.5f, 0.1f);
    JPH_CapsuleShapeSettings *ss = JPH_CapsuleShapeSettings_Create(half_h, params.radius);
    auto shape                  = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(ss);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);
    if (!shape) {
        return nullptr;
    }

    JPH_CharacterVirtualSettings settings;
    JPH_CharacterVirtualSettings_Init(&settings);
    settings.base.shape                         = shape;
    static constexpr float DegreesPerSemiCircle = 180.0f;
    settings.base.maxSlopeAngle = params.max_slope * (JPH_M_PI / DegreesPerSemiCircle);

    JPH_CharacterVirtual *j_char = JPH_CharacterVirtual_Create(
        &settings, &(JPH_RVec3){(double)pos.px, (double)pos.py, (double)pos.pz},
        &(JPH_Quat){0, 0, 0, 1}, 1, self->system);

    JPH_Shape_Destroy(shape);
    if (!j_char) {
        return nullptr;
    }

    if (self->char_vs_char_manager) {
        JPH_CharacterVsCharacterCollisionSimple_AddCharacter(self->char_vs_char_manager, j_char);
        JPH_CharacterVirtual_SetCharacterVsCharacterCollision(j_char, self->char_vs_char_manager);
    }
    return j_char;
}

// Helper 2: Shadow Buffer Registration (Atomic Commit)
static void register_char(PhysicsWorldObject *self, CharacterObject *obj,
                          JPH_CharacterVirtual *j_char, uint32_t slot) {
    SHADOW_LOCK(&self->shadow_lock);

    // 1. Resolve Handle Metadata
    uint32_t gen = atomic_load_explicit(&self->generations[slot], memory_order_relaxed);
    BodyHandle h = make_handle(slot, gen);

    // TSan Fix: Explicitly load the raw uint64_t once to use for all assignments
    uint64_t raw_h = h;

    // TSan Fix: Use raw_h to avoid implicit seq_cst load on Character object
    atomic_store_explicit(&obj->handle, raw_h, memory_order_relaxed);

    // 2. Dense Mapping
    auto dense_idx = (uint32_t)atomic_load_explicit(&self->count, memory_order_relaxed);
    JPH_BodyID bid = JPH_CharacterVirtual_GetInnerBodyID(j_char);
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);

    if (j_idx <= self->max_jolt_bodies) {
        // TSan Fix: Use raw_h to avoid implicit seq_cst load on shared world map
        // Release ensures that when a thread reads this handle, it sees the shadow buffer writes
        // below.
        atomic_store_explicit(&self->id_to_handle_map[j_idx], raw_h, memory_order_release);
    }

    // 3. Update Non-Atomic Shadow Buffers
    self->body_ids[dense_idx]      = bid;
    self->slot_to_dense[slot]      = dense_idx;
    self->dense_to_slot[dense_idx] = slot;
    self->user_data[dense_idx]     = 0;

    constexpr auto COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
    constexpr auto COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

    // Initialize Metadata (Fixes the collision/settling failure)
    self->categories[dense_idx]   = COLLISION_FILTER_ALL_CATEGORIES;
    self->masks[dense_idx]        = COLLISION_FILTER_ALL_MASKS;
    self->material_ids[dense_idx] = 0;

    // Fetch actual Jolt transform to populate shadow buffers
    JPH_STACK_ALLOC(JPH_RVec3, p);
    JPH_STACK_ALLOC(JPH_Quat, q);
    JPH_CharacterVirtual_GetPosition(j_char, p);
    JPH_CharacterVirtual_GetRotation(j_char, q);

    PosStride p_val = {p->x, p->y, p->z, 0.0};
    AuxStride r_val = {q->x, q->y, q->z, q->w};

    // 3. Populate shadow buffers (Prevents teleport streaks and garbage reads)
    ((PosStride *)self->positions)[dense_idx]          = p_val;
    ((PosStride *)self->prev_positions)[dense_idx]     = p_val;
    ((AuxStride *)self->rotations)[dense_idx]          = r_val;
    ((AuxStride *)self->prev_rotations)[dense_idx]     = r_val;
    ((AuxStride *)self->linear_velocities)[dense_idx]  = (AuxStride){0};
    ((AuxStride *)self->angular_velocities)[dense_idx] = (AuxStride){0};

    if (self->soft_shadows) {
        self->soft_shadows[dense_idx].vertices = nullptr;
    }

    // 4. Atomic Publication
    // Publish slot state first (Release semantics)
    atomic_store_explicit(&self->slot_states[slot], SLOT_CHARACTER, memory_order_release);

    // Then increment total world count (Release semantics)
    atomic_fetch_add_explicit(&self->count, 1, memory_order_release);

    // Finalize view metadata for Python memoryview consistency
    size_t final_count  = atomic_load_explicit(&self->count, memory_order_relaxed);
    self->view_shape[0] = (Py_ssize_t)final_count;

    // 5. Jolt Sync
    JPH_BodyInterface_SetUserData(self->body_interface, bid, raw_h);
    JPH_CharacterVirtual_SetUserData(j_char, raw_h);

    SHADOW_UNLOCK(&self->shadow_lock);
}

// Helper 3: Filter and Listener serialization (Trampoline Lock)
static void setup_char_filters(CharacterObject *obj) {
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock); 
    obj->listener     = JPH_CharacterContactListener_Create(obj);
    obj->body_filter  = JPH_BodyFilter_Create(nullptr);
    obj->shape_filter = JPH_ShapeFilter_Create(nullptr);
    obj->bp_filter    = JPH_BroadPhaseLayerFilter_Create(nullptr);
    obj->obj_filter   = JPH_ObjectLayerFilter_Create(nullptr);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    JPH_CharacterVirtual_SetListener(obj->character, obj->listener);
}

// Main Orchestrator
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_character(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  Py_ssize_t nargs,
                                                                  PyObject *kwnames) {
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    // 1. Setup Targets (Unchanged)
    PosStride pos                 = {.x = 0, .y = 0, .z = 0};
    constexpr auto DEFAULT_HEIGHT = 1.8f;
    constexpr auto DEFAULT_RADIUS = 0.4f;
    constexpr auto DEFUALT_STEP_H = 0.4f;
    constexpr auto DEFAULT_SLOPE  = 45.0f;
    float height                  = DEFAULT_HEIGHT;
    float radius                  = DEFAULT_RADIUS;
    float step_h                  = DEFUALT_STEP_H;
    float slope                   = DEFAULT_SLOPE;

    void *targets[CreateChar_COUNT] = {[IDX_CCHAR_POS]   = (void *)&pos,
                                       [IDX_CCHAR_H]     = (void *)&height,
                                       [IDX_CCHAR_R]     = (void *)&radius,
                                       [IDX_CCHAR_STEP]  = (void *)&step_h,
                                       [IDX_CCHAR_SLOPE] = (void *)&slope};

    if (!FastParse_Unified(args, nargs, kwnames, &st->parsers.CreateCharParser, targets)) {
        return nullptr;
    }

    // --- 2. WORLD LOGIC (ATOMIC REFACTOR) ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // TSan Fix: Atomic load of availability
    size_t available = atomic_load_explicit(&self->free_count, memory_order_acquire);

    if (available == 0) {
        if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return nullptr;
        }
        // Re-read available count after resize
        available = atomic_load_explicit(&self->free_count, memory_order_acquire);
    }

    // Pop from free stack and update count atomically
    uint32_t char_slot = self->free_slots[available - 1];
    atomic_store_explicit(&self->free_count, available - 1, memory_order_release);

    // Transition state atomically to prevent races with is_alive()
    atomic_store_explicit(&self->slot_states[char_slot], SLOT_PENDING_CREATE, memory_order_release);

    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. Resource Allocation (Unchanged)
    PositionVector pos_vec      = {pos.x, pos.y, pos.z};
    CharacterParams char_params = {height, radius, slope};

    JPH_CharacterVirtual *j_char = alloc_j_char(self, pos_vec, char_params);
    if (!j_char) {
        goto fail_jolt;
    }

    auto obj = (CharacterObject *)PyObject_GC_New(
        CharacterObject,
        (PyTypeObject *)get_culverin_state(PyType_GetModule(Py_TYPE(self)))->CharacterType);
    if (!obj) {
        goto fail_py;
    }

    obj->world     = (PhysicsWorldObject *)Py_NewRef(self);
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

    // Note: register_char has been refactored to handle atomic count/state updates
    register_char(self, obj, j_char, char_slot);
    setup_char_filters(obj);

    PyObject_GC_Track((PyObject *)obj);
    return (PyObject *)obj;

fail_py:
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)j_char);
fail_jolt:
    // --- ROLLBACK LOGIC (ATOMIC REFACTOR) ---
    SHADOW_LOCK(&self->shadow_lock);

    // Reset state atomically
    atomic_store_explicit(&self->slot_states[char_slot], SLOT_EMPTY, memory_order_relaxed);

    // Push back to free stack atomically
    size_t f_idx            = atomic_fetch_add_explicit(&self->free_count, 1, memory_order_relaxed);
    self->free_slots[f_idx] = char_slot;

    SHADOW_UNLOCK(&self->shadow_lock);
    return nullptr;
}