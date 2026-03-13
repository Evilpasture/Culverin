#include "culverin_character.h"
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_filters.h"
#include "culverin_parsers.h"
#include "culverin_physics_world_internal.h"

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL
char_on_contact_validate(void *userData, const JPH_CharacterVirtual *character,
                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
    return true; // Usually true, unless you want to walk through certain bodies
}

static void record_character_contact(CharacterObject *self, JPH_BodyID bodyID2,
                                     const JPH_RVec3 *pos, const JPH_Vec3 *norm,
                                     ContactEventType type) {
    auto *world    = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    BodyHandle h2  = 0;

    if (world->id_to_handle_map && j_idx < world->max_jolt_bodies) {
        h2 = world->id_to_handle_map[j_idx];
    }
    if (h2 == 0) {
        return; // Ignore unmapped bodies (like internal Jolt helpers)
    }

    BodyHandle h1 = self->handle;
    size_t idx    = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);

    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = (uint32_t)type;

        // Consistent ordering for Python set logic
        if (h1 < h2) {
            ev->body1 = h1;
            ev->body2 = h2;
        } else {
            ev->body1 = h2;
            ev->body2 = h1;
        }
        // hands are tied. precision is lost.
        ev->px               = (float)pos->x;
        ev->py               = (float)pos->y;
        ev->pz               = (float)pos->z;
        ev->nx               = norm->x;
        ev->ny               = norm->y;
        ev->nz               = norm->z;
        ev->impulse          = 1.0f; // Logical trigger strength
        ev->sliding_speed_sq = 0.0f;

        // Look up material of the object we hit
        auto slot2      = (uint32_t)(h2 & 0xFFFFFFFF);
        uint32_t dense2 = world->slot_to_dense[slot2];
        ev->mat1        = 0; // Characters don't have materials yet
        ev->mat2        = world->material_ids[dense2];

        atomic_thread_fence(memory_order_release);
    }
}

static void report_char_vs_char(CharacterObject *self, const JPH_CharacterVirtual *other,
                                const JPH_Vec3 *normal, const JPH_RVec3 *pos,
                                ContactEventType type) {
    auto *world   = self->world;
    BodyHandle h1 = self->handle;

    // 1. Get Inner Body ID
    JPH_BodyID other_bid = JPH_CharacterVirtual_GetInnerBodyID(other);

    // 2. Direct Jolt Lookup (Bypasses our map, which might be too small for
    // Virtual IDs)
    uint64_t userdata = JPH_BodyInterface_GetUserData(world->body_interface, other_bid);
    auto h2           = (BodyHandle)userdata;

    if (h2 == 0) {
        return; // Not a known Culverin object
    }

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = (uint32_t)type;

        // Canonicalize Order
        if (h1 < h2) {
            ev->body1 = h1;
            ev->body2 = h2;
        } else {
            ev->body1 = h2;
            ev->body2 = h1;
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

        atomic_thread_fence(memory_order_release);
    }
}
static void JPH_API_CALL char_on_character_contact_added(void *userData,
                                                         const JPH_CharacterVirtual *character,
                                                         const JPH_CharacterVirtual *otherCharacter,
                                                         JPH_SubShapeID subShapeID2,
                                                         const JPH_RVec3 *contactPosition,
                                                         const JPH_Vec3 *contactNormal,
                                                         JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto *self = (CharacterObject *)userData;
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

    if (dot > 0.1f) {
        float factor            = dot * strength;
        const float max_impulse = 5000.0f;
        if (factor > max_impulse) {
            factor = max_impulse;
        }

        JPH_Vec3 impulse;
        impulse.x = contactNormal->x * factor;

        // Flatten Y Response (allow kicking up, suppress crushing down)
        float y_push = contactNormal->y * factor;
        impulse.y    = (y_push > 0.0f) ? y_push : 0.0f;

        impulse.z = contactNormal->z * factor;

        JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
        JPH_BodyInterface_ActivateBody(bi, bodyID2);
    }
}

// --- Updated Added Callback ---
static void JPH_API_CALL char_on_contact_added(void *userData,
                                               const JPH_CharacterVirtual *character,
                                               JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2,
                                               const JPH_RVec3 *contactPosition,
                                               const JPH_Vec3 *contactNormal,
                                               JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto *self = (CharacterObject *)userData;
    if (!self) {
        return;
    }

    // Record Event
    record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_ADDED);

    // Apply Impulse
    apply_character_impulse(self, bodyID2, contactNormal);
}

static void JPH_API_CALL char_on_contact_persisted(void *userData,
                                                   const JPH_CharacterVirtual *character,
                                                   JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2,
                                                   const JPH_RVec3 *contactPosition,
                                                   const JPH_Vec3 *contactNormal,
                                                   JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter   = true;
    ioSettings->canReceiveImpulses = true;

    auto *self = (CharacterObject *)userData;
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
    auto *self = (CharacterObject *)userData;
    if (!self || !self->world) {
        return;
    }

    PhysicsWorldObject *world = self->world;
    uint32_t j_idx            = JPH_ID_TO_INDEX(bodyID2);

    auto h1       = self->handle;
    BodyHandle h2 = 0;
    if (world->id_to_handle_map && j_idx < world->max_jolt_bodies) {
        h2 = world->id_to_handle_map[j_idx];
    }
    if (h2 == 0) {
        return;
    }

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = EVENT_REMOVED;
        ev->body1        = (h1 < h2) ? h1 : h2;
        ev->body2        = (h1 < h2) ? h2 : h1;
        // Geometry is zeroed for removal
        memset(&ev->px, 0, sizeof(float) * 8);
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
    auto *self  = (CharacterObject *)userData;
    auto *world = self->world;

    BodyHandle h1 = self->handle;
    // We have to use the CharacterID to find the handle.
    // Jolt CharacterIDs usually map to the inner BodyID.
    uint32_t j_idx = JPH_ID_TO_INDEX(otherCharacterID);
    BodyHandle h2  = world->id_to_handle_map[j_idx];

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type         = EVENT_REMOVED;
        ev->body1        = (h1 < h2) ? h1 : h2;
        ev->body2        = (h1 < h2) ? h2 : h1;
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_adjust_velocity(void *userData,
                                                 const JPH_CharacterVirtual *character,
                                                 const JPH_Body *body2, JPH_Vec3 *ioLinearVelocity,
                                                 JPH_Vec3 *ioAngularVelocity) {

    // Usually, we want the default behavior (character follows the body).
    // TODO: add logic here if you want the character to "slip" on certain
    // materials.
}

// Map the procs
const JPH_CharacterContactListener_Procs char_listener_procs = {
    .OnContactValidate           = char_on_contact_validate,
    .OnContactAdded              = char_on_contact_added,
    .OnAdjustBodyVelocity        = char_on_adjust_velocity,   // ADDED
    .OnContactPersisted          = char_on_contact_persisted, // CHANGED from char_on_contact_added
    .OnContactRemoved            = char_on_contact_removed,   // ADDED
    .OnCharacterContactValidate  = NULL,                      // Default True is fine
    .OnCharacterContactAdded     = char_on_character_contact_added,
    .OnCharacterContactPersisted = char_on_character_contact_persisted, // ADDED
    .OnCharacterContactRemoved   = char_on_character_contact_removed,   // ADDED
    .OnContactSolve              = NULL                                 // Advanced, keep NULL
};

PyCFunction_DeclareMethodFromModule Character_move(CharacterObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    // 1. INTEGRATED FAST PARSE
    // The parser now writes directly into the Vec3f struct using your parse_vec3_f32 logic.
    Vec3f v_in = {.x = 0.0f, .y = 0.0f, .z = 0.0f};
    float dt   = 0.0f;

    void *targets[CharMove_COUNT];
    targets[IDX_CM_VEL] = &v_in;
    targets[IDX_CM_DT]  = &dt;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!FastParse_Unified(args, nargs, kwnames, &CharMoveParser, targets)) {
        return NULL;
    }

    // Validation (Inline macros/checks)
    VALIDATE_FINITE_VEC3(v_in.x, v_in.y, v_in.z, "Character velocity");

    // 2. SNAPSHOT (Shadow Lock)
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // Atomic stores for worker threads
    atomic_store_explicit(&self->last_vx, v_in.x, memory_order_relaxed);
    atomic_store_explicit(&self->last_vy, v_in.y, memory_order_relaxed);
    atomic_store_explicit(&self->last_vz, v_in.z, memory_order_relaxed);

    auto slot      = (uint32_t)(self->handle & 0xFFFFFFFF);
    uint32_t dense = self->world->slot_to_dense[slot];

    // Snapshot: Copy Current to Prev to provide a clean state for interpolation
    // We do this inside the lock to ensure rendering threads don't see half-updated strides.
    auto *shadow_pos  = (PosStride *)self->world->positions;
    auto *shadow_ppos = (PosStride *)self->world->prev_positions;
    auto *shadow_rot  = (AuxStride *)self->world->rotations;
    auto *shadow_prot = (AuxStride *)self->world->prev_rotations;

    shadow_ppos[dense] = shadow_pos[dense];
    shadow_prot[dense] = shadow_rot[dense];

    SHADOW_UNLOCK(&self->world->shadow_lock);

    // 3. JOLT EXECUTION (No GIL, No Shadow Lock)
    JPH_Vec3 j_v = {v_in.x, v_in.y, v_in.z};
    JPH_CharacterVirtual_SetLinearVelocity(self->character, &j_v);

    // Optimized settings: Initialized once
    static const JPH_ExtendedUpdateSettings update_settings = {
        .stickToFloorStepDown             = {0.0f, -0.5f, 0.0f},
        .walkStairsStepUp                 = {0.0f, 0.4f, 0.0f},
        .walkStairsMinStepForward         = 0.02f,
        .walkStairsStepForwardTest        = 0.15f,
        .walkStairsCosAngleForwardContact = 0.996f};

    // The trampoline lock protects Jolt's non-thread-safe internal Virtual Character callbacks
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    Py_BEGIN_ALLOW_THREADS JPH_CharacterVirtual_ExtendedUpdate(
        self->character, dt, &update_settings, 1, self->world->system, self->body_filter,
        self->shape_filter);
    Py_END_ALLOW_THREADS NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    // 4. POST-MOVE SYNC
    SHADOW_LOCK(&self->world->shadow_lock);

    JPH_STACK_ALLOC(JPH_RVec3, current_p);
    JPH_STACK_ALLOC(JPH_Quat, current_r);
    JPH_CharacterVirtual_GetPosition(self->character, current_p);
    JPH_CharacterVirtual_GetRotation(self->character, current_r);

    // Update the World Shadow Buffers so the GPU sees the new position next frame
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
        return NULL;
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
    // 1. Stack Allocation and Parser Targets
    PosStride pos = {};
    void *targets[SetPosChar_COUNT];
    targets[IDX_SPC_POS] = &pos; // The converter calls parse_vec3_r64 internally

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &SetPosCharParser, targets)) {
        return NULL;
    }

    // --- PHYSICS LOGIC ---
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // 3. Update Jolt
    // (Note: Using your existing stack allocation or direct struct pass)
    JPH_RVec3 j_pos = {pos.x, pos.y, pos.z};
    JPH_CharacterVirtual_SetPosition(self->character, &j_pos);

    // 4. Update Shadow Buffers
    // We update both current and prev to prevent "teleport streaks" (interpolation artifacts)
    auto slot          = (uint32_t)(self->handle & 0xFFFFFFFF);
    uint32_t dense_idx = self->world->slot_to_dense[slot];
    auto off           = (size_t)dense_idx * 4; // Assuming 4-float alignment/stride

    // Update Current
    self->world->positions[off + 0] = pos.x;
    self->world->positions[off + 1] = pos.y;
    self->world->positions[off + 2] = pos.z;

    // Update Previous
    self->world->prev_positions[off + 0] = pos.x;
    self->world->prev_positions[off + 1] = pos.y;
    self->world->prev_positions[off + 2] = pos.z;

    SHADOW_UNLOCK(&self->world->shadow_lock);
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethodFromModule Character_set_rotation(CharacterObject *self,
                                                           PyObject *const *args, Py_ssize_t nargs,
                                                           PyObject *kwnames) {
    // 1. Explicitly initialized stack target
    AuxStride rot = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    void *targets[SetRotChar_COUNT];
    targets[IDX_SRC_ROT] = &rot; // Parser triggers your parse_quat_f32 logic

    // 2. High Speed Parse
    if (!FastParse_Unified(args, nargs, kwnames, &SetRotCharParser, targets)) {
        return NULL;
    }

    // --- PHYSICS LOGIC ---
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    // 3. Update Jolt
    // We can cast our AuxStride directly to JPH_Quat if they are layout-compatible
    // or just initialize a temporary.
    JPH_Quat q = {.x = rot.x, .y = rot.y, .z = rot.z, .w = rot.w};
    JPH_CharacterVirtual_SetRotation(self->character, &q);

    // 4. Update Shadow Buffers (Zero-Streak Reset)
    auto slot          = (uint32_t)(self->handle & 0xFFFFFFFF);
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
    // 1. Explicitly initialized target
    float strength = 0.0f;

    void *targets[SetStrengthChar_COUNT];
    targets[IDX_SSC_STRENGTH] = &strength;

    // 2. High Speed Parse (Handles both world.set_strength(200.0) and
    // world.set_strength(strength=200.0))
    if (!FastParse_Unified(args, nargs, kwnames, &SetStrengthCharParser, targets)) {
        return NULL;
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
        return NULL;
    }

    // Clamp alpha to [0.0, 1.0]
    JPH_Real alpha = fmax(0.0, fmin(1.0, (JPH_Real)alpha_dbl));

    // --- 2. Snapshot State (Locked) ---
    SHADOW_LOCK(&self->world->shadow_lock);

    // Consistency Guard: Ensure we aren't reading while the world is stepping/swapping
    BLOCK_UNTIL_NOT_STEPPING(self->world);

    auto slot      = (uint32_t)(self->handle & 0xFFFFFFFF);
    uint32_t dense = self->world->slot_to_dense[slot];

    // Map world buffers using Strides
    auto *shadow_ppos = (PosStride *)self->world->prev_positions;
    auto *shadow_prot = (AuxStride *)self->world->prev_rotations;

    // Capture High-Precision "Start" state from the shadow buffer
    PosStride start_p = shadow_ppos[dense];
    AuxStride start_r = shadow_prot[dense];

    // Capture High-Precision "End" state from Jolt
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

    float mag_sq  = rx * rx + ry * ry + rz * rz + rw * rw;
    float inv_len = (mag_sq > 1e-9f) ? 1.0f / sqrtf(mag_sq) : 1.0f;
    rx *= inv_len;
    ry *= inv_len;
    rz *= inv_len;
    rw *= inv_len;

    // --- 4. Optimized Return (Manual Tuple Creation) ---
    // Instead of Py_BuildValue, we manually build the nested tuples.
    // This is significantly faster as it avoids format string parsing.
    PyObject *pos_tuple =
        PyTuple_Pack(3, PyFloat_FromDouble(px), PyFloat_FromDouble(py), PyFloat_FromDouble(pz));
    PyObject *rot_tuple = PyTuple_Pack(4, PyFloat_FromDouble(rx), PyFloat_FromDouble(ry),
                                       PyFloat_FromDouble(rz), PyFloat_FromDouble(rw));

    PyObject *result = PyTuple_Pack(2, pos_tuple, rot_tuple);

    // PyTuple_Pack increments refs for items, but the PyFloat_FromDouble
    // calls created new refs. We must decref the temporaries.
    Py_DECREF(pos_tuple);
    Py_DECREF(rot_tuple);

    return result;
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
PyType_DeclareSlot_StatusFromModule Character_traverse(CharacterObject *self, visitproc visit, void *arg) {
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

    // 1. DRAIN (Wait for world to stop)
    SHADOW_LOCK(&self->world->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self->world);
    BLOCK_UNTIL_NOT_QUERYING(self->world);
    uint32_t slot = (uint32_t)(self->handle & 0xFFFFFFFF);
    world_remove_body_slot(self->world, slot);
    SHADOW_UNLOCK(&self->world->shadow_lock);

    // 2. JOLT DESTRUCTION (Hard Serialized)
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
    auto *shape                  = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(ss);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);
    if (!shape) {
        return NULL;
    }

    JPH_CharacterVirtualSettings settings;
    JPH_CharacterVirtualSettings_Init(&settings);
    settings.base.shape         = shape;
    settings.base.maxSlopeAngle = params.max_slope * (JPH_M_PI / 180.0f);

    JPH_CharacterVirtual *j_char = JPH_CharacterVirtual_Create(
        &settings, &(JPH_RVec3){(double)pos.px, (double)pos.py, (double)pos.pz},
        &(JPH_Quat){0, 0, 0, 1}, 1, self->system);

    JPH_Shape_Destroy(shape);
    if (!j_char) {
        return NULL;
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

    BodyHandle h = make_handle(slot, self->generations[slot]);
    obj->handle  = h;

    auto dense_idx = (uint32_t)self->count;
    JPH_BodyID bid = JPH_CharacterVirtual_GetInnerBodyID(j_char);
    uint32_t j_idx = JPH_ID_TO_INDEX(bid);

    if (j_idx < self->max_jolt_bodies) {
        self->id_to_handle_map[j_idx] = h;
    }

    self->body_ids[dense_idx]      = bid;
    self->slot_to_dense[slot]      = dense_idx;
    self->dense_to_slot[dense_idx] = slot;
    self->slot_states[slot]        = SLOT_CHARACTER;
    self->user_data[dense_idx]     = 0;
    self->count++;
    self->view_shape[0] = (Py_ssize_t)self->count;

    JPH_BodyInterface_SetUserData(self->body_interface, bid, (uint64_t)h);
    SHADOW_UNLOCK(&self->shadow_lock);
}

// Helper 3: Filter and Listener serialization (Trampoline Lock)
static void setup_char_filters(CharacterObject *obj) {
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock); // Fix: Use Native
    JPH_CharacterContactListener_SetProcs(&char_listener_procs);
    obj->listener = JPH_CharacterContactListener_Create(obj);
    JPH_BodyFilter_SetProcs(&global_bf_procs);
    obj->body_filter = JPH_BodyFilter_Create(NULL);
    JPH_ShapeFilter_SetProcs(&global_sf_procs);
    obj->shape_filter = JPH_ShapeFilter_Create(NULL);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);

    JPH_CharacterVirtual_SetListener(obj->character, obj->listener);
    obj->bp_filter  = JPH_BroadPhaseLayerFilter_Create(NULL);
    obj->obj_filter = JPH_ObjectLayerFilter_Create(NULL);
}

// Main Orchestrator
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_character(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  Py_ssize_t nargs,
                                                                  PyObject *kwnames) {
    // 1. Setup Targets with Default Values
    // Note: Your parse_vec3 functions handle Py_None by doing nothing,
    // so we can initialize 'pos' to a default here.
    PosStride pos = {.x = 0, .y = 0, .z = 0};
    float height  = 1.8f;
    float radius  = 0.4f;
    float step_h  = 0.4f;
    float slope   = 45.0f;

    void *targets[CreateChar_COUNT];
    targets[IDX_CCHAR_POS]   = &pos;    // Parser calls parse_vec3_r64
    targets[IDX_CCHAR_H]     = &height; // Parser calls fp_conv_float
    targets[IDX_CCHAR_R]     = &radius;
    targets[IDX_CCHAR_STEP]  = &step_h;
    targets[IDX_CCHAR_SLOPE] = &slope;

    // 2. High Speed Parse (Handles positional, keywords, and vec3 unpacking)
    if (!FastParse_Unified(args, nargs, kwnames, &CreateCharParser, targets)) {
        return NULL;
    }

    // --- WORLD LOGIC ---
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    if (self->free_count == 0 && PhysicsWorld_resize(self, self->capacity * 2) < 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return NULL;
    }

    uint32_t char_slot           = self->free_slots[--self->free_count];
    self->slot_states[char_slot] = SLOT_PENDING_CREATE;
    SHADOW_UNLOCK(&self->shadow_lock);

    // 3. Resource Allocation
    PositionVector pos_vec      = {pos.x, pos.y, pos.z};
    CharacterParams char_params = {height, radius, slope};

    JPH_CharacterVirtual *j_char = alloc_j_char(self, pos_vec, char_params);
    if (!j_char) {
        goto fail_jolt;
    }

    auto *obj = (CharacterObject *)PyObject_GC_New(
        CharacterObject,
        (PyTypeObject *)get_culverin_state(PyType_GetModule(Py_TYPE(self)))->CharacterType);
    if (!obj) {
        goto fail_py;
    }

    // ... (rest of initialization remains the same) ...
    obj->world     = (PhysicsWorldObject *)Py_NewRef(self);
    obj->character = j_char;
    obj->prev_px   = pos.x;
    obj->prev_py   = pos.y;
    obj->prev_pz   = pos.z;

    register_char(self, obj, j_char, char_slot);
    setup_char_filters(obj);

    PyObject_GC_Track((PyObject *)obj);
    return (PyObject *)obj;

fail_py:
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)j_char);
fail_jolt:
    SHADOW_LOCK(&self->shadow_lock);
    self->slot_states[char_slot]         = SLOT_EMPTY;
    self->free_slots[self->free_count++] = char_slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    return NULL;
}
