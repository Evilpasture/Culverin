#pragma once
#include "culverin.h"
#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"
#include "culverin_debug_render.h"
#include "culverin_internal_query.h"
#include "culverin_physics_world_internal.h"
#include "culverin_soft_body.h"
#include "culverin_threading.h"
#include "culverin_types.h"
#include "joltc.h"
#include <Python.h>

// --- Material Registry ---
typedef struct {
    uint32_t id;
    float friction;
    float restitution;
    // Padding/Alignment isn't critical here as this is a lookup array, not a
    // stream
} MaterialData;

/**
 * CACHE ISOLATION MACRO
 * Using explicit padding instead of alignas() to remain compatible
 * with the Python allocator (which doesn't respect 64-byte alignment).
 */
#define CULV_CACHE_LINE_SPACER uint8_t CULV_CONCAT(_unused_pad_, __LINE__)[64]

// --- The Object Struct ---
typedef struct PhysicsWorldObject {
    PyObject_HEAD PyObject *weakreflist;

    /* ========================================================================
     * BUCKET 1: READ-ONLY / COLD DATA
     * ======================================================================== */
    JPH_PhysicsSystem *system;
    JPH_BodyInterface *body_interface;
    JPH_JobSystem *job_system;
    JPH_BroadPhaseLayerInterface *bp_interface;
    JPH_ObjectLayerPairFilter *pair_filter;
    JPH_ObjectVsBroadPhaseLayerFilter *bp_filter;
    JPH_ContactListener *contact_listener;
    JPH_TempAllocator *temp_allocator;
    JPH_CharacterVsCharacterCollision *char_vs_char_manager;
    uint32_t max_jolt_bodies;
    size_t contact_max_capacity;
    bool sync_ready;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 2: HOT SIMULATION STATE (The Stepper's Workspace)
     * ======================================================================== */
    double time;
    atomic_size_t count;
    size_t capacity;
    size_t slot_capacity;
    atomic_size_t free_count;

    // Hot Pointers (Shadow Buffers)
    JPH_Real *positions;
    JPH_Real *prev_positions;
    float *rotations;
    float *prev_rotations;
    float *linear_velocities;
    float *angular_velocities;
    JPH_BodyID *body_ids;
    uint64_t *user_data;
    uint32_t *material_ids;
    SoftBodyShadow *soft_shadows;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 3: GLOBAL SYNCHRONIZATION (Step Sync)
     * ======================================================================== */
    ShadowSync step_sync;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 4: COMMAND QUEUE & MUTEXES (The War Zone)
     * ======================================================================== */
    ShadowMutex shadow_lock;         // Primary lock for Python mutations
    NativeMutex jph_trampoline_lock; // Secondary lock for Jolt internals

    PhysicsCommand *command_queue;
    PhysicsCommand *command_queue_spare;
    size_t command_count;
    size_t command_capacity;
    size_t spare_capacity;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 5: VOLATILE ATOMIC FLAGS (Polling Targets)
     * ======================================================================== */
    CULV_CACHE_LINE_SPACER;
    atomic_bool is_stepping;
    atomic_bool step_requested;
    // #if !defined(Py_GIL_DISABLED)
    CULV_CACHE_LINE_SPACER;
    atomic_int waiting_threads;
    // #endif

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 6: QUERIES & BUFFER VIEWS
     * ======================================================================== */
    CULV_CACHE_LINE_SPACER;
    atomic_int active_queries;    // Hammered by batch raycasts
    atomic_int view_export_count; // Hammered by housekeeper (Numpy view)
    bool needs_optimization;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 7: CONTACTS & REGISTRIES
     * ======================================================================== */
    struct ContactEvent *contact_events;
    struct ContactEvent *contact_buffer;
    atomic_size_t contact_atomic_idx;
    size_t contact_count;
    size_t contact_capacity;

    MaterialData *materials;
    size_t material_count;
    size_t material_capacity;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 8: MAPPINGS & FILTERS (Categories/Masks)
     * ======================================================================== */
    ShapeEntry *shape_cache;
    size_t shape_cache_count;
    size_t shape_cache_capacity;

    const void **jolt_body_ptrs;

    CULV_ATOMIC(BodyHandle) * id_to_handle_map;
    uint32_t *slot_to_dense;
    uint32_t *dense_to_slot;
    uint32_t *free_slots;
    uint32_t *categories; // Filter Data
    uint32_t *masks;      // Filter Data
    CULV_ATOMIC(uint8_t) * slot_states;
    CULV_ATOMIC(uint32_t) * generations;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 9: CONSTRAINTS & DEFERRED GC
     * ======================================================================== */
    JPH_Constraint **constraints;
    uint32_t *constraint_generations;
    uint32_t *free_constraint_slots;
    uint8_t *constraint_states;
    size_t constraint_count;
    size_t constraint_capacity;
    size_t free_constraint_count;

    NewBuffers *trash_buffers;
    size_t trash_count;
    size_t trash_capacity;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 10: TAIL DATA (Debug & View Shapes)
     * ======================================================================== */
    JPH_DebugRenderer *debug_renderer;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;

    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];

} PhysicsWorldObject;

// --- Callback Logic ---
// Old ContactEvent for compatibility
typedef struct ContactEvent {
    CULV_ATOMIC(BodyHandle) body1;
    CULV_ATOMIC(BodyHandle) body2;
    float px, py, pz;
    float nx, ny, nz;
    float impulse;
    float sliding_speed_sq; // Scratching speed squared(tangential)
    uint32_t mat1;
    uint32_t mat2;
    uint32_t type;
    uint32_t _pad;
} ContactEvent;

static_assert(sizeof(ContactEvent) == MEMORY_ALIGNMENT_SIZE);

CULV_MAYBE_UNUSED static constexpr int CONTACT_MAX_CAPACITY = sizeof(ContactEvent) * 8 << 5;

// --- Handle Helper ---

CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
    return ((uint64_t)gen << HANDLE_INDEX_BITS) | (uint64_t)slot;
}

CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline bool unpack_handle(PhysicsWorldObject *self, BodyHandle h, uint32_t *slot) {
    // 1. 'h' is now a plain uint64_t (BodyHandle) passed by value.
    // There is no thread contention on a local variable, so we read it directly.
    // This eliminates the reinterpret_cast and the deleted constructor error.
    uint64_t h_val = h;

    *slot        = (uint32_t)(h_val & HANDLE_INDEX_MASK);
    uint32_t gen = (uint32_t)(h_val >> HANDLE_INDEX_BITS);

    // UNLIKELY is a compiler hint (builtin_expect) defined in specifics.h
    if (UNLIKELY(*slot >= self->slot_capacity)) {
        return false;
    }

    // 2. Read the current generation from the world's ATOMIC storage.
    // This MUST stay atomic because another thread (Physics Sim) could
    // be incrementing this value simultaneously.
    // This works in both C and C++ because 'generations' is CULV_ATOMIC(uint32_t)*
    uint32_t current_gen = atomic_load_explicit(&self->generations[*slot], memory_order_acquire);

    // 3. Logic check remains identical: Handle is valid if generations match.
    return (current_gen == gen);
}

#define CHECK_HANDLE(h_raw, slot_out)                                                              \
    do {                                                                                           \
        static_assert(sizeof(CULV_TYPE_OF(h_raw)) == sizeof(uint64_t));                            \
        static_assert(sizeof(CULV_TYPE_OF(slot_out)) == sizeof(uint32_t));                         \
        if (UNLIKELY(!unpack_handle(self, (BodyHandle)(h_raw), &(slot_out)))) {                    \
            SHADOW_UNLOCK(&self->shadow_lock);                                                     \
            RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)

#define CHECK_STATE(state_val, slot_mask)                                                          \
    do {                                                                                           \
        static_assert(sizeof(state_val) == sizeof(slot_mask));                                     \
        if (UNLIKELY(!is_state_valid((state_val), (slot_mask)))) {                                 \
            SHADOW_UNLOCK(&self->shadow_lock);                                                     \
            RAISE_STALE_HANDLE();                                                                  \
        }                                                                                          \
    } while (false)