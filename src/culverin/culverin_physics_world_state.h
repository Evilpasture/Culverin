#pragma once

// This header is the native world state boundary. It deliberately includes no
// CPython headers and contains no PyObject fields or interpreter-local state.
#include "culverin_jolt_config.h"
#include "culverin_jolt_utils.h"
#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"
#include "culverin_debug_render.h"
#include "culverin_internal_query.h"
#include "culverin_soft_body_core.h"
#include "culverin_slot_state.h"
#include "culverin_threading.h"
#include "culverin_types.h"
#include "joltc.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

typedef struct PhysicsWorld PhysicsWorld;

// Temporary container used by native world resize operations.
typedef struct NewBuffers {
    JPH_Real *pos, *ppos;
    float *rot, *prot, *lvel, *avel;
    JPH_BodyID *bids;
    uint64_t *udat;
    CULV_ATOMIC(uint32_t) *gens;
    uint32_t *s2d, *d2s, *free, *cats, *masks, *mats;
    CULV_ATOMIC(uint8_t) *stat;
    SoftBodyShadow *softs;
} NewBuffers;

typedef struct {
    int max_bodies;
    int max_pairs;
    int max_contact_constraints;
    int temp_allocator_size;
    int max_physics_jobs;
    int max_physics_barriers;
    int num_threads;
    float penetration_slop;
} WorldSettings;

typedef struct {
    float gx;
    float gy;
    float gz;
} GravityVector;

typedef enum PhysicsWorldStatus : int {
    PHYSICS_WORLD_STATUS_OK = 0,
    PHYSICS_WORLD_STATUS_OUT_OF_MEMORY = -1,
    PHYSICS_WORLD_STATUS_LIMIT_REACHED = -2,
    PHYSICS_WORLD_STATUS_BUFFER_EXPORTED = -3,
    PHYSICS_WORLD_STATUS_INVALID_ARGUMENT = -4,
} PhysicsWorldStatus;

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

// --- Native physics-world state (contains no Python object fields) ---
struct PhysicsWorld {
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
     * BUCKET 4: COMMAND QUEUE & MUTEXES
     * ======================================================================== */
    ShadowMutex shadow_lock;
    NativeMutex jph_trampoline_lock;

    PhysicsCommand *command_queue;
    PhysicsCommand *command_queue_spare;
    size_t command_count;
    size_t command_capacity;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 5: VOLATILE ATOMIC FLAGS (Polling Targets)
     * ======================================================================== */
    CULV_CACHE_LINE_SPACER;
    atomic_bool is_stepping;
    atomic_bool step_requested;
    CULV_CACHE_LINE_SPACER;
    atomic_int waiting_threads;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 6: QUERIES & BUFFER EXPORT STATE
     * ======================================================================== */
    CULV_CACHE_LINE_SPACER;
    atomic_int active_queries;
    atomic_int view_export_count;
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
     * BUCKET 8: MAPPINGS & FILTERS
     * ======================================================================== */
    ShapeEntry *shape_cache;
    size_t shape_cache_count;
    size_t shape_cache_capacity;
    const void **jolt_body_ptrs;
    CULV_ATOMIC(BodyHandle) * id_to_handle_map;
    uint32_t *slot_to_dense;
    uint32_t *dense_to_slot;
    uint32_t *free_slots;
    uint32_t *categories;
    uint32_t *masks;
    CULV_ATOMIC(uint8_t) * slot_states;
    CULV_ATOMIC(uint32_t) * generations;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 9: CONSTRAINTS & DEFERRED GC
     * ======================================================================== */
    JPH_Constraint **constraints;
    uint32_t *constraint_generations;
    uint32_t *free_constraint_slots;
    CULV_ATOMIC(uint8_t) * constraint_states;
    size_t constraint_count;
    size_t constraint_capacity;
    size_t free_constraint_count;
    int *constraint_types;
    NewBuffers *trash_buffers;
    size_t trash_count;
    size_t trash_capacity;

    CULV_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 10: DEBUG OUTPUT
     * ======================================================================== */
    JPH_DebugRenderer *debug_renderer;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;
};


// --- Native handle helpers ---
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
    return ((uint64_t)gen << HANDLE_INDEX_BITS) | (uint64_t)slot;
}

CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline bool unpack_handle(PhysicsWorld *world, BodyHandle h, uint32_t *slot) {
    const uint64_t h_val = h;
    *slot = (uint32_t)(h_val & HANDLE_INDEX_MASK);
    const uint32_t gen = (uint32_t)(h_val >> HANDLE_INDEX_BITS);

    if (UNLIKELY(*slot >= world->slot_capacity)) {
        return false;
    }

    const uint32_t current_gen =
        atomic_load_explicit(&world->generations[*slot], memory_order_acquire);
    return current_gen == gen;
}
