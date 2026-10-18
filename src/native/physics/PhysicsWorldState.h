#pragma once

// This header defines native world state without interpreter headers or
// interpreter-local state.
#include "jolt/JoltConfig.h"
#include "jolt/JoltUtils.h"
#include "physics/CommandBuffer.h"
#include "core/CompilerSpecifics.h"
#include "debug/DebugRender.h"
#include "physics/InternalQuery.h"
#include "soft_bodies/SoftBodyCore.h"
#include "physics/SlotState.h"
#include "core/Threading.h"
#include "core/Types.h"
#include "joltc.h"
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

typedef struct CPH_PhysicsWorld CPH_PhysicsWorld;

// Temporary container used by native world resize operations.
typedef struct CPH_NewBuffers {
    JPH_Real *pos, *ppos;
    float *rot, *prot, *lvel, *avel;
    JPH_BodyID *bids;
    uint64_t *udat;
    CPH_ATOMIC(uint32_t) *gens;
    uint32_t *s2d, *d2s, *free, *cats, *masks, *mats;
    CPH_ATOMIC(uint8_t) *stat;
    CPH_SoftBodyShadow *softs;
} CPH_NewBuffers;

typedef struct {
    int max_bodies;
    int max_pairs;
    int max_contact_constraints;
    int temp_allocator_size;
    int max_physics_jobs;
    int max_physics_barriers;
    int num_threads;
    float penetration_slop;
} CPH_WorldSettings;

typedef struct {
    float gx;
    float gy;
    float gz;
} CPH_GravityVector;

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
} CPH_MaterialData;

/**
 * CACHE ISOLATION MACRO
 * Using explicit padding instead of alignas() to remain compatible
 * with the Python allocator (which doesn't respect 64-byte alignment).
 */
#define CPH_CACHE_LINE_SPACER uint8_t CPH_CONCAT(_unused_pad_, __LINE__)[64]

// --- Native physics-world state (contains no Python object fields) ---
struct CPH_PhysicsWorld {
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

    CPH_CACHE_LINE_SPACER;

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
    CPH_SoftBodyShadow *soft_shadows;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 3: GLOBAL SYNCHRONIZATION (Step Sync)
     * ======================================================================== */
    CPH_ShadowSync step_sync;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 4: COMMAND QUEUE & MUTEXES
     * ======================================================================== */
    ShadowMutex shadow_lock;
    NativeMutex jph_trampoline_lock;

    PhysicsCommand *command_queue;
    PhysicsCommand *command_queue_spare;
    size_t command_count;
    size_t command_capacity;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 5: VOLATILE ATOMIC FLAGS (Polling Targets)
     * ======================================================================== */
    CPH_CACHE_LINE_SPACER;
    atomic_bool is_stepping;
    atomic_bool step_requested;
    CPH_CACHE_LINE_SPACER;
    atomic_int waiting_threads;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 6: QUERIES & BUFFER EXPORT STATE
     * ======================================================================== */
    CPH_CACHE_LINE_SPACER;
    atomic_int active_queries;
    atomic_int view_export_count;
    bool needs_optimization;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 7: CONTACTS & REGISTRIES
     * ======================================================================== */
    struct CPH_ContactEvent *contact_events;
    struct CPH_ContactEvent *contact_buffer;
    atomic_size_t contact_atomic_idx;
    size_t contact_count;
    size_t contact_capacity;

    CPH_MaterialData *materials;
    size_t material_count;
    size_t material_capacity;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 8: MAPPINGS & FILTERS
     * ======================================================================== */
    CPH_ShapeEntry *shape_cache;
    size_t shape_cache_count;
    size_t shape_cache_capacity;
    const void **jolt_body_ptrs;
    CPH_ATOMIC(BodyHandle) * id_to_handle_map;
    uint32_t *slot_to_dense;
    uint32_t *dense_to_slot;
    uint32_t *free_slots;
    uint32_t *categories;
    uint32_t *masks;
    CPH_ATOMIC(uint8_t) * slot_states;
    CPH_ATOMIC(uint32_t) * generations;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 9: CONSTRAINTS & DEFERRED GC
     * ======================================================================== */
    JPH_Constraint **constraints;
    uint32_t *constraint_generations;
    uint32_t *free_constraint_slots;
    CPH_ATOMIC(uint8_t) * constraint_states;
    size_t constraint_count;
    size_t constraint_capacity;
    size_t free_constraint_count;
    int *constraint_types;
    CPH_NewBuffers *trash_buffers;
    size_t trash_count;
    size_t trash_capacity;

    CPH_CACHE_LINE_SPACER;

    /* ========================================================================
     * BUCKET 10: DEBUG OUTPUT
     * ======================================================================== */
    JPH_DebugRenderer *debug_renderer;
    CPH_DebugBuffer debug_lines;
    CPH_DebugBuffer debug_triangles;
};


// --- Native handle helpers ---
[[nodiscard]] [[maybe_unused]]
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
    return ((uint64_t)gen << kHandleIndexBits) | (uint64_t)slot;
}

[[nodiscard]] [[maybe_unused]]
static inline bool unpack_handle(CPH_PhysicsWorld *world, BodyHandle h, uint32_t *slot) {
    const uint64_t h_val = h;
    *slot = (uint32_t)(h_val & kHandleIndexMask);
    const uint32_t gen = (uint32_t)(h_val >> kHandleIndexBits);

    if (CPH_UNLIKELY(*slot >= world->slot_capacity)) {
        return false;
    }

    const uint32_t current_gen =
        atomic_load_explicit(&world->generations[*slot], memory_order_acquire);
    return current_gen == gen;
}
