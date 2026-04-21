#pragma once
#include "culverin_command_buffer.h"
#include "culverin_compiler_specifics.h"
#include "culverin_debug_render.h"
#include "culverin_physics_world_internal.h"
#include "culverin_ragdoll.h"
#include "culverin_soft_body.h"
#include "culverin_threading.h"
#include "culverin_types.h"
#include "joltc.h"
#include <Python.h>

// --- The Object Struct ---
typedef struct PhysicsWorldObject {
    PyObject_HEAD // 16 bytes
        PyObject *weakreflist;

    // --- BUCKET 1: Pointers & 8-byte types (Zero Padding) ---
    JPH_PhysicsSystem *system;
    JPH_CharacterVsCharacterCollision *char_vs_char_manager;
    JPH_BodyInterface *body_interface;
    JPH_JobSystem *job_system;
    JPH_BroadPhaseLayerInterface *bp_interface;
    JPH_ObjectLayerPairFilter *pair_filter;
    JPH_ObjectVsBroadPhaseLayerFilter *bp_filter;
    JPH_ContactListener *contact_listener;

    // --- HOT SYNC BLOCK: Kept together for L1d Locality ---
    JPH_Real *positions;
    JPH_Real *prev_positions;
    float *rotations;
    float *prev_rotations;
    float *linear_velocities;
    float *angular_velocities;
    JPH_BodyID *body_ids;
    uint64_t *user_data;
    uint32_t *material_ids;

    // Array of SoftBodyShadow structs, parallel to body_ids.
    // If dense_idx is a rigid body, soft_shadows[dense_idx].vertices == nullptr
    SoftBodyShadow *soft_shadows;

    // --- Data Buffers ---
    struct ContactEvent *contact_events;
    struct ContactEvent *contact_buffer;
    MaterialData *materials;
    PhysicsCommand *command_queue;
    PhysicsCommand *command_queue_spare;
    struct ShapeEntry *shape_cache;
    CULV_ATOMIC(BodyHandle) * id_to_handle_map;
    JPH_Constraint **constraints;
    uint32_t *categories;
    uint32_t *masks;
    CULV_ATOMIC(uint32_t) * generations;
    uint32_t *slot_to_dense;
    uint32_t *dense_to_slot;
    uint32_t *free_slots;
    uint32_t *constraint_generations;
    uint32_t *free_constraint_slots;

    // --- Counters (8-byte) ---
    size_t contact_count;
    size_t contact_capacity;
    size_t contact_max_capacity;
    atomic_size_t contact_atomic_idx;
    size_t material_count;
    size_t material_capacity;
    atomic_size_t free_count;
    size_t slot_capacity;
    size_t command_count;
    size_t command_capacity;
    size_t spare_capacity;
    size_t shape_cache_count;
    size_t shape_cache_capacity;
    atomic_size_t count;
    size_t capacity;
    size_t constraint_count;
    size_t constraint_capacity;
    size_t free_constraint_count;
    double time;

    // --- DEFERRED GARBAGE COLLECTION ---
    NewBuffers *trash_buffers;
    size_t trash_count;
    size_t trash_capacity;

    // --- BUCKET 2: 4-byte types (Packed 2-per-slot) ---
    // These three now share 12 bytes total + 4 bytes padding at the end
    // instead of creating holes between every pointer.
    uint32_t max_jolt_bodies;
    atomic_int active_queries;
    atomic_int view_export_count;
#if !defined(Py_GIL_DISABLED)
    atomic_int waiting_threads;
#endif

    // --- BUCKET 3: Structs & Complex Types ---
    ShadowSync step_sync;    // 16 bytes (Internal 2-byte alignment)
    ShadowMutex shadow_lock; // MagMutex (usually 1 bytes)

    // --- BUCKET 4: Small types (Packed at the tail) ---
    CULV_ATOMIC(uint8_t) * slot_states;
    uint8_t *constraint_states;
    atomic_bool step_requested;
    atomic_bool is_stepping;
    bool needs_optimization;
    atomic_bool is_resizing;
    atomic_bool is_deallocating;

    // --- Large Tail Arrays ---
    Py_ssize_t view_shape[2];
    Py_ssize_t view_strides[2];

    // --- Debug Renderer ---
    JPH_DebugRenderer *debug_renderer;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;
} PhysicsWorldObject;

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