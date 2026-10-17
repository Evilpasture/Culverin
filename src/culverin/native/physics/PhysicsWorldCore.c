#include "physics/PhysicsWorldCore.h"
#include "core/NativeAlloc.h"
#include "physics/ContactEventDefinitions.h"
#include "physics/InternalQuery.h"
#include "physics/CommandBuffer.h"
#include "physics/ShadowSync.h"
#include <math.h>
#include <string.h>

static constexpr size_t AVX_ALIGNMENT                  = 32;
static constexpr size_t SHADOW_CAPACITY_PADDING        = 128;
static constexpr size_t SHADOW_CAPACITY_GROW           = 1024;
static constexpr size_t COMMAND_QUEUE_INITIAL_CAPACITY = 64;
static constexpr size_t INITIAL_BODY_CAPACITY = 1024;
static constexpr size_t CONTACT_MAX_CAPACITY = sizeof(ContactEvent) * 8 << 5;
static constexpr uint32_t ALL_LAYER_BITS               = 0xFFFFu;

static constexpr JPH_Real VERIFY_ABI_POS_X   = 10.0;
static constexpr JPH_Real VERIFY_ABI_POS_Y   = 20.0;
static constexpr JPH_Real VERIFY_ABI_POS_Z   = 30.0;
static constexpr double VERIFY_ABI_TOLERANCE = 0.1;


// Helper: Apply mass, sensor, CCD, and sleeping settings to the creation
// struct
void configure_body_settings(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                                    BodyConfig cfg) {
    // Use the members of the struct instead of loose variables
    if (cfg.is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }

    if (cfg.use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }

    if (cfg.motion_type == 2) { // MOTION_DYNAMIC
        JPH_BodyCreationSettings_SetAllowSleeping(settings, true);
    }

    if (cfg.allowed_dofs != 0U) {
        JPH_BodyCreationSettings_SetAllowedDOFs(settings, (JPH_AllowedDOFs)cfg.allowed_dofs);
    }

    JPH_BodyCreationSettings_SetFriction(settings, cfg.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, cfg.restitution);

    if (cfg.mass > 0.0f) {
        JPH_MassProperties mp;
        JPH_Shape_GetMassProperties(shape, &mp);
        float scale = cfg.mass / fmaxf(mp.mass, 1e-6f);
        mp.mass     = cfg.mass;
        for (int i = 0; i < 3; i++) {
            mp.inertia.column[i].x *= scale;
            mp.inertia.column[i].y *= scale;
            mp.inertia.column[i].z *= scale;
        }
        JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
        JPH_BodyCreationSettings_SetOverrideMassProperties(
            settings, JPH_OverrideMassProperties_CalculateInertia);
    }
}


// Helper 2: Apply physics properties (mass, friction, etc) to creation settings
void apply_body_creation_props(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                                      BodyCreationProps props) {
    if (props.mass > 0.0f) {
        JPH_MassProperties mp;
        JPH_Shape_GetMassProperties(shape, &mp);
        if (mp.mass > 1e-6f) {
            float scale = props.mass / mp.mass;
            mp.mass     = props.mass;
            for (int i = 0; i < 3; i++) {
                mp.inertia.column[i].x *= scale;
                mp.inertia.column[i].y *= scale;
                mp.inertia.column[i].z *= scale;
            }
            JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &mp);
            JPH_BodyCreationSettings_SetOverrideMassProperties(
                settings, JPH_OverrideMassProperties_CalculateInertia);
        }
    }

    if (props.is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }

    if (props.use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }

    JPH_BodyCreationSettings_SetAllowSleeping(settings, true);

    JPH_BodyCreationSettings_SetFriction(settings, props.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, props.restitution);
}

NativeMutex g_jph_init_lock;

void physics_world_init_state(PhysicsWorld *world) {
    // 1.1 Jolt Core Pointers
    world->system               = nullptr;
    world->char_vs_char_manager = nullptr;
    world->body_interface       = nullptr;
    world->job_system           = nullptr;
    world->bp_interface         = nullptr;
    world->pair_filter          = nullptr;
    world->bp_filter            = nullptr;
    world->contact_listener     = nullptr;
    world->temp_allocator       = nullptr;

    // 1.2. Hot Sync Shadow Buffers
    world->positions          = nullptr;
    world->prev_positions     = nullptr;
    world->rotations          = nullptr;
    world->prev_rotations     = nullptr;
    world->linear_velocities  = nullptr;
    world->angular_velocities = nullptr;
    world->body_ids           = nullptr;
    world->user_data          = nullptr;
    world->soft_shadows       = nullptr;
    world->material_ids       = nullptr;

    // 1.3. Data Buffers & Mapping Tables
    world->contact_events         = nullptr;
    world->contact_buffer         = nullptr;
    world->materials              = nullptr;
    world->command_queue          = nullptr;
    world->command_queue_spare    = nullptr;
    world->shape_cache            = nullptr;
    world->id_to_handle_map       = nullptr;
    world->constraints            = nullptr;
    world->categories             = nullptr;
    world->masks                  = nullptr;
    world->generations            = nullptr;
    world->slot_to_dense          = nullptr;
    world->dense_to_slot          = nullptr;
    world->free_slots             = nullptr;
    world->constraint_generations = nullptr;
    world->free_constraint_slots  = nullptr;
    world->constraint_types       = nullptr;
    world->trash_buffers          = nullptr;
    world->slot_states            = nullptr;
    world->constraint_states      = nullptr;

    // 1.4. Counters & Simulation State
    world->contact_count        = 0;
    world->contact_capacity     = 0;
    world->contact_max_capacity = 0;
    atomic_init(&world->contact_atomic_idx, 0);

    world->material_count    = 0;
    world->material_capacity = 0;
    atomic_init(&world->free_count, 0);
    world->slot_capacity        = 0;
    world->command_count        = 0;
    world->command_capacity     = 0;
    world->shape_cache_count    = 0;
    world->shape_cache_capacity = 0;
    atomic_init(&world->count, 0);
    world->capacity              = 0;
    world->constraint_count      = 0;
    world->constraint_capacity   = 0;
    world->free_constraint_count = 0;
    world->trash_count           = 0;
    world->trash_capacity        = 0;
    world->time                  = 0.0;

    // 1.5. Query & Sync State
    world->max_jolt_bodies = 0;
    atomic_init(&world->active_queries, 0);
    atomic_init(&world->view_export_count, 0);
    // #if !defined(Py_GIL_DISABLED)
    atomic_init(&world->waiting_threads, 0);
    // #endif
    atomic_init(&world->step_requested, false);
    atomic_init(&world->is_stepping, false);
    world->needs_optimization = false;

    // 1.6. Complex Structs
    world->step_sync = (ShadowSync){};
    INIT_LOCK(world->shadow_lock);
    INIT_NATIVE_MUTEX(world->jph_trampoline_lock);

    // 1.7. Debug Renderer
    world->debug_renderer  = nullptr;
    world->debug_lines     = (DebugBuffer){};
    world->debug_triangles = (DebugBuffer){};

    world->debug_renderer = JPH_DebugRenderer_Create(world);

    INIT_NATIVE_MUTEX(world->step_sync.mutex);
    INIT_NATIVE_COND(world->step_sync.cond);
}

int allocate_constraints(PhysicsWorld *world) {
    constexpr uint32_t CONSTRAINT_INITIAL_CAPACITY = 256;
    world->constraint_capacity                      = CONSTRAINT_INITIAL_CAPACITY;
    world->constraints =
        (JPH_Constraint **)CulvNative_Calloc(CONSTRAINT_INITIAL_CAPACITY, sizeof(JPH_Constraint *));
    world->constraint_generations = CulvNative_Calloc(CONSTRAINT_INITIAL_CAPACITY, sizeof(uint32_t));
    world->free_constraint_slots  = CulvNative_Malloc(CONSTRAINT_INITIAL_CAPACITY * sizeof(uint32_t));
    world->constraint_states      = CulvNative_Calloc(CONSTRAINT_INITIAL_CAPACITY, sizeof(uint8_t));
    world->constraint_types       = CulvNative_Calloc(CONSTRAINT_INITIAL_CAPACITY, sizeof(int));

    if (!world->constraints || !world->free_constraint_slots || !world->constraint_generations ||
        !world->constraint_states || !world->constraint_types) {
        return -1;
    }

    for (uint32_t i = 0; i < CONSTRAINT_INITIAL_CAPACITY; i++) {
        world->constraint_generations[i] = 1;
        world->free_constraint_slots[i]  = i;
        atomic_store_explicit(&world->constraint_states[i], SLOT_EMPTY, memory_order_relaxed);
    }
    world->free_constraint_count = CONSTRAINT_INITIAL_CAPACITY;
    return 0;
}

void initialize_free_slots(PhysicsWorld *world, size_t start_idx) {
    for (uint32_t i = (uint32_t)start_idx; i < (uint32_t)world->slot_capacity; i++) {
        // Initialize the atomic generation for this slot
        atomic_init(&world->generations[i], 1);

        // Atomically increment free_count and get the index to store the slot
        size_t f_idx = atomic_fetch_add_explicit(&world->free_count, 1, memory_order_relaxed);
        world->free_slots[f_idx] = i;
    }
}


void free_new_buffers(NewBuffers *nb) {
    if (!nb) {
        return;
    }

    // 1. Aligned Buffer Cleanup (Non-atomic)
    CULVERIN_SAFE_FREE_ALIGNED(nb->pos);
    CULVERIN_SAFE_FREE_ALIGNED(nb->rot);
    CULVERIN_SAFE_FREE_ALIGNED(nb->ppos);
    CULVERIN_SAFE_FREE_ALIGNED(nb->prot);
    CULVERIN_SAFE_FREE_ALIGNED(nb->lvel);
    CULVERIN_SAFE_FREE_ALIGNED(nb->avel);

    // 2. Standard Buffer Cleanup
    CULVERIN_SAFE_FREE(nb->bids);
    CULVERIN_SAFE_FREE(nb->udat);

    // 3. ATOMIC Buffer Cleanup
    // nb->gens is CULV_ATOMIC(uint32_t)*
    // nb->stat is CULV_ATOMIC(uint8_t)*
    // We cast to (void*) to ensure standard C-library free works without warnings
    CULVERIN_SAFE_FREE(nb->gens);
    CULVERIN_SAFE_FREE(nb->stat);

    // 4. Mapping Buffer Cleanup (Non-atomic)
    CULVERIN_SAFE_FREE(nb->s2d);
    CULVERIN_SAFE_FREE(nb->d2s);
    CULVERIN_SAFE_FREE(nb->free);
    CULVERIN_SAFE_FREE(nb->cats);
    CULVERIN_SAFE_FREE(nb->masks);
    CULVERIN_SAFE_FREE(nb->mats);
    CULVERIN_SAFE_FREE(nb->softs);

    // Defensive: Zero the struct to prevent Use-After-Free/Double-Free
    *nb = (NewBuffers){};
}

static bool NewBuffers_check_alloc(NewBuffers *nb) {
    // Just a list of the pointers that MUST be there
    return (nb->pos && nb->rot && nb->ppos && nb->prot && nb->lvel && nb->avel && nb->bids &&
            nb->udat && nb->gens && nb->s2d && nb->d2s && nb->stat && nb->free && nb->cats &&
            nb->masks && nb->mats && nb->softs) != 0;
}

CULV_NODISCARD
static int alloc_new_buffers(NewBuffers *nb, size_t cap) {
    *nb = (NewBuffers){};

    // SIMD-Heavy Buffers: AVX alignment (Non-atomic)
    nb->pos  = (JPH_Real *)CulvNative_AlignedMalloc(cap * sizeof(PosStride), AVX_ALIGNMENT);
    nb->ppos = (JPH_Real *)CulvNative_AlignedMalloc(cap * sizeof(PosStride), AVX_ALIGNMENT);
    nb->rot  = (float *)CulvNative_AlignedMalloc(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->prot = (float *)CulvNative_AlignedMalloc(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->lvel = (float *)CulvNative_AlignedMalloc(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->avel = (float *)CulvNative_AlignedMalloc(cap * sizeof(AuxStride), AVX_ALIGNMENT);

    // ATOMIC Data Buffers
    // gens is CULV_ATOMIC(uint32_t)*
    nb->gens = (CULV_ATOMIC(uint32_t) *)CulvNative_Malloc(cap * sizeof(CULV_ATOMIC(uint32_t)));
    // stat is CULV_ATOMIC(uint8_t)*
    nb->stat = (CULV_ATOMIC(uint8_t) *)CulvNative_Malloc(cap * sizeof(CULV_ATOMIC(uint8_t)));

    // Standard Data Buffers
    nb->bids  = (JPH_BodyID *)CulvNative_Malloc(cap * sizeof(JPH_BodyID));
    nb->udat  = (uint64_t *)CulvNative_Calloc(cap, sizeof(uint64_t));
    nb->s2d   = (uint32_t *)CulvNative_Malloc(cap * sizeof(uint32_t));
    nb->d2s   = (uint32_t *)CulvNative_Malloc(cap * sizeof(uint32_t));
    nb->free  = (uint32_t *)CulvNative_Malloc(cap * sizeof(uint32_t));
    nb->cats  = (uint32_t *)CulvNative_Malloc(cap * sizeof(uint32_t));
    nb->masks = (uint32_t *)CulvNative_Malloc(cap * sizeof(uint32_t));
    nb->mats  = (uint32_t *)CulvNative_Calloc(cap, sizeof(uint32_t));
    nb->softs = (SoftBodyShadow *)CulvNative_Calloc(cap, sizeof(SoftBodyShadow));

    // Validation
    if (!NewBuffers_check_alloc(nb)) {
        free_new_buffers(nb);
        return -1;
    }

    return 0;
}

static size_t migrate_and_init(PhysicsWorld *world, NewBuffers *nb, size_t new_cap) {
    // TSan Fix: Load counts atomically
    size_t current_count = atomic_load_explicit(&world->count, memory_order_acquire);
    size_t current_free  = atomic_load_explicit(&world->free_count, memory_order_acquire);

    // 1. Copy Active Dense Data (The bodies currently in simulation)
    if (current_count > 0) {
        // Positions use Stride 4
        size_t pos_bytes = current_count * sizeof(PosStride);
        memcpy(nb->pos, world->positions, pos_bytes);
        memcpy(nb->ppos, world->prev_positions, pos_bytes);

        // Rotations/Velocities use Stride 4
        size_t aux_bytes = current_count * sizeof(AuxStride);
        memcpy(nb->rot, world->rotations, aux_bytes);
        memcpy(nb->prot, world->prev_rotations, aux_bytes);
        memcpy(nb->lvel, world->linear_velocities, aux_bytes);
        memcpy(nb->avel, world->angular_velocities, aux_bytes);

        // Metadata (BodyIDs, UserData, Masks, etc.)
        memcpy(nb->bids, world->body_ids, current_count * sizeof(JPH_BodyID));
        memcpy(nb->udat, world->user_data, current_count * sizeof(uint64_t));
        memcpy(nb->cats, world->categories, current_count * sizeof(uint32_t));
        memcpy(nb->masks, world->masks, current_count * sizeof(uint32_t));
        memcpy(nb->mats, world->material_ids, current_count * sizeof(uint32_t));

        if (world->soft_shadows) {
            memcpy(nb->softs, world->soft_shadows, current_count * sizeof(SoftBodyShadow));
        }

        // Dense-to-Slot mapping
        memcpy(nb->d2s, world->dense_to_slot, current_count * sizeof(uint32_t));
    }

    // 2. Copy Mapping Tables (The Slot-based indirection)
    if (world->slot_capacity > 0) {
        // TSan Fix: Atomic arrays MUST be copied element-by-element
        for (size_t i = 0; i < world->slot_capacity; i++) {
            uint32_t gen = atomic_load_explicit(&world->generations[i], memory_order_relaxed);
            uint8_t stat = atomic_load_explicit(&world->slot_states[i], memory_order_relaxed);

            atomic_init(&nb->gens[i], gen);
            atomic_init(&nb->stat[i], stat);
        }

        // Standard buffers can still use memcpy
        size_t slot_bytes_u32 = world->slot_capacity * sizeof(uint32_t);
        memcpy(nb->s2d, world->slot_to_dense, slot_bytes_u32);

        // Copy the current free list
        if (current_free > 0) {
            memcpy(nb->free, world->free_slots, current_free * sizeof(uint32_t));
        }
    }

    // 3. Initialize Expanded Slots
    size_t local_free_count = current_free;
    for (size_t i = world->slot_capacity; i < new_cap; i++) {
        atomic_init(&nb->gens[i], 1);          // Starting generation
        atomic_init(&nb->stat[i], SLOT_EMPTY); // Starting state
        nb->free[local_free_count++] = (uint32_t)i;
    }
    // Guarantee new body IDs won't trigger false positives in active queries
    for (size_t i = current_count; i < new_cap; i++) {
        nb->bids[i] = JPH_INVALID_BODY_ID;
    }

    return local_free_count;
}

// helper: Allocate shadow buffers and indirection maps
CULV_NODISCARD
int allocate_buffers(PhysicsWorld *world, int max_bodies) {
    // TSan Fix: Load count atomically
    size_t current_count = atomic_load_explicit(&world->count, memory_order_relaxed);

    world->capacity = (size_t)max_bodies;
    if (world->capacity < current_count + SHADOW_CAPACITY_PADDING) {
        world->capacity = current_count + SHADOW_CAPACITY_GROW;
    }
    world->max_jolt_bodies = (uint32_t)max_bodies;

    constexpr auto INITIAL_BODY_CAPACITY = 64;

    size_t initial_cap =
        (max_bodies < INITIAL_BODY_CAPACITY) ? (size_t)max_bodies : INITIAL_BODY_CAPACITY;
    world->capacity      = initial_cap;
    world->slot_capacity = initial_cap;

    // SIMD Aligned Buffers (Not atomic, these are the heavy data buffers)
    world->positions =
        (JPH_Real *)CulvNative_AlignedMalloc(world->capacity * sizeof(PosStride), AVX_ALIGNMENT);
    world->prev_positions =
        (JPH_Real *)CulvNative_AlignedMalloc(world->capacity * sizeof(PosStride), AVX_ALIGNMENT);

    world->rotations =
        (float *)CulvNative_AlignedMalloc(world->capacity * sizeof(AuxStride), AVX_ALIGNMENT);
    world->prev_rotations =
        (float *)CulvNative_AlignedMalloc(world->capacity * sizeof(AuxStride), AVX_ALIGNMENT);
    world->linear_velocities =
        (float *)CulvNative_AlignedMalloc(world->capacity * sizeof(AuxStride), AVX_ALIGNMENT);
    world->angular_velocities =
        (float *)CulvNative_AlignedMalloc(world->capacity * sizeof(AuxStride), AVX_ALIGNMENT);

    world->body_ids =
        (typeof(world->body_ids))CulvNative_Malloc(world->capacity * sizeof(typeof(*world->body_ids)));
    world->user_data =
        (typeof(world->user_data))CulvNative_Calloc(world->capacity, sizeof(typeof(*world->user_data)));
    world->categories = (typeof(world->categories))CulvNative_Malloc(world->capacity *
                                                                 sizeof(typeof(*world->categories)));
    world->masks =
        (typeof(world->masks))CulvNative_Malloc(world->capacity * sizeof(typeof(*world->masks)));
    world->material_ids = (typeof(world->material_ids))CulvNative_Calloc(
        world->capacity, sizeof(typeof(*world->material_ids)));

    world->soft_shadows = (typeof(world->soft_shadows))CulvNative_Calloc(
        world->capacity, sizeof(typeof(*world->soft_shadows)));

    // ATOMIC BUFFER ALLOCATIONS
    // id_to_handle_map is BodyHandle*
    world->id_to_handle_map = (CULV_ATOMIC(typeof_unqual(*world->id_to_handle_map)) *)CulvNative_Malloc(
        (world->max_jolt_bodies + 1) * sizeof(typeof_unqual(*world->id_to_handle_map)));

    world->jolt_body_ptrs = (typeof(world->jolt_body_ptrs))CulvNative_Calloc(
        world->max_jolt_bodies + 1, sizeof(typeof(*world->jolt_body_ptrs)));

    // generations is CULV_ATOMIC(uint32_t)*
    world->generations = (CULV_ATOMIC(uint32_t) *)CulvNative_Malloc(world->slot_capacity *
                                                                 sizeof(CULV_ATOMIC(uint32_t)));

    // slot_states is CULV_ATOMIC(uint8_t)*
    world->slot_states = (CULV_ATOMIC(typeof_unqual(*world->slot_states)) *)CulvNative_Malloc(
        world->slot_capacity * sizeof(CULV_ATOMIC(typeof_unqual(*world->slot_states))));

    // Normal Indirection/Mapping Buffers
    world->slot_to_dense = (uint32_t *)CulvNative_Malloc(world->slot_capacity * sizeof(uint32_t));
    world->dense_to_slot = (uint32_t *)CulvNative_Malloc(world->slot_capacity * sizeof(uint32_t));
    world->free_slots    = (uint32_t *)CulvNative_Malloc(world->slot_capacity * sizeof(uint32_t));

    world->command_queue = (PhysicsCommand *)CulvNative_AlignedMalloc(
        COMMAND_QUEUE_INITIAL_CAPACITY * sizeof(PhysicsCommand), MEMORY_ALIGNMENT_SIZE);
    world->command_queue_spare = (PhysicsCommand *)CulvNative_AlignedMalloc(
        COMMAND_QUEUE_INITIAL_CAPACITY * sizeof(PhysicsCommand), MEMORY_ALIGNMENT_SIZE);
    world->command_capacity = COMMAND_QUEUE_INITIAL_CAPACITY;

    world->trash_capacity = 4;
    world->trash_count    = 0;
    world->trash_buffers  = (NewBuffers *)CulvNative_Calloc(world->trash_capacity, sizeof(NewBuffers));

    world->contact_max_capacity = CONTACT_MAX_CAPACITY;
    world->contact_buffer = (ContactEvent *)CulvNative_AlignedMalloc(
        world->contact_max_capacity * sizeof(ContactEvent), MEMORY_ALIGNMENT_SIZE);

    if (!world->positions || !world->prev_positions || !world->rotations ||
        !world->prev_rotations || !world->linear_velocities || !world->angular_velocities ||
        !world->body_ids || !world->id_to_handle_map || !world->command_queue ||
        !world->command_queue_spare || !world->trash_buffers || !world->contact_buffer ||
        !world->slot_states || !world->generations || !world->free_slots || !world->slot_to_dense ||
        !world->dense_to_slot || !world->user_data || !world->categories || !world->masks ||
        !world->material_ids || !world->soft_shadows || !world->jolt_body_ptrs) {
        return -1;
    }

    // INITIALIZE ATOMIC ARRAY ELEMENTS
    for (size_t i = 0; i <= world->max_jolt_bodies; i++) {
        atomic_init(&world->id_to_handle_map[i], 0);
    }
    for (size_t i = 0; i < world->slot_capacity; i++) {
        atomic_init(&world->generations[i], 0);
        atomic_init(&world->slot_states[i], SLOT_EMPTY);
    }

    // Zero-initialize the aligned buffers
    memset(world->positions, 0, world->capacity * sizeof(PosStride));
    memset(world->prev_positions, 0, world->capacity * sizeof(PosStride));
    memset(world->rotations, 0, world->capacity * sizeof(AuxStride));
    memset(world->prev_rotations, 0, world->capacity * sizeof(AuxStride));
    memset(world->linear_velocities, 0, world->capacity * sizeof(AuxStride));
    memset(world->angular_velocities, 0, world->capacity * sizeof(AuxStride));

    for (size_t i = 0; i < world->capacity; i++) {
        world->categories[i] = ALL_LAYER_BITS;
        world->masks[i]      = ALL_LAYER_BITS;
        world->body_ids[i]   = JPH_INVALID_BODY_ID;
    }

    return 0;
}
PhysicsWorldStatus physics_world_resize(PhysicsWorld *world, size_t new_capacity) {
    if (atomic_load_explicit(&world->view_export_count, memory_order_relaxed) > 0) {
        return PHYSICS_WORLD_STATUS_BUFFER_EXPORTED;
    }

    // --- VACUUM OLD TRASH ---
    // We do the deallocation here, in the mutator thread,
    // before we allocate new memory.
    if (world->trash_count > 0) {
        for (size_t i = 0; i < world->trash_count; i++) {
            free_new_buffers(&world->trash_buffers[i]);
        }
        world->trash_count = 0;
    }

    if (new_capacity > world->max_jolt_bodies) {
        new_capacity = world->max_jolt_bodies;
    }

    if (new_capacity <= world->capacity) {
        return PHYSICS_WORLD_STATUS_OK;
    }

    // 3. Prepare New Buffers
    NewBuffers nb;
    if (alloc_new_buffers(&nb, new_capacity) < 0) {
        return PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
    }

    // 4. Migrate Data
    size_t final_free_count = migrate_and_init(world, &nb, new_capacity);

    // 5. Expand Trash Bin
    if (UNLIKELY(world->trash_count >= world->trash_capacity)) {
        size_t next_cap = (world->trash_capacity == 0) ? 4 : world->trash_capacity * 2;
        void *new_trash = CulvNative_Realloc(world->trash_buffers, next_cap * sizeof(NewBuffers));
        if (!new_trash) {
            free_new_buffers(&nb);
            return PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
        }
        size_t added_elements = next_cap - world->trash_capacity;
        memset((NewBuffers *)new_trash + world->trash_capacity, 0,
               added_elements * sizeof(NewBuffers));
        world->trash_buffers  = (NewBuffers *)new_trash;
        world->trash_capacity = next_cap;
    }

    // 6. THE COMMIT (Swap pointers)
    NewBuffers old_bufs                      = {.pos   = world->positions,
                                                .ppos  = world->prev_positions,
                                                .rot   = world->rotations,
                                                .prot  = world->prev_rotations,
                                                .lvel  = world->linear_velocities,
                                                .avel  = world->angular_velocities,
                                                .bids  = world->body_ids,
                                                .udat  = world->user_data,
                                                .softs = world->soft_shadows,
                                                .gens  = world->generations, // Atomic pointer swap
                                                .s2d   = world->slot_to_dense,
                                                .d2s   = world->dense_to_slot,
                                                .stat  = world->slot_states, // Atomic pointer swap
                                                .free  = world->free_slots,
                                                .cats  = world->categories,
                                                .masks = world->masks,
                                                .mats  = world->material_ids};
    world->trash_buffers[world->trash_count++] = old_bufs;

    world->positions          = nb.pos;
    world->prev_positions     = nb.ppos;
    world->rotations          = nb.rot;
    world->prev_rotations     = nb.prot;
    world->linear_velocities  = nb.lvel;
    world->angular_velocities = nb.avel;
    world->body_ids           = nb.bids;
    world->user_data          = nb.udat;
    world->soft_shadows       = nb.softs;
    world->generations        = nb.gens; // Pointer to new atomic array
    world->slot_to_dense      = nb.s2d;
    world->dense_to_slot      = nb.d2s;
    world->slot_states        = nb.stat; // Pointer to new atomic array
    world->free_slots         = nb.free;
    world->categories         = nb.cats;
    world->masks              = nb.masks;
    world->material_ids       = nb.mats;

    // 7. Update metadata atomically
    atomic_store_explicit(&world->free_count, final_free_count, memory_order_release);
    world->capacity      = new_capacity;
    world->slot_capacity = new_capacity;

    // 8. Signal End

    return PHYSICS_WORLD_STATUS_OK;
}

static PhysicsWorldStatus reserve_body_slot(PhysicsWorld *world, uint32_t slot_state,
                                             BodyHandle *handle_out) {
    const size_t current_count = atomic_load_explicit(&world->count, memory_order_acquire);
    size_t available = atomic_load_explicit(&world->free_count, memory_order_acquire);

    if (UNLIKELY(current_count >= world->max_jolt_bodies)) {
        return PHYSICS_WORLD_STATUS_LIMIT_REACHED;
    }

    if (UNLIKELY(available == 0 || current_count + 1 > world->capacity)) {
        size_t next_capacity = world->capacity == 0 ? INITIAL_BODY_CAPACITY : world->capacity * 2;
        if (next_capacity > world->max_jolt_bodies) {
            next_capacity = world->max_jolt_bodies;
        }
        const PhysicsWorldStatus resize_status = physics_world_resize(world, next_capacity);
        if (resize_status != PHYSICS_WORLD_STATUS_OK) {
            return resize_status;
        }
        available = atomic_load_explicit(&world->free_count, memory_order_acquire);
        if (UNLIKELY(available == 0)) {
            return PHYSICS_WORLD_STATUS_LIMIT_REACHED;
        }
    }

    if (UNLIKELY(!ensure_command_capacity(world))) {
        return PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
    }

    const uint32_t slot = world->free_slots[--available];
    const uint32_t dense =
        (uint32_t)atomic_fetch_add_explicit(&world->count, 1, memory_order_relaxed);
    atomic_store_explicit(&world->free_count, available, memory_order_release);

    const uint32_t generation =
        atomic_load_explicit(&world->generations[slot], memory_order_relaxed);
    const BodyHandle handle = make_handle(slot, generation);

    world->slot_to_dense[slot] = dense;
    world->dense_to_slot[dense] = slot;
    world->body_ids[dense] = JPH_INVALID_BODY_ID;
    atomic_store_explicit(&world->slot_states[slot], slot_state, memory_order_release);

    *handle_out = handle;
    return PHYSICS_WORLD_STATUS_OK;
}

PhysicsWorldStatus physics_world_commit_create(PhysicsWorld *world,
                                               JPH_BodyCreationSettings *settings,
                                               uint32_t slot_state, BodyHandle *out_handle) {
    if (!world || !settings || !out_handle) {
        return PHYSICS_WORLD_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = 0;

    BodyHandle handle = 0;
    const PhysicsWorldStatus status = reserve_body_slot(world, slot_state, &handle);
    if (status != PHYSICS_WORLD_STATUS_OK) {
        return status;
    }

    JPH_BodyCreationSettings_SetUserData(settings, handle);
    *out_handle = handle;
    return PHYSICS_WORLD_STATUS_OK;
}

PhysicsWorldStatus physics_world_commit_create_soft(PhysicsWorld *world,
                                                    JPH_SoftBodyCreationSettings *settings,
                                                    uint32_t slot_state, BodyHandle *out_handle) {
    if (!world || !settings || !out_handle) {
        return PHYSICS_WORLD_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = 0;

    BodyHandle handle = 0;
    const PhysicsWorldStatus status = reserve_body_slot(world, slot_state, &handle);
    if (status != PHYSICS_WORLD_STATUS_OK) {
        return status;
    }

    JPH_SoftBodyCreationSettings_SetUserData(settings, handle);
    *out_handle = handle;
    return PHYSICS_WORLD_STATUS_OK;
}

void free_constraints(PhysicsWorld *world) {
    if (world->constraints) {
        for (size_t i = 0; i < world->constraint_capacity; i++) {
            if (!world->constraints[i]) {
                continue;
            }

            bool is_alive = (!world->constraint_states ||
                             atomic_load_explicit(&world->constraint_states[i],
                                                  memory_order_relaxed) == SLOT_ALIVE) != 0;
            if (is_alive) {
                if (world->system) {
                    JPH_PhysicsSystem_RemoveConstraint(world->system, world->constraints[i]);
                }
                JPH_Constraint_Destroy(world->constraints[i]);
            }
            world->constraints[i] = nullptr;
        }
        CULVERIN_SAFE_FREE(world->constraints);
    }
    CULVERIN_SAFE_FREE(world->constraint_generations);
    CULVERIN_SAFE_FREE(world->free_constraint_slots);
    CULVERIN_SAFE_FREE(world->constraint_states);
    CULVERIN_SAFE_FREE(world->constraint_types);
}

void free_shadow_buffers(PhysicsWorld *world) {
    // 1. Aligned buffers (stride types)
    CULVERIN_SAFE_FREE_ALIGNED(world->positions);
    CULVERIN_SAFE_FREE_ALIGNED(world->prev_positions);
    CULVERIN_SAFE_FREE_ALIGNED(world->rotations);
    CULVERIN_SAFE_FREE_ALIGNED(world->prev_rotations);
    CULVERIN_SAFE_FREE_ALIGNED(world->linear_velocities);
    CULVERIN_SAFE_FREE_ALIGNED(world->angular_velocities);

    // 2. ATOMIC buffers
    // Generations is CULV_ATOMIC(uint32_t)*
    // Slot States is CULV_ATOMIC(uint8_t)*
    CULVERIN_SAFE_FREE(world->generations);
    CULVERIN_SAFE_FREE(world->slot_states);

    // 3. Regular buffers
    CULVERIN_SAFE_FREE(world->body_ids);
    CULVERIN_SAFE_FREE(world->slot_to_dense);
    CULVERIN_SAFE_FREE(world->dense_to_slot);
    CULVERIN_SAFE_FREE(world->free_slots);
    CULVERIN_SAFE_FREE_ALIGNED(world->command_queue);
    CULVERIN_SAFE_FREE(world->user_data);
    CULVERIN_SAFE_FREE(world->categories);

    if (world->soft_shadows) {
        for (size_t i = 0; i < world->capacity; i++) {
            CULVERIN_SAFE_FREE_ALIGNED(world->soft_shadows[i].vertices);
            CULVERIN_SAFE_FREE_ALIGNED(world->soft_shadows[i].velocities);
            CULVERIN_SAFE_FREE_ALIGNED(world->soft_shadows[i].normals);
        }
        CULVERIN_SAFE_FREE(world->soft_shadows);
    }
    CULVERIN_SAFE_FREE(world->masks);
    CULVERIN_SAFE_FREE(world->material_ids);
    CULVERIN_SAFE_FREE(world->materials);
    CULVERIN_SAFE_FREE(world->jolt_body_ptrs);
}

void physics_world_free_native(PhysicsWorld *world) {
    if (!world) {
        return;
    }

    clear_command_queue(world);
    free_constraints(world);

    if (world->system) {
        JPH_PhysicsSystem_Destroy(world->system);
        world->system        = nullptr;
        world->body_interface = nullptr;
        // JoltC owns these three interfaces after PhysicsSystem_Create succeeds.
        world->bp_interface = nullptr;
        world->pair_filter  = nullptr;
        world->bp_filter    = nullptr;
    } else {
        // Initialization can fail before ownership transfers to JPH_PhysicsSystem.
        if (world->bp_filter) {
            JPH_ObjectVsBroadPhaseLayerFilter_Destroy(world->bp_filter);
            world->bp_filter = nullptr;
        }
        if (world->pair_filter) {
            JPH_ObjectLayerPairFilter_Destroy(world->pair_filter);
            world->pair_filter = nullptr;
        }
        if (world->bp_interface) {
            JPH_BroadPhaseLayerInterface_Destroy(world->bp_interface);
            world->bp_interface = nullptr;
        }
    }
    if (world->char_vs_char_manager) {
        JPH_CharacterVsCharacterCollision_Destroy(world->char_vs_char_manager);
        world->char_vs_char_manager = nullptr;
    }
    if (world->job_system) {
        JPH_JobSystem_Destroy(world->job_system);
        world->job_system = nullptr;
    }
    if (world->temp_allocator) {
        JPH_TempAllocator_Destroy(world->temp_allocator);
        world->temp_allocator = nullptr;
    }

    if (world->debug_renderer) {
        JPH_DebugRenderer_Destroy(world->debug_renderer);
        world->debug_renderer = nullptr;
    }
    debug_buffer_free(&world->debug_lines);
    debug_buffer_free(&world->debug_triangles);
    free_shape_cache(world);

    if (world->contact_listener) {
        JPH_ContactListener_Destroy(world->contact_listener);
        world->contact_listener = nullptr;
    }
    CULVERIN_SAFE_FREE_ALIGNED(world->contact_buffer);
    CULVERIN_SAFE_FREE_ALIGNED(world->contact_events);

    if (world->trash_buffers) {
        for (size_t i = 0; i < world->trash_count; i++) {
            free_new_buffers(&world->trash_buffers[i]);
        }
        CULVERIN_SAFE_FREE(world->trash_buffers);
    }
    world->trash_count = 0;

    CULVERIN_SAFE_FREE_ALIGNED(world->command_queue_spare);
    free_shadow_buffers(world);
    CULVERIN_SAFE_FREE(world->id_to_handle_map);

    FREE_LOCK(world->shadow_lock);
    FREE_NATIVE_MUTEX(world->jph_trampoline_lock);
    FREE_NATIVE_MUTEX(world->step_sync.mutex);
    FREE_NATIVE_COND(world->step_sync.cond);
}

int init_jolt_core(PhysicsWorld *world, WorldSettings settings, GravityVector gravity) {
    int num_workers = settings.num_threads;
#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    // When running ThreadSanitizer, disable Jolt's background workers.
    // TSan cannot understand Jolt's highly-optimized C++ lock-free job queues
    // and will generate false positives. Running Jolt synchronously allows TSan
    // to focus exclusively on finding real concurrency bugs in the Culverin/Python layer.
    num_workers = 0;
#endif
    JobSystemThreadPoolConfig job_cfg = {.maxJobs     = (uint32_t)settings.max_physics_jobs,
                                         .maxBarriers = (uint32_t)settings.max_physics_barriers,
                                         .numThreads  = num_workers};

    // TSan Fix: Serialize the first PhysicsSystem creation.
    // This allows Jolt's internal lazy-statics to initialize safely.
    NATIVE_MUTEX_LOCK(g_jph_init_lock);
    world->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
    if (!world->job_system) {
        NATIVE_MUTEX_UNLOCK(g_jph_init_lock);
        return -1;
    }

#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    world->temp_allocator = JPH_TempAllocatorMalloc_Create();
#else
    world->temp_allocator = JPH_TempAllocator_Create((uint32_t)settings.temp_allocator_size);
#endif
    if (!world->temp_allocator) {
        NATIVE_MUTEX_UNLOCK(g_jph_init_lock);
        return -1;
    }

    // --- 3 LAYERS: 0=Static, 1=Dynamic, 2=VehicleRay ---
    world->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(OBJECT_LAYER_COUNT, 3);
    if (!world->bp_interface) {
        NATIVE_MUTEX_UNLOCK(g_jph_init_lock);
        return -1;
    }
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(world->bp_interface,
                                                                 OBJECT_LAYER_STATIC, 0);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(world->bp_interface,
                                                                 OBJECT_LAYER_DYNAMIC, 1);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(world->bp_interface,
                                                                 OBJECT_LAYER_VEHICLE, 2);

    world->pair_filter = JPH_ObjectLayerPairFilterTable_Create(3);
    if (!world->pair_filter) {
        NATIVE_MUTEX_UNLOCK(g_jph_init_lock);
        return -1;
    }

    // Matrix:
    // 0 (Static)  vs 1 (Dynamic) -> ON
    // 0 (Static)  vs 2 (Ray)     -> ON
    // 1 (Dynamic) vs 1 (Dynamic) -> ON
    // 1 (Dynamic) vs 2 (Ray)     -> OFF (Fixes self-collision)
    // 2 (Ray)     vs 2 (Ray)     -> OFF
    JPH_ObjectLayerPairFilterTable_EnableCollision(world->pair_filter, 0, 1);
    JPH_ObjectLayerPairFilterTable_EnableCollision(world->pair_filter, 0, 2);
    JPH_ObjectLayerPairFilterTable_EnableCollision(world->pair_filter, 1, 1);
    JPH_ObjectLayerPairFilterTable_DisableCollision(world->pair_filter, 1, 2);

    world->bp_filter =
        JPH_ObjectVsBroadPhaseLayerFilterTable_Create(world->bp_interface, 3, world->pair_filter, 3);
    if (!world->bp_filter) {
        NATIVE_MUTEX_UNLOCK(g_jph_init_lock);
        return -1;
    }

    JPH_PhysicsSystemSettings phys_settings = {.maxBodies    = (uint32_t)settings.max_bodies,
                                               .maxBodyPairs = (uint32_t)settings.max_pairs,
                                               .maxContactConstraints =
                                                   (uint32_t)settings.max_contact_constraints,
                                               .broadPhaseLayerInterface      = world->bp_interface,
                                               .objectLayerPairFilter         = world->pair_filter,
                                               .objectVsBroadPhaseLayerFilter = world->bp_filter};

    world->system = JPH_PhysicsSystem_Create(&phys_settings);
    NATIVE_MUTEX_UNLOCK(g_jph_init_lock);
    if (!world->system) {
        return -1;
    }

    world->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
    if (!world->char_vs_char_manager) {
        return -1;
    }
    JPH_PhysicsSystem_SetGravity(world->system, &(JPH_Vec3){gravity.gx, gravity.gy, gravity.gz});
    world->body_interface = JPH_PhysicsSystem_GetBodyInterface(world->system);
    return world->body_interface ? 0 : -1;
}

PhysicsCommand *physics_world_detach_command_queue(PhysicsWorld *world, size_t *command_count) {
    PhysicsCommand *queue = world->command_queue;
    *command_count = world->command_count;

    world->command_queue = world->command_queue_spare;
    world->command_queue_spare = queue;
    world->command_count = 0;
    return queue;
}

void physics_world_step_native(PhysicsWorld *world, float dt, PhysicsCommand *queue,
                               size_t command_count) {
    NATIVE_MUTEX_LOCK(world->jph_trampoline_lock);

    CULV_PROFILE_BEGIN(jolt_step);

    if (command_count > 0) {
        flush_commands_internal(world, queue, command_count);
        world->needs_optimization = true;
    }

    if (dt <= 0.0f) {
        JPH_PhysicsSystem_OptimizeBroadPhase(world->system);
        world->needs_optimization = false;
    } else {
        JPH_PhysicsSystem_Update2(world->system, dt, 1, world->temp_allocator, world->job_system);
        if (world->needs_optimization) {
            JPH_PhysicsSystem_OptimizeBroadPhase(world->system);
            world->needs_optimization = false;
        }
    }

    // Wait for exported query views to finish before writing the shadow buffers.
    NATIVE_MUTEX_LOCK(world->step_sync.mutex);
    while (atomic_load_explicit(&world->active_queries, memory_order_acquire) > 0) {
        NATIVE_COND_WAIT(world->step_sync.cond, world->step_sync.mutex);
    }
    NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);

    culverin_sync_shadow_buffers(world);

    CULV_PROFILE_END(jolt_step, "Jolt Physics Crunch", (unsigned int)command_count);
    NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
}

size_t physics_world_finalize_step(PhysicsWorld *world, float dt) {
    const size_t contact_count =
        atomic_load_explicit(&world->contact_atomic_idx, memory_order_acquire);
    world->contact_count =
        contact_count > world->contact_max_capacity ? world->contact_max_capacity : contact_count;

    const size_t body_count = atomic_load_explicit(&world->count, memory_order_acquire);
    world->time += (double)dt;

    atomic_store_explicit(&world->is_stepping, false, memory_order_release);
    atomic_store_explicit(&world->step_requested, false, memory_order_release);

    NATIVE_MUTEX_LOCK(world->step_sync.mutex);
    NATIVE_COND_BROADCAST(world->step_sync.cond);
    NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);

    return body_count;
}

int verify_abi_alignment(JPH_BodyInterface *bi) {
    JPH_BoxShapeSettings *bs = JPH_BoxShapeSettings_Create(&(JPH_Vec3){1, 1, 1}, 0.0f);
    auto shape               = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(bs);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)bs);
    if (!shape) {
        return -1;
    }

    JPH_BodyCreationSettings *bcs = JPH_BodyCreationSettings_Create3(
        shape, &(JPH_RVec3){VERIFY_ABI_POS_X, VERIFY_ABI_POS_Y, VERIFY_ABI_POS_Z},
        &(JPH_Quat){0, 0, 0, 1}, JPH_MotionType_Static, 0);
    JPH_Shape_Destroy(shape);
    if (!bcs) {
        return -1;
    }

    JPH_BodyID bid = JPH_BodyInterface_CreateAndAddBody(bi, bcs, JPH_Activation_Activate);
    JPH_BodyCreationSettings_Destroy(bcs);

    JPH_STACK_ALLOC(JPH_RVec3, p_check);
    JPH_BodyInterface_GetPosition(bi, bid, p_check);
    JPH_BodyInterface_RemoveBody(bi, bid);
    JPH_BodyInterface_DestroyBody(bi, bid);

    if (fabs(p_check->x - VERIFY_ABI_POS_X) > VERIFY_ABI_TOLERANCE ||
        fabs(p_check->y - VERIFY_ABI_POS_Y) > VERIFY_ABI_TOLERANCE) {
        return -1;
    }
    return 0;
}