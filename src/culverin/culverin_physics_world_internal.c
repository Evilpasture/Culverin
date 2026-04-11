#include "culverin_physics_world_internal.h"
#include "culverin_compiler_specifics.h"
#include "culverin_physics_sync.h"
#include "culverin_threading.h"

// Constants for buffer alignment, capacity growth and defaults
static constexpr size_t AVX_ALIGNMENT                  = 32; // 32-byte alignment for AVX-256
static constexpr size_t SHADOW_CAPACITY_PADDING        = 128;
static constexpr size_t SHADOW_CAPACITY_GROW           = 1024;
static constexpr size_t COMMAND_QUEUE_INITIAL_CAPACITY = 64;
static constexpr uint32_t ALL_LAYER_BITS               = 0xFFFFu;
static constexpr uint32_t MAX_CONTACT_CONSTRAINTS      = 1024u * 32u;
static constexpr uint32_t JOB_SYSTEM_MAX_JOBS          = 1024u;
static constexpr uint32_t JOB_SYSTEM_MAX_BARRIERS      = 8u;

// Baked scene formats (tuple indices and per-object strides)
static constexpr int BAKED_INDEX_POS      = 1;
static constexpr int BAKED_INDEX_ROT      = 2;
static constexpr int BAKED_INDEX_SHAPE    = 3;
static constexpr int BAKED_INDEX_MOT      = 4;
static constexpr int BAKED_INDEX_LAYER    = 5;
static constexpr int BAKED_INDEX_USERDATA = 6;
static constexpr int BAKED_POS_STRIDE     = 4;
static constexpr int BAKED_ROT_STRIDE     = 4;
static constexpr int BAKED_SHAPE_STRIDE   = 5;

// ABI verification constants
static constexpr JPH_Real VERIFY_ABI_POS_X   = 10.0;
static constexpr JPH_Real VERIFY_ABI_POS_Y   = 20.0;
static constexpr JPH_Real VERIFY_ABI_POS_Z   = 30.0;
static constexpr double VERIFY_ABI_TOLERANCE = 0.1;

void free_new_buffers(NewBuffers *nb) {
    if (!nb) {
        return;
    }

    // 1. Aligned Buffer Cleanup (Non-atomic)
    CulvMem_RawFreeAligned(nb->pos);
    CulvMem_RawFreeAligned(nb->rot);
    CulvMem_RawFreeAligned(nb->ppos);
    CulvMem_RawFreeAligned(nb->prot);
    CulvMem_RawFreeAligned(nb->lvel);
    CulvMem_RawFreeAligned(nb->avel);

    // 2. Standard Buffer Cleanup
    CULV_RAW_FREE(nb->bids);
    CULV_RAW_FREE(nb->udat);
    
    // 3. ATOMIC Buffer Cleanup
    // nb->gens is _Atomic uint32_t*
    // nb->stat is _Atomic uint8_t*
    // We cast to (void*) to ensure standard C-library free works without warnings
    CULV_RAW_FREE((void *)nb->gens);
    CULV_RAW_FREE((void *)nb->stat);

    // 4. Mapping Buffer Cleanup (Non-atomic)
    CULV_RAW_FREE(nb->s2d);
    CULV_RAW_FREE(nb->d2s);
    CULV_RAW_FREE(nb->free);
    CULV_RAW_FREE(nb->cats);
    CULV_RAW_FREE(nb->masks);
    CULV_RAW_FREE(nb->mats);

    // Defensive: Zero the struct to prevent Use-After-Free/Double-Free
    memset(nb, 0, sizeof(NewBuffers));
}
CULV_NODISCARD
static int alloc_new_buffers(NewBuffers *nb, size_t cap) {
    memset(nb, 0, sizeof(NewBuffers));

    // SIMD-Heavy Buffers: AVX alignment (Non-atomic)
    nb->pos  = (JPH_Real *)CulvMem_RawMallocAligned(cap * sizeof(PosStride), AVX_ALIGNMENT);
    nb->ppos = (JPH_Real *)CulvMem_RawMallocAligned(cap * sizeof(PosStride), AVX_ALIGNMENT);
    nb->rot  = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->prot = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->lvel = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->avel = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);

    // ATOMIC Data Buffers
    // gens is _Atomic uint32_t*
    nb->gens = (_Atomic uint32_t *)CULV_RAW_MALLOC(cap * sizeof(_Atomic uint32_t));
    // stat is _Atomic uint8_t*
    nb->stat = (_Atomic uint8_t *)CULV_RAW_MALLOC(cap * sizeof(_Atomic uint8_t));

    // Standard Data Buffers
    nb->bids  = (JPH_BodyID *)CULV_RAW_MALLOC(cap * sizeof(JPH_BodyID));
    nb->udat  = (uint64_t *)CULV_RAW_CALLOC(cap, sizeof(uint64_t));
    nb->s2d   = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->d2s   = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->free  = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->cats  = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->masks = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->mats  = (uint32_t *)CULV_RAW_CALLOC(cap, sizeof(uint32_t));

    // Validation
    if (!nb->pos || !nb->rot || !nb->ppos || !nb->prot || !nb->lvel || !nb->avel || 
        !nb->bids || !nb->udat || !nb->gens || !nb->s2d || !nb->d2s || 
        !nb->stat || !nb->free || !nb->cats || !nb->masks || !nb->mats) {
        free_new_buffers(nb);
        return -1;
    }
    
    return 0;
}

static size_t migrate_and_init(PhysicsWorldObject *self, NewBuffers *nb, size_t new_cap) {
    // TSan Fix: Load counts atomically
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    size_t current_free  = atomic_load_explicit(&self->free_count, memory_order_acquire);

    // 1. Copy Active Dense Data (The bodies currently in simulation)
    if (current_count > 0) {
        // Positions use Stride 4
        size_t pos_bytes = current_count * sizeof(PosStride);
        memcpy(nb->pos, self->positions, pos_bytes);
        memcpy(nb->ppos, self->prev_positions, pos_bytes);

        // Rotations/Velocities use Stride 4
        size_t aux_bytes = current_count * sizeof(AuxStride);
        memcpy(nb->rot, self->rotations, aux_bytes);
        memcpy(nb->prot, self->prev_rotations, aux_bytes);
        memcpy(nb->lvel, self->linear_velocities, aux_bytes);
        memcpy(nb->avel, self->angular_velocities, aux_bytes);

        // Metadata (BodyIDs, UserData, Masks, etc.)
        memcpy(nb->bids, self->body_ids, current_count * sizeof(JPH_BodyID));
        memcpy(nb->udat, self->user_data, current_count * sizeof(uint64_t));
        memcpy(nb->cats, self->categories, current_count * sizeof(uint32_t));
        memcpy(nb->masks, self->masks, current_count * sizeof(uint32_t));
        memcpy(nb->mats, self->material_ids, current_count * sizeof(uint32_t));

        // Dense-to-Slot mapping
        memcpy(nb->d2s, self->dense_to_slot, current_count * sizeof(uint32_t));
    }

    // 2. Copy Mapping Tables (The Slot-based indirection)
    if (self->slot_capacity > 0) {
        // TSan Fix: Atomic arrays MUST be copied element-by-element
        for (size_t i = 0; i < self->slot_capacity; i++) {
            uint32_t gen = atomic_load_explicit(&self->generations[i], memory_order_relaxed);
            uint8_t stat = atomic_load_explicit(&self->slot_states[i], memory_order_relaxed);
            
            atomic_init(&nb->gens[i], gen);
            atomic_init(&nb->stat[i], stat);
        }

        // Standard buffers can still use memcpy
        size_t slot_bytes_u32 = self->slot_capacity * sizeof(uint32_t);
        memcpy(nb->s2d, self->slot_to_dense, slot_bytes_u32);

        // Copy the current free list
        if (current_free > 0) {
            memcpy(nb->free, self->free_slots, current_free * sizeof(uint32_t));
        }
    }

    // 3. Initialize Expanded Slots
    size_t local_free_count = current_free;
    for (size_t i = self->slot_capacity; i < new_cap; i++) {
        atomic_init(&nb->gens[i], 1);           // Starting generation
        atomic_init(&nb->stat[i], SLOT_EMPTY);  // Starting state
        nb->free[local_free_count++] = (uint32_t)i;
    }

    return local_free_count;
}

// helper: Allocate shadow buffers and indirection maps
CULV_NODISCARD
int allocate_buffers(PhysicsWorldObject *self, int max_bodies) {
    // TSan Fix: Load count atomically
    size_t current_count = atomic_load_explicit(&self->count, memory_order_relaxed);

    self->capacity = (size_t)max_bodies;
    if (self->capacity < current_count + SHADOW_CAPACITY_PADDING) {
        self->capacity = current_count + SHADOW_CAPACITY_GROW;
    }
    self->max_jolt_bodies = (uint32_t)max_bodies;

    size_t initial_cap = (max_bodies < 64) ? (size_t)max_bodies : 64;
    self->capacity = initial_cap;
    self->slot_capacity = initial_cap;

    // SIMD Aligned Buffers (Not atomic, these are the heavy data buffers)
    self->positions =
        (JPH_Real *)CulvMem_RawMallocAligned(self->capacity * sizeof(PosStride), AVX_ALIGNMENT);
    self->prev_positions =
        (JPH_Real *)CulvMem_RawMallocAligned(self->capacity * sizeof(PosStride), AVX_ALIGNMENT);

    self->rotations =
        (float *)CulvMem_RawMallocAligned(self->capacity * sizeof(AuxStride), AVX_ALIGNMENT);
    self->prev_rotations =
        (float *)CulvMem_RawMallocAligned(self->capacity * sizeof(AuxStride), AVX_ALIGNMENT);
    self->linear_velocities =
        (float *)CulvMem_RawMallocAligned(self->capacity * sizeof(AuxStride), AVX_ALIGNMENT);
    self->angular_velocities =
        (float *)CulvMem_RawMallocAligned(self->capacity * sizeof(AuxStride), AVX_ALIGNMENT);

    self->body_ids     = (JPH_BodyID *)CULV_RAW_MALLOC(self->capacity * sizeof(JPH_BodyID));
    self->user_data    = (uint64_t *)CULV_RAW_CALLOC(self->capacity, sizeof(uint64_t));
    self->categories   = (uint32_t *)CULV_RAW_MALLOC(self->capacity * sizeof(uint32_t));
    self->masks        = (uint32_t *)CULV_RAW_MALLOC(self->capacity * sizeof(uint32_t));
    self->material_ids = (uint32_t *)CULV_RAW_CALLOC(self->capacity, sizeof(uint32_t));

    // ATOMIC BUFFER ALLOCATIONS
    // id_to_handle_map is _Atomic BodyHandle*
    self->id_to_handle_map = (_Atomic BodyHandle *)CULV_RAW_MALLOC((self->max_jolt_bodies + 1) * sizeof(BodyHandle));
    
    // generations is _Atomic uint32_t*
    self->generations   = (_Atomic uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(_Atomic uint32_t));
    
    // slot_states is _Atomic uint8_t*
    self->slot_states   = (_Atomic uint8_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(_Atomic uint8_t));

    // Normal Indirection/Mapping Buffers
    self->slot_to_dense = (uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(uint32_t));
    self->dense_to_slot = (uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(uint32_t));
    self->free_slots    = (uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(uint32_t));

    self->command_queue =
        (PhysicsCommand *)CULV_RAW_MALLOC(COMMAND_QUEUE_INITIAL_CAPACITY * sizeof(PhysicsCommand));
    self->command_capacity = COMMAND_QUEUE_INITIAL_CAPACITY;

    self->trash_capacity = 4;
    self->trash_count    = 0;
    self->trash_buffers  = (NewBuffers *)CULV_RAW_CALLOC(self->trash_capacity, sizeof(NewBuffers));

    if (!self->positions || !self->rotations || !self->id_to_handle_map || !self->command_queue ||
        !self->slot_states || !self->generations) {
        return -1;
    }

    // INITIALIZE ATOMIC ARRAY ELEMENTS
    for (size_t i = 0; i <= self->max_jolt_bodies; i++) {
        atomic_init(&self->id_to_handle_map[i], 0);
    }
    for (size_t i = 0; i < self->slot_capacity; i++) {
        atomic_init(&self->generations[i], 0);
        atomic_init(&self->slot_states[i], SLOT_EMPTY);
    }

    // Zero-initialize the aligned buffers
    memset(self->positions, 0, self->capacity * sizeof(PosStride));
    memset(self->prev_positions, 0, self->capacity * sizeof(PosStride));
    memset(self->rotations, 0, self->capacity * sizeof(AuxStride));
    memset(self->prev_rotations, 0, self->capacity * sizeof(AuxStride));
    memset(self->linear_velocities, 0, self->capacity * sizeof(AuxStride));
    memset(self->angular_velocities, 0, self->capacity * sizeof(AuxStride));

    for (size_t i = 0; i < self->capacity; i++) {
        self->categories[i] = ALL_LAYER_BITS;
        self->masks[i]      = ALL_LAYER_BITS;
    }
    
    return 0;
}
CULV_NODISCARD
int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
    // 1. Signal Start
    atomic_store_explicit(&self->is_resizing, true, memory_order_release);

    if (atomic_load_explicit(&self->view_export_count, memory_order_relaxed) > 0) {
        atomic_store_explicit(&self->is_resizing, false, memory_order_relaxed);
        PyErr_SetString(PyExc_BufferError, "Cannot resize while memoryview is active.");
        return -1;
    }

    // 2. Concurrency Guard
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    if (new_capacity > self->max_jolt_bodies) {
        new_capacity = self->max_jolt_bodies;
    }

    if (new_capacity <= self->capacity) {
        atomic_store_explicit(&self->is_resizing, false, memory_order_relaxed);
        return 0;
    }

    // 3. Prepare New Buffers
    NewBuffers nb;
    if (alloc_new_buffers(&nb, new_capacity) < 0) {
        atomic_store_explicit(&self->is_resizing, false, memory_order_relaxed);
        PyErr_NoMemory();
        return -1;
    }

    // 4. Migrate Data
    size_t final_free_count = migrate_and_init(self, &nb, new_capacity);

    // 5. Expand Trash Bin
    if (self->trash_count >= self->trash_capacity) {
        size_t next_cap = (self->trash_capacity == 0) ? 4 : self->trash_capacity * 2;
        void *new_trash = CULV_RAW_REALLOC(self->trash_buffers, next_cap * sizeof(NewBuffers));
        if (!new_trash) {
            free_new_buffers(&nb);
            atomic_store_explicit(&self->is_resizing, false, memory_order_relaxed);
            return -1;
        }
        size_t added_elements = next_cap - self->trash_capacity;
        memset((NewBuffers *)new_trash + self->trash_capacity, 0, added_elements * sizeof(NewBuffers));
        self->trash_buffers  = (NewBuffers *)new_trash;
        self->trash_capacity = next_cap;
    }

    // 6. THE COMMIT (Swap pointers)
    NewBuffers old_bufs = {
        .pos   = self->positions,
        .ppos  = self->prev_positions,
        .rot   = self->rotations,
        .prot  = self->prev_rotations,
        .lvel  = self->linear_velocities,
        .avel  = self->angular_velocities,
        .bids  = self->body_ids,
        .udat  = self->user_data,
        .gens  = self->generations, // Atomic pointer swap
        .s2d   = self->slot_to_dense,
        .d2s   = self->dense_to_slot,
        .stat  = self->slot_states,  // Atomic pointer swap
        .free  = self->free_slots,
        .cats  = self->categories,
        .masks = self->masks,
        .mats  = self->material_ids
    };
    self->trash_buffers[self->trash_count++] = old_bufs;

    self->positions          = nb.pos;
    self->prev_positions     = nb.ppos;
    self->rotations          = nb.rot;
    self->prev_rotations     = nb.prot;
    self->linear_velocities  = nb.lvel;
    self->angular_velocities = nb.avel;
    self->body_ids           = nb.bids;
    self->user_data          = nb.udat;
    self->generations        = nb.gens; // Pointer to new atomic array
    self->slot_to_dense      = nb.s2d;
    self->dense_to_slot      = nb.d2s;
    self->slot_states        = nb.stat; // Pointer to new atomic array
    self->free_slots         = nb.free;
    self->categories         = nb.cats;
    self->masks              = nb.masks;
    self->material_ids       = nb.mats;

    // 7. Update metadata atomically
    atomic_store_explicit(&self->free_count, final_free_count, memory_order_release);
    self->capacity      = new_capacity;
    self->slot_capacity = new_capacity;

    // 8. Signal End
    atomic_store_explicit(&self->is_resizing, false, memory_order_release);

    return 0;
}

void free_constraints(PhysicsWorldObject *self) {
    if (self->constraints) {
        for (size_t i = 0; i < self->constraint_capacity; i++) {
            if (!self->constraints[i]) {
                continue;
            }

            bool is_alive =
                (!self->constraint_states || self->constraint_states[i] == SLOT_ALIVE) != 0;
            if (is_alive) {
                if (self->system) {
                    JPH_PhysicsSystem_RemoveConstraint(self->system, self->constraints[i]);
                }
                JPH_Constraint_Destroy(self->constraints[i]);
            }
            self->constraints[i] = nullptr;
        }
        CULV_RAW_FREE((void *)self->constraints);
        self->constraints = nullptr;
    }
    CULV_RAW_FREE(self->constraint_generations);
    self->constraint_generations = nullptr;
    CULV_RAW_FREE(self->free_constraint_slots);
    self->free_constraint_slots = nullptr;
    CULV_RAW_FREE(self->constraint_states);
    self->constraint_states = nullptr;
}

void free_shadow_buffers(PhysicsWorldObject *self) {
    // 1. Aligned buffers (stride types)
    CulvMem_RawFreeAligned(self->positions);
    self->positions = nullptr;
    CulvMem_RawFreeAligned(self->prev_positions);
    self->prev_positions = nullptr;
    CulvMem_RawFreeAligned(self->rotations);
    self->rotations = nullptr;
    CulvMem_RawFreeAligned(self->prev_rotations);
    self->prev_rotations = nullptr;
    CulvMem_RawFreeAligned(self->linear_velocities);
    self->linear_velocities = nullptr;
    CulvMem_RawFreeAligned(self->angular_velocities);
    self->angular_velocities = nullptr;

    // 2. ATOMIC buffers
    // Generations is _Atomic uint32_t*
    // Slot States is _Atomic uint8_t*
    CULV_RAW_FREE((void *)self->generations);
    self->generations = nullptr;
    CULV_RAW_FREE((void *)self->slot_states);
    self->slot_states = nullptr;

    // 3. Regular buffers
    CULV_RAW_FREE(self->body_ids);
    self->body_ids = nullptr;
    CULV_RAW_FREE(self->slot_to_dense);
    self->slot_to_dense = nullptr;
    CULV_RAW_FREE(self->dense_to_slot);
    self->dense_to_slot = nullptr;
    CULV_RAW_FREE(self->free_slots);
    self->free_slots = nullptr;
    CULV_RAW_FREE(self->command_queue);
    self->command_queue = nullptr;
    CULV_RAW_FREE(self->user_data);
    self->user_data = nullptr;
    CULV_RAW_FREE(self->categories);
    self->categories = nullptr;
    CULV_RAW_FREE(self->masks);
    self->masks = nullptr;
    CULV_RAW_FREE(self->material_ids);
    self->material_ids = nullptr;
    CULV_RAW_FREE(self->materials);
    self->materials = nullptr;
}

// --- Helper: Resource Cleanup (Idempotent) ---
// SAFETY:
// - Must not be called while PhysicsSystem is stepping
// - Must not be called from a Jolt callback
// - Must not race with Python memoryview access
void PhysicsWorld_free_members(PhysicsWorldObject *self) {
    // 1. Clear and free the ACTIVE command queue
    if (self->command_queue) {
        clear_command_queue(self);
        CULV_RAW_FREE(self->command_queue);
        self->command_queue = nullptr;
    }

    // 2. Free the SPARE command queue
    if (self->command_queue_spare) {
        CULV_RAW_FREE(self->command_queue_spare);
        self->command_queue_spare = nullptr;
    }

    // 3. Constraints (Must go before PhysicsSystem)
    free_constraints(self);

    // 4. Jolt Core Systems
    if (self->system) {
        JPH_PhysicsSystem_Destroy(self->system);
        self->system = nullptr;
    }
    if (self->char_vs_char_manager) {
        JPH_CharacterVsCharacterCollision_Destroy(self->char_vs_char_manager);
        self->char_vs_char_manager = nullptr;
    }
    if (self->job_system) {
        JPH_JobSystem_Destroy(self->job_system);
        self->job_system = nullptr;
    }

    // 5. Debug Utilities
    if (self->debug_renderer) {
        JPH_DebugRenderer_Destroy(self->debug_renderer);
        self->debug_renderer = nullptr;
    }
    debug_buffer_free(&self->debug_lines);
    debug_buffer_free(&self->debug_triangles);

    // 6. Shape Cache
    free_shape_cache(self);

    // 7. Contact Listener & Buffers
    if (self->contact_listener) {
        JPH_ContactListener_Destroy(self->contact_listener);
        self->contact_listener = nullptr;
    }
    CULV_RAW_FREE(self->contact_buffer);
    self->contact_buffer = nullptr;

    // 8. Deferred Trash Cleanup
    // Note: free_new_buffers has been updated to handle internal atomics
    if (self->trash_buffers) {
        for (size_t i = 0; i < self->trash_count; i++) {
            free_new_buffers(&self->trash_buffers[i]);
        }
        CULV_RAW_FREE(self->trash_buffers);
        self->trash_buffers = nullptr;
        self->trash_count   = 0;
    }

    // 9. Dense/Shadow Buffers
    // Note: free_shadow_buffers has been updated to handle gens/slot_states atomics
    free_shadow_buffers(self);

    // 10. Handle Mapping
    // TSan Fix: Cast pointer to atomic array to void* for free()
    if (self->id_to_handle_map) {
        CULV_RAW_FREE((void *)self->id_to_handle_map);
        self->id_to_handle_map = nullptr;
    }

    // 11. Threading Primitives
    FREE_LOCK(self->shadow_lock);
    FREE_NATIVE_MUTEX(self->step_sync.mutex);
    FREE_NATIVE_COND(self->step_sync.cond);
}

// helper: Initialize settings via Python helper
CULV_NODISCARD
int init_settings(PhysicsWorldObject *self, PyObject *settings_dict, float *gx, float *gy,
                  float *gz, int *max_bodies, int *max_pairs) {
    PyObject *st_module = PyType_GetModule(Py_TYPE(self));
    CulverinState *st   = get_culverin_state(st_module);
    PyObject *val_func  = PyObject_GetAttrString(st->helper, "validate_settings");
    if (!val_func) {
        return -1;
    }

    PyObject *norm =
        PyObject_CallFunctionObjArgs(val_func, settings_dict ? settings_dict : Py_None, nullptr);
    Py_DECREF(val_func);
    if (!norm) {
        return -1;
    }

    float slop;
    int ok = PyArg_ParseTuple(norm, "ffffii", gx, gy, gz, &slop, max_bodies, max_pairs);
    Py_DECREF(norm);
    return ok ? 0 : -1;
}

// helper: Initialize Jolt Core Systems
CULV_NODISCARD
int init_jolt_core(PhysicsWorldObject *self, WorldLimits limits, GravityVector gravity) {
#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    // When running ThreadSanitizer, disable Jolt's background workers.
    // TSan cannot understand Jolt's highly-optimized C++ lock-free job queues
    // and will generate false positives. Running Jolt synchronously allows TSan
    // to focus exclusively on finding real concurrency bugs in the Culverin/Python layer.
    constexpr int num_workers = 0;
#else
    constexpr int num_workers = -1;
#endif
    JobSystemThreadPoolConfig job_cfg = {
        .maxJobs = JOB_SYSTEM_MAX_JOBS, .maxBarriers = JOB_SYSTEM_MAX_BARRIERS, .numThreads = num_workers};

    // TSan Fix: Serialize the first PhysicsSystem creation. 
    // This allows Jolt's internal lazy-statics to initialize safely.
    NATIVE_MUTEX_LOCK(g_jph_trampoline_lock);
    self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);

    // --- 3 LAYERS: 0=Static, 1=Dynamic, 2=VehicleRay ---
    self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(3, 3);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 0, 0);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 1, 1);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 2, 2);

    self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(3);

    // Matrix:
    // 0 (Static)  vs 1 (Dynamic) -> ON
    // 0 (Static)  vs 2 (Ray)     -> ON
    // 1 (Dynamic) vs 1 (Dynamic) -> ON
    // 1 (Dynamic) vs 2 (Ray)     -> OFF (Fixes self-collision)
    // 2 (Ray)     vs 2 (Ray)     -> OFF
    JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 0, 1);
    JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 0, 2);
    JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1);
    JPH_ObjectLayerPairFilterTable_DisableCollision(self->pair_filter, 1, 2);

    self->bp_filter =
        JPH_ObjectVsBroadPhaseLayerFilterTable_Create(self->bp_interface, 3, self->pair_filter, 3);

    JPH_PhysicsSystemSettings phys_settings = {.maxBodies             = (uint32_t)limits.max_bodies,
                                               .maxBodyPairs          = (uint32_t)limits.max_pairs,
                                               .maxContactConstraints = MAX_CONTACT_CONSTRAINTS,
                                               .broadPhaseLayerInterface      = self->bp_interface,
                                               .objectLayerPairFilter         = self->pair_filter,
                                               .objectVsBroadPhaseLayerFilter = self->bp_filter};

    self->system               = JPH_PhysicsSystem_Create(&phys_settings);
    NATIVE_MUTEX_UNLOCK(g_jph_trampoline_lock);
    self->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
    JPH_PhysicsSystem_SetGravity(self->system, &(JPH_Vec3){gravity.gx, gravity.gy, gravity.gz});
    self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);
    return 0;
}

// helper: Iterate over baked Python data to create initial Jolt bodies
CULV_NODISCARD
int load_baked_scene(PhysicsWorldObject *self, PyObject *baked) {
    // TSan Fix: Load current count atomically
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);

    PyObject *pos_bytes_obj = PyTuple_GetItem(baked, BAKED_INDEX_POS);
    Py_ssize_t pos_len      = PyBytes_Size(pos_bytes_obj);

    if (pos_len < (Py_ssize_t)(current_count * sizeof(PosStride))) {
        PyErr_Format(PyExc_ValueError,
                     "Baked position buffer too small. Expected %zu bytes, got %zd.",
                     current_count * sizeof(PosStride), pos_len);
        return -1;
    }

    // 1. EXTRACT RAW BYTES AS PACKED ARRAYS
    JPH_Real *baked_pos = (JPH_Real *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_POS));
    float *baked_rot    = (float *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_ROT));
    float *baked_shape  = (float *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_SHAPE));
    unsigned char *u_mot =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_MOT));
    unsigned char *u_layer =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_LAYER));
    uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, BAKED_INDEX_USERDATA));

    if (!baked_pos || !baked_rot || !baked_shape || !u_mot || !u_layer || !u_data) {
        PyErr_SetString(PyExc_ValueError, "Invalid or truncated baked data bytes");
        return -1;
    }

    int result = 0;
    SHADOW_LOCK(&self->shadow_lock);

    JPH_BodyInterface *bi = self->body_interface;
    auto *shadow_pos      = (PosStride *)self->positions;
    auto *shadow_rot      = (AuxStride *)self->rotations;

    for (size_t i = 0; i < current_count; i++) {
        // A. Shape Lookup
        float *s_data    = &baked_shape[i * BAKED_SHAPE_STRIDE];
        float params[4]  = {s_data[1], s_data[2], s_data[3], s_data[4]};
        JPH_Shape *shape = find_or_create_shape_locked(self, (int)s_data[0], params);

        if (UNLIKELY(!shape)) {
            result = -1;
            break;
        }

        // B. EXTRACT POSITION (Shadow buffers are non-atomic)
        shadow_pos[i].x = baked_pos[i * BAKED_POS_STRIDE + 0];
        shadow_pos[i].y = baked_pos[i * BAKED_POS_STRIDE + 1];
        shadow_pos[i].z = baked_pos[i * BAKED_POS_STRIDE + 2];
        shadow_pos[i].w = 0.0;

        shadow_rot[i].x = baked_rot[i * BAKED_ROT_STRIDE + 0];
        shadow_rot[i].y = baked_rot[i * BAKED_ROT_STRIDE + 1];
        shadow_rot[i].z = baked_rot[i * BAKED_ROT_STRIDE + 2];
        shadow_rot[i].w = baked_rot[i * BAKED_ROT_STRIDE + 3];

        // C. Jolt Create Settings
        JPH_RVec3 j_pos = {shadow_pos[i].x, shadow_pos[i].y, shadow_pos[i].z};
        JPH_Quat j_rot  = {shadow_rot[i].x, shadow_rot[i].y, shadow_rot[i].z, shadow_rot[i].w};

        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
            shape, &j_pos, &j_rot, (JPH_MotionType)u_mot[i], (JPH_ObjectLayer)u_layer[i]);

        // TSan Fix: Initialize the atomic generation for this slot
        atomic_store_explicit(&self->generations[i], 1, memory_order_relaxed);
        
        // BodyHandle is _Atomic uint64_t. We create it locally.
        BodyHandle handle = make_handle((uint32_t)i, 1);
        
        // OPTIMIZATION: Use explicit relaxed load to avoid seq_cst penalty for Jolt
        uint64_t raw_h = atomic_load_explicit(&handle, memory_order_relaxed);
        JPH_BodyCreationSettings_SetUserData(creation, raw_h);
        
        if (u_mot[i] == 2) {
            JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
        }

        self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(bi, creation, JPH_Activation_Activate);

        uint32_t j_idx = JPH_ID_TO_INDEX(self->body_ids[i]);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            // TSan Fix: Store to shared map using relaxed (world is not yet simulating)
            atomic_store_explicit(&self->id_to_handle_map[j_idx], raw_h, memory_order_relaxed);
        }

        self->slot_to_dense[i] = (uint32_t)i;
        self->dense_to_slot[i] = (uint32_t)i;
        
        // TSan Fix: Atomic state update
        atomic_store_explicit(&self->slot_states[i], SLOT_ALIVE, memory_order_relaxed);
        
        self->user_data[i] = u_data[i];
        JPH_BodyCreationSettings_Destroy(creation);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    return result;
}
CULV_NODISCARD
int verify_abi_alignment(JPH_BodyInterface *bi) {
    JPH_BoxShapeSettings *bs = JPH_BoxShapeSettings_Create(&(JPH_Vec3){1, 1, 1}, 0.0f);
    auto *shape              = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(bs);
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
        PyErr_SetString(PyExc_RuntimeError, "JoltC ABI Mismatch: Precision issue.");
        return -1;
    }
    return 0;
}

PyType_DeclareSlot_StatusFromModule PhysicsWorld_getbuffer(PhysicsWorldObject *self, 
                                                           Py_buffer *view, CULV_MAYBE_UNUSED int flags) {
    SHADOW_LOCK(&self->shadow_lock);
    
    // TSan Fix: Read the atomic count safely
    size_t current_count = atomic_load_explicit(&self->count, memory_order_acquire);
    
    // We export the positions buffer as the default buffer for the object
    view->buf = self->positions;
    view->len = (Py_ssize_t)(current_count * sizeof(PosStride));
    view->readonly = 0;
    view->itemsize = sizeof(JPH_Real);
    view->format = (sizeof(JPH_Real) == sizeof(double)) ? "d" : "f";
    view->ndim = 2;
    view->shape = self->view_shape;
    view->strides = self->view_strides;
    view->suboffsets = NULL;
    view->internal = NULL;

    // view_export_count is a standard int protected by shadow_lock
    atomic_fetch_add_explicit(&self->view_export_count, 1, memory_order_relaxed);
    
    SHADOW_UNLOCK(&self->shadow_lock);
    return 0;
}

// Buffer Release Slot
PyType_DeclareSlot_VoidFromModule PhysicsWorld_releasebuffer(PhysicsWorldObject *self,
                                                   Py_buffer *Py_UNUSED(view)) {
    SHADOW_LOCK(&self->shadow_lock);
    
    // Release logic remains simple as no atomic counters are mutated here
    if (atomic_load_explicit(&self->view_export_count, memory_order_relaxed) > 0) {
        atomic_fetch_sub_explicit(&self->view_export_count, 1, memory_order_relaxed);
    }
    
    SHADOW_UNLOCK(&self->shadow_lock);
}