#include "culverin_physics_world_internal.h"
#include "culverin_compiler_specifics.h"
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

    // Use the aligned wrapper for every buffer allocated via CulvMem_RawMallocAligned
    CulvMem_RawFreeAligned(nb->pos);
    CulvMem_RawFreeAligned(nb->rot);
    CulvMem_RawFreeAligned(nb->ppos);
    CulvMem_RawFreeAligned(nb->prot);
    CulvMem_RawFreeAligned(nb->lvel);
    CulvMem_RawFreeAligned(nb->avel);

    CULV_RAW_FREE(nb->bids);
    CULV_RAW_FREE(nb->udat);
    CULV_RAW_FREE(nb->gens);
    CULV_RAW_FREE(nb->s2d);
    CULV_RAW_FREE(nb->d2s);
    CULV_RAW_FREE(nb->stat);
    CULV_RAW_FREE(nb->free);
    CULV_RAW_FREE(nb->cats);
    CULV_RAW_FREE(nb->masks);
    CULV_RAW_FREE(nb->mats);

    // Defensive: Zero the struct so we don't accidentally Use-After-Free
    memset(nb, 0, sizeof(NewBuffers));
}
CULV_NODISCARD
static int alloc_new_buffers(NewBuffers *nb, size_t cap) {
    memset(nb, 0, sizeof(NewBuffers));

    // SIMD-Heavy Buffers: AVX alignment
    nb->pos  = (JPH_Real *)CulvMem_RawMallocAligned(cap * sizeof(PosStride), AVX_ALIGNMENT);
    nb->ppos = (JPH_Real *)CulvMem_RawMallocAligned(cap * sizeof(PosStride), AVX_ALIGNMENT);
    nb->rot  = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->prot = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->lvel = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);
    nb->avel = (float *)CulvMem_RawMallocAligned(cap * sizeof(AuxStride), AVX_ALIGNMENT);

    // Standard Data Buffers: no special alignment needed
    nb->bids  = (JPH_BodyID *)CULV_RAW_MALLOC(cap * sizeof(JPH_BodyID));
    nb->udat  = (uint64_t *)CULV_RAW_CALLOC(cap, sizeof(uint64_t));
    nb->gens  = (uint32_t *)CULV_RAW_CALLOC(cap, sizeof(uint32_t));
    nb->s2d   = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->d2s   = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->stat  = (uint8_t *)CULV_RAW_CALLOC(cap, sizeof(uint8_t));
    nb->free  = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->cats  = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->masks = (uint32_t *)CULV_RAW_MALLOC(cap * sizeof(uint32_t));
    nb->mats  = (uint32_t *)CULV_RAW_CALLOC(cap, sizeof(uint32_t));

    if (!nb->pos || !nb->rot || !nb->ppos || !nb->prot || !nb->lvel || !nb->avel || !nb->bids ||
        !nb->udat || !nb->gens || !nb->s2d || !nb->d2s || !nb->stat || !nb->free || !nb->cats ||
        !nb->masks || !nb->mats) {
        free_new_buffers(nb);
        return -1;
    }
    return 0;
}

static size_t migrate_and_init(PhysicsWorldObject *self, NewBuffers *nb, size_t new_cap) {
    // 1. Copy Active Dense Data (The bodies currently in simulation)
    if (self->count > 0) {
        // Positions use Stride 4 (32 bytes in Double Precision)
        size_t pos_bytes = self->count * sizeof(PosStride);
        memcpy(nb->pos, self->positions, pos_bytes);
        memcpy(nb->ppos, self->prev_positions, pos_bytes);

        // Rotations/Velocities use Stride 4 (16 bytes)
        size_t aux_bytes = self->count * sizeof(AuxStride);
        memcpy(nb->rot, self->rotations, aux_bytes);
        memcpy(nb->prot, self->prev_rotations, aux_bytes);
        memcpy(nb->lvel, self->linear_velocities, aux_bytes);
        memcpy(nb->avel, self->angular_velocities, aux_bytes);

        // Metadata (BodyIDs, UserData, Masks, etc.)
        memcpy(nb->bids, self->body_ids, self->count * sizeof(JPH_BodyID));
        memcpy(nb->udat, self->user_data, self->count * sizeof(uint64_t));
        memcpy(nb->cats, self->categories, self->count * sizeof(uint32_t));
        memcpy(nb->masks, self->masks, self->count * sizeof(uint32_t));
        memcpy(nb->mats, self->material_ids, self->count * sizeof(uint32_t));

        // Dense-to-Slot mapping
        memcpy(nb->d2s, self->dense_to_slot, self->count * sizeof(uint32_t));
    }

    // 2. Copy Mapping Tables (The Slot-based indirection)
    if (self->slot_capacity > 0) {
        size_t slot_bytes_u32 = self->slot_capacity * sizeof(uint32_t);
        memcpy(nb->gens, self->generations, slot_bytes_u32);
        memcpy(nb->s2d, self->slot_to_dense, slot_bytes_u32);
        memcpy(nb->stat, self->slot_states, self->slot_capacity * sizeof(uint8_t));

        // Copy the current free list
        memcpy(nb->free, self->free_slots, self->free_count * sizeof(uint32_t));
    }

    // 3. Initialize Expanded Slots
    // We calculate the new free count locally
    size_t local_free_count = self->free_count;
    for (size_t i = self->slot_capacity; i < new_cap; i++) {
        nb->gens[i]                  = 1; // Starting generation
        nb->stat[i]                  = SLOT_EMPTY;
        nb->free[local_free_count++] = (uint32_t)i; // Add to free list
    }

    return local_free_count;
}

// helper: Allocate shadow buffers and indirection maps
CULV_NODISCARD
int allocate_buffers(PhysicsWorldObject *self, int max_bodies) {
    self->capacity = (size_t)max_bodies;
    if (self->capacity < self->count + SHADOW_CAPACITY_PADDING) {
        self->capacity = self->count + SHADOW_CAPACITY_GROW;
    }
    self->max_jolt_bodies = (uint32_t)max_bodies;
    self->slot_capacity   = self->capacity;

    // Fix: Allocate with proper alignment for SIMD operations
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

    self->id_to_handle_map =
        (BodyHandle *)CULV_RAW_CALLOC(self->max_jolt_bodies, sizeof(BodyHandle));

    self->generations   = (uint32_t *)CULV_RAW_CALLOC(self->slot_capacity, sizeof(uint32_t));
    self->slot_to_dense = (uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(uint32_t));
    self->dense_to_slot = (uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(uint32_t));
    self->free_slots    = (uint32_t *)CULV_RAW_MALLOC(self->slot_capacity * sizeof(uint32_t));
    self->slot_states   = (uint8_t *)CULV_RAW_CALLOC(self->slot_capacity, sizeof(uint8_t));

    self->command_queue =
        (PhysicsCommand *)CULV_RAW_MALLOC(COMMAND_QUEUE_INITIAL_CAPACITY * sizeof(PhysicsCommand));
    self->command_capacity = COMMAND_QUEUE_INITIAL_CAPACITY;

    self->trash_capacity = 4;
    self->trash_count    = 0;
    self->trash_buffers  = (NewBuffers *)CULV_RAW_CALLOC(self->trash_capacity, sizeof(NewBuffers));

    if (!self->positions || !self->rotations || !self->id_to_handle_map || !self->command_queue ||
        !self->slot_states) {
        return -1;
    }

    // Zero-initialize the aligned buffers (since we can't use Calloc with custom alignment)
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
    // 1. Buffer View Guard
    if (self->view_export_count > 0) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize while memoryview is active.");
        return -1;
    }

    // 2. Concurrency Guard: Wait for Step Thread to finish Phase 5 (Sync)
    // and wait for any active Raycast/Query to finish.
    BLOCK_UNTIL_NOT_STEPPING(self);
    BLOCK_UNTIL_NOT_QUERYING(self);

    // CAP: Never exceed the Jolt limit established at init
    if (new_capacity > self->max_jolt_bodies) {
        new_capacity = self->max_jolt_bodies;
    }

    if (new_capacity <= self->capacity) {
        return 0;
    }

    // 3. Prepare New Buffers (Transactional)
    NewBuffers nb;
    if (alloc_new_buffers(&nb, new_capacity) < 0) {
        PyErr_NoMemory();
        return -1;
    }

    // 4. Migrate Data
    size_t final_free_count = migrate_and_init(self, &nb, new_capacity);

    // 5. Expand Trash Bin if needed
    if (self->trash_count >= self->trash_capacity) {
        size_t next_cap = (self->trash_capacity == 0) ? 4 : self->trash_capacity * 2;
        void *new_trash = CULV_RAW_REALLOC(self->trash_buffers, next_cap * sizeof(NewBuffers));
        if (!new_trash) {
            free_new_buffers(&nb);
            return -1;
        }

        // Zero-init the NEW portion of the trash array to prevent double-frees
        size_t added_elements = next_cap - self->trash_capacity;
        memset((NewBuffers *)new_trash + self->trash_capacity, 0,
               added_elements * sizeof(NewBuffers));

        self->trash_buffers  = (NewBuffers *)new_trash;
        self->trash_capacity = next_cap;
    }

    // 6. THE COMMIT (Critical Section)
    // Package current pointers to be freed later by the Stepper
    NewBuffers old_bufs                      = {.pos   = self->positions,
                                                .ppos  = self->prev_positions,
                                                .rot   = self->rotations,
                                                .prot  = self->prev_rotations,
                                                .lvel  = self->linear_velocities,
                                                .avel  = self->angular_velocities,
                                                .bids  = self->body_ids,
                                                .udat  = self->user_data,
                                                .gens  = self->generations,
                                                .s2d   = self->slot_to_dense,
                                                .d2s   = self->dense_to_slot,
                                                .stat  = self->slot_states,
                                                .free  = self->free_slots,
                                                .cats  = self->categories,
                                                .masks = self->masks,
                                                .mats  = self->material_ids};
    self->trash_buffers[self->trash_count++] = old_bufs;

    // Swap pointers to the new, larger arrays
    self->positions          = nb.pos;
    self->prev_positions     = nb.ppos;
    self->rotations          = nb.rot;
    self->prev_rotations     = nb.prot;
    self->linear_velocities  = nb.lvel;
    self->angular_velocities = nb.avel;
    self->body_ids           = nb.bids;
    self->user_data          = nb.udat;
    self->generations        = nb.gens;
    self->slot_to_dense      = nb.s2d;
    self->dense_to_slot      = nb.d2s;
    self->slot_states        = nb.stat;
    self->free_slots         = nb.free;
    self->categories         = nb.cats;
    self->masks              = nb.masks;
    self->material_ids       = nb.mats;

    // Update metadata
    self->free_count    = final_free_count;
    self->capacity      = new_capacity;
    self->slot_capacity = new_capacity;

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
            self->constraints[i] = NULL;
        }
        CULV_RAW_FREE((void *)self->constraints);
        self->constraints = NULL;
    }
    CULV_RAW_FREE(self->constraint_generations);
    self->constraint_generations = NULL;
    CULV_RAW_FREE(self->free_constraint_slots);
    self->free_constraint_slots = NULL;
    CULV_RAW_FREE(self->constraint_states);
    self->constraint_states = NULL;
}

void free_shadow_buffers(PhysicsWorldObject *self) {
    // Aligned buffers (stride types) must use CulvMem_RawFreeAligned
    CulvMem_RawFreeAligned(self->positions);
    self->positions = NULL;
    CulvMem_RawFreeAligned(self->prev_positions);
    self->prev_positions = NULL;
    CulvMem_RawFreeAligned(self->rotations);
    self->rotations = NULL;
    CulvMem_RawFreeAligned(self->prev_rotations);
    self->prev_rotations = NULL;
    CulvMem_RawFreeAligned(self->linear_velocities);
    self->linear_velocities = NULL;
    CulvMem_RawFreeAligned(self->angular_velocities);
    self->angular_velocities = NULL;

    // Regular buffers
    CULV_RAW_FREE(self->body_ids);
    self->body_ids = NULL;
    CULV_RAW_FREE(self->generations);
    self->generations = NULL;
    CULV_RAW_FREE(self->slot_to_dense);
    self->slot_to_dense = NULL;
    CULV_RAW_FREE(self->dense_to_slot);
    self->dense_to_slot = NULL;
    CULV_RAW_FREE(self->free_slots);
    self->free_slots = NULL;
    CULV_RAW_FREE(self->slot_states);
    self->slot_states = NULL;
    CULV_RAW_FREE(self->command_queue);
    self->command_queue = NULL;
    CULV_RAW_FREE(self->user_data);
    self->user_data = NULL;
    CULV_RAW_FREE(self->categories);
    self->categories = NULL;
    CULV_RAW_FREE(self->masks);
    self->masks = NULL;
    CULV_RAW_FREE(self->material_ids);
    self->material_ids = NULL;
    CULV_RAW_FREE(self->materials);
    self->materials = NULL;
}

// --- Helper: Resource Cleanup (Idempotent) ---
// SAFETY:
// - Must not be called while PhysicsSystem is stepping
// - Must not be called from a Jolt callback
// - Must not race with Python memoryview access
void PhysicsWorld_free_members(PhysicsWorldObject *self) {
    // 1. Clear and free the ACTIVE command queue
    // This one definitely contains live Jolt pointers from create_body calls
    // made since the last world.step().
    if (self->command_queue) {
        clear_command_queue(self); 
        CULV_RAW_FREE(self->command_queue);
        self->command_queue = NULL;
    }

    // 2. Free the SPARE command queue
    // Note: We don't call clear_command_queue here because flush_commands_internal
    // already destroyed the settings pointers in this buffer during the last step.
    // We just release the raw memory block.
    if (self->command_queue_spare) {
        CULV_RAW_FREE(self->command_queue_spare);
        self->command_queue_spare = NULL;
    }

    // 3. Constraints (Must go before PhysicsSystem)
    free_constraints(self);

    // 4. Jolt Core Systems
    if (self->system) {
        JPH_PhysicsSystem_Destroy(self->system);
        self->system = NULL;
    }
    if (self->char_vs_char_manager) {
        JPH_CharacterVsCharacterCollision_Destroy(self->char_vs_char_manager);
        self->char_vs_char_manager = NULL;
    }
    if (self->job_system) {
        JPH_JobSystem_Destroy(self->job_system);
        self->job_system = NULL;
    }

    // 5. Debug Utilities
    if (self->debug_renderer) {
        JPH_DebugRenderer_Destroy(self->debug_renderer);
        self->debug_renderer = NULL;
    }
    debug_buffer_free(&self->debug_lines);
    debug_buffer_free(&self->debug_triangles);

    // 6. Shape Cache (THE BIG ONE)
    free_shape_cache(self);

    // 7. Contact Listener & Buffers
    if (self->contact_listener) {
        JPH_ContactListener_Destroy(self->contact_listener);
        self->contact_listener = NULL;
    }
    CULV_RAW_FREE(self->contact_buffer);
    self->contact_buffer = NULL;

    // 8. Deferred Trash Cleanup
    if (self->trash_buffers) {
        for (size_t i = 0; i < self->trash_count; i++) {
            free_new_buffers(&self->trash_buffers[i]);
        }
        CULV_RAW_FREE(self->trash_buffers);
        self->trash_buffers = NULL;
        self->trash_count   = 0;
    }

    // 9. Dense/Shadow Buffers
    free_shadow_buffers(self);

    // 10. Handle Mapping
    CULV_RAW_FREE(self->id_to_handle_map);
    self->id_to_handle_map = NULL;

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
        PyObject_CallFunctionObjArgs(val_func, settings_dict ? settings_dict : Py_None, NULL);
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
    JobSystemThreadPoolConfig job_cfg = {
        .maxJobs = JOB_SYSTEM_MAX_JOBS, .maxBarriers = JOB_SYSTEM_MAX_BARRIERS, .numThreads = -1};
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
    self->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
    JPH_PhysicsSystem_SetGravity(self->system, &(JPH_Vec3){gravity.gx, gravity.gy, gravity.gz});
    self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);
    return 0;
}

// helper: Iterate over baked Python data to create initial Jolt bodies
CULV_NODISCARD
int load_baked_scene(PhysicsWorldObject *self, PyObject *baked) {
    PyObject *pos_bytes_obj = PyTuple_GetItem(baked, BAKED_INDEX_POS);
    Py_ssize_t pos_len      = PyBytes_Size(pos_bytes_obj);

    // Safety check: Expected Stride 4 * sizeof(JPH_Real)
    if (pos_len < (Py_ssize_t)(self->count * sizeof(PosStride))) {
        PyErr_Format(PyExc_ValueError,
                     "Baked position buffer too small. Expected %zu bytes, got %zd. Check Python "
                     "bake_scene precision.",
                     self->count * sizeof(PosStride), pos_len);
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

    for (size_t i = 0; i < self->count; i++) {
        // A. Shape Lookup
        float *s_data    = &baked_shape[i * BAKED_SHAPE_STRIDE];
        float params[4]  = {s_data[1], s_data[2], s_data[3], s_data[4]};
        JPH_Shape *shape = find_or_create_shape_locked(self, (int)s_data[0], params);

        if (UNLIKELY(!shape)) {
            result = -1;
            break;
        }

        // B. EXTRACT POSITION (Python packed 4 JPH_Real per body)
        shadow_pos[i].x = baked_pos[i * BAKED_POS_STRIDE + 0];
        shadow_pos[i].y = baked_pos[i * BAKED_POS_STRIDE + 1];
        shadow_pos[i].z = baked_pos[i * BAKED_POS_STRIDE + 2];
        shadow_pos[i].w = 0.0;

        // Quaternions are always floats
        shadow_rot[i].x = baked_rot[i * BAKED_ROT_STRIDE + 0];
        shadow_rot[i].y = baked_rot[i * BAKED_ROT_STRIDE + 1];
        shadow_rot[i].z = baked_rot[i * BAKED_ROT_STRIDE + 2];
        shadow_rot[i].w = baked_rot[i * BAKED_ROT_STRIDE + 3];

        // C. Jolt Create Settings
        JPH_RVec3 j_pos = {shadow_pos[i].x, shadow_pos[i].y, shadow_pos[i].z};
        JPH_Quat j_rot  = {shadow_rot[i].x, shadow_rot[i].y, shadow_rot[i].z, shadow_rot[i].w};

        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
            shape, &j_pos, &j_rot, (JPH_MotionType)u_mot[i], (JPH_ObjectLayer)u_layer[i]);

        self->generations[i] = 1;
        JPH_BodyCreationSettings_SetUserData(creation, (uint64_t)make_handle((uint32_t)i, 1));
        if (u_mot[i] == 2) {
            JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
        }

        self->body_ids[i] =
            JPH_BodyInterface_CreateAndAddBody(bi, creation, JPH_Activation_Activate);

        uint32_t j_idx = JPH_ID_TO_INDEX(self->body_ids[i]);
        if (self->id_to_handle_map && j_idx < self->max_jolt_bodies) {
            self->id_to_handle_map[j_idx] = make_handle((uint32_t)i, 1);
        }

        self->slot_to_dense[i] = (uint32_t)i;
        self->dense_to_slot[i] = (uint32_t)i;
        self->slot_states[i]   = SLOT_ALIVE;
        self->user_data[i]     = u_data[i];

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

// Buffer Release Slot
PyType_DeclareSlot_Void PhysicsWorld_releasebuffer(PhysicsWorldObject *self,
                                                   Py_buffer *Py_UNUSED(view)) {
    SHADOW_LOCK(&self->shadow_lock);
    if (self->view_export_count > 0) {
        self->view_export_count--;
    }
    SHADOW_UNLOCK(&self->shadow_lock);
}
