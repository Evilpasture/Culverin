// --- START OF FILE culverin_shadow_sync.c ---

#include "culverin_shadow_sync.h"
#include "culverin_compiler_specifics.h"

static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

static constexpr int BATCH_SIZE = 32;

// =================================================================================================
// HOT PATH: Fully Unrolled, Linearized, Aligned Stores
// =================================================================================================
static CULV_FORCE_INLINE void process_full_batch(PhysicsWorldObject *self,
                                                 const SyncWorkItem *worklist) {
    auto *CULV_RESTRICT s_pos  = (PosStride *)CULV_ASSUME_ALIGNED(self->positions, 32);
    auto *CULV_RESTRICT s_ppos = (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, 32);
    auto *CULV_RESTRICT s_rot  = (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, 16);
    auto *CULV_RESTRICT s_prot = (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, 16);
    auto *CULV_RESTRICT s_lvel = (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, 16);
    auto *CULV_RESTRICT s_avel = (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, 16);

#pragma clang loop unroll(full)
    for (uint32_t j = 0; j < BATCH_SIZE; j++) {
        uint32_t D  = worklist[j].dense_idx;
        JPH_Body *b = (JPH_Body *)worklist[j].body;

        // 1. Snapshot (Always safe)
        s_ppos[D] = s_pos[D];
        s_prot[D] = s_rot[D];

        // 2. Position (DVec3 -> double[4])
        // Use the API. It handles JPH_DOUBLE_PRECISION offsets automatically.
        JPH_RVec3 p;
        JPH_Body_GetPosition(b, &p);
        s_pos[D].x = p.x;
        s_pos[D].y = p.y;
        s_pos[D].z = p.z;
        s_pos[D].w = 0.0;

        // 3. Rotation (Quat -> float[4])
        JPH_Quat q;
        JPH_Body_GetRotation(b, &q);
        s_rot[D] = (AuxStride){q.x, q.y, q.z, q.w};

        // 4. Velocities (Vec3 -> float[4])
        JPH_Vec3 lv;
        JPH_Vec3 av;
        JPH_Body_GetLinearVelocity(b, &lv);
        JPH_Body_GetAngularVelocity(b, &av);
        s_lvel[D] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
        s_avel[D] = (AuxStride){av.x, av.y, av.z, 0.0f};
    }
}

// =================================================================================================
// COLD PATH: Remainder Handling (0 to 31 items)
// =================================================================================================
static void process_partial_batch(PhysicsWorldObject *self, const SyncWorkItem *worklist,
                                  uint32_t count) {
    if (count == 0) {
        return;
    }

    // We still use aligned hints because the arrays don't move,
    // but we don't force unrolling.
    auto *CULV_RESTRICT s_pos  = (PosStride *)CULV_ASSUME_ALIGNED(self->positions, 32);
    auto *CULV_RESTRICT s_ppos = (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, 32);
    auto *CULV_RESTRICT s_rot  = (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, 16);
    auto *CULV_RESTRICT s_prot = (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, 16);
    auto *CULV_RESTRICT s_lvel = (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, 16);
    auto *CULV_RESTRICT s_avel = (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, 16);

    for (uint32_t j = 0; j < count; j++) {
        uint32_t D  = worklist[j].dense_idx;
        JPH_Body *b = (JPH_Body *)worklist[j].body;

        s_ppos[D] = s_pos[D];
        s_prot[D] = s_rot[D];

        JPH_RVec3 p;
        JPH_Body_GetPosition(b, &p);
        s_pos[D].x = p.x;
        s_pos[D].y = p.y;
        s_pos[D].z = p.z;
        s_pos[D].w = 0.0;

        JPH_Quat q;
        JPH_Body_GetRotation(b, &q);
        s_rot[D] = (AuxStride){q.x, q.y, q.z, q.w};

        JPH_Vec3 lv;
        JPH_Vec3 av;
        JPH_Body_GetLinearVelocity(b, &lv);
        JPH_Body_GetAngularVelocity(b, &av);
        s_lvel[D] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
        s_avel[D] = (AuxStride){av.x, av.y, av.z, 0.0f};
    }
}

// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================
void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
#ifdef CULVERIN_PROFILE_SYNC
    uint64_t start = rdtsc();
#endif

    const auto *sys       = self->system;
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys, JPH_BodyType_Rigid);

    if (UNLIKELY(active_count == 0 || !self->positions)) {
        return;
    }

    const JPH_BodyID *active_ids = JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys, JPH_BodyType_Rigid);

    SHADOW_LOCK(&self->shadow_lock);
    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;

    // Stack allocated worklist (fits in L1 cache comfortably)
    alignas(MEMORY_ALIGNMENT_SIZE) SyncWorkItem worklist[BATCH_SIZE];
    uint32_t work_ptr = 0;

    for (uint32_t i = 0; i < active_count; i++) {
        // 1. Prefetch Logic (Lookahead)
        // We look 4 items ahead to hide DRAM latency, as the optimized hot path
        // consumes bodies extremely fast.
        if (i + 4 < active_count) {
            const void *next_id_ptr = &active_ids[i + 4];
            CULV_PREFETCH(next_id_ptr);
        }

        // 2. Load Body
        const JPH_Body *b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i]);
        if (UNLIKELY(!b)) {
            continue;
        }

        // 3. Filter & Validate
        uint64_t handle = JPH_Body_GetUserData((JPH_Body *)b);
        auto slot       = (uint32_t)(handle & 0xFFFFFFFF);
        auto gen        = (uint32_t)(handle >> 32);

        if (LIKELY(slot < self->slot_capacity && self->generations[slot] == gen &&
                   self->slot_states[slot] == SLOT_ALIVE)) {

            worklist[work_ptr].body      = b;
            worklist[work_ptr].dense_idx = s2d[slot];
            work_ptr++;

            // --- HOT PATH TRIGGER ---
            if (work_ptr == BATCH_SIZE) {
                process_full_batch(self, worklist);
                work_ptr = 0;
            }
        }
    }

    // --- COLD PATH (REMAINDER) ---
    if (work_ptr > 0) {
        process_partial_batch(self, worklist, work_ptr);
    }

    SHADOW_UNLOCK(&self->shadow_lock);

#ifdef CULVERIN_PROFILE_SYNC
    uint64_t elapsed = rdtsc() - start;
    if (active_count > 0) {
        fprintf(stderr, "Sync: %llu cycles for %u bodies (%.1f cyc/body)\n", elapsed, active_count,
                (double)elapsed / active_count);
    }
#endif
}
