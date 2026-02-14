#include "culverin_shadow_sync.h"
#include "culverin.h"

#define CULVERIN_PROFILE_SYNC

#ifdef CULVERIN_PROFILE_SYNC
#include <stdio.h>
static inline uint64_t rdtsc() {
#ifdef _MSC_VER
    return __rdtsc();
#else
    uint32_t lo, hi;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
#endif
}
#endif

/**
 * High-Performance Shadow Sync
 * Decouples Jolt State from Shadow Buffers via a stack-allocated worklist.
 */
void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
#ifdef CULVERIN_PROFILE_SYNC
    uint64_t start = rdtsc();
#endif

    const auto *sys = self->system;
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys, JPH_BodyType_Rigid);
    if (active_count == 0) return;

    const JPH_BodyID *active_ids = JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys, JPH_BodyType_Rigid);
    if (UNLIKELY(!active_ids || !self->positions)) return;

    // Hoist pointers to local registers (tells compiler they are stable)
    auto *CULV_RESTRICT s_pos  = (PosStride *)self->positions;
    auto *CULV_RESTRICT s_ppos = (PosStride *)self->prev_positions;
    auto *CULV_RESTRICT s_rot  = (AuxStride *)self->rotations;
    auto *CULV_RESTRICT s_prot = (AuxStride *)self->prev_rotations;
    auto *CULV_RESTRICT s_lvel = (AuxStride *)self->linear_velocities;
    auto *CULV_RESTRICT s_avel = (AuxStride *)self->angular_velocities;
    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;

    // Batching Worklist (Stack allocated - 512 bytes total)
    SyncWorkItem worklist[32];
    uint32_t work_ptr = 0;

    for (uint32_t i = 0; i < active_count; i++) {
        const JPH_Body* b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i]);
        if (UNLIKELY(!b)) continue;

        // --- PHASE 1: PREPARATION (No Lock) ---
        uint64_t handle = JPH_Body_GetUserData((JPH_Body *)b);
        uint32_t slot = (uint32_t)(handle & 0xFFFFFFFF);
        uint32_t gen = (uint32_t)(handle >> 32);

        // Filter and Validate logic outside the lock
        if (LIKELY(slot < self->slot_capacity && 
                   self->generations[slot] == gen && 
                   self->slot_states[slot] == SLOT_ALIVE)) {
            
            uint32_t dense = s2d[slot];
            
            // Prefetch Jolt body data into L1 cache for Phase 2
            #if defined(__clang__) || defined(__GNUC__)
                __builtin_prefetch(((const char*)b) + 48, 0, 3);
            #elif defined(_MSC_VER)
                _mm_prefetch(((const char*)b) + 48, _MM_HINT_T0);
            #endif

            worklist[work_ptr++] = (SyncWorkItem){b, dense};
        }

        // --- PHASE 2: BURST SYNC (Hold Shadow Lock) ---
        if (work_ptr == 32 || (i == active_count - 1 && work_ptr > 0)) {
            SHADOW_LOCK(&self->shadow_lock);

            // ========== PHASE A: SNAPSHOT (Shadow → Shadow) ==========
            // This is a pure memory copy with known stride, easy to vectorize
            #pragma clang loop unroll(full) vectorize(enable)
            for (uint32_t j = 0; j < work_ptr; j++) {
                uint32_t D = worklist[j].dense_idx;
                s_ppos[D] = s_pos[D];   // 32-byte AVX move
                s_prot[D] = s_rot[D];   // 16-byte SSE move
            }

            // ========== PHASE B: SYNC (Jolt → Shadow) ==========
            #pragma clang loop unroll(full)
            for (uint32_t j = 0; j < work_ptr; j++) {
                const JPH_Body* B = worklist[j].body;
                uint32_t D = worklist[j].dense_idx;

                // Use stack locals as "landing zones" for the getters
                JPH_RVec3 p;
                JPH_Quat q;
                JPH_Vec3 lv, av;

                // The compiler can only optimize these if it can see the source code
                JPH_Body_GetPosition(B, &p);
                JPH_Body_GetRotation(B, &q);
                JPH_Body_GetLinearVelocity((JPH_Body *)B, &lv);
                JPH_Body_GetAngularVelocity((JPH_Body *)B, &av);

                // Slam into Shadow
                s_pos[D]  = (PosStride){p.x, p.y, p.z, 0.0};
                s_rot[D]  = (AuxStride){q.x, q.y, q.z, q.w};
                s_lvel[D] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
                s_avel[D] = (AuxStride){av.x, av.y, av.z, 0.0f};
            }

            SHADOW_UNLOCK(&self->shadow_lock);
            work_ptr = 0; // Reset for next batch
        }
    }

#ifdef CULVERIN_PROFILE_SYNC
    uint64_t elapsed = rdtsc() - start;
    if (active_count > 0) {
        fprintf(stderr, "Sync: %llu cycles for %u bodies (%.1f cyc/body)\n",
                elapsed, active_count, (double)elapsed / active_count);
    }
#endif
}