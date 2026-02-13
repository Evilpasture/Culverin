#include "culverin_shadow_sync.h"
#include "culverin.h"

/**
 * High-Performance Shadow Sync
 * Optimized for 1,000,000+ bodies.
 * 
 * Strategy:
 * 1. Only sync bodies Jolt marks as "Active" (Algorithmic win: O(Active) instead of O(Total)).
 * 2. Resolve pointers in chunks of 16 to maximize pipeline depth.
 * 3. Use __builtin_prefetch to pull Body data into L1 before we need it.
 * 4. Use Stride Structs to ensure the compiler generates packed SIMD stores.
 */
void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
    const auto *sys = self->system;
    
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys, JPH_BodyType_Rigid);
    if (active_count == 0) return;
    const JPH_BodyID *active_ids = JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys, JPH_BodyType_Rigid);
    if (!active_ids) return;

    auto *CULV_RESTRICT s_pos  = (PosStride *)self->positions;
    auto *CULV_RESTRICT s_ppos = (PosStride *)self->prev_positions;
    auto *CULV_RESTRICT s_rot  = (AuxStride *)self->rotations;
    auto *CULV_RESTRICT s_prot = (AuxStride *)self->prev_rotations;
    auto *CULV_RESTRICT s_lvel = (AuxStride *)self->linear_velocities;
    auto *CULV_RESTRICT s_avel = (AuxStride *)self->angular_velocities;

    const JPH_Body* chunk[16];
    constexpr int CHUNK_SIZE = 16;

    // Cache the lookup table pointer for speed
    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;

    for (auto i = 0u; i < active_count; i += CHUNK_SIZE) {
        auto rem = (active_count - i < CHUNK_SIZE) ? (active_count - i) : CHUNK_SIZE;

        // Phase 1: Resolve & Dual-Prefetch
        for (auto j = 0u; j < rem; j++) {
            const JPH_Body* b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i + j]);
            chunk[j] = b;
            
            if (LIKELY(b)) {
                // 1. Prefetch SOURCE (Jolt Body)
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(((const char*)b) + 48, 0, 3);
                #elif defined(_MSC_VER)
                    _mm_prefetch(((const char*)b) + 48, _MM_HINT_T0);
                #endif

                // 2. Prefetch DESTINATION (Shadow Buffer)
                // We need to calculate the dense index early
                uint64_t handle = JPH_Body_GetUserData((JPH_Body *)b);
                auto dense = s2d[(uint32_t)(handle & 0xFFFFFFFF)];
                
                // Prefetch the Write Location (1 = Write access)
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(&s_pos[dense], 1, 3);
                #elif defined(_MSC_VER)
                    _mm_prefetch((const char*)&s_pos[dense], _MM_HINT_T0);
                #endif
            }
        }

        // Phase 2: Burst Write (Shadow Lock Held)
        SHADOW_LOCK(&self->shadow_lock);
        
        // Compiler Hint: Assume loop count is small but > 0
        // This encourages unrolling without massive overhead
        #if defined(__clang__)
        #pragma clang loop unroll(full)
        #endif
        for (uint32_t j = 0; j < rem; j++) {
            const JPH_Body* b = chunk[j];
            if (UNLIKELY(!b)) continue;

            uint64_t handle = JPH_Body_GetUserData((JPH_Body *)b);
            auto dense = self->slot_to_dense[(uint32_t)(handle & 0xFFFFFFFF)];

            // 1. Read Jolt (Load into Registers)
            JPH_RVec3 p; JPH_Quat q; JPH_Vec3 lv, av;
            JPH_Body_GetPosition(b, &p);
            JPH_Body_GetRotation(b, &q);
            JPH_Body_GetLinearVelocity(b, &lv);
            JPH_Body_GetAngularVelocity(b, &av);

            // 2. Snapshot (Memory to Memory)
            // Note: We read the *current* Shadow values, not the Jolt values here.
            // This is cache-friendly because we are about to overwrite this cache line.
            s_ppos[dense] = s_pos[dense];
            s_prot[dense] = s_rot[dense];

            // 3. Write New (Registers to Memory)
            s_pos[dense]  = (PosStride){p.x, p.y, p.z};
            s_rot[dense]  = (AuxStride){q.x, q.y, q.z, q.w};
            s_lvel[dense] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
            s_avel[dense] = (AuxStride){av.x, av.y, av.z, 0.0f};
        }
        SHADOW_UNLOCK(&self->shadow_lock);
    }
}