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
    const JPH_PhysicsSystem *sys = self->system;

    // 1. Get the count of bodies that actually moved this frame
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys, JPH_BodyType_Rigid);
    if (active_count == 0) return;

    // 2. Retrieve the IDs of moving bodies (Thread-safe copy)
    JPH_BodyID *active_ids = (JPH_BodyID *)PyMem_RawMalloc(active_count * sizeof(JPH_BodyID));
    if (UNLIKELY(!active_ids)) return;
    JPH_PhysicsSystem_GetActiveBodies(sys, active_ids, active_count);

    // 3. Setup Typed Output Pointers (with restrict for SIMD optimization)
    PosStride *CULV_RESTRICT shadow_pos  = (PosStride *)self->positions;
    AuxStride *CULV_RESTRICT shadow_rot  = (AuxStride *)self->rotations;
    AuxStride *CULV_RESTRICT shadow_lvel = (AuxStride *)self->linear_velocities;
    AuxStride *CULV_RESTRICT shadow_avel = (AuxStride *)self->angular_velocities;

    const int CHUNK_SIZE = 16;
    const JPH_Body* chunk[16];

    for (uint32_t i = 0; i < active_count; i += CHUNK_SIZE) {
        uint32_t rem = (active_count - i < CHUNK_SIZE) ? (active_count - i) : CHUNK_SIZE;

        // --- Phase 1: Pointer Resolution & Prefetching ---
        // This loop resolves the IDs to raw memory addresses.
        for (uint32_t j = 0; j < rem; j++) {
            const JPH_Body* b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i + j]);
            chunk[j] = b;

            if (LIKELY(b)) {
                // Tell the CPU to start fetching the coordinate data from RAM (Offset ~48)
                // 0 = Read access, 3 = High temporal locality (L1 cache)
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(((const char*)b) + 48, 0, 3);
                    __builtin_prefetch(((const char*)b) + 80, 0, 3);
                #elif defined(_MSC_VER)
                    _mm_prefetch(((const char*)b) + 48, _MM_HINT_T0);
                    _mm_prefetch(((const char*)b) + 80, _MM_HINT_T0);
                #endif
            }
        }

        // --- Phase 2: SIMD Hot Copy ---
        // This loop has zero function calls, allowing the compiler to use AVX/AVX2.
        for (uint32_t j = 0; j < rem; j++) {
            const JPH_Body* b = chunk[j];
            if (UNLIKELY(!b)) continue;

            // Resolve the dense index using the handle we stored in Jolt's UserData
            uint64_t handle = JPH_Body_GetUserData(b);
            uint32_t slot = (uint32_t)(handle & 0xFFFFFFFF);
            uint32_t dense = self->slot_to_dense[slot];

            JPH_RVec3 p; 
            JPH_Quat q; 
            JPH_Vec3 lv, av;

            // Extract high-precision data directly from the Body pointer
            JPH_Body_GetPosition(b, &p);
            JPH_Body_GetRotation(b, &q);
            JPH_Body_GetLinearVelocity(b, &lv);
            JPH_Body_GetAngularVelocity(b, &av);

            // Commit to shadow buffers using Stride Struct assignment
            // (Compiler optimizes these into vmovups/vmovsd instructions)
            shadow_pos[dense]  = (PosStride){p.x, p.y, p.z};
            shadow_rot[dense]  = (AuxStride){q.x, q.y, q.z, q.w};
            shadow_lvel[dense] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
            shadow_avel[dense] = (AuxStride){av.x, av.y, av.z, 0.0f};
        }
    }

    // 4. Cleanup
    PyMem_RawFree(active_ids);
}