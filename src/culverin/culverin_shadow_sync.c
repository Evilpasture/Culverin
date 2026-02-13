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
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys, JPH_BodyType_Rigid);
    if (active_count == 0) return;

    JPH_BodyID *active_ids = (JPH_BodyID *)PyMem_RawMalloc(active_count * sizeof(JPH_BodyID));
    if (!active_ids) return;
    JPH_PhysicsSystem_GetActiveBodies(sys, active_ids, active_count);

    PosStride *CULV_RESTRICT shadow_pos  = (PosStride *)self->positions;
    PosStride *CULV_RESTRICT shadow_ppos = (PosStride *)self->prev_positions;
    AuxStride *CULV_RESTRICT shadow_rot  = (AuxStride *)self->rotations;
    AuxStride *CULV_RESTRICT shadow_prot = (AuxStride *)self->prev_rotations;
    AuxStride *CULV_RESTRICT shadow_lvel = (AuxStride *)self->linear_velocities;
    AuxStride *CULV_RESTRICT shadow_avel = (AuxStride *)self->angular_velocities;

    const JPH_Body* chunk[16];
    const int CHUNK_SIZE = 16;

    for (uint32_t i = 0; i < active_count; i += CHUNK_SIZE) {
        uint32_t rem = (active_count - i < CHUNK_SIZE) ? (active_count - i) : CHUNK_SIZE;

        // Phase 1: Resolve & Prefetch
        for (uint32_t j = 0; j < rem; j++) {
            const JPH_Body* b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i + j]);
            chunk[j] = b;
            if (b) {
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(((const char*)b) + 48, 0, 3);
                #endif
            }
        }

        // Phase 2: Double-Sync (Current to Prev, then Jolt to Current)
        for (uint32_t j = 0; j < rem; j++) {
            const JPH_Body* b = chunk[j];
            if (!b) continue;

            uint64_t handle = JPH_Body_GetUserData(b);
            uint32_t dense = self->slot_to_dense[(uint32_t)(handle & 0xFFFFFFFF)];

            // --- THE SELECTIVE SNAPSHOT ---
            // Move "Old Current" to "Previous" for this specific body.
            // This is extremely fast because shadow_pos[dense] is likely 
            // already in the CPU cache from the previous frame's renderer read.
            shadow_ppos[dense] = shadow_pos[dense];
            shadow_prot[dense] = shadow_rot[dense];

            // Now perform the standard Jolt-to-Shadow sync
            JPH_RVec3 p; JPH_Quat q; JPH_Vec3 lv, av;
            JPH_Body_GetPosition(b, &p);
            JPH_Body_GetRotation(b, &q);
            JPH_Body_GetLinearVelocity(b, &lv);
            JPH_Body_GetAngularVelocity(b, &av);

            shadow_pos[dense]  = (PosStride){p.x, p.y, p.z};
            shadow_rot[dense]  = (AuxStride){q.x, q.y, q.z, q.w};
            shadow_lvel[dense] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
            shadow_avel[dense] = (AuxStride){av.x, av.y, av.z, 0.0f};
        }
    }
    PyMem_RawFree(active_ids);
}