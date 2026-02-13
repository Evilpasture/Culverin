#include "culverin_shadow_sync.h"
#include "culverin.h"

void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
    const JPH_PhysicsSystem *sys = self->system;
    const size_t count = self->count;
    const JPH_BodyID *CULV_RESTRICT body_ids = self->body_ids;

    PosStride *CULV_RESTRICT shadow_pos  = (PosStride *)self->positions;
    AuxStride *CULV_RESTRICT shadow_rot  = (AuxStride *)self->rotations;
    AuxStride *CULV_RESTRICT shadow_lvel = (AuxStride *)self->linear_velocities;
    AuxStride *CULV_RESTRICT shadow_avel = (AuxStride *)self->angular_velocities;

    const JPH_Body* chunk[16]; // Increased chunk to 16 for better prefetch depth
    const int CHUNK_SIZE = 16;

    for (size_t i = 0; i < count; i += CHUNK_SIZE) {
        int rem = (count - i < CHUNK_SIZE) ? (int)(count - i) : CHUNK_SIZE;

        // --- Phase 1: Resolve & Prefetch ---
        for (int j = 0; j < rem; j++) {
            JPH_BodyID bid = body_ids[i + j];
            if (bid != JPH_INVALID_BODY_ID) {
                const JPH_Body* b = JPH_PhysicsSystem_GetBodyPtr(sys, bid);
                chunk[j] = b;

                // PREFETCH: We know coordinates start at offset ~48.
                // 0 = Read access, 3 = High temporal locality (Keep in L1)
                #if defined(__GNUC__) || defined(__clang__)
                    __builtin_prefetch(((const char*)b) + 48, 0, 3);
                    __builtin_prefetch(((const char*)b) + 80, 0, 3); // Also pull velocity block
                #elif defined(_MSC_VER)
                    _mm_prefetch(((const char*)b) + 48, _MM_HINT_T0);
                    _mm_prefetch(((const char*)b) + 80, _MM_HINT_T0);
                #endif
            } else {
                chunk[j] = NULL;
            }
        }

        // --- Phase 2: Hot Copy ---
        for (int j = 0; j < rem; j++) {
            const JPH_Body* b = chunk[j];
            if (!b) continue;
            size_t idx = i + j;

            JPH_RVec3 p; JPH_Quat q; JPH_Vec3 lv; JPH_Vec3 av;
            JPH_Body_GetPosition(b, &p);
            JPH_Body_GetRotation(b, &q);
            JPH_Body_GetLinearVelocity(b, &lv);
            JPH_Body_GetAngularVelocity(b, &av);

            shadow_pos[idx]  = (PosStride){p.x, p.y, p.z};
            shadow_rot[idx]  = (AuxStride){q.x, q.y, q.z, q.w};
            shadow_lvel[idx] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
            shadow_avel[idx] = (AuxStride){av.x, av.y, av.z, 0.0f};
        }
    }
}