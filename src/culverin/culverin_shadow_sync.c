#include "culverin_shadow_sync.h"
#include "culverin.h"

void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
    const JPH_PhysicsSystem *sys = self->system; // Use system directly
    const size_t count = self->count;
    const JPH_BodyID *CULV_RESTRICT body_ids = self->body_ids;

    PosStride *CULV_RESTRICT shadow_pos  = (PosStride *)self->positions;
    AuxStride *CULV_RESTRICT shadow_rot  = (AuxStride *)self->rotations;
    AuxStride *CULV_RESTRICT shadow_lvel = (AuxStride *)self->linear_velocities;
    AuxStride *CULV_RESTRICT shadow_avel = (AuxStride *)self->angular_velocities;

    const JPH_Body* chunk[8];

    for (size_t i = 0; i < count; i += 8) {
        int rem = (count - i < 8) ? (int)(count - i) : 8;

        // Phase 1: Pointer Resolution (1 call per body)
        for (int j = 0; j < rem; j++) {
            JPH_BodyID bid = body_ids[i + j];
            if (bid != JPH_INVALID_BODY_ID) {
                chunk[j] = JPH_PhysicsSystem_GetBodyPtr(sys, bid);
            } else {
                chunk[j] = NULL;
            }
        }

        // Phase 2: Hot Copy (No calls, Compiler Vectorizes)
        for (int j = 0; j < rem; j++) {
            const JPH_Body* b = chunk[j];
            if (!b) continue;
            size_t idx = i + j;

            JPH_RVec3 p; JPH_Quat q; JPH_Vec3 lv; JPH_Vec3 av;

            // These remain fast because they take a Body* and resolve to direct offsets
            JPH_Body_GetPosition(b, &p);
            JPH_Body_GetRotation(b, &q);
            JPH_Body_GetLinearVelocity((JPH_Body*)b, &lv);
            JPH_Body_GetAngularVelocity((JPH_Body*)b, &av);

            shadow_pos[idx]  = (PosStride){p.x, p.y, p.z};
            shadow_rot[idx]  = (AuxStride){q.x, q.y, q.z, q.w};
            shadow_lvel[idx] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
            shadow_avel[idx] = (AuxStride){av.x, av.y, av.z, 0.0f};
        }
    }
}