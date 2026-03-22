// --- START OF FILE culverin_shadow_sync.cpp ---

#include "culverin_shadow_sync.h"
#include "culverin_compiler_specifics.h"

// 2. Include native Jolt headers
#include <Jolt/Jolt.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/Body/BodyLockInterface.h>

static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4, "PosStride size mismatch");
static_assert(sizeof(AuxStride) == sizeof(float) * 4, "AuxStride size mismatch");

static constexpr int BATCH_SIZE = 32;

// Safe C++ wrapper for our worklist so we don't use opaque C pointers internally
struct CppSyncWorkItem {
    const JPH::Body *body;
    uint32_t dense_idx;
};

// =================================================================================================
// HOT PATH: Fully Unrolled, C++ Inlined, SIMD Vectorized Stores
// =================================================================================================
static CULV_FORCE_INLINE void process_full_batch(PhysicsWorldObject *self,
                                                 const CppSyncWorkItem *worklist) {
    auto *CULV_RESTRICT s_pos  = (PosStride *)CULV_ASSUME_ALIGNED(self->positions, 32);
    auto *CULV_RESTRICT s_ppos = (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, 32);
    auto *CULV_RESTRICT s_rot  = (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, 16);
    auto *CULV_RESTRICT s_prot = (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, 16);
    auto *CULV_RESTRICT s_lvel = (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, 16);
    auto *CULV_RESTRICT s_avel = (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, 16);

#pragma clang loop unroll(full)
    for (uint32_t j = 0; j < BATCH_SIZE; j++) {
        uint32_t D = worklist[j].dense_idx;

        // Native C++ Pointer - GUARANTEED SAFE
        const JPH::Body *b = worklist[j].body;

        // Snapshot previous state
        s_ppos[D] = s_pos[D];
        s_prot[D] = s_rot[D];

        // Positions
        JPH::RVec3 p = b->GetPosition();
        s_pos[D].x = p.GetX();
        s_pos[D].y = p.GetY();
        s_pos[D].z = p.GetZ();
        s_pos[D].w = 0.0;

        // Rotations
        JPH::Quat q = b->GetRotation();
        s_rot[D].x = q.GetX();
        s_rot[D].y = q.GetY();
        s_rot[D].z = q.GetZ();
        s_rot[D].w = q.GetW();

        // Velocities
        JPH::Vec3 lv = b->GetLinearVelocity();
        JPH::Vec3 av = b->GetAngularVelocity();
        s_lvel[D].x = lv.GetX();
        s_lvel[D].y = lv.GetY();
        s_lvel[D].z = lv.GetZ();
        s_lvel[D].w = 0.0f;

        s_avel[D].x = av.GetX();
        s_avel[D].y = av.GetY();
        s_avel[D].z = av.GetZ();
        s_avel[D].w = 0.0f;
    }
}

// =================================================================================================
// COLD PATH: Remainder Handling (0 to 31 items)
// =================================================================================================
static void process_partial_batch(PhysicsWorldObject *self, const CppSyncWorkItem *worklist,
                                  uint32_t count) {
    if (count == 0) {
        return;
    }

    auto *CULV_RESTRICT s_pos  = (PosStride *)CULV_ASSUME_ALIGNED(self->positions, 32);
    auto *CULV_RESTRICT s_ppos = (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, 32);
    auto *CULV_RESTRICT s_rot  = (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, 16);
    auto *CULV_RESTRICT s_prot = (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, 16);
    auto *CULV_RESTRICT s_lvel = (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, 16);
    auto *CULV_RESTRICT s_avel = (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, 16);

    for (uint32_t j = 0; j < count; j++) {
        uint32_t D = worklist[j].dense_idx;
        const JPH::Body *b = worklist[j].body;

        s_ppos[D] = s_pos[D];
        s_prot[D] = s_rot[D];

        JPH::RVec3 p = b->GetPosition();
        s_pos[D].x = p.GetX();
        s_pos[D].y = p.GetY();
        s_pos[D].z = p.GetZ();
        s_pos[D].w = 0.0;

        JPH::Quat q = b->GetRotation();
        s_rot[D].x = q.GetX();
        s_rot[D].y = q.GetY();
        s_rot[D].z = q.GetZ();
        s_rot[D].w = q.GetW();

        JPH::Vec3 lv = b->GetLinearVelocity();
        JPH::Vec3 av = b->GetAngularVelocity();
        s_lvel[D].x = lv.GetX();
        s_lvel[D].y = lv.GetY();
        s_lvel[D].z = lv.GetZ();
        s_lvel[D].w = 0.0f;

        s_avel[D].x = av.GetX();
        s_avel[D].y = av.GetY();
        s_avel[D].z = av.GetZ();
        s_avel[D].w = 0.0f;
    }
}

// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================
extern "C" void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
#ifdef CULVERIN_PROFILE_SYNC
    uint64_t start = rdtsc();
#endif

    const auto *sys_c = self->system;
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys_c, JPH_BodyType_Rigid);

    if (UNLIKELY(active_count == 0 || !self->positions)) {
        return;
    }

    // Cast the opaque system pointer to the native C++ System
    auto *native_sys = reinterpret_cast<JPH::PhysicsSystem *>(sys_c);
    
    // Use Jolt's native BodyLockInterface for guaranteed memory safety
    const JPH::BodyLockInterfaceNoLock &bli = native_sys->GetBodyLockInterfaceNoLock();

    // Still use the fast, zero-allocation array from JoltC
    const JPH_BodyID *active_ids = JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys_c, JPH_BodyType_Rigid);

    SHADOW_LOCK(&self->shadow_lock);
    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;

    // Stack allocated worklist (fits in L1 cache comfortably)
    alignas(MEMORY_ALIGNMENT_SIZE) CppSyncWorkItem worklist[BATCH_SIZE];
    uint32_t work_ptr = 0;

    for (uint32_t i = 0; i < active_count; i++) {
        // 1. Prefetch Logic (Lookahead)
        if (i + 4 < active_count) {
            const void *next_id_ptr = &active_ids[i + 4];
            CULV_PREFETCH(next_id_ptr);
        }

        // 2. Safely look up the Native Body
        // TryGetBody returns nullptr if the ID is invalid or destroyed, preventing SEGVs.
        JPH::BodyID native_id(active_ids[i]);
        const JPH::Body *b = bli.TryGetBody(native_id);

        if (UNLIKELY(!b)) {
            continue;
        }

        // 3. Filter & Validate (Inline C++)
        uint64_t handle = b->GetUserData();
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