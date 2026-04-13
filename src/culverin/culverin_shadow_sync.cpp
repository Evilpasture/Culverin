#include "culverin_shadow_sync.h"
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"

// Include native Jolt headers for ultra-fast C++ bypass
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodyVertex.h>

static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4, "PosStride size mismatch");
static_assert(sizeof(AuxStride) == sizeof(float) * 4, "AuxStride size mismatch");

static constexpr int BATCH_SIZE = 128;

namespace {
struct CppSyncWorkItem {
    const JPH::Body *body;
    uint32_t dense_idx;
};

// =================================================================================================
// HOT PATH: Unrolled, Prefetched, and SIMD Vectorized Stores (RIGID BODIES)
// =================================================================================================

CULV_FORCE_INLINE void process_full_batch(PhysicsWorldObject *const CULV_RESTRICT self,
                                          const CppSyncWorkItem *const CULV_RESTRICT worklist) {
    PosStride * CULV_RESTRICT s_pos =
        (PosStride *)CULV_ASSUME_ALIGNED(self->positions, sizeof(PosStride));
    PosStride * CULV_RESTRICT s_ppos =
        (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, sizeof(PosStride));
    AuxStride * CULV_RESTRICT s_rot =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, sizeof(AuxStride));
    AuxStride * CULV_RESTRICT s_prot =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, sizeof(AuxStride));
    AuxStride * CULV_RESTRICT s_lvel =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, sizeof(AuxStride));
    AuxStride * CULV_RESTRICT s_avel =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, sizeof(AuxStride));

    // Prevent I-Cache bloat and register spilling by limiting unroll count
    CULV_UNROLL_LOOP(4)
    for (uint32_t j = 0; j < BATCH_SIZE; j++) {
        const uint32_t D = worklist[j].dense_idx;

        // Native C++ Pointer - GUARANTEED SAFE
        const JPH::Body *b = worklist[j].body;

        // Snapshot previous state (Wide 128/256-bit copy)
        const PosStride old_pos = s_pos[D];
        const AuxStride old_rot = s_rot[D];
        s_ppos[D]               = old_pos;
        s_prot[D]               = old_rot;

#ifndef JPH_DOUBLE_PRECISION
        JPH::Vec4(b->GetCenterOfMassPosition(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_pos[D]));
#else
        b->GetCenterOfMassPosition().StoreDouble3(reinterpret_cast<JPH::Double3 *>(&s_pos[D]));
        s_pos[D].w = 0.0;
#endif

        b->GetRotation().GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_rot[D]));
        JPH::Vec4(b->GetLinearVelocity(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_lvel[D]));
        JPH::Vec4(b->GetAngularVelocity(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_avel[D]));
    }
}

// =================================================================================================
// COLD PATH: Remainder Handling (0 to 31 items)
// =================================================================================================
CULV_FORCE_INLINE void process_partial_batch(PhysicsWorldObject *const CULV_RESTRICT self,
                                             const CppSyncWorkItem *const CULV_RESTRICT worklist,
                                             const uint32_t count) {
    if (count == 0) {
        return;
    }

    PosStride * CULV_RESTRICT s_pos =
        (PosStride *)CULV_ASSUME_ALIGNED(self->positions, sizeof(PosStride));
    PosStride * CULV_RESTRICT s_ppos =
        (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, sizeof(PosStride));
    AuxStride * CULV_RESTRICT s_rot =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, sizeof(AuxStride));
    AuxStride * CULV_RESTRICT s_prot =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, sizeof(AuxStride));
    AuxStride * CULV_RESTRICT s_lvel =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, sizeof(AuxStride));
    AuxStride * CULV_RESTRICT s_avel =
        (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, sizeof(AuxStride));

    for (uint32_t j = 0; j < count; j++) {
        if (j + 2 < count) {
            const uint32_t future_D = worklist[j + 2].dense_idx;
            CULV_PREFETCH_WRITE(&s_pos[future_D]);
            CULV_PREFETCH_WRITE(&s_rot[future_D]);
        }

        const uint32_t D   = worklist[j].dense_idx;
        const JPH::Body *b = worklist[j].body;

        const PosStride old_pos = s_pos[D];
        const AuxStride old_rot = s_rot[D];
        s_ppos[D]               = old_pos;
        s_prot[D]               = old_rot;

#ifndef JPH_DOUBLE_PRECISION
        JPH::Vec4(b->GetCenterOfMassPosition(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_pos[D]));
#else
        b->GetCenterOfMassPosition().StoreDouble3(reinterpret_cast<JPH::Double3 *>(&s_pos[D]));
        s_pos[D].w = 0.0;
#endif

        b->GetRotation().GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_rot[D]));
        JPH::Vec4(b->GetLinearVelocity(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_lvel[D]));
        JPH::Vec4(b->GetAngularVelocity(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_avel[D]));
    }
}
} // namespace

// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================
extern "C" void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {

    if (UNLIKELY(!self)) {
        return;
    }

    if (UNLIKELY(!self->system)) {
        return;
    }

    // Check the death flag! If the main thread is deallocating, we must not touch any pointers or
    // issue Jolt calls.
    if (UNLIKELY(atomic_load_explicit(&self->is_deallocating, JPH::memory_order_acquire))) {
        return;
    }

    // If the Main Thread is reallocating, DO NOT touch the pointers.
    // The main thread is holding the shadow_lock or about to move buffers.
    if (UNLIKELY(atomic_load_explicit(&self->is_resizing, JPH::memory_order_acquire))) {
        return;
    }

    const auto *sys_c = self->system;

    // Check for both Rigid and Soft active bodies
    const uint32_t active_rigid_count =
        JPH_PhysicsSystem_GetNumActiveBodies(sys_c, JPH_BodyType_Rigid);
    const uint32_t active_soft_count =
        JPH_PhysicsSystem_GetNumActiveBodies(sys_c, JPH_BodyType_Soft);

    if (UNLIKELY(!(active_rigid_count || active_soft_count))) {
        return;
    }

    if (UNLIKELY(!self->positions)) {
        return;
    }
    if (UNLIKELY(!self->slot_to_dense)) {
        return;
    }
    if (UNLIKELY(!self->generations)) {
        return;
    }
    if (UNLIKELY(!self->slot_states)) {
        return;
    }

    static constexpr size_t MIN_CYCLES           = 0xFFFFFFFFFFFFFFFFULL;
    CULV_MAYBE_UNUSED static CulvStat sync_stats = {
        .total_cycles = 0, .min_cycles = MIN_CYCLES, .max_cycles = 0, .count = 0};

    CULV_PROFILE_BEGIN(sync);

    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;
    PosStride * CULV_RESTRICT s_pos         = (PosStride *)self->positions;
    AuxStride * CULV_RESTRICT s_rot         = (AuxStride *)self->rotations;

    const JPH::BodyLockInterfaceNoLock *lock_iface =
        reinterpret_cast<const JPH::BodyLockInterfaceNoLock *>(
            JPH_PhysicsSystem_GetBodyLockInterfaceNoLock(sys_c));

    // ========================================================================
    // PASS 1: RIGID BODIES
    // ========================================================================
    if (active_rigid_count > 0) {
        const JPH_BodyID *active_rigid_ids =
            JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys_c, JPH_BodyType_Rigid);
        if (LIKELY(active_rigid_ids)) {
            alignas(MEMORY_ALIGNMENT_SIZE) CppSyncWorkItem worklist[BATCH_SIZE];
            uint32_t work_ptr = 0;

            for (uint32_t i = 0; i < active_rigid_count; i++) {
                const JPH::Body *b = lock_iface->TryGetBody(JPH::BodyID(active_rigid_ids[i]));
                if (UNLIKELY(b == nullptr)) {
                    continue;
                }

                const uint64_t handle = b->GetUserData();
                const auto slot       = (uint32_t)(handle & HANDLE_INDEX_MASK);
                const auto gen        = (uint32_t)(handle >> HANDLE_INDEX_BITS);

                const uint32_t safe_slot = (slot < self->slot_capacity) ? slot : 0;

                const uint8_t state =
                    __atomic_load_n((uint8_t *)&self->slot_states[safe_slot], __ATOMIC_ACQUIRE);
                const uint32_t current_gen =
                    __atomic_load_n((uint32_t *)&self->generations[safe_slot], __ATOMIC_ACQUIRE);

                const uint32_t state_bad = (state == SLOT_ALIVE || state == SLOT_CHARACTER) ? 0 : 1;
                const uint32_t bad       = static_cast<uint32_t>(slot >= self->slot_capacity) |
                                           (current_gen ^ gen) | state_bad;

                const uint32_t d_idx = s2d[safe_slot];

                CULV_PREFETCH_WRITE(&s_pos[d_idx]);
                CULV_PREFETCH_WRITE(&s_rot[d_idx]);

                CULV_ASSUME(work_ptr < BATCH_SIZE);
                const uint32_t is_valid = static_cast<uint32_t>(bad == 0);
                worklist[work_ptr].body = (is_valid != 0u) ? b : worklist[work_ptr].body;
                worklist[work_ptr].dense_idx =
                    (is_valid != 0u) ? d_idx : worklist[work_ptr].dense_idx;
                work_ptr += is_valid;

                if (work_ptr == BATCH_SIZE) {
                    process_full_batch(self, worklist);
                    work_ptr = 0;
                }
            }

            if (work_ptr > 0) {
                process_partial_batch(self, worklist, work_ptr);
            }
        }
    }

    // ========================================================================
    // PASS 2: SOFT BODIES
    // ========================================================================
    if (active_soft_count > 0 && self->soft_shadows != nullptr) {
        const JPH_BodyID *active_soft_ids =
            JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys_c, JPH_BodyType_Soft);

        if (LIKELY(active_soft_ids)) {
            PosStride * CULV_RESTRICT s_ppos = (PosStride *)self->prev_positions;
            AuxStride * CULV_RESTRICT s_prot = (AuxStride *)self->prev_rotations;

            for (uint32_t i = 0; i < active_soft_count; i++) {
                const JPH::Body *b = lock_iface->TryGetBody(JPH::BodyID(active_soft_ids[i]));
                if (UNLIKELY(!(b && b->IsSoftBody()))) {
                    continue;
                }

                const uint64_t handle    = b->GetUserData();
                const auto slot          = (uint32_t)(handle & HANDLE_INDEX_MASK);
                const auto gen           = (uint32_t)(handle >> HANDLE_INDEX_BITS);
                const uint32_t safe_slot = (slot < self->slot_capacity) ? slot : 0;

                const uint8_t state =
                    __atomic_load_n((uint8_t *)&self->slot_states[safe_slot], __ATOMIC_ACQUIRE);
                const uint32_t current_gen =
                    __atomic_load_n((uint32_t *)&self->generations[safe_slot], __ATOMIC_ACQUIRE);

                if (slot >= self->slot_capacity || current_gen != gen || state != SLOT_SOFT_BODY) {
                    continue;
                }

                const uint32_t d_idx = s2d[safe_slot];

                // 1. Sync Center of Mass & Rotation (Keeps traditional rigid-body query functions
                // happy)
                s_ppos[d_idx] = s_pos[d_idx];
                s_prot[d_idx] = s_rot[d_idx];

#ifndef JPH_DOUBLE_PRECISION
                JPH::Vec4(b->GetCenterOfMassPosition(), 0.0f)
                    .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_pos[d_idx]));
#else
                b->GetCenterOfMassPosition().StoreDouble3(
                    reinterpret_cast<JPH::Double3 *>(&s_pos[d_idx]));
                s_pos[d_idx].w = 0.0;
#endif
                b->GetRotation().GetXYZW().StoreFloat4(
                    reinterpret_cast<JPH::Float4 *>(&s_rot[d_idx]));

                // 2. Sync Soft Body Vertices
                const JPH::MotionProperties *mp = b->GetMotionProperties();
                if ((mp != nullptr) && b->IsSoftBody()) {
                    const auto *soft_mp = static_cast<const JPH::SoftBodyMotionProperties *>(mp);
                    const JPH::Array<JPH::SoftBodyVertex> &jolt_verts = soft_mp->GetVertices();

                    SoftBodyShadow &shadow = self->soft_shadows[d_idx];

                    // Safety check to ensure arrays align
                    if ((shadow.vertices != nullptr) && shadow.num_vertices == jolt_verts.size()) {
                        auto *dst_verts = reinterpret_cast<PosStride *>(shadow.vertices);

                        // Jolt's soft body vertices are local to the Center of Mass transform.
                        JPH::RMat44 com_transform = b->GetCenterOfMassTransform();

                        const size_t num_v = shadow.num_vertices;
                        for (size_t v = 0; v < num_v; ++v) {
                            JPH::Vec3 local_pos(jolt_verts[v].mPosition);

#ifndef JPH_DOUBLE_PRECISION
                            JPH::Vec3 world_pos = com_transform * local_pos;
                            JPH::Vec4(world_pos, 0.0f)
                                .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&dst_verts[v]));
#else
                            JPH::RVec3 world_pos = com_transform * local_pos;
                            world_pos.StoreDouble3(reinterpret_cast<JPH::Double3 *>(&dst_verts[v]));
                            dst_verts[v].w = 0.0;
#endif
                        }
                    }
                }
            }
        }
    }

    CULV_PROFILE_ACCUMULATE(sync, &sync_stats);
#ifdef CULVERIN_PROFILE_SYNC
    if (sync_stats.count >= 50) {
        fprintf(stderr, "[culverin] Sync Stat Avg: %" PRIu64 " | Max: %" PRIu64 "\n",
                sync_stats.total_cycles / sync_stats.count, sync_stats.max_cycles);

        // Reset
        sync_stats = (CulvStat){0, MIN_CYCLES, 0, 0};
    }
#endif
}