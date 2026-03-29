// --- START OF FILE culverin_shadow_sync.cpp ---

#include "culverin_shadow_sync.h"
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"

// Include native Jolt headers for ultra-fast C++ bypass
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsSystem.h>

static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4, "PosStride size mismatch");
static_assert(sizeof(AuxStride) == sizeof(float) * 4, "AuxStride size mismatch");

static constexpr int BATCH_SIZE = 32;

// Safe C++ wrapper for our worklist so we don't use opaque C pointers internally
namespace {
struct CppSyncWorkItem {
    const JPH::Body *body;
    uint32_t dense_idx;
};

// =================================================================================================
// HOT PATH: Unrolled, Prefetched, and SIMD Vectorized Stores
// =================================================================================================

CULV_FORCE_INLINE void process_full_batch(PhysicsWorldObject *self,
                                          const CppSyncWorkItem *worklist) {
    auto *CULV_RESTRICT s_pos  = (PosStride *)CULV_ASSUME_ALIGNED(self->positions, 32);
    auto *CULV_RESTRICT s_ppos = (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, 32);
    auto *CULV_RESTRICT s_rot  = (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, 16);
    auto *CULV_RESTRICT s_prot = (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, 16);
    auto *CULV_RESTRICT s_lvel = (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, 16);
    auto *CULV_RESTRICT s_avel = (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, 16);

// Prevent I-Cache bloat and register spilling by limiting unroll count
#if defined(__clang__)
#    pragma clang loop unroll_count(4)
#elif defined(__GNUC__)
#    pragma GCC unroll 4
#endif
    for (uint32_t j = 0; j < BATCH_SIZE; j++) {
        // [OPTIMIZATION]: Lookahead prefetch destination addresses for scatter-writes.
        // Mitigates L1/L2 cache misses when dense_idx is highly randomized.
        if (j + 4 < BATCH_SIZE) {
            uint32_t future_D = worklist[j + 4].dense_idx;
            CULV_PREFETCH_WRITE(&s_pos[future_D]);
            CULV_PREFETCH_WRITE(&s_ppos[future_D]);
            CULV_PREFETCH_WRITE(&s_rot[future_D]);
            CULV_PREFETCH_WRITE(&s_prot[future_D]);
        }

        uint32_t D = worklist[j].dense_idx;

        // Native C++ Pointer - GUARANTEED SAFE
        const JPH::Body *b = worklist[j].body;

        // Snapshot previous state (Wide 128/256-bit copy)
        s_ppos[D] = s_pos[D];
        s_prot[D] = s_rot[D];

        // Positions: Safe scalar fallback due to JPH_DOUBLE_PRECISION toggles.
        JPH::RVec3 p = b->GetPosition();
        s_pos[D].x   = p.GetX();
        s_pos[D].y   = p.GetY();
        s_pos[D].z   = p.GetZ();
        s_pos[D].w   = 0.0;

        // [OPTIMIZATION]: 128-bit SIMD Store Rotations (X, Y, Z, W)
        b->GetRotation().GetXYZW().StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_rot[D]));

        // [OPTIMIZATION]: 128-bit SIMD Store Velocities (Forces W to 0.0f safely)
        JPH::Vec4(b->GetLinearVelocity(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_lvel[D]));
        JPH::Vec4(b->GetAngularVelocity(), 0.0f)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *>(&s_avel[D]));
    }
}

// =================================================================================================
// COLD PATH: Remainder Handling (0 to 31 items)
// =================================================================================================
void process_partial_batch(PhysicsWorldObject *self, const CppSyncWorkItem *worklist,
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
        // Minor prefetch for the remainder loop
        if (j + 2 < count) {
            uint32_t future_D = worklist[j + 2].dense_idx;
            CULV_PREFETCH_WRITE(&s_pos[future_D]);
            CULV_PREFETCH_WRITE(&s_rot[future_D]);
        }

        uint32_t D         = worklist[j].dense_idx;
        const JPH::Body *b = worklist[j].body;

        s_ppos[D] = s_pos[D];
        s_prot[D] = s_rot[D];

        JPH::RVec3 p = b->GetPosition();
        s_pos[D].x   = p.GetX();
        s_pos[D].y   = p.GetY();
        s_pos[D].z   = p.GetZ();
        s_pos[D].w   = 0.0;

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
#ifdef CULVERIN_PROFILE_SYNC
    uint64_t start = rdtsc();
#endif

    if (UNLIKELY(!self)) {
        return;
    }

    if (UNLIKELY(!self->system)) {
        return;
    }

    // If the Main Thread is reallocating, DO NOT touch the pointers.
    // The main thread is holding the shadow_lock or about to move buffers.
    if (UNLIKELY(atomic_load_explicit(&self->is_resizing, std::memory_order_acquire))) {
        return;
    }

    const auto *sys_c     = self->system;
    uint32_t active_count = JPH_PhysicsSystem_GetNumActiveBodies(sys_c, JPH_BodyType_Rigid);

    if (UNLIKELY(active_count == 0)) {
        return;
    }

    const JPH_BodyID *active_ids =
        JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys_c, JPH_BodyType_Rigid);
    if (UNLIKELY(!active_ids)) {
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

    const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;

    // Stack allocated worklist (fits in L1 cache comfortably)
    alignas(MEMORY_ALIGNMENT_SIZE) CppSyncWorkItem worklist[BATCH_SIZE];
    uint32_t work_ptr = 0;

    for (uint32_t i = 0; i < active_count; i++) {
        if (i + 4 < active_count) {
            const void *next_id_ptr = &active_ids[i + 4];
            CULV_PREFETCH(next_id_ptr);
        }

        // [THE FIX] We use the C-API to safely navigate the hidden struct and Jolt's Read Locks.
        const JPH_Body *opaque_body = JPH_PhysicsSystem_GetBodyPtr(sys_c, active_ids[i]);

        if (UNLIKELY(opaque_body == nullptr)) {
            continue;
        }

        const JPH::Body *b = reinterpret_cast<const JPH::Body *>(opaque_body);

        uint64_t handle = b->GetUserData();
        auto slot       = (uint32_t)(handle & HANDLE_INDEX_MASK);
        auto gen        = (uint32_t)(handle >> HANDLE_INDEX_BITS);

        // 1. Calculate the bounds check (0 if in bounds, non-zero if out)
        auto out_of_bounds = (uint32_t)(slot >= self->slot_capacity);

        // 2. Bitwise OR the conditions
        // If any condition is true (non-zero), the result is non-zero.
        if (UNLIKELY(out_of_bounds | (self->generations[slot] ^ gen) |
                     (self->slot_states[slot] ^ SLOT_ALIVE))) {
            return;
        }
        // Now the "Hot Path" is flat and easy to read
        worklist[work_ptr].body      = b;
        worklist[work_ptr].dense_idx = s2d[slot];
        work_ptr++;

        if (work_ptr == BATCH_SIZE) {
            process_full_batch(self, worklist);
            work_ptr = 0;
        }
    }

    // --- COLD PATH (REMAINDER) ---
    if (work_ptr > 0) {
        process_partial_batch(self, worklist, work_ptr);
    }

#ifdef CULVERIN_PROFILE_SYNC
    uint64_t elapsed = rdtsc() - start;
    if (active_count > 0) {
        fprintf(stderr, "Sync: %llu cycles for %u bodies (%.1f cyc/body)\n", elapsed, active_count,
                (double)elapsed / active_count);
    }
#endif
}