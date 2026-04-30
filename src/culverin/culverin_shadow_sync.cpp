#include "culverin_shadow_sync.h"
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"

// Include native Jolt headers for ultra-fast C++ bypass
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodyVertex.h>
#include <cstddef>

static_assert(sizeof(PosStride) == sizeof(JPH::Real) * 4, "PosStride size mismatch");
static_assert(sizeof(AuxStride) == sizeof(float) * 4, "AuxStride size mismatch");

static constexpr int BATCH_SIZE = 128;

namespace {

struct SyncWorkItem {
    const JPH::Body *body;
    uint32_t dense_idx;
};

struct WorldDataCreateInfo {
    PosStride *const CULV_RESTRICT shadow_pos;
    PosStride *const CULV_RESTRICT shadow_ppos;
    AuxStride *const CULV_RESTRICT shadow_rot;
    AuxStride *const CULV_RESTRICT shadow_prot;
    AuxStride *const CULV_RESTRICT shadow_lvel;
    AuxStride *const CULV_RESTRICT shadow_avel;
    const SoftBodyShadow *const CULV_RESTRICT soft_shadows;
};

static_assert(std::is_trivially_copyable<WorldDataCreateInfo>());

// =================================================================================================
// UNIFIED ITEM PROCESSOR
// Automatically eliminates dead code branches at compile time via `if constexpr`.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline]] inline void process_item(const uint32_t D,
                                                const JPH::Body *const CULV_RESTRICT b,
                                                const WorldDataCreateInfo world) noexcept {

    // 1. Snapshot previous state
    world.shadow_ppos[D] = world.shadow_pos[D];
    world.shadow_prot[D] = world.shadow_rot[D];

    // 2. Write Current COM
#ifndef JPH_DOUBLE_PRECISION
    [[clang::always_inline]] JPH::Vec4(b->GetCenterOfMassPosition(), 0.0f)
        .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&world.shadow_pos[D]));
#else
    [[clang::always_inline]] b->GetCenterOfMassPosition().StoreDouble3(
        reinterpret_cast<JPH::Double3 *const CULV_RESTRICT>(&world.shadow_pos[D]));
    world.shadow_pos[D].w = 0.0;
#endif

    // 3. Write Current Rotation
    [[clang::always_inline]] b->GetRotation().GetXYZW().StoreFloat4(
        reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&world.shadow_rot[D]));

    // 4. Type-Specific Sub-Data
    if constexpr (TType == JPH::EBodyType::RigidBody) {
        [[clang::always_inline]] JPH::Vec4(b->GetLinearVelocity(), 0.0F)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&world.shadow_lvel[D]));
        [[clang::always_inline]] JPH::Vec4(b->GetAngularVelocity(), 0.0F)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&world.shadow_avel[D]));
    } else if constexpr (TType == JPH::EBodyType::SoftBody) {
        const auto *const CULV_RESTRICT soft_mp =
            static_cast<const JPH::SoftBodyMotionProperties *const CULV_RESTRICT>(
                b->GetMotionProperties());
        const JPH::Array<JPH::SoftBodyVertex> &jolt_verts = soft_mp->GetVertices();
        const SoftBodyShadow &shadow                      = world.soft_shadows[D];

        if ((shadow.vertices != nullptr) && shadow.num_vertices == jolt_verts.size()) [[likely]] {
            auto *const CULV_RESTRICT dst_verts =
                reinterpret_cast<PosStride *const CULV_RESTRICT>(shadow.vertices);
            const JPH::Quat rotation     = b->GetRotation();
            const JPH::RVec3 translation = b->GetCenterOfMassPosition();
            const size_t num_v           = shadow.num_vertices;

            CULV_UNROLL_LOOP(8)
            for (size_t v = 0; v < num_v; ++v) {
                if (v + 8 < num_v) {
                    CULV_PREFETCH_WRITE(&dst_verts[v + 8]);
                }

                const JPH::Vec3 local_pos(jolt_verts[v].mPosition);
#ifndef JPH_DOUBLE_PRECISION
                const JPH::Vec3 world_pos = (rotation * local_pos) + translation;
                [[clang::always_inline]] JPH::Vec4(world_pos, 0.0f)
                    .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&dst_verts[v]));
#else
                const JPH::RVec3 world_pos = JPH::RVec3(rotation * local_pos) + translation;
                [[clang::always_inline]] world_pos.StoreDouble3(
                    reinterpret_cast<JPH::Double3 *const CULV_RESTRICT>(&dst_verts[v]));
                dst_verts[v].w = 0.0;
#endif
            }
        }
    }
}

// =================================================================================================
// UNIFIED BATCH PROCESSOR
// Routes to either fully unrolled loops (FixedCount > 0) or prefetched remainder loops.
// =================================================================================================
template <JPH::EBodyType TType, uint32_t FixedCount = 0>
[[gnu::always_inline, gnu::hot, gnu::flatten]] inline void
process_batch(const WorldDataCreateInfo world, const SyncWorkItem *const CULV_RESTRICT worklist,
              const uint32_t dynamic_count = 0) noexcept {

    if constexpr (FixedCount > 0) {
        CULV_UNROLL_LOOP(8)
        for (uint32_t j = 0; j < FixedCount; j++) {
            process_item<TType>(worklist[j].dense_idx, worklist[j].body, world);
        }
    } else {
        CULV_UNROLL_LOOP(4)
        for (uint32_t j = 0; j < dynamic_count; j++) {
            if (j + 2 < dynamic_count) {
                CULV_PREFETCH_WRITE(&world.shadow_pos[worklist[j + 2].dense_idx]);
                CULV_PREFETCH_WRITE(&world.shadow_rot[worklist[j + 2].dense_idx]);
            }
            process_item<TType>(worklist[j].dense_idx, worklist[j].body, world);
        }
    }
}

} // namespace

// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================

extern "C" [[gnu::flatten, gnu::hot, gnu::nonnull(1)]] void
culverin_sync_shadow_buffers(const PhysicsWorldObject *const CULV_RESTRICT self) noexcept {
    using namespace JPH;
    if (!self->sync_ready) [[unlikely]] {
        return;
    }
    const auto *const CULV_RESTRICT system =
        static_cast<PhysicsSystem *>(JPH_PhysicsSystem_GetPhysicsSystemInstance(self->system));

    // Check for both Rigid and Soft active bodies
    const uint32_t active_rigid_count = system->GetNumActiveBodies(EBodyType::RigidBody);
    const uint32_t active_soft_count  = system->GetNumActiveBodies(EBodyType::SoftBody);

    if ((active_rigid_count == 0U) && (active_soft_count == 0U)) [[unlikely]] {
        return;
    }
    static constexpr uint64_t MIN_CYCLES         = 0xFFFFFFFFFFFFFFFFULL;
    CULV_MAYBE_UNUSED static CulvStat sync_stats = {
        .total_cycles = 0, .min_cycles = MIN_CYCLES, .max_cycles = 0, .count = 0};

    CULV_PROFILE_BEGIN(sync);

    const uint32_t *const CULV_RESTRICT s2d = self->slot_to_dense;
    auto *const CULV_RESTRICT s_pos         = reinterpret_cast<PosStride *const CULV_RESTRICT>(
        CULV_ASSUME_ALIGNED(self->positions, sizeof(PosStride)));
    auto *const CULV_RESTRICT s_ppos = reinterpret_cast<PosStride *const CULV_RESTRICT>(
        CULV_ASSUME_ALIGNED(self->prev_positions, sizeof(PosStride)));
    auto *const CULV_RESTRICT s_rot = reinterpret_cast<AuxStride *const CULV_RESTRICT>(
        CULV_ASSUME_ALIGNED(self->rotations, sizeof(AuxStride)));
    auto *const CULV_RESTRICT s_prot = reinterpret_cast<AuxStride *const CULV_RESTRICT>(
        CULV_ASSUME_ALIGNED(self->prev_rotations, sizeof(AuxStride)));
    auto *const CULV_RESTRICT s_lvel = reinterpret_cast<AuxStride *const CULV_RESTRICT>(
        CULV_ASSUME_ALIGNED(self->linear_velocities, sizeof(AuxStride)));
    auto *const CULV_RESTRICT s_avel = reinterpret_cast<AuxStride *const CULV_RESTRICT>(
        CULV_ASSUME_ALIGNED(self->angular_velocities, sizeof(AuxStride)));

    const auto *const CULV_RESTRICT lock_iface = &system->GetBodyLockInterfaceNoLock();

    // ========================================================================
    // PASS 1: RIGID BODIES
    // ========================================================================
    if (active_rigid_count > 0) {
        const BodyID *const CULV_RESTRICT active_rigid_ids =
            system->GetActiveBodiesUnsafe(EBodyType::RigidBody);
        if (active_rigid_ids != nullptr) [[unlikely]] {
            alignas(MEMORY_ALIGNMENT_SIZE) SyncWorkItem worklist[BATCH_SIZE];
            uint32_t work_ptr = 0;

            const void *const CULV_RESTRICT *const CULV_RESTRICT body_ptrs = self->jolt_body_ptrs;
            const auto *const CULV_RESTRICT slot_states                    = self->slot_states;
            const auto *const CULV_RESTRICT generations                    = self->generations;
            const size_t slot_capacity                                     = self->slot_capacity;
            for (uint32_t i = 0; i < active_rigid_count; i++) {
                const uint32_t raw_jolt_id = active_rigid_ids[i].GetIndexAndSequenceNumber();
                const uint32_t j_idx       = raw_jolt_id & JPH::BodyID::cMaxBodyIndex;

                // Read from our flat cache
                const auto *CULV_RESTRICT b =
                    static_cast<const Body * CULV_RESTRICT>(body_ptrs[j_idx]);

                // Validate pointer and Sequence ID (in case Jolt destroyed and reused this slot)
                [[clang::always_inline]] if (b == nullptr ||
                                             b->GetID().GetIndexAndSequenceNumber() != raw_jolt_id)
                    [[unlikely]] {
                    // Cache miss: Fallback to Jolt lookup and update our cache
                    [[clang::always_inline]] b  = lock_iface->TryGetBody(BodyID(raw_jolt_id));
                    self->jolt_body_ptrs[j_idx] = b;
                }

                if (b == nullptr) [[unlikely]] {
                    continue;
                }

                const uint64_t handle = b->GetUserData();
                const auto slot       = static_cast<const uint32_t>(handle & HANDLE_INDEX_MASK);
                const auto gen        = static_cast<const uint32_t>(handle >> HANDLE_INDEX_BITS);

                const uint32_t safe_slot = (slot < slot_capacity) ? slot : 0;

                const uint8_t state        = slot_states[safe_slot].load(std::memory_order_acquire);
                const uint32_t current_gen = generations[safe_slot].load(std::memory_order_acquire);

                const uint32_t state_bad = (state == SLOT_ALIVE || state == SLOT_CHARACTER) ? 0 : 1;
                const uint32_t bad       = static_cast<const uint32_t>(slot >= slot_capacity) |
                                           (current_gen ^ gen) | state_bad;

                const uint32_t d_idx = s2d[safe_slot];

                CULV_PREFETCH_WRITE(&s_pos[d_idx]);
                CULV_PREFETCH_WRITE(&s_rot[d_idx]);

                CULV_ASSUME(work_ptr < BATCH_SIZE);
                const auto is_valid          = static_cast<const uint32_t>(bad == 0);
                worklist[work_ptr].body      = b;
                worklist[work_ptr].dense_idx = d_idx;
                work_ptr += is_valid;

                if (work_ptr == BATCH_SIZE) {
                    process_batch<EBodyType::RigidBody, BATCH_SIZE>(
                        WorldDataCreateInfo{

                            .shadow_pos   = s_pos,
                            .shadow_ppos  = s_ppos,
                            .shadow_rot   = s_rot,
                            .shadow_prot  = s_prot,
                            .shadow_lvel  = s_lvel,
                            .shadow_avel  = s_avel,
                            .soft_shadows = nullptr

                        },
                        worklist);
                    work_ptr = 0;
                }
            }

            if (work_ptr > 0) {
                process_batch<EBodyType::RigidBody, 0>(
                    WorldDataCreateInfo{

                        .shadow_pos   = s_pos,
                        .shadow_ppos  = s_ppos,
                        .shadow_rot   = s_rot,
                        .shadow_prot  = s_prot,
                        .shadow_lvel  = s_lvel,
                        .shadow_avel  = s_avel,
                        .soft_shadows = nullptr

                    },
                    worklist, work_ptr);
            }
        }
    }

    // ========================================================================
    // PASS 2: SOFT BODIES (Branchless Dispatch)
    // ========================================================================
    if (active_soft_count > 0 && self->soft_shadows != nullptr) {
        const BodyID *const CULV_RESTRICT active_soft_ids =
            system->GetActiveBodiesUnsafe(EBodyType::SoftBody);

        if (active_soft_ids != nullptr) [[likely]] {
            alignas(MEMORY_ALIGNMENT_SIZE) SyncWorkItem soft_worklist[BATCH_SIZE];
            uint32_t soft_work_ptr = 0;

            const void *const CULV_RESTRICT *const CULV_RESTRICT body_ptrs = self->jolt_body_ptrs;
            const auto *const CULV_RESTRICT slot_states                    = self->slot_states;
            const auto *const CULV_RESTRICT generations                    = self->generations;
            const size_t slot_capacity                                     = self->slot_capacity;
            const auto *const CULV_RESTRICT soft_shadows                   = self->soft_shadows;
            for (uint32_t i = 0; i < active_soft_count; i++) {
                const uint32_t raw_jolt_id = active_soft_ids[i].GetIndexAndSequenceNumber();
                const uint32_t j_idx       = raw_jolt_id & JPH::BodyID::cMaxBodyIndex;

                const auto *CULV_RESTRICT b =
                    static_cast<const Body * CULV_RESTRICT>(body_ptrs[j_idx]);

                [[clang::always_inline]] if (b == nullptr ||
                                             b->GetID().GetIndexAndSequenceNumber() != raw_jolt_id)
                    [[unlikely]] {
                    [[clang::always_inline]] b  = lock_iface->TryGetBody(BodyID(raw_jolt_id));
                    self->jolt_body_ptrs[j_idx] = b;
                }

                if ((b == nullptr) || !b->IsSoftBody()) [[unlikely]] {
                    continue;
                }

                const uint64_t handle    = b->GetUserData();
                const auto slot          = static_cast<const uint32_t>(handle & HANDLE_INDEX_MASK);
                const auto gen           = static_cast<const uint32_t>(handle >> HANDLE_INDEX_BITS);
                const uint32_t safe_slot = (slot < slot_capacity) ? slot : 0;

                const uint8_t state        = slot_states[safe_slot].load(std::memory_order_acquire);
                const uint32_t current_gen = generations[safe_slot].load(std::memory_order_acquire);

                // --- BRANCHLESS VALIDATION ---
                const uint32_t state_bad = (state == SLOT_SOFT_BODY) ? 0 : 1;
                const uint32_t bad       = static_cast<const uint32_t>(slot >= slot_capacity) |
                                           (current_gen ^ gen) | state_bad;

                const uint32_t d_idx = s2d[safe_slot];
                const auto is_valid  = static_cast<uint32_t>(bad == 0);

                // Prefetch the vertex shadow buffer metadata
                CULV_PREFETCH_READ(&soft_shadows[d_idx]);

                CULV_ASSUME(soft_work_ptr < BATCH_SIZE);
                soft_worklist[soft_work_ptr].body      = b;
                soft_worklist[soft_work_ptr].dense_idx = d_idx;
                soft_work_ptr += is_valid;

                if (soft_work_ptr == BATCH_SIZE) {
                    process_batch<EBodyType::SoftBody, BATCH_SIZE>(
                        WorldDataCreateInfo{

                            .shadow_pos   = s_pos,
                            .shadow_ppos  = s_ppos,
                            .shadow_rot   = s_rot,
                            .shadow_prot  = s_prot,
                            .shadow_lvel  = nullptr,
                            .shadow_avel  = nullptr,
                            .soft_shadows = soft_shadows

                        },
                        soft_worklist);
                    soft_work_ptr = 0;
                }
            }

            if (soft_work_ptr > 0) {
                process_batch<EBodyType::SoftBody, 0>(
                    WorldDataCreateInfo{

                        .shadow_pos   = s_pos,
                        .shadow_ppos  = s_ppos,
                        .shadow_rot   = s_rot,
                        .shadow_prot  = s_prot,
                        .shadow_lvel  = nullptr,
                        .shadow_avel  = nullptr,
                        .soft_shadows = soft_shadows

                    },
                    soft_worklist, soft_work_ptr);
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