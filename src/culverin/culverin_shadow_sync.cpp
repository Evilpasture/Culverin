#include "culverin_shadow_sync.h"
#include "CulverinPrefetch"
#include "CulverinSpan"
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"

// Include native Jolt headers for ultra-fast C++ bypass
#include <Jolt/Jolt.h>
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodyVertex.h>
#include <array>
#include <cstddef>
#include <memory>

static_assert(sizeof(PosStride) == sizeof(JPH::Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(CPH::Float32) * 4);

namespace CPH {

constexpr Unsigned32 BATCH_SIZE = 128;
struct SyncWorkItem {
    const JPH::Body *body;
    Unsigned32 dense_idx;
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

struct MappingDataCreateInfo {
    const void *CULV_RESTRICT *const CULV_RESTRICT body_ptrs;
    const std::atomic<Unsigned8> *const CULV_RESTRICT slot_states;
    const std::atomic<Unsigned32> *const CULV_RESTRICT generations;
    const SizeType slot_capacity;
    const Unsigned32 *const CULV_RESTRICT slot_to_dense;
};

static_assert(std::is_trivially_copyable<WorldDataCreateInfo>());
static_assert(std::is_trivial<WorldDataCreateInfo>());
static_assert(std::is_trivially_copyable<MappingDataCreateInfo>());
static_assert(std::is_trivial<MappingDataCreateInfo>());

#ifdef JPH_DOUBLE_PRECISION
inline constexpr bool double_precision = true;
#else
inline constexpr bool double_precision = false;
#endif

using PosPointerType = JPH::Double3 *const CULV_RESTRICT;
using AuxPointerType = JPH::Float4 *const CULV_RESTRICT;

// =================================================================================================
// UNIFIED ITEM PROCESSOR
// Automatically eliminates dead code branches at compile time via `if constexpr`.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline, gnu::nonnull(2)]] inline auto
ProcessItem(const Unsigned32 D, const JPH::Body *const CULV_RESTRICT b,
            const WorldDataCreateInfo world) noexcept -> void {

    // 1. Snapshot previous state
    world.shadow_ppos[D] = world.shadow_pos[D];
    world.shadow_prot[D] = world.shadow_rot[D];

    // 2. Write Current COM
    auto *const target = &world.shadow_pos[D];

    const auto &rotation    = b->GetRotation();
    const auto &translation = b->GetCenterOfMassPosition();

    if constexpr (double_precision) {
        [[clang::always_inline]] translation.StoreDouble3(reinterpret_cast<PosPointerType>(target));
        target->w = 0.0;
    } else {
        [[clang::always_inline]] JPH::Vec4(JPH::Vec3(translation), 0.0F)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(target));
    }

    // 3. Write Current Rotation

    [[clang::always_inline]] rotation.GetXYZW().StoreFloat4(
        reinterpret_cast<AuxPointerType>(&world.shadow_rot[D]));

    // 4. Type-Specific Sub-Data
    if constexpr (TType == JPH::EBodyType::RigidBody) {
        [[clang::always_inline]] JPH::Vec4(b->GetLinearVelocity(), 0.0F)
            .StoreFloat4(reinterpret_cast<AuxPointerType>(&world.shadow_lvel[D]));
        [[clang::always_inline]] JPH::Vec4(b->GetAngularVelocity(), 0.0F)
            .StoreFloat4(reinterpret_cast<AuxPointerType>(&world.shadow_avel[D]));
    } else if constexpr (TType == JPH::EBodyType::SoftBody) {
        const auto *const CULV_RESTRICT soft_mp =
            static_cast<const JPH::SoftBodyMotionProperties *const CULV_RESTRICT>(
                b->GetMotionProperties());
        const JPH::Array<JPH::SoftBodyVertex> &jolt_verts = soft_mp->GetVertices();
        const SoftBodyShadow &shadow                      = world.soft_shadows[D];

        if ((shadow.vertices == nullptr) || shadow.num_vertices != jolt_verts.size()) [[unlikely]] {
            return;
        }

        auto *const CULV_RESTRICT dst_verts =
            reinterpret_cast<PosStride *const CULV_RESTRICT>(shadow.vertices);

        const SizeType num_v = shadow.num_vertices;
        CULV_UNROLL_LOOP(8)
        for (SizeType v = 0; v < num_v; ++v) {
            if (v + 8 < num_v) {
                CPH::Prefetch<CPH::AccessType::Write>(&dst_verts[v + 8]);
            }

            const JPH::Vec3 local_pos(jolt_verts[v].mPosition);

            if constexpr (double_precision) {
                const JPH::RVec3 world_pos = JPH::RVec3(rotation * local_pos) + translation;

                [[clang::always_inline]] world_pos.StoreDouble3(
                    reinterpret_cast<PosPointerType>(&dst_verts[v]));
                dst_verts[v].w = 0.0;
            } else {
                const JPH::Vec3 world_pos = (rotation * local_pos) + JPH::Vec3(translation);

                [[clang::always_inline]] JPH::Vec4(world_pos, 0.0F)
                    .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&dst_verts[v]));
            }
        }
    }
}

// =================================================================================================
// UNIFIED BATCH PROCESSOR
// Routes to either fully unrolled loops (FixedCount > 0) or prefetched remainder loops.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline, gnu::hot, gnu::flatten]]
inline auto ProcessBatch(const WorldDataCreateInfo world,
                         RestrictSpan<const CPH::SyncWorkItem> items) noexcept -> void {

    const SizeType count = items.size();
    [[assume(count > 0), assume(count <= BATCH_SIZE)]];

    CULV_UNROLL_LOOP(4)
    for (Unsigned32 j = 0; j < count; j++) {
        if (j + 2 < count) {
            const Unsigned32 next_idx = items[j + 2].dense_idx;
            CPH::Prefetch<CPH::AccessType::Write>(&world.shadow_pos[next_idx]);
            CPH::Prefetch<CPH::AccessType::Write>(&world.shadow_rot[next_idx]);
        }

        ProcessItem<TType>(items[j].dense_idx, items[j].body, world);
    }
}

// =================================================================================================
// UNIFIED SYNC PASS EXECUTOR
// Deduplicates the logic for iterating over Rigid and Soft bodies.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline, gnu::flatten, gnu::nonnull(2)]] inline auto
ExecuteSyncPass(const Unsigned32 active_count, const JPH::PhysicsSystem *const CULV_RESTRICT system,
                MappingDataCreateInfo map, const WorldDataCreateInfo world) noexcept -> void {
    if (active_count == 0) {
        return;
    }

    const JPH::BodyID *const CULV_RESTRICT active_ids = system->GetActiveBodiesUnsafe(TType);
    if (active_ids == nullptr) {
        [[unlikely]] return;
    }

    const auto *const CULV_RESTRICT lock_iface = &system->GetBodyLockInterfaceNoLock();

    alignas(MEMORY_ALIGNMENT_SIZE) std::array<SyncWorkItem, BATCH_SIZE> worklist;

    Unsigned32 work_ptr = 0;

    for (Unsigned32 i = 0; i < active_count; i++) {
        const Unsigned32 raw_jolt_id = active_ids[i].GetIndexAndSequenceNumber();
        const Unsigned32 j_idx       = raw_jolt_id & JPH::BodyID::cMaxBodyIndex;

        const auto *CULV_RESTRICT b =
            static_cast<const JPH::Body * CULV_RESTRICT>(map.body_ptrs[j_idx]);

        [[clang::always_inline]] if (b == nullptr || b->GetID().GetIndexAndSequenceNumber() !=
                                                         raw_jolt_id) [[unlikely]] {
            [[clang::always_inline]] b = lock_iface->TryGetBody(JPH::BodyID(raw_jolt_id));
            map.body_ptrs[j_idx]       = b;
        }
        // Verified by flush_commands_internal.
        [[assume(b != nullptr)]];

        const Unsigned64 handle    = b->GetUserData();
        const auto slot            = static_cast<const Unsigned32>(handle & HANDLE_INDEX_MASK);
        const auto gen             = static_cast<const Unsigned32>(handle >> HANDLE_INDEX_BITS);
        const Unsigned32 safe_slot = (slot < map.slot_capacity) ? slot : 0;

        const Unsigned8 state        = map.slot_states[safe_slot].load(std::memory_order_relaxed);
        const Unsigned32 current_gen = map.generations[safe_slot].load(std::memory_order_relaxed);

        // --- BRANCHLESS VALIDATION ---
        [[assume(state < SLOT_COUNT)]];
        const Unsigned32 state_bad = [state]() -> Unsigned32 {
            if constexpr (TType == JPH::EBodyType::RigidBody) {
                // Create a mask of the bits we want (bit 1 and bit 2)
                // (1 << 2) | (1 << 4) is not correct because the values are 2 and 4, not bit
                // positions. We use the values directly:
                constexpr Unsigned8 mask = (1 << SLOT_ALIVE) | (1 << SLOT_CHARACTER);

                // Use the state as a shift amount to index into our 'valid' bitmask
                // If state is 2 or 4, (mask >> state) & 1 will be 1.
                // We XOR with 1 to flip it: 1 (bad) if state is NOT in the mask.
                return ((mask >> state) & 1) ^ 1;
            } else {
                // For SoftBody, just check if state == 5
                return (state != SLOT_SOFT_BODY);
            }
        }();
        const Unsigned32 bad   = static_cast<const Unsigned32>(slot >= map.slot_capacity) |
                                 (current_gen ^ gen) | state_bad;
        const Unsigned32 d_idx = map.slot_to_dense[safe_slot];
        const auto is_valid    = static_cast<Unsigned32>(bad == 0);

        [[assume(work_ptr < BATCH_SIZE)]];
        worklist[work_ptr].body      = b;
        worklist[work_ptr].dense_idx = d_idx;
        work_ptr += is_valid;

        if (work_ptr == BATCH_SIZE) {
            // "Convert worklist to span, then take the first BATCH_SIZE items"
            ProcessBatch<TType>(world, CPH::RestrictSpan(worklist));
            work_ptr = 0;
        }
    }

    // Flush remainder
    if (work_ptr > 0) {
        // "Convert worklist to span, then take the first 'work_ptr' items"
        ProcessBatch<TType>(world, CPH::RestrictSpan(worklist).first(work_ptr));
    }
}

} // namespace CPH

// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================

extern "C" [[gnu::flatten, gnu::hot, gnu::nonnull(1)]] auto
culverin_sync_shadow_buffers(const PhysicsWorldObject *const CULV_RESTRICT self) noexcept -> void {
    using namespace JPH;
    using namespace CPH;
    const auto *const CULV_RESTRICT system = static_cast<const PhysicsSystem *const CULV_RESTRICT>(
        JPH_PhysicsSystem_GetPhysicsSystemInstance(self->system));

    const Unsigned32 active_rigid_count = system->GetNumActiveBodies(EBodyType::RigidBody);
    const Unsigned32 active_soft_count  = system->GetNumActiveBodies(EBodyType::SoftBody);

    if ((active_rigid_count == 0U) && (active_soft_count == 0U)) {
        [[unlikely]] return;
    }

    constexpr Unsigned64 MIN_CYCLES             = 0xFFFFFFFFFFFFFFFFULL;
    [[maybe_unused]] static CulvStat sync_stats = {
        .total_cycles = 0, .min_cycles = MIN_CYCLES, .max_cycles = 0, .count = 0};

    CULV_PROFILE_BEGIN(sync);

    const CPH::WorldDataCreateInfo world = {
        .shadow_pos = std::assume_aligned<sizeof(PosStride)>(
            reinterpret_cast<PosStride *const CULV_RESTRICT>(self->positions)),

        .shadow_ppos = std::assume_aligned<sizeof(PosStride)>(
            reinterpret_cast<PosStride *const CULV_RESTRICT>(self->prev_positions)),

        .shadow_rot = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CULV_RESTRICT>(self->rotations)),

        .shadow_prot = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CULV_RESTRICT>(self->prev_rotations)),

        .shadow_lvel = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CULV_RESTRICT>(self->linear_velocities)),

        .shadow_avel = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CULV_RESTRICT>(self->angular_velocities)),

        .soft_shadows = self->soft_shadows};

    const CPH::MappingDataCreateInfo mapping_data = {
        .body_ptrs     = self->jolt_body_ptrs,
        .slot_states   = self->slot_states,
        .generations   = self->generations,
        .slot_capacity = self->slot_capacity,
        .slot_to_dense = self->slot_to_dense,
    };

    ExecuteSyncPass<EBodyType::RigidBody>(active_rigid_count, system, mapping_data, world);

    if (self->soft_shadows != nullptr) {
        ExecuteSyncPass<EBodyType::SoftBody>(active_soft_count, system, mapping_data, world);
    }

    CULV_PROFILE_ACCUMULATE(sync, &sync_stats);
#ifdef CULVERIN_PROFILE_SYNC
    if (sync_stats.count >= 50) {
        fprintf(stderr, "[culverin] Sync Stat Avg: %" PRIu64 " | Max: %" PRIu64 "\n",
                sync_stats.total_cycles / sync_stats.count, sync_stats.max_cycles);
        sync_stats = (CulvStat){0, MIN_CYCLES, 0, 0};
    }
#endif
}
