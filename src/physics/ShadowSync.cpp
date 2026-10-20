#include "physics/ShadowSync.hpp"
#include "core/Compiler.hpp"
#include "core/Profiling.hpp"
#include "core/Types.hpp"
#include "cpp/Prefetch.hpp"
#include "cpp/Span.hpp"

// Include native Jolt headers for ultra-fast C++ bypass
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Body/Body.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodyVertex.h>
#include <array>
#include <atomic>
#include <memory> // IWYU pragma: keep


namespace CPH {

static_assert(sizeof(PosStride) == sizeof(JPH::Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(CPH::Float32) * 4);
static_assert(alignof(PosStride) == 32);
static_assert(alignof(AuxStride) == 16);
static_assert(sizeof(PosStride) % 32 == 0);


constexpr Unsigned32 BATCH_SIZE = 128;
struct SyncWorkItem {
    const JPH::Body *body;
    Unsigned32 dense_idx;
};

struct WorldDataCreateInfo {
    PosStride *const CPH_RESTRICT shadow_pos;
    PosStride *const CPH_RESTRICT shadow_ppos;
    AuxStride *const CPH_RESTRICT shadow_rot;
    AuxStride *const CPH_RESTRICT shadow_prot;
    AuxStride *const CPH_RESTRICT shadow_lvel;
    AuxStride *const CPH_RESTRICT shadow_avel;
    const SoftBodyShadow *const CPH_RESTRICT soft_shadows;
};

struct MappingDataCreateInfo {
    const void *CPH_RESTRICT *const CPH_RESTRICT body_ptrs;
    const std::atomic<Unsigned8> *const CPH_RESTRICT slot_states;
    const std::atomic<Unsigned32> *const CPH_RESTRICT generations;
    const SizeType slot_capacity;
    const Unsigned32 *const CPH_RESTRICT slot_to_dense;
};

static_assert(std::atomic<Unsigned8>::is_always_lock_free);
static_assert(std::atomic<Unsigned32>::is_always_lock_free);

#ifdef JPH_DOUBLE_PRECISION
inline constexpr bool double_precision = true;
#else
inline constexpr bool double_precision = false;
#endif

using PosPointerType = JPH::Double3 *const CPH_RESTRICT;
using AuxPointerType = JPH::Float4 *const CPH_RESTRICT;

// =================================================================================================
// UNIFIED ITEM PROCESSOR
// Automatically eliminates dead code branches at compile time via `if constexpr`.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline, gnu::nonnull(2)]] inline auto
ProcessItem(const Unsigned32 D, const JPH::Body *const CPH_RESTRICT b,
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
            .StoreFloat4(reinterpret_cast<JPH::Float4 *const CPH_RESTRICT>(target));
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
        const auto *const CPH_RESTRICT soft_mp =
            static_cast<const JPH::SoftBodyMotionProperties *const CPH_RESTRICT>(
                b->GetMotionProperties());
        const JPH::Array<JPH::SoftBodyVertex> &jolt_verts = soft_mp->GetVertices();
        const SoftBodyShadow &shadow                      = world.soft_shadows[D];

        if (!shadow.vertices || shadow.vertices.size() != jolt_verts.size()) [[unlikely]] {
            return;
        }

        auto *const CPH_RESTRICT dst_verts =
            shadow.vertices.data();

        const SizeType num_v = shadow.vertices.size();
        CPH_UNROLL_LOOP(8)
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
                    .StoreFloat4(reinterpret_cast<JPH::Float4 *const CPH_RESTRICT>(&dst_verts[v]));
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

    CPH_UNROLL_LOOP(4)
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
ExecuteSyncPass(const Unsigned32 active_count, const JPH::PhysicsSystem *const CPH_RESTRICT system,
                MappingDataCreateInfo map, const WorldDataCreateInfo world) noexcept -> void {
    if (active_count == 0) {
        return;
    }

    const auto *const CPH_RESTRICT active_ids = reinterpret_cast<const JPH_BodyID *const CPH_RESTRICT>(
        system->GetActiveBodiesUnsafe(TType));
    if (active_ids == nullptr) {
        [[unlikely]] return;
    }

    alignas(kMemoryAlignmentSize) std::array<SyncWorkItem, BATCH_SIZE> worklist;

    Unsigned32 work_ptr = 0;

    for (Unsigned32 i = 0; i < active_count; i++) {
        const JPH_BodyID raw_jolt_id = active_ids[i];
        const Unsigned32 j_idx       = raw_jolt_id & JPH::BodyID::cMaxBodyIndex;

        const auto *CPH_RESTRICT b =
            static_cast<const JPH::Body * CPH_RESTRICT>(map.body_ptrs[j_idx]);

        [[clang::always_inline]] if (b == nullptr || b->GetID().GetIndexAndSequenceNumber() !=
                                                         raw_jolt_id) [[unlikely]] {
            b = system->GetBodyLockInterface().TryGetBody(JPH::BodyID(raw_jolt_id));
            map.body_ptrs[j_idx] = b;
        }
        if (b == nullptr) [[unlikely]] {
            continue;
        }

        const Unsigned64 handle    = b->GetUserData();
        const auto slot            = static_cast<const Unsigned32>(handle & kHandleIndexMask);
        const auto gen             = static_cast<const Unsigned32>(handle >> kHandleIndexBits);
        const Unsigned32 safe_slot = (slot < map.slot_capacity) ? slot : 0;

        const Unsigned8 state        = map.slot_states[safe_slot].load(std::memory_order::relaxed);
        const Unsigned32 current_gen = map.generations[safe_slot].load(std::memory_order::relaxed);

        // --- BRANCHLESS VALIDATION ---
        const Unsigned32 state_bad = [state]() -> Unsigned32 {
            if constexpr (TType == JPH::EBodyType::RigidBody) {
                // Create a mask of the bits we want (bit 1 and bit 2)
                // (1 << 2) | (1 << 4) is not correct because the values are 2 and 4, not bit
                // positions. We use the values directly:
                constexpr Unsigned8 mask = (1 << std::to_underlying(SlotState::Alive)) | (1 << std::to_underlying(SlotState::Character));

                // Use the state as a shift amount to index into our 'valid' bitmask
                // If state is 2 or 4, (mask >> state) & 1 will be 1.
                // We XOR with 1 to flip it: 1 (bad) if state is NOT in the mask.
                return ((mask >> state) & 1) ^ 1;
            } else {
                // For SoftBody, just check if state == 5
                return (state != std::to_underlying(SlotState::SoftBody));
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


// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================

[[gnu::flatten, gnu::hot, gnu::nonnull(1)]] auto
SyncShadowBuffers(const PhysicsWorld *const CPH_RESTRICT world) noexcept -> void {
    using namespace CPH;
    const JPH::PhysicsSystem *const CPH_RESTRICT system = &world->physics_system;

    const Unsigned32 active_rigid_count =
        system->GetNumActiveBodies(JPH::EBodyType::RigidBody);
    const Unsigned32 active_soft_count = system->GetNumActiveBodies(JPH::EBodyType::SoftBody);

    if ((active_rigid_count == 0U) && (active_soft_count == 0U)) {
        [[unlikely]] return;
    }

    [[maybe_unused]] static CPH::Stat sync_stats;

    CPH_PROFILE_BEGIN(sync);

    const CPH::WorldDataCreateInfo world_data = {
        .shadow_pos = std::assume_aligned<sizeof(PosStride)>(
            reinterpret_cast<PosStride *const CPH_RESTRICT>(world->bodies.positions.data())),

        .shadow_ppos = std::assume_aligned<sizeof(PosStride)>(
            reinterpret_cast<PosStride *const CPH_RESTRICT>(world->bodies.prev_positions.data())),

        .shadow_rot = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CPH_RESTRICT>(world->bodies.rotations.data())),

        .shadow_prot = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CPH_RESTRICT>(world->bodies.prev_rotations.data())),

        .shadow_lvel = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CPH_RESTRICT>(world->bodies.linear_velocities.data())),

        .shadow_avel = std::assume_aligned<sizeof(AuxStride)>(
            reinterpret_cast<AuxStride *const CPH_RESTRICT>(world->bodies.angular_velocities.data())),

        .soft_shadows = world->bodies.soft_shadows.data()};

    const CPH::MappingDataCreateInfo mapping_data = {
        .body_ptrs     = world->jolt_body_ptrs.data(),
        .slot_states   = world->bodies.slot_states.data(),
        .generations   = world->bodies.generations.data(),
        .slot_capacity = world->slot_capacity,
        .slot_to_dense = world->bodies.slot_to_dense.data(),
    };

    ExecuteSyncPass<JPH::EBodyType::RigidBody>(active_rigid_count, system, mapping_data, world_data);

    if (world->bodies.soft_shadows) {
        ExecuteSyncPass<JPH::EBodyType::SoftBody>(active_soft_count, system, mapping_data, world_data);
    }

    CPH_PROFILE_ACCUMULATE(sync, &sync_stats);
#ifdef CULVERIN_PROFILE_SYNC
    if (sync_stats.count >= 50) {
        fprintf(stderr, "[culverin] Sync Stat Avg: %" PRIu64 " ns | Max: %" PRIu64 " ns\n",
                sync_stats.total_ns / sync_stats.count, sync_stats.max_ns);
        sync_stats = CPH::Stat{};
    }
#endif
}

} // namespace CPH
