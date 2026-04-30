#include "culverin_shadow_sync.h"
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
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

namespace {

// Locality hints for clarity
enum class CacheLevel : uint8_t { L1 = 3, L2 = 2, L3 = 1, stream = 0 };
enum class AccessType : uint8_t { Read = 0, Write = 1 };

template <AccessType Access = AccessType::Read, CacheLevel Level = CacheLevel::L1>
[[gnu::always_inline]] inline void prefetch(const void *addr) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(addr, static_cast<int>(Access), static_cast<int>(Level));
#elif defined(_MSC_VER)
    // MSVC doesn't have a direct 1:1 for __builtin_prefetch's rw param
    if constexpr (Access == AccessType::Write) {
        // PREFETCHW support is CPU-specific; T0 is the standard fallback
        _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
    } else {
        if constexpr (Level == CacheLevel::L1)
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
        else if constexpr (Level == CacheLevel::L2)
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T1);
        else
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_NTA);
    }
#endif
}

template <size_t N, typename F> constexpr void unroll(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        (f(std::integral_constant<size_t, Is>{}), ...);
    }(std::make_index_sequence<N>{});
}

template <size_t N, typename F> constexpr void repeat(F &&f) {
    [&f]<size_t... Is>(std::index_sequence<Is...>) -> auto {
        ((static_cast<void>(Is), f()), ...);
    }(std::make_index_sequence<N>{});
}

constexpr int BATCH_SIZE = 128;
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

struct MappingDataCreateInfo {
    const void *CULV_RESTRICT *const CULV_RESTRICT body_ptrs;
    const std::atomic<uint8_t> *const CULV_RESTRICT slot_states;
    const std::atomic<uint32_t> *const CULV_RESTRICT generations;
    const size_t slot_capacity;
    const uint32_t *const CULV_RESTRICT slot_to_dense;
};

static_assert(std::is_trivially_copyable<WorldDataCreateInfo>());
static_assert(std::is_trivial<WorldDataCreateInfo>());

#ifdef JPH_DOUBLE_PRECISION
inline constexpr bool double_precision = true;
#else
inline constexpr bool double_precision = false;
#endif

using pos_ptr_t = JPH::Double3 *const CULV_RESTRICT;
using aux_ptr_t = JPH::Float4 *const CULV_RESTRICT;

// =================================================================================================
// UNIFIED ITEM PROCESSOR
// Automatically eliminates dead code branches at compile time via `if constexpr`.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline, gnu::nonnull(2)]] inline auto
process_item(const uint32_t D, const JPH::Body *const CULV_RESTRICT b,
             const WorldDataCreateInfo world) noexcept -> void {

    // 1. Snapshot previous state
    world.shadow_ppos[D] = world.shadow_pos[D];
    world.shadow_prot[D] = world.shadow_rot[D];

    // 2. Write Current COM
    auto *const target = &world.shadow_pos[D];

    if constexpr (double_precision) {
        [[clang::always_inline]] b->GetCenterOfMassPosition().StoreDouble3(
            reinterpret_cast<pos_ptr_t>(target));
        target->w = 0.0;
    } else {
        [[clang::always_inline]] JPH::Vec4(JPH::Vec3(b->GetCenterOfMassPosition()), 0.0F)
            .StoreFloat4(reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(target));
    }

    // 3. Write Current Rotation
    [[clang::always_inline]] b->GetRotation().GetXYZW().StoreFloat4(
        reinterpret_cast<aux_ptr_t>(&world.shadow_rot[D]));

    // 4. Type-Specific Sub-Data
    if constexpr (TType == JPH::EBodyType::RigidBody) {
        [[clang::always_inline]] JPH::Vec4(b->GetLinearVelocity(), 0.0F)
            .StoreFloat4(reinterpret_cast<aux_ptr_t>(&world.shadow_lvel[D]));
        [[clang::always_inline]] JPH::Vec4(b->GetAngularVelocity(), 0.0F)
            .StoreFloat4(reinterpret_cast<aux_ptr_t>(&world.shadow_avel[D]));
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
                    prefetch<AccessType::Write>(&dst_verts[v + 8]);
                }

                const JPH::Vec3 local_pos(jolt_verts[v].mPosition);

                if constexpr (double_precision) {
                    const JPH::RVec3 world_pos = JPH::RVec3(rotation * local_pos) + translation;

                    [[clang::always_inline]] world_pos.StoreDouble3(
                        reinterpret_cast<pos_ptr_t>(&dst_verts[v]));
                    dst_verts[v].w = 0.0;
                } else {
                    const JPH::Vec3 world_pos = (rotation * local_pos) + JPH::Vec3(translation);

                    [[clang::always_inline]] JPH::Vec4(world_pos, 0.0F)
                        .StoreFloat4(
                            reinterpret_cast<JPH::Float4 *const CULV_RESTRICT>(&dst_verts[v]));
                }
            }
        }
    }
}

// =================================================================================================
// UNIFIED BATCH PROCESSOR
// Routes to either fully unrolled loops (FixedCount > 0) or prefetched remainder loops.
// =================================================================================================
template <JPH::EBodyType TType, uint32_t FixedCount = 0>
[[gnu::always_inline, gnu::hot, gnu::flatten, gnu::nonnull(2)]] inline auto
process_batch(const WorldDataCreateInfo world, const SyncWorkItem *const CULV_RESTRICT worklist,
              const uint32_t dynamic_count = 0) noexcept -> void {

    if constexpr (FixedCount > 0) {
        unroll<FixedCount>([&](auto j) -> auto {
            process_item<TType>(worklist[j].dense_idx, worklist[j].body, world);
        });
    } else {
        [[assume(dynamic_count > 0)]];

        CULV_UNROLL_LOOP(4)
        for (uint32_t j = 0; j < dynamic_count; j++) {
            if (j + 2 < dynamic_count) {
                prefetch<AccessType::Write>(&world.shadow_pos[worklist[j + 2].dense_idx]);
                prefetch<AccessType::Write>(&world.shadow_rot[worklist[j + 2].dense_idx]);
            }
            process_item<TType>(worklist[j].dense_idx, worklist[j].body, world);
        }
    }
}

template <JPH::EBodyType TType, uint32_t FixedCount>
inline void process_batch(const WorldDataCreateInfo world,
                          const std::array<SyncWorkItem, FixedCount> &arr) noexcept {
    process_batch<TType, FixedCount>(world, arr.data());
}

// =================================================================================================
// UNIFIED SYNC PASS EXECUTOR
// Deduplicates the logic for iterating over Rigid and Soft bodies.
// =================================================================================================
template <JPH::EBodyType TType>
[[gnu::always_inline, gnu::flatten, gnu::nonnull(2)]] inline auto
execute_sync_pass(const uint32_t active_count, const JPH::PhysicsSystem *const CULV_RESTRICT system,
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

    uint32_t work_ptr = 0;

    for (uint32_t i = 0; i < active_count; i++) {
        const uint32_t raw_jolt_id = active_ids[i].GetIndexAndSequenceNumber();
        const uint32_t j_idx       = raw_jolt_id & JPH::BodyID::cMaxBodyIndex;

        const auto *CULV_RESTRICT b =
            static_cast<const JPH::Body * CULV_RESTRICT>(map.body_ptrs[j_idx]);

        [[clang::always_inline]] if (b == nullptr || b->GetID().GetIndexAndSequenceNumber() !=
                                                         raw_jolt_id) [[unlikely]] {
            [[clang::always_inline]] b = lock_iface->TryGetBody(JPH::BodyID(raw_jolt_id));
            map.body_ptrs[j_idx]       = b;
        }
        // Verified by flush_commands_internal.
        [[assume(b != nullptr)]];

        const uint64_t handle    = b->GetUserData();
        const auto slot          = static_cast<const uint32_t>(handle & HANDLE_INDEX_MASK);
        const auto gen           = static_cast<const uint32_t>(handle >> HANDLE_INDEX_BITS);
        const uint32_t safe_slot = (slot < map.slot_capacity) ? slot : 0;

        const uint8_t state        = map.slot_states[safe_slot].load(std::memory_order_acquire);
        const uint32_t current_gen = map.generations[safe_slot].load(std::memory_order_acquire);

        // --- BRANCHLESS VALIDATION ---
        const uint32_t state_bad = [state]() -> uint32_t {
            if constexpr (TType == JPH::EBodyType::RigidBody) {
                // Create a mask of the bits we want (bit 1 and bit 2)
                // (1 << 2) | (1 << 4) is not correct because the values are 2 and 4, not bit
                // positions. We use the values directly:
                constexpr uint8_t mask = (1 << SLOT_ALIVE) | (1 << SLOT_CHARACTER);

                // Use the state as a shift amount to index into our 'valid' bitmask
                // If state is 2 or 4, (mask >> state) & 1 will be 1.
                // We XOR with 1 to flip it: 1 (bad) if state is NOT in the mask.
                return ((mask >> state) & 1) ^ 1;
            } else {
                // For SoftBody, just check if state == 5
                return (state != SLOT_SOFT_BODY);
            }
        }();
        const uint32_t bad =
            static_cast<const uint32_t>(slot >= map.slot_capacity) | (current_gen ^ gen) | state_bad;
        const uint32_t d_idx = map.slot_to_dense[safe_slot];
        const auto is_valid  = static_cast<uint32_t>(bad == 0);

        prefetch<AccessType::Write>(&world.shadow_pos[d_idx]);
        prefetch<AccessType::Write>(&world.shadow_rot[d_idx]);
        if constexpr (TType == JPH::EBodyType::SoftBody) {
            prefetch<AccessType::Read>(&world.soft_shadows[d_idx]);
        }

        [[assume(work_ptr < BATCH_SIZE)]];
        worklist[work_ptr].body      = b;
        worklist[work_ptr].dense_idx = d_idx;
        work_ptr += is_valid;

        if (work_ptr == BATCH_SIZE) {
            process_batch<TType, BATCH_SIZE>(world, worklist);
            work_ptr = 0;
        }
    }

    if (work_ptr > 0) {
        process_batch<TType, 0>(world, worklist.data(), work_ptr);
    }
}

} // namespace

// =================================================================================================
// MAIN SYNC ROUTINE
// =================================================================================================

extern "C" [[gnu::flatten, gnu::hot, gnu::nonnull(1)]] auto
culverin_sync_shadow_buffers(const PhysicsWorldObject *const CULV_RESTRICT self) noexcept -> void {
    using namespace JPH;
    const auto *const CULV_RESTRICT system = static_cast<const PhysicsSystem *const CULV_RESTRICT>(
        JPH_PhysicsSystem_GetPhysicsSystemInstance(self->system));

    const uint32_t active_rigid_count = system->GetNumActiveBodies(EBodyType::RigidBody);
    const uint32_t active_soft_count  = system->GetNumActiveBodies(EBodyType::SoftBody);

    if ((active_rigid_count == 0U) && (active_soft_count == 0U)) {
        [[unlikely]] return;
    }

    constexpr uint64_t MIN_CYCLES               = 0xFFFFFFFFFFFFFFFFULL;
    [[maybe_unused]] static CulvStat sync_stats = {
        .total_cycles = 0, .min_cycles = MIN_CYCLES, .max_cycles = 0, .count = 0};

    CULV_PROFILE_BEGIN(sync);

    const WorldDataCreateInfo world = {
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

    const MappingDataCreateInfo mapping_data = {
        .body_ptrs     = self->jolt_body_ptrs,
        .slot_states   = self->slot_states,
        .generations   = self->generations,
        .slot_capacity = self->slot_capacity,
        .slot_to_dense = self->slot_to_dense,
    };

    execute_sync_pass<EBodyType::RigidBody>(active_rigid_count, system, mapping_data, world);

    if (self->soft_shadows != nullptr) {
        execute_sync_pass<EBodyType::SoftBody>(active_soft_count, system, mapping_data, world);
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