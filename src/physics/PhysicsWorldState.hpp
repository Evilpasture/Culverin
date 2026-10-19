#pragma once

// Native physics-world state. Contains no Python object fields; the
// CPython binding owns interpreter concerns and placement-constructs this
// struct inside the PhysicsWorld Python object.

#include "core/Buffer.hpp"
#include "core/Compiler.hpp"
#include "core/Threading.hpp"
#include "core/Types.hpp"
#include "debug/DebugRender.hpp"
#include "jolt/JoltConfig.hpp"
#include "jolt/JoltUtils.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/SlotState.hpp"
#include "soft_bodies/SoftBodyCore.hpp"
#include <joltc.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>


namespace CPH {

struct WorldSettings {
    int max_bodies;
    int max_pairs;
    int max_contact_constraints;
    int temp_allocator_size;
    int max_physics_jobs;
    int max_physics_barriers;
    int num_threads;
    float penetration_slop;
};

struct GravityVector {
    float gx;
    float gy;
    float gz;
};

enum PhysicsWorldStatus : int {
    PHYSICS_WORLD_STATUS_OK                = 0,
    PHYSICS_WORLD_STATUS_OUT_OF_MEMORY     = -1,
    PHYSICS_WORLD_STATUS_LIMIT_REACHED     = -2,
    PHYSICS_WORLD_STATUS_BUFFER_EXPORTED   = -3,
    PHYSICS_WORLD_STATUS_INVALID_ARGUMENT  = -4,
};

struct MaterialData {
    std::uint32_t id;
    float friction;
    float restitution;
};

// All per-body storage that resizes together. Dense arrays (positions …
// soft_shadows) are indexed by dense index [0, count); the maps
// (slot_to_dense, generations, slot_states, free_slots) are indexed by
// sparse slot. capacity == slot_capacity in PhysicsWorld, so every
// array here has the same size.
//
// The SoA layout is a public contract (Python buffer protocol) and the
// layout ShadowSync streams with SIMD; only ownership changed in the
// C++23 rewrite — each array is a Buffer, not a raw malloc.

// Default collision category/mask: a body collides with everything until
// told otherwise.
inline constexpr std::uint32_t kAllLayerBits = 0xFFFFU;

struct BodyBuffers {
    Buffer<PosStride, kAvxAlignment> positions;
    Buffer<PosStride, kAvxAlignment> prev_positions;
    Buffer<AuxStride, kAvxAlignment> rotations;
    Buffer<AuxStride, kAvxAlignment> prev_rotations;
    Buffer<AuxStride, kAvxAlignment> linear_velocities;
    Buffer<AuxStride, kAvxAlignment> angular_velocities;
    Buffer<JPH_BodyID> body_ids;
    Buffer<std::uint64_t> user_data;
    Buffer<std::uint32_t> categories;
    Buffer<std::uint32_t> masks;
    Buffer<std::uint32_t> material_ids;
    Buffer<SoftBodyShadow> soft_shadows;

    Buffer<std::uint32_t> slot_to_dense;
    Buffer<std::uint32_t> dense_to_slot;
    Buffer<std::uint32_t> free_slots;
    Buffer<std::atomic<std::uint32_t>> generations;
    Buffer<std::atomic<std::uint8_t>> slot_states;

    // All-or-nothing: on failure *this is unchanged and false is returned.
    // With for_overwrite, elements are default-constructed only (see
    // Buffer::allocate_for_overwrite) — used by resize, which copies the
    // live ranges and initializes the newly exposed ones itself.
    [[nodiscard]] bool allocate(std::size_t capacity, bool for_overwrite = false) noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return positions.size(); }
};

struct ConstraintSlot {
    JPH_Constraint *constraint = nullptr;
    std::uint32_t generation   = 1;
    std::atomic<std::uint8_t> state{SLOT_EMPTY};
    int type = 0;
};

// Cache isolation via explicit padding: PhysicsWorld lives inside a
// CPython-allocated object, whose allocator does not respect 64-byte
// alignment of the containing struct.
#define CPH_CACHE_LINE_SPACER std::uint8_t CPH_CONCAT(_unused_pad_, __LINE__)[64]

struct PhysicsWorld {
    PhysicsWorld();
    ~PhysicsWorld();

    PhysicsWorld(const PhysicsWorld &)            = delete;
    PhysicsWorld &operator=(const PhysicsWorld &) = delete;
    PhysicsWorld(PhysicsWorld &&)                 = delete;
    PhysicsWorld &operator=(PhysicsWorld &&)      = delete;

    // --- Jolt handles (owned; destroyed by ~PhysicsWorld) ---
    JPH_PhysicsSystem *system                                 = nullptr;
    JPH_BodyInterface *body_interface                         = nullptr;
    JPH_JobSystem *job_system                                 = nullptr;
    JPH_BroadPhaseLayerInterface *bp_interface                = nullptr;
    JPH_ObjectLayerPairFilter *pair_filter                    = nullptr;
    JPH_ObjectVsBroadPhaseLayerFilter *bp_filter              = nullptr;
    JPH_ContactListener *contact_listener                     = nullptr;
    JPH_TempAllocator *temp_allocator                         = nullptr;
    JPH_CharacterVsCharacterCollision *char_vs_char_manager   = nullptr;
    std::uint32_t max_jolt_bodies                             = 0;
    std::size_t contact_max_capacity                          = 0;

    CPH_CACHE_LINE_SPACER;

    // --- Hot simulation state ---
    double time = 0.0;
    std::atomic<std::size_t> count{0};
    std::size_t capacity      = 0; // == slot_capacity == bodies.size()
    std::size_t slot_capacity = 0;
    std::atomic<std::size_t> free_count{0};

    BodyBuffers bodies;
    // Storage retired by the most recent resize. Released at the next
    // resize (or teardown) so a lock-free reader of the previous frame's
    // shadow buffers can never observe freed memory.
    BodyBuffers retired_bodies;

    CPH_CACHE_LINE_SPACER;

    // --- Synchronization (see core/Threading.hpp for the hierarchy) ---
    ShadowSync step_sync;

    CPH_CACHE_LINE_SPACER;

    std::mutex shadow_lock;
    std::mutex jph_trampoline_lock;

    Buffer<PhysicsCommand> command_queue;
    Buffer<PhysicsCommand> command_queue_spare;
    std::size_t command_count = 0;

    CPH_CACHE_LINE_SPACER;
    std::atomic<bool> is_stepping{false};
    std::atomic<bool> step_requested{false};
    CPH_CACHE_LINE_SPACER;
    std::atomic<int> waiting_threads{0};

    CPH_CACHE_LINE_SPACER;
    std::atomic<int> active_queries{0};
    std::atomic<int> view_export_count{0};
    bool needs_optimization = false;

    CPH_CACHE_LINE_SPACER;

    // --- Contacts, materials, shapes ---
    Buffer<ContactEvent> contact_buffer;
    std::atomic<std::size_t> contact_atomic_idx{0};
    std::size_t contact_count = 0;

    Buffer<MaterialData> materials;
    std::size_t material_count = 0;

    Buffer<ShapeEntry> shape_cache;
    std::size_t shape_cache_count = 0;

    // Indexed by Jolt BodyID index (size max_jolt_bodies + 1).
    Buffer<const void *> jolt_body_ptrs;
    Buffer<std::atomic<BodyHandle>> id_to_handle_map;

    CPH_CACHE_LINE_SPACER;

    // --- Constraints ---
    Buffer<ConstraintSlot> constraints;
    Buffer<std::uint32_t> free_constraint_slots;
    std::size_t free_constraint_count = 0;

    CPH_CACHE_LINE_SPACER;

    // --- Debug output ---
    JPH_DebugRenderer *debug_renderer = nullptr;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;
};

// --- Native handle helpers ---

[[nodiscard]] inline BodyHandle make_handle(std::uint32_t slot, std::uint32_t gen) noexcept {
    return (static_cast<std::uint64_t>(gen) << kHandleIndexBits) | slot;
}

// Resolves a handle to its slot if the generation still matches.
[[nodiscard]] inline bool unpack_handle(PhysicsWorld *world, BodyHandle handle,
                                        std::uint32_t *slot) noexcept {
    *slot              = static_cast<std::uint32_t>(handle & kHandleIndexMask);
    const auto gen     = static_cast<std::uint32_t>(handle >> kHandleIndexBits);
    if (CPH_UNLIKELY(*slot >= world->slot_capacity)) {
        return false;
    }
    return world->bodies.generations[*slot].load(std::memory_order::acquire) == gen;
}

// --- Shared step/query arbitration (defined in PhysicsWorld.cpp) ---

// Acquires shadow_lock, parking on step_sync while a step is in progress.
// Returns the held lock; structured bindings / named locals only — never
// call this as a temporary.
[[nodiscard]] std::unique_lock<std::mutex>
LockShadowNotStepping(PhysicsWorld *world);

// With shadow_lock held (via `shadow`), waits until no query is active.
void WaitUntilNotQuerying(PhysicsWorld *world,
                              std::unique_lock<std::mutex> &shadow);

// Wakes every thread parked on step_sync.cond.
// Append a command; capacity must already be ensured via
// EnsureCommandCapacity. Caller holds shadow_lock.
template <typename Payload>
inline void EnqueueCommand(PhysicsWorld *world, std::uint32_t slot, Payload &&payload) {
    world->command_queue[world->command_count++] =
        PhysicsCommand{slot, std::forward<Payload>(payload)};
}

inline void NotifyStepSync(PhysicsWorld *world) {
    const std::scoped_lock lock(world->step_sync.mutex);
    world->step_sync.cond.notify_all();
}

} // namespace CPH
