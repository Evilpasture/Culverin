#pragma once

// Native physics-world state. Contains no Python object fields; the
// CPython binding owns interpreter concerns and placement-constructs this
// struct inside the PhysicsWorld Python object.

#include "core/Buffer.hpp"
#include "core/Compiler.hpp"
#include "core/Error.hpp"
#include "core/Threading.hpp"
#include "core/Types.hpp"
#include "debug/DebugRender.hpp"
#include "jolt/Jolt.hpp"
#include "jolt/JoltConfig.hpp"
#include "jolt/JoltUtils.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/ContactListener.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/Query.hpp"
#include "physics/SlotState.hpp"
#include "soft_bodies/SoftBodyCore.hpp"
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Constraints/Constraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <joltc.h>


#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <span>


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

enum class WorldError : std::uint32_t {
    OutOfMemory     = 1,
    LimitReached    = 2,
    BufferExported  = 3,
    InvalidArgument = 4,
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
    // Owning reference to the Jolt constraint: PhysicsSystem::AddConstraint
    // takes its own, so this is released with Constraint::Release().
    JPH::Constraint *constraint = nullptr;
    std::uint32_t generation   = 1;
    std::atomic<std::uint8_t> state{std::to_underlying(SlotState::Empty)};
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

    // --- Lifecycle & stepping (PhysicsWorld.cpp) ---

    // Allocates body/constraint/contact storage for `max_bodies`, with room
    // for `initial_count` bodies available immediately.
    [[nodiscard]] std::expected<void, Error> AllocateBuffers(int max_bodies,
                                                             std::size_t initial_count);
    void InitializeFreeSlots(std::size_t start_idx);
    [[nodiscard]] bool InitJolt(WorldSettings settings, GravityVector gravity);
    [[nodiscard]] std::expected<void, Error> LoadBakedScene(const BakedSceneData &scene);
    // Slot and resize mutations require shadow_lock and a non-stepping world.
    [[nodiscard]] std::expected<void, Error> Resize(std::size_t new_capacity);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CommitCreate(JPH::BodyCreationSettings &settings, std::uint32_t slot_state);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CommitCreateSoft(JPH::SoftBodyCreationSettings &settings, std::uint32_t slot_state);
    // Caller holds shadow_lock. Swaps the active command queue with the spare
    // and returns the captured commands (still owned by the spare buffer;
    // flush them before the next detach).
    [[nodiscard]] std::span<PhysicsCommand> DetachCommands();
    // Called with the GIL released after the binding marked the world
    // stepping and detached the commands; flushes commands, steps Jolt,
    // syncs shadows.
    void StepNative(float dt, std::span<PhysicsCommand> commands);
    // Caller holds shadow_lock; finalizes step counters/flags and wakes
    // waiters. Returns the live body count.
    std::size_t FinalizeStep(float dt);

    // --- Command queue (CommandBuffer.cpp) ---

    // Swap-remove `slot` from the dense body arrays and recycle it.
    // Caller holds shadow_lock.
    void RemoveBodySlot(std::uint32_t slot);
    // Queue growth; caller holds shadow_lock and the world is not stepping.
    [[nodiscard]] bool EnsureCommandCapacity();
    [[nodiscard]] bool EnsureCommandBulkCapacity(std::size_t batch_size);
    // Executes and clears each command. Caller holds jph_trampoline_lock.
    void FlushCommands(std::span<PhysicsCommand> queue);
    // Drops pending commands, destroying payloads they own.
    void ClearCommandQueue();
    // Append a command; capacity must already be ensured via
    // EnsureCommandCapacity. Caller holds shadow_lock.
    template <typename Payload>
    void EnqueueCommand(std::uint32_t slot, Payload &&payload) {
        command_queue[command_count++] = PhysicsCommand{slot, std::forward<Payload>(payload)};
    }

    // --- Step/query arbitration (PhysicsWorld.cpp) ---

    // Acquires shadow_lock, parking on step_sync while a step is in
    // progress. Returns the held lock; named locals only — never call
    // this as a temporary.
    [[nodiscard]] std::unique_lock<std::mutex> LockShadowNotStepping();
    // With shadow_lock held (via `shadow`), waits until no query is active.
    void WaitUntilNotQuerying(std::unique_lock<std::mutex> &shadow);
    // Wakes every thread parked on step_sync.cond.
    void NotifyStepSync() {
        const std::scoped_lock lock(step_sync.mutex);
        step_sync.cond.notify_all();
    }
    // Resolves a handle to its slot if the generation still matches.
    [[nodiscard]] std::optional<std::uint32_t> UnpackHandle(BodyHandle handle) const noexcept {
        const auto slot = static_cast<std::uint32_t>(handle & kHandleIndexMask);
        const auto gen  = static_cast<std::uint32_t>(handle >> kHandleIndexBits);
        if (CPH_UNLIKELY(slot >= slot_capacity)) {
            return std::nullopt;
        }
        if (bodies.generations[slot].load(std::memory_order::acquire) != gen) {
            return std::nullopt;
        }
        return slot;
    }

    // --- Shape cache (InternalQuery.cpp) ---

    // REQUIRES: shadow_lock held (documented at the definition).
    // Returns an empty ref on failure.
    [[nodiscard]] JPH::ShapeRefC
    FindOrCreateShapeLocked(int type, std::span<const float, 4> params);
    // Locking wrapper around FindOrCreateShapeLocked.
    [[nodiscard]] JPH::ShapeRefC FindOrCreateShape(int type, std::span<const float, 4> params);

    // --- Body creation (PhysicsWorldCreation.cpp) ---

    [[nodiscard]] ResolvedMaterial ResolveMaterial(std::uint32_t material_id, float input_friction,
                                                   float input_restitution);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateBody(const PhysicsWorldBodyCreateRequest &request);
    // `out_handles` must have the same length as `positions`/`sizes`;
    // failed entries are set to 0.
    [[nodiscard]] std::expected<void, Error>
    CreateBodiesBatch(std::span<const PosStride> positions, std::span<const ShapeParams> sizes,
                      int shape_type, int motion_type, std::span<BodyHandle> out_handles);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateSoftBody(const PhysicsWorldSoftBodyCreateRequest &request);
    [[nodiscard]] std::expected<RagdollInstance, Error>
    CreateRagdollInstance(JPH::RagdollSettings *settings, JPH_RVec3 position, JPH_Quat rotation,
                            std::uint64_t user_data);
    // `body_slots` must have room for the ragdoll's body count.
    // Returns the world body count after registration.
    [[nodiscard]] std::expected<std::size_t, Error>
    RegisterRagdollBodies(JPH::Ragdoll *ragdoll, std::span<std::uint32_t> body_slots,
                          std::uint64_t user_data, std::uint32_t category, std::uint32_t mask,
                          std::uint32_t material_id);
    void DestroyRagdoll(JPH::Ragdoll *ragdoll);
    // `points` is flat XYZ triples.
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateConvexHull(std::span<const float> points, const PhysicsWorldBodyCreateRequest &request);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateCompoundBody(std::span<const PhysicsWorldCompoundPart> parts,
                       const PhysicsWorldBodyCreateRequest &request);
    // `vertices` is flat XYZ triples, `indices` flat triangle triples.
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateMeshBody(std::span<const float> vertices, std::span<const std::uint32_t> indices,
                   JPH_RVec3 position, JPH_Quat rotation, std::uint32_t category,
                   std::uint32_t mask, std::uint64_t user_data);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateHeightfieldBody(std::span<const float> heights, std::size_t grid_size, JPH_Vec3 scale,
                          JPH_RVec3 position, JPH_Quat rotation, float friction, float restitution,
                          std::uint32_t category, std::uint32_t mask, std::uint32_t material_id,
                          std::uint64_t user_data);

    // --- Body operations (PhysicsWorldOps.cpp) ---

    // Executes an engine-side body mutation: command staging, shadow-buffer
    // mirroring, and immediate Jolt application for impulse-style ops on
    // already-live bodies. Stale-handle errors are translated by the binding.
    [[nodiscard]] std::expected<void, Error>
    ApplyBodyOperation(const PhysicsBodyOperation &operation);
    [[nodiscard]] std::expected<void, Error> QueueBodyDestroy(BodyHandle handle);
    [[nodiscard]] std::expected<void, Error>
    QueueBodyDestroyBatch(std::span<const BodyHandle> handles);

    // --- Runtime operations (PhysicsWorldRuntime.cpp) ---

    [[nodiscard]] bool AttachContactListener();
    void SyncLoadedState(std::size_t saved_count);
    [[nodiscard]] std::expected<Buffer<ContactEvent>, Error> TakeContactEvents();
    [[nodiscard]] std::expected<Buffer<std::uint8_t>, Error> SaveSnapshot();
    // Returns the number of bodies restored.
    [[nodiscard]] std::expected<std::size_t, Error> LoadSnapshot(std::span<const std::uint8_t> data);
    [[nodiscard]] std::expected<void, Error> SetGravity(JPH::Vec3 gravity);
    [[nodiscard]] JPH::Vec3 GetGravity();
    [[nodiscard]] std::expected<PhysicsBodyStats, Error> GetBodyStats(BodyHandle handle);
    [[nodiscard]] std::expected<std::uint32_t, Error> GetSoftBodyVertexCount(BodyHandle handle);
    [[nodiscard]] std::expected<SoftBodyVertexPosition, Error>
    GetSoftBodyVertexPosition(BodyHandle handle, std::uint32_t vertex_index);
    [[nodiscard]] std::expected<Buffer<JPH_Vec3>, Error>
    GetSoftBodyLocalVertices(BodyHandle handle);
    // Returns whether the body is submerged; a stale handle reports
    // "not submerged" rather than an error (legacy behavior).
    [[nodiscard]] std::expected<bool, Error>
    ApplyBuoyancy(BodyHandle handle, JPH_Real surface_y, float buoyancy, float linear_drag,
                  float angular_drag, JPH::Vec3 fluid_velocity, float dt);
    [[nodiscard]] std::expected<void, Error>
    ApplyBuoyancyBatch(std::span<const BodyHandle> handles, JPH_Real surface_y, float buoyancy,
                       float linear_drag, float angular_drag, JPH::Vec3 fluid_velocity, float dt);
    [[nodiscard]] std::expected<MotionType, Error> GetMotionType(BodyHandle handle);
    [[nodiscard]] std::expected<std::uint64_t, Error> GetUserData(BodyHandle handle);
    [[nodiscard]] std::expected<std::uint32_t, Error> GetIndex(BodyHandle handle);
    [[nodiscard]] bool IsAlive(BodyHandle handle);
    [[nodiscard]] std::expected<bool, Error> IsActive(BodyHandle handle);
    // Fills `indices` (resized as needed) and returns how many entries
    // are valid.
    [[nodiscard]] std::expected<std::size_t, Error>
    GetActiveIndices(Buffer<std::uint32_t> &indices);
    // Interpolated render state: 7 floats per body (position + quaternion).
    [[nodiscard]] std::expected<Buffer<float>, Error> GetRenderState(float alpha);
    [[nodiscard]] std::expected<void, Error>
    SetCollisionFilter(BodyHandle handle, std::uint32_t category, std::uint32_t mask);
    [[nodiscard]] std::expected<void, Error>
    RegisterMaterial(std::uint32_t id, float friction, float restitution);
    [[nodiscard]] std::expected<DebugData, Error>
    BuildDebugData(bool draw_shapes, bool draw_constraints, bool draw_bounding_box,
                   bool draw_centers, bool wireframe);

    // --- Spatial queries (Query.cpp) ---

    // Balances the active_queries registration performed by the query
    // entry points (also used by native ops that register as queries).
    void QueryEnd();
    [[nodiscard]] std::expected<void, Error>
    OverlapSphere(JPH::RVec3 center, float radius, PhysicsQueryHitBuffer &hits);
    [[nodiscard]] std::expected<void, Error>
    OverlapAabb(JPH::RVec3 min, JPH::RVec3 max, PhysicsQueryHitBuffer &hits);
    [[nodiscard]] std::expected<PhysicsRaycastHit, Error>
    Raycast(JPH::RVec3 origin, JPH::Vec3 direction, float max_distance, BodyHandle ignore_handle);
    // `starts`/`directions` are flat XYZ triples, one per result.
    [[nodiscard]] std::expected<void, Error>
    RaycastBatch(std::span<const float> starts, std::span<const float> directions,
                 float max_distance, std::span<RayCastBatchResult> results);
    [[nodiscard]] std::expected<PhysicsShapecastHit, Error>
    Shapecast(int shape_type, JPH::RVec3 position, JPH::Quat rotation, JPH::Vec3 sweep_direction,
              std::span<const float, 4> size, BodyHandle ignore_handle);

    // --- Ragdoll bookkeeping (Ragdoll.cpp) ---

    void CopyRagdollBodyHandles(std::span<const std::uint32_t> body_slots,
                                 std::span<PhysicsRagdollBodyHandleInfo> out_handles);
    [[nodiscard]] bool CopyRagdollDebugInfo(JPH::Ragdoll &ragdoll,
                                            std::span<PhysicsRagdollBodyDebugInfo> out_info);

    // JoltC view of physics_system.GetBodyInterface(), retained until the
    // body-operation subsystems move to the direct Jolt API in A4/A5.
    JPH_BodyInterface *body_interface = nullptr;
    // Native listener, constructed with `this` (A3). Jolt only stores the
    // pointer, so it must outlive physics_system — it does: declared here,
    // physics_system is declared last and therefore destroyed first.
    WorldContactListener contact_listener;
    std::uint32_t max_jolt_bodies    = 0;
    std::size_t contact_max_capacity = 0;

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

    // Acquisition order is jph_trampoline_lock -> shadow_lock -> step_sync
    // (step_sync lives in the block above). See core/Threading.hpp.
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
    // Heap-owned, not a value member: JPH::DebugRendererSimple stores an
    // RVec3 (32-byte aligned in double precision), which would force
    // PhysicsWorld itself to alignof 32 — and the world lives inside a
    // CPython-allocated object that only guarantees max_align_t. Same reason
    // the job system is heap-owned. One allocation per world, exactly like
    // the JoltC renderer it replaces.
    std::unique_ptr<WorldDebugRenderer> debug_renderer;
    DebugBuffer debug_lines;
    DebugBuffer debug_triangles;

    // --- Direct Jolt core ---
    // Declared last so member destruction tears physics_system down before
    // the body, shape, and constraint storage declared above. The thread
    // pool is heap-owned: JobSystemThreadPool requires 64-byte alignment,
    // which CPython allocation of the enclosing world object cannot provide.
    JPH::CharacterVsCharacterCollisionSimple character_vs_character;
    std::unique_ptr<JPH::TempAllocator> temp_allocator;
    std::unique_ptr<JPH::JobSystemThreadPool> job_system;
    JPH::PhysicsSystem physics_system;
    bool jolt_initialized = false;
};

// --- Native handle helpers ---

[[nodiscard]] inline BodyHandle make_handle(std::uint32_t slot, std::uint32_t gen) noexcept {
    return (static_cast<std::uint64_t>(gen) << kHandleIndexBits) | slot;
}

} // namespace CPH
