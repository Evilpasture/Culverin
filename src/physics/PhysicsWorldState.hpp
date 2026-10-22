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
    int maxBodies;
    int maxPairs;
    int maxContactConstraints;
    int tempAllocatorSize;
    int maxPhysicsJobs;
    int maxPhysicsBarriers;
    int numThreads;
    float penetrationSlop;
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
    Buffer<PosStride, kAvxAlignment> prevPositions;
    Buffer<AuxStride, kAvxAlignment> rotations;
    Buffer<AuxStride, kAvxAlignment> prevRotations;
    Buffer<AuxStride, kAvxAlignment> linearVelocities;
    Buffer<AuxStride, kAvxAlignment> angularVelocities;
    Buffer<JPH::BodyID> bodyIds;
    Buffer<std::uint64_t> userData;
    Buffer<std::uint32_t> categories;
    Buffer<std::uint32_t> masks;
    Buffer<std::uint32_t> materialIds;
    Buffer<SoftBodyShadow> softShadows;

    Buffer<std::uint32_t> slotToDense;
    Buffer<std::uint32_t> denseToSlot;
    Buffer<std::uint32_t> freeSlots;
    Buffer<std::atomic<std::uint32_t>> generations;
    Buffer<std::atomic<std::uint8_t>> slotStates;

    // All-or-nothing: on failure *this is unchanged and false is returned.
    // With for_overwrite, elements are default-constructed only (see
    // Buffer::allocate_for_overwrite) — used by resize, which copies the
    // live ranges and initializes the newly exposed ones itself.
    [[nodiscard]] bool allocate(std::size_t capacity, bool for_overwrite = false) noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return positions.size(); }
};

struct ConstraintSlot {
    // Borrowed pointer to a Jolt constraint owned by the physics system:
    // AddConstraint takes the only reference and RemoveConstraint releases it.
    // Use the slot to look the constraint up, never to end its lifetime.
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
        commandQueue[commandCount++] = PhysicsCommand{slot, std::forward<Payload>(payload)};
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
        const std::scoped_lock lock(stepSync.mutex);
        stepSync.cond.notify_all();
    }
    // Resolves a typed handle to its slot, but only while the generation
    // still matches -- a handle to a recycled body resolves to nullopt
    // even though its index is in range.
    [[nodiscard]] std::optional<std::uint32_t> Resolve(BodyHandle handle) const noexcept {
        const auto slot = handle.Index();
        const auto gen  = handle.Generation();
        if (slot >= slotCapacity) [[unlikely]] {
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
    CreateRagdollInstance(JPH::RagdollSettings *settings, JPH::RVec3 position, JPH::Quat rotation,
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
                   JPH::RVec3 position, JPH::Quat rotation, std::uint32_t category,
                   std::uint32_t mask, std::uint64_t user_data);
    [[nodiscard]] std::expected<BodyHandle, Error>
    CreateHeightfieldBody(std::span<const float> heights, std::size_t grid_size, JPH::Vec3 scale,
                          JPH::RVec3 position, JPH::Quat rotation, float friction, float restitution,
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
    [[nodiscard]] std::expected<Buffer<JPH::Float3>, Error>
    GetSoftBodyLocalVertices(BodyHandle handle);
    // Returns whether the body is submerged; a stale handle reports
    // "not submerged" rather than an error (legacy behavior).
    [[nodiscard]] std::expected<bool, Error>
    ApplyBuoyancy(BodyHandle handle, JPH::Real surface_y, float buoyancy, float linear_drag,
                  float angular_drag, JPH::Vec3 fluid_velocity, float dt);
    [[nodiscard]] std::expected<void, Error>
    ApplyBuoyancyBatch(std::span<const BodyHandle> handles, JPH::Real surface_y, float buoyancy,
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

    // Alias for physics_system.GetBodyInterface(). It was a JoltC view until
    // A5; it is a plain Jolt reference now, but it is still nullable and the
    // world's "is it initialized" gate for several call sites, so it stays a
    // pointer rather than a reference.
    JPH::BodyInterface *bodyInterface = nullptr;
    // Native listener, constructed with `this` (A3). Jolt only stores the
    // pointer, so it must outlive physics_system — it does: declared here,
    // physics_system is declared last and therefore destroyed first.
    WorldContactListener contactListener;
    std::uint32_t maxJoltBodies    = 0;
    std::size_t contactMaxCapacity = 0;

    CPH_CACHE_LINE_SPACER;

    // --- Hot simulation state ---
    double time = 0.0;
    std::atomic<std::size_t> count{0};
    std::size_t capacity      = 0; // == slot_capacity == bodies.size()
    std::size_t slotCapacity = 0;
    std::atomic<std::size_t> freeCount{0};

    BodyBuffers bodies;
    // Storage retired by the most recent resize. Released at the next
    // resize (or teardown) so a lock-free reader of the previous frame's
    // shadow buffers can never observe freed memory.
    BodyBuffers retiredBodies;

    CPH_CACHE_LINE_SPACER;

    // --- Synchronization (see core/Threading.hpp for the hierarchy) ---
    ShadowSync stepSync;

    CPH_CACHE_LINE_SPACER;

    // Acquisition order is jph_trampoline_lock -> shadow_lock -> step_sync
    // (step_sync lives in the block above). See core/Threading.hpp.
    std::mutex shadowLock;
    std::mutex jphTrampolineLock;

    Buffer<PhysicsCommand> commandQueue;
    Buffer<PhysicsCommand> commandQueueSpare;
    std::size_t commandCount = 0;

    CPH_CACHE_LINE_SPACER;
    std::atomic<bool> isStepping{false};
    std::atomic<bool> stepRequested{false};
    CPH_CACHE_LINE_SPACER;
    std::atomic<int> waitingThreads{0};

    CPH_CACHE_LINE_SPACER;
    std::atomic<int> activeQueries{0};
    std::atomic<int> viewExportCount{0};
    bool needsOptimization = false;

    CPH_CACHE_LINE_SPACER;

    // --- Contacts, materials, shapes ---
    Buffer<ContactEvent> contactBuffer;
    std::atomic<std::size_t> contactAtomicIdx{0};
    std::size_t contactCount = 0;

    Buffer<MaterialData> materials;
    std::size_t materialCount = 0;

    Buffer<ShapeEntry> shapeCache;
    std::size_t shapeCacheCount = 0;

    // Indexed by Jolt BodyID index (size max_jolt_bodies + 1).
    Buffer<const void *> joltBodyPtrs;
    Buffer<std::atomic<BodyHandle>> idToHandleMap;

    CPH_CACHE_LINE_SPACER;

    // --- Constraints ---
    Buffer<ConstraintSlot> constraints;
    Buffer<std::uint32_t> freeConstraintSlots;
    std::size_t freeConstraintCount = 0;

    CPH_CACHE_LINE_SPACER;

    // --- Debug output ---
    // Heap-owned, not a value member: JPH::DebugRendererSimple stores an
    // RVec3 (32-byte aligned in double precision), which would force
    // PhysicsWorld itself to alignof 32 — and the world lives inside a
    // CPython-allocated object that only guarantees max_align_t. Same reason
    // the job system is heap-owned. One allocation per world, exactly like
    // the JoltC renderer it replaces.
    std::unique_ptr<WorldDebugRenderer> debugRenderer;
    DebugBuffer debugLines;
    DebugBuffer debugTriangles;

    // --- Direct Jolt core ---
    // Declared last so member destruction tears physics_system down before
    // the body, shape, and constraint storage declared above. The thread
    // pool is heap-owned: JobSystemThreadPool requires 64-byte alignment,
    // which CPython allocation of the enclosing world object cannot provide.
    JPH::CharacterVsCharacterCollisionSimple characterVsCharacter;
    std::unique_ptr<JPH::TempAllocator> tempAllocator;
    std::unique_ptr<JPH::JobSystemThreadPool> jobSystem;

    // PhysicsSystem is heap-owned for the same reason the job system is, and
    // for a subtler one: it carries alignof 16 (its RVec3 storage), while the
    // guarantee CPython's allocator makes is alignof(std::max_align_t). Those
    // coincide on Linux x86-64 (16) and do not on any 64-bit platform whose
    // long double is 8 bytes wide — macOS arm64, where max_align_t is 8. As a
    // value member it made alignof(PhysicsWorldObject) 16 and tripped the
    // binding's allocator static_assert. Factory-allocated (not a bare `new`)
    // so -fno-exceptions terminates on OOM exactly as it does for the two
    // allocations above; `physics_system` stays a reference so the ~70 call
    // sites that spell `world.physics_system.X()` are untouched.
    std::unique_ptr<JPH::PhysicsSystem> physicsSystemStorage;
    JPH::PhysicsSystem &physicsSystem;
    bool joltInitialized = false;
};

} // namespace CPH
