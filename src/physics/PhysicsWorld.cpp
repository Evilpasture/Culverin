#include "physics/PhysicsWorld.hpp"

#include "core/Profiling.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/ShadowSync.hpp"
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayer.h>
#include <Jolt/Physics/Collision/Shape/BoxShape.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>


namespace CPH {

namespace {

constexpr std::size_t kCommandQueueInitialCapacity = 64;
constexpr std::size_t kInitialBodyCapacity         = 1024;
constexpr std::size_t kConstraintInitialCapacity   = 256;
constexpr std::size_t kContactMaxCapacity          = sizeof(ContactEvent) * 8 << 5;

constexpr JPH::Real kVerifyAbiPosX    = 10.0;
constexpr JPH::Real kVerifyAbiPosY    = 20.0;
constexpr JPH::Real kVerifyAbiPosZ    = 30.0;
constexpr double kVerifyAbiTolerance  = 0.1;

void set_scaled_mass(JPH::BodyCreationSettings &settings, JPH::MassProperties properties,
                     float mass, float scale) {
    properties.mMass = mass;
    for (JPH::uint column = 0; column < 4; ++column) {
        const JPH::Vec4 value = properties.mInertia.GetColumn4(column);
        properties.mInertia.SetColumn4(
            column, JPH::Vec4(value.GetX() * scale, value.GetY() * scale, value.GetZ() * scale,
                              value.GetW()));
    }
    settings.mMassPropertiesOverride = properties;
    settings.mOverrideMassProperties = JPH::EOverrideMassProperties::CalculateInertia;
}

// Culverin has three object layers and uses the identical numbering for
// broadphase layers. The pair matrix is fixed: Static collides with Dynamic
// and Vehicle, and Dynamic collides with itself. The remaining pairs
// (Static/Static, Dynamic/Vehicle, Vehicle/Vehicle) do not collide.
[[nodiscard]] constexpr bool object_layers_collide(std::uint32_t layer1,
                                                   std::uint32_t layer2) noexcept {
    if (layer1 > layer2) {
        const std::uint32_t temporary = layer1;
        layer1                        = layer2;
        layer2                        = temporary;
    }
    return (layer1 == std::to_underlying(ObjectLayer::Static) &&
            (layer2 == std::to_underlying(ObjectLayer::Dynamic) ||
             layer2 == std::to_underlying(ObjectLayer::Vehicle))) ||
           (layer1 == std::to_underlying(ObjectLayer::Dynamic) &&
            layer2 == std::to_underlying(ObjectLayer::Dynamic));
}

class WorldBroadPhaseLayerInterface final : public JPH::BroadPhaseLayerInterface {
  public:
    [[nodiscard]] JPH::uint GetNumBroadPhaseLayers() const override {
        return std::to_underlying(ObjectLayer::Count);
    }

    [[nodiscard]] JPH::BroadPhaseLayer
    GetBroadPhaseLayer(JPH::ObjectLayer layer) const override {
        return layer < std::to_underlying(ObjectLayer::Count)
                   ? JPH::BroadPhaseLayer(static_cast<JPH::BroadPhaseLayer::Type>(layer))
                   : JPH::cBroadPhaseLayerInvalid;
    }

#if defined(JPH_EXTERNAL_PROFILE) || defined(JPH_PROFILE_ENABLED)
    [[nodiscard]] const char *GetBroadPhaseLayerName(JPH::BroadPhaseLayer layer) const override {
        switch (static_cast<std::uint32_t>(layer.GetValue())) {
        case std::to_underlying(ObjectLayer::Static):
            return "Static";
        case std::to_underlying(ObjectLayer::Dynamic):
            return "Dynamic";
        case std::to_underlying(ObjectLayer::Vehicle):
            return "Vehicle";
        default:
            return "Invalid";
        }
    }
#endif
};

class WorldObjectLayerPairFilter final : public JPH::ObjectLayerPairFilter {
  public:
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer layer1,
                                     JPH::ObjectLayer layer2) const override {
        return object_layers_collide(static_cast<std::uint32_t>(layer1),
                                     static_cast<std::uint32_t>(layer2));
    }
};

class WorldObjectVsBroadPhaseLayerFilter final : public JPH::ObjectVsBroadPhaseLayerFilter {
  public:
    [[nodiscard]] bool ShouldCollide(JPH::ObjectLayer layer,
                                     JPH::BroadPhaseLayer broadphase_layer) const override {
        // Broadphase numbering is identical to object-layer numbering.
        return object_layers_collide(static_cast<std::uint32_t>(layer),
                                     static_cast<std::uint32_t>(broadphase_layer.GetValue()));
    }
};

} // namespace

// --- Body settings helpers ---

void ConfigureBodySettings(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                               BodyConfig cfg) {
    if (cfg.is_sensor) {
        settings.mIsSensor = true;
    }
    if (cfg.use_ccd) {
        settings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    }
    if (cfg.motion_type == std::to_underlying(MotionType::Dynamic)) {
        settings.mAllowSleeping = true;
    }
    if (cfg.allowed_dofs != 0U) {
        settings.mAllowedDOFs = static_cast<JPH::EAllowedDOFs>(cfg.allowed_dofs);
    }
    settings.mFriction    = cfg.friction;
    settings.mRestitution = cfg.restitution;

    if (cfg.mass > 0.0F) {
        const JPH::MassProperties properties = shape->GetMassProperties();
        set_scaled_mass(settings, properties, cfg.mass,
                        cfg.mass / std::fmax(properties.mMass, 1e-6F));
    }
}

void ApplyBodyCreationProps(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                                BodyCreationProps props) {
    if (props.mass > 0.0F) {
        const JPH::MassProperties properties = shape->GetMassProperties();
        if (properties.mMass > 1e-6F) {
            set_scaled_mass(settings, properties, props.mass, props.mass / properties.mMass);
        }
    }
    if (props.is_sensor) {
        settings.mIsSensor = true;
    }
    if (props.use_ccd) {
        settings.mMotionQuality = JPH::EMotionQuality::LinearCast;
    }
    settings.mAllowSleeping = true;
    settings.mFriction      = props.friction;
    settings.mRestitution   = props.restitution;
}

// --- Synchronization helpers (see core/Threading.hpp) ---

std::unique_lock<std::mutex> PhysicsWorld::LockShadowNotStepping() {
    std::unique_lock shadow(shadow_lock);
    if (!is_stepping.load(std::memory_order::relaxed)) {
        return shadow;
    }

    waiting_threads.fetch_add(1, std::memory_order::relaxed);
    while (is_stepping.load(std::memory_order::relaxed)) {
        shadow.unlock();
        {
            std::unique_lock sync(step_sync.mutex);
            step_sync.cond.wait(sync, [&] {
                return !is_stepping.load(std::memory_order::relaxed);
            });
        }
        shadow.lock();
    }
    waiting_threads.fetch_sub(1, std::memory_order::relaxed);
    return shadow;
}

void PhysicsWorld::WaitUntilNotQuerying(std::unique_lock<std::mutex> &shadow) {
    while (active_queries.load(std::memory_order::acquire) > 0) {
        shadow.unlock();
        {
            std::unique_lock sync(step_sync.mutex);
            step_sync.cond.wait(sync, [&] {
                return active_queries.load(std::memory_order::relaxed) <= 0;
            });
        }
        shadow.lock();
    }
}

// --- Construction / destruction ---

// Both listeners back-point at the world, so the world is fully wired the
// moment its storage exists — no create/destroy pair and no null checks
// downstream. The renderer is heap-owned for alignment reasons (see the
// member declaration).
PhysicsWorld::PhysicsWorld()
    : contact_listener(this), debug_renderer(std::make_unique<WorldDebugRenderer>(this)) {}

PhysicsWorld::~PhysicsWorld() {
    ClearCommandQueue();

    // Constraints must leave the physics system before it is destroyed.
    for (auto &slot : constraints) {
        if (slot.constraint != nullptr &&
            slot.state.load(std::memory_order::relaxed) == std::to_underlying(SlotState::Alive)) {
            if (jolt_initialized) {
                physics_system.RemoveConstraint(slot.constraint);
            }
            slot.constraint->Release();
            slot.constraint = nullptr;
        }
    }

    // Unlink the listener before the system goes away; the listener object
    // itself is a member and dies with the rest of the storage below.
    if (jolt_initialized) {
        physics_system.SetContactListener(nullptr);
    }
    body_interface   = nullptr;
    jolt_initialized = false;

    // Buffers, shape cache entries, debug streams, the debug renderer
    // (DebugRenderer's singleton bookkeeping), and the contact listener free
    // themselves as members (declaration-reverse order) after this body runs.
    // physics_system is declared last, so it is torn down first.
}

// --- Storage ---

bool BodyBuffers::allocate(std::size_t capacity, bool for_overwrite) noexcept {
    BodyBuffers next;
    const auto alloc = [capacity, for_overwrite](auto &buffer) {
        return for_overwrite ? buffer.allocate_for_overwrite(capacity)
                             : buffer.allocate(capacity);
    };
    const bool ok =
        alloc(next.positions) && alloc(next.prev_positions) && alloc(next.rotations) &&
        alloc(next.prev_rotations) && alloc(next.linear_velocities) &&
        alloc(next.angular_velocities) && alloc(next.body_ids) && alloc(next.user_data) &&
        alloc(next.categories) && alloc(next.masks) && alloc(next.material_ids) &&
        alloc(next.soft_shadows) && alloc(next.slot_to_dense) && alloc(next.dense_to_slot) &&
        alloc(next.free_slots) && alloc(next.generations) && alloc(next.slot_states);
    if (!ok) {
        return false;
    }
    *this = std::move(next);
    return true;
}

std::expected<void, Error> PhysicsWorld::AllocateBuffers(int max_bodies,
                                                         std::size_t initial_count) {
    const std::size_t max_capacity = max_bodies > 0 ? static_cast<std::size_t>(max_bodies) : 0;
    max_jolt_bodies         = static_cast<std::uint32_t>(max_bodies);

    constexpr std::size_t kInitialSlotCapacity = 64;
    std::size_t initial_capacity =
        max_capacity < kInitialSlotCapacity ? max_capacity : kInitialSlotCapacity;
    if (initial_count > initial_capacity && initial_count <= max_capacity) {
        initial_capacity = initial_count;
    }
    capacity      = initial_capacity;
    slot_capacity = initial_capacity;

    if (!bodies.allocate(initial_capacity) ||
        !id_to_handle_map.allocate(max_jolt_bodies + 1) ||
        !jolt_body_ptrs.allocate(max_jolt_bodies + 1) ||
        !command_queue.allocate(kCommandQueueInitialCapacity) ||
        !command_queue_spare.allocate(kCommandQueueInitialCapacity) ||
        !constraints.allocate(kConstraintInitialCapacity) ||
        !free_constraint_slots.allocate(kConstraintInitialCapacity)) {
        return std::unexpected(WorldError::OutOfMemory);
    }

    contact_max_capacity = kContactMaxCapacity;
    // for_overwrite: value-initializing 32k events (4 MB) dominated world
    // construction time, and the bytes are never read before being
    // written — every event writer zeroes its claimed slot before filling
    // it, and TakeContactEvents only copies slots a writer claimed.
    if (!contact_buffer.allocate_for_overwrite(contact_max_capacity)) {
        return std::unexpected(WorldError::OutOfMemory);
    }

    for (std::size_t i = 0; i < capacity; ++i) {
        bodies.categories[i] = kAllLayerBits;
        bodies.masks[i]      = kAllLayerBits;
        bodies.body_ids[i]   = JPH::BodyID(); // default ctor == cInvalidBodyID
    }
    for (std::uint32_t i = 0; i < kConstraintInitialCapacity; ++i) {
        free_constraint_slots[i] = i;
    }
    free_constraint_count = kConstraintInitialCapacity;
    return {};
}

void PhysicsWorld::InitializeFreeSlots(std::size_t start_idx) {
    for (auto i = static_cast<std::uint32_t>(start_idx);
         i < static_cast<std::uint32_t>(slot_capacity); ++i) {
        bodies.generations[i].store(1, std::memory_order::relaxed);
        const std::size_t free_index =
            free_count.fetch_add(1, std::memory_order::relaxed);
        bodies.free_slots[free_index] = i;
    }
}

std::expected<void, Error> PhysicsWorld::Resize(std::size_t new_capacity) {
    if (view_export_count.load(std::memory_order::relaxed) > 0) {
        return std::unexpected(WorldError::BufferExported);
    }

    // Retire the previous epoch now, in the mutator thread, before
    // allocating the replacement storage.
    retired_bodies = BodyBuffers{};

    if (new_capacity > max_jolt_bodies) {
        new_capacity = max_jolt_bodies;
    }
    if (new_capacity <= capacity) {
        return {};
    }

    BodyBuffers next;
    // for_overwrite: live ranges are memcpy'd below and the newly exposed
    // dense range is deliberately not touched here (fresh pages fault in
    // lazily as bodies are created — value-initializing tens of thousands
    // of elements × 17 arrays is exactly the work legacy C's plain malloc
    // did not do, and it dominated resize cost).
    if (!next.allocate(new_capacity, true)) {
        return std::unexpected(WorldError::OutOfMemory);
    }

    BodyBuffers &current            = bodies;
    const std::size_t live_count    = count.load(std::memory_order::acquire);
    const std::size_t current_free  = free_count.load(std::memory_order::acquire);

    // Dense data for live bodies.
    if (live_count > 0) {
        std::memcpy(next.positions.data(), current.positions.data(),
                    live_count * sizeof(PosStride));
        std::memcpy(next.prev_positions.data(), current.prev_positions.data(),
                    live_count * sizeof(PosStride));
        std::memcpy(next.rotations.data(), current.rotations.data(),
                    live_count * sizeof(AuxStride));
        std::memcpy(next.prev_rotations.data(), current.prev_rotations.data(),
                    live_count * sizeof(AuxStride));
        std::memcpy(next.linear_velocities.data(), current.linear_velocities.data(),
                    live_count * sizeof(AuxStride));
        std::memcpy(next.angular_velocities.data(), current.angular_velocities.data(),
                    live_count * sizeof(AuxStride));
        std::memcpy(next.body_ids.data(), current.body_ids.data(), live_count * sizeof(JPH::BodyID));
        std::memcpy(next.user_data.data(), current.user_data.data(),
                    live_count * sizeof(std::uint64_t));
        std::memcpy(next.categories.data(), current.categories.data(),
                    live_count * sizeof(std::uint32_t));
        std::memcpy(next.masks.data(), current.masks.data(), live_count * sizeof(std::uint32_t));
        std::memcpy(next.material_ids.data(), current.material_ids.data(),
                    live_count * sizeof(std::uint32_t));
        std::memcpy(next.dense_to_slot.data(), current.dense_to_slot.data(),
                    live_count * sizeof(std::uint32_t));
        for (std::size_t i = 0; i < live_count; ++i) {
            next.soft_shadows[i] = std::move(current.soft_shadows[i]);
        }
    }

    // Slot maps. Atomic arrays are copied element-wise.
    for (std::size_t i = 0; i < slot_capacity; ++i) {
        next.generations[i].store(current.generations[i].load(std::memory_order::relaxed),
                                   std::memory_order::relaxed);
        next.slot_states[i].store(current.slot_states[i].load(std::memory_order::relaxed),
                                  std::memory_order::relaxed);
    }
    if (slot_capacity > 0) {
        std::memcpy(next.slot_to_dense.data(), current.slot_to_dense.data(),
                    slot_capacity * sizeof(std::uint32_t));
    }
    if (current_free > 0) {
        std::memcpy(next.free_slots.data(), current.free_slots.data(),
                    current_free * sizeof(std::uint32_t));
    }

    // Expanded slots start empty at generation 1, on the free stack.
    std::size_t new_free_count = current_free;
    for (std::size_t i = slot_capacity; i < new_capacity; ++i) {
        next.generations[i].store(1, std::memory_order::relaxed);
        next.slot_states[i].store(std::to_underlying(SlotState::Empty), std::memory_order::relaxed);
        next.slot_to_dense[i]         = 0;
        next.free_slots[new_free_count++] = static_cast<std::uint32_t>(i);
    }
    // The expanded dense range [live_count, new_capacity) stays untouched on
    // purpose (see allocate above): every path that claims a dense index
    // defines all of its fields before the body becomes visible — paths
    // that don't take explicit values write creation defaults at claim
    // time (reserve_body_slot for single/soft creates, the batch loop in
    // CreateBodiesBatch for batch creates).

    retired_bodies = std::exchange(bodies, std::move(next));
    free_count.store(new_free_count, std::memory_order::release);
    capacity      = new_capacity;
    slot_capacity = new_capacity;
    return {};
}

// --- Baked scene loading ---

std::expected<void, Error> PhysicsWorld::LoadBakedScene(const BakedSceneData &scene) {
    if (scene.count == 0) {
        return {};
    }
    if (scene.positions == nullptr || scene.rotations == nullptr || scene.shapes == nullptr ||
        scene.motion_types == nullptr || scene.object_layers == nullptr ||
        scene.user_data == nullptr || body_interface == nullptr) {
        return std::unexpected(WorldError::InvalidArgument);
    }
    if (scene.count > capacity || scene.count > slot_capacity ||
        scene.count > max_jolt_bodies) {
        return std::unexpected(WorldError::LimitReached);
    }

    Error error;
    {
        // Lock order (core/Threading.hpp): jph_trampoline_lock is outermost,
        // shadow_lock nests inside it. Every other path that needs both takes
        // the trampoline first (StepNative/FlushCommands, SyncAndFlushInternal,
        // ragdoll and compound construction), so taking shadow_lock first here
        // closed a cycle that TSan reports as lock-order-inversion:
        //   shadow_lock -> jph_trampoline_lock (this function)
        //   jph_trampoline_lock -> shadow_lock (FlushCommands, per command)
        // The cycle is a genuine deadlock as soon as a scene load races a step.
        const std::scoped_lock trampoline(jph_trampoline_lock);
        const std::scoped_lock shadow(shadow_lock);
        for (std::size_t i = 0; i < scene.count; ++i) {
            const float *const shape_data = &scene.shapes[i * kBakedShapeStride];
            const float params[4] = {shape_data[1], shape_data[2], shape_data[3], shape_data[4]};
            const JPH::ShapeRefC shape =
                FindOrCreateShapeLocked(static_cast<int>(shape_data[0]), params);
            if (shape == nullptr) [[unlikely]] {
                error = WorldError::OutOfMemory;
                break;
            }

            const std::size_t pos_idx = i * kBakedPositionStride;
            const std::size_t rot_idx = i * kBakedRotationStride;
            PosStride &shadow_pos = bodies.positions[i];
            AuxStride &shadow_rot = bodies.rotations[i];
            shadow_pos = {scene.positions[pos_idx + 0], scene.positions[pos_idx + 1],
                          scene.positions[pos_idx + 2], 0.0};
            shadow_rot = {scene.rotations[rot_idx + 0], scene.rotations[rot_idx + 1],
                          scene.rotations[rot_idx + 2], scene.rotations[rot_idx + 3]};

            const std::uint8_t motion_type = scene.motion_types[i];
            JPH::BodyCreationSettings creation(
                shape.GetPtr(), JPH::RVec3(shadow_pos.x, shadow_pos.y, shadow_pos.z),
                JPH::Quat(shadow_rot.x, shadow_rot.y, shadow_rot.z, shadow_rot.w),
                static_cast<JPH::EMotionType>(motion_type),
                static_cast<JPH::ObjectLayer>(scene.object_layers[i]));

            bodies.generations[i].store(1, std::memory_order::relaxed);
            const BodyHandle handle = make_handle(static_cast<std::uint32_t>(i), 1);
            creation.mUserData      = handle;
            if (motion_type == std::to_underlying(MotionType::Dynamic)) {
                creation.mAllowSleeping = true;
            }

            const JPH::BodyID body_id = physics_system.GetBodyInterface().CreateAndAddBody(
                creation, JPH::EActivation::Activate);
            if (body_id.IsInvalid()) [[unlikely]] {
                error = WorldError::OutOfMemory;
                break;
            }

            bodies.body_ids[i] = body_id;
            const std::uint32_t jolt_idx = body_id.GetIndex();
            if (id_to_handle_map && jolt_idx < max_jolt_bodies) {
                id_to_handle_map[jolt_idx].store(handle, std::memory_order::relaxed);
            }
            bodies.slot_to_dense[i] = static_cast<std::uint32_t>(i);
            bodies.dense_to_slot[i] = static_cast<std::uint32_t>(i);
            bodies.slot_states[i].store(std::to_underlying(SlotState::Alive), std::memory_order::relaxed);
            bodies.user_data[i] = scene.user_data[i];
        }
    }

    if (!error) {
        // Publish the initial body count only after every body was created.
        count.store(scene.count, std::memory_order::relaxed);
    }
    if (error) {
        return std::unexpected(error);
    }
    return {};
}

// --- Slot reservation for deferred creation ---

namespace {

std::expected<void, Error> reserve_body_slot(PhysicsWorld *world, std::uint32_t slot_state,
                                     BodyHandle *handle_out) {
    const std::size_t current_count = world->count.load(std::memory_order::acquire);
    std::size_t available           = world->free_count.load(std::memory_order::acquire);

    if (current_count >= world->max_jolt_bodies) [[unlikely]] {
        return std::unexpected(WorldError::LimitReached);
    }
    if (available == 0 || current_count + 1 > world->capacity) [[unlikely]] {
        std::size_t next_capacity =
            world->capacity == 0 ? kInitialBodyCapacity : world->capacity * 2;
        if (next_capacity > world->max_jolt_bodies) {
            next_capacity = world->max_jolt_bodies;
        }
        const std::expected<void, Error> resize_status = world->Resize(next_capacity);
        if (!resize_status) {
            return resize_status;
        }
        available = world->free_count.load(std::memory_order::acquire);
        if (available == 0) [[unlikely]] {
            return std::unexpected(WorldError::LimitReached);
        }
    }

    if (!world->EnsureCommandCapacity()) [[unlikely]] {
        return std::unexpected(WorldError::OutOfMemory);
    }

    const std::uint32_t slot = world->bodies.free_slots[--available];
    const auto dense = static_cast<std::uint32_t>(
        world->count.fetch_add(1, std::memory_order::relaxed));
    world->free_count.store(available, std::memory_order::release);

    const std::uint32_t generation =
        world->bodies.generations[slot].load(std::memory_order::relaxed);
    const BodyHandle handle = make_handle(slot, generation);

    world->bodies.slot_to_dense[slot]  = dense;
    world->bodies.dense_to_slot[dense] = slot;
    world->bodies.body_ids[dense]      = JPH::BodyID();
    // Creation defaults for the fields resize leaves indeterminate. The
    // single-body path overwrites all of these from its request; the
    // soft-body path overwrites categories/masks/user_data only.
    world->bodies.linear_velocities[dense]  = {};
    world->bodies.angular_velocities[dense] = {};
    world->bodies.user_data[dense]          = 0;
    world->bodies.categories[dense]         = kAllLayerBits;
    world->bodies.masks[dense]              = kAllLayerBits;
    world->bodies.material_ids[dense]       = 0;
    world->bodies.slot_states[slot].store(static_cast<std::uint8_t>(slot_state),
                                          std::memory_order::release);

    *handle_out = handle;
    return {};
}

} // namespace

std::expected<BodyHandle, Error>
PhysicsWorld::CommitCreate(JPH::BodyCreationSettings &settings, std::uint32_t slot_state) {
    BodyHandle handle = 0;
    const std::expected<void, Error> status = reserve_body_slot(this, slot_state, &handle);
    if (!status) {
        return std::unexpected(status.error());
    }
    settings.mUserData = handle;
    return handle;
}

std::expected<BodyHandle, Error>
PhysicsWorld::CommitCreateSoft(JPH::SoftBodyCreationSettings &settings,
                               std::uint32_t slot_state) {
    BodyHandle handle = 0;
    const std::expected<void, Error> status = reserve_body_slot(this, slot_state, &handle);
    if (!status) {
        return std::unexpected(status.error());
    }
    settings.mUserData = handle;
    return handle;
}

// --- Jolt bring-up ---

bool PhysicsWorld::InitJolt(WorldSettings settings, GravityVector gravity) {
    if (jolt_initialized) {
        return false;
    }

    int num_workers = settings.num_threads > 0 ? settings.num_threads : -1;
#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    // TSan cannot model Jolt's lock-free job queues; run Jolt synchronously
    // so TSan focuses on Culverin's own synchronization.
    num_workers = 0;
#endif

    const auto max_jobs = settings.max_physics_jobs > 0
                              ? static_cast<JPH::uint>(settings.max_physics_jobs)
                              : JPH::cMaxPhysicsJobs;
    const auto max_barriers = settings.max_physics_barriers > 0
                                  ? static_cast<JPH::uint>(settings.max_physics_barriers)
                                  : JPH::cMaxPhysicsBarriers;
    const auto max_bodies =
        settings.max_bodies > 0 ? static_cast<JPH::uint>(settings.max_bodies) : 10240U;
    const auto max_pairs =
        settings.max_pairs > 0 ? static_cast<JPH::uint>(settings.max_pairs) : 65536U;
    const auto max_contact_constraints = settings.max_contact_constraints > 0
                                             ? static_cast<JPH::uint>(settings.max_contact_constraints)
                                             : 10240U;

    // Serialize PhysicsSystem initialization across worlds. Direct Jolt
    // initialization still reaches process-wide lazy state, and TSan cannot
    // prove those first-touch paths race-free; retaining the established
    // bring-up lock preserves Culverin's initialization guarantee.
    const std::scoped_lock init_guard(g_jph_init_lock);

    job_system =
        std::make_unique<JPH::JobSystemThreadPool>(max_jobs, max_barriers, num_workers);

#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    temp_allocator = std::make_unique<JPH::TempAllocatorMalloc>();
#else
    // WithMallocFallback preserves the previous allocator behavior: the
    // fixed buffer serves normal steps, and overflow falls back to malloc
    // instead of failing the step.
    temp_allocator = std::make_unique<JPH::TempAllocatorImplWithMallocFallback>(
        settings.temp_allocator_size > 0 ? static_cast<std::size_t>(settings.temp_allocator_size)
                                         : 0U);
#endif

    static const WorldBroadPhaseLayerInterface broadphase_layer_interface;
    static const WorldObjectLayerPairFilter object_layer_pair_filter;
    static const WorldObjectVsBroadPhaseLayerFilter object_vs_broadphase_filter;

    physics_system.Init(max_bodies, 0, max_pairs, max_contact_constraints,
                               broadphase_layer_interface, object_vs_broadphase_filter,
                               object_layer_pair_filter);
    physics_system.SetGravity(JPH::Vec3(gravity.gx, gravity.gy, gravity.gz));
    body_interface = &physics_system.GetBodyInterface();
    jolt_initialized = true;
    return true;
}

// --- Stepping ---

std::span<PhysicsCommand> PhysicsWorld::DetachCommands() {
    const std::size_t captured = command_count;
    std::swap(command_queue, command_queue_spare);
    command_count = 0;
    return {command_queue_spare.data(), captured};
}

void PhysicsWorld::StepNative(float dt, std::span<PhysicsCommand> commands) {
    const std::scoped_lock trampoline(jph_trampoline_lock);

    CPH_PROFILE_BEGIN(jolt_step);

    if (!commands.empty()) {
        FlushCommands(commands);
        needs_optimization = true;
    }

    if (dt <= 0.0F) {
        physics_system.OptimizeBroadPhase();
        needs_optimization = false;
    } else {
        [[maybe_unused]] const JPH::EPhysicsUpdateError update_error =
            physics_system.Update(dt, 1, temp_allocator.get(),
                                         job_system.get());
        if (needs_optimization) {
            physics_system.OptimizeBroadPhase();
            needs_optimization = false;
        }
    }

    // Shadow buffers are written only once in-flight queries have drained.
    {
        std::unique_lock sync(step_sync.mutex);
        step_sync.cond.wait(
            sync, [&] { return active_queries.load(std::memory_order::acquire) <= 0; });
    }

    SyncShadowBuffers(this);

    CPH_PROFILE_END(jolt_step, "Jolt Physics Crunch", static_cast<unsigned int>(commands.size()));
}

std::size_t PhysicsWorld::FinalizeStep(float dt) {
    const std::size_t body_count = count.load(std::memory_order::acquire);
    time += static_cast<double>(dt);

    is_stepping.store(false, std::memory_order::release);
    step_requested.store(false, std::memory_order::release);
    NotifyStepSync();
    return body_count;
}

// See the declaration in PhysicsWorld.hpp. Two things about this function are
// load-bearing beyond the obvious check:
//
//   1. It must run BEFORE the caller creates any body. Creating and destroying
//      a body here advances Jolt's BodyID free-list and touches the broadphase,
//      which changes the BodyID every later body receives. Body order feeds the
//      solver, so removing this call changes ragdoll and soft-body numerics
//      (~888 lines of ragdoll_probe.py output moved when it was dropped) even
//      though nothing about the simulation logic changed. Keep the call, and
//      keep it before the first user body.
//   2. It was never a pure check: prefer to fix it rather than delete it.
int VerifyAbiAlignment(JPH::BodyInterface &body_interface) {
    const JPH::BoxShapeSettings box_settings(JPH::Vec3(1.0f, 1.0f, 1.0f), 0.0f);
    const auto shape_result = box_settings.Create();
    if (!shape_result.IsValid()) {
        return -1;
    }
    const JPH::ShapeRefC shape = shape_result.Get();

    JPH::BodyCreationSettings settings(shape.GetPtr(),
                                       JPH::RVec3(kVerifyAbiPosX, kVerifyAbiPosY, kVerifyAbiPosZ),
                                       JPH::Quat::sIdentity(), JPH::EMotionType::Static, 0);

    const JPH::BodyID bid = body_interface.CreateAndAddBody(settings, JPH::EActivation::Activate);
    if (bid.IsInvalid()) {
        return -1;
    }

    const JPH::RVec3 check = body_interface.GetPosition(bid);
    body_interface.RemoveBody(bid);
    body_interface.DestroyBody(bid);

    if (std::fabs(check.GetX() - kVerifyAbiPosX) > kVerifyAbiTolerance ||
        std::fabs(check.GetY() - kVerifyAbiPosY) > kVerifyAbiTolerance) {
        return -1;
    }
    return 0;
}

} // namespace CPH
