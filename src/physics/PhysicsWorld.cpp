#include "physics/PhysicsWorld.hpp"

#include "core/Profiling.hpp"
#include "physics/ContactEventDefinitions.hpp"
#include "physics/InternalQuery.hpp"
#include "physics/ShadowSync.hpp"

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

constexpr JPH_Real kVerifyAbiPosX    = 10.0;
constexpr JPH_Real kVerifyAbiPosY    = 20.0;
constexpr JPH_Real kVerifyAbiPosZ    = 30.0;
constexpr double kVerifyAbiTolerance = 0.1;

using ShapeSettingsPtr = JoltPtr<JPH_ShapeSettings, JPH_ShapeSettings_Destroy>;
using ShapePtr         = JoltPtr<JPH_Shape, JPH_Shape_Destroy>;
using BodySettingsPtr  = JoltPtr<JPH_BodyCreationSettings, JPH_BodyCreationSettings_Destroy>;

void set_scaled_mass(JPH_BodyCreationSettings *settings, JPH_MassProperties &properties,
                     float mass, float scale) {
    properties.mass = mass;
    for (auto &column : properties.inertia.column) {
        column.x *= scale;
        column.y *= scale;
        column.z *= scale;
    }
    JPH_BodyCreationSettings_SetMassPropertiesOverride(settings, &properties);
    JPH_BodyCreationSettings_SetOverrideMassProperties(
        settings, JPH_OverrideMassProperties_CalculateInertia);
}

} // namespace

// --- Body settings helpers ---

void ConfigureBodySettings(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                               BodyConfig cfg) {
    if (cfg.is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }
    if (cfg.use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }
    if (cfg.motion_type == MOTION_DYNAMIC) {
        JPH_BodyCreationSettings_SetAllowSleeping(settings, true);
    }
    if (cfg.allowed_dofs != 0U) {
        JPH_BodyCreationSettings_SetAllowedDOFs(settings,
                                                static_cast<JPH_AllowedDOFs>(cfg.allowed_dofs));
    }
    JPH_BodyCreationSettings_SetFriction(settings, cfg.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, cfg.restitution);

    if (cfg.mass > 0.0F) {
        JPH_MassProperties properties;
        JPH_Shape_GetMassProperties(shape, &properties);
        set_scaled_mass(settings, properties, cfg.mass,
                        cfg.mass / std::fmax(properties.mass, 1e-6F));
    }
}

void ApplyBodyCreationProps(JPH_BodyCreationSettings *settings, JPH_Shape *shape,
                                BodyCreationProps props) {
    if (props.mass > 0.0F) {
        JPH_MassProperties properties;
        JPH_Shape_GetMassProperties(shape, &properties);
        if (properties.mass > 1e-6F) {
            set_scaled_mass(settings, properties, props.mass, props.mass / properties.mass);
        }
    }
    if (props.is_sensor) {
        JPH_BodyCreationSettings_SetIsSensor(settings, true);
    }
    if (props.use_ccd) {
        JPH_BodyCreationSettings_SetMotionQuality(settings, JPH_MotionQuality_LinearCast);
    }
    JPH_BodyCreationSettings_SetAllowSleeping(settings, true);
    JPH_BodyCreationSettings_SetFriction(settings, props.friction);
    JPH_BodyCreationSettings_SetRestitution(settings, props.restitution);
}

// --- Synchronization helpers (see core/Threading.hpp) ---

std::unique_lock<std::mutex> LockShadowNotStepping(PhysicsWorld *world) {
    std::unique_lock shadow(world->shadow_lock);
    if (!world->is_stepping.load(std::memory_order::relaxed)) {
        return shadow;
    }

    world->waiting_threads.fetch_add(1, std::memory_order::relaxed);
    while (world->is_stepping.load(std::memory_order::relaxed)) {
        shadow.unlock();
        {
            std::unique_lock sync(world->step_sync.mutex);
            world->step_sync.cond.wait(sync, [&] {
                return !world->is_stepping.load(std::memory_order::relaxed);
            });
        }
        shadow.lock();
    }
    world->waiting_threads.fetch_sub(1, std::memory_order::relaxed);
    return shadow;
}

void WaitUntilNotQuerying(PhysicsWorld *world,
                              std::unique_lock<std::mutex> &shadow) {
    while (world->active_queries.load(std::memory_order::acquire) > 0) {
        shadow.unlock();
        {
            std::unique_lock sync(world->step_sync.mutex);
            world->step_sync.cond.wait(sync, [&] {
                return world->active_queries.load(std::memory_order::relaxed) <= 0;
            });
        }
        shadow.lock();
    }
}

// --- Construction / destruction ---

PhysicsWorld::PhysicsWorld() {
    debug_renderer = JPH_DebugRenderer_Create(this);
}

PhysicsWorld::~PhysicsWorld() {
    ClearCommandQueue(this);

    // Constraints must leave the physics system before it is destroyed.
    for (auto &slot : constraints) {
        if (slot.constraint != nullptr &&
            slot.state.load(std::memory_order::relaxed) == SLOT_ALIVE) {
            if (system != nullptr) {
                JPH_PhysicsSystem_RemoveConstraint(system, slot.constraint);
            }
            JPH_Constraint_Destroy(slot.constraint);
            slot.constraint = nullptr;
        }
    }

    if (system != nullptr) {
        JPH_PhysicsSystem_Destroy(system);
        system = nullptr;
        body_interface = nullptr;
        // Ownership of these three interfaces transferred to the system at
        // creation time.
        bp_interface = nullptr;
        pair_filter  = nullptr;
        bp_filter    = nullptr;
    } else {
        // Initialization failed before ownership transferred.
        if (bp_filter != nullptr) {
            JPH_ObjectVsBroadPhaseLayerFilter_Destroy(bp_filter);
        }
        if (pair_filter != nullptr) {
            JPH_ObjectLayerPairFilter_Destroy(pair_filter);
        }
        if (bp_interface != nullptr) {
            JPH_BroadPhaseLayerInterface_Destroy(bp_interface);
        }
    }
    if (char_vs_char_manager != nullptr) {
        JPH_CharacterVsCharacterCollision_Destroy(char_vs_char_manager);
    }
    if (job_system != nullptr) {
        JPH_JobSystem_Destroy(job_system);
    }
    if (temp_allocator != nullptr) {
        JPH_TempAllocator_Destroy(temp_allocator);
    }
    if (debug_renderer != nullptr) {
        JPH_DebugRenderer_Destroy(debug_renderer);
    }
    if (contact_listener != nullptr) {
        JPH_ContactListener_Destroy(contact_listener);
    }
    // Buffers, shape cache entries, and debug streams free themselves as
    // members (declaration-reverse order) after this body runs.
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

int AllocateBuffers(PhysicsWorld *world, int max_bodies, std::size_t initial_count) {
    const std::size_t max_capacity = max_bodies > 0 ? static_cast<std::size_t>(max_bodies) : 0;
    world->max_jolt_bodies         = static_cast<std::uint32_t>(max_bodies);

    constexpr std::size_t kInitialSlotCapacity = 64;
    std::size_t initial_capacity =
        max_capacity < kInitialSlotCapacity ? max_capacity : kInitialSlotCapacity;
    if (initial_count > initial_capacity && initial_count <= max_capacity) {
        initial_capacity = initial_count;
    }
    world->capacity      = initial_capacity;
    world->slot_capacity = initial_capacity;

    if (!world->bodies.allocate(initial_capacity) ||
        !world->id_to_handle_map.allocate(world->max_jolt_bodies + 1) ||
        !world->jolt_body_ptrs.allocate(world->max_jolt_bodies + 1) ||
        !world->command_queue.allocate(kCommandQueueInitialCapacity) ||
        !world->command_queue_spare.allocate(kCommandQueueInitialCapacity) ||
        !world->constraints.allocate(kConstraintInitialCapacity) ||
        !world->free_constraint_slots.allocate(kConstraintInitialCapacity)) {
        return -1;
    }

    world->contact_max_capacity = kContactMaxCapacity;
    // for_overwrite: value-initializing 32k events (4 MB) dominated world
    // construction time, and the bytes are never read before being
    // written — every event writer zeroes its claimed slot before filling
    // it, and TakeContactEvents only copies slots a writer claimed.
    if (!world->contact_buffer.allocate_for_overwrite(world->contact_max_capacity)) {
        return -1;
    }

    for (std::size_t i = 0; i < world->capacity; ++i) {
        world->bodies.categories[i] = kAllLayerBits;
        world->bodies.masks[i]      = kAllLayerBits;
        world->bodies.body_ids[i]   = kJphInvalidBodyId;
    }
    for (std::uint32_t i = 0; i < kConstraintInitialCapacity; ++i) {
        world->free_constraint_slots[i] = i;
    }
    world->free_constraint_count = kConstraintInitialCapacity;
    return 0;
}

void InitializeFreeSlots(PhysicsWorld *world, std::size_t start_idx) {
    for (auto i = static_cast<std::uint32_t>(start_idx);
         i < static_cast<std::uint32_t>(world->slot_capacity); ++i) {
        world->bodies.generations[i].store(1, std::memory_order::relaxed);
        const std::size_t free_index =
            world->free_count.fetch_add(1, std::memory_order::relaxed);
        world->bodies.free_slots[free_index] = i;
    }
}

PhysicsWorldStatus PhysicsWorldResize(PhysicsWorld *world, std::size_t new_capacity) {
    if (world->view_export_count.load(std::memory_order::relaxed) > 0) {
        return PHYSICS_WORLD_STATUS_BUFFER_EXPORTED;
    }

    // Retire the previous epoch now, in the mutator thread, before
    // allocating the replacement storage.
    world->retired_bodies = BodyBuffers{};

    if (new_capacity > world->max_jolt_bodies) {
        new_capacity = world->max_jolt_bodies;
    }
    if (new_capacity <= world->capacity) {
        return PHYSICS_WORLD_STATUS_OK;
    }

    BodyBuffers next;
    // for_overwrite: live ranges are memcpy'd below and the newly exposed
    // dense range is deliberately not touched here (fresh pages fault in
    // lazily as bodies are created — value-initializing tens of thousands
    // of elements × 17 arrays is exactly the work legacy C's plain malloc
    // did not do, and it dominated resize cost).
    if (!next.allocate(new_capacity, true)) {
        return PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
    }

    BodyBuffers &current            = world->bodies;
    const std::size_t count         = world->count.load(std::memory_order::acquire);
    const std::size_t current_free  = world->free_count.load(std::memory_order::acquire);

    // Dense data for live bodies.
    if (count > 0) {
        std::memcpy(next.positions.data(), current.positions.data(),
                    count * sizeof(PosStride));
        std::memcpy(next.prev_positions.data(), current.prev_positions.data(),
                    count * sizeof(PosStride));
        std::memcpy(next.rotations.data(), current.rotations.data(),
                    count * sizeof(AuxStride));
        std::memcpy(next.prev_rotations.data(), current.prev_rotations.data(),
                    count * sizeof(AuxStride));
        std::memcpy(next.linear_velocities.data(), current.linear_velocities.data(),
                    count * sizeof(AuxStride));
        std::memcpy(next.angular_velocities.data(), current.angular_velocities.data(),
                    count * sizeof(AuxStride));
        std::memcpy(next.body_ids.data(), current.body_ids.data(), count * sizeof(JPH_BodyID));
        std::memcpy(next.user_data.data(), current.user_data.data(),
                    count * sizeof(std::uint64_t));
        std::memcpy(next.categories.data(), current.categories.data(),
                    count * sizeof(std::uint32_t));
        std::memcpy(next.masks.data(), current.masks.data(), count * sizeof(std::uint32_t));
        std::memcpy(next.material_ids.data(), current.material_ids.data(),
                    count * sizeof(std::uint32_t));
        std::memcpy(next.dense_to_slot.data(), current.dense_to_slot.data(),
                    count * sizeof(std::uint32_t));
        for (std::size_t i = 0; i < count; ++i) {
            next.soft_shadows[i] = std::move(current.soft_shadows[i]);
        }
    }

    // Slot maps. Atomic arrays are copied element-wise.
    for (std::size_t i = 0; i < world->slot_capacity; ++i) {
        next.generations[i].store(current.generations[i].load(std::memory_order::relaxed),
                                   std::memory_order::relaxed);
        next.slot_states[i].store(current.slot_states[i].load(std::memory_order::relaxed),
                                  std::memory_order::relaxed);
    }
    if (world->slot_capacity > 0) {
        std::memcpy(next.slot_to_dense.data(), current.slot_to_dense.data(),
                    world->slot_capacity * sizeof(std::uint32_t));
    }
    if (current_free > 0) {
        std::memcpy(next.free_slots.data(), current.free_slots.data(),
                    current_free * sizeof(std::uint32_t));
    }

    // Expanded slots start empty at generation 1, on the free stack.
    std::size_t free_count = current_free;
    for (std::size_t i = world->slot_capacity; i < new_capacity; ++i) {
        next.generations[i].store(1, std::memory_order::relaxed);
        next.slot_states[i].store(SLOT_EMPTY, std::memory_order::relaxed);
        next.slot_to_dense[i]         = 0;
        next.free_slots[free_count++] = static_cast<std::uint32_t>(i);
    }
    // The expanded dense range [count, new_capacity) stays untouched on
    // purpose (see allocate above): every path that claims a dense index
    // defines all of its fields before the body becomes visible — paths
    // that don't take explicit values write creation defaults at claim
    // time (reserve_body_slot for single/soft creates, the batch loop in
    // PhysicsWorldCreateBodiesBatch for batch creates).

    world->retired_bodies = std::exchange(world->bodies, std::move(next));
    world->free_count.store(free_count, std::memory_order::release);
    world->capacity      = new_capacity;
    world->slot_capacity = new_capacity;
    return PHYSICS_WORLD_STATUS_OK;
}

// --- Baked scene loading ---

PhysicsWorldStatus PhysicsWorldLoadBakedScene(PhysicsWorld *world,
                                                  const BakedSceneData *scene) {
    if (world == nullptr || scene == nullptr) {
        return PHYSICS_WORLD_STATUS_INVALID_ARGUMENT;
    }
    if (scene->count == 0) {
        return PHYSICS_WORLD_STATUS_OK;
    }
    if (scene->positions == nullptr || scene->rotations == nullptr || scene->shapes == nullptr ||
        scene->motion_types == nullptr || scene->object_layers == nullptr ||
        scene->user_data == nullptr || world->body_interface == nullptr) {
        return PHYSICS_WORLD_STATUS_INVALID_ARGUMENT;
    }
    if (scene->count > world->capacity || scene->count > world->slot_capacity ||
        scene->count > world->max_jolt_bodies) {
        return PHYSICS_WORLD_STATUS_LIMIT_REACHED;
    }

    PhysicsWorldStatus status = PHYSICS_WORLD_STATUS_OK;
    {
        const std::scoped_lock shadow(world->shadow_lock);
        const std::scoped_lock trampoline(world->jph_trampoline_lock);
        for (std::size_t i = 0; i < scene->count; ++i) {
            const float *const shape_data = &scene->shapes[i * BAKED_SHAPE_STRIDE];
            const float params[4] = {shape_data[1], shape_data[2], shape_data[3], shape_data[4]};
            JPH_Shape *const shape =
                FindOrCreateShapeLocked(world, static_cast<int>(shape_data[0]), params);
            if (CPH_UNLIKELY(shape == nullptr)) {
                status = PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
                break;
            }

            const std::size_t pos_idx = i * BAKED_POSITION_STRIDE;
            const std::size_t rot_idx = i * BAKED_ROTATION_STRIDE;
            PosStride &shadow_pos = world->bodies.positions[i];
            AuxStride &shadow_rot = world->bodies.rotations[i];
            shadow_pos = {scene->positions[pos_idx + 0], scene->positions[pos_idx + 1],
                          scene->positions[pos_idx + 2], 0.0};
            shadow_rot = {scene->rotations[rot_idx + 0], scene->rotations[rot_idx + 1],
                          scene->rotations[rot_idx + 2], scene->rotations[rot_idx + 3]};

            JPH_RVec3 j_pos = {shadow_pos.x, shadow_pos.y, shadow_pos.z};
            JPH_Quat j_rot  = {shadow_rot.x, shadow_rot.y, shadow_rot.z, shadow_rot.w};
            const std::uint8_t motion_type = scene->motion_types[i];
            const BodySettingsPtr creation(JPH_BodyCreationSettings_Create3(
                shape, &j_pos, &j_rot, static_cast<JPH_MotionType>(motion_type),
                static_cast<JPH_ObjectLayer>(scene->object_layers[i])));
            if (CPH_UNLIKELY(!creation)) {
                status = PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
                break;
            }

            world->bodies.generations[i].store(1, std::memory_order::relaxed);
            const BodyHandle handle = make_handle(static_cast<std::uint32_t>(i), 1);
            JPH_BodyCreationSettings_SetUserData(creation.get(), handle);
            if (motion_type == MOTION_DYNAMIC) {
                JPH_BodyCreationSettings_SetAllowSleeping(creation.get(), true);
            }

            const JPH_BodyID body_id = JPH_BodyInterface_CreateAndAddBody(
                world->body_interface, creation.get(), JPH_Activation_Activate);
            if (CPH_UNLIKELY(body_id == kJphInvalidBodyId)) {
                status = PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
                break;
            }

            world->bodies.body_ids[i] = body_id;
            const std::uint32_t jolt_idx = JPH_ID_TO_INDEX(body_id);
            if (world->id_to_handle_map && jolt_idx < world->max_jolt_bodies) {
                world->id_to_handle_map[jolt_idx].store(handle, std::memory_order::relaxed);
            }
            world->bodies.slot_to_dense[i] = static_cast<std::uint32_t>(i);
            world->bodies.dense_to_slot[i] = static_cast<std::uint32_t>(i);
            world->bodies.slot_states[i].store(SLOT_ALIVE, std::memory_order::relaxed);
            world->bodies.user_data[i] = scene->user_data[i];
        }
    }

    if (status == PHYSICS_WORLD_STATUS_OK) {
        // Publish the initial body count only after every body was created.
        world->count.store(scene->count, std::memory_order::relaxed);
    }
    return status;
}

// --- Slot reservation for deferred creation ---

namespace {

PhysicsWorldStatus reserve_body_slot(PhysicsWorld *world, std::uint32_t slot_state,
                                     BodyHandle *handle_out) {
    const std::size_t current_count = world->count.load(std::memory_order::acquire);
    std::size_t available           = world->free_count.load(std::memory_order::acquire);

    if (CPH_UNLIKELY(current_count >= world->max_jolt_bodies)) {
        return PHYSICS_WORLD_STATUS_LIMIT_REACHED;
    }
    if (CPH_UNLIKELY(available == 0 || current_count + 1 > world->capacity)) {
        std::size_t next_capacity =
            world->capacity == 0 ? kInitialBodyCapacity : world->capacity * 2;
        if (next_capacity > world->max_jolt_bodies) {
            next_capacity = world->max_jolt_bodies;
        }
        const PhysicsWorldStatus resize_status = PhysicsWorldResize(world, next_capacity);
        if (resize_status != PHYSICS_WORLD_STATUS_OK) {
            return resize_status;
        }
        available = world->free_count.load(std::memory_order::acquire);
        if (CPH_UNLIKELY(available == 0)) {
            return PHYSICS_WORLD_STATUS_LIMIT_REACHED;
        }
    }

    if (CPH_UNLIKELY(!EnsureCommandCapacity(world))) {
        return PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
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
    world->bodies.body_ids[dense]      = kJphInvalidBodyId;
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
    return PHYSICS_WORLD_STATUS_OK;
}

} // namespace

PhysicsWorldStatus PhysicsWorldCommitCreate(PhysicsWorld *world,
                                                 JPH_BodyCreationSettings *settings,
                                                 std::uint32_t slot_state, BodyHandle *out_handle) {
    if (!world || !settings || !out_handle) {
        return PHYSICS_WORLD_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = 0;

    BodyHandle handle = 0;
    const PhysicsWorldStatus status = reserve_body_slot(world, slot_state, &handle);
    if (status != PHYSICS_WORLD_STATUS_OK) {
        return status;
    }
    JPH_BodyCreationSettings_SetUserData(settings, handle);
    *out_handle = handle;
    return PHYSICS_WORLD_STATUS_OK;
}

PhysicsWorldStatus PhysicsWorldCommitCreateSoft(PhysicsWorld *world,
                                                     JPH_SoftBodyCreationSettings *settings,
                                                     std::uint32_t slot_state,
                                                     BodyHandle *out_handle) {
    if (!world || !settings || !out_handle) {
        return PHYSICS_WORLD_STATUS_INVALID_ARGUMENT;
    }
    *out_handle = 0;

    BodyHandle handle = 0;
    const PhysicsWorldStatus status = reserve_body_slot(world, slot_state, &handle);
    if (status != PHYSICS_WORLD_STATUS_OK) {
        return status;
    }
    JPH_SoftBodyCreationSettings_SetUserData(settings, handle);
    *out_handle = handle;
    return PHYSICS_WORLD_STATUS_OK;
}

// --- Jolt bring-up ---

int PhysicsWorldInitJolt(PhysicsWorld *world, WorldSettings settings,
                             GravityVector gravity) {
    int num_workers = settings.num_threads;
#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    // TSan cannot model Jolt's lock-free job queues; run Jolt synchronously
    // so TSan focuses on Culverin's own synchronization.
    num_workers = 0;
#endif
    const JobSystemThreadPoolConfig job_cfg = {
        .maxJobs     = static_cast<std::uint32_t>(settings.max_physics_jobs),
        .maxBarriers = static_cast<std::uint32_t>(settings.max_physics_barriers),
        .numThreads  = num_workers};

    // Serialize first PhysicsSystem creation (Jolt lazy statics, see TSan note).
    std::unique_lock init_guard(g_jph_init_lock);

    world->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
    if (!world->job_system) {
        return -1;
    }

#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
    world->temp_allocator = JPH_TempAllocatorMalloc_Create();
#else
    world->temp_allocator = JPH_TempAllocator_Create(static_cast<std::uint32_t>(settings.temp_allocator_size));
#endif
    if (!world->temp_allocator) {
        return -1;
    }

    // 3 layers: 0=Static, 1=Dynamic, 2=VehicleRay.
    world->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(OBJECT_LAYER_COUNT, 3);
    if (!world->bp_interface) {
        return -1;
    }
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(world->bp_interface,
                                                                 OBJECT_LAYER_STATIC, 0);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(world->bp_interface,
                                                                 OBJECT_LAYER_DYNAMIC, 1);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(world->bp_interface,
                                                                 OBJECT_LAYER_VEHICLE, 2);

    world->pair_filter = JPH_ObjectLayerPairFilterTable_Create(3);
    if (!world->pair_filter) {
        return -1;
    }
    JPH_ObjectLayerPairFilterTable_EnableCollision(world->pair_filter, 0, 1);
    JPH_ObjectLayerPairFilterTable_EnableCollision(world->pair_filter, 0, 2);
    JPH_ObjectLayerPairFilterTable_EnableCollision(world->pair_filter, 1, 1);
    JPH_ObjectLayerPairFilterTable_DisableCollision(world->pair_filter, 1, 2);

    world->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(world->bp_interface, 3,
                                                                     world->pair_filter, 3);
    if (!world->bp_filter) {
        return -1;
    }

    const JPH_PhysicsSystemSettings phys_settings = {
        .maxBodies             = static_cast<std::uint32_t>(settings.max_bodies),
        .numBodyMutexes        = 0, // 0 = Jolt default
        .maxBodyPairs          = static_cast<std::uint32_t>(settings.max_pairs),
        .maxContactConstraints =
            static_cast<std::uint32_t>(settings.max_contact_constraints),
        ._padding                      = 0,
        .broadPhaseLayerInterface      = world->bp_interface,
        .objectLayerPairFilter         = world->pair_filter,
        .objectVsBroadPhaseLayerFilter = world->bp_filter};

    world->system = JPH_PhysicsSystem_Create(&phys_settings);
    init_guard.unlock();
    if (!world->system) {
        return -1;
    }

    world->char_vs_char_manager = JPH_CharacterVsCharacterCollision_CreateSimple();
    if (!world->char_vs_char_manager) {
        return -1;
    }
    JPH_Vec3 jolt_gravity = {gravity.gx, gravity.gy, gravity.gz};
    JPH_PhysicsSystem_SetGravity(world->system, &jolt_gravity);
    world->body_interface = JPH_PhysicsSystem_GetBodyInterface(world->system);
    return world->body_interface ? 0 : -1;
}

// --- Stepping ---

std::span<PhysicsCommand> PhysicsWorldDetachCommands(PhysicsWorld *world) {
    const std::size_t captured = world->command_count;
    std::swap(world->command_queue, world->command_queue_spare);
    world->command_count = 0;
    return {world->command_queue_spare.data(), captured};
}

void PhysicsWorldStepNative(PhysicsWorld *world, float dt,
                                std::span<PhysicsCommand> commands) {
    const std::scoped_lock trampoline(world->jph_trampoline_lock);

    CPH_PROFILE_BEGIN(jolt_step);

    if (!commands.empty()) {
        FlushCommandsInternal(world, commands);
        world->needs_optimization = true;
    }

    if (dt <= 0.0F) {
        JPH_PhysicsSystem_OptimizeBroadPhase(world->system);
        world->needs_optimization = false;
    } else {
        JPH_PhysicsSystem_Update2(world->system, dt, 1, world->temp_allocator, world->job_system);
        if (world->needs_optimization) {
            JPH_PhysicsSystem_OptimizeBroadPhase(world->system);
            world->needs_optimization = false;
        }
    }

    // Shadow buffers are written only once in-flight queries have drained.
    {
        std::unique_lock sync(world->step_sync.mutex);
        world->step_sync.cond.wait(
            sync, [&] { return world->active_queries.load(std::memory_order::acquire) <= 0; });
    }

    SyncShadowBuffers(world);

    CPH_PROFILE_END(jolt_step, "Jolt Physics Crunch", static_cast<unsigned int>(commands.size()));
}

std::size_t PhysicsWorldFinalizeStep(PhysicsWorld *world, float dt) {
    const std::size_t contact_count =
        world->contact_atomic_idx.load(std::memory_order::acquire);
    world->contact_count = std::min(contact_count, world->contact_max_capacity);

    const std::size_t body_count = world->count.load(std::memory_order::acquire);
    world->time += static_cast<double>(dt);

    world->is_stepping.store(false, std::memory_order::release);
    world->step_requested.store(false, std::memory_order::release);
    NotifyStepSync(world);
    return body_count;
}

int VerifyAbiAlignment(JPH_BodyInterface *body_interface) {
    JPH_Vec3 half_extents = {1, 1, 1};
    const ShapeSettingsPtr box_settings(
        reinterpret_cast<JPH_ShapeSettings *>(JPH_BoxShapeSettings_Create(&half_extents, 0.0F)));
    if (!box_settings) {
        return -1;
    }
    const ShapePtr shape(reinterpret_cast<JPH_Shape *>(
        JPH_BoxShapeSettings_CreateShape(reinterpret_cast<JPH_BoxShapeSettings *>(box_settings.get()))));
    if (!shape) {
        return -1;
    }

    JPH_RVec3 position = {kVerifyAbiPosX, kVerifyAbiPosY, kVerifyAbiPosZ};
    JPH_Quat rotation  = {0, 0, 0, 1};
    const BodySettingsPtr settings(JPH_BodyCreationSettings_Create3(
        shape.get(), &position, &rotation, JPH_MotionType_Static, 0));
    if (!settings) {
        return -1;
    }

    const JPH_BodyID bid =
        JPH_BodyInterface_CreateAndAddBody(body_interface, settings.get(), JPH_Activation_Activate);
    if (bid == kJphInvalidBodyId) {
        return -1;
    }

    alignas(kAvxAlignment) JPH_RVec3 check{};
    JPH_BodyInterface_GetPosition(body_interface, bid, &check);
    JPH_BodyInterface_RemoveBody(body_interface, bid);
    JPH_BodyInterface_DestroyBody(body_interface, bid);

    if (std::fabs(check.x - kVerifyAbiPosX) > kVerifyAbiTolerance ||
        std::fabs(check.y - kVerifyAbiPosY) > kVerifyAbiTolerance) {
        return -1;
    }
    return 0;
}

} // namespace CPH
