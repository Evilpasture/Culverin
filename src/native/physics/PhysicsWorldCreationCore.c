#include "physics/PhysicsWorldCreationCore.h"
#include "core/NativeAlloc.h"
#include "physics/CommandBuffer.h"
#include "physics/InternalQuery.h"
#include "physics/PhysicsWorldCore.h"
#include "physics/SlotState.h"
#include <string.h>

static constexpr float DEFAULT_MATERIAL_FRICTION      = 0.2f;
static constexpr float CONVEX_HULL_TOLERANCE          = 0.05f;
static constexpr size_t INITIAL_BODY_CAPACITY         = 1024;
static constexpr size_t RAGDOLL_BODY_BUFFER_INCREMENT = 1024;

static void lock_until_not_stepping(PhysicsWorld *world) {
    SHADOW_LOCK(&world->shadow_lock);
    if (!atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
        return;
    }

    atomic_fetch_add_explicit(&world->waiting_threads, 1, memory_order_relaxed);
    while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        NATIVE_MUTEX_LOCK(world->step_sync.mutex);
        while (atomic_load_explicit(&world->is_stepping, memory_order_relaxed)) {
            NATIVE_COND_WAIT(world->step_sync.cond, world->step_sync.mutex);
        }
        NATIVE_MUTEX_UNLOCK(world->step_sync.mutex);
        SHADOW_LOCK(&world->shadow_lock);
    }
    atomic_fetch_sub_explicit(&world->waiting_threads, 1, memory_order_relaxed);
}

static void initialize_body_shadow(PhysicsWorld *world, BodyHandle handle,
                                   const PhysicsWorldBodyCreateRequest *request) {
    const uint32_t slot      = (uint32_t)(handle & HANDLE_INDEX_MASK);
    const uint32_t dense     = world->slot_to_dense[slot];
    const PosStride position = {request->position.x, request->position.y, request->position.z, 0.0};
    const AuxStride rotation = {request->rotation.x, request->rotation.y, request->rotation.z,
                                request->rotation.w};

    ((PosStride *)world->positions)[dense]          = position;
    ((PosStride *)world->prev_positions)[dense]     = position;
    ((AuxStride *)world->rotations)[dense]          = rotation;
    ((AuxStride *)world->prev_rotations)[dense]     = rotation;
    ((AuxStride *)world->linear_velocities)[dense]  = (AuxStride){};
    ((AuxStride *)world->angular_velocities)[dense] = (AuxStride){};
    world->categories[dense]                        = request->category;
    world->masks[dense]                             = request->mask;
    world->material_ids[dense]                      = request->material_id;
    world->user_data[dense]                         = request->user_data;
}

static PhysicsWorldCreationStatus
enqueue_body_settings(PhysicsWorld *world, JPH_BodyCreationSettings *settings,
                      JPH_Shape *owned_shape, const PhysicsWorldBodyCreateRequest *request,
                      BodyHandle *out_handle, size_t *out_count,
                      PhysicsWorldStatus *out_native_status) {
    lock_until_not_stepping(world);
    BodyHandle handle = 0;
    const PhysicsWorldStatus native_status =
        physics_world_commit_create(world, settings, SLOT_PENDING_CREATE, &handle);
    if (native_status != PHYSICS_WORLD_STATUS_OK) {
        SHADOW_UNLOCK(&world->shadow_lock);
        JPH_BodyCreationSettings_Destroy(settings);
        if (owned_shape) {
            JPH_Shape_Destroy(owned_shape);
        }
        *out_native_status = native_status;
        return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
    }

    initialize_body_shadow(world, handle, request);
    const uint32_t slot         = (uint32_t)(handle & HANDLE_INDEX_MASK);
    PhysicsCommand *command     = &world->command_queue[world->command_count++];
    command->header             = CMD_HEADER(CMD_CREATE_BODY, slot);
    command->create.settings    = settings;
    command->create.user_data   = request->user_data;
    command->create.category    = request->category;
    command->create.mask        = request->mask;
    command->create.material_id = request->material_id;

    *out_handle = handle;
    *out_count  = atomic_load_explicit(&world->count, memory_order_relaxed);
    SHADOW_UNLOCK(&world->shadow_lock);
    if (owned_shape) {
        JPH_Shape_Destroy(owned_shape);
    }
    return PHYSICS_WORLD_CREATION_STATUS_OK;
}

static PhysicsWorldCreationStatus
create_body_from_shape(PhysicsWorld *world, JPH_Shape *shape, bool owns_shape,
                       const PhysicsWorldBodyCreateRequest *request, JPH_MotionType motion_type,
                       uint32_t object_layer, PhysicsWorldBodySetup setup, BodyHandle *out_handle,
                       size_t *out_count, PhysicsWorldStatus *out_native_status) {
    if (!shape) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }

    JPH_RVec3 position = request->position;
    JPH_Quat rotation  = request->rotation;
    JPH_BodyCreationSettings *settings =
        JPH_BodyCreationSettings_Create3(shape, &position, &rotation, motion_type, object_layer);
    if (!settings) {
        if (owns_shape) {
            JPH_Shape_Destroy(shape);
        }
        return PHYSICS_WORLD_CREATION_STATUS_BODY_SETTINGS_FAILED;
    }

    switch (setup) {
    case PHYSICS_WORLD_BODY_SETUP_CONFIGURED:
        configure_body_settings(settings, shape, request->config);
        break;
    case PHYSICS_WORLD_BODY_SETUP_CREATION_PROPS: {
        const BodyCreationProps props = {
            request->config.mass,      request->config.friction, request->config.restitution,
            request->config.is_sensor, request->config.use_ccd,
        };
        apply_body_creation_props(settings, shape, props);
        break;
    }
    case PHYSICS_WORLD_BODY_SETUP_MATERIAL_ONLY:
        JPH_BodyCreationSettings_SetFriction(settings, request->config.friction);
        JPH_BodyCreationSettings_SetRestitution(settings, request->config.restitution);
        break;
    case PHYSICS_WORLD_BODY_SETUP_NONE:
        break;
    }

    return enqueue_body_settings(world, settings, owns_shape ? shape : nullptr, request, out_handle,
                                 out_count, out_native_status);
}

void physics_world_resolve_material(PhysicsWorld *world, uint32_t material_id, float input_friction,
                                    float input_restitution, float *out_friction,
                                    float *out_restitution) {
    if (!world || !out_friction || !out_restitution) {
        return;
    }

    float friction    = DEFAULT_MATERIAL_FRICTION;
    float restitution = 0.0f;
    if (material_id > 0) {
        SHADOW_LOCK(&world->shadow_lock);
        for (size_t i = 0; i < world->material_count; i++) {
            if (world->materials[i].id == material_id) {
                friction    = world->materials[i].friction;
                restitution = world->materials[i].restitution;
                break;
            }
        }
        SHADOW_UNLOCK(&world->shadow_lock);
    }

    *out_friction    = input_friction >= 0.0f ? input_friction : friction;
    *out_restitution = input_restitution >= 0.0f ? input_restitution : restitution;
}

PhysicsWorldCreationStatus physics_world_create_body(PhysicsWorld *world,
                                                     const PhysicsWorldBodyCreateRequest *request,
                                                     BodyHandle *out_handle, size_t *out_count,
                                                     PhysicsWorldStatus *out_native_status) {
    if (!world || !request || !out_handle || !out_count || !out_native_status) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_handle        = 0;
    *out_count         = 0;
    *out_native_status = PHYSICS_WORLD_STATUS_OK;

    SHADOW_LOCK(&world->shadow_lock);
    JPH_Shape *shape =
        find_or_create_shape_locked(world, request->shape_type, request->shape_params);
    SHADOW_UNLOCK(&world->shadow_lock);
    if (!shape) {
        return PHYSICS_WORLD_CREATION_STATUS_BODY_SETTINGS_FAILED;
    }

    const uint32_t object_layer = request->config.motion_type == MOTION_KINEMATIC ||
                                          request->config.motion_type == MOTION_STATIC
                                      ? OBJECT_LAYER_STATIC
                                      : OBJECT_LAYER_DYNAMIC;
    return create_body_from_shape(
        world, shape, false, request, (JPH_MotionType)request->config.motion_type, object_layer,
        PHYSICS_WORLD_BODY_SETUP_CONFIGURED, out_handle, out_count, out_native_status);
}

PhysicsWorldCreationStatus physics_world_create_soft_body(
    PhysicsWorld *world, const PhysicsWorldSoftBodyCreateRequest *request, BodyHandle *out_handle,
    size_t *out_count, PhysicsWorldStatus *out_native_status) {
    if (!world || !request || !request->shared_settings || !out_handle || !out_count ||
        !out_native_status) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_handle        = 0;
    *out_count         = 0;
    *out_native_status = PHYSICS_WORLD_STATUS_OK;

    JPH_SoftBodyCreationSettings *settings = JPH_SoftBodyCreationSettings_Create();
    if (!settings) {
        return PHYSICS_WORLD_CREATION_STATUS_BODY_SETTINGS_FAILED;
    }
    JPH_SoftBodyCreationSettings_SetSettings(settings, request->shared_settings);
    JPH_SoftBodyCreationSettings_SetPressure(settings, request->pressure);
    JPH_SoftBodyCreationSettings_SetVertexRadius(settings, request->vertex_radius);
    JPH_SoftBodyCreationSettings_SetLinearDamping(settings, request->linear_damping);
    JPH_SoftBodyCreationSettings_SetNumIterations(settings, request->num_iterations);
    JPH_SoftBodyCreationSettings_SetMaxLinearVelocity(settings, request->max_linear_velocity);
    JPH_SoftBodyCreationSettings_SetGravityFactor(settings, request->gravity_factor);
    JPH_SoftBodyCreationSettings_SetFriction(settings, request->friction);
    JPH_SoftBodyCreationSettings_SetRestitution(settings, request->restitution);
    JPH_SoftBodyCreationSettings_SetMakeRotationIdentity(settings, request->make_rotation_identity);

    JPH_RVec3 position = request->position;
    JPH_Quat rotation  = request->rotation;
    JPH_SoftBodyCreationSettings_SetPosition(settings, &position);
    JPH_SoftBodyCreationSettings_SetRotation(settings, &rotation);
    JPH_SoftBodyCreationSettings_SetObjectLayer(settings, OBJECT_LAYER_DYNAMIC);
    JPH_SoftBodyCreationSettings_SetAllowSleeping(settings, true);
    JPH_SoftBodyCreationSettings_SetMakeRotationIdentity(settings, request->make_rotation_identity);
    JPH_SoftBodyCreationSettings_SetUpdatePosition(settings, request->update_position);
    JPH_SoftBodyCreationSettings_SetFacesDoubleSided(settings, request->faces_double_sided);

    lock_until_not_stepping(world);
    BodyHandle handle = 0;
    const PhysicsWorldStatus native_status =
        physics_world_commit_create_soft(world, settings, SLOT_PENDING_CREATE, &handle);
    if (native_status != PHYSICS_WORLD_STATUS_OK) {
        SHADOW_UNLOCK(&world->shadow_lock);
        JPH_SoftBodyCreationSettings_Destroy(settings);
        *out_native_status = native_status;
        return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
    }

    const uint32_t slot                         = (uint32_t)(handle & HANDLE_INDEX_MASK);
    const uint32_t dense                        = world->slot_to_dense[slot];
    const PosStride position_shadow             = {request->position.x, request->position.y,
                                                   request->position.z, 0.0};
    const AuxStride rotation_shadow             = {request->rotation.x, request->rotation.y,
                                                   request->rotation.z, request->rotation.w};
    ((PosStride *)world->positions)[dense]      = position_shadow;
    ((PosStride *)world->prev_positions)[dense] = position_shadow;
    ((AuxStride *)world->rotations)[dense]      = rotation_shadow;
    ((AuxStride *)world->prev_rotations)[dense] = rotation_shadow;
    world->categories[dense]                    = request->category;
    world->masks[dense]                         = request->mask;
    world->user_data[dense]                     = request->user_data;

    PhysicsCommand *command           = &world->command_queue[world->command_count++];
    command->header                   = CMD_HEADER(CMD_CREATE_SOFT_BODY, slot);
    command->create_soft.settings     = settings;
    command->create_soft.category     = request->category;
    command->create_soft.mask         = request->mask;
    command->create_soft.num_vertices = request->vertex_count;

    *out_handle = handle;
    *out_count  = atomic_load_explicit(&world->count, memory_order_relaxed);
    SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_CREATION_STATUS_OK;
}

JPH_RagdollSettings *physics_world_create_ragdoll_settings(JPH_Skeleton *skeleton) {
    if (!skeleton) {
        return nullptr;
    }
    JPH_RagdollSettings *settings = JPH_RagdollSettings_Create();
    if (settings) {
        JPH_RagdollSettings_SetSkeleton(settings, skeleton);
    }
    return settings;
}

PhysicsWorldCreationStatus
physics_world_create_ragdoll_instance(PhysicsWorld *world, JPH_RagdollSettings *settings,
                                      JPH_RVec3 position, JPH_Quat rotation, uint64_t user_data,
                                      JPH_Ragdoll **out_ragdoll, size_t *out_body_count) {
    if (!world || !settings || !out_ragdoll || !out_body_count || !world->system) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_ragdoll    = nullptr;
    *out_body_count = 0;

    NATIVE_MUTEX_LOCK(world->jph_trampoline_lock);
    JPH_RagdollSettings_CalculateBodyIndexToConstraintIndex(settings);
    JPH_RagdollSettings_CalculateConstraintIndexToBodyIdxPair(settings);
    JPH_Ragdoll *ragdoll = JPH_RagdollSettings_CreateRagdoll(settings, world->system, 0, user_data);
    if (!ragdoll) {
        NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
        return PHYSICS_WORLD_CREATION_STATUS_RAGDOLL_FAILED;
    }

    const size_t joint_count =
        (size_t)JPH_Skeleton_GetJointCount(JPH_RagdollSettings_GetSkeleton(settings));
    if (joint_count > SIZE_MAX / sizeof(JPH_Mat4)) {
        JPH_Ragdoll_Destroy(ragdoll);
        NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
        return PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY;
    }
    JPH_Mat4 *neutral_matrices =
        joint_count ? CulvNative_Malloc(joint_count * sizeof(JPH_Mat4)) : nullptr;
    if (joint_count && !neutral_matrices) {
        JPH_Ragdoll_Destroy(ragdoll);
        NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
        return PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY;
    }

    JPH_RVec3 zero_root = {0, 0, 0};
    JPH_Ragdoll_GetPose2(ragdoll, &zero_root, neutral_matrices, true);
    JPH_Mat4 rotation_matrix;
    JPH_Mat4_Rotation(&rotation_matrix, &rotation);
    for (size_t i = 0; i < joint_count; i++) {
        JPH_Mat4 result;
        JPH_Mat4_Multiply(&rotation_matrix, &neutral_matrices[i], &result);
        neutral_matrices[i] = result;
    }

    JPH_Ragdoll_SetPose2(ragdoll, &position, neutral_matrices, true);
    JPH_Ragdoll_AddToPhysicsSystem(ragdoll, JPH_Activation_Activate, true);
    *out_body_count = (size_t)JPH_Ragdoll_GetBodyCount(ragdoll);
    *out_ragdoll    = ragdoll;
    CulvNative_Free(neutral_matrices);
    NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
    return PHYSICS_WORLD_CREATION_STATUS_OK;
}

PhysicsWorldCreationStatus physics_world_register_ragdoll_bodies(
    PhysicsWorld *world, JPH_Ragdoll *ragdoll, size_t body_count, uint64_t user_data,
    uint32_t category, uint32_t mask, uint32_t material_id, uint32_t *body_slots,
    size_t *out_world_count, PhysicsWorldStatus *out_native_status) {
    if (!world || !ragdoll || !body_slots || !out_world_count || !out_native_status) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_world_count   = 0;
    *out_native_status = PHYSICS_WORLD_STATUS_OK;

    lock_until_not_stepping(world);
    const size_t current_count = atomic_load_explicit(&world->count, memory_order_acquire);
    size_t available           = atomic_load_explicit(&world->free_count, memory_order_acquire);
    if (current_count > world->max_jolt_bodies ||
        body_count > world->max_jolt_bodies - current_count) {
        SHADOW_UNLOCK(&world->shadow_lock);
        *out_native_status = PHYSICS_WORLD_STATUS_LIMIT_REACHED;
        return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
    }
    if (available < body_count) {
        if (body_count > SIZE_MAX - world->capacity - RAGDOLL_BODY_BUFFER_INCREMENT) {
            SHADOW_UNLOCK(&world->shadow_lock);
            *out_native_status = PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
            return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
        }
        size_t capacity = world->capacity + body_count + RAGDOLL_BODY_BUFFER_INCREMENT;
        if (capacity > world->max_jolt_bodies) {
            capacity = world->max_jolt_bodies;
        }
        const PhysicsWorldStatus resize_status = physics_world_resize(world, capacity);
        if (resize_status != PHYSICS_WORLD_STATUS_OK) {
            SHADOW_UNLOCK(&world->shadow_lock);
            *out_native_status = resize_status;
            return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
        }
        available = atomic_load_explicit(&world->free_count, memory_order_acquire);
        if (available < body_count) {
            SHADOW_UNLOCK(&world->shadow_lock);
            *out_native_status = PHYSICS_WORLD_STATUS_LIMIT_REACHED;
            return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
        }
    }

    JPH_BodyInterface *body_interface = world->body_interface;
    PosStride *positions              = (PosStride *)world->positions;
    PosStride *previous_positions     = (PosStride *)world->prev_positions;
    AuxStride *rotations              = (AuxStride *)world->rotations;
    AuxStride *previous_rotations     = (AuxStride *)world->prev_rotations;
    AuxStride *linear_velocities      = (AuxStride *)world->linear_velocities;
    AuxStride *angular_velocities     = (AuxStride *)world->angular_velocities;
    for (size_t i = 0; i < body_count; i++) {
        const JPH_BodyID body_id = JPH_Ragdoll_GetBodyID(ragdoll, (int)i);
        const size_t free_index =
            atomic_fetch_sub_explicit(&world->free_count, 1, memory_order_relaxed) - 1;
        const uint32_t slot = world->free_slots[free_index];
        body_slots[i]       = slot;
        const uint32_t dense =
            (uint32_t)atomic_fetch_add_explicit(&world->count, 1, memory_order_relaxed);
        JPH_RVec3 position;
        JPH_Quat rotation;
        JPH_BodyInterface_GetPosition(body_interface, body_id, &position);
        JPH_BodyInterface_GetRotation(body_interface, body_id, &rotation);

        positions[dense]            = (PosStride){position.x, position.y, position.z, 0.0};
        previous_positions[dense]   = positions[dense];
        rotations[dense]            = (AuxStride){rotation.x, rotation.y, rotation.z, rotation.w};
        previous_rotations[dense]   = rotations[dense];
        linear_velocities[dense]    = (AuxStride){};
        angular_velocities[dense]   = (AuxStride){};
        world->body_ids[dense]      = body_id;
        world->slot_to_dense[slot]  = dense;
        world->dense_to_slot[dense] = slot;

        const uint32_t generation =
            atomic_load_explicit(&world->generations[slot], memory_order_relaxed);
        const BodyHandle handle   = make_handle(slot, generation);
        const uint32_t jolt_index = JPH_ID_TO_INDEX(body_id);
        if (world->id_to_handle_map && jolt_index < world->max_jolt_bodies) {
            atomic_store_explicit(&world->id_to_handle_map[jolt_index], handle,
                                  memory_order_release);
        }
        JPH_BodyInterface_SetUserData(body_interface, body_id, handle);
        atomic_store_explicit(&world->slot_states[slot], SLOT_ALIVE, memory_order_release);
        world->user_data[dense]    = user_data;
        world->categories[dense]   = category;
        world->masks[dense]        = mask;
        world->material_ids[dense] = material_id;
    }

    *out_world_count = atomic_load_explicit(&world->count, memory_order_relaxed);
    SHADOW_UNLOCK(&world->shadow_lock);
    return PHYSICS_WORLD_CREATION_STATUS_OK;
}

void physics_world_destroy_ragdoll(PhysicsWorld *world, JPH_Ragdoll *ragdoll) {
    if (!world || !ragdoll) {
        return;
    }
    NATIVE_MUTEX_LOCK(world->jph_trampoline_lock);
    JPH_Ragdoll_Destroy(ragdoll);
    NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
}

PhysicsWorldCreationStatus
physics_world_create_bodies_batch(PhysicsWorld *world, const PosStride *positions,
                                  const ShapeParams *sizes, size_t batch_count, int shape_type,
                                  int motion_type, BodyHandle *out_handles, size_t *out_count,
                                  PhysicsWorldStatus *out_native_status) {
    if (!world || !positions || !sizes || !out_handles || !out_count || !out_native_status ||
        batch_count == 0 || batch_count > SIZE_MAX / sizeof(JPH_BodyCreationSettings *)) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_count         = 0;
    *out_native_status = PHYSICS_WORLD_STATUS_OK;

    JPH_BodyCreationSettings **settings =
        CulvNative_Calloc(batch_count, sizeof(JPH_BodyCreationSettings *));
    if (!settings) {
        return PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY;
    }

    JPH_Shape *last_shape      = nullptr;
    const ShapeParams sentinel = {.p = {-1.0f, -1.0f, -1.0f, -1.0f}};
    ShapeParams last_size      = sentinel;
    for (size_t i = 0; i < batch_count; i++) {
        JPH_Shape *shape      = last_shape;
        const float *current  = sizes[i].p;
        const float *previous = last_size.p;
        if (current[0] != previous[0] || current[1] != previous[1] || current[2] != previous[2] ||
            current[3] != previous[3]) {
            SHADOW_LOCK(&world->shadow_lock);
            shape = find_or_create_shape_locked(world, shape_type, current);
            SHADOW_UNLOCK(&world->shadow_lock);
            last_shape = shape;
            last_size  = sizes[i];
        }

        if (shape) {
            JPH_RVec3 position = {positions[i].x, positions[i].y, positions[i].z};
            JPH_Quat rotation  = {0, 0, 0, 1};
            const uint32_t object_layer =
                motion_type == MOTION_STATIC ? OBJECT_LAYER_STATIC : OBJECT_LAYER_DYNAMIC;
            settings[i] = JPH_BodyCreationSettings_Create3(
                shape, &position, &rotation, (JPH_MotionType)motion_type, object_layer);
        }
    }

    lock_until_not_stepping(world);
    size_t current_count = atomic_load_explicit(&world->count, memory_order_acquire);
    size_t available     = atomic_load_explicit(&world->free_count, memory_order_acquire);
    if (available < batch_count || current_count + batch_count > world->capacity) {
        size_t needed = current_count + batch_count + INITIAL_BODY_CAPACITY;
        if (needed > world->max_jolt_bodies) {
            needed = world->max_jolt_bodies;
        }
        const PhysicsWorldStatus resize_status = physics_world_resize(world, needed);
        if (resize_status != PHYSICS_WORLD_STATUS_OK) {
            SHADOW_UNLOCK(&world->shadow_lock);
            for (size_t i = 0; i < batch_count; i++) {
                if (settings[i]) {
                    JPH_BodyCreationSettings_Destroy(settings[i]);
                }
            }
            CulvNative_Free(settings);
            *out_native_status = resize_status;
            return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
        }
        available = atomic_load_explicit(&world->free_count, memory_order_acquire);
    }

    if (current_count + batch_count > world->max_jolt_bodies) {
        SHADOW_UNLOCK(&world->shadow_lock);
        for (size_t i = 0; i < batch_count; i++) {
            if (settings[i]) {
                JPH_BodyCreationSettings_Destroy(settings[i]);
            }
        }
        CulvNative_Free(settings);
        *out_native_status = PHYSICS_WORLD_STATUS_LIMIT_REACHED;
        return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
    }

    if (!ensure_command_bulk_capacity(world, batch_count)) {
        SHADOW_UNLOCK(&world->shadow_lock);
        for (size_t i = 0; i < batch_count; i++) {
            if (settings[i]) {
                JPH_BodyCreationSettings_Destroy(settings[i]);
            }
        }
        CulvNative_Free(settings);
        *out_native_status = PHYSICS_WORLD_STATUS_OUT_OF_MEMORY;
        return PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR;
    }

    const uint32_t base_dense =
        (uint32_t)atomic_fetch_add_explicit(&world->count, batch_count, memory_order_relaxed);
    const size_t free_head = available - batch_count;
    atomic_store_explicit(&world->free_count, free_head, memory_order_release);
    for (size_t i = 0; i < batch_count; i++) {
        if (!settings[i]) {
            out_handles[i] = 0;
            continue;
        }

        const uint32_t slot  = world->free_slots[free_head + i];
        const uint32_t dense = base_dense + (uint32_t)i;
        const uint32_t generation =
            atomic_load_explicit(&world->generations[slot], memory_order_relaxed);
        const BodyHandle handle = ((uint64_t)generation << HANDLE_INDEX_BITS) | slot;
        JPH_BodyCreationSettings_SetUserData(settings[i], handle);
        world->slot_to_dense[slot]  = dense;
        world->dense_to_slot[dense] = slot;
        world->body_ids[dense]      = JPH_INVALID_BODY_ID;

        PosStride position                          = positions[i];
        position.w                                  = 0.0;
        ((PosStride *)world->positions)[dense]      = position;
        ((PosStride *)world->prev_positions)[dense] = position;
        ((AuxStride *)world->rotations)[dense]      = (AuxStride){0, 0, 0, 1};
        ((AuxStride *)world->prev_rotations)[dense] = (AuxStride){0, 0, 0, 1};
        atomic_store_explicit(&world->slot_states[slot], SLOT_PENDING_CREATE, memory_order_release);

        PhysicsCommand *command  = &world->command_queue[world->command_count++];
        command->header          = CMD_HEADER(CMD_CREATE_BODY, slot);
        command->create.settings = settings[i];
        out_handles[i]           = handle;
    }
    *out_count = atomic_load_explicit(&world->count, memory_order_relaxed);
    SHADOW_UNLOCK(&world->shadow_lock);
    CulvNative_Free(settings);
    return PHYSICS_WORLD_CREATION_STATUS_OK;
}

PhysicsWorldCreationStatus
physics_world_create_convex_hull(PhysicsWorld *world, const float *points, size_t point_count,
                                 const PhysicsWorldBodyCreateRequest *request,
                                 BodyHandle *out_handle, size_t *out_count,
                                 PhysicsWorldStatus *out_native_status) {
    if (!world || !points || !request || !out_handle || !out_count || !out_native_status) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    if (point_count > UINT32_MAX) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_native_status = PHYSICS_WORLD_STATUS_OK;

    JPH_Vec3 *jolt_points = CulvNative_Malloc(point_count * sizeof(*jolt_points));
    if (!jolt_points) {
        return PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < point_count; i++) {
        jolt_points[i] = (JPH_Vec3){points[i * 3], points[i * 3 + 1], points[i * 3 + 2]};
    }

    JPH_ConvexHullShapeSettings *hull_settings = JPH_ConvexHullShapeSettings_Create(
        jolt_points, (uint32_t)point_count, CONVEX_HULL_TOLERANCE);
    CulvNative_Free(jolt_points);
    if (!hull_settings) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }
    JPH_Shape *shape = (JPH_Shape *)JPH_ConvexHullShapeSettings_CreateShape(hull_settings);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)hull_settings);
    if (!shape) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }

    const uint32_t object_layer =
        request->config.motion_type == MOTION_STATIC ? OBJECT_LAYER_STATIC : OBJECT_LAYER_DYNAMIC;
    return create_body_from_shape(
        world, shape, true, request, (JPH_MotionType)request->config.motion_type, object_layer,
        PHYSICS_WORLD_BODY_SETUP_CONFIGURED, out_handle, out_count, out_native_status);
}

static JPH_Shape *create_compound_shape(PhysicsWorld *world, const PhysicsWorldCompoundPart *parts,
                                        size_t part_count) {
    NATIVE_MUTEX_LOCK(world->jph_trampoline_lock);
    JPH_StaticCompoundShapeSettings *compound_settings = JPH_StaticCompoundShapeSettings_Create();
    if (!compound_settings) {
        NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
        return nullptr;
    }

    for (size_t i = 0; i < part_count; i++) {
        SHADOW_LOCK(&world->shadow_lock);
        JPH_Shape *sub_shape =
            find_or_create_shape_locked(world, parts[i].shape_type, parts[i].shape_params);
        SHADOW_UNLOCK(&world->shadow_lock);
        if (sub_shape) {
            JPH_Vec3 local_position = parts[i].local_position;
            JPH_Quat local_rotation = parts[i].local_rotation;
            JPH_CompoundShapeSettings_AddShape2((JPH_CompoundShapeSettings *)compound_settings,
                                                &local_position, &local_rotation, sub_shape, 0);
        }
    }

    JPH_Shape *shape = (JPH_Shape *)JPH_StaticCompoundShape_Create(compound_settings);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)compound_settings);
    NATIVE_MUTEX_UNLOCK(world->jph_trampoline_lock);
    return shape;
}

PhysicsWorldCreationStatus
physics_world_create_compound_body(PhysicsWorld *world, const PhysicsWorldCompoundPart *parts,
                                   size_t part_count, const PhysicsWorldBodyCreateRequest *request,
                                   BodyHandle *out_handle, size_t *out_count,
                                   PhysicsWorldStatus *out_native_status) {
    if (!world || !parts || part_count == 0 || !request || !out_handle || !out_count ||
        !out_native_status) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_native_status = PHYSICS_WORLD_STATUS_OK;
    JPH_Shape *shape   = create_compound_shape(world, parts, part_count);
    if (!shape) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }

    const uint32_t object_layer =
        request->config.motion_type == MOTION_STATIC ? OBJECT_LAYER_STATIC : OBJECT_LAYER_DYNAMIC;
    return create_body_from_shape(
        world, shape, true, request, (JPH_MotionType)request->config.motion_type, object_layer,
        PHYSICS_WORLD_BODY_SETUP_CREATION_PROPS, out_handle, out_count, out_native_status);
}

PhysicsWorldCreationStatus
physics_world_create_mesh_body(PhysicsWorld *world, const float *vertices, size_t vertex_count,
                               const uint32_t *indices, size_t triangle_count, JPH_RVec3 position,
                               JPH_Quat rotation, uint32_t category, uint32_t mask,
                               uint64_t user_data, BodyHandle *out_handle, size_t *out_count,
                               PhysicsWorldStatus *out_native_status, uint32_t out_bad_indices[3]) {
    if (!world || !vertices || !indices || !out_handle || !out_count || !out_native_status ||
        vertex_count == 0 || triangle_count == 0 || vertex_count > UINT32_MAX ||
        triangle_count > UINT32_MAX) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_native_status = PHYSICS_WORLD_STATUS_OK;
    if (triangle_count > SIZE_MAX / sizeof(JPH_IndexedTriangle) ||
        vertex_count > SIZE_MAX / sizeof(JPH_Vec3)) {
        return PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY;
    }
    JPH_Vec3 *jolt_vertices        = CulvNative_Malloc(vertex_count * sizeof(*jolt_vertices));
    JPH_IndexedTriangle *triangles = CulvNative_Malloc(triangle_count * sizeof(*triangles));
    if (!jolt_vertices || !triangles) {
        CulvNative_Free(jolt_vertices);
        CulvNative_Free(triangles);
        return PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < vertex_count; i++) {
        jolt_vertices[i] = (JPH_Vec3){vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]};
    }
    for (size_t triangle = 0; triangle < triangle_count; triangle++) {
        const uint32_t i1 = indices[triangle * 3];
        const uint32_t i2 = indices[(triangle * 3) + 1];
        const uint32_t i3 = indices[(triangle * 3) + 2];
        if (i1 >= vertex_count || i2 >= vertex_count || i3 >= vertex_count) {
            if (out_bad_indices) {
                out_bad_indices[0] = i1;
                out_bad_indices[1] = i2;
                out_bad_indices[2] = i3;
            }
            CulvNative_Free(jolt_vertices);
            CulvNative_Free(triangles);
            return PHYSICS_WORLD_CREATION_STATUS_INVALID_MESH_INDEX;
        }
        triangles[triangle] = (JPH_IndexedTriangle){
            .i1            = i1,
            .i2            = i2,
            .i3            = i3,
            .materialIndex = 0,
            .userData      = 0,
        };
    }

    JPH_MeshShapeSettings *mesh_settings = JPH_MeshShapeSettings_Create2(
        jolt_vertices, (uint32_t)vertex_count, triangles, (uint32_t)triangle_count);
    CulvNative_Free(jolt_vertices);
    CulvNative_Free(triangles);
    if (!mesh_settings) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }
    JPH_Shape *shape = (JPH_Shape *)JPH_MeshShapeSettings_CreateShape(mesh_settings);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)mesh_settings);
    if (!shape) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }

    const PhysicsWorldBodyCreateRequest request = {
        .position  = position,
        .rotation  = rotation,
        .config    = {.motion_type = MOTION_STATIC},
        .category  = category,
        .mask      = mask,
        .user_data = user_data,
    };
    return create_body_from_shape(world, shape, true, &request, JPH_MotionType_Static,
                                  OBJECT_LAYER_STATIC, PHYSICS_WORLD_BODY_SETUP_NONE, out_handle,
                                  out_count, out_native_status);
}

PhysicsWorldCreationStatus physics_world_create_heightfield_body(
    PhysicsWorld *world, const float *heights, size_t grid_size, JPH_Vec3 scale, JPH_RVec3 position,
    JPH_Quat rotation, float friction, float restitution, uint32_t category, uint32_t mask,
    uint32_t material_id, uint64_t user_data, BodyHandle *out_handle, size_t *out_count,
    PhysicsWorldStatus *out_native_status) {
    if (!world || !heights || !out_handle || !out_count || !out_native_status || grid_size == 0 ||
        grid_size > UINT32_MAX || grid_size > SIZE_MAX / grid_size ||
        grid_size * grid_size > SIZE_MAX / sizeof(float)) {
        return PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT;
    }
    *out_native_status = PHYSICS_WORLD_STATUS_OK;

    JPH_Vec3 offset = {};
    JPH_HeightFieldShapeSettings *heightfield_settings =
        JPH_HeightFieldShapeSettings_Create(heights, &offset, &scale, (uint32_t)grid_size, nullptr);
    if (!heightfield_settings) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }
    JPH_Shape *shape = (JPH_Shape *)JPH_HeightFieldShapeSettings_CreateShape(heightfield_settings);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)heightfield_settings);
    if (!shape) {
        return PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED;
    }

    const PhysicsWorldBodyCreateRequest request = {
        .position = position,
        .rotation = rotation,
        .config = {.friction = friction, .restitution = restitution, .motion_type = MOTION_STATIC},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };
    return create_body_from_shape(world, shape, true, &request, JPH_MotionType_Static,
                                  OBJECT_LAYER_STATIC, PHYSICS_WORLD_BODY_SETUP_MATERIAL_ONLY,
                                  out_handle, out_count, out_native_status);
}
