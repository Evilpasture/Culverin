#include <math.h>
#include "bindings/physics/PhysicsWorld.h"
#include "characters/Character.h" // IWYU pragma: keep
#include "bindings/physics/Constraint.h" // IWYU pragma: keep
#include "physics/PhysicsWorld.h"
#include "physics/PhysicsWorldCreation.h"
#include "physics/PhysicsWorldOps.h"
#include "physics/PhysicsWorldRuntime.h"
#include "bindings/physics/WorldAdapters.h"
#include "core/NativeAlloc.h"
#include "bindings/parsing/ArgIndices.h"
#include "physics/ContactEventDefinitions.h"
#include "core/CompilerSpecifics.h"
#include "bindings/FastBuild.h"
#include "bindings/Getters.h"
#include "math/MathCore.h"
#include "bindings/ModuleState.h"
#include "bindings/physics/PhysicsSync.h"
#include "bindings/PythonSupport.h"
#include "bindings/physics/QueryMethods.h"
#include "ragdolls/Ragdoll.h"
#include "physics/ShadowSync.h"
#include "soft_bodies/SoftBody.h"

// ============================================================================
// Semantic Constants - Magic Number Replacements
// ============================================================================

// Memory and Alignment

// Physics Simulation
static constexpr float DEFAULT_FRAME_TIME    = 1.0f / 60.0f;
static constexpr float DEFAULT_LINEAR_DRAG   = 0.5f;
static constexpr float DEFAULT_ANGULAR_DRAG  = 0.5f;
static constexpr float DEFAULT_FRICTION      = 0.2f;
static constexpr uint32_t NO_ALLOWED_DOFS_OVERRIDE = 0U;

// Collision Filtering
static constexpr uint32_t COLLISION_FILTER_ALL_CATEGORIES = 0xFFFF;
static constexpr uint32_t COLLISION_FILTER_ALL_MASKS      = 0xFFFF;

// Numerical Tolerances

// Array Indices and Counts
static constexpr float RESTITUTION_BUFFER           = 0.5f; // Default restitution/bounce
static constexpr size_t VERTEX_STRIDE_BYTES         = 12;   // 3 floats (x, y, z) * 4 bytes
static constexpr float DEFAULT_BODY_SIZE            = 0.5f;

// Jolt Physics collision masks (all layers/categories)
static constexpr uint32_t JOLT_ALL_LAYER_BITS = 0xFFFF;

// Buffer allocation increments

// --- Lifecycle: Deallocation ---
PyType_DeclareSlot_Status PhysicsWorld_traverse(PhysicsWorldObject *self, visitproc visit,
                                                void *arg) {
    // 1. Visit the type itself (Required for all heap types)
    Py_VISIT(Py_TYPE(self));

    // 2. If you add any other PyObject* members to your struct in the future,
    // you MUST visit them here.

    return 0;
}

PyType_DeclareSlot_Status PhysicsWorld_clear([[maybe_unused]] PhysicsWorldObject *self) {
    // Currently nothing to clear.
    return 0;
}

PyType_DeclareSlot_Void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
    auto tp = Py_TYPE(self);

    // 1. The GC "Safety Shield"
    // Use the check-then-untrack to avoid the 'already untracked' abort
    if (PyObject_GC_IsTracked((PyObject *)self)) {
        PyObject_GC_UnTrack(self);
    }

    // 2. Weakref cleanup
    if (self->weakreflist != nullptr) {
        PyObject_ClearWeakRefs((PyObject *)self);
    }

    // Clean up local parsers
    if (self->parsers != nullptr) {
        culverin_free_world_parsers(self->parsers);
        PyMem_Free(self->parsers);
    }

    // 3. The "Manual" work
    PhysicsWorld_free_members(self);

    // 4. Final destruction
    // For Heap Types, we use the type's free function
    tp->tp_free((PyObject *)self);

    // 5. Release the type itself
    Py_DECREF(tp);
}
// --- Lifecycle: Initialization ---

// Orchestrator function
PyType_DeclareSlot_Status PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                                            PyObject *kwds) {
    if (self->physics.system != nullptr) {
        // Please don't call __init__() again.
        PyErr_SetString(PyExc_RuntimeError,
                        "PhysicsWorld instance has already been initialized and "
                        "cannot be re-initialized.");
        return -1;
    }
    if (self->parsers == nullptr) {
        self->parsers = (WorldParsers *)PyMem_Malloc(sizeof(WorldParsers));
        if (self->parsers == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
        culverin_init_world_parsers(self->parsers);
    }

    PyObject *settings_dict     = nullptr;
    PyObject *bodies_list       = nullptr;
    AUTO_DECREF PyObject *baked = nullptr;
    size_t baked_count          = 0;
    GravityVector gravity;
    WorldSettings settings;

    const void *const restrict targets[WorldInit_COUNT] = {
        [IDX_SETTINGS] = (const void *const restrict)&settings_dict,
        [IDX_BODIES]   = (const void *const restrict)&bodies_list};

    if (!CULV_FAST_PARSE_UNIFIED(args, kwds, nullptr, &self->parsers->WorldInitParser, targets)) {
        return -1;
    }

    // 1. Initial State
    physics_world_init_state(&self->physics);
    self->view_shape[0] = self->view_shape[1] = 0;
    self->view_strides[0] = self->view_strides[1] = 0;

    // 2. Settings & Jolt Init
    if (init_settings(self, settings_dict, &gravity, &settings) < 0) {
        goto fail;
    }

    if (physics_world_init_jolt(&self->physics, settings, gravity) < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to initialize native Jolt world resources.");
        }
        goto fail;
    }

    if (verify_abi_alignment(self->physics.body_interface) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "JoltC ABI mismatch: position precision check failed.");
        goto fail;
    }

    if (!physics_world_attach_contact_listener(&self->physics)) {
        PyErr_NoMemory();
        goto fail;
    }

    // 3. Bake & Buffers
    if (bodies_list && bodies_list != Py_None) {
        PyObject *st_helper = get_culverin_state(PyType_GetModule(Py_TYPE(self)))->helper;
        AUTO_DECREF PyObject *bake_func = PyObject_GetAttrString(st_helper, "bake_scene");
        baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, nullptr);
        if (!baked) {
            goto fail;
        }
        if (get_baked_scene_count(baked, &baked_count) < 0) {
            goto fail;
        }
    }

    if (allocate_buffers(&self->physics, settings.max_bodies, baked_count) < 0) {
        PyErr_NoMemory();
        goto fail;
    }

    // 4. Constraints & Data Loading
    if (allocate_constraints(&self->physics) < 0) {
        PyErr_NoMemory();
        goto fail;
    }

    if (baked && load_baked_scene(self, baked) < 0) {
        goto fail;
    }

    size_t start_idx = atomic_load_explicit(&self->physics.count, memory_order_relaxed);
    initialize_free_slots(&self->physics, start_idx);

    SHADOW_LOCK(&self->physics.shadow_lock);
    culverin_sync_shadow_buffers(&self->physics);
    SHADOW_UNLOCK(&self->physics.shadow_lock);
    return 0;

fail:
    PhysicsWorld_free_members(self);
    return -1;
}
// Translate native status codes into Python exceptions at the binding boundary.
static void set_physics_world_error(PhysicsWorldObject *self, PhysicsWorldStatus status) {
    switch (status) {
    case PHYSICS_WORLD_STATUS_OUT_OF_MEMORY:
        PyErr_NoMemory();
        break;
    case PHYSICS_WORLD_STATUS_LIMIT_REACHED:
        PyErr_Format(PyExc_RuntimeError, "PhysicsWorld limit reached: %u bodies",
                     self->physics.max_jolt_bodies);
        break;
    case PHYSICS_WORLD_STATUS_BUFFER_EXPORTED:
        PyErr_SetString(PyExc_BufferError, "Cannot resize while memoryview is active.");
        break;
    default:
        PyErr_SetString(PyExc_RuntimeError, "Native PhysicsWorld operation failed.");
        break;
    }
}

static PyObject *body_operation_status_result(PhysicsBodyOperationStatus status) {
    switch (status) {
    case PHYSICS_BODY_OPERATION_STATUS_OK:
        Py_RETURN_NONE;
    case PHYSICS_BODY_OPERATION_STATUS_OUT_OF_MEMORY:
        return PyErr_NoMemory();
    case PHYSICS_BODY_OPERATION_STATUS_STALE_HANDLE:
        RAISE_STALE_HANDLE();
        return nullptr;
    case PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT:
        PyErr_SetString(PyExc_RuntimeError, "Invalid native physics body operation");
        return nullptr;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unknown native physics body operation status");
    return nullptr;
}

static PyObject *run_body_operation(PhysicsWorldObject *self,
                                    const PhysicsBodyOperation *operation) {
    PhysicsBodyOperationStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_apply_body_operation(&self->physics, operation);
    Py_END_ALLOW_THREADS return body_operation_status_result(status);
}

static void set_creation_error(PhysicsWorldObject *self, PhysicsWorldCreationStatus status,
                               PhysicsWorldStatus native_status, const char *shape_message,
                               const char *settings_message) {
    switch (status) {
    case PHYSICS_WORLD_CREATION_STATUS_OK:
        return;
    case PHYSICS_WORLD_CREATION_STATUS_BODY_SETTINGS_FAILED:
        if (settings_message) {
            PyErr_SetString(PyExc_RuntimeError, settings_message);
        } else {
            PyErr_NoMemory();
        }
        return;
    case PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR:
        set_physics_world_error(self, native_status);
        return;
    case PHYSICS_WORLD_CREATION_STATUS_SHAPE_FAILED:
        PyErr_SetString(PyExc_RuntimeError,
                        shape_message ? shape_message : "Shape creation failed");
        return;
    case PHYSICS_WORLD_CREATION_STATUS_RAGDOLL_FAILED:
        PyErr_SetString(PyExc_RuntimeError,
                        shape_message ? shape_message : "Jolt failed to create Ragdoll instance");
        return;
    case PHYSICS_WORLD_CREATION_STATUS_OUT_OF_MEMORY:
        PyErr_NoMemory();
        return;
    case PHYSICS_WORLD_CREATION_STATUS_INVALID_MESH_INDEX:
        PyErr_SetString(PyExc_ValueError, "Mesh index out of range");
        return;
    case PHYSICS_WORLD_CREATION_STATUS_INVALID_ARGUMENT:
        PyErr_SetString(PyExc_RuntimeError, "Invalid native PhysicsWorld creation request");
        return;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unknown native PhysicsWorld creation status");
}

static PyObject *snapshot_status_error(PhysicsWorldSnapshotStatus status, size_t world_capacity,
                                       size_t saved_capacity) {
    switch (status) {
    case PHYSICS_WORLD_SNAPSHOT_STATUS_OK:
        return nullptr;
    case PHYSICS_WORLD_SNAPSHOT_STATUS_OUT_OF_MEMORY:
        return PyErr_NoMemory();
    case PHYSICS_WORLD_SNAPSHOT_STATUS_INVALID_ARGUMENT:
        PyErr_SetString(PyExc_RuntimeError, "Invalid PhysicsWorld snapshot operation");
        return nullptr;
    case PHYSICS_WORLD_SNAPSHOT_STATUS_TRUNCATED:
        PyErr_SetString(PyExc_ValueError, "Snapshot buffer truncated or stride mismatch");
        return nullptr;
    case PHYSICS_WORLD_SNAPSHOT_STATUS_CAPACITY_MISMATCH:
        PyErr_Format(PyExc_ValueError, "Capacity mismatch: World is %zu, Snapshot is %zu",
                     world_capacity, saved_capacity);
        return nullptr;
    case PHYSICS_WORLD_SNAPSHOT_STATUS_OVERFLOW:
        PyErr_SetString(PyExc_OverflowError, "Snapshot is too large");
        return nullptr;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unknown PhysicsWorld snapshot status");
    return nullptr;
}

static PyObject *runtime_status_error(PhysicsWorldRuntimeStatus status,
                                      const char *pending_create_message) {
    switch (status) {
    case PHYSICS_WORLD_RUNTIME_STATUS_OK:
        return nullptr;
    case PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_MEMORY:
        return PyErr_NoMemory();
    case PHYSICS_WORLD_RUNTIME_STATUS_STALE_HANDLE:
#if defined(STRICT_HANDLE_ENABLED)
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return nullptr;
#else
        return Py_NewRef(Py_None);
#endif
    case PHYSICS_WORLD_RUNTIME_STATUS_PENDING_CREATE:
        PyErr_SetString(PyExc_RuntimeError, pending_create_message
                                                ? pending_create_message
                                                : "Body has not been flushed to Jolt yet");
        return nullptr;
    case PHYSICS_WORLD_RUNTIME_STATUS_OVERFLOW:
        PyErr_SetString(PyExc_OverflowError, "Body count exceeds Jolt limit");
        return nullptr;
    case PHYSICS_WORLD_RUNTIME_STATUS_INVALID_ARGUMENT:
        PyErr_SetString(PyExc_RuntimeError, "Invalid native PhysicsWorld operation");
        return nullptr;
    case PHYSICS_WORLD_RUNTIME_STATUS_WRONG_BODY_TYPE:
        PyErr_SetString(PyExc_TypeError, "Handle does not belong to a soft body");
        return nullptr;
    case PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_RANGE:
        PyErr_SetString(PyExc_IndexError, "Soft body vertex index is out of bounds");
        return nullptr;
    case PHYSICS_WORLD_RUNTIME_STATUS_MISSING_BUFFER:
        PyErr_SetString(PyExc_RuntimeError, "Soft body shadow buffer missing");
        return nullptr;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unknown native PhysicsWorld operation status");
    return nullptr;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    uint64_t h_raw;
    float x;
    float y;
    float z;
    const void *const restrict targets[Vec3_COUNT] = {[IDX_V3_H] =
                                                          (const void *const restrict)&h_raw,
                                                      [IDX_V3_X] = (const void *const restrict)&x,
                                                      [IDX_V3_Y] = (const void *const restrict)&y,
                                                      [IDX_V3_Z] = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ImpulseParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Impulse");

    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_IMPULSE,
        .handle = (BodyHandle)h_raw,
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_impulse_at(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    uint64_t h_raw;
    float ix;
    float iy;
    float iz;
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    const void *const restrict targets[ImpAt_COUNT] = {
        [IDX_IMPAT_H]  = (const void *const restrict)&h_raw,
        [IDX_IMPAT_IX] = (const void *const restrict)&ix,
        [IDX_IMPAT_IY] = (const void *const restrict)&iy,
        [IDX_IMPAT_IZ] = (const void *const restrict)&iz,
        [IDX_IMPAT_PX] = (const void *const restrict)&px,
        [IDX_IMPAT_PY] = (const void *const restrict)&py,
        [IDX_IMPAT_PZ] = (const void *const restrict)&pz};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ImpulseAtParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(ix, iy, iz, "Impulse");
    VALIDATE_FINITE_VEC3(px, py, pz, "Impulse position");

    const PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_APPLY_IMPULSE_AT,
        .handle   = (BodyHandle)h_raw,
        .position = {px, py, pz},
        .vector   = {ix, iy, iz},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_angular_impulse(PhysicsWorldObject *self,
                                                             PyObject *const *args, size_t nargsf,
                                                             PyObject *kwnames) {
    uint64_t h_raw;
    float x;
    float y;
    float z;
    const void *const restrict targets[Vec3_COUNT] = {[IDX_V3_H] =
                                                          (const void *const restrict)&h_raw,
                                                      [IDX_V3_X] = (const void *const restrict)&x,
                                                      [IDX_V3_Y] = (const void *const restrict)&y,
                                                      [IDX_V3_Z] = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->AngImpulseParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Angular impulse");

    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE,
        .handle = (BodyHandle)h_raw,
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_force(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    const void *const restrict targets[Vec3_COUNT] = {[IDX_V3_H] =
                                                          (const void *const restrict)&h_raw,
                                                      [IDX_V3_X] = (const void *const restrict)&x,
                                                      [IDX_V3_Y] = (const void *const restrict)&y,
                                                      [IDX_V3_Z] = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ForceParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Force");

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_FORCE,
        .handle = (BodyHandle)h_raw,
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_torque(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    const void *const restrict targets[Vec3_COUNT] = {[IDX_V3_H] =
                                                          (const void *const restrict)&h_raw,
                                                      [IDX_V3_X] = (const void *const restrict)&x,
                                                      [IDX_V3_Y] = (const void *const restrict)&y,
                                                      [IDX_V3_Z] = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->TorqueParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Torque");

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_TORQUE,
        .handle = (BodyHandle)h_raw,
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_gravity(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    float x;
    float y;
    float z;
    const void *const restrict targets[XYZ_COUNT] = {
        [IDX_XYZ_X] = (const void *const restrict)&x,
        [IDX_XYZ_Y] = (const void *const restrict)&y,
        [IDX_XYZ_Z] = (const void *const restrict)&z,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->GravityParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "Gravity");

    const JPH_Vec3 gravity = {x, y, z};
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_set_gravity(&self->physics, gravity);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_gravity(PhysicsWorldObject *self,
                                                   PyObject *ignored [[maybe_unused]]) {
    JPH_Vec3 gravity;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_get_gravity(&self->physics, &gravity);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return CULV_PY_BUILD_TUPLE(CULV_PY_BUILD_VALUE(gravity.x), CULV_PY_BUILD_VALUE(gravity.y),
                               CULV_PY_BUILD_VALUE(gravity.z));
}

PyCFunction_DeclareMethod PhysicsWorld_get_body_stats(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->HOnlyParser, targets)) {
        return nullptr;
    }

    PhysicsBodyStats stats;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_body_stats(&self->physics, (BodyHandle)h_raw, &stats);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }

    return CULV_PY_BUILD_TUPLE(
        CULV_PY_BUILD_TUPLE(stats.position.x, stats.position.y, stats.position.z),
        CULV_PY_BUILD_TUPLE(stats.rotation.x, stats.rotation.y, stats.rotation.z, stats.rotation.w),
        CULV_PY_BUILD_TUPLE(stats.linear_velocity.x, stats.linear_velocity.y,
                            stats.linear_velocity.z));
}
PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    uint64_t h_raw;
    double surface_y;
    float buoyancy                                 = 1.0f;
    float lin_drag                                 = DEFAULT_LINEAR_DRAG;
    float ang_drag                                 = DEFAULT_ANGULAR_DRAG;
    float dt                                       = DEFAULT_FRAME_TIME;
    PyObject *o_vel                                = nullptr;
    const void *const restrict targets[Buoy_COUNT] = {
        [IDX_BUOY_HANDLE]    = (const void *const restrict)&h_raw,
        [IDX_BUOY_SURFACE_Y] = (const void *const restrict)&surface_y,
        [IDX_BUOY_BUOYANCY]  = (const void *const restrict)&buoyancy,
        [IDX_BUOY_LIN_DRAG]  = (const void *const restrict)&lin_drag,
        [IDX_BUOY_ANG_DRAG]  = (const void *const restrict)&ang_drag,
        [IDX_BUOY_DT]        = (const void *const restrict)&dt,
        [IDX_BUOY_VEL]       = (const void *const restrict)&o_vel,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->BuoyParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_FLOAT(buoyancy, "buoyancy");
    VALIDATE_FINITE_FLOAT(lin_drag, "linear drag");
    VALIDATE_FINITE_FLOAT(ang_drag, "angular drag");
    VALIDATE_FINITE_FLOAT(dt, "dt");

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    if (o_vel && o_vel != Py_None) {
        if (!parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
            return nullptr;
        }
        VALIDATE_FINITE_VEC3(vx, vy, vz, "fluid velocity");
    }

    const JPH_Vec3 fluid_velocity = {vx, vy, vz};
    bool submerged = false;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = physics_world_apply_buoyancy(&self->physics, (BodyHandle)h_raw,
                                          (JPH_Real)surface_y, buoyancy, lin_drag, ang_drag,
                                          fluid_velocity, dt, &submerged);
    Py_END_ALLOW_THREADS
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyBool_FromLong((int)submerged);
}

PyCFunction_DeclareMethod PhysicsWorld_apply_buoyancy_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    PyObject *o_handles = nullptr;
    PyObject *o_vel = nullptr;
    JPH_Real surface_y                                  = 0.0;
    float buoyancy                                      = 1.0f;
    float lin_drag                                      = DEFAULT_LINEAR_DRAG;
    float ang_drag                                      = DEFAULT_ANGULAR_DRAG;
    float dt                                            = DEFAULT_FRAME_TIME;
    const void *const restrict targets[BatchBuoy_COUNT] = {
        [IDX_BBUOY_HANDLES]   = (const void *const restrict)&o_handles,
        [IDX_BBUOY_SURFACE_Y] = (const void *const restrict)&surface_y,
        [IDX_BBUOY_BUOYANCY]  = (const void *const restrict)&buoyancy,
        [IDX_BBUOY_LIN_DRAG]  = (const void *const restrict)&lin_drag,
        [IDX_BBUOY_ANG_DRAG]  = (const void *const restrict)&ang_drag,
        [IDX_BBUOY_DT]        = (const void *const restrict)&dt,
        [IDX_BBUOY_VEL]       = (const void *const restrict)&o_vel,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->BatchBuoyParser, targets)) {
        return nullptr;
    }

    Py_buffer handles_view;
    if (PyObject_GetBuffer(o_handles, &handles_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (UNLIKELY(handles_view.itemsize != 8 && handles_view.len % 8 != 0)) {
        PyBuffer_Release(&handles_view);
        PyErr_SetString(PyExc_ValueError, "Handle buffer must be uint64 array");
        return nullptr;
    }

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    if (o_vel && o_vel != Py_None && !parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
        PyBuffer_Release(&handles_view);
        return nullptr;
    }

    const size_t handle_count = (size_t)handles_view.len / sizeof(BodyHandle);
    if (handle_count == 0) {
        PyBuffer_Release(&handles_view);
        Py_RETURN_NONE;
    }

    BodyHandle *handles = (BodyHandle *)CULV_RAW_MALLOC(handle_count * sizeof(BodyHandle));
    if (!handles) {
        PyBuffer_Release(&handles_view);
        return PyErr_NoMemory();
    }
    memcpy(handles, handles_view.buf, handle_count * sizeof(BodyHandle));
    PyBuffer_Release(&handles_view);

    const JPH_Vec3 fluid_velocity = {vx, vy, vz};
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_apply_buoyancy_batch(&self->physics, handles, handle_count, surface_y,
                                           buoyancy, lin_drag, ang_drag, fluid_velocity, dt);
    Py_END_ALLOW_THREADS CULV_RAW_FREE(handles);
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_save_state(PhysicsWorldObject *self,
                                                  PyObject *unused [[maybe_unused]]) {
    uint8_t *snapshot = nullptr;
    size_t snapshot_size = 0;
    PhysicsWorldSnapshotStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = physics_world_save_snapshot(&self->physics, &snapshot, &snapshot_size);
    Py_END_ALLOW_THREADS
    if (status != PHYSICS_WORLD_SNAPSHOT_STATUS_OK) {
        return snapshot_status_error(status, 0, 0);
    }
    if (snapshot_size > (size_t)PY_SSIZE_T_MAX) {
        CulvNative_Free(snapshot);
        PyErr_SetString(PyExc_OverflowError, "Snapshot is too large for a Python bytes object");
        return nullptr;
    }

    PyObject *bytes = PyBytes_FromStringAndSize((const char *)snapshot,
                                                (Py_ssize_t)snapshot_size);
    CulvNative_Free(snapshot);
    return bytes;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_load_state(PhysicsWorldObject *self, PyObject *const *args,
                                                  Py_ssize_t nargs, PyObject *kwnames) {
    PyObject *state_obj = nullptr;
    const void *const restrict targets[LoadState_COUNT] = {
        [IDX_LS_STATE] = (const void *const restrict)&state_obj,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->LoadStateParser, targets)) {
        return nullptr;
    }

    Py_buffer view = {0};
    if (PyObject_GetBuffer(state_obj, &view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    const size_t data_size = (size_t)view.len;
    uint8_t *snapshot = CulvNative_Malloc(data_size ? data_size : 1);
    if (!snapshot) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    if (data_size != 0) {
        memcpy(snapshot, view.buf, data_size);
    }
    PyBuffer_Release(&view);

    size_t saved_count = 0;
    size_t world_capacity = 0;
    size_t saved_capacity = 0;
    PhysicsWorldSnapshotStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = physics_world_load_snapshot(&self->physics, snapshot, data_size, &saved_count,
                                         &world_capacity, &saved_capacity);
    Py_END_ALLOW_THREADS
    CulvNative_Free(snapshot);
    if (status != PHYSICS_WORLD_SNAPSHOT_STATUS_OK) {
        return snapshot_status_error(status, world_capacity, saved_capacity);
    }

    self->view_shape[0] = (Py_ssize_t)saved_count;
    Py_RETURN_NONE;
}

typedef struct {
    PyThreadState *tstate;
    bool once;
} PyThreadCtx;

[[gnu::always_inline]]
static inline void internal_auto_gil_restore(PyThreadCtx *ctx) {
    if (ctx->tstate) {
        PyEval_RestoreThread(ctx->tstate);
        ctx->tstate = nullptr;
    }
}

#define WITH_ALLOW_THREADS                                                                         \
    for ([[gnu::cleanup(internal_auto_gil_restore)]] PyThreadCtx _ctx = {PyEval_SaveThread(),      \
                                                                         true};                    \
         _ctx.once; _ctx.once                                         = false)

[[gnu::flatten]]
PyCFunction_DeclareMethod PhysicsWorld_step(PhysicsWorldObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
    float dt                                       = DEFAULT_FRAME_TIME;
    const void *const restrict targets[Step_COUNT] = {
        [IDX_STEP_DT] = (const void *const restrict)&dt,
    };

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->StepParser, targets)) {
        return nullptr;
    }

    VALIDATE_FINITE_FLOAT(dt, "dt");

    // --- PHASE 0: RE-ENTRANCY GUARD ---
    if (atomic_load_explicit(&self->physics.is_stepping, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "Concurrent step detected.");
        return nullptr;
    }

    // --- PHASE 1: SHADOW STATE LOCK-DOWN ---
    SHADOW_LOCK(&self->physics.shadow_lock);

    // ANTI-STARVATION: Yield to waiting Python threads (Getters/Mutators)
    // This is necessary for performance.
    while (atomic_load_explicit(&self->physics.waiting_threads, memory_order_relaxed) > 0) {
        SHADOW_UNLOCK(&self->physics.shadow_lock);
        WITH_ALLOW_THREADS { culverin_yield(); }
        SHADOW_LOCK(&self->physics.shadow_lock);
    }

    atomic_thread_fence(memory_order_acquire);

    // Raise flags
    atomic_store_explicit(&self->physics.is_stepping, true, memory_order_relaxed);
    atomic_store_explicit(&self->physics.step_requested, true, memory_order_relaxed);

    // Drain in-flight queries
    if (atomic_load_explicit(&self->physics.active_queries, memory_order_acquire) > 0) {
        SHADOW_UNLOCK(&self->physics.shadow_lock);
        WITH_ALLOW_THREADS {
            NATIVE_MUTEX_LOCK(self->physics.step_sync.mutex);
            while (atomic_load_explicit(&self->physics.active_queries, memory_order_relaxed) > 0) {
                NATIVE_COND_WAIT(self->physics.step_sync.cond, self->physics.step_sync.mutex);
            }
            NATIVE_MUTEX_UNLOCK(self->physics.step_sync.mutex);
        }
        SHADOW_LOCK(&self->physics.shadow_lock);
    }

    // Transfer the active command queue to the native step core.
    size_t captured_count = 0;
    PhysicsCommand *captured_queue =
        physics_world_detach_command_queue(&self->physics, &captured_count);

    atomic_store_explicit(&self->physics.contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->physics.shadow_lock);

    // --- PHASE 2: NATIVE JOLT CRUNCH (GIL Released) ---
    WITH_ALLOW_THREADS {
        physics_world_step_native(&self->physics, dt, captured_queue, captured_count);
    }

    // --- PHASE 3: FINALIZATION ---
    SHADOW_LOCK(&self->physics.shadow_lock);

    const size_t final_count = physics_world_finalize_step(&self->physics, dt);
    self->view_shape[0]      = (Py_ssize_t)final_count;

    SHADOW_UNLOCK(&self->physics.shadow_lock);

    Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_create_convex_hull(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {

    // 1. FAST PARSE
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_points   = nullptr;
    int motion_type      = 2;
    float mass           = -1.0f;
    float friction       = DEFAULT_FRICTION;
    float restitution    = 0.0f;
    uint64_t user_data   = 0;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;

    const void *const restrict targets[HC_COUNT] = {
        [IDX_HC_POS]       = (void *const restrict)&o_pos,
        [IDX_HC_ROT]       = (void *const restrict)&o_rot,
        [IDX_HC_DATA]      = (void *const restrict)&o_points,
        [IDX_HC_MOTION]    = (void *const restrict)&motion_type,
        [IDX_HC_MASS]      = (void *const restrict)&mass,
        [IDX_HC_USER_DATA] = (void *const restrict)&user_data,
        [IDX_HC_SENSOR]    = (void *const restrict)&is_sensor,
        [IDX_HC_CAT]       = (void *const restrict)&category,
        [IDX_HC_MASK]      = (void *const restrict)&mask,
        [IDX_HC_MAT_ID]    = (void *const restrict)&material_id,
        [IDX_HC_FRIC]      = (void *const restrict)&friction,
        [IDX_HC_REST]      = (void *const restrict)&restitution,
        [IDX_HC_CCD]       = (void *const restrict)&use_ccd};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ConvexHullParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR/POINT EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");

    Py_buffer points_view;
    if (PyObject_GetBuffer(o_points, &points_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    size_t num_points = points_view.len / VERTEX_STRIDE_BYTES;
    if (UNLIKELY(num_points < 3)) {
        PyBuffer_Release(&points_view);
        return PyErr_Format(PyExc_ValueError, "Need >= 3 points");
    }

    // 3. Native shape creation and world staging (the buffer remains pinned).
    const PhysicsWorldBodyCreateRequest request = {
        .position    = {px, py, pz},
        .rotation    = {rx, ry, rz, rw},
        .config      = {mass, friction, restitution, (int)is_sensor, (int)use_ccd, motion_type,
                        NO_ALLOWED_DOFS_OVERRIDE},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_create_convex_hull(&self->physics, (const float *)points_view.buf, num_points,
                                         &request, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS PyBuffer_Release(&points_view);

    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, "Convex Hull build failed",
                           "Failed to create BodySettings");
        return nullptr;
    }
    self->view_shape[0] = (Py_ssize_t)body_count;
    return PyLong_FromUnsignedLongLong(handle);
}

// Parse Python compound-part objects into a native request buffer.
static PhysicsWorldCompoundPart *parse_compound_parts(PyObject *parts, size_t *out_count) {
    if (!PyList_Check(parts)) {
        PyErr_SetString(PyExc_TypeError, "Compound parts must be a list");
        return nullptr;
    }

    const Py_ssize_t part_count = PyList_GET_SIZE(parts);
    if (part_count == 0) {
        PyErr_SetString(PyExc_ValueError, "Compound shape must have at least one part");
        return nullptr;
    }
    if ((size_t)part_count > SIZE_MAX / sizeof(PhysicsWorldCompoundPart)) {
        PyErr_NoMemory();
        return nullptr;
    }

    PhysicsWorldCompoundPart *buffer =
        CULV_RAW_MALLOC((size_t)part_count * sizeof(PhysicsWorldCompoundPart));
    if (!buffer) {
        PyErr_NoMemory();
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < part_count; i++) {
        PyObject *item = PyList_GET_ITEM(parts, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) < 4) {
            CULV_RAW_FREE(buffer);
            PyErr_Format(PyExc_ValueError, "Part %zd must be a tuple(pos, rot, type, size)", i);
            return nullptr;
        }

        PyObject *part_position = PyTuple_GET_ITEM(item, 0);
        PyObject *part_rotation = PyTuple_GET_ITEM(item, 1);
        const long type         = PyLong_AsLong(PyTuple_GET_ITEM(item, 2));
        PyObject *part_size     = PyTuple_GET_ITEM(item, 3);
        if (PyErr_Occurred()) {
            CULV_RAW_FREE(buffer);
            return nullptr;
        }

        buffer[i] = (PhysicsWorldCompoundPart){.shape_type = (int)type};
        if (PyTuple_Check(part_position) && PyTuple_GET_SIZE(part_position) == 3) {
            buffer[i].local_position.x =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_position, 0));
            buffer[i].local_position.y =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_position, 1));
            buffer[i].local_position.z =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_position, 2));
        }
        if (PyTuple_Check(part_rotation) && PyTuple_GET_SIZE(part_rotation) == 4) {
            buffer[i].local_rotation.x =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 0));
            buffer[i].local_rotation.y =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 1));
            buffer[i].local_rotation.z =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 2));
            buffer[i].local_rotation.w =
                (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 3));
        } else {
            buffer[i].local_rotation.w = 1.0f;
        }
        if (PyTuple_Check(part_size)) {
            const Py_ssize_t size_count = PyTuple_GET_SIZE(part_size);
            for (Py_ssize_t j = 0; j < 4 && j < size_count; j++) {
                buffer[i].shape_params[j] = (float)PyFloat_AsDouble(PyTuple_GET_ITEM(part_size, j));
            }
        } else if (PyFloat_Check(part_size) || PyLong_Check(part_size)) {
            buffer[i].shape_params[0] = (float)PyFloat_AsDouble(part_size);
        }
    }

    if (PyErr_Occurred()) {
        CULV_RAW_FREE(buffer);
        return nullptr;
    }
    *out_count = (size_t)part_count;
    return buffer;
}

// Orchestrator
PyCFunction_DeclareMethod PhysicsWorld_create_compound_body(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_parts    = nullptr;
    int motion_type      = 2;
    float mass           = -1.0f;
    float friction       = DEFAULT_FRICTION;
    float restitution    = 0.0f;
    uint64_t user_data   = 0;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;

    const void *const restrict targets[HC_COUNT] = {
        [IDX_HC_POS]       = (const void *const restrict)&o_pos,
        [IDX_HC_ROT]       = (const void *const restrict)&o_rot,
        [IDX_HC_DATA]      = (const void *const restrict)&o_parts,
        [IDX_HC_MOTION]    = (const void *const restrict)&motion_type,
        [IDX_HC_MASS]      = (const void *const restrict)&mass,
        [IDX_HC_USER_DATA] = (const void *const restrict)&user_data,
        [IDX_HC_SENSOR]    = (const void *const restrict)&is_sensor,
        [IDX_HC_CAT]       = (const void *const restrict)&category,
        [IDX_HC_MASK]      = (const void *const restrict)&mask,
        [IDX_HC_MAT_ID]    = (const void *const restrict)&material_id,
        [IDX_HC_FRIC]      = (const void *const restrict)&friction,
        [IDX_HC_REST]      = (const void *const restrict)&restitution,
        [IDX_HC_CCD]       = (const void *const restrict)&use_ccd};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->CompoundParser, targets)) {
        return nullptr;
    }

    // 2. EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }
    if (UNLIKELY(!PyList_Check(o_parts))) {
        PyErr_SetString(PyExc_TypeError, "'parts' must be list");
        return nullptr;
    }

    // 3. Parse the Python list; native code builds the shape and stages the body.
    size_t part_count;
    PhysicsWorldCompoundPart *parts = parse_compound_parts(o_parts, &part_count);
    if (!parts) {
        return nullptr;
    }
    const PhysicsWorldBodyCreateRequest request = {
        .position    = {px, py, pz},
        .rotation    = {rx, ry, rz, rw},
        .config      = {mass, friction, restitution, (int)is_sensor, (int)use_ccd, motion_type,
                        NO_ALLOWED_DOFS_OVERRIDE},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_create_compound_body(
        &self->physics, parts, part_count, &request, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS CULV_RAW_FREE(parts);

    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, "Failed to create compound shape",
                           "Failed to create compound BodySettings");
        return nullptr;
    }
    self->view_shape[0] = (Py_ssize_t)body_count;
    return PyLong_FromUnsignedLongLong(handle);
}

// Main Orchestrator
PyCFunction_DeclareMethod PhysicsWorld_create_body(PhysicsWorldObject *self, PyObject *const *args,
                                                   size_t nargsf, PyObject *kwnames) {
    auto nargs = PyVectorcall_NARGS(nargsf);

    // 1. DEFAULT VALUES
    JPH_Real px           = 0.0;
    JPH_Real py           = 0.0;
    JPH_Real pz           = 0.0;
    float rx              = 0.0f;
    float ry              = 0.0f;
    float rz              = 0.0f;
    float rw              = 1.0f;
    float mass            = -1.0f;
    float friction        = -1.0f;
    float restitution     = -1.0f;
    int shape_type        = 0;
    int motion_type       = 2;
    uint32_t category     = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask         = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id  = 0;
    uint32_t allowed_dofs = NO_ALLOWED_DOFS_OVERRIDE;
    uint64_t user_data    = 0;
    bool is_sensor        = false;
    bool use_ccd          = false;
    PyObject *o_pos       = nullptr;
    PyObject *o_rot       = nullptr;
    PyObject *o_size      = nullptr;

    // 2. TARGET MAPPING (Explicitly mapped via Enum)
    // Using explicit indices [IDX_...] makes this reorder-proof.
    const void *const restrict targets[Body_COUNT] = {
        [IDX_POS]          = (const void *const restrict)&o_pos,
        [IDX_ROT]          = (const void *const restrict)&o_rot,
        [IDX_SIZE]         = (const void *const restrict)&o_size,
        [IDX_SHAPE]        = (const void *const restrict)&shape_type,
        [IDX_MOTION]       = (const void *const restrict)&motion_type,
        [IDX_USER_DATA]    = (const void *const restrict)&user_data,
        [IDX_SENSOR]       = (const void *const restrict)&is_sensor,
        [IDX_MASS]         = (const void *const restrict)&mass,
        [IDX_CAT]          = (const void *const restrict)&category,
        [IDX_MASK]         = (const void *const restrict)&mask,
        [IDX_FRIC]         = (const void *const restrict)&friction,
        [IDX_REST]         = (const void *const restrict)&restitution,
        [IDX_MAT]          = (const void *const restrict)&material_id,
        [IDX_CCD]          = (const void *const restrict)&use_ccd,
        [IDX_ALLOWED_DOFS] = (const void *const restrict)&allowed_dofs,
    };

    // 3. THE FAST PARSE
    // We pass the global 'BodyParser' defined in the God Init helper
    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->BodyParser, targets)) {
        return nullptr;
    }

    // 4. CONVERT COMPLEX TYPES
    if (o_pos && o_pos != Py_None) {
        if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
            return nullptr; // PyErr already set
        }
        // GUARD: This is the exact point test_numerical_stability checks
        VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    }

    if (o_rot && o_rot != Py_None) {
        if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
            return nullptr;
        }
        VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");
    }

    // Validation
    if (shape_type == 4 && motion_type != 0) {
        return PyErr_Format(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
    }

    // GUARD: Floats (Only check if they aren't the 'unset' -1.0 default)
    if (mass != -1.0f) {
        VALIDATE_FINITE_FLOAT(mass, "mass");
    }
    if (friction != -1.0f) {
        VALIDATE_FINITE_FLOAT(friction, "friction");
    }
    if (restitution != -1.0f) {
        VALIDATE_FINITE_FLOAT(restitution, "restitution");
    }

    // Resolve defaults and retain parsed size data at the Python boundary.
    float resolved_friction;
    float resolved_restitution;
    physics_world_resolve_material(&self->physics, material_id, friction, restitution,
                                   &resolved_friction, &resolved_restitution);
    float size[4] = {DEFAULT_BODY_SIZE, DEFAULT_BODY_SIZE, DEFAULT_BODY_SIZE, 0.0f};
    parse_body_size(o_size, size);
    VALIDATE_FINITE_VEC4(size[0], size[1], size[2], size[3], "Shape size");

    const PhysicsWorldBodyCreateRequest request = {
        .shape_type   = shape_type,
        .shape_params = {size[0], size[1], size[2], size[3]},
        .position     = {px, py, pz},
        .rotation     = {rx, ry, rz, rw},
        .config      = {mass, resolved_friction, resolved_restitution, (int)is_sensor, (int)use_ccd,
                        motion_type, allowed_dofs},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };

    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_create_body(&self->physics, &request, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS

        if (status == PHYSICS_WORLD_CREATION_STATUS_BODY_SETTINGS_FAILED) {
        return PyErr_Format(PyExc_RuntimeError, "Failed to create BodySettings");
    }
    if (status == PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR) {
        set_physics_world_error(self, native_status);
        return nullptr;
    }
    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid native body creation request");
        return nullptr;
    }

    // Update exported buffer shape only after native body data is stable.
    self->view_shape[0] = (Py_ssize_t)body_count;
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod PhysicsWorld_create_bodies_batch(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {

    // 1. FAST PARSE (Zero Lock Contention)
    PyObject *py_pos   = nullptr;
    PyObject *py_sizes = nullptr;
    int shape_type     = 0;
    int motion_type    = 2;

    const void *const restrict targets[BatchCreate_COUNT] = {
        [IDX_BC_POSITIONS] = (const void *const restrict)&py_pos,
        [IDX_BC_SIZES]     = (const void *const restrict)&py_sizes,
        [IDX_BC_SHAPE]     = (const void *const restrict)&shape_type,
        [IDX_BC_MOTION]    = (const void *const restrict)&motion_type};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->BatchCreateParser, targets)) {
        return nullptr;
    }

    if (!PyList_Check(py_pos) || !PyList_Check(py_sizes)) {
        return PyErr_Format(PyExc_TypeError, "Inputs must be lists");
    }

    const Py_ssize_t batch_count = PyList_GET_SIZE(py_pos);
    if (PyList_GET_SIZE(py_sizes) != batch_count) {
        return PyErr_Format(PyExc_ValueError, "List length mismatch");
    }
    if (batch_count == 0) {
        return PyList_New(0);
    }

    // 2. Parse list contents into native buffers before releasing the GIL.
    const size_t item_size =
        sizeof(PosStride) + sizeof(ShapeParams) + sizeof(BodyHandle) + sizeof(PyObject *);
    if ((size_t)batch_count > SIZE_MAX / item_size) {
        return PyErr_Format(PyExc_OverflowError, "Batch is too large");
    }
    void *arena = CULV_RAW_MALLOC((size_t)batch_count * item_size);
    if (UNLIKELY(!arena)) {
        return PyErr_NoMemory();
    }

    PosStride *positions = arena;
    ShapeParams *sizes   = (ShapeParams *)(positions + batch_count);
    BodyHandle *handles  = (BodyHandle *)(sizes + batch_count);
    PyObject **results   = (PyObject **)(handles + batch_count);
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (!parse_py_vec3(PyList_GET_ITEM(py_pos, i), &positions[i])) {
            CULV_RAW_FREE(arena);
            return nullptr;
        }
        parse_body_size(PyList_GET_ITEM(py_sizes, i), sizes[i].p);
        if (PyErr_Occurred()) {
            CULV_RAW_FREE(arena);
            return nullptr;
        }
    }

    size_t body_count                = 0;
    PhysicsWorldStatus native_status = PHYSICS_WORLD_STATUS_OK;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_create_bodies_batch(
        &self->physics, positions, sizes, (size_t)batch_count, shape_type, motion_type, handles,
        &body_count, &native_status);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        CULV_RAW_FREE(arena);
        if (status == PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR &&
            native_status == PHYSICS_WORLD_STATUS_LIMIT_REACHED) {
            PyErr_SetString(PyExc_RuntimeError, "Batch exceeds world limit");
        } else {
            set_creation_error(self, status, native_status, "Batch shape creation failed", nullptr);
        }
        return nullptr;
    }
    self->view_shape[0] = (Py_ssize_t)body_count;

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        results[i] = handles[i] ? CULV_PY_BUILD_VALUE(handles[i]) : CULV_PY_BUILD_VALUE(nullptr);
    }
    PyObject *result_list = culv_py_build_pack_list((size_t)batch_count, results);
    CULV_RAW_FREE(arena);
    return result_list;
}

/**
 * Main Orchestrator
 */

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    // 1. FAST PARSE
    PyObject *o_pos     = nullptr;
    PyObject *o_rot     = nullptr;
    PyObject *o_verts   = nullptr;
    PyObject *o_indices = nullptr;
    uint64_t user_data  = 0;
    uint32_t cat        = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask       = COLLISION_FILTER_ALL_MASKS;

    const void *const restrict targets[Mesh_COUNT] = {
        [IDX_MSH_POS]       = (const void *const restrict)&o_pos,
        [IDX_MSH_ROT]       = (const void *const restrict)&o_rot,
        [IDX_MSH_VERTS]     = (const void *const restrict)&o_verts,
        [IDX_MSH_INDICES]   = (const void *const restrict)&o_indices,
        [IDX_MSH_USER_DATA] = (const void *const restrict)&user_data,
        [IDX_MSH_CAT]       = (const void *const restrict)&cat,
        [IDX_MSH_MASK]      = (const void *const restrict)&mask};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->MeshParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR/QUAT EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    Py_buffer v_view = {0};
    Py_buffer i_view = {0};
    if (PyObject_GetBuffer(o_verts, &v_view, PyBUF_SIMPLE) != 0 ||
        PyObject_GetBuffer(o_indices, &i_view, PyBUF_SIMPLE) != 0) {
        if (v_view.buf) {
            PyBuffer_Release(&v_view);
        }
        return nullptr;
    }

    if (UNLIKELY(v_view.len % VERTEX_STRIDE_BYTES != 0 || i_view.len % VERTEX_STRIDE_BYTES != 0)) {
        PyBuffer_Release(&v_view);
        PyBuffer_Release(&i_view);
        PyErr_SetString(PyExc_ValueError, "Buffer size mismatch");
        return nullptr;
    }

    // 3. Native mesh shape construction and world staging (buffers stay pinned).
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status = PHYSICS_WORLD_STATUS_OK;
    PhysicsWorldCreationStatus status;
    uint32_t bad_indices[3]       = {};
    const size_t vertex_count     = (size_t)(v_view.len / VERTEX_STRIDE_BYTES);
    const size_t triangle_count   = (size_t)(i_view.len / VERTEX_STRIDE_BYTES);
    Py_BEGIN_ALLOW_THREADS status = physics_world_create_mesh_body(
        &self->physics, (const float *)v_view.buf, vertex_count, (const uint32_t *)i_view.buf,
        triangle_count, (JPH_RVec3){px, py, pz}, (JPH_Quat){rx, ry, rz, rw}, cat, mask, user_data,
        &handle, &body_count, &native_status, bad_indices);
    Py_END_ALLOW_THREADS

        PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    if (status == PHYSICS_WORLD_CREATION_STATUS_INVALID_MESH_INDEX) {
        return PyErr_Format(PyExc_ValueError, "Mesh index out of range: %u/%u/%u >= %zu",
                            bad_indices[0], bad_indices[1], bad_indices[2], vertex_count);
    }
    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status,
                           "Jolt Mesh BVH build failed (Triangle data degenerate?)", nullptr);
        return nullptr;
    }
    self->view_shape[0] = (Py_ssize_t)body_count;
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_body(PhysicsWorldObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {
    uint64_t raw_handle;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&raw_handle};
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->DestroyParser, targets)) {
        return nullptr;
    }

    PhysicsBodyOperationStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = physics_world_queue_body_destroy(&self->physics, (BodyHandle)raw_handle);
    Py_END_ALLOW_THREADS
    return body_operation_status_result(status);
}

PyCFunction_DeclareMethod PhysicsWorld_destroy_bodies_batch(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    PyObject *py_handles = nullptr;
    const void *const restrict targets[BatchDestroy_COUNT] = {
        [IDX_BD_HANDLES] = (const void *const restrict)&py_handles};
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->BatchDestroyParser, targets)) {
        return nullptr;
    }

    PyObject *sequence = PySequence_Fast(py_handles, "handles must be a sequence");
    if (!sequence) {
        return nullptr;
    }
    const Py_ssize_t input_count = PySequence_Fast_GET_SIZE(sequence);
    if (input_count <= 0) {
        Py_DECREF(sequence);
        Py_RETURN_NONE;
    }
    if ((size_t)input_count > SIZE_MAX / sizeof(BodyHandle)) {
        Py_DECREF(sequence);
        return PyErr_NoMemory();
    }

    BodyHandle *handles = CULV_RAW_MALLOC((size_t)input_count * sizeof(BodyHandle));
    if (!handles) {
        Py_DECREF(sequence);
        return PyErr_NoMemory();
    }
    PyObject **items = PySequence_Fast_ITEMS(sequence);
    size_t handle_count = 0;
    for (Py_ssize_t i = 0; i < input_count; i++) {
        const uint64_t raw_handle = PyLong_AsUnsignedLongLong(items[i]);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            continue;
        }
        handles[handle_count++] = (BodyHandle)raw_handle;
    }
    Py_DECREF(sequence);

    PhysicsBodyOperationStatus status;
    Py_BEGIN_ALLOW_THREADS
    status = physics_world_queue_body_destroy_batch(&self->physics, handles, handle_count);
    Py_END_ALLOW_THREADS
    CULV_RAW_FREE(handles);
    return body_operation_status_result(status);
}

PyCFunction_DeclareMethod PhysicsWorld_set_position(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;
    const void *const restrict targets[SetPos_COUNT] = {
        [IDX_SETPOS_HANDLE] = (const void *const restrict)&h_raw,
        [IDX_SETPOS_X]      = (const void *const restrict)&x,
        [IDX_SETPOS_Y]      = (const void *const restrict)&y,
        [IDX_SETPOS_Z]      = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetPosParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "SetPosition");

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_SET_POSITION,
        .handle   = (BodyHandle)h_raw,
        .position = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_rotation(PhysicsWorldObject *self, PyObject *const *args,
                                                    size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    float w;
    const void *const restrict targets[SetRot_COUNT] = {
        [IDX_SETROT_H] = (const void *const restrict)&h_raw,
        [IDX_SETROT_X] = (const void *const restrict)&x,
        [IDX_SETROT_Y] = (const void *const restrict)&y,
        [IDX_SETROT_Z] = (const void *const restrict)&z,
        [IDX_SETROT_W] = (const void *const restrict)&w};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetRotParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_QUAT(x, y, z, w, "SetRotation");

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_SET_ROTATION,
        .handle   = (BodyHandle)h_raw,
        .rotation = {x, y, z, w},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    const void *const restrict targets[Vec3_COUNT] = {[IDX_V3_H] =
                                                          (const void *const restrict)&h_raw,
                                                      [IDX_V3_X] = (const void *const restrict)&x,
                                                      [IDX_V3_Y] = (const void *const restrict)&y,
                                                      [IDX_V3_Z] = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetLinVelParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "LinearVelocity");

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_LINEAR_VELOCITY,
        .handle = (BodyHandle)h_raw,
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    const void *const restrict targets[Vec3_COUNT] = {[IDX_V3_H] =
                                                          (const void *const restrict)&h_raw,
                                                      [IDX_V3_X] = (const void *const restrict)&x,
                                                      [IDX_V3_Y] = (const void *const restrict)&y,
                                                      [IDX_V3_Z] = (const void *const restrict)&z};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetAngVelParser, targets)) {
        return nullptr;
    }
    VALIDATE_FINITE_VEC3(x, y, z, "AngularVelocity");

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY,
        .handle = (BodyHandle)h_raw,
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->GetMotionParser, targets)) {
        return nullptr;
    }

    JPH_MotionType motion_type;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_motion_type(&self->physics, (BodyHandle)h_raw, &motion_type);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(
            status, "Cannot query motion type of a body that has not been flushed to Jolt yet");
    }
    return PyLong_FromLong((long)motion_type);
}

PyCFunction_DeclareMethod PhysicsWorld_set_motion_type(PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    int motion_type;
    const void *const restrict targets[SetMotion_COUNT] = {
        [IDX_SM_H] = (const void *const restrict)&h_raw,
        [IDX_SM_M] = (const void *const restrict)&motion_type};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetMotionParser, targets)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_MOTION_TYPE,
        .handle = (BodyHandle)h_raw,
        .value  = motion_type,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    uint64_t data_raw;
    const void *const restrict targets[SetUserData_COUNT] = {
        [IDX_SUD_H] = (const void *const restrict)&h_raw,
        [IDX_SUD_D] = (const void *const restrict)&data_raw};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetUserDataParser, targets)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type      = PHYSICS_BODY_OP_SET_USER_DATA,
        .handle    = (BodyHandle)h_raw,
        .user_data = data_raw,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_get_user_data(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->GetUserDataParser, targets)) {
        return nullptr;
    }

    uint64_t user_data;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_user_data(&self->physics, (BodyHandle)h_raw, &user_data);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyLong_FromUnsignedLongLong(user_data);
}

PyCFunction_DeclareMethod PhysicsWorld_activate(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {[IDX_H_H] =
                                                           (const void *const restrict)&h_raw};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ActivateParser, targets)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_ACTIVATE,
        .handle = (BodyHandle)h_raw,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_deactivate(PhysicsWorldObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {[IDX_H_H] =
                                                           (const void *const restrict)&h_raw};

    // Note: Reusing ActivateParser as the schema is identical (single handle)
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ActivateParser, targets)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_DEACTIVATE,
        .handle = (BodyHandle)h_raw,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_transform(PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    PyObject *o_pos                                   = nullptr;
    PyObject *o_rot                                   = nullptr;
    const void *const restrict targets[SetTrns_COUNT] = {
        [IDX_ST_HANDLE] = (const void *const restrict)&h_raw,
        [IDX_ST_POS]    = (const void *const restrict)&o_pos,
        [IDX_ST_ROT]    = (const void *const restrict)&o_rot};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->SetTrnsParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION (Outside Lock)
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz)) {
        return nullptr;
    }
    if (!parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(px, py, pz, "SetTransform position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "SetTransform rotation");

    // 3. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_SET_TRANSFORM,
        .handle   = (BodyHandle)h_raw,
        .position = {px, py, pz},
        .rotation = {rx, ry, rz, rw},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_set_ccd(PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    bool enabled;
    const void *const restrict targets[CCD_COUNT] = {
        [IDX_CCD_H] = (const void *const restrict)&h_raw,
        [IDX_CCD_E] = (const void *const restrict)&enabled};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->CCDParser, targets)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_CCD,
        .handle = (BodyHandle)h_raw,
        .value  = (int)enabled,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod PhysicsWorld_get_index(PhysicsWorldObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ActivateParser, targets)) {
        return nullptr;
    }

    uint32_t index;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_index(&self->physics, (BodyHandle)h_raw, &index);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyLong_FromUnsignedLong(index);
}

PyCFunction_DeclareMethod PhysicsWorld_is_alive(PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ActivateParser, targets)) {
        return nullptr;
    }

    bool alive;
    Py_BEGIN_ALLOW_THREADS alive = physics_world_is_alive(&self->physics, (BodyHandle)h_raw);
    Py_END_ALLOW_THREADS

        if (alive) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod PhysicsWorld_is_active(PhysicsWorldObject *self, PyObject *const *args,
                                                 size_t nargsf, PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {
        [IDX_H_H] = (const void *const restrict)&h_raw,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ActivateParser, targets)) {
        return nullptr;
    }

    bool active;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_is_active(&self->physics, (BodyHandle)h_raw, &active);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (active) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod PhysicsWorld_get_active_indices(PhysicsWorldObject *self,
                                                          PyObject *args [[maybe_unused]]) {
    uint32_t *indices   = nullptr;
    size_t active_count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_active_indices(&self->physics, &indices, &active_count);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }

    if (active_count > (size_t)PY_SSIZE_T_MAX / sizeof(uint32_t)) {
        CulvNative_Free(indices);
        return PyErr_NoMemory();
    }
    PyObject *bytes_obj = PyBytes_FromStringAndSize((const char *)indices,
                                                    (Py_ssize_t)(active_count * sizeof(uint32_t)));
    CulvNative_Free(indices);
    return bytes_obj;
}

PyCFunction_DeclareMethod PhysicsWorld_get_render_state(PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    float alpha;
    const void *const restrict targets[Render_COUNT] = {
        [IDX_RND_ALPHA] = (const void *const restrict)&alpha,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->RenderParser, targets)) {
        return nullptr;
    }

    alpha        = fmaxf(0.0f, fminf(1.0f, alpha));
    float *state = nullptr;
    size_t count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_render_state(&self->physics, alpha, &state, &count);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }

    const size_t byte_count = count * 7 * sizeof(float);
    if (byte_count > (size_t)PY_SSIZE_T_MAX) {
        CulvNative_Free(state);
        PyErr_SetString(PyExc_OverflowError, "Render state is too large for a Python bytes object");
        return nullptr;
    }
    PyObject *bytes_obj = PyBytes_FromStringAndSize((const char *)state, (Py_ssize_t)byte_count);
    CulvNative_Free(state);
    return bytes_obj;
}

PyCFunction_DeclareMethod PhysicsWorld_set_collision_filter(PhysicsWorldObject *self,
                                                            PyObject *const *args, size_t nargsf,
                                                            PyObject *kwnames) {
    uint64_t h_raw;
    uint32_t category;
    uint32_t mask;
    const void *const restrict targets[ColFilter_COUNT] = {
        [IDX_CF_H] = (const void *const restrict)&h_raw,
        [IDX_CF_C] = (const void *const restrict)&category,
        [IDX_CF_M] = (const void *const restrict)&mask,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->ColFilterParser, targets)) {
        return nullptr;
    }

    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_set_collision_filter(&self->physics, (BodyHandle)h_raw, category, mask);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(
            status, "Cannot update collision filters of a body until it has been flushed to Jolt");
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_register_material(PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    uint32_t id;
    float friction                                   = RESTITUTION_BUFFER;
    float restitution                                = 0.0f;
    const void *const restrict targets[RegMat_COUNT] = {
        [IDX_RM_ID]   = (const void *const restrict)&id,
        [IDX_RM_FRIC] = (const void *const restrict)&friction,
        [IDX_RM_REST] = (const void *const restrict)&restitution,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->RegMatParser, targets)) {
        return nullptr;
    }

    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_register_material(&self->physics, id, friction, restitution);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod PhysicsWorld_create_heightfield(PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_scale    = nullptr;
    PyObject *o_heights  = nullptr;
    int grid_size        = 0;
    uint64_t user_data   = 0;
    uint32_t category    = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask        = COLLISION_FILTER_ALL_MASKS;
    uint32_t material_id = 0;
    float friction       = DEFAULT_FRICTION;
    float restitution    = 0.0f;

    const void *const restrict targets[Heightfield_COUNT] = {
        [IDX_HF_POS]       = (const void *const restrict)&o_pos,
        [IDX_HF_ROT]       = (const void *const restrict)&o_rot,
        [IDX_HF_SCALE]     = (const void *const restrict)&o_scale,
        [IDX_HF_HEIGHTS]   = (const void *const restrict)&o_heights,
        [IDX_HF_GRID_SIZE] = (const void *const restrict)&grid_size,
        [IDX_HF_USER_DATA] = (const void *const restrict)&user_data,
        [IDX_HF_CAT]       = (const void *const restrict)&category,
        [IDX_HF_MASK]      = (const void *const restrict)&mask,
        [IDX_HF_MAT_ID]    = (const void *const restrict)&material_id,
        [IDX_HF_FRIC]      = (const void *const restrict)&friction,
        [IDX_HF_REST]      = (const void *const restrict)&restitution};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->HeightfieldParser, targets)) {
        return nullptr;
    }

    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    float sx;
    float sy;
    float sz;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw) ||
        !parse_vec3_direct(o_scale, &sx, &sy, &sz)) {
        return nullptr;
    }

    Py_buffer h_view;
    if (PyObject_GetBuffer(o_heights, &h_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (UNLIKELY(h_view.len != (Py_ssize_t)((Py_ssize_t)grid_size * grid_size * sizeof(float)))) {
        PyBuffer_Release(&h_view);
        return PyErr_Format(PyExc_ValueError, "Height buffer size mismatch");
    }

    // 2. Native heightfield construction and world staging (buffer stays pinned).
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status = PHYSICS_WORLD_STATUS_OK;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_create_heightfield_body(
        &self->physics, (const float *)h_view.buf, (size_t)grid_size, (JPH_Vec3){sx, sy, sz},
        (JPH_RVec3){px, py, pz}, (JPH_Quat){rx, ry, rz, rw}, friction, restitution, category, mask,
        material_id, user_data, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS PyBuffer_Release(&h_view);

    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, "HeightField build failed",
                           "Failed to create HeightField BodySettings");
        return nullptr;
    }
    self->view_shape[0] = (Py_ssize_t)body_count;
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod PhysicsWorld_get_debug_data(PhysicsWorldObject *self,
                                                      PyObject *const *args, size_t nargsf,
                                                      PyObject *kwnames) {
    bool draw_shapes                                    = true;
    bool draw_constraints                               = true;
    bool draw_bounding_box                              = false;
    bool draw_centers                                   = false;
    bool wireframe                                      = true;
    const void *const restrict targets[DebugData_COUNT] = {
        [IDX_DD_SHAPES]      = (const void *const restrict)&draw_shapes,
        [IDX_DD_CONSTRAINTS] = (const void *const restrict)&draw_constraints,
        [IDX_DD_BBOX]        = (const void *const restrict)&draw_bounding_box,
        [IDX_DD_CENTERS]     = (const void *const restrict)&draw_centers,
        [IDX_DD_WIREFRAME]   = (const void *const restrict)&wireframe,
    };
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->DebugDataParser, targets)) {
        return nullptr;
    }

    DebugVertex *lines     = nullptr;
    size_t line_count      = 0;
    DebugVertex *triangles = nullptr;
    size_t triangle_count  = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_build_debug_data(
        &self->physics, draw_shapes, draw_constraints, draw_bounding_box, draw_centers, wireframe,
        &lines, &line_count, &triangles, &triangle_count);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (line_count > (size_t)PY_SSIZE_T_MAX / sizeof(DebugVertex) ||
        triangle_count > (size_t)PY_SSIZE_T_MAX / sizeof(DebugVertex)) {
        CulvNative_Free(lines);
        CulvNative_Free(triangles);
        PyErr_SetString(PyExc_OverflowError, "Debug data is too large for Python bytes objects");
        return nullptr;
    }

    PyObject *lines_bytes = PyBytes_FromStringAndSize(
        (const char *)lines, (Py_ssize_t)(line_count * sizeof(DebugVertex)));
    PyObject *triangles_bytes = PyBytes_FromStringAndSize(
        (const char *)triangles, (Py_ssize_t)(triangle_count * sizeof(DebugVertex)));
    CulvNative_Free(lines);
    CulvNative_Free(triangles);
    if (UNLIKELY(!lines_bytes || !triangles_bytes)) {
        Py_XDECREF(lines_bytes);
        Py_XDECREF(triangles_bytes);
        return nullptr;
    }

    PyObject *result = PyTuple_Pack(2, lines_bytes, triangles_bytes);
    Py_DECREF(lines_bytes);
    Py_DECREF(triangles_bytes);
    return result;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_soft_body(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf,
                                                                  PyObject *kwnames) {

    // 1. FAST PARSE
    PyObject *o_shared        = nullptr;
    PyObject *o_pos           = nullptr;
    PyObject *o_rot           = nullptr;
    uint64_t user_data        = 0;
    uint32_t category         = COLLISION_FILTER_ALL_CATEGORIES;
    uint32_t mask             = COLLISION_FILTER_ALL_MASKS;
    float pressure            = 0.0f;
    float vertex_radius       = 0.05f;
    float linear_damping      = 0.1f;
    uint32_t num_iterations   = 10;
    float max_linear_velocity = 500.0f;
    float gravity_factor      = 1.0f;
    float friction            = 0.2f;
    float restitution         = 0.0f;
    bool make_rot_identity    = false;
    bool update_position      = true; // Jolt default is usually true
    bool faces_double_sided   = false;

    const void *const restrict targets[CreateSoftBody_COUNT] = {
        [IDX_CSB_SHARED]     = (const void *const restrict)&o_shared,
        [IDX_CSB_POS]        = (const void *const restrict)&o_pos,
        [IDX_CSB_ROT]        = (const void *const restrict)&o_rot,
        [IDX_CSB_USER_DATA]  = (const void *const restrict)&user_data,
        [IDX_CSB_CAT]        = (const void *const restrict)&category,
        [IDX_CSB_MASK]       = (const void *const restrict)&mask,
        [IDX_CSB_PRESSURE]   = (const void *const restrict)&pressure,
        [IDX_CSB_V_RADIUS]   = (const void *const restrict)&vertex_radius,
        [IDX_CSB_LIN_DAMP]   = (const void *const restrict)&linear_damping,
        [IDX_CSB_ITER]       = (const void *const restrict)&num_iterations,
        [IDX_CSB_MAX_VEL]    = (const void *const restrict)&max_linear_velocity,
        [IDX_CSB_GRAV]       = (const void *const restrict)&gravity_factor,
        [IDX_CSB_FRIC]       = (const void *const restrict)&friction,
        [IDX_CSB_REST]       = (const void *const restrict)&restitution,
        [IDX_CSB_ROT_ID]     = (const void *const restrict)&make_rot_identity,
        [IDX_CSB_UPDATE_POS] = (const void *const restrict)&update_position,
        [IDX_CSB_FACE_DS]    = (const void *const restrict)&faces_double_sided};

    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->CreateSoftBodyParser, targets)) {
        return nullptr;
    }

    // 2. VECTOR EXTRACTION
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;
    float rx;
    float ry;
    float rz;
    float rw;
    if (!parse_vec3_direct(o_pos, &px, &py, &pz) || !parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");

    CulverinState *state = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(o_shared, (PyTypeObject *)state->SoftBodySharedSettingsType)) {
        PyErr_SetString(PyExc_TypeError, "shared must be a SoftBodySharedSettings object");
        return nullptr;
    }
    auto py_shared = (SoftBodySharedSettingsObject *)o_shared;

    // 3. Native Jolt settings construction and world staging.
    const PhysicsWorldSoftBodyCreateRequest request = {
        .shared_settings        = py_shared->settings,
        .vertex_count           = py_shared->num_vertices,
        .position               = {px, py, pz},
        .rotation               = {rx, ry, rz, rw},
        .user_data              = user_data,
        .category               = category,
        .mask                   = mask,
        .pressure               = pressure,
        .vertex_radius          = vertex_radius,
        .linear_damping         = linear_damping,
        .num_iterations         = num_iterations,
        .max_linear_velocity    = max_linear_velocity,
        .gravity_factor         = gravity_factor,
        .friction               = friction,
        .restitution            = restitution,
        .make_rotation_identity = make_rot_identity,
        .update_position        = update_position,
        .faces_double_sided     = faces_double_sided,
    };
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status = PHYSICS_WORLD_STATUS_OK;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_create_soft_body(
        &self->physics, &request, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, nullptr,
                           "Failed to create SoftBodyCreationSettings");
        return nullptr;
    }
    self->view_shape[0] = (Py_ssize_t)body_count;
    return PyLong_FromUnsignedLongLong(handle);
}

// Getters

PyCFunction_DeclareMethodFromModule
PhysicsWorld_get_soft_body_vertex_count(PhysicsWorldObject *self, PyObject *const *args,
                                        size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    const void *const restrict targets[HOnly_COUNT] = {[IDX_H_H] =
                                                           (const void *const restrict)&handle};
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->HOnlyParser, targets)) {
        return nullptr;
    }

    uint32_t count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_soft_body_vertex_count(&self->physics, handle, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyLong_FromUnsignedLong(count);
}

PyCFunction_DeclareMethodFromModule
PhysicsWorld_get_soft_body_vertex_position(PhysicsWorldObject *self, PyObject *const *args,
                                           size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    uint32_t index;
    const void *const restrict targets[GetSbVertex_COUNT] = {
        [IDX_GSBV_H] = (const void *const restrict)&handle,
        [IDX_GSBV_I] = (const void *const restrict)&index};
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->GetSbVertexParser, targets)) {
        return nullptr;
    }

    JPH_Vec3 position;
    uint32_t count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = physics_world_get_soft_body_vertex_position(
        &self->physics, handle, index, &position, &count);
    Py_END_ALLOW_THREADS if (status == PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_RANGE) {
        return PyErr_Format(PyExc_IndexError, "Vertex index %u out of bounds (count: %u)", index,
                            count);
    }
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return CULV_PY_BUILD_TUPLE(position.x, position.y, position.z);
}

PyCFunction_DeclareMethodFromModule
PhysicsWorld_get_soft_body_local_vertices(PhysicsWorldObject *self, PyObject *const *args,
                                          size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    const void *const restrict targets[HOnly_COUNT] = {[IDX_H_H] =
                                                           (const void *const restrict)&handle};
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->HOnlyParser, targets)) {
        return nullptr;
    }

    JPH_Vec3 *positions = nullptr;
    uint32_t count      = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_get_soft_body_local_vertices(&self->physics, handle, &positions, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if ((size_t)count > (size_t)PY_SSIZE_T_MAX / sizeof(JPH_Vec3)) {
        CulvNative_Free(positions);
        PyErr_SetString(PyExc_OverflowError, "Soft body vertex data is too large");
        return nullptr;
    }

    PyObject *bytes = PyBytes_FromStringAndSize((const char *)positions,
                                                (Py_ssize_t)((size_t)count * sizeof(JPH_Vec3)));
    CulvNative_Free(positions);
    return bytes;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_get_soft_body_vertices(PhysicsWorldObject *self,
                                                                        PyObject *const *args,
                                                                        size_t nargsf,
                                                                        PyObject *kwnames) {
    uint64_t h_raw;
    const void *const restrict targets[HOnly_COUNT] = {[IDX_H_H] =
                                                           (const void *const restrict)&h_raw};
    if (!CULV_FAST_PARSE_UNIFIED(args, PyVectorcall_NARGS(nargsf), kwnames,
                                 &self->parsers->HOnlyParser, targets)) {
        return nullptr;
    }

    SHADOW_LOCK(&self->physics.shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    CHECK_HANDLE(h_raw, slot);

    const uint8_t state =
        atomic_load_explicit(&self->physics.slot_states[slot], memory_order_acquire);
    if (state != SLOT_SOFT_BODY) {
        SHADOW_UNLOCK(&self->physics.shadow_lock);
        return PyErr_Format(PyExc_TypeError, "Handle does not belong to a soft body");
    }

    uint32_t dense_idx     = self->physics.slot_to_dense[slot];
    SoftBodyShadow *shadow = &self->physics.soft_shadows[dense_idx];

    if (!shadow->vertices) {
        SHADOW_UNLOCK(&self->physics.shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Soft body shadow buffer missing");
    }

    // We can reuse BufferProxyObject, but we need to tell it to use THIS specific pointer
    // and THIS specific length, rather than the global positions array.
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    BufferProxyObject *proxy =
        PyObject_GC_New(BufferProxyObject, (PyTypeObject *)st->BufferProxyType);
    proxy->owner = (PyObject *)self;
    Py_INCREF(self);

    proxy->buf_type    = PROXY_DYNAMIC;
    proxy->dynamic_ptr = shadow->vertices;
    proxy->format      = JPH_REAL_STRING;
    proxy->itemsize    = sizeof(JPH_Real);
    proxy->stride      = 4; // PosStride
    proxy->shape[0]    = (Py_ssize_t)shadow->num_vertices * 4;

    atomic_fetch_add_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);

    SHADOW_UNLOCK(&self->physics.shadow_lock);

    PyObject_GC_Track(proxy);
    return (PyObject *)proxy;
}

PyCFunction_DeclareMethod PhysicsWorld_benchmark_parse([[maybe_unused]] PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    constexpr size_t NUM_ARGS = 64;
    uint64_t values[NUM_ARGS] = {};
    const void *restrict targets[NUM_ARGS];

    // Map targets to values array (Loop is efficient here, unrolled by compiler)
    for (size_t i = 0; i < NUM_ARGS; ++i) {
        targets[i] = (const void *const restrict)&values[i];
    }

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (UNLIKELY(!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->StressTestParser,
                                          targets))) {
        return nullptr;
    }

    Py_RETURN_NONE;
}

/**
 * Benchmark for CULV_PY_BUILD_TUPLE using METH_NOARGS.
 * This eliminates all argument parsing overhead to isolate the
 * performance of the Culverin Fast Build engine.
 */
PyCFunction_DeclareMethod PhysicsWorld_benchmark_build([[maybe_unused]] PyObject *self,
                                                       [[maybe_unused]] PyObject *args) {
    // 1. Define dummy C data to be "built" into Python
    constexpr int trash         = 42;
    constexpr float trashpi     = 3.14f;
    constexpr double trasheuler = 2.718281828;
    int i_val                   = trash;
    float f_val                 = trashpi;
    double d_val                = trasheuler;
    const char *s_val =
        "Pater noster qui es in caelis, sanctificetur nomen tuum. Adveniat regnum tuum. Fiat "
        "voluntas tua, sicut in caelo et in terra. Panem nostrum quotidianum da nobis hodie, et "
        "dimitte nobis debita nostra sicut et nos dimittimus debitoribus nostris. Et ne nos "
        "inducas in tentationem, sed libera nos a malo. Amen.";
    bool b_val = true;

    // 2. Execute the Build
    // CULV_PY_BUILD_TUPLE uses CULV_PY_BUILD_VAL to route types at compile-time.
    // It then uses culv_py_build_pack_tuple to perform O(1) allocation and
    // O(N) reference stealing.
    constexpr float stfu = 200.0f;
    PyObject *result     = CULV_PY_BUILD_TUPLE(
        i_val, f_val, d_val, s_val, b_val, (int)100, (float)stfu,
        "Lorem ipsum dolor sit amet consectetur adipiscing elit, sed do eiusmod "
        "tempor incididunt ut labore et dolore magna aliqua.",
        false);

    // 3. Error handling
    if (UNLIKELY(!result)) {
        // culv_py_build_pack_tuple handles internal cleanup on failure
        return nullptr;
    }

    // Return the new reference to the caller
    return result;
}

// --- Ragdoll Settings Implementation ---
void culverin_init_ragdoll_settings_parsers(RagdollSettingsParsers *rsp);
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll_settings(PhysicsWorldObject *self,
                                                                         PyObject *const *args,
                                                                         Py_ssize_t nargs,
                                                                         PyObject *kwnames) {
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *py_skel_obj                                     = nullptr;
    const void *const restrict targets[RagdollSettings_COUNT] = {
        [IDX_RS_SKELETON] = (const void *const restrict)&py_skel_obj,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->RagdollSettingsParser,
                                 targets)) {
        return nullptr;
    }

    // --- 2. TYPE VALIDATION ---

    // Manual type check (replaces O! format string)
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(py_skel_obj, (PyTypeObject *)st->SkeletonType)) {
        PyErr_SetString(PyExc_TypeError, "skeleton must be a Skeleton object");
        return nullptr;
    }
    SkeletonObject *py_skel = (SkeletonObject *)py_skel_obj;

    // --- 3. OBJECT CREATION ---
    auto obj = (RagdollSettingsObject *)PyObject_New(RagdollSettingsObject,
                                                     (PyTypeObject *)st->RagdollSettingsType);
    if (!obj) {
        return PyErr_NoMemory();
    }
    obj->parsers = (RagdollSettingsParsers *)PyMem_Malloc(sizeof(RagdollSettingsParsers));
    if (!obj->parsers) {
        Py_DECREF(obj);
        return PyErr_NoMemory();
    }
    culverin_init_ragdoll_settings_parsers(obj->parsers);

    // Create native Jolt settings without crossing the Python boundary from core code.
    Py_BEGIN_ALLOW_THREADS obj->settings = physics_world_create_ragdoll_settings(py_skel->skeleton);
    Py_END_ALLOW_THREADS if (!obj->settings) {
        Py_DECREF(obj);
        return PyErr_NoMemory();
    }

    obj->world = self;
    Py_INCREF(self);

    return (PyObject *)obj;
}
void culverin_init_ragdoll_parsers(RagdollParsers *rp);
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ragdoll(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames) {
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *settings_obj = nullptr;
    PosStride pos          = {.x = 0, .y = 0, .z = 0};
    AuxStride rot          = {.x = 0, .y = 0, .z = 0, .w = 1.0f};
    uint64_t user_data     = 0;
    uint32_t category      = JOLT_ALL_LAYER_BITS;
    uint32_t mask          = JOLT_ALL_LAYER_BITS;
    uint32_t material_id   = 0;

    const void *const restrict targets[CreateRagdoll_COUNT] = {
        [IDX_CR_SETTINGS] = (const void *const restrict)&settings_obj,
        [IDX_CR_POS]      = (const void *const restrict)&pos,
        [IDX_CR_ROT]      = (const void *const restrict)&rot,
        [IDX_CR_USER]     = (const void *const restrict)&user_data,
        [IDX_CR_CAT]      = (const void *const restrict)&category,
        [IDX_CR_MASK]     = (const void *const restrict)&mask,
        [IDX_CR_MAT]      = (const void *const restrict)&material_id,
    };

    if (!CULV_FAST_PARSE_UNIFIED(args, nargs, kwnames, &self->parsers->CreateRagdollParser,
                                 targets)) {
        return nullptr;
    }

    // Type Safety Check (Replaces original O! logic)
    CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(settings_obj, (PyTypeObject *)st->RagdollSettingsType)) {
        PyErr_SetString(PyExc_TypeError, "settings must be a RagdollSettings object");
        return nullptr;
    }
    auto py_settings = (RagdollSettingsObject *)settings_obj;

    // --- 2. Native Jolt creation and pose setup ---
    JPH_Ragdoll *j_rag = nullptr;
    size_t body_count  = 0;
    PhysicsWorldCreationStatus creation_status;
    Py_BEGIN_ALLOW_THREADS creation_status = physics_world_create_ragdoll_instance(
        &self->physics, py_settings->settings, (JPH_RVec3){pos.x, pos.y, pos.z},
        (JPH_Quat){rot.x, rot.y, rot.z, rot.w}, user_data, &j_rag, &body_count);
    Py_END_ALLOW_THREADS if (creation_status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, creation_status, PHYSICS_WORLD_STATUS_OK,
                           "Jolt failed to create Ragdoll instance", nullptr);
        return nullptr;
    }

    uint32_t *body_slots = nullptr;
    if (body_count > SIZE_MAX / sizeof(uint32_t) ||
        !(body_slots = CULV_RAW_MALLOC(body_count * sizeof(uint32_t)))) {
        Py_BEGIN_ALLOW_THREADS physics_world_destroy_ragdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS return PyErr_NoMemory();
    }

    auto obj = (RagdollObject *)PyObject_New(RagdollObject, (PyTypeObject *)st->RagdollType);
    if (!obj) {
        CULV_RAW_FREE(body_slots);
        Py_BEGIN_ALLOW_THREADS physics_world_destroy_ragdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS return nullptr;
    }
    obj->ragdoll    = nullptr;
    obj->world      = nullptr;
    obj->body_count = 0;
    obj->body_slots = body_slots;
    obj->parsers    = nullptr;
    obj->parsers    = (RagdollParsers *)PyMem_Malloc(sizeof(RagdollParsers));
    if (!obj->parsers) {
        Py_DECREF(obj);
        Py_BEGIN_ALLOW_THREADS physics_world_destroy_ragdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS return PyErr_NoMemory();
    }
    culverin_init_ragdoll_parsers(obj->parsers);

    obj->ragdoll = j_rag;
    obj->world   = self;
    Py_INCREF(self);
    obj->body_count = body_count;

    size_t world_count                     = 0;
    PhysicsWorldStatus native_status       = PHYSICS_WORLD_STATUS_OK;
    Py_BEGIN_ALLOW_THREADS creation_status = physics_world_register_ragdoll_bodies(
        &self->physics, j_rag, body_count, user_data, category, mask, material_id, body_slots,
        &world_count, &native_status);
    Py_END_ALLOW_THREADS if (creation_status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        obj->ragdoll = nullptr;
        Py_DECREF(obj);
        Py_BEGIN_ALLOW_THREADS physics_world_destroy_ragdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS set_creation_error(self, creation_status, native_status, nullptr,
                                                nullptr);
        return nullptr;
    }

    self->view_shape[0] = (Py_ssize_t)world_count;
    return (PyObject *)obj;
}

// Fixed get_contact_events to be safer with locking
PyCFunction_DeclareMethodFromModule
PhysicsWorld_get_contact_events(PhysicsWorldObject *self, PyObject *args [[maybe_unused]]) {
    ContactEvent *scratch = nullptr;
    size_t count          = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_take_contact_events(&self->physics, &scratch, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (count == 0) {
        return PyList_New(0);
    }

    // --- 2. BUILD PHASE (Unlocked & FastBuild Integrated) ---
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) {
        CulvNative_AlignedFree(scratch);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEventSlim *slim = GetSlimHeader(&scratch[i]);

        // TSan Fix: Explicitly load the handles from the atomic members in the struct.
        // We use relaxed because this 'scratch' copy is thread-local and synchronized.
        uint64_t b1_raw = atomic_load_explicit(&slim->body1, memory_order_relaxed);
        uint64_t b2_raw = atomic_load_explicit(&slim->body2, memory_order_relaxed);

        /**
         * OPTIMIZATION: CULV_PY_BUILD_TUPLE
         * 1. culv_py_build_from_u64 converts b1_raw and b2_raw
         * 2. culv_py_build_from_float converts impulse and sliding_speed
         * 3. culv_py_build_pack_tuple performs a single O(1) allocation
         */
        PyObject *item = CULV_PY_BUILD_TUPLE(b1_raw, b2_raw, slim->impulse, slim->sliding_speed);

        if (UNLIKELY(!item)) {
            Py_DECREF(list);
            CulvNative_AlignedFree(scratch);
            return nullptr;
        }

        // PyList_SET_ITEM steals the reference from CULV_PY_BUILD_TUPLE
        PyList_SET_ITEM(list, (Py_ssize_t)i, item);
    }

    CulvNative_AlignedFree(scratch);
    return list;
}

PyCFunction_DeclareMethodFromModule
PhysicsWorld_get_contact_events_ex(PhysicsWorldObject *self, PyObject *args [[maybe_unused]]) {
    ContactEvent *scratch = nullptr;
    size_t count          = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_take_contact_events(&self->physics, &scratch, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (count == 0) {
        return PyList_New(0);
    }

    // --- 2. KEY INTERNING (Persistent) ---
    static PyObject *k_bodies = nullptr;
    static PyObject *k_pos    = nullptr;
    static PyObject *k_norm   = nullptr;
    static PyObject *k_str    = nullptr;
    static PyObject *k_slide  = nullptr;
    static PyObject *k_mat    = nullptr;
    static PyObject *k_type   = nullptr;

    if (UNLIKELY(!k_bodies)) {
        k_bodies = PyUnicode_InternFromString("bodies");
        k_pos    = PyUnicode_InternFromString("position");
        k_norm   = PyUnicode_InternFromString("normal");
        k_str    = PyUnicode_InternFromString("impulse");
        k_slide  = PyUnicode_InternFromString("slide_speed"); // Updated key to reflect lack of _sq
        k_mat    = PyUnicode_InternFromString("materials");
        k_type   = PyUnicode_InternFromString("type");
    }

    // --- 3. BUILD PHASE (FastBuild Engine) ---
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) {
        CulvNative_AlignedFree(scratch);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEventSlim *slim  = GetSlimHeader(&scratch[i]);
        ContactEventFatExt *fat = GetFatExtension(&scratch[i]);

        // TSan Fix: Explicit relaxed loads for atomic handles
        uint64_t b1 = atomic_load_explicit(&slim->body1, memory_order_relaxed);
        uint64_t b2 = atomic_load_explicit(&slim->body2, memory_order_relaxed);

        /**
         * OPTIMIZATION: CULV_PY_BUILD_DICT
         * We compose the nested tuples (pos, normal, bodies, mats)
         * and the dictionary in a single, readable expression.
         */
        PyObject *dict =
            CULV_PY_BUILD_DICT(k_bodies, CULV_PY_BUILD_TUPLE(b1, b2), k_pos,
                               CULV_PY_BUILD_TUPLE(slim->px, slim->py, slim->pz), k_norm,
                               CULV_PY_BUILD_TUPLE(slim->nx, slim->ny, slim->nz), k_mat,
                               CULV_PY_BUILD_TUPLE(fat->mat1, fat->mat2), k_str, slim->impulse,
                               k_slide, slim->sliding_speed, k_type, slim->flags);

        if (UNLIKELY(!dict)) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, (Py_ssize_t)i, Py_None);
            continue;
        }

        // Steals ref to dict created by FastBuild
        PyList_SET_ITEM(list, (Py_ssize_t)i, dict);
    }

    CulvNative_AlignedFree(scratch);
    return list;
}

// ContactEvent layout:
// 128 bytes total (64 Slim Header, 64 Fat Extension)
// See ContactEventDefinitions.h for exact alignment and padding.
PyCFunction_DeclareMethodFromModule
PhysicsWorld_get_contact_events_raw(PhysicsWorldObject *self, PyObject *args [[maybe_unused]]) {
    ContactEvent *events = nullptr;
    size_t count         = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        physics_world_take_contact_events(&self->physics, &events, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (count == 0) {
        PyObject *empty = PyBytes_FromStringAndSize("", 0);
        if (!empty) {
            return nullptr;
        }
        PyObject *view = PyMemoryView_FromObject(empty);
        Py_DECREF(empty);
        return view;
    }
    if (count > (size_t)PY_SSIZE_T_MAX / sizeof(ContactEvent)) {
        CulvNative_AlignedFree(events);
        PyErr_SetString(PyExc_OverflowError, "Contact event data is too large");
        return nullptr;
    }

    const size_t bytes_size = count * sizeof(ContactEvent);
    PyObject *raw_bytes = PyBytes_FromStringAndSize((const char *)events, (Py_ssize_t)bytes_size);
    CulvNative_AlignedFree(events);
    if (!raw_bytes) {
        return nullptr;
    }
    PyObject *view = PyMemoryView_FromObject(raw_bytes);
    Py_DECREF(raw_bytes);
    return view;
}

PyType_DeclareSlot_StatusFromModule
PhysicsWorld_getbuffer(PhysicsWorldObject *self, Py_buffer *view, [[maybe_unused]] int flags) {
    SHADOW_LOCK(&self->physics.shadow_lock);

    // TSan Fix: Read the atomic count safely
    size_t current_count = atomic_load_explicit(&self->physics.count, memory_order_acquire);

    // We export the positions buffer as the default buffer for the object
    view->buf        = self->physics.positions;
    view->len        = (Py_ssize_t)(current_count * sizeof(PosStride));
    view->readonly   = 0;
    view->itemsize   = sizeof(JPH_Real);
    view->format     = (sizeof(JPH_Real) == sizeof(double)) ? "d" : "f";
    view->ndim       = 2;
    view->shape      = self->view_shape;
    view->strides    = self->view_strides;
    view->suboffsets = nullptr;
    view->internal   = nullptr;

    // view_export_count is a standard int protected by shadow_lock
    atomic_fetch_add_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);

    SHADOW_UNLOCK(&self->physics.shadow_lock);
    return 0;
}

// Buffer Release Slot
PyType_DeclareSlot_VoidFromModule PhysicsWorld_releasebuffer(PhysicsWorldObject *self,
                                                             Py_buffer *view [[maybe_unused]]) {
    SHADOW_LOCK(&self->physics.shadow_lock);

    // Release logic remains simple as no atomic counters are mutated here
    if (atomic_load_explicit(&self->physics.view_export_count, memory_order_relaxed) > 0) {
        atomic_fetch_sub_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);
    }

    SHADOW_UNLOCK(&self->physics.shadow_lock);
}

// User-facing macros for context methods
#define PW_FASTCALL(name) CULV_FEAT(PhysicsWorld, name, METH_FASTCALL | METH_KEYWORDS)
#define PW_NOARGS(name) CULV_FEAT(PhysicsWorld, name, METH_NOARGS)
#define PW_O(name) CULV_FEAT(PhysicsWorld, name, METH_O)

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_vehicle(PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                Py_ssize_t nargs,
                                                                PyObject *kwnames);

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_tracked_vehicle(PhysicsWorldObject *self,
                                                                        PyObject *const *args,
                                                                        Py_ssize_t nargs,
                                                                        PyObject *kwnames);
PyCFunction_DeclareMethodFromModule PhysicsWorld_create_ship(PhysicsWorldObject *self,
                                                             PyObject *const *args,
                                                             Py_ssize_t nargs, PyObject *kwnames);
PyType_Spec PhysicsWorld_spec = {
    .name      = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(PhysicsWorldObject),
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_MANAGED_DICT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = PyType_GenericNew},
            {.slot = Py_tp_init, .pfunc = PhysicsWorld_init},
            {.slot = Py_tp_dealloc, .pfunc = PhysicsWorld_dealloc},
            {.slot = Py_tp_call, .pfunc = nullptr},
            {.slot = Py_tp_methods,
             .pfunc =
                 (PyMethodDef[]){

                     // --- Lifecycle ---
                     PW_FASTCALL(step),
                     PW_FASTCALL(create_body),
                     PW_FASTCALL(create_bodies_batch),
                     PW_FASTCALL(destroy_body),
                     PW_FASTCALL(destroy_bodies_batch),
                     PW_FASTCALL(create_mesh_body),
                     PW_FASTCALL(create_constraint),
                     PW_FASTCALL(destroy_constraint),
                     PW_FASTCALL(get_constraint_type),
                     PW_FASTCALL(create_vehicle),
                     PW_FASTCALL(create_tracked_vehicle),
                     PW_FASTCALL(create_ship),
                     PW_FASTCALL(create_ragdoll_settings),
                     PW_FASTCALL(create_ragdoll),
                     PW_FASTCALL(create_heightfield),
                     PW_FASTCALL(create_convex_hull),
                     PW_FASTCALL(create_compound_body),
                     PW_FASTCALL(create_soft_body),

                     // --- Interaction ---
                     PW_FASTCALL(apply_impulse),
                     PW_FASTCALL(apply_angular_impulse),
                     PW_FASTCALL(apply_impulse_at),
                     PW_FASTCALL(apply_force),
                     PW_FASTCALL(apply_torque),
                     PW_FASTCALL(set_gravity),
                     PW_NOARGS(get_gravity),
                     PW_FASTCALL(apply_buoyancy),
                     PW_FASTCALL(apply_buoyancy_batch),
                     PW_FASTCALL(set_position),
                     PW_FASTCALL(set_rotation),
                     PW_FASTCALL(set_linear_velocity),
                     PW_FASTCALL(set_angular_velocity),
                     PW_FASTCALL(set_transform),
                     PW_FASTCALL(set_collision_filter),
                     PW_FASTCALL(register_material),
                     PW_FASTCALL(set_constraint_target),

                     // --- Motion Control ---
                     PW_FASTCALL(get_motion_type),
                     PW_FASTCALL(set_motion_type),
                     PW_FASTCALL(activate),
                     PW_FASTCALL(deactivate),
                     PW_FASTCALL(set_ccd),

                     // --- Queries ---
                     PW_FASTCALL(get_soft_body_vertices),
                     PW_FASTCALL(get_soft_body_vertex_count),
                     PW_FASTCALL(get_soft_body_vertex_position),
                     PW_FASTCALL(get_soft_body_local_vertices),
                     PW_FASTCALL(raycast),
                     PW_FASTCALL(raycast_batch),
                     PW_FASTCALL(shapecast),
                     PW_FASTCALL(overlap_sphere),
                     PW_FASTCALL(overlap_aabb),

                     // --- Utilities ---
                     PW_FASTCALL(get_index),
                     PW_FASTCALL(is_alive),
                     PW_FASTCALL(is_active),
                     PW_NOARGS(get_active_indices),
                     PW_FASTCALL(get_render_state),
                     PW_FASTCALL(get_debug_data),
                     PW_FASTCALL(get_body_stats),

                     // --- User Data ---
                     PW_FASTCALL(get_user_data),
                     PW_FASTCALL(set_user_data),

                     // -- Event Logic ---
                     PW_NOARGS(get_contact_events),
                     PW_NOARGS(get_contact_events_ex),
                     PW_NOARGS(get_contact_events_raw),

                     // --- State & Advanced ---
                     PW_NOARGS(save_state),
                     PW_FASTCALL(load_state),
                     PW_FASTCALL(create_character),

                     // --- Internal/Debug ---
                     // Not for public use, therefore can't use macros
                     {"_benchmark_parse", CULV_CAST(PhysicsWorld_benchmark_parse),
                      METH_FASTCALL | METH_KEYWORDS, nullptr},

                     {"_benchmark_build", CULV_CAST(PhysicsWorld_benchmark_build), METH_NOARGS,
                      nullptr},

                     {}

                 }},
            {.slot = Py_tp_members,
             .pfunc =
                 (PyMemberDef[]){

                     {.name   = "__weaklistoffset__",
                      .type   = Py_T_PYSSIZET,
                      .offset = offsetof(PhysicsWorldObject, weakreflist),
                      .flags  = Py_READONLY,
                      .doc    = nullptr},
                     {}

                 }},
            {.slot = Py_tp_getset,
             .pfunc =
                 (PyGetSetDef[]){

                     GETSET("positions", get_positions),
                     GETSET("rotations", get_rotations),
                     GETSET("velocities", get_velocities),
                     GETSET("angular_velocities", get_angular_velocities),
                     GETSET("count", get_count),
                     GETSET("time", get_time),
                     GETSET("user_data", get_user_data_buffer),
                     GETSET("shape_count", get_shape_count),
                     GETSET("is_step_pending", get_is_step_pending),
                     GETSET("max_bodies", PhysicsWorld_get_max_bodies),
                     GETSET("remaining_capacity", PhysicsWorld_get_remaining_capacity),
                     {}

                 }},
            {.slot = Py_bf_getbuffer, .pfunc = PhysicsWorld_getbuffer},
            {.slot = Py_bf_releasebuffer, .pfunc = PhysicsWorld_releasebuffer},
            {.slot = Py_tp_traverse, .pfunc = PhysicsWorld_traverse},
            {.slot = Py_tp_clear, .pfunc = PhysicsWorld_clear},
            {},

        },
};
