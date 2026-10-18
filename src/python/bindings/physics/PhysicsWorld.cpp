#include "bindings/physics/PhysicsWorld.h"
#include "bindings/BindingCommon.h"
#include "bindings/FastBuild.h"
#include "bindings/Getters.h"
#include "bindings/ModuleState.h"
#include "bindings/PythonSupport.h"
#include "bindings/parsing/ParserRegistry.hpp"
#include "bindings/physics/Constraint.h" // IWYU pragma: keep
#include "bindings/physics/PhysicsSync.h"
#include "bindings/physics/QueryMethods.h"
#include "bindings/physics/WorldAdapters.h"
#include "characters/Character.h" // IWYU pragma: keep
#include "core/CompilerSpecifics.h"
#include "core/NativeAlloc.h"
#include "math/MathCore.h"
#include "physics/ContactEventDefinitions.h"
#include "physics/PhysicsWorld.h"
#include "physics/PhysicsWorldCreation.h"
#include "physics/PhysicsWorldOps.h"
#include "physics/PhysicsWorldRuntime.h"
#include "physics/ShadowSync.h"
#include "ragdolls/Ragdoll.h"
#include "soft_bodies/SoftBody.h"
#include <math.h>

// ============================================================================
// Semantic Constants - Magic Number Replacements
// ============================================================================

// Memory and Alignment

// Physics Simulation
static constexpr float kDefaultFrameTime         = 1.0f / 60.0f;
static constexpr float kDefaultLinearDrag        = 0.5f;
static constexpr float kDefaultAngularDrag       = 0.5f;
static constexpr float kDefaultFriction          = 0.2f;
static constexpr uint32_t kNoAllowedDofsOverride = 0U;

// Collision Filtering
static constexpr uint32_t kCollisionFilterAllCategories = 0xFFFF;
static constexpr uint32_t kCollisionFilterAllMasks      = 0xFFFF;

// Numerical Tolerances

// Array Indices and Counts
static constexpr float kRestitutionBuffer  = 0.5f; // Default restitution/bounce
static constexpr size_t kVertexStrideBytes = 12;   // 3 floats (x, y, z) * 4 bytes
static constexpr float kDefaultBodySize    = 0.5f;

// Jolt Physics collision masks (all layers/categories)
static constexpr uint32_t kJoltAllLayerBits = 0xFFFF;

// Buffer allocation increments

// --- Lifecycle: Deallocation ---
PyType_DeclareSlot_Status CPH_PyWorld_Traverse(CPH_PhysicsWorldObject *self, visitproc visit,
                                               void *arg) {
    // 1. Visit the type itself (Required for all heap types)
    Py_VISIT(Py_TYPE(self));

    // 2. If you add any other PyObject* members to your struct in the future,
    // you MUST visit them here.

    return 0;
}

PyType_DeclareSlot_Status CPH_PyWorld_Clear([[maybe_unused]] CPH_PhysicsWorldObject *self) {
    // Currently nothing to clear.
    return 0;
}

PyType_DeclareSlot_Void CPH_PyWorld_Dealloc(CPH_PhysicsWorldObject *self) {
    auto tp = Py_TYPE(self);

    // 1. The GC "Safety Shield"
    // Use the check-then-untrack to avoid the 'already untracked' abort
    if (PyObject_GC_IsTracked(reinterpret_cast<PyObject *>(self))) {
        PyObject_GC_UnTrack(self);
    }

    // 2. Weakref cleanup
    if (self->weakreflist != nullptr) {
        PyObject_ClearWeakRefs(reinterpret_cast<PyObject *>(self));
    }

    // Clean up local parsers
    if (self->parsers != nullptr) {

        delete self->parsers;
    }

    // 3. The "Manual" work
    CPH_PyWorld_FreeMembers(self);

    // 4. Final destruction
    // For Heap Types, we use the type's free function
    tp->tp_free(reinterpret_cast<PyObject *>(self));

    // 5. Release the type itself
    Py_DECREF(tp);
}
// --- Lifecycle: Initialization ---

// Orchestrator function
PyType_DeclareSlot_Status CPH_PyWorld_Init(CPH_PhysicsWorldObject *self, PyObject *args,
                                           PyObject *kwds) {
    if (self->physics.system != nullptr) {
        // Please don't call __init__() again.
        PyErr_SetString(PyExc_RuntimeError,
                        "PhysicsWorld instance has already been initialized and "
                        "cannot be re-initialized.");
        return -1;
    }
    if (self->parsers == nullptr) {
        self->parsers = CPH::NewParserGroup<CPH_WorldParsers>();
        if (self->parsers == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
        self->parsers->Init();
    }

    PyObject *settings_dict         = nullptr;
    PyObject *bodies_list           = nullptr;
    CPH_AUTO_DECREF PyObject *baked = nullptr;
    size_t baked_count              = 0;
    CPH_GravityVector gravity;
    CPH_WorldSettings settings;
    size_t start_idx = 0;

    if (!self->parsers->WorldInit.Parse(args, kwds, settings_dict, bodies_list)) {
        return -1;
    }

    // 1. Initial State
    CPH_PhysicsWorldInitState(&self->physics);
    self->view_shape[0] = self->view_shape[1] = 0;
    self->view_strides[0] = self->view_strides[1] = 0;

    // 2. Settings & Jolt Init
    if (CPH_InitSettings(self, settings_dict, &gravity, &settings) < 0) {
        goto fail;
    }

    if (CPH_PhysicsWorldInitJolt(&self->physics, settings, gravity) < 0) {
        if (!PyErr_Occurred()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "Failed to initialize native Jolt world resources.");
        }
        goto fail;
    }

    if (CPH_VerifyAbiAlignment(self->physics.body_interface) < 0) {
        PyErr_SetString(PyExc_RuntimeError, "JoltC ABI mismatch: position precision check failed.");
        goto fail;
    }

    if (!CPH_PhysicsWorldAttachContactListener(&self->physics)) {
        PyErr_NoMemory();
        goto fail;
    }

    // 3. Bake & Buffers
    if (bodies_list && bodies_list != Py_None) {
        PyObject *st_helper = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)))->helper;
        CPH_AUTO_DECREF PyObject *bake_func = PyObject_GetAttrString(st_helper, "bake_scene");
        baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, nullptr);
        if (!baked) {
            goto fail;
        }
        if (CPH_GetBakedSceneCount(baked, &baked_count) < 0) {
            goto fail;
        }
    }

    if (CPH_AllocateBuffers(&self->physics, settings.max_bodies, baked_count) < 0) {
        PyErr_NoMemory();
        goto fail;
    }

    // 4. Constraints & Data Loading
    if (CPH_AllocateConstraints(&self->physics) < 0) {
        PyErr_NoMemory();
        goto fail;
    }

    if (baked && CPH_LoadBakedScene(self, baked) < 0) {
        goto fail;
    }

    start_idx = atomic_load_explicit(&self->physics.count, memory_order_relaxed);
    CPH_InitializeFreeSlots(&self->physics, start_idx);

    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    CPH_SyncShadowBuffers(&self->physics);
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
    return 0;

fail:
    CPH_PyWorld_FreeMembers(self);
    return -1;
}
// Translate native status codes into Python exceptions at the binding boundary.
static void set_physics_world_error(CPH_PhysicsWorldObject *self, PhysicsWorldStatus status) {
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
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return nullptr;
    case PHYSICS_BODY_OPERATION_STATUS_INVALID_ARGUMENT:
        PyErr_SetString(PyExc_RuntimeError, "Invalid native physics body operation");
        return nullptr;
    }
    PyErr_SetString(PyExc_RuntimeError, "Unknown native physics body operation status");
    return nullptr;
}

static PyObject *run_body_operation(CPH_PhysicsWorldObject *self,
                                    const CPH_PhysicsBodyOperation *operation) {
    PhysicsBodyOperationStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldApplyBodyOperation(&self->physics, operation);
    Py_END_ALLOW_THREADS return body_operation_status_result(status);
}

static void set_creation_error(CPH_PhysicsWorldObject *self, PhysicsWorldCreationStatus status,
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
PyCFunction_DeclareMethod CPH_PyWorld_ApplyImpulse(CPH_PhysicsWorldObject *self,
                                                   PyObject *const *args, size_t nargsf,
                                                   PyObject *kwnames) {
    uint64_t h_raw;
    float x;
    float y;
    float z;

    if (!self->parsers->Impulse.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y, z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "Impulse");

    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_IMPULSE,
        .handle = static_cast<BodyHandle>(h_raw),
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_ApplyImpulseAt(CPH_PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    uint64_t h_raw;
    float ix;
    float iy;
    float iz;
    JPH_Real px;
    JPH_Real py;
    JPH_Real pz;

    if (!self->parsers->ImpulseAt.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, ix, iy,
                                        iz, px, py, pz)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_VEC3(ix, iy, iz, "Impulse");
    CPH_VALIDATE_FINITE_VEC3(px, py, pz, "Impulse position");

    const CPH_PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_APPLY_IMPULSE_AT,
        .handle   = static_cast<BodyHandle>(h_raw),
        .position = {px, py, pz},
        .vector   = {ix, iy, iz},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_ApplyAngularImpulse(CPH_PhysicsWorldObject *self,
                                                          PyObject *const *args, size_t nargsf,
                                                          PyObject *kwnames) {
    uint64_t h_raw;
    float x;
    float y;
    float z;

    if (!self->parsers->AngImpulse.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y,
                                         z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "Angular impulse");

    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_ANGULAR_IMPULSE,
        .handle = static_cast<BodyHandle>(h_raw),
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_ApplyForce(CPH_PhysicsWorldObject *self,
                                                 PyObject *const *args, size_t nargsf,
                                                 PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;

    if (!self->parsers->Force.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y, z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "Force");

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_FORCE,
        .handle = static_cast<BodyHandle>(h_raw),
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_ApplyTorque(CPH_PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;

    if (!self->parsers->Torque.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y, z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "Torque");

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_APPLY_TORQUE,
        .handle = static_cast<BodyHandle>(h_raw),
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetGravity(CPH_PhysicsWorldObject *self,
                                                 PyObject *const *args, size_t nargsf,
                                                 PyObject *kwnames) {
    float x;
    float y;
    float z;

    if (!self->parsers->Gravity.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, x, y, z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "Gravity");

    const JPH_Vec3 gravity = {x, y, z};
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldSetGravity(&self->physics, gravity);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod CPH_PyWorld_GetGravity(CPH_PhysicsWorldObject *self,
                                                 PyObject *ignored [[maybe_unused]]) {
    JPH_Vec3 gravity;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldGetGravity(&self->physics, &gravity);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return CPH_PY_BUILD_TUPLE(CPH_PY_BUILD_VALUE(gravity.x), CPH_PY_BUILD_VALUE(gravity.y),
                              CPH_PY_BUILD_VALUE(gravity.z));
}

PyCFunction_DeclareMethod CPH_PyWorld_GetBodyStats(CPH_PhysicsWorldObject *self,
                                                   PyObject *const *args, size_t nargsf,
                                                   PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->HOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    CPH_PhysicsBodyStats stats;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetBodyStats(&self->physics, static_cast<BodyHandle>(h_raw), &stats);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }

    return CPH_PY_BUILD_TUPLE(
        CPH_PY_BUILD_TUPLE(stats.position.x, stats.position.y, stats.position.z),
        CPH_PY_BUILD_TUPLE(stats.rotation.x, stats.rotation.y, stats.rotation.z, stats.rotation.w),
        CPH_PY_BUILD_TUPLE(stats.linear_velocity.x, stats.linear_velocity.y,
                           stats.linear_velocity.z));
}
PyCFunction_DeclareMethod CPH_PyWorld_ApplyBuoyancy(CPH_PhysicsWorldObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {
    uint64_t h_raw;
    double surface_y;
    float buoyancy  = 1.0f;
    float lin_drag  = kDefaultLinearDrag;
    float ang_drag  = kDefaultAngularDrag;
    float dt        = kDefaultFrameTime;
    PyObject *o_vel = nullptr;

    if (!self->parsers->Buoy.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, surface_y,
                                   buoyancy, lin_drag, ang_drag, dt, o_vel)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_FLOAT(buoyancy, "buoyancy");
    CPH_VALIDATE_FINITE_FLOAT(lin_drag, "linear drag");
    CPH_VALIDATE_FINITE_FLOAT(ang_drag, "angular drag");
    CPH_VALIDATE_FINITE_FLOAT(dt, "dt");

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    if (o_vel && o_vel != Py_None) {
        if (!CPH_parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
            return nullptr;
        }
        CPH_VALIDATE_FINITE_VEC3(vx, vy, vz, "fluid velocity");
    }

    const JPH_Vec3 fluid_velocity = {vx, vy, vz};
    bool submerged                = false;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldApplyBuoyancy(
        &self->physics, static_cast<BodyHandle>(h_raw), static_cast<JPH_Real>(surface_y), buoyancy,
        lin_drag, ang_drag, fluid_velocity, dt, &submerged);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyBool_FromLong(static_cast<int>(submerged));
}

PyCFunction_DeclareMethod CPH_PyWorld_ApplyBuoyancyBatch(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    PyObject *o_handles = nullptr;
    PyObject *o_vel     = nullptr;
    JPH_Real surface_y  = 0.0;
    float buoyancy      = 1.0f;
    float lin_drag      = kDefaultLinearDrag;
    float ang_drag      = kDefaultAngularDrag;
    float dt            = kDefaultFrameTime;

    if (!self->parsers->BatchBuoy.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_handles,
                                        surface_y, buoyancy, lin_drag, ang_drag, dt, o_vel)) {
        return nullptr;
    }

    Py_buffer handles_view;
    if (PyObject_GetBuffer(o_handles, &handles_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (CPH_UNLIKELY(handles_view.itemsize != 8 && handles_view.len % 8 != 0)) {
        PyBuffer_Release(&handles_view);
        PyErr_SetString(PyExc_ValueError, "Handle buffer must be uint64 array");
        return nullptr;
    }

    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;
    if (o_vel && o_vel != Py_None && !CPH_parse_vec3_direct(o_vel, &vx, &vy, &vz)) {
        PyBuffer_Release(&handles_view);
        return nullptr;
    }

    const size_t handle_count = static_cast<size_t>(handles_view.len) / sizeof(BodyHandle);
    if (handle_count == 0) {
        PyBuffer_Release(&handles_view);
        Py_RETURN_NONE;
    }

    BodyHandle *handles = static_cast<BodyHandle *>(CPH_PyAlloc(handle_count * sizeof(BodyHandle)));
    if (!handles) {
        PyBuffer_Release(&handles_view);
        return PyErr_NoMemory();
    }
    memcpy(handles, handles_view.buf, handle_count * sizeof(BodyHandle));
    PyBuffer_Release(&handles_view);

    const JPH_Vec3 fluid_velocity = {vx, vy, vz};
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldApplyBuoyancyBatch(&self->physics, handles, handle_count, surface_y,
                                           buoyancy, lin_drag, ang_drag, fluid_velocity, dt);
    Py_END_ALLOW_THREADS CPH_PyFree(handles);
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod CPH_PyWorld_SaveState(CPH_PhysicsWorldObject *self,
                                                PyObject *unused [[maybe_unused]]) {
    uint8_t *snapshot    = nullptr;
    size_t snapshot_size = 0;
    PhysicsWorldSnapshotStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldSaveSnapshot(&self->physics, &snapshot, &snapshot_size);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_SNAPSHOT_STATUS_OK) {
        return snapshot_status_error(status, 0, 0);
    }
    if (snapshot_size > static_cast<size_t>(PY_SSIZE_T_MAX)) {
        CPH_Free(snapshot);
        PyErr_SetString(PyExc_OverflowError, "Snapshot is too large for a Python bytes object");
        return nullptr;
    }

    PyObject *bytes = PyBytes_FromStringAndSize(reinterpret_cast<const char *>(snapshot),
                                                static_cast<Py_ssize_t>(snapshot_size));
    CPH_Free(snapshot);
    return bytes;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod CPH_PyWorld_LoadState(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject *state_obj    = nullptr;

    if (!self->parsers->LoadState.Parse(args, nargs, kwnames, state_obj)) {
        return nullptr;
    }

    Py_buffer view{};
    if (PyObject_GetBuffer(state_obj, &view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    const size_t data_size = static_cast<size_t>(view.len);
    uint8_t *snapshot      = static_cast<uint8_t *>(CPH_Alloc(data_size ? data_size : 1));
    if (!snapshot) {
        PyBuffer_Release(&view);
        return PyErr_NoMemory();
    }
    if (data_size != 0) {
        memcpy(snapshot, view.buf, data_size);
    }
    PyBuffer_Release(&view);

    size_t saved_count    = 0;
    size_t world_capacity = 0;
    size_t saved_capacity = 0;
    PhysicsWorldSnapshotStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldLoadSnapshot(
        &self->physics, snapshot, data_size, &saved_count, &world_capacity, &saved_capacity);
    Py_END_ALLOW_THREADS CPH_Free(snapshot);
    if (status != PHYSICS_WORLD_SNAPSHOT_STATUS_OK) {
        return snapshot_status_error(status, world_capacity, saved_capacity);
    }

    self->view_shape[0] = static_cast<Py_ssize_t>(saved_count);
    Py_RETURN_NONE;
}

typedef struct {
    PyThreadState *tstate;
    bool once;
} CPH_PyThreadCtx;

[[gnu::always_inline]]
static inline void internal_auto_gil_restore(CPH_PyThreadCtx *ctx) {
    if (ctx->tstate) {
        PyEval_RestoreThread(ctx->tstate);
        ctx->tstate = nullptr;
    }
}

#define CPH_WITH_ALLOW_THREADS                                                                     \
    for ([[gnu::cleanup(internal_auto_gil_restore)]] CPH_PyThreadCtx _ctx = {PyEval_SaveThread(),  \
                                                                             true};                \
         _ctx.once; _ctx.once                                             = false)

[[gnu::flatten]]
PyCFunction_DeclareMethod CPH_PyWorld_Step(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                           size_t nargsf, PyObject *kwnames) {
    float dt = kDefaultFrameTime;

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (!self->parsers->Step.Parse(args, nargs, kwnames, dt)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_FLOAT(dt, "dt");

    // --- PHASE 0: RE-ENTRANCY GUARD ---
    if (atomic_load_explicit(&self->physics.is_stepping, memory_order_acquire)) {
        PyErr_SetString(PyExc_RuntimeError, "Concurrent step detected.");
        return nullptr;
    }

    // --- PHASE 1: SHADOW STATE LOCK-DOWN ---
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    // ANTI-STARVATION: Yield to waiting Python threads (Getters/Mutators)
    // This is necessary for performance.
    while (atomic_load_explicit(&self->physics.waiting_threads, memory_order_relaxed) > 0) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        CPH_WITH_ALLOW_THREADS { CPH_CpuYield(); }
        CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    }

    atomic_thread_fence(memory_order_acquire);

    // Raise flags
    atomic_store_explicit(&self->physics.is_stepping, true, memory_order_relaxed);
    atomic_store_explicit(&self->physics.step_requested, true, memory_order_relaxed);

    // Drain in-flight queries
    if (atomic_load_explicit(&self->physics.active_queries, memory_order_acquire) > 0) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        CPH_WITH_ALLOW_THREADS {
            CPH_NATIVE_MUTEX_LOCK(self->physics.step_sync.mutex);
            while (atomic_load_explicit(&self->physics.active_queries, memory_order_relaxed) > 0) {
                CPH_NATIVE_COND_WAIT(self->physics.step_sync.cond, self->physics.step_sync.mutex);
            }
            CPH_NATIVE_MUTEX_UNLOCK(self->physics.step_sync.mutex);
        }
        CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    }

    // Transfer the active command queue to the native step core.
    size_t captured_count = 0;
    PhysicsCommand *captured_queue =
        CPH_PhysicsWorldDetachCommandQueue(&self->physics, &captured_count);

    atomic_store_explicit(&self->physics.contact_atomic_idx, 0, memory_order_relaxed);
    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    // --- PHASE 2: NATIVE JOLT CRUNCH (GIL Released) ---
    CPH_WITH_ALLOW_THREADS {
        CPH_PhysicsWorldStepNative(&self->physics, dt, captured_queue, captured_count);
    }

    // --- PHASE 3: FINALIZATION ---
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    const size_t final_count = CPH_PhysicsWorldFinalizeStep(&self->physics, dt);
    self->view_shape[0]      = static_cast<Py_ssize_t>(final_count);

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    Py_RETURN_NONE;
}
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod CPH_PyWorld_CreateConvexHull(CPH_PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {

    // 1. FAST PARSE
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_points   = nullptr;
    int motion_type      = 2;
    float mass           = -1.0f;
    float friction       = kDefaultFriction;
    float restitution    = 0.0f;
    uint64_t user_data   = 0;
    uint32_t category    = kCollisionFilterAllCategories;
    uint32_t mask        = kCollisionFilterAllMasks;
    uint32_t material_id = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;

    if (!self->parsers->ConvexHull.Parse(
            args, PyVectorcall_NARGS(nargsf), kwnames, o_pos, o_rot, o_points, motion_type, mass,
            user_data, is_sensor, category, mask, material_id, friction, restitution, use_ccd)) {
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
    if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz) ||
        !CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    CPH_VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");

    Py_buffer points_view;
    if (PyObject_GetBuffer(o_points, &points_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    size_t num_points = points_view.len / kVertexStrideBytes;
    if (CPH_UNLIKELY(num_points < 3)) {
        PyBuffer_Release(&points_view);
        return PyErr_Format(PyExc_ValueError, "Need >= 3 points");
    }

    // 3. Native shape creation and world staging (the buffer remains pinned).
    const CPH_PhysicsWorldBodyCreateRequest request = {
        .position    = {px, py, pz},
        .rotation    = {rx, ry, rz, rw},
        .config      = {mass, friction, restitution, static_cast<int>(is_sensor),
                        static_cast<int>(use_ccd), motion_type, kNoAllowedDofsOverride},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldCreateConvexHull(
        &self->physics, static_cast<const float *>(points_view.buf), num_points, &request, &handle,
        &body_count, &native_status);
    Py_END_ALLOW_THREADS PyBuffer_Release(&points_view);

    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, "Convex Hull build failed",
                           "Failed to create BodySettings");
        return nullptr;
    }
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);
    return PyLong_FromUnsignedLongLong(handle);
}

// Parse Python compound-part objects into a native request buffer.
static CPH_PhysicsWorldCompoundPart *parse_compound_parts(PyObject *parts, size_t *out_count) {
    if (!PyList_Check(parts)) {
        PyErr_SetString(PyExc_TypeError, "Compound parts must be a list");
        return nullptr;
    }

    const Py_ssize_t part_count = PyList_GET_SIZE(parts);
    if (part_count == 0) {
        PyErr_SetString(PyExc_ValueError, "Compound shape must have at least one part");
        return nullptr;
    }
    if (static_cast<size_t>(part_count) > SIZE_MAX / sizeof(CPH_PhysicsWorldCompoundPart)) {
        PyErr_NoMemory();
        return nullptr;
    }

    CPH_PhysicsWorldCompoundPart *buffer = static_cast<CPH_PhysicsWorldCompoundPart *>(
        CPH_PyAlloc(static_cast<size_t>(part_count) * sizeof(CPH_PhysicsWorldCompoundPart)));
    if (!buffer) {
        PyErr_NoMemory();
        return nullptr;
    }

    for (Py_ssize_t i = 0; i < part_count; i++) {
        PyObject *item = PyList_GET_ITEM(parts, i);
        if (!PyTuple_Check(item) || PyTuple_GET_SIZE(item) < 4) {
            CPH_PyFree(buffer);
            PyErr_Format(PyExc_ValueError, "Part %zd must be a tuple(pos, rot, type, size)", i);
            return nullptr;
        }

        PyObject *part_position = PyTuple_GET_ITEM(item, 0);
        PyObject *part_rotation = PyTuple_GET_ITEM(item, 1);
        const long type         = PyLong_AsLong(PyTuple_GET_ITEM(item, 2));
        PyObject *part_size     = PyTuple_GET_ITEM(item, 3);
        if (PyErr_Occurred()) {
            CPH_PyFree(buffer);
            return nullptr;
        }

        buffer[i] = CPH_PhysicsWorldCompoundPart{.shape_type = static_cast<int>(type)};
        if (PyTuple_Check(part_position) && PyTuple_GET_SIZE(part_position) == 3) {
            buffer[i].local_position.x =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_position, 0)));
            buffer[i].local_position.y =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_position, 1)));
            buffer[i].local_position.z =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_position, 2)));
        }
        if (PyTuple_Check(part_rotation) && PyTuple_GET_SIZE(part_rotation) == 4) {
            buffer[i].local_rotation.x =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 0)));
            buffer[i].local_rotation.y =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 1)));
            buffer[i].local_rotation.z =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 2)));
            buffer[i].local_rotation.w =
                static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_rotation, 3)));
        } else {
            buffer[i].local_rotation.w = 1.0f;
        }
        if (PyTuple_Check(part_size)) {
            const Py_ssize_t size_count = PyTuple_GET_SIZE(part_size);
            for (Py_ssize_t j = 0; j < 4 && j < size_count; j++) {
                buffer[i].shape_params[j] =
                    static_cast<float>(PyFloat_AsDouble(PyTuple_GET_ITEM(part_size, j)));
            }
        } else if (PyFloat_Check(part_size) || PyLong_Check(part_size)) {
            buffer[i].shape_params[0] = static_cast<float>(PyFloat_AsDouble(part_size));
        }
    }

    if (PyErr_Occurred()) {
        CPH_PyFree(buffer);
        return nullptr;
    }
    *out_count = static_cast<size_t>(part_count);
    return buffer;
}

// Orchestrator
PyCFunction_DeclareMethod CPH_PyWorld_CreateCompoundBody(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_parts    = nullptr;
    int motion_type      = 2;
    float mass           = -1.0f;
    float friction       = kDefaultFriction;
    float restitution    = 0.0f;
    uint64_t user_data   = 0;
    uint32_t category    = kCollisionFilterAllCategories;
    uint32_t mask        = kCollisionFilterAllMasks;
    uint32_t material_id = 0;
    bool is_sensor       = false;
    bool use_ccd         = false;

    if (!self->parsers->Compound.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_pos, o_rot,
                                       o_parts, motion_type, mass, user_data, is_sensor, category,
                                       mask, material_id, friction, restitution, use_ccd)) {
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
    if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz) ||
        !CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }
    if (CPH_UNLIKELY(!PyList_Check(o_parts))) {
        PyErr_SetString(PyExc_TypeError, "'parts' must be list");
        return nullptr;
    }

    // 3. Parse the Python list; native code builds the shape and stages the body.
    size_t part_count;
    CPH_PhysicsWorldCompoundPart *parts = parse_compound_parts(o_parts, &part_count);
    if (!parts) {
        return nullptr;
    }
    const CPH_PhysicsWorldBodyCreateRequest request = {
        .position    = {px, py, pz},
        .rotation    = {rx, ry, rz, rw},
        .config      = {mass, friction, restitution, static_cast<int>(is_sensor),
                        static_cast<int>(use_ccd), motion_type, kNoAllowedDofsOverride},
        .category    = category,
        .mask        = mask,
        .material_id = material_id,
        .user_data   = user_data,
    };
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldCreateCompoundBody(
        &self->physics, parts, part_count, &request, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS CPH_PyFree(parts);

    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, "Failed to create compound shape",
                           "Failed to create compound BodySettings");
        return nullptr;
    }
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);
    return PyLong_FromUnsignedLongLong(handle);
}

// Main Orchestrator
PyCFunction_DeclareMethod CPH_PyWorld_CreateBody(CPH_PhysicsWorldObject *self,
                                                 PyObject *const *args, size_t nargsf,
                                                 PyObject *kwnames) {
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
    uint32_t category     = kCollisionFilterAllCategories;
    uint32_t mask         = kCollisionFilterAllMasks;
    uint32_t material_id  = 0;
    uint32_t allowed_dofs = kNoAllowedDofsOverride;
    uint64_t user_data    = 0;
    bool is_sensor        = false;
    bool use_ccd          = false;
    PyObject *o_pos       = nullptr;
    PyObject *o_rot       = nullptr;
    PyObject *o_size      = nullptr;

    // 2. TARGET MAPPING (Explicitly mapped via Enum)
    // Using explicit indices [IDX_...] makes this reorder-proof.

    // 3. THE FAST PARSE
    // We pass the global 'BodyParser' defined in the God Init helper
    if (!self->parsers->Body.Parse(args, nargs, kwnames, o_pos, o_rot, o_size, shape_type,
                                   motion_type, user_data, is_sensor, mass, category, mask,
                                   friction, restitution, material_id, use_ccd, allowed_dofs)) {
        return nullptr;
    }

    // 4. CONVERT COMPLEX TYPES
    if (o_pos && o_pos != Py_None) {
        if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz)) {
            return nullptr; // PyErr already set
        }
        // GUARD: This is the exact point test_numerical_stability checks
        CPH_VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    }

    if (o_rot && o_rot != Py_None) {
        if (!CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
            return nullptr;
        }
        CPH_VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");
    }

    // Validation
    if (shape_type == 4 && motion_type != 0) {
        return PyErr_Format(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
    }

    // GUARD: Floats (Only check if they aren't the 'unset' -1.0 default)
    if (mass != -1.0f) {
        CPH_VALIDATE_FINITE_FLOAT(mass, "mass");
    }
    if (friction != -1.0f) {
        CPH_VALIDATE_FINITE_FLOAT(friction, "friction");
    }
    if (restitution != -1.0f) {
        CPH_VALIDATE_FINITE_FLOAT(restitution, "restitution");
    }

    // Resolve defaults and retain parsed size data at the Python boundary.
    float resolved_friction;
    float resolved_restitution;
    CPH_PhysicsWorldResolveMaterial(&self->physics, material_id, friction, restitution,
                                    &resolved_friction, &resolved_restitution);
    float size[4] = {kDefaultBodySize, kDefaultBodySize, kDefaultBodySize, 0.0f};
    CPH_ParseBodySize(o_size, size);
    CPH_VALIDATE_FINITE_VEC4(size[0], size[1], size[2], size[3], "Shape size");

    const CPH_PhysicsWorldBodyCreateRequest request = {
        .shape_type   = shape_type,
        .shape_params = {size[0], size[1], size[2], size[3]},
        .position     = {px, py, pz},
        .rotation     = {rx, ry, rz, rw},
        .config       = {mass, resolved_friction, resolved_restitution, static_cast<int>(is_sensor),
                         static_cast<int>(use_ccd), motion_type, allowed_dofs},
        .category     = category,
        .mask         = mask,
        .material_id  = material_id,
        .user_data    = user_data,
    };

    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldCreateBody(&self->physics, &request, &handle, &body_count, &native_status);
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
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod CPH_PyWorld_CreateBodiesBatch(CPH_PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {

    // 1. FAST PARSE (Zero Lock Contention)
    PyObject *py_pos   = nullptr;
    PyObject *py_sizes = nullptr;
    int shape_type     = 0;
    int motion_type    = 2;

    if (!self->parsers->BatchCreate.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, py_pos,
                                          py_sizes, shape_type, motion_type)) {
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
        sizeof(CPH_PosStride) + sizeof(CPH_ShapeParams) + sizeof(BodyHandle) + sizeof(PyObject *);
    if (static_cast<size_t>(batch_count) > SIZE_MAX / item_size) {
        return PyErr_Format(PyExc_OverflowError, "Batch is too large");
    }
    void *arena = CPH_PyAlloc(static_cast<size_t>(batch_count) * item_size);
    if (CPH_UNLIKELY(!arena)) {
        return PyErr_NoMemory();
    }

    CPH_PosStride *positions = static_cast<CPH_PosStride *>(arena);
    CPH_ShapeParams *sizes   = reinterpret_cast<CPH_ShapeParams *>(positions + batch_count);
    BodyHandle *handles      = reinterpret_cast<BodyHandle *>(sizes + batch_count);
    PyObject **results       = reinterpret_cast<PyObject **>(handles + batch_count);
    for (Py_ssize_t i = 0; i < batch_count; i++) {
        if (!CPH_parse_py_vec3(PyList_GET_ITEM(py_pos, i), &positions[i])) {
            CPH_PyFree(arena);
            return nullptr;
        }
        CPH_ParseBodySize(PyList_GET_ITEM(py_sizes, i), sizes[i].p);
        if (PyErr_Occurred()) {
            CPH_PyFree(arena);
            return nullptr;
        }
    }

    size_t body_count                = 0;
    PhysicsWorldStatus native_status = PHYSICS_WORLD_STATUS_OK;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldCreateBodiesBatch(
        &self->physics, positions, sizes, static_cast<size_t>(batch_count), shape_type, motion_type,
        handles, &body_count, &native_status);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        CPH_PyFree(arena);
        if (status == PHYSICS_WORLD_CREATION_STATUS_NATIVE_ERROR &&
            native_status == PHYSICS_WORLD_STATUS_LIMIT_REACHED) {
            PyErr_SetString(PyExc_RuntimeError, "Batch exceeds world limit");
        } else {
            set_creation_error(self, status, native_status, "Batch shape creation failed", nullptr);
        }
        return nullptr;
    }
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);

    for (Py_ssize_t i = 0; i < batch_count; i++) {
        results[i] = handles[i] ? CPH_PY_BUILD_VALUE(handles[i]) : CPH_PY_BUILD_VALUE(nullptr);
    }
    PyObject *result_list = CPH_PyBuildPackList(static_cast<size_t>(batch_count), results);
    CPH_PyFree(arena);
    return result_list;
}

/**
 * Main Orchestrator
 */

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyCFunction_DeclareMethod CPH_PyWorld_CreateMeshBody(CPH_PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    // 1. FAST PARSE
    PyObject *o_pos     = nullptr;
    PyObject *o_rot     = nullptr;
    PyObject *o_verts   = nullptr;
    PyObject *o_indices = nullptr;
    uint64_t user_data  = 0;
    uint32_t cat        = kCollisionFilterAllCategories;
    uint32_t mask       = kCollisionFilterAllMasks;

    if (!self->parsers->Mesh.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_pos, o_rot, o_verts,
                                   o_indices, user_data, cat, mask)) {
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
    if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz) ||
        !CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    Py_buffer v_view{};
    Py_buffer i_view{};
    if (PyObject_GetBuffer(o_verts, &v_view, PyBUF_SIMPLE) != 0 ||
        PyObject_GetBuffer(o_indices, &i_view, PyBUF_SIMPLE) != 0) {
        if (v_view.buf) {
            PyBuffer_Release(&v_view);
        }
        return nullptr;
    }

    if (CPH_UNLIKELY(v_view.len % kVertexStrideBytes != 0 ||
                     i_view.len % kVertexStrideBytes != 0)) {
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
    const size_t vertex_count     = (v_view.len / kVertexStrideBytes);
    const size_t triangle_count   = (i_view.len / kVertexStrideBytes);
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldCreateMeshBody(
        &self->physics, static_cast<const float *>(v_view.buf), vertex_count,
        static_cast<const uint32_t *>(i_view.buf), triangle_count, JPH_RVec3{px, py, pz},
        JPH_Quat{rx, ry, rz, rw}, cat, mask, user_data, &handle, &body_count, &native_status,
        bad_indices);
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
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod CPH_PyWorld_DestroyBody(CPH_PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {
    uint64_t raw_handle;

    if (!self->parsers->Destroy.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, raw_handle)) {
        return nullptr;
    }

    PhysicsBodyOperationStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldQueueBodyDestroy(&self->physics, static_cast<BodyHandle>(raw_handle));
    Py_END_ALLOW_THREADS return body_operation_status_result(status);
}

PyCFunction_DeclareMethod CPH_PyWorld_DestroyBodiesBatch(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    PyObject *py_handles = nullptr;

    if (!self->parsers->BatchDestroy.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, py_handles)) {
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
    if (static_cast<size_t>(input_count) > SIZE_MAX / sizeof(BodyHandle)) {
        Py_DECREF(sequence);
        return PyErr_NoMemory();
    }

    BodyHandle *handles = static_cast<BodyHandle *>(
        CPH_PyAlloc(static_cast<size_t>(input_count) * sizeof(BodyHandle)));
    if (!handles) {
        Py_DECREF(sequence);
        return PyErr_NoMemory();
    }
    PyObject **items    = PySequence_Fast_ITEMS(sequence);
    size_t handle_count = 0;
    for (Py_ssize_t i = 0; i < input_count; i++) {
        const uint64_t raw_handle = PyLong_AsUnsignedLongLong(items[i]);
        if (PyErr_Occurred()) {
            PyErr_Clear();
            continue;
        }
        handles[handle_count++] = static_cast<BodyHandle>(raw_handle);
    }
    Py_DECREF(sequence);

    PhysicsBodyOperationStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldQueueBodyDestroyBatch(&self->physics, handles, handle_count);
    Py_END_ALLOW_THREADS CPH_PyFree(handles);
    return body_operation_status_result(status);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetPosition(CPH_PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;

    if (!self->parsers->SetPos.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y, z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "SetPosition");

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_SET_POSITION,
        .handle   = static_cast<BodyHandle>(h_raw),
        .position = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetRotation(CPH_PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;
    float w;

    if (!self->parsers->SetRot.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y, z,
                                     w)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_QUAT(x, y, z, w, "SetRotation");

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_SET_ROTATION,
        .handle   = static_cast<BodyHandle>(h_raw),
        .rotation = {x, y, z, w},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetLinearVelocity(CPH_PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;

    if (!self->parsers->SetLinVel.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y,
                                        z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "LinearVelocity");

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_LINEAR_VELOCITY,
        .handle = static_cast<BodyHandle>(h_raw),
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetAngularVelocity(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    float x;
    float y;
    float z;

    if (!self->parsers->SetAngVel.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, x, y,
                                        z)) {
        return nullptr;
    }
    CPH_VALIDATE_FINITE_VEC3(x, y, z, "AngularVelocity");

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_ANGULAR_VELOCITY,
        .handle = static_cast<BodyHandle>(h_raw),
        .vector = {x, y, z},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_GetMotionType(CPH_PhysicsWorldObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->GetMotion.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    JPH_MotionType motion_type;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetMotionType(&self->physics, static_cast<BodyHandle>(h_raw), &motion_type);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(
            status, "Cannot query motion type of a body that has not been flushed to Jolt yet");
    }
    return PyLong_FromLong(static_cast<long>(motion_type));
}

PyCFunction_DeclareMethod CPH_PyWorld_SetMotionType(CPH_PhysicsWorldObject *self,
                                                    PyObject *const *args, size_t nargsf,
                                                    PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    int motion_type;

    if (!self->parsers->SetMotion.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw,
                                        motion_type)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_MOTION_TYPE,
        .handle = static_cast<BodyHandle>(h_raw),
        .value  = motion_type,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetUserData(CPH_PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    uint64_t data_raw;

    if (!self->parsers->SetUserData.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw,
                                          data_raw)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type      = PHYSICS_BODY_OP_SET_USER_DATA,
        .handle    = static_cast<BodyHandle>(h_raw),
        .user_data = data_raw,
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_GetUserData(CPH_PhysicsWorldObject *self,
                                                  PyObject *const *args, size_t nargsf,
                                                  PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->GetUserData.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    uint64_t user_data;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetUserData(&self->physics, static_cast<BodyHandle>(h_raw), &user_data);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyLong_FromUnsignedLongLong(user_data);
}

PyCFunction_DeclareMethod CPH_PyWorld_Activate(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;

    if (!self->parsers->Activate.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_ACTIVATE,
        .handle = static_cast<BodyHandle>(h_raw),
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_Deactivate(CPH_PhysicsWorldObject *self,
                                                 PyObject *const *args, size_t nargsf,
                                                 PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;

    // Note: Reusing ActivateParser as the schema is identical (single handle)
    if (!self->parsers->Activate.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_DEACTIVATE,
        .handle = static_cast<BodyHandle>(h_raw),
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetTransform(CPH_PhysicsWorldObject *self,
                                                   PyObject *const *args, size_t nargsf,
                                                   PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    PyObject *o_pos = nullptr;
    PyObject *o_rot = nullptr;

    if (!self->parsers->SetTrns.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, o_pos,
                                      o_rot)) {
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
    if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz)) {
        return nullptr;
    }
    if (!CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_VEC3(px, py, pz, "SetTransform position");
    CPH_VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "SetTransform rotation");

    // 3. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type     = PHYSICS_BODY_OP_SET_TRANSFORM,
        .handle   = static_cast<BodyHandle>(h_raw),
        .position = {px, py, pz},
        .rotation = {rx, ry, rz, rw},
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_SetCcd(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    uint64_t h_raw;
    bool enabled;

    if (!self->parsers->CCD.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, enabled)) {
        return nullptr;
    }

    // 2. ENGINE OPERATION
    const CPH_PhysicsBodyOperation operation = {
        .type   = PHYSICS_BODY_OP_SET_CCD,
        .handle = static_cast<BodyHandle>(h_raw),
        .value  = static_cast<int>(enabled),
    };
    return run_body_operation(self, &operation);
}

PyCFunction_DeclareMethod CPH_PyWorld_GetIndex(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->Activate.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    uint32_t index;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetIndex(&self->physics, static_cast<BodyHandle>(h_raw), &index);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyLong_FromUnsignedLong(index);
}

PyCFunction_DeclareMethod CPH_PyWorld_IsAlive(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                              size_t nargsf, PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->Activate.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    bool alive;
    Py_BEGIN_ALLOW_THREADS alive =
        CPH_PhysicsWorldIsAlive(&self->physics, static_cast<BodyHandle>(h_raw));
    Py_END_ALLOW_THREADS

        if (alive) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod CPH_PyWorld_IsActive(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                               size_t nargsf, PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->Activate.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    bool active;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldIsActive(&self->physics, static_cast<BodyHandle>(h_raw), &active);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (active) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

PyCFunction_DeclareMethod CPH_PyWorld_GetActiveIndices(CPH_PhysicsWorldObject *self,
                                                       PyObject *args [[maybe_unused]]) {
    uint32_t *indices   = nullptr;
    size_t active_count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetActiveIndices(&self->physics, &indices, &active_count);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }

    if (active_count > static_cast<size_t>(PY_SSIZE_T_MAX) / sizeof(uint32_t)) {
        CPH_Free(indices);
        return PyErr_NoMemory();
    }
    PyObject *bytes_obj =
        PyBytes_FromStringAndSize(reinterpret_cast<const char *>(indices),
                                  static_cast<Py_ssize_t>(active_count * sizeof(uint32_t)));
    CPH_Free(indices);
    return bytes_obj;
}

PyCFunction_DeclareMethod CPH_PyWorld_GetRenderState(CPH_PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    float alpha;

    if (!self->parsers->Render.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, alpha)) {
        return nullptr;
    }

    alpha        = fmaxf(0.0f, fminf(1.0f, alpha));
    float *state = nullptr;
    size_t count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetRenderState(&self->physics, alpha, &state, &count);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }

    const size_t byte_count = count * 7 * sizeof(float);
    if (byte_count > static_cast<size_t>(PY_SSIZE_T_MAX)) {
        CPH_Free(state);
        PyErr_SetString(PyExc_OverflowError, "Render state is too large for a Python bytes object");
        return nullptr;
    }
    PyObject *bytes_obj = PyBytes_FromStringAndSize(reinterpret_cast<const char *>(state),
                                                    static_cast<Py_ssize_t>(byte_count));
    CPH_Free(state);
    return bytes_obj;
}

PyCFunction_DeclareMethod CPH_PyWorld_SetCollisionFilter(CPH_PhysicsWorldObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    uint64_t h_raw;
    uint32_t category;
    uint32_t mask;

    if (!self->parsers->ColFilter.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw, category,
                                        mask)) {
        return nullptr;
    }

    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldSetCollisionFilter(
        &self->physics, static_cast<BodyHandle>(h_raw), category, mask);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(
            status, "Cannot update collision filters of a body until it has been flushed to Jolt");
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod CPH_PyWorld_RegisterMaterial(CPH_PhysicsWorldObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    uint32_t id;
    float friction    = kRestitutionBuffer;
    float restitution = 0.0f;

    if (!self->parsers->RegMat.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, id, friction,
                                     restitution)) {
        return nullptr;
    }

    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldRegisterMaterial(&self->physics, id, friction, restitution);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    Py_RETURN_NONE;
}

PyCFunction_DeclareMethod CPH_PyWorld_CreateHeightfield(CPH_PhysicsWorldObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {

    // 1. FAST PARSE & VALIDATION
    PyObject *o_pos      = nullptr;
    PyObject *o_rot      = nullptr;
    PyObject *o_scale    = nullptr;
    PyObject *o_heights  = nullptr;
    int grid_size        = 0;
    uint64_t user_data   = 0;
    uint32_t category    = kCollisionFilterAllCategories;
    uint32_t mask        = kCollisionFilterAllMasks;
    uint32_t material_id = 0;
    float friction       = kDefaultFriction;
    float restitution    = 0.0f;

    if (!self->parsers->Heightfield.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, o_pos, o_rot,
                                          o_scale, o_heights, grid_size, user_data, category, mask,
                                          material_id, friction, restitution)) {
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
    if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz) ||
        !CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw) ||
        !CPH_parse_vec3_direct(o_scale, &sx, &sy, &sz)) {
        return nullptr;
    }

    Py_buffer h_view;
    if (PyObject_GetBuffer(o_heights, &h_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (CPH_UNLIKELY(h_view.len != static_cast<Py_ssize_t>(static_cast<Py_ssize_t>(grid_size) *
                                                           grid_size * sizeof(float)))) {
        PyBuffer_Release(&h_view);
        return PyErr_Format(PyExc_ValueError, "Height buffer size mismatch");
    }

    // 2. Native heightfield construction and world staging (buffer stays pinned).
    BodyHandle handle;
    size_t body_count;
    PhysicsWorldStatus native_status = PHYSICS_WORLD_STATUS_OK;
    PhysicsWorldCreationStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldCreateHeightfieldBody(
        &self->physics, static_cast<const float *>(h_view.buf), static_cast<size_t>(grid_size),
        JPH_Vec3{sx, sy, sz}, JPH_RVec3{px, py, pz}, JPH_Quat{rx, ry, rz, rw}, friction,
        restitution, category, mask, material_id, user_data, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS PyBuffer_Release(&h_view);

    if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, "HeightField build failed",
                           "Failed to create HeightField BodySettings");
        return nullptr;
    }
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);
    return PyLong_FromUnsignedLongLong(handle);
}

PyCFunction_DeclareMethod CPH_PyWorld_GetDebugData(CPH_PhysicsWorldObject *self,
                                                   PyObject *const *args, size_t nargsf,
                                                   PyObject *kwnames) {
    bool draw_shapes       = true;
    bool draw_constraints  = true;
    bool draw_bounding_box = false;
    bool draw_centers      = false;
    bool wireframe         = true;

    if (!self->parsers->DebugData.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, draw_shapes,
                                        draw_constraints, draw_bounding_box, draw_centers,
                                        wireframe)) {
        return nullptr;
    }

    CPH_DebugVertex *lines     = nullptr;
    size_t line_count          = 0;
    CPH_DebugVertex *triangles = nullptr;
    size_t triangle_count      = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldBuildDebugData(
        &self->physics, draw_shapes, draw_constraints, draw_bounding_box, draw_centers, wireframe,
        &lines, &line_count, &triangles, &triangle_count);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (line_count > static_cast<size_t>(PY_SSIZE_T_MAX) / sizeof(CPH_DebugVertex) ||
        triangle_count > static_cast<size_t>(PY_SSIZE_T_MAX) / sizeof(CPH_DebugVertex)) {
        CPH_Free(lines);
        CPH_Free(triangles);
        PyErr_SetString(PyExc_OverflowError, "Debug data is too large for Python bytes objects");
        return nullptr;
    }

    PyObject *lines_bytes =
        PyBytes_FromStringAndSize(reinterpret_cast<const char *>(lines),
                                  static_cast<Py_ssize_t>(line_count * sizeof(CPH_DebugVertex)));
    PyObject *triangles_bytes = PyBytes_FromStringAndSize(
        reinterpret_cast<const char *>(triangles),
        static_cast<Py_ssize_t>(triangle_count * sizeof(CPH_DebugVertex)));
    CPH_Free(lines);
    CPH_Free(triangles);
    if (CPH_UNLIKELY(!lines_bytes || !triangles_bytes)) {
        Py_XDECREF(lines_bytes);
        Py_XDECREF(triangles_bytes);
        return nullptr;
    }

    PyObject *result = PyTuple_Pack(2, lines_bytes, triangles_bytes);
    Py_DECREF(lines_bytes);
    Py_DECREF(triangles_bytes);
    return result;
}

PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateSoftBody(CPH_PhysicsWorldObject *self,
                                                               PyObject *const *args, size_t nargsf,
                                                               PyObject *kwnames) {

    // 1. FAST PARSE
    PyObject *o_shared        = nullptr;
    PyObject *o_pos           = nullptr;
    PyObject *o_rot           = nullptr;
    uint64_t user_data        = 0;
    uint32_t category         = kCollisionFilterAllCategories;
    uint32_t mask             = kCollisionFilterAllMasks;
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

    if (!self->parsers->CreateSoftBody.Parse(
            args, PyVectorcall_NARGS(nargsf), kwnames, o_shared, o_pos, o_rot, user_data, category,
            mask, pressure, vertex_radius, linear_damping, num_iterations, max_linear_velocity,
            gravity_factor, friction, restitution, make_rot_identity, update_position,
            faces_double_sided)) {
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
    if (!CPH_parse_vec3_direct(o_pos, &px, &py, &pz) ||
        !CPH_parse_quat_direct(o_rot, &rx, &ry, &rz, &rw)) {
        return nullptr;
    }

    CPH_VALIDATE_FINITE_VEC3(px, py, pz, "Position");
    CPH_VALIDATE_FINITE_QUAT(rx, ry, rz, rw, "Rotation");

    CPH_ModuleState *state = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(o_shared,
                            reinterpret_cast<PyTypeObject *>(state->SoftBodySharedSettingsType))) {
        PyErr_SetString(PyExc_TypeError, "shared must be a SoftBodySharedSettings object");
        return nullptr;
    }
    auto py_shared = reinterpret_cast<CPH_SoftBodySharedSettingsObject *>(o_shared);

    // 3. Native Jolt settings construction and world staging.
    const CPH_PhysicsWorldSoftBodyCreateRequest request = {
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
    Py_BEGIN_ALLOW_THREADS status = CPH_PhysicsWorldCreateSoftBody(
        &self->physics, &request, &handle, &body_count, &native_status);
    Py_END_ALLOW_THREADS

        if (status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, status, native_status, nullptr,
                           "Failed to create SoftBodyCreationSettings");
        return nullptr;
    }
    self->view_shape[0] = static_cast<Py_ssize_t>(body_count);
    return PyLong_FromUnsignedLongLong(handle);
}

// Getters

PyCFunction_DeclareMethodFromModule CPH_PyWorld_GetSoftBodyVertexCount(CPH_PhysicsWorldObject *self,
                                                                       PyObject *const *args,
                                                                       size_t nargsf,
                                                                       PyObject *kwnames) {
    uint64_t handle;

    if (!self->parsers->HOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle)) {
        return nullptr;
    }

    uint32_t count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetSoftBodyVertexCount(&self->physics, handle, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return PyLong_FromUnsignedLong(count);
}

PyCFunction_DeclareMethodFromModule
CPH_PyWorld_GetSoftBodyVertexPosition(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                      size_t nargsf, PyObject *kwnames) {
    uint64_t handle;
    uint32_t index;

    if (!self->parsers->GetSbVertex.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle,
                                          index)) {
        return nullptr;
    }

    JPH_Vec3 position;
    uint32_t count = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetSoftBodyVertexPosition(&self->physics, handle, index, &position, &count);
    Py_END_ALLOW_THREADS if (status == PHYSICS_WORLD_RUNTIME_STATUS_OUT_OF_RANGE) {
        return PyErr_Format(PyExc_IndexError, "Vertex index %u out of bounds (count: %u)", index,
                            count);
    }
    if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    return CPH_PY_BUILD_TUPLE(position.x, position.y, position.z);
}

PyCFunction_DeclareMethodFromModule
CPH_PyWorld_GetSoftBodyLocalVertices(CPH_PhysicsWorldObject *self, PyObject *const *args,
                                     size_t nargsf, PyObject *kwnames) {
    uint64_t handle;

    if (!self->parsers->HOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle)) {
        return nullptr;
    }

    JPH_Vec3 *positions = nullptr;
    uint32_t count      = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldGetSoftBodyLocalVertices(&self->physics, handle, &positions, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (static_cast<size_t>(count) > static_cast<size_t>(PY_SSIZE_T_MAX) / sizeof(JPH_Vec3)) {
        CPH_Free(positions);
        PyErr_SetString(PyExc_OverflowError, "Soft body vertex data is too large");
        return nullptr;
    }

    PyObject *bytes = PyBytes_FromStringAndSize(
        reinterpret_cast<const char *>(positions),
        static_cast<Py_ssize_t>(static_cast<size_t>(count) * sizeof(JPH_Vec3)));
    CPH_Free(positions);
    return bytes;
}

PyCFunction_DeclareMethodFromModule CPH_PyWorld_GetSoftBodyVertices(CPH_PhysicsWorldObject *self,
                                                                    PyObject *const *args,
                                                                    size_t nargsf,
                                                                    PyObject *kwnames) {
    uint64_t h_raw;

    if (!self->parsers->HOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, h_raw)) {
        return nullptr;
    }

    CPH_SHADOW_LOCK(&self->physics.shadow_lock);
    CPH_BLOCK_UNTIL_NOT_STEPPING(self);

    uint32_t slot = 0;
    if (!CPH_ValidateHandle(self, h_raw, &slot)) {
        return nullptr;
    }

    const uint8_t state =
        atomic_load_explicit(&self->physics.slot_states[slot], memory_order_acquire);
    if (state != SLOT_SOFT_BODY) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        return PyErr_Format(PyExc_TypeError, "Handle does not belong to a soft body");
    }

    uint32_t dense_idx         = self->physics.slot_to_dense[slot];
    CPH_SoftBodyShadow *shadow = &self->physics.soft_shadows[dense_idx];

    if (!shadow->vertices) {
        CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
        return PyErr_Format(PyExc_RuntimeError, "Soft body shadow buffer missing");
    }

    // We can reuse BufferProxyObject, but we need to tell it to use THIS specific pointer
    // and THIS specific length, rather than the global positions array.
    CPH_ModuleState *st          = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    CPH_BufferProxyObject *proxy = PyObject_GC_New(
        CPH_BufferProxyObject, reinterpret_cast<PyTypeObject *>(st->BufferProxyType));
    proxy->owner = reinterpret_cast<PyObject *>(self);
    Py_INCREF(self);

    proxy->buf_type    = PROXY_DYNAMIC;
    proxy->dynamic_ptr = shadow->vertices;
    proxy->format      = JPH_REAL_STRING;
    proxy->itemsize    = sizeof(JPH_Real);
    proxy->stride      = 4; // PosStride
    proxy->shape[0]    = static_cast<Py_ssize_t>(shadow->num_vertices) * 4;

    atomic_fetch_add_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);

    PyObject_GC_Track(proxy);
    return reinterpret_cast<PyObject *>(proxy);
}

PyCFunction_DeclareMethod CPH_PyWorld_BenchmarkParse([[maybe_unused]] CPH_PhysicsWorldObject *self,
                                                     PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    constexpr size_t NUM_ARGS = 64;
    uint64_t values[NUM_ARGS] = {};

    auto nargs = PyVectorcall_NARGS(nargsf);
    if (CPH_UNLIKELY(!self->parsers->StressTest.Parse(
            args, nargs, kwnames, values[0], values[1], values[2], values[3], values[4], values[5],
            values[6], values[7], values[8], values[9], values[10], values[11], values[12],
            values[13], values[14], values[15], values[16], values[17], values[18], values[19],
            values[20], values[21], values[22], values[23], values[24], values[25], values[26],
            values[27], values[28], values[29], values[30], values[31], values[32], values[33],
            values[34], values[35], values[36], values[37], values[38], values[39], values[40],
            values[41], values[42], values[43], values[44], values[45], values[46], values[47],
            values[48], values[49], values[50], values[51], values[52], values[53], values[54],
            values[55], values[56], values[57], values[58], values[59], values[60], values[61],
            values[62], values[63]))) {
        return nullptr;
    }

    Py_RETURN_NONE;
}

/**
 * Benchmark for CULV_PY_BUILD_TUPLE using METH_NOARGS.
 * This eliminates all argument parsing overhead to isolate the
 * performance of the Culverin Fast Build engine.
 */
PyCFunction_DeclareMethod CPH_PyWorld_BenchmarkBuild([[maybe_unused]] PyObject *self,
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
    // It then uses CPH_PyBuildPackTuple to perform O(1) allocation and
    // O(N) reference stealing.
    constexpr float stfu = 200.0f;
    PyObject *result =
        CPH_PY_BUILD_TUPLE(i_val, f_val, d_val, s_val, b_val, 100, stfu,
                           "Lorem ipsum dolor sit amet consectetur adipiscing elit, sed do eiusmod "
                           "tempor incididunt ut labore et dolore magna aliqua.",
                           false);

    // 3. Error handling
    if (CPH_UNLIKELY(!result)) {
        // CPH_PyBuildPackTuple handles internal cleanup on failure
        return nullptr;
    }

    // Return the new reference to the caller
    return result;
}

// --- Ragdoll Settings Implementation ---
PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateRagdollSettings(CPH_PhysicsWorldObject *self,
                                                                      PyObject *const *args,
                                                                      size_t nargsf,
                                                                      PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *py_skel_obj = nullptr;

    if (!self->parsers->RagdollSettings.Parse(args, nargs, kwnames, py_skel_obj)) {
        return nullptr;
    }

    // --- 2. TYPE VALIDATION ---

    // Manual type check (replaces O! format string)
    CPH_ModuleState *st = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(py_skel_obj, reinterpret_cast<PyTypeObject *>(st->SkeletonType))) {
        PyErr_SetString(PyExc_TypeError, "skeleton must be a Skeleton object");
        return nullptr;
    }
    CPH_SkeletonObject *py_skel = reinterpret_cast<CPH_SkeletonObject *>(py_skel_obj);

    // --- 3. OBJECT CREATION ---
    auto obj = PyObject_New(CPH_RagdollSettingsObject,
                            reinterpret_cast<PyTypeObject *>(st->RagdollSettingsType));
    if (!obj) {
        return PyErr_NoMemory();
    }
    obj->parsers = CPH::NewParserGroup<CPH_RagdollSettingsParsers>();
    if (!obj->parsers) {
        Py_DECREF(obj);
        return PyErr_NoMemory();
    }
    obj->parsers->Init();

    // Create native Jolt settings without crossing the Python boundary from core code.
    Py_BEGIN_ALLOW_THREADS obj->settings = CPH_PhysicsWorldCreateRagdollSettings(py_skel->skeleton);
    Py_END_ALLOW_THREADS if (!obj->settings) {
        Py_DECREF(obj);
        return PyErr_NoMemory();
    }

    obj->world = self;
    Py_INCREF(self);

    return reinterpret_cast<PyObject *>(obj);
}
PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateRagdoll(CPH_PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    // --- 1. FAST ARGUMENT PARSING ---
    PyObject *settings_obj = nullptr;
    CPH_PosStride pos{};
    CPH_AuxStride rot    = {.x = 0, .y = 0, .z = 0, .w = 1.0f};
    uint64_t user_data   = 0;
    uint32_t category    = kJoltAllLayerBits;
    uint32_t mask        = kJoltAllLayerBits;
    uint32_t material_id = 0;

    if (!self->parsers->CreateRagdoll.Parse(args, nargs, kwnames, settings_obj, pos, rot, user_data,
                                            category, mask, material_id)) {
        return nullptr;
    }

    // Type Safety Check (Replaces original O! logic)
    CPH_ModuleState *st = CPH_GetModuleState(PyType_GetModule(Py_TYPE(self)));
    if (!PyObject_TypeCheck(settings_obj,
                            reinterpret_cast<PyTypeObject *>(st->RagdollSettingsType))) {
        PyErr_SetString(PyExc_TypeError, "settings must be a RagdollSettings object");
        return nullptr;
    }
    auto py_settings = reinterpret_cast<CPH_RagdollSettingsObject *>(settings_obj);

    // --- 2. Native Jolt creation and pose setup ---
    JPH_Ragdoll *j_rag = nullptr;
    size_t body_count  = 0;
    PhysicsWorldCreationStatus creation_status;
    JPH_RVec3 initial_position{pos.x, pos.y, pos.z};
    JPH_Quat initial_rotation{rot.x, rot.y, rot.z, rot.w};
    Py_BEGIN_ALLOW_THREADS creation_status = CPH_PhysicsWorldCreateRagdollInstance(
        &self->physics, py_settings->settings, initial_position, initial_rotation, user_data,
        &j_rag, &body_count);
    Py_END_ALLOW_THREADS if (creation_status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        set_creation_error(self, creation_status, PHYSICS_WORLD_STATUS_OK,
                           "Jolt failed to create Ragdoll instance", nullptr);
        return nullptr;
    }

    uint32_t *body_slots = nullptr;
    if (body_count > SIZE_MAX / sizeof(uint32_t) ||
        !(body_slots = static_cast<uint32_t *>(CPH_PyAlloc(body_count * sizeof(uint32_t))))) {
        Py_BEGIN_ALLOW_THREADS CPH_PhysicsWorldDestroyRagdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS return PyErr_NoMemory();
    }

    auto obj = PyObject_New(CPH_RagdollObject, reinterpret_cast<PyTypeObject *>(st->RagdollType));
    if (!obj) {
        CPH_PyFree(body_slots);
        Py_BEGIN_ALLOW_THREADS CPH_PhysicsWorldDestroyRagdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS return nullptr;
    }
    obj->ragdoll    = nullptr;
    obj->world      = nullptr;
    obj->body_count = 0;
    obj->body_slots = body_slots;
    obj->parsers    = nullptr;
    obj->parsers    = CPH::NewParserGroup<CPH_RagdollParsers>();
    if (!obj->parsers) {
        Py_DECREF(obj);
        Py_BEGIN_ALLOW_THREADS CPH_PhysicsWorldDestroyRagdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS return PyErr_NoMemory();
    }
    obj->parsers->Init();

    obj->ragdoll = j_rag;
    obj->world   = self;
    Py_INCREF(self);
    obj->body_count = body_count;

    size_t world_count                     = 0;
    PhysicsWorldStatus native_status       = PHYSICS_WORLD_STATUS_OK;
    Py_BEGIN_ALLOW_THREADS creation_status = CPH_PhysicsWorldRegisterRagdollBodies(
        &self->physics, j_rag, body_count, user_data, category, mask, material_id, body_slots,
        &world_count, &native_status);
    Py_END_ALLOW_THREADS if (creation_status != PHYSICS_WORLD_CREATION_STATUS_OK) {
        obj->ragdoll = nullptr;
        Py_DECREF(obj);
        Py_BEGIN_ALLOW_THREADS CPH_PhysicsWorldDestroyRagdoll(&self->physics, j_rag);
        Py_END_ALLOW_THREADS set_creation_error(self, creation_status, native_status, nullptr,
                                                nullptr);
        return nullptr;
    }

    self->view_shape[0] = static_cast<Py_ssize_t>(world_count);
    return reinterpret_cast<PyObject *>(obj);
}

// Fixed get_contact_events to be safer with locking
PyCFunction_DeclareMethodFromModule CPH_PyWorld_GetContactEvents(CPH_PhysicsWorldObject *self,
                                                                 PyObject *args [[maybe_unused]]) {
    CPH_ContactEvent *scratch = nullptr;
    size_t count              = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldTakeContactEvents(&self->physics, &scratch, &count);
    Py_END_ALLOW_THREADS if (status != PHYSICS_WORLD_RUNTIME_STATUS_OK) {
        return runtime_status_error(status, nullptr);
    }
    if (count == 0) {
        return PyList_New(0);
    }

    // --- 2. BUILD PHASE (Unlocked & FastBuild Integrated) ---
    PyObject *list = PyList_New(static_cast<Py_ssize_t>(count));
    if (!list) {
        CPH_AlignedFree(scratch);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        CPH_ContactEventSlim *slim = GetSlimHeader(&scratch[i]);

        // TSan Fix: Explicitly load the handles from the atomic members in the struct.
        // We use relaxed because this 'scratch' copy is thread-local and synchronized.
        uint64_t b1_raw = atomic_load_explicit(&slim->body1, memory_order_relaxed);
        uint64_t b2_raw = atomic_load_explicit(&slim->body2, memory_order_relaxed);

        /**
         * OPTIMIZATION: CULV_PY_BUILD_TUPLE
         * 1. CPH_PyBuildFromU64 converts b1_raw and b2_raw
         * 2. CPH_PyBuildFromFloat converts impulse and sliding_speed
         * 3. CPH_PyBuildPackTuple performs a single O(1) allocation
         */
        PyObject *item = CPH_PY_BUILD_TUPLE(b1_raw, b2_raw, slim->impulse, slim->sliding_speed);

        if (CPH_UNLIKELY(!item)) {
            Py_DECREF(list);
            CPH_AlignedFree(scratch);
            return nullptr;
        }

        // PyList_SET_ITEM steals the reference from CULV_PY_BUILD_TUPLE
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
    }

    CPH_AlignedFree(scratch);
    return list;
}

PyCFunction_DeclareMethodFromModule
CPH_PyWorld_GetContactEventsEx(CPH_PhysicsWorldObject *self, PyObject *args [[maybe_unused]]) {
    CPH_ContactEvent *scratch = nullptr;
    size_t count              = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldTakeContactEvents(&self->physics, &scratch, &count);
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

    if (CPH_UNLIKELY(!k_bodies)) {
        k_bodies = PyUnicode_InternFromString("bodies");
        k_pos    = PyUnicode_InternFromString("position");
        k_norm   = PyUnicode_InternFromString("normal");
        k_str    = PyUnicode_InternFromString("impulse");
        k_slide  = PyUnicode_InternFromString("slide_speed"); // Updated key to reflect lack of _sq
        k_mat    = PyUnicode_InternFromString("materials");
        k_type   = PyUnicode_InternFromString("type");
    }

    // --- 3. BUILD PHASE (FastBuild Engine) ---
    PyObject *list = PyList_New(static_cast<Py_ssize_t>(count));
    if (!list) {
        CPH_AlignedFree(scratch);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        CPH_ContactEventSlim *slim  = GetSlimHeader(&scratch[i]);
        CPH_ContactEventFatExt *fat = GetFatExtension(&scratch[i]);

        // TSan Fix: Explicit relaxed loads for atomic handles
        uint64_t b1 = atomic_load_explicit(&slim->body1, memory_order_relaxed);
        uint64_t b2 = atomic_load_explicit(&slim->body2, memory_order_relaxed);

        /**
         * OPTIMIZATION: CULV_PY_BUILD_DICT
         * We compose the nested tuples (pos, normal, bodies, mats)
         * and the dictionary in a single, readable expression.
         */
        PyObject *dict =
            CPH_PY_BUILD_DICT(k_bodies, CPH_PY_BUILD_TUPLE(b1, b2), k_pos,
                              CPH_PY_BUILD_TUPLE(slim->px, slim->py, slim->pz), k_norm,
                              CPH_PY_BUILD_TUPLE(slim->nx, slim->ny, slim->nz), k_mat,
                              CPH_PY_BUILD_TUPLE(fat->mat1, fat->mat2), k_str, slim->impulse,
                              k_slide, slim->sliding_speed, k_type, slim->flags);

        if (CPH_UNLIKELY(!dict)) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), Py_None);
            continue;
        }

        // Steals ref to dict created by FastBuild
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), dict);
    }

    CPH_AlignedFree(scratch);
    return list;
}

// ContactEvent layout:
// 128 bytes total (64 Slim Header, 64 Fat Extension)
// See ContactEventDefinitions.h for exact alignment and padding.
PyCFunction_DeclareMethodFromModule
CPH_PyWorld_GetContactEventsRaw(CPH_PhysicsWorldObject *self, PyObject *args [[maybe_unused]]) {
    CPH_ContactEvent *events = nullptr;
    size_t count             = 0;
    PhysicsWorldRuntimeStatus status;
    Py_BEGIN_ALLOW_THREADS status =
        CPH_PhysicsWorldTakeContactEvents(&self->physics, &events, &count);
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
    if (count > static_cast<size_t>(PY_SSIZE_T_MAX) / sizeof(CPH_ContactEvent)) {
        CPH_AlignedFree(events);
        PyErr_SetString(PyExc_OverflowError, "Contact event data is too large");
        return nullptr;
    }

    const size_t bytes_size = count * sizeof(CPH_ContactEvent);
    PyObject *raw_bytes     = PyBytes_FromStringAndSize(reinterpret_cast<const char *>(events),
                                                        static_cast<Py_ssize_t>(bytes_size));
    CPH_AlignedFree(events);
    if (!raw_bytes) {
        return nullptr;
    }
    PyObject *view = PyMemoryView_FromObject(raw_bytes);
    Py_DECREF(raw_bytes);
    return view;
}

PyType_DeclareSlot_StatusFromModule
CPH_PyWorld_Getbuffer(CPH_PhysicsWorldObject *self, Py_buffer *view, [[maybe_unused]] int flags) {
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    // TSan Fix: Read the atomic count safely
    size_t current_count = atomic_load_explicit(&self->physics.count, memory_order_acquire);

    // We export the positions buffer as the default buffer for the object
    view->buf        = self->physics.positions;
    view->len        = static_cast<Py_ssize_t>(current_count * sizeof(CPH_PosStride));
    view->readonly   = 0;
    view->itemsize   = sizeof(JPH_Real);
    view->format     = const_cast<char *>((sizeof(JPH_Real) == sizeof(double)) ? "d" : "f");
    view->ndim       = 2;
    view->shape      = self->view_shape;
    view->strides    = self->view_strides;
    view->suboffsets = nullptr;
    view->internal   = nullptr;

    // view_export_count is a standard int protected by shadow_lock
    atomic_fetch_add_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
    return 0;
}

// Buffer Release Slot
PyType_DeclareSlot_VoidFromModule CPH_PyWorld_Releasebuffer(CPH_PhysicsWorldObject *self,
                                                            Py_buffer *view [[maybe_unused]]) {
    CPH_SHADOW_LOCK(&self->physics.shadow_lock);

    // Release logic remains simple as no atomic counters are mutated here
    if (atomic_load_explicit(&self->physics.view_export_count, memory_order_relaxed) > 0) {
        atomic_fetch_sub_explicit(&self->physics.view_export_count, 1, memory_order_relaxed);
    }

    CPH_SHADOW_UNLOCK(&self->physics.shadow_lock);
}

// Declarations for context methods defined in other translation units.
PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateVehicle(CPH_PhysicsWorldObject *self,
                                                              PyObject *const *args, size_t nargsf,
                                                              PyObject *kwnames);

PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateTrackedVehicle(CPH_PhysicsWorldObject *self,
                                                                     PyObject *const *args,
                                                                     size_t nargsf,
                                                                     PyObject *kwnames);
PyCFunction_DeclareMethodFromModule CPH_PyWorld_CreateShip(CPH_PhysicsWorldObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames);
static PyMethodDef kPhysicsWorldMethods[] = {

    // --- Lifecycle ---
    CPH::Python::FastCallMethod("step", CPH_PyWorld_Step),
    CPH::Python::FastCallMethod("create_body", CPH_PyWorld_CreateBody),
    CPH::Python::FastCallMethod("create_bodies_batch", CPH_PyWorld_CreateBodiesBatch),
    CPH::Python::FastCallMethod("destroy_body", CPH_PyWorld_DestroyBody),
    CPH::Python::FastCallMethod("destroy_bodies_batch", CPH_PyWorld_DestroyBodiesBatch),
    CPH::Python::FastCallMethod("create_mesh_body", CPH_PyWorld_CreateMeshBody),
    CPH::Python::FastCallMethod("create_constraint", CPH_PyWorld_CreateConstraint),
    CPH::Python::FastCallMethod("destroy_constraint", CPH_PyWorld_DestroyConstraint),
    CPH::Python::FastCallMethod("get_constraint_type", CPH_PyWorld_GetConstraintType),
    CPH::Python::FastCallMethod("create_vehicle", CPH_PyWorld_CreateVehicle),
    CPH::Python::FastCallMethod("create_tracked_vehicle", CPH_PyWorld_CreateTrackedVehicle),
    CPH::Python::FastCallMethod("create_ship", CPH_PyWorld_CreateShip),
    CPH::Python::FastCallMethod("create_ragdoll_settings", CPH_PyWorld_CreateRagdollSettings),
    CPH::Python::FastCallMethod("create_ragdoll", CPH_PyWorld_CreateRagdoll),
    CPH::Python::FastCallMethod("create_heightfield", CPH_PyWorld_CreateHeightfield),
    CPH::Python::FastCallMethod("create_convex_hull", CPH_PyWorld_CreateConvexHull),
    CPH::Python::FastCallMethod("create_compound_body", CPH_PyWorld_CreateCompoundBody),
    CPH::Python::FastCallMethod("create_soft_body", CPH_PyWorld_CreateSoftBody),

    // --- Interaction ---
    CPH::Python::FastCallMethod("apply_impulse", CPH_PyWorld_ApplyImpulse),
    CPH::Python::FastCallMethod("apply_angular_impulse", CPH_PyWorld_ApplyAngularImpulse),
    CPH::Python::FastCallMethod("apply_impulse_at", CPH_PyWorld_ApplyImpulseAt),
    CPH::Python::FastCallMethod("apply_force", CPH_PyWorld_ApplyForce),
    CPH::Python::FastCallMethod("apply_torque", CPH_PyWorld_ApplyTorque),
    CPH::Python::FastCallMethod("set_gravity", CPH_PyWorld_SetGravity),
    CPH::Python::NoArgsMethod("get_gravity", CPH_PyWorld_GetGravity),
    CPH::Python::FastCallMethod("apply_buoyancy", CPH_PyWorld_ApplyBuoyancy),
    CPH::Python::FastCallMethod("apply_buoyancy_batch", CPH_PyWorld_ApplyBuoyancyBatch),
    CPH::Python::FastCallMethod("set_position", CPH_PyWorld_SetPosition),
    CPH::Python::FastCallMethod("set_rotation", CPH_PyWorld_SetRotation),
    CPH::Python::FastCallMethod("set_linear_velocity", CPH_PyWorld_SetLinearVelocity),
    CPH::Python::FastCallMethod("set_angular_velocity", CPH_PyWorld_SetAngularVelocity),
    CPH::Python::FastCallMethod("set_transform", CPH_PyWorld_SetTransform),
    CPH::Python::FastCallMethod("set_collision_filter", CPH_PyWorld_SetCollisionFilter),
    CPH::Python::FastCallMethod("register_material", CPH_PyWorld_RegisterMaterial),
    CPH::Python::FastCallMethod("set_constraint_target", CPH_PyWorld_SetConstraintTarget),

    // --- Motion Control ---
    CPH::Python::FastCallMethod("get_motion_type", CPH_PyWorld_GetMotionType),
    CPH::Python::FastCallMethod("set_motion_type", CPH_PyWorld_SetMotionType),
    CPH::Python::FastCallMethod("activate", CPH_PyWorld_Activate),
    CPH::Python::FastCallMethod("deactivate", CPH_PyWorld_Deactivate),
    CPH::Python::FastCallMethod("set_ccd", CPH_PyWorld_SetCcd),

    // --- Queries ---
    CPH::Python::FastCallMethod("get_soft_body_vertices", CPH_PyWorld_GetSoftBodyVertices),
    CPH::Python::FastCallMethod("get_soft_body_vertex_count", CPH_PyWorld_GetSoftBodyVertexCount),
    CPH::Python::FastCallMethod("get_soft_body_vertex_position",
                                CPH_PyWorld_GetSoftBodyVertexPosition),
    CPH::Python::FastCallMethod("get_soft_body_local_vertices",
                                CPH_PyWorld_GetSoftBodyLocalVertices),
    CPH::Python::FastCallMethod("raycast", CPH_PyWorld_Raycast),
    CPH::Python::FastCallMethod("raycast_batch", CPH_PyWorld_RaycastBatch),
    CPH::Python::FastCallMethod("shapecast", CPH_PyWorld_Shapecast),
    CPH::Python::FastCallMethod("overlap_sphere", CPH_PyWorld_OverlapSphere),
    CPH::Python::FastCallMethod("overlap_aabb", CPH_PyWorld_OverlapAabb),

    // --- Utilities ---
    CPH::Python::FastCallMethod("get_index", CPH_PyWorld_GetIndex),
    CPH::Python::FastCallMethod("is_alive", CPH_PyWorld_IsAlive),
    CPH::Python::FastCallMethod("is_active", CPH_PyWorld_IsActive),
    CPH::Python::NoArgsMethod("get_active_indices", CPH_PyWorld_GetActiveIndices),
    CPH::Python::FastCallMethod("get_render_state", CPH_PyWorld_GetRenderState),
    CPH::Python::FastCallMethod("get_debug_data", CPH_PyWorld_GetDebugData),
    CPH::Python::FastCallMethod("get_body_stats", CPH_PyWorld_GetBodyStats),

    // --- User Data ---
    CPH::Python::FastCallMethod("get_user_data", CPH_PyWorld_GetUserData),
    CPH::Python::FastCallMethod("set_user_data", CPH_PyWorld_SetUserData),

    // -- Event Logic ---
    CPH::Python::NoArgsMethod("get_contact_events", CPH_PyWorld_GetContactEvents),
    CPH::Python::NoArgsMethod("get_contact_events_ex", CPH_PyWorld_GetContactEventsEx),
    CPH::Python::NoArgsMethod("get_contact_events_raw", CPH_PyWorld_GetContactEventsRaw),

    // --- State & Advanced ---
    CPH::Python::NoArgsMethod("save_state", CPH_PyWorld_SaveState),
    CPH::Python::FastCallMethod("load_state", CPH_PyWorld_LoadState),
    CPH::Python::FastCallMethod("create_character", CPH_PyWorld_CreateCharacter),

    // --- Internal/Debug ---
    // Private parser/build microbenchmarks.
    CPH::Python::FastCallMethod("_benchmark_parse", CPH_PyWorld_BenchmarkParse),

    CPH::Python::NoArgsMethod("_benchmark_build", CPH_PyWorld_BenchmarkBuild),

    {nullptr, nullptr, 0, nullptr}};

PyType_Spec CPH_PyWorld_Spec = {
    .name      = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(CPH_PhysicsWorldObject),
    .flags =
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_HAVE_GC | Py_TPFLAGS_MANAGED_DICT,
    .slots =
        (PyType_Slot[]){

            {.slot = Py_tp_new, .pfunc = reinterpret_cast<void *>(PyType_GenericNew)},
            {.slot = Py_tp_init, .pfunc = reinterpret_cast<void *>(CPH_PyWorld_Init)},
            {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(CPH_PyWorld_Dealloc)},
            {.slot = Py_tp_call, .pfunc = nullptr},
            {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kPhysicsWorldMethods)},
            {.slot = Py_tp_members,
             .pfunc =
                 (PyMemberDef[]){

                     {.name   = "__weaklistoffset__",
                      .type   = Py_T_PYSSIZET,
                      .offset = offsetof(CPH_PhysicsWorldObject, weakreflist),
                      .flags  = Py_READONLY,
                      .doc    = nullptr},
                     {}

                 }},
            {.slot = Py_tp_getset,
             .pfunc =
                 (PyGetSetDef[]){

                     CPH::Python::GetSet("positions", CPH_PyWorld_GetPositions),
                     CPH::Python::GetSet("rotations", CPH_PyWorld_GetRotations),
                     CPH::Python::GetSet("velocities", CPH_PyWorld_GetVelocities),
                     CPH::Python::GetSet("angular_velocities", CPH_PyWorld_GetAngularVelocities),
                     CPH::Python::GetSet("count", CPH_PyWorld_GetCount),
                     CPH::Python::GetSet("time", CPH_PyWorld_GetTime),
                     CPH::Python::GetSet("user_data", CPH_PyWorld_GetUserDataBuffer),
                     CPH::Python::GetSet("shape_count", CPH_PyWorld_GetShapeCount),
                     CPH::Python::GetSet("is_step_pending", CPH_PyWorld_GetIsStepPending),
                     CPH::Python::GetSet("max_bodies", CPH_PyWorld_GetMaxBodies),
                     CPH::Python::GetSet("remaining_capacity", CPH_PyWorld_GetRemainingCapacity),
                     {}

                 }},
            {.slot = Py_bf_getbuffer, .pfunc = reinterpret_cast<void *>(CPH_PyWorld_Getbuffer)},
            {.slot  = Py_bf_releasebuffer,
             .pfunc = reinterpret_cast<void *>(CPH_PyWorld_Releasebuffer)},
            {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(CPH_PyWorld_Traverse)},
            {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(CPH_PyWorld_Clear)},
            {},

        },
};
