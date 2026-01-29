#ifndef PYJOLT_H
#define PYJOLT_H

#define PY_SSIZE_T_CLEAN
#include "joltc.h"
#include <Python.h>
#include <float.h>
#include <math.h>
#include <stdatomic.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__) || defined(__apple__)
#include <sched.h>
#include <unistd.h>
#endif

#ifndef JPH_INVALID_BODY_ID
#define JPH_INVALID_BODY_ID 0xFFFFFFFF
#endif

// Allocate 'Type' on the stack with guaranteed 32-byte alignment.
// USAGE: JPH_STACK_ALLOC(JPH_RVec3, my_vec);
#if defined(_MSC_VER)
// Microsoft Visual Studio syntax
#define JPH_ALIGNED_STORAGE(Type, Name, Align)                                 \
  __declspec(align(Align)) Type Name
#elif defined(__GNUC__) || defined(__clang__)
// GCC/Clang syntax (supports both C and C++)
#define JPH_ALIGNED_STORAGE(Type, Name, Align)                                 \
  Type Name __attribute__((aligned(Align)))
#else
// Standard C11 (requires <stdalign.h> if not in C++)
#include <stdalign.h>
#define JPH_ALIGNED_STORAGE(Type, Name, Align) alignas(Align) Type Name
#endif

#define JPH_STACK_ALLOC(Type, Name)                                            \
  JPH_ALIGNED_STORAGE(Type, Name##_storage, 32);                               \
  Type *Name = &Name##_storage

// --- Threading Primitives (Python 3.14t support) ---
#if PY_VERSION_HEX >= 0x030D0000
typedef PyMutex ShadowMutex;
#define SHADOW_LOCK(m) PyMutex_Lock(m)
#define SHADOW_UNLOCK(m) PyMutex_Unlock(m)
#else
typedef PyThread_type_lock ShadowMutex;
#define SHADOW_LOCK(m) PyThread_acquire_lock(m, 1)
#define SHADOW_UNLOCK(m) PyThread_release_lock(m)
#endif

static ShadowMutex g_jph_trampoline_lock; // Global lock for JPH callbacks

// Comment this line out to disable all debug prints
#define CULVERIN_DEBUG

#ifdef CULVERIN_DEBUG
#define DEBUG_LOG(fmt, ...)                                                    \
  fprintf(stderr, "[Culverin] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

#define GUARD_STEPPING(self)                                                   \
  do                                                                           \
    if ((self)->is_stepping) {                                                 \
      SHADOW_UNLOCK(&(self)->shadow_lock);                                     \
      PyErr_SetString(PyExc_RuntimeError,                                      \
                      "Cannot modify physics world during simulation step");   \
      return NULL;                                                             \
    }                                                                          \
  while (0)
/* Must use while in lock(deprecated, only use for debug) */

// Processor-level hint to save power during spin-waits
static inline void culverin_cpu_relax() {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
// MSVC and Intel use intrinsics
#include <immintrin.h>
  _mm_pause();
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__i386__) || defined(__x86_64__)
  __asm__ __volatile__("pause");
#elif defined(__arm__) || defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
#endif
}

static inline void culverin_yield() {
  // 1. Give the CPU a break (Hardware level)
  culverin_cpu_relax();

// 2. Give the OS a break (Kernel level)
#if defined(_WIN32)
  // SwitchToThread() is the gold standard for Windows yielding
  if (SwitchToThread() == FALSE) {
    Sleep(0);
  }
#elif defined(__linux__) || defined(__FreeBSD__)
  sched_yield();
#elif defined(__APPLE__)
  // macOS deprecated sched_yield behavior; usleep(0) is often preferred
  // for thread arbitration in user-space.
  usleep(0);
#else
  // Fallback for unknown POSIX systems
  sleep(0);
#endif
}

// Blocks until the world is not mid-step.
// Must be called while holding SHADOW_LOCK. Re-acquires it before returning.
#define BLOCK_UNTIL_NOT_STEPPING(self)                                         \
  do {                                                                         \
    while ((self)->is_stepping) {                                              \
      SHADOW_UNLOCK(&(self)->shadow_lock);                                     \
      Py_BEGIN_ALLOW_THREADS culverin_yield();                                 \
      Py_END_ALLOW_THREADS SHADOW_LOCK(&(self)->shadow_lock);                  \
    }                                                                          \
  } while (0)

// Blocks until no queries (raycasts/shapecasts) are running.
// Must be called while holding SHADOW_LOCK.
#define BLOCK_UNTIL_NOT_QUERYING(self)                                         \
  do {                                                                         \
    while (atomic_load_explicit(&(self)->active_queries,                       \
                                memory_order_acquire) > 0) {                   \
      SHADOW_UNLOCK(&(self)->shadow_lock);                                     \
      Py_BEGIN_ALLOW_THREADS culverin_yield();                                 \
      Py_END_ALLOW_THREADS SHADOW_LOCK(&(self)->shadow_lock);                  \
    }                                                                          \
  } while (0)

// --- Shape Caching ---
typedef struct {
  uint32_t type; // 0=Box, 1=Sphere, 2=Capsule, 3=Cylinder, 4=Plane
  float p1, p2, p3, p4;
} ShapeKey;

typedef struct {
  ShapeKey key;
  JPH_Shape *shape;
} ShapeEntry;

// Minimal Handle Helper
// Python handles will be 64-bit integers: (Generation << 32) | SlotIndex
typedef uint64_t BodyHandle;

// Constraint Types
typedef enum {
  CONSTRAINT_FIXED = 0,
  CONSTRAINT_POINT = 1,
  CONSTRAINT_HINGE = 2,
  CONSTRAINT_SLIDER = 3,
  CONSTRAINT_DISTANCE = 4,
  CONSTRAINT_CONE = 5
} ConstraintType;

// Minimal Handle for Constraints (Distinct from BodyHandle)
typedef uint64_t ConstraintHandle;

// --- Slot State Machine ---
typedef enum {
  SLOT_EMPTY = 0,
  SLOT_PENDING_CREATE = 1,
  SLOT_ALIVE = 2,
  SLOT_PENDING_DESTROY = 3
} SlotState;

// --- Command Buffer ---
typedef enum {
  CMD_CREATE_BODY,
  CMD_DESTROY_BODY,
  CMD_SET_POS,
  CMD_SET_ROT,
  CMD_SET_TRNS, // Position + Rotation
  CMD_SET_LINVEL,
  CMD_SET_ANGVEL,
  CMD_SET_MOTION,
  CMD_ACTIVATE,
  CMD_DEACTIVATE,
  CMD_SET_USER_DATA
} CommandType;

typedef struct {
  CommandType type;
  uint32_t slot; // Every command targets a slot
  union {
    // CMD_CREATE_BODY
    struct {
      JPH_BodyCreationSettings *settings;
      uint64_t user_data;
      uint32_t category;
      uint32_t mask;
      uint32_t material_id;
    } create;

    // CMD_SET_POS, CMD_SET_ROT, CMD_SET_LINVEL, CMD_SET_ANGVEL
    struct {
      float x, y, z, w; // w used for rotation
    } vec;

    // CMD_SET_TRNS (Combined Pos + Rot)
    struct {
      float px, py, pz;
      float rx, ry, rz, rw;
    } transform;

    // CMD_SET_MOTION
    int motion_type;

    // CMD_SET_USER_DATA
    uint64_t user_data_val;
  } data;
} PhysicsCommand;

#ifdef _MSC_VER
#pragma pack(push, 1)
#endif

// --- Callback Logic ---
typedef struct
#ifndef _MSC_VER
    __attribute__((packed))
#endif
    ContactEvent {
  uint64_t body1;
  uint64_t body2;
  float px, py, pz;
  float nx, ny, nz;
  float impulse;
  float sliding_speed_sq; // Scratching speed squared(tangential)
  uint32_t mat1;          // 4 (New)
  uint32_t mat2;          // 4 (New)
  uint32_t _pad[2];       // 8 (Padding to 64 bytes)
} ContactEvent;

#ifdef _MSC_VER
#pragma pack(pop)
#endif

// --- Raycast Batch Result (Aligned to 16-bytes, Total 48-bytes) ---
#ifdef _MSC_VER
#pragma pack(push, 1)
#endif
typedef struct
#ifndef _MSC_VER
    __attribute__((packed))
#endif
{
  uint64_t handle;      // 8 bytes
  float fraction;       // 4 bytes
  float nx, ny, nz;     // 12 bytes
  float px, py, pz;     // 12 bytes
  uint32_t subShapeID;  // 4 bytes
  uint32_t material_id; // 4 bytes
  uint32_t _pad;
} RayCastBatchResult;
#ifdef _MSC_VER
#pragma pack(pop)
#endif

_Static_assert(sizeof(RayCastBatchResult) == 48,
               "RayCastBatchResult size mismatch");

_Static_assert(sizeof(ContactEvent) == 64, "ContactEvent size mismatch");

// --- Material Registry ---
typedef struct {
  uint32_t id;
  float friction;
  float restitution;
  // Padding/Alignment isn't critical here as this is a lookup array, not a
  // stream
} MaterialData;

// --- The Object Struct ---
typedef struct {
  PyObject_HEAD

      // Jolt Handles
      JPH_PhysicsSystem *system;
  JPH_CharacterVsCharacterCollision *char_vs_char_manager;
  JPH_BodyInterface *body_interface;
  JPH_JobSystem *job_system;

  // Filters
  JPH_BroadPhaseLayerInterface *bp_interface;
  JPH_ObjectLayerPairFilter *pair_filter;
  JPH_ObjectVsBroadPhaseLayerFilter *bp_filter;

  // --- Global Contact Listener ---
  JPH_ContactListener *contact_listener;

  // --- Event Buffer ---
  ContactEvent *contact_events;
  size_t contact_count;
  size_t contact_capacity;

  // Change contact_count to an atomic
  atomic_size_t contact_atomic_idx;

  atomic_int active_queries; // Tracks threads currently inside Jolt queries

  // The buffer must be large and pre-allocated
  ContactEvent *contact_buffer;
  size_t contact_max_capacity;

  // Shadow Buffers
  float *positions;
  float *rotations;
  // Previous State Buffers (For Interpolation)
  float *prev_positions;
  float *prev_rotations;

  float *linear_velocities;
  float *angular_velocities;
  JPH_BodyID *body_ids;
  uint64_t *user_data;

  // --- Indirection System ---
  uint32_t *categories;   // [Dense Index]
  uint32_t *masks;        // [Dense Index]
  uint32_t *material_ids; // Shadow buffer (Per-Body)

  // Registry (Global for this world)
  MaterialData *materials;
  size_t material_count;
  size_t material_capacity;

  uint32_t *generations;   // [Slot] -> Generation
  uint32_t *slot_to_dense; // [Slot] -> Dense Index
  uint32_t *dense_to_slot; // [Dense Index] -> Slot

  uint32_t *free_slots; // Stack of available slots
  uint8_t *slot_states;
  size_t free_count;
  size_t slot_capacity; // Size of the mapping arrays

  PhysicsCommand *command_queue;
  size_t command_count;
  size_t command_capacity;

  ShapeEntry *shape_cache;
  size_t shape_cache_count;
  size_t shape_cache_capacity;

  size_t count;
  size_t capacity;
  double time;

  // --- Constraint Registry ---
  JPH_Constraint **constraints;
  uint32_t *constraint_generations;
  uint32_t *free_constraint_slots;
  uint8_t *constraint_states; // ALIVE / EMPTY

  size_t constraint_count;
  size_t constraint_capacity;
  size_t free_constraint_count;

  // MemoryView Safety
  int view_export_count; // Tracks active memoryviews to prevent unsafe resize

  ShadowMutex shadow_lock;
  bool is_stepping;

  Py_ssize_t view_shape[2];
  Py_ssize_t view_strides[2];
} PhysicsWorldObject;

// --- Character Object ---
typedef struct {
  PyObject_HEAD JPH_CharacterVirtual *character;
  PhysicsWorldObject *world;
  BodyHandle handle;

  // Filters and listeners
  JPH_BodyFilter *body_filter;
  JPH_ShapeFilter *shape_filter;
  JPH_BroadPhaseLayerFilter *bp_filter;
  JPH_ObjectLayerFilter *obj_filter;
  JPH_CharacterContactListener *listener;

  // ATOMIC INPUTS: Read by Jolt worker threads in callbacks
  _Atomic float push_strength;
  _Atomic float last_vx;
  _Atomic float last_vy;
  _Atomic float last_vz;

  // Non-atomic: Used by main thread only for rendering
  float prev_px, prev_py, prev_pz;
  float prev_rx, prev_ry, prev_rz, prev_rw;
} CharacterObject;

extern const PyType_Spec Character_spec;

typedef struct {
  PyObject_HEAD JPH_VehicleConstraint *vehicle;
  JPH_VehicleCollisionTester *tester;
  PhysicsWorldObject *world;

  // Ownership tracking for cleanup
  JPH_WheelSettings **wheel_settings;
  JPH_VehicleControllerSettings *controller_settings;
  JPH_VehicleTransmissionSettings *transmission_settings; // NEW: Keep alive
  JPH_LinearCurve *friction_curve;
  JPH_LinearCurve *torque_curve;

  uint32_t num_wheels;
  int current_gear;
} VehicleObject;

extern const PyType_Spec Vehicle_spec;

// --- Ragdoll Structures ---

typedef struct {
  PyObject_HEAD JPH_Skeleton *skeleton;
} SkeletonObject;

extern const PyType_Spec Skeleton_spec;

typedef struct {
  PyObject_HEAD JPH_RagdollSettings *settings;
  PhysicsWorldObject *world; // Kept to access Shape Cache
} RagdollSettingsObject;

extern const PyType_Spec RagdollSettings_spec;

typedef struct {
  PyObject_HEAD JPH_Ragdoll *ragdoll;
  PhysicsWorldObject *world;

  // We must track the handles of the parts so we can
  // invalid the slots when the ragdoll is destroyed.
  size_t body_count;
  uint32_t *body_slots;
} RagdollObject;

extern const PyType_Spec Ragdoll_spec;

typedef struct {
  PhysicsWorldObject *world;
  PyObject *result_list; // Python List to append handles to
} QueryContext;

// Helper for Overlap Callbacks
typedef struct {
  PhysicsWorldObject *world;
  uint64_t *hits; // C array to store baked handles
  size_t count;
  size_t capacity;
} OverlapContext;

typedef struct {
  JPH_ShapeCastResult hit;
  bool has_hit;
} CastShapeContext;

// Optional filter to ignore a specific body
typedef struct {
  JPH_BodyID ignore_id;
} CastShapeFilter;

// --- Module State (PEP 489) ---
typedef struct {
  PyObject *helper;           // Reference to culverin._culverin module
  PyObject *PhysicsWorldType; // Reference to the class
  PyObject *CharacterType;    // Reference to the character class
  PyObject *VehicleType;      // Reference to the vehicle class
  PyObject *SkeletonType;
  PyObject *RagdollSettingsType;
  PyObject *RagdollType;
} CulverinState;

// Helper to retrieve state from the module object
static inline CulverinState *get_culverin_state(PyObject *module) {
  return (CulverinState *)PyModule_GetState(module);
}

static inline bool JPH_API_CALL CastShape_BodyFilter(void *userData,
                                                     JPH_BodyID bodyID) {
  CastShapeFilter *ctx = (CastShapeFilter *)userData;
  return (ctx->ignore_id == 0 || bodyID != ctx->ignore_id);
}

// Sync function (defined in shadow_sync.c)
void culverin_sync_shadow_buffers(PhysicsWorldObject *self);

#endif