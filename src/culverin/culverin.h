#pragma once

#define PY_SSIZE_T_CLEAN
#include "joltc.h" // Amer Koleci's JoltC binder.
#include <Python.h>

#include "culverin_types.h"
#include "culverin_command_buffer.h"
#include "culverin_debug_render.h"
#include "culverin_internal_query.h"
#include "culverin_tracked_vehicle.h"
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

// Jolt BodyID layout: [8 bits sequence | 24 bits index]
#ifndef JPH_BODY_ID_INDEX_MASK
#define JPH_BODY_ID_INDEX_MASK 0x00FFFFFF
#endif

#ifndef JPH_DOUBLE_PRECISION
#define JPH_DOUBLE_PRECISION 1
#endif

// Use restrict keyword to tell the compiler these buffers do not overlap.
// This is the single best way to enable SIMD auto-vectorization.
#ifdef _MSC_VER
#define CULV_RESTRICT __restrict
#else
#define CULV_RESTRICT __restrict__
#endif

constexpr int CONTACT_MAX_CAPACITY = 16384;

// Mask for the raw array index (Stripping the 24th bit used for Static flags)
#define JPH_ID_TO_INDEX(id) ((id) & 0x7FFFFF)

// Allocate 'Type' on the stack with guaranteed 32-byte alignment.
// USAGE: JPH_STACK_ALLOC(JPH_RVec3, my_vec);
#if defined(__clang__) || defined(__GNUC__)
// Clang/LLVM alignment logic (highly robust)
#define JPH_ALIGNED_STORAGE(Type, Name, Align)                                 \
  Type Name __attribute__((aligned(Align)))
#elif defined(_MSC_VER)
#define JPH_ALIGNED_STORAGE(Type, Name, Align)                                 \
  __declspec(align(Align)) Type Name
#else
#include <stdalign.h>
#define JPH_ALIGNED_STORAGE(Type, Name, Align) alignas(Align) Type Name
#endif

#define JPH_STACK_ALLOC(Type, Name)                                            \
  JPH_ALIGNED_STORAGE(Type, Name##_storage, 32);                               \
  Type *Name = &Name##_storage

// --- Lock Helpers in culverin.h ---
#if PY_VERSION_HEX >= 0x030D0000
// Python 3.13+ uses PyMutex (no allocation needed, just zero init)
#define INIT_LOCK(m) memset(&(m), 0, sizeof(ShadowMutex))
#define FREE_LOCK(m)
#else
// Older Python versions use PyThread_type_lock (requires allocation)
#define INIT_LOCK(m) (m) = PyThread_allocate_lock()
#define FREE_LOCK(m)                                                           \
  do {                                                                         \
    if (m)                                                                     \
      PyThread_free_lock(m);                                                   \
    (m) = NULL;                                                                \
  } while (0)
#endif

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

// --- Compiler Hints ---
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
// Fallback for MSVC or other compilers that don't support built-in expect
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif

// Comment this line out to disable all debug prints
#define CULVERIN_DEBUG

#ifdef CULVERIN_DEBUG
#define DEBUG_LOG(fmt, ...)                                                    \
  fprintf(stderr, "[Culverin] " fmt "\n", ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// Processor-level hint to save power during spin-waits
static inline void culverin_cpu_relax() {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
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

// Minimal Handle Helper
// Python handles will be 64-bit integers: (Generation << 32) | SlotIndex
typedef uint64_t BodyHandle;

// Constraint Types
typedef enum ConstraintType : uint8_t {
  CONSTRAINT_FIXED = 0,
  CONSTRAINT_POINT = 1,
  CONSTRAINT_HINGE = 2,
  CONSTRAINT_SLIDER = 3,
  CONSTRAINT_DISTANCE = 4,
  CONSTRAINT_CONE = 5
} ConstraintType;

// Minimal Handle for Constraints (Distinct from BodyHandle)
typedef uint64_t ConstraintHandle;

// --- Contact Lifecycle Types ---
typedef enum ContactEventType : uint8_t {
  EVENT_ADDED = 0,
  EVENT_PERSISTED = 1,
  EVENT_REMOVED = 2
} ContactEventType;

// --- Callback Logic ---
typedef struct ContactEvent {
  uint64_t body1;
  uint64_t body2;
  float px, py, pz;
  float nx, ny, nz;
  float impulse;
  float sliding_speed_sq; // Scratching speed squared(tangential)
  uint32_t mat1;
  uint32_t mat2;
  uint32_t type;
  uint32_t _pad;
} ContactEvent;

_Static_assert(sizeof(ContactEvent) == 64,
               "ContactEvent must be 64 bytes for performance");

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

typedef struct {
  int max_bodies;
  int max_pairs;
} WorldLimits;

typedef struct {
  float gx;
  float gy;
  float gz;
} GravityVector;

typedef struct {
  float px;
  float py;
  float pz;
} PositionVector; // Can use GravityVector if it's identical

typedef struct {
  float mass;
  float friction;
  float restitution;
  int is_sensor;
  int use_ccd;
} BodyCreationProps;

typedef struct {
  float friction;
  float restitution;
} MaterialSettings;

typedef struct {
  float mass;
  float friction;
  float restitution;
  int is_sensor;
  int use_ccd;
  int motion_type;
} BodyConfig;

// Struct to hold parsed Python data safely in C
typedef struct {
    JPH_Vec3 local_p;
    JPH_Quat local_q;
    float params[4];
    int type;
} CompoundPart;

// --- Native Condition Variable Support ---

#ifdef _WIN32
  typedef SRWLOCK NativeMutex;
  typedef CONDITION_VARIABLE NativeCond;
  #define INIT_NATIVE_MUTEX(m) InitializeSRWLock(&(m))
  #define FREE_NATIVE_MUTEX(m) (void)(m) // No cleanup needed for SRWLock
  #define NATIVE_MUTEX_LOCK(m) AcquireSRWLockExclusive(&(m))
  #define NATIVE_MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(&(m))
  
  #define INIT_NATIVE_COND(c) InitializeConditionVariable(&(c))
  #define FREE_NATIVE_COND(c) (void)(c) // No cleanup needed
  #define NATIVE_COND_WAIT(c, m) SleepConditionVariableSRW(&(c), &(m), INFINITE, 0)
  #define NATIVE_COND_BROADCAST(c) WakeAllConditionVariable(&(c))
#else
  #include <pthread.h>
  typedef pthread_mutex_t NativeMutex;
  typedef pthread_cond_t NativeCond;
  #define INIT_NATIVE_MUTEX(m) pthread_mutex_init(&(m), NULL)
  #define FREE_NATIVE_MUTEX(m) pthread_mutex_destroy(&(m))
  #define NATIVE_MUTEX_LOCK(m) pthread_mutex_lock(&(m))
  #define NATIVE_MUTEX_UNLOCK(m) pthread_mutex_unlock(&(m))

  #define INIT_NATIVE_COND(c) pthread_cond_init(&(c), NULL)
  #define FREE_NATIVE_COND(c) pthread_cond_destroy(&(c))
  #define NATIVE_COND_WAIT(c, m) pthread_cond_wait(&(c), &(m))
  #define NATIVE_COND_BROADCAST(c) pthread_cond_broadcast(&(c))
#endif

// Blocks until the world is not mid-step.
// Must be called while holding SHADOW_LOCK. Re-acquires it before returning.
#define BLOCK_UNTIL_NOT_STEPPING(self)                                         \
  do {                                                                         \
    if (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed)) {    \
      SHADOW_UNLOCK(&(self)->shadow_lock);                                     \
      Py_BEGIN_ALLOW_THREADS                                                   \
      NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                              \
      /* The Double Check: check again after acquiring native lock */          \
      while (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed)) { \
        NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);     \
      }                                                                        \
      NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                            \
      Py_END_ALLOW_THREADS                                                     \
      SHADOW_LOCK(&(self)->shadow_lock);                                       \
    }                                                                          \
  } while (0)

#define BLOCK_UNTIL_NOT_QUERYING(self)                                         \
  do {                                                                         \
    if (atomic_load_explicit(&(self)->active_queries, memory_order_acquire) > 0) { \
      SHADOW_UNLOCK(&(self)->shadow_lock);                                     \
      Py_BEGIN_ALLOW_THREADS                                                   \
      NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                              \
      /* The Double Check */                                                   \
      while (atomic_load_explicit(&(self)->active_queries, memory_order_relaxed) > 0) { \
        NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);     \
      }                                                                        \
      NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                            \
      Py_END_ALLOW_THREADS                                                     \
      SHADOW_LOCK(&(self)->shadow_lock);                                       \
    }                                                                          \
  } while (0)

// Queries use this to wait if a Step is about to happen
#define BLOCK_IF_STEP_PENDING(self)                                            \
  do {                                                                         \
    if (atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) { \
      SHADOW_UNLOCK(&(self)->shadow_lock);                                     \
      Py_BEGIN_ALLOW_THREADS                                                   \
      NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                              \
      while (atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) { \
        NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);     \
      }                                                                        \
      NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                            \
      Py_END_ALLOW_THREADS                                                     \
      SHADOW_LOCK(&(self)->shadow_lock);                                       \
    }                                                                          \
  } while (0)

// A container to sync state changes (stepping finished, query finished)
typedef struct {
    NativeMutex mutex;
    NativeCond cond;
} ShadowSync;

// --- The Object Struct ---
typedef struct PhysicsWorldObject {
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
  JPH_Real *positions;
  float *rotations;
  // Previous State Buffers (For Interpolation)
  JPH_Real *prev_positions;
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
  PhysicsCommand *command_queue_spare;
  size_t command_count;
  size_t command_capacity;
  size_t spare_capacity;

  ShapeEntry *shape_cache;
  size_t shape_cache_count;
  size_t shape_cache_capacity;

  size_t count;
  size_t capacity;
  double time;

  // Fast index-to-handle lookup to avoid Jolt locks in callbacks
  BodyHandle *id_to_handle_map;
  uint32_t max_jolt_bodies;

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
  ShadowSync step_sync; 
  atomic_bool step_requested; // New: Stepper wants to run
  _Atomic bool is_stepping;

  Py_ssize_t view_shape[2];
  Py_ssize_t view_strides[2];

  // --- Debug Renderer ---
  JPH_DebugRenderer *debug_renderer;
  DebugBuffer debug_lines;
  DebugBuffer debug_triangles;
} PhysicsWorldObject;

// Temporary container for resize
typedef struct {
  JPH_Real *pos, *ppos;
  float *rot, *prot, *lvel, *avel;
  JPH_BodyID *bids;
  uint64_t *udat;
  uint32_t *gens, *s2d, *d2s, *free, *cats, *masks, *mats;
  uint8_t *stat;
} NewBuffers;

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

// --- Handle Helper ---
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
  return ((uint64_t)gen << 32) | (uint64_t)slot;
}

static inline bool unpack_handle(PhysicsWorldObject *self, BodyHandle h,
                                 uint32_t *slot) {
  *slot = (uint32_t)(h & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(h >> 32);

  if (*slot >= self->slot_capacity) {
    return false;
  }
  return self->generations[*slot] == gen;
}

extern NativeMutex g_jph_trampoline_lock;
