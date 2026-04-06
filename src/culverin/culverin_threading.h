#pragma once
#include "culverin_compiler_specifics.h"
#include <Python.h>
#include <stdatomic.h>

/**
 * Culverin Threading Invariants & Lock Hierarchy
 * ---------------------------------------------
 * * 1. HIERARCHY: ShadowMutex (High) -> NativeMutex (Low).
 * - Always acquire shadow_lock BEFORE step_sync.mutex.
 * - To avoid deadlock, release shadow_lock before blocking on condition
 * variables.
 * * 2. OWNERSHIP:
 * - SHADOW_LOCK protects the Command Queue, Slot States, and Shadow Buffers.
 * - NATIVE_MUTEX/COND handles thread arbitration (parking/waking).
 * - g_jph_trampoline_lock protects the non-thread-safe JPH Physics System
 * state.
 * * 3. STEPPING INVARIANT:
 * - is_stepping = true  => No external thread may read/write Shadow Buffers.
 * - is_stepping = false => External (Python) threads may read Shadow Buffers
 * under SHADOW_LOCK.
 * * 4. DOUBLE-BUFFERING:
 * - Command queues are swapped under SHADOW_LOCK but flushed under
 * g_jph_trampoline_lock.
 * - This allows Python to queue new commands while the previous batch is being
 * simulated.
 */

#ifdef CULVERIN_PROFILE_SYNC
#    define SYNC_START_TIMER(start)                                                                \
        struct timespec start;                                                                     \
        clock_gettime(CLOCK_MONOTONIC, &start)
#    define SYNC_END_TIMER(start, self)                                                            \
        do {                                                                                       \
            struct timespec end;                                                                   \
            clock_gettime(CLOCK_MONOTONIC, &end);                                                  \
            uint64_t ns =                                                                          \
                (end.tv_sec - start.tv_sec) * 1000000000ULL + (end.tv_nsec - start.tv_nsec);       \
            atomic_fetch_add_explicit(&(self)->step_sync.total_blocked_ns, ns,                     \
                                      memory_order_relaxed);                                       \
        } while (0)
#else
#    define SYNC_START_TIMER(start)
#    define SYNC_END_TIMER(start, self)
#endif

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
#elif defined(__linux__) || defined(__apple__)
#    include <sched.h>
#    include <unistd.h>
#endif

#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
#    include <immintrin.h>
#endif

// Processor-level hint to save power during spin-waits
static inline void culverin_cpu_relax() {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER)
    _mm_pause();
#elif defined(__GNUC__) || defined(__clang__)
#    if defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__("pause");
#    elif defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("yield");
#    endif
#endif
}

CULV_MAYBE_UNUSED static inline void culverin_yield() {
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
#    include <unistd.h>
    // macOS deprecated sched_yield behavior; usleep(0) is often preferred
    // for thread arbitration in user-space.
    usleep(0);
#else
    // Fallback for unknown POSIX systems
    sleep(0);
#endif
}

// --- Native Condition Variable Support ---

#ifdef _WIN32
typedef SRWLOCK NativeMutex;
typedef CONDITION_VARIABLE NativeCond;

// --- Native Mutex Inlines ---

static inline int shadow_init_native_mutex(NativeMutex *m) {
    InitializeSRWLock(m);
    return 0; // SRWLock init always succeeds and returns void
}

static inline void shadow_free_native_mutex(NativeMutex *m) {
    (void)m; // SRWLock requires no explicit cleanup
}

static inline void shadow_native_mutex_lock(NativeMutex *m) {
    AcquireSRWLockExclusive(m);
}

static inline void shadow_native_mutex_unlock(NativeMutex *m) {
    ReleaseSRWLockExclusive(m);
}

// --- Native Condition Variable Inlines ---

static inline int shadow_init_native_cond(NativeCond *c) {
    InitializeConditionVariable(c);
    return 0; // ConditionVariable init always succeeds and returns void
}

static inline void shadow_free_native_cond(NativeCond *c) {
    (void)c; // No explicit cleanup needed
}

static inline void shadow_native_cond_wait(NativeCond *c, NativeMutex *m) {
    // Windows returns a BOOL, but since you use INFINITE, it only 
    // returns once the lock is re-acquired.
    SleepConditionVariableSRW(c, m, INFINITE, 0);
}

static inline void shadow_native_cond_broadcast(NativeCond *c) {
    WakeAllConditionVariable(c);
}

// --- Macro Wrappers ---

#    define INIT_NATIVE_MUTEX(m)      shadow_init_native_mutex(&(m))
#    define FREE_NATIVE_MUTEX(m)      shadow_free_native_mutex(&(m))
#    define NATIVE_MUTEX_LOCK(m)      shadow_native_mutex_lock(&(m))
#    define NATIVE_MUTEX_UNLOCK(m)    shadow_native_mutex_unlock(&(m))

#    define INIT_NATIVE_COND(c)       shadow_init_native_cond(&(c))
#    define FREE_NATIVE_COND(c)       shadow_free_native_cond(&(c))
#    define NATIVE_COND_WAIT(c, m)    shadow_native_cond_wait(&(c), &(m))
#    define NATIVE_COND_BROADCAST(c)  shadow_native_cond_broadcast(&(c))

#else
#    include <pthread.h>

typedef pthread_mutex_t NativeMutex;
typedef pthread_cond_t NativeCond;

// --- Native Mutex Inlines ---

static inline int shadow_init_native_mutex(NativeMutex *m) {
    return pthread_mutex_init(m, nullptr);
}

static inline int shadow_free_native_mutex(NativeMutex *m) {
    return pthread_mutex_destroy(m);
}

static inline void shadow_native_mutex_lock(NativeMutex *m) {
    pthread_mutex_lock(m);
}

static inline void shadow_native_mutex_unlock(NativeMutex *m) {
    pthread_mutex_unlock(m);
}

// --- Native Condition Variable Inlines ---

static inline int shadow_init_native_cond(NativeCond *c) {
    return pthread_cond_init(c, nullptr);
}

static inline int shadow_free_native_cond(NativeCond *c) {
    return pthread_cond_destroy(c);
}

static inline void shadow_native_cond_wait(NativeCond *c, NativeMutex *m) {
    pthread_cond_wait(c, m);
}

static inline void shadow_native_cond_broadcast(NativeCond *c) {
    pthread_cond_broadcast(c);
}

// --- Macro Wrappers (Automatic Address-of) ---

#    define INIT_NATIVE_MUTEX(m)      shadow_init_native_mutex(&(m))
#    define FREE_NATIVE_MUTEX(m)      shadow_free_native_mutex(&(m))
#    define NATIVE_MUTEX_LOCK(m)      shadow_native_mutex_lock(&(m))
#    define NATIVE_MUTEX_UNLOCK(m)    shadow_native_mutex_unlock(&(m))

#    define INIT_NATIVE_COND(c)       shadow_init_native_cond(&(c))
#    define FREE_NATIVE_COND(c)       shadow_free_native_cond(&(c))
#    define NATIVE_COND_WAIT(c, m)    shadow_native_cond_wait(&(c), &(m))
#    define NATIVE_COND_BROADCAST(c)  shadow_native_cond_broadcast(&(c))

#endif

// --- Threading Primitives (ShadowMutex Shim) ---

#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
/**
 * 1. TSan Fallback: Map ShadowMutex to NativeMutex (struct-based).
 * We treat m as a pointer to the NativeMutex instance.
 */
typedef NativeMutex ShadowMutex;

#    define SHADOW_LOCK(m)                                                                         \
        do {                                                                                       \
            static_assert(_Generic((m), NativeMutex *: 1, default: 0));                            \
            NATIVE_MUTEX_LOCK(*(m));                                                               \
        } while (false)

#    define SHADOW_UNLOCK(m)                                                                       \
        do {                                                                                       \
            static_assert(_Generic((m), NativeMutex *: 1, default: 0));                            \
            NATIVE_MUTEX_UNLOCK(*(m));                                                             \
        } while (false)

#    define INIT_LOCK(m)                                                                           \
        (static_assert(_Generic((m), NativeMutex: 1, default: 0)), INIT_NATIVE_MUTEX(m))

#    define FREE_LOCK(m)                                                                           \
        (static_assert(_Generic((m), NativeMutex: 1, default: 0)), FREE_NATIVE_MUTEX(m))

#elif PY_VERSION_HEX >= 0x030D0000
/**
 * 1. Python 3.13+ Production: PyMutex (1-byte, stack/struct allocated)
 */
typedef PyMutex ShadowMutex;

#    define SHADOW_LOCK(m)                                                                         \
        do {                                                                                       \
            static_assert(_Generic((m), PyMutex *: 1, default: 0));                                \
            PyMutex_Lock(m);                                                                       \
        } while (false)

#    define SHADOW_UNLOCK(m)                                                                       \
        do {                                                                                       \
            static_assert(_Generic((m), PyMutex *: 1, default: 0));                                \
            PyMutex_Unlock(m);                                                                     \
        } while (false)

#    define INIT_LOCK(m)                                                                           \
        do {                                                                                       \
            static_assert(_Generic((m), PyMutex: 1, default: 0));                                  \
            memset(&(m), 0, sizeof(PyMutex));                                                      \
        } while (false)

#    define FREE_LOCK(m) /* No-op for PyMutex */

#else
/**
 * 2. Legacy CPython 3.12 and older: PyThread_type_lock (void* handle)
 */
typedef PyThread_type_lock ShadowMutex;

#    define SHADOW_LOCK(m)                                                                         \
        do {                                                                                       \
            static_assert(_Generic((m), PyThread_type_lock *: 1, default: 0));                     \
            PyThread_acquire_lock(*(m), 1);                                                        \
        } while (false)

#    define SHADOW_UNLOCK(m)                                                                       \
        do {                                                                                       \
            static_assert(_Generic((m), PyThread_type_lock *: 1, default: 0));                     \
            PyThread_release_lock(*(m));                                                           \
        } while (false)

#    define INIT_LOCK(m)                                                                           \
        do {                                                                                       \
            static_assert(_Generic((m), PyThread_type_lock: 1, default: 0));                       \
            (m) = PyThread_allocate_lock();                                                        \
        } while (false)

#    define FREE_LOCK(m)                                                                           \
        do {                                                                                       \
            static_assert(_Generic((m), PyThread_type_lock: 1, default: 0));                       \
            if ((m))                                                                               \
                PyThread_free_lock((m));                                                           \
            (m) = nullptr;                                                                         \
        } while (false)
#endif

// Blocks until the world is not mid-step.
// Must be called while holding SHADOW_LOCK. Re-acquires it before returning.
#define BLOCK_UNTIL_NOT_STEPPING(self)                                                             \
    do {                                                                                           \
        if (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed)) {                    \
            atomic_fetch_add_explicit(&(self)->waiting_threads, 1, memory_order_relaxed);          \
            while (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed)) {             \
                SHADOW_UNLOCK(&(self)->shadow_lock);                                               \
                Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                 \
                while (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed)) {         \
                    NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);             \
                }                                                                                  \
                NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                                      \
                Py_END_ALLOW_THREADS SHADOW_LOCK(&(self)->shadow_lock);                            \
            }                                                                                      \
            atomic_fetch_sub_explicit(&(self)->waiting_threads, 1, memory_order_relaxed);          \
        }                                                                                          \
    } while (0)

#define BLOCK_UNTIL_NOT_QUERYING(self)                                                             \
    do {                                                                                           \
        while (atomic_load_explicit(&(self)->active_queries, memory_order_acquire) > 0) {          \
            SHADOW_UNLOCK(&(self)->shadow_lock);                                                   \
            Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                     \
            /* The Double Check */                                                                 \
            while (atomic_load_explicit(&(self)->active_queries, memory_order_relaxed) > 0) {      \
                NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);                 \
            }                                                                                      \
            NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                                          \
            Py_END_ALLOW_THREADS SHADOW_LOCK(&(self)->shadow_lock);                                \
        }                                                                                          \
    } while (0)

// Queries use this to wait if a Step is about to happen
#define BLOCK_IF_STEP_PENDING(self)                                                                \
    do {                                                                                           \
        if (atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) {                 \
            atomic_fetch_add_explicit(&(self)->waiting_threads, 1, memory_order_relaxed);          \
            while (atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) {          \
                SHADOW_UNLOCK(&(self)->shadow_lock);                                               \
                Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                 \
                while (atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) {      \
                    NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);             \
                }                                                                                  \
                NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                                      \
                Py_END_ALLOW_THREADS SHADOW_LOCK(&(self)->shadow_lock);                            \
            }                                                                                      \
            atomic_fetch_sub_explicit(&(self)->waiting_threads, 1, memory_order_relaxed);          \
        }                                                                                          \
    } while (0)

#define BLOCK_UNTIL_CAN_QUERY(self)                                                                \
    do {                                                                                           \
        if (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed) ||                    \
            atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) {                 \
            atomic_fetch_add_explicit(&(self)->waiting_threads, 1, memory_order_relaxed);          \
            while (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed) ||             \
                   atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) {          \
                SHADOW_UNLOCK(&(self)->shadow_lock);                                               \
                Py_BEGIN_ALLOW_THREADS NATIVE_MUTEX_LOCK((self)->step_sync.mutex);                 \
                while (atomic_load_explicit(&(self)->is_stepping, memory_order_relaxed) ||         \
                       atomic_load_explicit(&(self)->step_requested, memory_order_relaxed)) {      \
                    NATIVE_COND_WAIT((self)->step_sync.cond, (self)->step_sync.mutex);             \
                }                                                                                  \
                NATIVE_MUTEX_UNLOCK((self)->step_sync.mutex);                                      \
                Py_END_ALLOW_THREADS SHADOW_LOCK(&(self)->shadow_lock);                            \
            }                                                                                      \
            atomic_fetch_sub_explicit(&(self)->waiting_threads, 1, memory_order_relaxed);          \
        }                                                                                          \
    } while (0)

// A container to sync state changes (stepping finished, query finished)
typedef struct {
    NativeMutex mutex;
    NativeCond cond;
} ShadowSync;

extern NativeMutex g_jph_trampoline_lock;
