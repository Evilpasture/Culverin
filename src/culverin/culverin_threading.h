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
#    define INIT_NATIVE_MUTEX(m) InitializeSRWLock(&(m))
#    define FREE_NATIVE_MUTEX(m) (void)(m) // No cleanup needed for SRWLock
#    define NATIVE_MUTEX_LOCK(m) AcquireSRWLockExclusive(&(m))
#    define NATIVE_MUTEX_UNLOCK(m) ReleaseSRWLockExclusive(&(m))

#    define INIT_NATIVE_COND(c) InitializeConditionVariable(&(c))
#    define FREE_NATIVE_COND(c) (void)(c) // No cleanup needed
#    define NATIVE_COND_WAIT(c, m) SleepConditionVariableSRW(&(c), &(m), INFINITE, 0)
#    define NATIVE_COND_BROADCAST(c) WakeAllConditionVariable(&(c))
#else
#    include <pthread.h>
typedef pthread_mutex_t NativeMutex;
typedef pthread_cond_t NativeCond;
#    define INIT_NATIVE_MUTEX(m) pthread_mutex_init(&(m), nullptr)
#    define FREE_NATIVE_MUTEX(m) pthread_mutex_destroy(&(m))
#    define NATIVE_MUTEX_LOCK(m) pthread_mutex_lock(&(m))
#    define NATIVE_MUTEX_UNLOCK(m) pthread_mutex_unlock(&(m))

#    define INIT_NATIVE_COND(c) pthread_cond_init(&(c), nullptr)
#    define FREE_NATIVE_COND(c) pthread_cond_destroy(&(c))
#    define NATIVE_COND_WAIT(c, m) pthread_cond_wait(&(c), &(m))
#    define NATIVE_COND_BROADCAST(c) pthread_cond_broadcast(&(c))
#endif

// --- Threading Primitives (ShadowMutex Shim) ---

#if defined(__SANITIZE_THREAD__) || defined(ENABLE_SANITIZER)
// 1. TSan Fallback: TSan cannot see inside PyMutex's inline atomics, leading to false positives.
// Fallback to interceptable OS native mutexes during TSan builds.
typedef NativeMutex ShadowMutex;
#    define SHADOW_LOCK(m) NATIVE_MUTEX_LOCK(*(m))
#    define SHADOW_UNLOCK(m) NATIVE_MUTEX_UNLOCK(*(m))
#    define INIT_LOCK(m) INIT_NATIVE_MUTEX(m)
#    define FREE_LOCK(m) FREE_NATIVE_MUTEX(m)

#elif PY_VERSION_HEX >= 0x030D0000
// 2. Python 3.13+ Production: Use the ultra-fast, 1-byte PyMutex
typedef PyMutex ShadowMutex;
#    define SHADOW_LOCK(m) PyMutex_Lock(m)
#    define SHADOW_UNLOCK(m) PyMutex_Unlock(m)
#    define INIT_LOCK(m) memset(&(m), 0, sizeof(ShadowMutex))
#    define FREE_LOCK(m)

#else
// 3. Legacy CPython 3.12 and older
typedef PyThread_type_lock ShadowMutex;
#    define SHADOW_LOCK(m) PyThread_acquire_lock(*(m), 1)
#    define SHADOW_UNLOCK(m) PyThread_release_lock(*(m))
#    define INIT_LOCK(m) (m) = PyThread_allocate_lock()
#    define FREE_LOCK(m)                                                                           \
        do {                                                                                       \
            if (m)                                                                                 \
                PyThread_free_lock(m);                                                             \
            (m) = nullptr;                                                                            \
        } while (0)
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
