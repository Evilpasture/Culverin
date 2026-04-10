#pragma once
#include "culverin_compiler_specifics.h"
#include <Python.h>
#include <stdatomic.h>
// This is a custom Mutex I made. 1 byte!
#include "mag_mutex.h"

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

/**
 * MagMutex & MagCond Abstraction Layer
 * Replaces SRWLOCK/CONDITION_VARIABLE (Win) and pthread_mutex/cond (POSIX)
 */


typedef MagMutex NativeMutex;
typedef MagCond NativeCond;

// --- Mutex Operations ---

static inline int internal_native_mutex_init(NativeMutex *m) {
    // MagMutex is safe to zero-initialize, but atomic_init is more explicit
#ifdef __cplusplus
    m->bits.store(MAG_UNLOCKED, std::memory_order_relaxed);
#else
    atomic_init(&m->bits, MAG_UNLOCKED);
#endif
    return 0;
}

static inline void internal_native_mutex_lock(NativeMutex *m) {
    MagMutex_Lock(m);
}

static inline void internal_native_mutex_unlock(NativeMutex *m) {
    MagMutex_Unlock(m);
}

static inline int internal_native_mutex_free(NativeMutex *m) {
    (void)m; // MagMutex is a 1-byte value type; no OS resources to free.
    return 0;
}

// --- Condition Variable Operations ---

static inline int internal_native_cond_init(NativeCond *c) {
    MagCond_Init(c);
    return 0;
}

static inline void internal_native_cond_wait(NativeCond *c, NativeMutex *m) {
    MagCond_Wait(c, m);
}

static inline void internal_native_cond_broadcast(NativeCond *c) {
    MagCond_Broadcast(c);
}

static inline int internal_native_cond_free(NativeCond *c) {
    (void)c; // MagCond is a 1-byte value type; no OS resources to free.
    return 0;
}

/* ============================================================================
 * 2. SHADOW MUTEX IMPLEMENTATIONS
 * ============================================================================ */

/**
 * SHIM LOGIC: 
 * On Python 3.13+ (Free-threaded), we call MagMutex directly.
 * On Python 3.12, we wrap MagMutex in Allow/End macros to prevent GIL deadlocks.
 */
typedef NativeMutex ShadowMutex;

static inline int internal_shadow_init(ShadowMutex *m) { 
    return internal_native_mutex_init(m); 
}

static inline void internal_shadow_lock(ShadowMutex *m) { 
#if PY_VERSION_HEX < 0x030D0000
    /* Python 3.12 shim: Release GIL so the thread can 'park' in C 
       without blocking the interpreter or signals like Ctrl+C. */
    Py_BEGIN_ALLOW_THREADS
    internal_native_mutex_lock(m); 
    Py_END_ALLOW_THREADS
#else
    /* Python 3.13+: Direct call. The runtime is parallel-friendly. */
    internal_native_mutex_lock(m); 
#endif
}

static inline void internal_shadow_unlock(ShadowMutex *m) { 
    /* Unlock is usually fast enough to keep the GIL, but 3.12 
       sometimes needs the same 'blink' if there's heavy contention. */
    internal_native_mutex_unlock(m); 
}

static inline int internal_shadow_free(ShadowMutex *m) { 
    return internal_native_mutex_free(m); 
}


/* ============================================================================
 * 3. PUBLIC API MACROS
 * ============================================================================ */

#define INIT_NATIVE_MUTEX(m) internal_native_mutex_init(&(m))
#define FREE_NATIVE_MUTEX(m) internal_native_mutex_free(&(m))
#define NATIVE_MUTEX_LOCK(m) internal_native_mutex_lock(&(m))
#define NATIVE_MUTEX_UNLOCK(m) internal_native_mutex_unlock(&(m))

#define INIT_NATIVE_COND(c) internal_native_cond_init(&(c))
#define FREE_NATIVE_COND(c) internal_native_cond_free(&(c))
#define NATIVE_COND_WAIT(c, m) internal_native_cond_wait(&(c), &(m))
#define NATIVE_COND_BROADCAST(c) internal_native_cond_broadcast(&(c))

#define INIT_LOCK(m) internal_shadow_init(&(m))
#define FREE_LOCK(m) internal_shadow_free(&(m))

/* These two accept the pointer directly as per culverin.c usage */
#define SHADOW_LOCK(m) internal_shadow_lock(m)
#define SHADOW_UNLOCK(m) internal_shadow_unlock(m)

/* ============================================================================
 * 4. STRUCTURES
 * ============================================================================ */


// Use a standard constant for cache line width
static constexpr auto CACHE_LINE_SIZE = 64;

/* A helper macro to calculate remaining space in a cache line */
#define CACHE_ISOLATE_PAD(current_size) (CACHE_LINE_SIZE - ((current_size) % CACHE_LINE_SIZE))

typedef struct {
    // 1. Isolate from the main PhysicsWorld fields (like active_queries)
    uint8_t _pad_before[CACHE_LINE_SIZE];

    // 2. The actual synchronization primitives (2 bytes)
    NativeMutex mutex;
    NativeCond cond;

    // 3. Isolate from trailing fields (like shadow_lock)
    // We use a calculated constant so if you change Mutex size, 
    // the padding adjusts automatically.
    uint8_t _pad_after[CACHE_ISOLATE_PAD(sizeof(NativeMutex) + sizeof(NativeCond)) + CACHE_LINE_SIZE];
} ShadowSync;

extern NativeMutex g_jph_trampoline_lock;