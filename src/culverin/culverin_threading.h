#pragma once
#include "culverin_compiler_specifics.h"
#include "mag_mutex.h" // 1-byte Mutex and Cond
#include <Python.h>
#include <stdatomic.h>

/**
 * Culverin Threading Invariants & Lock Hierarchy
 * ---------------------------------------------
 * 1. HIERARCHY: ShadowMutex (High) -> NativeMutex (Low).
 * - Always acquire shadow_lock BEFORE step_sync.mutex.
 * - To avoid deadlock, release shadow_lock before blocking on condition variables.
 * 2. OWNERSHIP:
 * - SHADOW_LOCK protects the Command Queue, Slot States, and Shadow Buffers.
 * - NATIVE_MUTEX/COND handles thread arbitration (parking/waking).
 * - self->jph_trampoline_lock protects the non-thread-safe JPH Physics System state.
 * 3. STEPPING INVARIANT:
 * - is_stepping = true  => No external thread may read/write Shadow Buffers.
 * - is_stepping = false => External (Python) threads may read Shadow Buffers under SHADOW_LOCK.
 * 4. DOUBLE-BUFFERING:
 * - Command queues are swapped under SHADOW_LOCK but flushed under self->jph_trampoline_lock.
 */

/* ============================================================================
 * 1. PROFILING & HARDWARE YIELDING
 * ============================================================================ */

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
    for (int i = 0; i < 100; i++) {
        culverin_cpu_relax();
    }
#if defined(_WIN32)
    if (SwitchToThread() == FALSE)
        Sleep(0);
#elif defined(__APPLE__)
    usleep(0);
#else
    sched_yield();
#endif
}

/* ============================================================================
 * 2. UNIFIED LOCK API (Directly mapping to MagMutex/MagCond)
 * ============================================================================ */

typedef MagMutex NativeMutex;
typedef MagCond NativeCond;
typedef MagMutex ShadowMutex;

// Mutex Operations
#define INIT_NATIVE_MUTEX(m) atomic_init(&(m).bits, MAG_UNLOCKED)
#define FREE_NATIVE_MUTEX(m) (void)(m)
#define NATIVE_MUTEX_LOCK(m) MagMutex_Lock(&(m))
#define NATIVE_MUTEX_UNLOCK(m) MagMutex_Unlock(&(m))

// Condition Variable Operations
#define INIT_NATIVE_COND(c) MagCond_Init(&(c))
#define FREE_NATIVE_COND(c) (void)(c)
#define NATIVE_COND_WAIT(c, m) MagCond_Wait(&(c), &(m))
#define NATIVE_COND_BROADCAST(c) MagCond_Broadcast(&(c))

// Shadow Mutex Operations (Maps directly to Native operations)
#define INIT_LOCK(m) atomic_init(&(m).bits, MAG_UNLOCKED)
#define FREE_LOCK(m) (void)(m)
#define SHADOW_LOCK(m) MagMutex_Lock(m)
#define SHADOW_UNLOCK(m) MagMutex_Unlock(m)

/* ============================================================================
 * 3. STRUCTURES & CACHE ISOLATION
 * ============================================================================ */

constexpr auto CACHE_LINE_SIZE = 64;
#define CACHE_ISOLATE_PAD(current_size) (CACHE_LINE_SIZE - ((current_size) % CACHE_LINE_SIZE))

typedef struct {
    uint8_t _pad_before[CACHE_LINE_SIZE];

    NativeMutex mutex;
    NativeCond cond;

    // Automatically adjusts if MagMutex/MagCond ever change size
    uint8_t
        _pad_after[CACHE_ISOLATE_PAD(sizeof(NativeMutex) + sizeof(NativeCond)) + CACHE_LINE_SIZE];
} ShadowSync;

extern NativeMutex g_jph_init_lock;