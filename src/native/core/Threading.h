#pragma once
#include "core/CompilerSpecifics.h"
#include "threading/Mutex.h" // Native mutex and condition primitives
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
 * - jph_trampoline_lock protects the non-thread-safe JPH Physics System state.
 * 3. STEPPING INVARIANT:
 * - is_stepping = true  => No external thread may read/write Shadow Buffers.
 * - is_stepping = false => External (Python) threads may read Shadow Buffers under SHADOW_LOCK.
 * 4. DOUBLE-BUFFERING:
 * - Command queues are swapped under SHADOW_LOCK but flushed under jph_trampoline_lock.
 */

/* ============================================================================
 * 1. HARDWARE YIELDING
 * ============================================================================ */

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

[[maybe_unused]] static inline void culverin_yield() {
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
 * 2. UNIFIED LOCK API (Directly mapping to CulvMutex/CulvCond)
 * ============================================================================ */

typedef CulvMutex NativeMutex;
typedef CulvCond NativeCond;
typedef CulvMutex ShadowMutex;

// Mutex Operations
#define INIT_NATIVE_MUTEX(m) atomic_init(&(m).bits, CULV_MUTEX_UNLOCKED)
#define FREE_NATIVE_MUTEX(m) (void)(m)
#define NATIVE_MUTEX_LOCK(m) culv_mutex_lock(&(m))
#define NATIVE_MUTEX_UNLOCK(m) culv_mutex_unlock(&(m))

// Condition Variable Operations
#define INIT_NATIVE_COND(c) culv_cond_init(&(c))
#define FREE_NATIVE_COND(c) (void)(c)
#define NATIVE_COND_WAIT(c, m) culv_cond_wait(&(c), &(m))
#define NATIVE_COND_BROADCAST(c) culv_cond_broadcast(&(c))

// Shadow Mutex Operations (Maps directly to Native operations)
#define INIT_LOCK(m) atomic_init(&(m).bits, CULV_MUTEX_UNLOCKED)
#define FREE_LOCK(m) (void)(m)
#define SHADOW_LOCK(m) culv_mutex_lock(m)
#define SHADOW_UNLOCK(m) culv_mutex_unlock(m)

/* ============================================================================
 * 3. STRUCTURES & CACHE ISOLATION
 * ============================================================================ */

constexpr auto CACHE_LINE_SIZE = 64;
#define CACHE_ISOLATE_PAD(current_size) (CACHE_LINE_SIZE - ((current_size) % CACHE_LINE_SIZE))

typedef struct {
    uint8_t _pad_before[CACHE_LINE_SIZE];

    NativeMutex mutex;
    NativeCond cond;

    // Automatically adjusts if the native mutex or condition size changes
    uint8_t
        _pad_after[CACHE_ISOLATE_PAD(sizeof(NativeMutex) + sizeof(NativeCond)) + CACHE_LINE_SIZE];
} ShadowSync;

extern NativeMutex g_jph_init_lock;