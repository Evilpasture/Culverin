#pragma once
#include "core/CompilerSpecifics.h"
#include "threading/Mutex.h" // Native mutex and condition primitives
#include <stdatomic.h>
#if !defined(_WIN32)
#    include <sched.h>
#endif

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
static inline void CPH_CpuRelax() {
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

[[maybe_unused]] static inline void CPH_CpuYield() {
    for (int i = 0; i < 100; i++) {
        CPH_CpuRelax();
    }
#if defined(_WIN32)
    if (SwitchToThread() == FALSE)
        Sleep(0);
#else
    (void)sched_yield();
#endif
}

/* ============================================================================
 * 2. UNIFIED LOCK API (Directly mapping to CPH_Mutex/CPH_Cond)
 * ============================================================================ */

typedef CPH_Mutex NativeMutex;
typedef CPH_Cond NativeCond;
typedef CPH_Mutex ShadowMutex;

// Mutex Operations
#define CPH_INIT_NATIVE_MUTEX(m) atomic_init(&(m).bits, kMutexUnlocked)
#define CPH_FREE_NATIVE_MUTEX(m) (void)(m)
#define CPH_NATIVE_MUTEX_LOCK(m) CPH_mutex_lock(&(m))
#define CPH_NATIVE_MUTEX_UNLOCK(m) CPH_mutex_unlock(&(m))

// Condition Variable Operations
#define CPH_INIT_NATIVE_COND(c) CPH_cond_init(&(c))
#define CPH_FREE_NATIVE_COND(c) (void)(c)
#define CPH_NATIVE_COND_WAIT(c, m) CPH_CondWait(&(c), &(m))
#define CPH_NATIVE_COND_BROADCAST(c) CPH_CondBroadcast(&(c))

// Shadow Mutex Operations (Maps directly to Native operations)
#define CPH_INIT_LOCK(m) atomic_init(&(m).bits, kMutexUnlocked)
#define CPH_FREE_LOCK(m) (void)(m)
#define CPH_SHADOW_LOCK(m) CPH_mutex_lock(m)
#define CPH_SHADOW_UNLOCK(m) CPH_mutex_unlock(m)

/* ============================================================================
 * 3. STRUCTURES & CACHE ISOLATION
 * ============================================================================ */

constexpr auto kCacheLineSize = 64;
#define CPH_CACHE_ISOLATE_PAD(current_size) (kCacheLineSize - ((current_size) % kCacheLineSize))

typedef struct {
    uint8_t _pad_before[kCacheLineSize];

    NativeMutex mutex;
    NativeCond cond;

    // Automatically adjusts if the native mutex or condition size changes
    uint8_t
        _pad_after[CPH_CACHE_ISOLATE_PAD(sizeof(NativeMutex) + sizeof(NativeCond)) + kCacheLineSize];
} CPH_ShadowSync;

extern NativeMutex g_jph_init_lock;