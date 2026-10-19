#pragma once

// Culverin Threading Invariants & Lock Hierarchy
// ---------------------------------------------
// 1. HIERARCHY: shadow_lock (High) -> step_sync.mutex (Low).
//    Always acquire shadow_lock BEFORE step_sync.mutex, and release
//    shadow_lock before blocking on step_sync.cond.
// 2. OWNERSHIP:
//    - shadow_lock protects the command queue, slot states, and shadow buffers.
//    - step_sync handles thread arbitration (parking/waking).
//    - jph_trampoline_lock protects the non-thread-safe JPH PhysicsSystem.
// 3. STEPPING INVARIANT:
//    - is_stepping = true  => no external thread may read/write shadow buffers.
//    - is_stepping = false => external threads may read shadow buffers under shadow_lock.
// 4. DOUBLE-BUFFERING:
//    - Command queues are swapped under shadow_lock but flushed under
//      jph_trampoline_lock.
//
// Synchronization uses std::mutex / std::condition_variable / std::atomic.
// The legacy 1-byte custom mutex optimized per-mutex footprint, but a world
// owns three mutexes total; correctness tooling (TSan, libc debug checks)
// understands the standard primitives and did not understand the custom one.

#include <condition_variable>
#include <cstdint>
#include <mutex>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    undef MemoryBarrier
#    undef far
#    undef near
#    undef ERROR
#else
#    include <sched.h>
#endif

// Processor-level hint to save power during spin-waits.
inline void CPH_CpuRelax() noexcept {
#if defined(__GNUC__) || defined(__clang__)
#    if defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__("pause");
#    elif defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("yield");
#    endif
#endif
}

inline void CPH_CpuYield() noexcept {
    for (int i = 0; i < 100; i++) {
        CPH_CpuRelax();
    }
#if defined(_WIN32)
    if (SwitchToThread() == FALSE) {
        Sleep(0);
    }
#else
    (void)sched_yield();
#endif
}

// Parking/waking endpoint for step and query arbitration. Cache isolation
// is provided by the spacer members around it in CPH_PhysicsWorld (the
// world lives in CPython-allocated storage, so it cannot be over-aligned).
struct CPH_ShadowSync {
    std::mutex mutex;
    std::condition_variable cond;
};

// Serializes first-time Jolt PhysicsSystem creation across worlds (TSan:
// lets Jolt's internal lazy statics initialize safely). Inline variable:
// exactly one instance, constructed before first use, no init call needed.
inline std::mutex g_jph_init_lock;
