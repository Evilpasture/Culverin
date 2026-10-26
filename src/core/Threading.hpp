#pragma once

// Culverin Threading Invariants & Ownership Domains
// -------------------------------------------------
// The concurrency model is now organised around three isolated ownership
// domains (see docs/SUBSYSTEM_REFACTOR.md for the full rationale):
//
//   * CommandMailbox  - every Python-visible mutation is staged here and
//     flushed, in order, by the step thread before PhysicsSystem::Update().
//     Producers take only a lightweight push mutex; they never touch Jolt.
//   * BodyRegistry    - fixed-capacity slot/generation tables. Because they
//     are allocated once at max_bodies and never move, Resolve(handle) is a
//     zero-lock bounds check + one acquire load.
//   * TransformBuffer - double-buffered transform output. The step writes the
//     back frame and flips an atomic index; zero-copy memoryviews always read
//     the stable front frame. Readers never wait on writers.
//
// With transform reads lock-free and every mutation deferred, the coarse
// shadow_lock is demoted from a "universal gate" to a narrow mutex that only
// serialises the mailbox push / slot-table structural changes and the Jolt
// query & step arbitration below. The following hierarchy still documents the
// locks that remain for Jolt-touching operations; it is deliberately no
// longer on the transform-read hot path:
//
//        jph_trampoline_lock  ->  shadow_lock  ->  step_sync.mutex
//
//    - jph_trampoline_lock BEFORE shadow_lock for the few synchronous Jolt
//      structural operations (scene load, ragdoll/compound construction,
//      vehicle/character creation) that have not yet migrated to the mailbox.
//    - shadow_lock BEFORE step_sync.mutex; release shadow_lock before
//      blocking on step_sync.cond.
//    - step_sync is a narrow arbitration endpoint for step/query overlap of
//      non-transform operations, not a gate on reading positions/rotations.
//
// Synchronization uses std::mutex / std::condition_variable / std::atomic.
// The legacy 1-byte custom mutex optimized per-mutex footprint, but a world
// owns three mutexes total; correctness tooling (TSan, libc debug checks)
// understands the standard primitives and did not understand the custom one.

#include <condition_variable>
#include <cstdint>
#include <mutex>

// Platform headers must be included at global scope, never inside namespace
// CPH. <windows.h> is not self-contained: winnt.h includes <ctype.h> (line 34)
// and windef.h a dozen others, and every name those headers declare lands in
// whichever namespace is open at the include site. Wrapping the include opened
// CPH::isalnum, CPH::isalpha, CPH::isdigit ... while leaving the global
// namespace empty, which MSVC's <cctype> — a header that is nothing but
// `using ::isalnum;` — cannot survive. The reported failure was three thousand
// lines away ("no member named 'isalnum' in the global namespace", reached
// through Jolt's <sstream>), and the compiler's suggestion of iswalnum, which
// <cwchar> had declared at global scope earlier, pointed at the wrong header
// entirely.
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

namespace CPH {

// Processor-level hint to save power during spin-waits.
inline void CpuRelax() noexcept {
#if defined(__GNUC__) || defined(__clang__)
#    if defined(__i386__) || defined(__x86_64__)
    __asm__ __volatile__("pause");
#    elif defined(__arm__) || defined(__aarch64__)
    __asm__ __volatile__("yield");
#    endif
#endif
}

inline void CpuYield() noexcept {
    for (int i = 0; i < 100; i++) {
        CpuRelax();
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
// is provided by the spacer members around it in PhysicsWorld (the
// world lives in CPython-allocated storage, so it cannot be over-aligned).
struct ShadowSync {
    std::mutex mutex;
    std::condition_variable cond;
};

namespace Global {

// Serializes first-time Jolt PhysicsSystem creation across worlds (TSan: lets
// Jolt's internal lazy statics initialize safely).
//
// Declared here and defined in Threading.cpp, not as an inline variable: an
// `inline` global is still a global, and this is the only place in the tree
// that needs one in a header. Everything with static storage duration in the
// first-party engine lives in CPH::Global and has exactly one definition.
extern std::mutex jph_init_lock;

} // namespace Global

} // namespace CPH
