#pragma once

/**
 * @file Mutex.h
 * @brief CulvMutex: A High-Performance, Memory-Efficient 1-Byte Mutex.
 *
 * DESIGN GOALS:
 * 1. Minimal Footprint: 1 byte per mutex in Release mode (ideal for ECS/Large Object Graphs).
 * 2. High Contention Throughput: Uses a Decoupled Unlock Protocol and Address Sharding
 *    to eliminate "Thundering Herd" and False-Sharing bottlenecks.
 * 3. Cross-Platform: Specialized assembly for ARM64 (Apple Silicon) and x86_64.
 * 4. Safety: Built-in Deadlock Detection and Recursive Lock checks in Debug mode.
 *
 * BASIC USAGE:
 * @code
 *    CulvMutex my_lock = { .bits = CULV_MUTEX_UNLOCKED };
 *    // or zero-init is fine
 *    CulvMutex my_lock = {};
 *
 *    culv_mutex_lock(&my_lock);
 *    // ... critical section ...
 *    culv_mutex_unlock(&my_lock);
 * @endcode
 *
 * INITIALIZATION:
 * CulvMutex can be zero-initialized or initialized with { .bits = CULV_MUTEX_UNLOCKED }.
 * In Release mode (NDEBUG defined), the struct is exactly 8 bits. In Debug mode,
 * the struct grows to include ownership and diagnostic metadata.
 */

// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <assert.h>
#ifdef __cplusplus
#    include <atomic>
// Map C11/C23 names to C++ names for the inline functions
using std::memory_order_acq_rel;
using std::memory_order_acquire;
using std::memory_order_relaxed;
using std::memory_order_release;
#    define CULV_MUTEX_ATOMIC(T) std::atomic<T>
#else
#    include <stdatomic.h>
#    define CULV_MUTEX_ATOMIC(T) _Atomic T
#endif
#include <stdint.h>

#ifdef __cplusplus
#    define CULV_MUTEX_ATOMIC_CX(obj, exp, des, succ, fail)                                               \
        std::atomic_compare_exchange_strong_explicit(obj, exp, des, succ, fail)
#else
#    define CULV_MUTEX_ATOMIC_CX(obj, exp, des, succ, fail)                                               \
        atomic_compare_exchange_strong_explicit(obj, exp, des, succ, fail)
#endif

#ifndef NDEBUG
#    define CULV_MUTEX_DEBUG 1
#endif

#if defined(__GNUC__) || defined(__clang__)
#    define CULV_MUTEX_LIKELY(x) __builtin_expect(!!(x), 1)
#    define CULV_MUTEX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#    define CULV_MUTEX_COLD __attribute__((cold))
#    define CULV_MUTEX_ALWAYS_INLINE __attribute__((always_inline))
#    define CULV_MUTEX_ASSUME(x) __builtin_assume(x)
#else
#    define CULV_MUTEX_LIKELY(x) (x)
#    define CULV_MUTEX_UNLIKELY(x) (x)
#    define CULV_MUTEX_COLD
#    define CULV_MUTEX_ALWAYS_INLINE
#    define CULV_MUTEX_ASSUME(x) (x)
#endif

#if defined(CULV_MUTEX_EXPORT_INTERNAL) || !defined(__GNUC__) || defined(CULV_MUTEX_BUILDING_TESTS)
#    define CULV_MUTEX_INTERNAL [[gnu::visibility("default")]]
#else
#    define CULV_MUTEX_INTERNAL [[gnu::visibility("hidden")]]
#endif

// NOLINTBEGIN(readability-identifier-naming)
#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>

/* --- I HATE WINDOWS --- */
#    undef MemoryBarrier
#    undef far
#    undef near
#    undef ERROR
/* ---------------------- */
typedef CRITICAL_SECTION culv_platform_mutex_t;
typedef CONDITION_VARIABLE culv_platform_cond_t;
typedef DWORD culv_platform_thread_id_t;
#    define CULV_THREAD_PLATFORM_MUTEX_INIT(mod) InitializeCriticalSection(mod)
#    define CULV_THREAD_PLATFORM_MUTEX_LOCK(mod) EnterCriticalSection(mod)
#    define CULV_THREAD_PLATFORM_MUTEX_UNLOCK(mod) LeaveCriticalSection(mod)
#    define CULV_THREAD_PLATFORM_COND_INIT(condvar) InitializeConditionVariable(condvar)
#    define CULV_THREAD_PLATFORM_COND_WAIT(condvar, mod) SleepConditionVariableCS(condvar, mod, INFINITE)
#    define CULV_THREAD_PLATFORM_COND_SIGNAL(condvar) WakeConditionVariable(condvar)
#    define CULV_THREAD_PLATFORM_COND_DESTROY(condvar)
#    define CULV_THREAD_PLATFORM_CURRENT_ID() GetCurrentThreadId()
#    define CULV_THREAD_PLATFORM_THREADS_EQUAL(t1, t2) ((t1) == (t2))
#else
#    include <pthread.h>
#    include <sched.h>
typedef pthread_mutex_t culv_platform_mutex_t;
typedef pthread_cond_t culv_platform_cond_t;
typedef pthread_t culv_platform_thread_id_t;
#    define CULV_THREAD_PLATFORM_MUTEX_INIT(mod) pthread_mutex_init(mod, nullptr)
#    define CULV_THREAD_PLATFORM_MUTEX_LOCK(mod) pthread_mutex_lock(mod)
#    define CULV_THREAD_PLATFORM_MUTEX_UNLOCK(mod) pthread_mutex_unlock(mod)
#    define CULV_THREAD_PLATFORM_COND_INIT(condvar) pthread_cond_init(condvar, nullptr)
#    define CULV_THREAD_PLATFORM_COND_WAIT(condvar, mod) pthread_cond_wait(condvar, mod)
#    define CULV_THREAD_PLATFORM_COND_SIGNAL(condvar) pthread_cond_signal(condvar)
#    define CULV_THREAD_PLATFORM_COND_DESTROY(condvar) pthread_cond_destroy(condvar)
#    define CULV_THREAD_PLATFORM_CURRENT_ID() pthread_self()
#    define CULV_THREAD_PLATFORM_THREADS_EQUAL(t1, t2) pthread_equal((t1), (t2))
#endif
// NOLINTEND(readability-identifier-naming)
// --- Memory Ordering Configuration ---

#if defined(CULV_MUTEX_FORCE_CASAL)
// Experimental: Use Full Barriers for both Lock and Unlock
#    define CULV_MUTEX_LOCK_ORDER memory_order_acq_rel
#    define CULV_MUTEX_UNLOCK_ORDER memory_order_acq_rel
#else
// Standard: Standard Acquire for Lock, Release for Unlock
#    define CULV_MUTEX_LOCK_ORDER memory_order_acquire
#    define CULV_MUTEX_UNLOCK_ORDER memory_order_release
#endif
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

// --- CulvMutex States ---

constexpr uint8_t CULV_MUTEX_UNLOCKED    = 0x00;
constexpr uint8_t CULV_MUTEX_LOCKED      = 0x01;
constexpr uint8_t CULV_MUTEX_HAS_WAITERS = 0x02;
constexpr uint8_t CULV_MUTEX_POISONED    = 0x04;
#ifdef CULV_MUTEX_DEBUG
constexpr uint8_t CULV_MUTEX_DEBUG_ALIGNMENT = 32;
#endif
/**
 * @struct CulvMutex
 * @brief The core mutex structure.
 */
typedef struct CulvMutex {
#ifdef CULV_MUTEX_DEBUG
    alignas(CULV_MUTEX_DEBUG_ALIGNMENT)
#endif
        CULV_MUTEX_ATOMIC(uint8_t) bits;
#ifdef CULV_MUTEX_DEBUG
    CULV_MUTEX_ATOMIC(bool) has_owner;
    CULV_MUTEX_ATOMIC(uintptr_t) owner;
    CULV_MUTEX_ATOMIC(uint64_t) spin_success_count;
    CULV_MUTEX_ATOMIC(uint64_t) park_count;
#endif
} CulvMutex;

#ifdef __cplusplus
extern "C" {
#endif

// --- Debug Prototypes ---

#ifdef CULV_MUTEX_DEBUG
void culv_mutex_debug_check_pre_lock(CulvMutex *mod);
void culv_mutex_debug_post_lock(CulvMutex *mod);
void culv_mutex_debug_pre_unlock(CulvMutex *mod);
void culv_mutex_debug_clear_owner(CulvMutex *mod);
#else
static inline void culv_mutex_debug_check_pre_lock(CulvMutex *mod) { (void)mod; }
static inline void culv_mutex_debug_post_lock(CulvMutex *mod) { (void)mod; }
static inline void culv_mutex_debug_pre_unlock(CulvMutex *mod) { (void)mod; }
static inline void culv_mutex_debug_clear_owner(CulvMutex *mod) { (void)mod; }
#endif

[[gnu::cold, gnu::noinline, gnu::nonnull(1)]] CULV_MUTEX_INTERNAL
    // NOLINTNEXTLINE(readability-identifier-naming)
    void culv_mutex_lock_slow(CulvMutex *mod);
[[gnu::cold, gnu::noinline, gnu::nonnull(1)]] CULV_MUTEX_INTERNAL
    // NOLINTNEXTLINE(readability-identifier-naming)
    void culv_mutex_unlock_slow(CulvMutex *mod);

// --- Public API ---

// NOLINTBEGIN(hicpp-no-assembler, cppcoreguidelines-pro-type-inline-assembly)
/**
 * culv_cpu_relax: Hardware-specific hint to the CPU that we are in a spin-loop.
 * On ARM64 (Apple Silicon), this uses the 'yield' instruction to save power
 * and potentially allow the other SMT thread to progress.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
static inline void culv_cpu_relax() {
#if defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#elif defined(_WIN32)
    YieldProcessor();
#endif
}
// NOLINTEND(hicpp-no-assembler, cppcoreguidelines-pro-type-inline-assembly)

/**
 * @brief Permanently poisons the mutex. Any subsequent lock attempt will abort.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
static inline void culv_mutex_poison(CulvMutex *mod) {
    atomic_fetch_or_explicit(&mod->bits, CULV_MUTEX_POISONED, memory_order_release);
}

/**
 * @brief Attempts to acquire the lock without blocking.
 * @return true if acquired, false if contended.
 */
[[gnu::flatten]] [[gnu::hot]] [[gnu::always_inline]] [[gnu::artificial]]
// NOLINTNEXTLINE(readability-identifier-naming)
static inline bool culv_mutex_try_lock(CulvMutex *mod) {
    culv_mutex_debug_check_pre_lock(mod);
    uint8_t expected = CULV_MUTEX_UNLOCKED;
    bool success     = CULV_MUTEX_ATOMIC_CX(&mod->bits, &expected, CULV_MUTEX_LOCKED, memory_order_acquire,
                                     memory_order_relaxed);
    if (success) {
        culv_mutex_debug_post_lock(mod);
    }
    return success;
}

/**
 * @brief Acquires the lock.
 * On ARM64 with CULV_MUTEX_FORCE_CASAL, this emits 'casalb'.
 * Otherwise, emits 'casab' (Acquire).
 */
[[gnu::flatten, gnu::hot, gnu::always_inline, gnu::artificial, gnu::nonnull(1)]]
// NOLINTNEXTLINE(readability-identifier-naming)
static inline void culv_mutex_lock(CulvMutex *mod) {
    culv_mutex_debug_check_pre_lock(mod);
    uint8_t expected = CULV_MUTEX_UNLOCKED;

    if (CULV_MUTEX_LIKELY(CULV_MUTEX_ATOMIC_CX(&mod->bits, &expected, CULV_MUTEX_LOCKED, CULV_MUTEX_LOCK_ORDER,
                                 memory_order_relaxed))) {
        culv_mutex_debug_post_lock(mod);
        return;
    }
    culv_mutex_lock_slow(mod);
}

/**
 * @brief Releases the lock.
 * On ARM64 with CULV_MUTEX_FORCE_CASAL, this emits 'casalb'.
 * Otherwise, emits 'caslb' (Release).
 */
[[gnu::flatten, gnu::hot, gnu::always_inline, gnu::artificial, gnu::nonnull(1)]]
// NOLINTNEXTLINE(readability-identifier-naming)
static inline void culv_mutex_unlock(CulvMutex *mod) {
    culv_mutex_debug_pre_unlock(mod);
    culv_mutex_debug_clear_owner(mod);

    uint8_t expected = CULV_MUTEX_LOCKED;

    if (CULV_MUTEX_LIKELY(CULV_MUTEX_ATOMIC_CX(&mod->bits, &expected, CULV_MUTEX_UNLOCKED, CULV_MUTEX_UNLOCK_ORDER,
                                 memory_order_relaxed))) {
        return;
    }
    culv_mutex_unlock_slow(mod);
}

// --- CulvCond (Condition Variable) ---

/**
 * @struct CulvCond
 * @brief A 1-Byte Condition Variable.
 *
 * Works seamlessly with CulvMutex. Uses the same parking lot backend to
 * provide ultra-low memory footprint synchronization.
 */
typedef struct CulvCond {
    CULV_MUTEX_ATOMIC(uint8_t) bits;
} CulvCond;

/**
 * @brief Initializes a CulvCond.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
static inline void culv_cond_init(CulvCond *condvar) {
#ifdef __cplusplus
    condvar->bits.store(0, std::memory_order_relaxed);
#else
    atomic_init(&condvar->bits, 0);
#endif
}

/**
 * @brief Atomically releases the mutex and causes the calling thread to block on the condition
 * variable. Upon successful return, the mutex shall have been locked and is owned by the calling
 * thread.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
void culv_cond_wait(CulvCond *condvar, CulvMutex *mod);

/**
 * @brief Unblocks at least one of the threads that are blocked on the specified condition variable.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
void culv_cond_signal(CulvCond *condvar);

/**
 * @brief Unblocks all threads currently blocked on the specified condition variable.
 */
// NOLINTNEXTLINE(readability-identifier-naming)
void culv_cond_broadcast(CulvCond *condvar);

#ifdef __cplusplus
} // extern "C"
#endif
