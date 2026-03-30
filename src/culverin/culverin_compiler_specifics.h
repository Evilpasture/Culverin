#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Comment this line out to disable all debug prints
// #define CULVERIN_DEBUG

// --- Compiler Hints ---
#if defined(__GNUC__) || defined(__clang__)
#    define LIKELY(x) __builtin_expect(!!(x), 1)
#    define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
// Fallback for MSVC or other compilers that don't support built-in expect
#    define LIKELY(x) (x)
#    define UNLIKELY(x) (x)
#endif

// Use restrict keyword to tell the compiler these buffers do not overlap.
// This is the single best way to enable SIMD auto-vectorization.
#ifdef _MSC_VER
#    define CULV_RESTRICT __restrict
// MSVC doesn't have a direct equivalent to assume_aligned,
// though __assume( ((intptr_t)x & 31) == 0 ) is sometimes used.
#    define CULV_ASSUME_ALIGNED(x, alignment) (x)
#    define CULV_FORCE_INLINE __forceinline
#else
#    define CULV_RESTRICT __restrict__
#    define CULV_ASSUME_ALIGNED(x, alignment) __builtin_assume_aligned((x), (alignment))
#    define CULV_FORCE_INLINE inline __attribute__((always_inline))
#endif

// Use a prefixed function to avoid collision
[[noreturn]]
static inline void culv_unreachable(void) {
#if defined(CULVERIN_DEBUG)
    fprintf(stderr, "Unreachable hit at %s:%d\n", __FILE__, __LINE__);
    abort();
#elif defined(_MSC_VER)
    __assume(false);
#else
    __builtin_unreachable();
#endif
}

#define CULVERIN_PROFILE_SYNC

#ifdef CULVERIN_PROFILE_SYNC
#   include <stdio.h>
#   include <stdint.h>
#   include <inttypes.h>

/* ── Platform counter ───────────────────────────────────────────────────────
 *
 * x86-64: rdtsc/rdtscp — actual CPU cycle counter.
 *   Start: cpuid (full serialize) + rdtsc
 *   End:   rdtscp (load serialize) + cpuid (prevent later loads creeping in)
 *
 * ARM64: the PMU cycle counter (pmccntr_el0) is what we want, but it requires
 *   PMUSERENR_EL0.EN=1 which the kernel may not grant to userspace.
 *   We probe it at init and fall back to cntvct_el0 (virtual timer ticks,
 *   NOT cycles — caller is warned via a runtime message).
 *
 * MSVC/x86: __rdtsc() + _mm_lfence() standing in for cpuid.
 * ────────────────────────────────────────────────────────────────────────── */

#if defined(__aarch64__) || defined(__arm64__)

static int culv_use_pmccntr = -1; /* -1 = uninitialised */

static inline void culv_probe_pmu(void) {
    if (culv_use_pmccntr != -1) return;
    uint64_t val = 0;
    /* PMUSERENR_EL0 bit 0 grants user access; a SIGILL here means denied */
    __asm__ __volatile__("mrs %0, pmccntr_el0" : "=r"(val));
    culv_use_pmccntr = 1;
}

/* Call once before profiling; installs SIGILL handler to detect PMU access */
#   include <signal.h>
#   include <setjmp.h>
static volatile sig_atomic_t culv_pmu_fault;
static jmp_buf culv_pmu_jmp;
static void culv_pmu_sigill(int sig) { (void)sig; culv_pmu_fault = 1; longjmp(culv_pmu_jmp, 1); }

static inline void culv_init_counters(void) {
    struct sigaction sa = {0}, old;
    sa.sa_handler = culv_pmu_sigill;
    sigaction(SIGILL, &sa, &old);
    culv_pmu_fault = 0;
    if (setjmp(culv_pmu_jmp) == 0) {
        uint64_t v;
        __asm__ __volatile__("mrs %0, pmccntr_el0" : "=r"(v));
        culv_use_pmccntr = 1;
    } else {
        culv_use_pmccntr = 0;
        fprintf(stderr, "[culverin] pmccntr_el0 denied — falling back to "
                        "cntvct_el0 (timer ticks, NOT cycles)\n");
    }
    sigaction(SIGILL, &old, NULL);
}

static inline uint64_t culv_read_start(void) {
    uint64_t val;
    if (culv_use_pmccntr) {
        __asm__ __volatile__(
            "isb\n\t"
            "mrs %0, pmccntr_el0"
            : "=r"(val) :: "memory");
    } else {
        __asm__ __volatile__(
            "isb\n\t"
            "mrs %0, cntvct_el0"
            : "=r"(val) :: "memory");
    }
    return val;
}

static inline uint64_t culv_read_end(void) {
    uint64_t val;
    if (culv_use_pmccntr) {
        __asm__ __volatile__(
            "mrs %0, pmccntr_el0\n\t"
            "isb"
            : "=r"(val) :: "memory");
    } else {
        __asm__ __volatile__(
            "mrs %0, cntvct_el0\n\t"
            "isb"
            : "=r"(val) :: "memory");
    }
    return val;
}

#   define CULV_INIT_PROFILER() culv_init_counters()

#elif defined(_MSC_VER)

#   include <intrin.h>

static inline uint64_t culv_read_start(void) {
    _mm_lfence();
    return __rdtsc();
}
static inline uint64_t culv_read_end(void) {
    /* rdtscp serialises the end read; lfence stops later loads from retiring
     * before we capture the counter */
    unsigned int aux;
    uint64_t v = __rdtscp(&aux);
    _mm_lfence();
    return v;
}
#   define CULV_INIT_PROFILER() ((void)0)

#else /* GCC/Clang x86-64 */

static inline uint64_t culv_read_start(void) {
    uint32_t lo, hi;
    /* cpuid is the only reliable full serialising instruction on x86 */
    __asm__ __volatile__(
        "cpuid\n\t"
        "rdtsc"
        : "=a"(lo), "=d"(hi)
        : "a"(0)
        : "%rbx", "%rcx", "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t culv_read_end(void) {
    uint32_t lo, hi;
    uint32_t aux; /* rdtscp writes TSC_AUX (CPU id) into ecx — must consume it */
    __asm__ __volatile__(
        "rdtscp\n\t"
        "mov %%eax, %0\n\t"
        "mov %%edx, %1\n\t"
        "mov %%ecx, %2\n\t"
        "cpuid"                       /* fence: stops the block after leaking in */
        : "=r"(lo), "=r"(hi), "=r"(aux)
        :
        : "%rax", "%rbx", "%rcx", "%rdx", "memory");
    return ((uint64_t)hi << 32) | lo;
}
#   define CULV_INIT_PROFILER() ((void)0)

#endif /* platform */

/* ── Public macros ──────────────────────────────────────────────────────────
 *
 * Usage:
 *   CULV_INIT_PROFILER();               // once at startup (ARM only needs it)
 *
 *   CULV_PROFILE_BEGIN(tag);
 *   ... work ...
 *   CULV_PROFILE_END(tag, label, count);
 *
 * `tag` is a plain identifier — must be unique within the scope.
 * `count` is the body/item count for cyc/body output; pass 0 to suppress.
 * ────────────────────────────────────────────────────────────────────────── */

#   define CULV_PROFILE_BEGIN(tag) \
        uint64_t _culv_start_##tag = culv_read_start()

#   define CULV_PROFILE_END(tag, label, count) \
        do { \
            uint64_t _culv_end = culv_read_end(); \
            uint64_t _culv_elapsed = _culv_end - _culv_start_##tag; \
            unsigned int _culv_c = (unsigned int)(count); \
            if (_culv_c > 0) { \
                fprintf(stderr, "[culverin] %s: %" PRIu64 " cycles for %u items" \
                                " (%.1f cyc/item)\n", \
                        label, _culv_elapsed, _culv_c, \
                        (double)_culv_elapsed / _culv_c); \
            } else { \
                fprintf(stderr, "[culverin] %s: %" PRIu64 " cycles\n", \
                        label, _culv_elapsed); \
            } \
        } while (0)

#else /* CULVERIN_PROFILE_SYNC not defined */

#   define CULV_INIT_PROFILER()           ((void)0)
#   define CULV_PROFILE_BEGIN(tag)        ((void)0)
#   define CULV_PROFILE_END(tag, label, count) ((void)0)

#endif /* CULVERIN_PROFILE_SYNC */

#if defined(__clang__) || defined(__GNUC__)
// __builtin_prefetch(addr, rw, locality)
// rw: 0 = read, 1 = write
// locality: 0 = none, 3 = high (keep in L1)
#    define CULV_PREFETCH(addr) __builtin_prefetch((const void *)(addr), 0, 3)
#    define CULV_PREFETCH_WRITE(addr) __builtin_prefetch((const void *)(addr), 1, 3)
#elif defined(_MSC_VER)
#    include <mmintrin.h>
// _mm_prefetch(addr, hint)
// _MM_HINT_T0 = Prefetch into all cache levels (L1, L2, L3)
#    define CULV_PREFETCH(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#    define CULV_PREFETCH_WRITE(addr) _mm_prefetch((const char *)(addr), _MM_HINT_T0)
#else
// Fallback for compilers that don't support prefetching
#    define CULV_PREFETCH(addr) ((void)0)
#    define CULV_PREFETCH_WRITE(addr) ((void)0)
#endif

// Use a nested check to avoid the "macro not defined" evaluation error
#if defined(__has_c_attribute)
#    if __has_c_attribute(nodiscard)
#        define CULV_NODISCARD [[nodiscard]]
#        define CULV_MAYBE_UNUSED [[maybe_unused]]
#    else
#        define CULV_NODISCARD
#        define CULV_MAYBE_UNUSED
#    endif
#elif defined(_MSC_VER)
#    define CULV_NODISCARD _Check_return_
#    define CULV_MAYBE_UNUSED
#elif defined(__GNUC__) || defined(__clang__)
#    define CULV_NODISCARD __attribute__((warn_unused_result))
#    define CULV_MAYBE_UNUSED __attribute__((unused))
#else
#    define CULV_NODISCARD
#    define CULV_MAYBE_UNUSED
#endif

// --- Compiler Assume Hint ---
// Tells the compiler an expression is guaranteed to be true, allowing it to
// optimize away range checks and improve loop unrolling.
#if defined(CULVERIN_DEBUG)
#  define CULV_ASSUME(x) do { if (!(x)) { \
       fprintf(stderr, "Assumption failed: %s at %s:%d\n", #x, __FILE__, __LINE__); \
       abort(); } } while(0)
#elif defined(__clang__) || defined(__GNUC__)
#  define CULV_ASSUME(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#elif defined(_MSC_VER)
#  define CULV_ASSUME(x) __assume(x)
#else
#  define CULV_ASSUME(x) ((void)0)
#endif

// Force TSan to ignore a specific function
#if defined(__has_feature)
#  if __has_feature(thread_sanitizer)
#    define CULV_NO_TSAN __attribute__((no_sanitize("thread")))
#  else
#    define CULV_NO_TSAN
#  endif
#elif defined(__SANITIZE_THREAD__)
#  define CULV_NO_TSAN __attribute__((no_sanitize("thread")))
#else
#  define CULV_NO_TSAN
#endif

// NOLINTNEXTLINE(readability-identifier-naming)
#define PyCFunction_DeclareMethod CULV_NODISCARD static PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyCFunction_DeclareMethodFromModule CULV_NODISCARD extern PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyGetSet_DeclareGetter CULV_NODISCARD extern PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyGetSet_DeclareSetter CULV_NODISCARD extern PyObject *

// For tp_alloc and factory-style slots
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_Object CULV_NODISCARD static PyObject *
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_ObjectFromModule CULV_NODISCARD extern PyObject *

// For tp_init and status-returning slots
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_Status CULV_NODISCARD static int
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_StatusFromModule CULV_NODISCARD extern int

// For tp_dealloc (Cannot be nodiscard because it returns void)
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_Void static void
// NOLINTNEXTLINE(readability-identifier-naming)
#define PyType_DeclareSlot_VoidFromModule extern void

CULV_MAYBE_UNUSED static constexpr size_t MEMORY_ALIGNMENT_SIZE = 64;

#if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 202311L
#    define CULV_REPRODUCIBLE [[reproducible]]
#    define CULV_UNSEQUENCED [[unsequenced]]
#else
#    define CULV_REPRODUCIBLE
#    define CULV_UNSEQUENCED
#endif
