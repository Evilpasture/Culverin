#pragma once

// #define CULVERIN_PROFILE_SYNC

#include <stdint.h>

#ifdef CULVERIN_PROFILE_SYNC
#    include <inttypes.h>
#    include <stdatomic.h>
#    include <stdio.h>
#    include <time.h>
#endif

typedef struct {
    uint64_t total_cycles;
    uint64_t min_cycles;
    uint64_t max_cycles;
    uint32_t count;
} CPH_Stat;

#ifdef CULVERIN_PROFILE_SYNC
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

#    if defined(__aarch64__) || defined(__arm64__)

static int cph_use_pmccntr = -1; /* -1 = uninitialised */

static inline void CPH_ProbePmu(void) {
    if (cph_use_pmccntr != -1)
        return;
    uint64_t val = 0;
    /* PMUSERENR_EL0 bit 0 grants user access; a SIGILL here means denied */
    __asm__ __volatile__("mrs %0, pmccntr_el0" : "=r"(val));
    cph_use_pmccntr = 1;
}

/* Call once before profiling; installs SIGILL handler to detect PMU access */
#        include <setjmp.h>
#        include <signal.h>

static volatile sig_atomic_t cph_pmu_fault;
static jmp_buf cph_pmu_jmp;
static void CPH_PmuSigill(int sig) {
    (void)sig;
    cph_pmu_fault = 1;
    longjmp(cph_pmu_jmp, 1);
}

static inline void CPH_InitCounters(void) {
    struct sigaction sa = {0}, old;
    sa.sa_handler       = CPH_PmuSigill;
    sigaction(SIGILL, &sa, &old);
    cph_pmu_fault = 0;
    if (setjmp(cph_pmu_jmp) == 0) {
        uint64_t v;
        __asm__ __volatile__("mrs %0, pmccntr_el0" : "=r"(v));
        cph_use_pmccntr = 1;
    } else {
        cph_use_pmccntr = 0;
        fprintf(stderr, "[culverin] pmccntr_el0 denied — falling back to "
                        "cntvct_el0 (timer ticks, NOT cycles)\n");
    }
    sigaction(SIGILL, &old, nullptr);
}

// On Apple Silicon, we cannot use pmccntr_el0 (Cycles) in user-space.
// We use cntvct_el0 (Virtual Timer). It frequency is typically 24MHz.
static inline uint64_t CPH_ReadStart(void) {
    uint64_t val;
    // ISB (Instruction Synchronization Barrier) ensures all previous
    // instructions finished before we read the timer.
    __asm__ __volatile__("isb\n\t"
                         "mrs %0, cntvct_el0"
                         : "=r"(val)::"memory");
    return val;
}

static inline uint64_t CPH_ReadEnd(void) {
    uint64_t val;
    // ISB ensures the work is finished before we read the timer.
    __asm__ __volatile__("mrs %0, cntvct_el0\n\t"
                         "isb"
                         : "=r"(val)::"memory");
    return val;
}

#        define CPH_INIT_PROFILER()                                                               \
            fprintf(stderr, "[culverin] Using ARM Virtual Timer (cntvct_el0)\n")

#    elif defined(_MSC_VER)

#        include <intrin.h>

static inline uint64_t CPH_ReadStart(void) {
    _mm_lfence();
    return __rdtsc();
}
static inline uint64_t CPH_ReadEnd(void) {
    /* rdtscp serialises the end read; lfence stops later loads from retiring
     * before we capture the counter */
    unsigned int aux;
    uint64_t v = __rdtscp(&aux);
    _mm_lfence();
    return v;
}
#        define CPH_INIT_PROFILER() ((void)0)

#    else /* GCC/Clang x86-64 */

static inline uint64_t CPH_ReadStart(void) {
    uint32_t lo, hi;
    /* cpuid is the only reliable full serialising instruction on x86 */
    __asm__ __volatile__("cpuid\n\t"
                         "rdtsc"
                         : "=a"(lo), "=d"(hi)
                         : "a"(0)
                         : "%rbx", "%rcx", "memory");
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t CPH_ReadEnd(void) {
    uint32_t lo, hi;
    uint32_t aux; /* rdtscp writes TSC_AUX (CPU id) into ecx — must consume it */
    __asm__ __volatile__("rdtscp\n\t"
                         "mov %%eax, %0\n\t"
                         "mov %%edx, %1\n\t"
                         "mov %%ecx, %2\n\t"
                         "cpuid" /* fence: stops the block after leaking in */
                         : "=r"(lo), "=r"(hi), "=r"(aux)
                         :
                         : "%rax", "%rbx", "%rcx", "%rdx", "memory");
    return ((uint64_t)hi << 32) | lo;
}
#        define CPH_INIT_PROFILER() ((void)0)

#    endif /* platform */

/* ── Public macros ──────────────────────────────────────────────────────────
 *
 * Usage:
 *   CPH_INIT_PROFILER();               // once at startup (ARM only needs it)
 *
 *   CPH_PROFILE_BEGIN(tag);
 *   ... work ...
 *   CPH_PROFILE_END(tag, label, count);
 *
 * `tag` is a plain identifier — must be unique within the scope.
 * `count` is the body/item count for cyc/body output; pass 0 to suppress.
 * ────────────────────────────────────────────────────────────────────────── */

#    define CPH_PROFILE_BEGIN(tag) uint64_t _cph_start_##tag = CPH_ReadStart()

#    define CPH_PROFILE_END(tag, label, count)                                                    \
        do {                                                                                       \
            uint64_t _cph_end     = CPH_ReadEnd();                                              \
            uint64_t _cph_elapsed = _cph_end - _cph_start_##tag;                                \
            unsigned int _cph_count   = (unsigned int)(count);                                        \
            if (_cph_count > 0) {                                                                     \
                fprintf(stderr,                                                                    \
                        "[culverin] %s: %" PRIu64 " cycles for %u items"                           \
                        " (%.1f cyc/item)\n",                                                      \
                        label, _cph_elapsed, _cph_count, (double)_cph_elapsed / _cph_count);           \
            } else {                                                                               \
                fprintf(stderr, "[culverin] %s: %" PRIu64 " cycles\n", label, _cph_elapsed);      \
            }                                                                                      \
        } while (0)

#    define CPH_PROFILE_ACCUMULATE(tag, stat_ptr)                                                 \
        do {                                                                                       \
            uint64_t _end     = CPH_ReadEnd();                                                   \
            uint64_t _elapsed = _end - _cph_start_##tag;                                          \
            (stat_ptr)->total_cycles += _elapsed;                                                  \
            if (_elapsed < (stat_ptr)->min_cycles)                                                 \
                (stat_ptr)->min_cycles = _elapsed;                                                 \
            if (_elapsed > (stat_ptr)->max_cycles)                                                 \
                (stat_ptr)->max_cycles = _elapsed;                                                 \
            (stat_ptr)->count++;                                                                   \
        } while (0)

#else /* CULVERIN_PROFILE_SYNC not defined */

#    define CPH_INIT_PROFILER() ((void)0)
#    define CPH_PROFILE_BEGIN(tag) ((void)0)
#    define CPH_PROFILE_END(tag, label, count) ((void)0)
#    define CPH_PROFILE_ACCUMULATE(tag, stat_ptr) ((void)0)

#endif /* CULVERIN_PROFILE_SYNC */

#ifdef CULVERIN_PROFILE_SYNC
#    define CPH_SYNC_START_TIMER(start)                                                                \
        struct timespec start;                                                                     \
        clock_gettime(CLOCK_MONOTONIC, &start)
#    define CPH_SYNC_END_TIMER(start, self)                                                            \
        do {                                                                                       \
            struct timespec end;                                                                   \
            clock_gettime(CLOCK_MONOTONIC, &end);                                                  \
            uint64_t ns =                                                                          \
                (end.tv_sec - start.tv_sec) * 1000000000ULL + (end.tv_nsec - start.tv_nsec);       \
            atomic_fetch_add_explicit(&(self)->step_sync.total_blocked_ns, ns,                     \
                                      memory_order_relaxed);                                       \
        } while (0)
#else
#    define CPH_SYNC_START_TIMER(start)
#    define CPH_SYNC_END_TIMER(start, self)
#endif
