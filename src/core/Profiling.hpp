#pragma once

// Optional timing instrumentation for the shadow-sync/step paths.
// Disabled by default; defining CULVERIN_PROFILE_SYNC compiles the macros
// to std::chrono::steady_clock measurements. (The legacy implementation
// read raw cycle counters with inline assembly and probed ARM PMU access
// behind a SIGILL handler to report the same numbers in cycles; wall-clock
// nanoseconds answer the same questions and cut on every platform.)

#include <cstdint>

#ifdef CULVERIN_PROFILE_SYNC
#    include <chrono>
#    include <cstdio>
#endif

namespace CPH {

struct Stat {
    std::uint64_t total_ns = 0;
    std::uint64_t min_ns   = UINT64_MAX;
    std::uint64_t max_ns   = 0;
    std::uint32_t count    = 0;

    void accumulate(std::uint64_t elapsed_ns) noexcept {
        total_ns += elapsed_ns;
        if (elapsed_ns < min_ns) {
            min_ns = elapsed_ns;
        }
        if (elapsed_ns > max_ns) {
            max_ns = elapsed_ns;
        }
        count++;
    }
};

#ifdef CULVERIN_PROFILE_SYNC

using ProfClock = std::chrono::steady_clock;

inline void ProfPrintElapsed(const char *label, std::uint64_t elapsed_ns, unsigned int count) {
    if (count > 0) {
        std::fprintf(stderr, "[culverin] %s: %llu ns for %u items (%.1f ns/item)\n", label,
                     static_cast<unsigned long long>(elapsed_ns), count,
                     static_cast<double>(elapsed_ns) / count);
    } else {
        std::fprintf(stderr, "[culverin] %s: %llu ns\n", label,
                     static_cast<unsigned long long>(elapsed_ns));
    }
}

inline std::uint64_t ProfElapsedNsSince(ProfClock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(ProfClock::now() - start).count());
}

#endif // CULVERIN_PROFILE_SYNC

} // namespace CPH

#ifdef CULVERIN_PROFILE_SYNC

#    define CPH_INIT_PROFILER() (void)0
#    define CPH_PROFILE_BEGIN(tag) const auto _cph_start_##tag = ::CPH::ProfClock::now()
#    define CPH_PROFILE_END(tag, label, count)                                                     \
        ::CPH::ProfPrintElapsed(label, ::CPH::ProfElapsedNsSince(_cph_start_##tag),                \
                                static_cast<unsigned int>(count))
#    define CPH_PROFILE_ACCUMULATE(tag, stat_ptr)                                                  \
        (stat_ptr)->accumulate(::CPH::ProfElapsedNsSince(_cph_start_##tag))

#else

#    define CPH_INIT_PROFILER() (void)0
#    define CPH_PROFILE_BEGIN(tag) (void)0
#    define CPH_PROFILE_END(tag, label, count) (void)0
#    define CPH_PROFILE_ACCUMULATE(tag, stat_ptr) (void)0

#endif
