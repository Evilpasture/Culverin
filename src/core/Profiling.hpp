#pragma once

// Optional timing instrumentation for the shadow-sync/step paths.
// Disabled by default; defining CULVERIN_PROFILE_SYNC turns the RAII scopes
// below into std::chrono::steady_clock measurements. (The legacy
// implementation read raw cycle counters with inline assembly and probed ARM
// PMU access behind a SIGILL handler to report the same numbers in cycles;
// wall-clock nanoseconds answer the same questions and cut on every platform.)
//
// Instrumentation is scoped, never manual: a ProfileScope region ends when the
// guard dies, so an early return cannot skip the report the way a paired
// BEGIN/END macro pair could. That is why these are classes and not macros.

#include <cstdint>

#ifdef CULVERIN_PROFILE_SYNC
#    include <chrono>
#    include <cstdio>
#endif

namespace CPH {

struct Stat {
    std::uint64_t totalNs = 0;
    std::uint64_t minNs   = UINT64_MAX;
    std::uint64_t maxNs   = 0;
    std::uint32_t count    = 0;

    void accumulate(std::uint64_t elapsed_ns) noexcept {
        totalNs += elapsed_ns;
        if (elapsed_ns < minNs) {
            minNs = elapsed_ns;
        }
        if (elapsed_ns > maxNs) {
            maxNs = elapsed_ns;
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

// Reports one elapsed interval to stderr on destruction.
class ProfileScope {
  public:
    explicit ProfileScope(const char *label, unsigned int count = 0) noexcept
        : label_(label), count_(count), start_(ProfClock::now()) {}

    ~ProfileScope() { ProfPrintElapsed(label_, ProfElapsedNsSince(start_), count_); }

    ProfileScope(const ProfileScope &)                    = delete;
    ProfileScope &operator=(const ProfileScope &)         = delete;
    ProfileScope(ProfileScope &&)                         = delete;
    ProfileScope &operator=(ProfileScope &&)              = delete;

  private:
    const char   *label_;
    unsigned int  count_;
    ProfClock::time_point start_;
};

// Folds one elapsed interval into a running Stat on destruction.
class ProfileAccumulator {
  public:
    explicit ProfileAccumulator(Stat &stat) noexcept : stat_(stat), start_(ProfClock::now()) {}

    ~ProfileAccumulator() { stat_.accumulate(ProfElapsedNsSince(start_)); }

    ProfileAccumulator(const ProfileAccumulator &)             = delete;
    ProfileAccumulator &operator=(const ProfileAccumulator &)  = delete;
    ProfileAccumulator(ProfileAccumulator &&)                  = delete;
    ProfileAccumulator &operator=(ProfileAccumulator &&)       = delete;

  private:
    Stat                 &stat_;
    ProfClock::time_point start_;
};

#else // !CULVERIN_PROFILE_SYNC

// Counting/accumulating instrumentation compiles away entirely; the guards
// still exist so call sites stay scoped and readable.
class ProfileScope {
  public:
    explicit ProfileScope(const char *, unsigned int = 0) noexcept {}
};

class ProfileAccumulator {
  public:
    explicit ProfileAccumulator(Stat &) noexcept {}
};

#endif // CULVERIN_PROFILE_SYNC

} // namespace CPH
