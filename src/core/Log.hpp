#pragma once

// The one debug logger.
//
// This used to exist twice: `CPH::Python::DebugLog` in
// src/python/bindings/PythonCommon.hpp and a byte-identical copy in
// src/vehicles/Vehicle.cpp. The copy was there because PythonCommon.hpp is
// binding glue and nothing under src/ may include it -- a rule worth keeping --
// so the native side had no shared logger to call and grew its own.
//
// Both were gated on CULVERIN_DEBUG and both wrote "[Culverin] " plus a newline
// to stderr. This header is that function, once, in a place both sides may
// include: everything depends on culverin_native_common, and nothing in here
// touches CPython.
//
// Gating: the flag is read here rather than at the call site, so a release
// build compiles the body away entirely -- every `DebugLog(...)` still
// evaluates its arguments, as it did before, but `if constexpr` drops the
// formatting, the va_list and the writes.

#include <cstdarg>
#include <cstdio>

namespace CPH {

#if defined(CULVERIN_DEBUG)
inline constexpr bool kDebugLogging = true;
#else
inline constexpr bool kDebugLogging = false;
#endif

inline void DebugLog(const char *format, ...) noexcept {
    if constexpr (kDebugLogging) {
        std::fputs("[Culverin] ", stderr);
        va_list args;
        va_start(args, format);
        std::vfprintf(stderr, format, args);
        va_end(args);
        std::fputc('\n', stderr);
    } else {
        (void)format;
    }
}

} // namespace CPH
