#pragma once

// Culverin's Jolt trace/assert bridge, native side (Phase 2, step A3).
//
// Jolt exposes two process-wide function pointers for this:
//
//     JPH::Trace        = void (*)(const char *inFMT, ...)
//     JPH::AssertFailed = bool (*)(const char *expr, const char *msg,
//                                  const char *file, JPH::uint line)
//                         (only exists when JPH_ENABLE_ASSERTS is defined)
//
// Note the trace hook is *variadic*: the JoltC wrapper used to format for us,
// so the bridge must do its own vsnprintf before the pretty-printer runs.

#include "jolt/Jolt.hpp"


namespace CPH {

// Installs both hooks.
//
// ORDER MATTERS while JoltC is still linked: JPH_Init() assigns
// JPH::Trace = TraceImpl and JPH::AssertFailed = AssertFailedImpl itself, so
// calling this before JPH_Init() is silently undone (trace output would fall
// back to raw stdout). Call it after Jolt initialization. Step A5 deletes
// JPH_Init(), at which point this becomes the only initialization path.
void InstallJoltLogging() noexcept;

} // namespace CPH
