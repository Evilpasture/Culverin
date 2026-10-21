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
// This is now the only path that installs them. (Historically it had to run
// after JPH_Init(), which assigned Jolt's defaults and would silently undo an
// earlier call; A5 removed JoltC and with it JPH_Init(), and the inlined
// RegisterTypes sequence in Module.cpp does not touch either hook.)
void InstallJoltLogging() noexcept;

} // namespace CPH
