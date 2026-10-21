#pragma once

// Single entry point for the native Jolt C++ API (Phase 2, step A0 —
// see docs/CPP_INTERFACE_PLAN.md). Jolt requires <Jolt/Jolt.h> to
// precede every other Jolt header; including this header first
// guarantees that and pins the ABI configuration Culverin is built
// against.
//
// The configuration itself is owned by the Jolt CMake options
// (DOUBLE_PRECISION, OBJECT_LAYER_BITS, DEBUG_RENDERER_IN_DEBUG_AND_RELEASE),
// which Jolt exports as PUBLIC compile definitions on the Jolt::Jolt
// target; every Culverin target reaches them through
// culverin_native_support. The checks below turn any drift between
// that configuration and what Culverin expects into a compile error
// here instead of silent ODR/layout corruption across the boundary.

#ifndef JPH_DOUBLE_PRECISION
#error "Culverin requires JPH_DOUBLE_PRECISION (positions are float64 on the wire). Check the DOUBLE_PRECISION Jolt option."
#endif

// Debug drawing is a shipped feature (world.build_debug_data), not a
// debug-build convenience, so JPH::DebugRenderer must exist in every
// configuration. Checked here so the failure names the CMake option instead
// of surfacing as DebugRendererSimple.h's own #error.
#ifndef JPH_DEBUG_RENDERER
#error "Culverin requires JPH_DEBUG_RENDERER (debug draw is part of the Python API). Check the DEBUG_RENDERER_IN_DEBUG_AND_RELEASE Jolt option."
#endif
#if JPH_OBJECT_LAYER_BITS != 32
#error "Culverin requires JPH_OBJECT_LAYER_BITS=32 (ObjectLayer is uint32 in handles/settings)."
#endif

// --- MSVC ABI prelude: pin the narrow-ctype declaration set ---------------
// MSVC's <cctype> is nothing but `using _CSTD isalnum; ...` — it projects the
// C library's character names out of the global namespace and does not declare
// them itself. UCRT's <ctype.h> declares them under `#if !defined __midl &&
// !defined RC_INVOKED`, behind a guard that is *only* `#pragma once`, so the
// first inclusion of that header decides the rest of the translation unit:
// include it once where the narrow block is skipped and every later inclusion
// is a no-op, leaving <cctype> to fail with "no member named 'isalnum' in the
// global namespace" while the wide names survive — <corecrt_wctype.h> is
// included ahead of the guard, so iswalnum & co. are always visible.
//
// Jolt's Core.h reaches <cctype> through <sstream>, deep inside its own
// warning-suppression and namespace machinery; Culverin cannot reorder that.
// What it can do is make the first inclusion happen at the one place every
// translation unit is guaranteed to pass through with nothing open, above.
#if defined(_MSC_VER)
#    include <cctype>
#endif

#include <Jolt/Jolt.h>

// Foundational ID/layer types referenced by the ABI checks below (and by
// essentially every physics TU).
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <type_traits>

// Double-precision Jolt: Real is double, RVec3 is the 32-byte DVec3.
// ShadowSync streams JPH::Body positions directly into PosStride
// (4 doubles) and the Python buffer protocol exposes float64 stride-4.
static_assert(std::is_same_v<JPH::Real, double>);
static_assert(sizeof(JPH::RVec3) == 32);
static_assert(sizeof(JPH::ObjectLayer) == 4);
// BodyBuffers store body IDs directly as JPH::BodyID, a standard-layout
// single-uint32 wrapper, so the SoA array stays 4 bytes per body.
static_assert(sizeof(JPH::BodyID) == 4);
static_assert(std::is_standard_layout_v<JPH::BodyID>);
