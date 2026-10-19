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
#if JPH_OBJECT_LAYER_BITS != 32
#error "Culverin requires JPH_OBJECT_LAYER_BITS=32 (ObjectLayer is uint32 in handles/settings)."
#endif

#include <Jolt/Jolt.h>

// Foundational ID/layer types referenced by the ABI checks below (and by
// essentially every physics TU).
#include <Jolt/Physics/Body/BodyID.h>
#include <Jolt/Physics/Collision/ObjectLayer.h>

#include <type_traits>

namespace culverin::jolt_abi {

// Double-precision Jolt: Real is double, RVec3 is the 32-byte DVec3.
// ShadowSync streams JPH::Body positions directly into CPH_PosStride
// (4 doubles) and the Python buffer protocol exposes float64 stride-4.
static_assert(std::is_same_v<JPH::Real, double>);
static_assert(sizeof(JPH::RVec3) == 32);
static_assert(sizeof(JPH::ObjectLayer) == 4);
// BodyBuffers store body IDs in a uint32 SoA array (JPH_BodyID).
static_assert(sizeof(JPH::BodyID) == 4);

} // namespace culverin::jolt_abi
