#pragma once

// Global allow-all query filters registered with Jolt at module init.
// Per-query filtering uses UnifiedBodyFilter (physics/InternalQuery.hpp).

#include "physics/InternalQuery.hpp"
#include <joltc.h>

namespace {

[[maybe_unused]] bool JPH_API_CALL
filter_allow_all_bp([[maybe_unused]] void *userData, [[maybe_unused]] JPH_BroadPhaseLayer layer) {
    return true;
}
[[maybe_unused]] bool JPH_API_CALL
filter_allow_all_obj([[maybe_unused]] void *userData, [[maybe_unused]] JPH_ObjectLayer layer) {
    return true;
}
[[maybe_unused]] bool JPH_API_CALL
filter_true_shape([[maybe_unused]] void *userData, [[maybe_unused]] const JPH_Shape *shape,
                  [[maybe_unused]] const JPH_SubShapeID *id) {
    return true;
}

} // namespace

inline const JPH_BroadPhaseLayerFilter_Procs kGlobalBpProcs = {
    .ShouldCollide = filter_allow_all_bp};
inline const JPH_ObjectLayerFilter_Procs kGlobalObjProcs = {
    .ShouldCollide = filter_allow_all_obj};
inline const JPH_BodyFilter_Procs kGlobalBfProcs  = {.ShouldCollide = UnifiedBodyFilter};
inline const JPH_ShapeFilter_Procs kGlobalSfProcs = {.ShouldCollide = filter_true_shape};
