#pragma once
#include "core/CompilerSpecifics.h"
#include "physics/InternalQuery.h"
#include "joltc.h"

// --- Query Filters ---
[[nodiscard]] [[maybe_unused]] static inline bool JPH_API_CALL
filter_allow_all_bp([[maybe_unused]] void *userData, [[maybe_unused]] JPH_BroadPhaseLayer layer) {
    return true; // Allow ray to see all broadphase regions
}
[[nodiscard]] [[maybe_unused]] static inline bool JPH_API_CALL
filter_allow_all_obj([[maybe_unused]] void *userData, [[maybe_unused]] JPH_ObjectLayer layer) {
    return true; // Allow ray to see all object layers (0 and 1)
}
[[nodiscard]] [[maybe_unused]] static inline bool JPH_API_CALL
filter_true_body([[maybe_unused]] void *userData, [[maybe_unused]] JPH_BodyID bodyID) {
    return true;
}
[[nodiscard]] [[maybe_unused]] static inline bool JPH_API_CALL
filter_true_shape([[maybe_unused]] void *userData, [[maybe_unused]] const JPH_Shape *shape,
                  [[maybe_unused]] const JPH_SubShapeID *id) {
    return true;
}

// 3. Define the GLOBAL procedure tables
[[maybe_unused]] static const JPH_BroadPhaseLayerFilter_Procs global_bp_procs = {
    .ShouldCollide = filter_allow_all_bp};

[[maybe_unused]] static const JPH_ObjectLayerFilter_Procs global_obj_procs = {
    .ShouldCollide = filter_allow_all_obj};

[[maybe_unused]] static const JPH_BodyFilter_Procs global_bf_procs  = {.ShouldCollide =
                                                                           UnifiedBodyFilter};
[[maybe_unused]] static const JPH_ShapeFilter_Procs global_sf_procs = {.ShouldCollide =
                                                                           filter_true_shape};
