#pragma once
#include "culverin_compiler_specifics.h"
#include "culverin_internal_query.h"
#include "joltc.h"

// --- Query Filters ---
CULV_MAYBE_UNUSED static inline bool JPH_API_CALL filter_allow_all_bp(CULV_MAYBE_UNUSED void *userData,
                                                    CULV_MAYBE_UNUSED JPH_BroadPhaseLayer layer) {
    return true; // Allow ray to see all broadphase regions
}
CULV_MAYBE_UNUSED static inline bool JPH_API_CALL filter_allow_all_obj(CULV_MAYBE_UNUSED void *userData,
                                                     CULV_MAYBE_UNUSED JPH_ObjectLayer layer) {
    return true; // Allow ray to see all object layers (0 and 1)
}

CULV_MAYBE_UNUSED static inline bool JPH_API_CALL filter_true_body(CULV_MAYBE_UNUSED void *userData,
                                                 CULV_MAYBE_UNUSED JPH_BodyID bodyID) {
    return true;
}
CULV_MAYBE_UNUSED static inline bool JPH_API_CALL filter_true_shape(CULV_MAYBE_UNUSED void *userData,
                                                  CULV_MAYBE_UNUSED const JPH_Shape *shape,
                                                  CULV_MAYBE_UNUSED const JPH_SubShapeID *id) {
    return true;
}

// 3. Define the GLOBAL procedure tables
CULV_MAYBE_UNUSED static const JPH_BroadPhaseLayerFilter_Procs global_bp_procs = {.ShouldCollide =
                                                                    filter_allow_all_bp};

CULV_MAYBE_UNUSED static const JPH_ObjectLayerFilter_Procs global_obj_procs = {.ShouldCollide = filter_allow_all_obj};

CULV_MAYBE_UNUSED static const JPH_BodyFilter_Procs global_bf_procs  = {.ShouldCollide = UnifiedBodyFilter};
CULV_MAYBE_UNUSED static const JPH_ShapeFilter_Procs global_sf_procs = {.ShouldCollide = filter_true_shape};
