#pragma once
#include "joltc.h"

// --- Query Filters ---
static inline bool JPH_API_CALL filter_allow_all_bp(void *userData,
                                                    JPH_BroadPhaseLayer layer) {
  return true; // Allow ray to see all broadphase regions
}
static inline bool JPH_API_CALL filter_allow_all_obj(void *userData,
                                                     JPH_ObjectLayer layer) {
  return true; // Allow ray to see all object layers (0 and 1)
}

static inline bool JPH_API_CALL filter_true_body(void *userData,
                                                 JPH_BodyID bodyID) {
  return true;
}
static inline bool JPH_API_CALL filter_true_shape(void *userData,
                                                  const JPH_Shape *shape,
                                                  const JPH_SubShapeID *id) {
  return true;
}

static const JPH_BodyFilter_Procs global_bf_procs = {.ShouldCollide =
                                                         filter_true_body};
static const JPH_ShapeFilter_Procs global_sf_procs = {.ShouldCollide =
                                                          filter_true_shape};