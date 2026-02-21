#pragma once

#include "culverin_fast_parse.h"
#include "culverin_types.h"

/**
 * SCHEMA DEFINITIONS
 * Format: X(IndexName, PythonName, C-Type, IsRequired)
 */

#define SCHEMA_BODY(X)                                                         \
  X(IDX_POS, "pos", PyObject *, 0)                                             \
  X(IDX_ROT, "rot", PyObject *, 0)                                             \
  X(IDX_SIZE, "size", PyObject *, 0)                                           \
  X(IDX_SHAPE, "shape", int, 0)                                                \
  X(IDX_MOTION, "motion", int, 0)                                              \
  X(IDX_USER_DATA, "user_data", uint64_t, 0)                                   \
  X(IDX_SENSOR, "is_sensor", bool, 0)                                          \
  X(IDX_MASS, "mass", float, 0)                                                \
  X(IDX_CAT, "category", uint32_t, 0)                                          \
  X(IDX_MASK, "mask", uint32_t, 0)                                             \
  X(IDX_FRIC, "friction", float, 0)                                            \
  X(IDX_REST, "restitution", float, 0)                                         \
  X(IDX_MAT, "material_id", uint32_t, 0)                                       \
  X(IDX_CCD, "ccd", bool, 0)

#define SCHEMA_SET_POS(X)                                                      \
  X(IDX_SETPOS_HANDLE, "handle", BodyHandle, 1)                                \
  X(IDX_SETPOS_X, "x", JPH_Real, 1)                                            \
  X(IDX_SETPOS_Y, "y", JPH_Real, 1)                                            \
  X(IDX_SETPOS_Z, "z", JPH_Real, 1)

// Shared by Impulse, AngImpulse, Force, and Torque
#define SCHEMA_VEC3(X)                                                         \
  X(IDX_V3_H, "handle", BodyHandle, 1)                                         \
  X(IDX_V3_X, "x", float, 1)                                                   \
  X(IDX_V3_Y, "y", float, 1)                                                   \
  X(IDX_V3_Z, "z", float, 1)

#define SCHEMA_IMPULSE_AT(X)                                                   \
  X(IDX_IMPAT_H, "handle", BodyHandle, 1)                                      \
  X(IDX_IMPAT_IX, "ix", float, 1)                                              \
  X(IDX_IMPAT_IY, "iy", float, 1)                                              \
  X(IDX_IMPAT_IZ, "iz", float, 1)                                              \
  X(IDX_IMPAT_PX, "px", JPH_Real, 1)                                           \
  X(IDX_IMPAT_PY, "py", JPH_Real, 1)                                           \
  X(IDX_IMPAT_PZ, "pz", JPH_Real, 1)

#define SCHEMA_BUOYANCY(X)                                                     \
  X(IDX_BUOY_HANDLE, "handle", BodyHandle, 1)                                  \
  X(IDX_BUOY_SURFACE_Y, "surface_y", double, 1)                                \
  X(IDX_BUOY_BUOYANCY, "buoyancy", float, 0)                                   \
  X(IDX_BUOY_LIN_DRAG, "linear_drag", float, 0)                                \
  X(IDX_BUOY_ANG_DRAG, "angular_drag", float, 0)                               \
  X(IDX_BUOY_DT, "dt", float, 0)                                               \
  X(IDX_BUOY_VEL, "fluid_velocity", PyObject *, 0)

#define SCHEMA_HULL_OR_COMP(X)                                                 \
  X(IDX_HC_POS, "pos", PyObject *, 1)                                          \
  X(IDX_HC_ROT, "rot", PyObject *, 1)                                          \
  X(IDX_HC_DATA, "points_or_parts", PyObject *, 1)                             \
  X(IDX_HC_MOTION, "motion", int, 0)                                           \
  X(IDX_HC_MASS, "mass", float, 0)                                             \
  X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                                \
  X(IDX_HC_SENSOR, "is_sensor", bool, 0)                                       \
  X(IDX_HC_CAT, "category", uint32_t, 0)                                       \
  X(IDX_HC_MASK, "mask", uint32_t, 0)                                          \
  X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)                                 \
  X(IDX_HC_FRIC, "friction", float, 0)                                         \
  X(IDX_HC_REST, "restitution", float, 0)                                      \
  X(IDX_HC_CCD, "ccd", bool, 0)

#define SCHEMA_MESH(X)                                                         \
  X(IDX_MSH_POS, "pos", PyObject *, 1)                                         \
  X(IDX_MSH_ROT, "rot", PyObject *, 1)                                         \
  X(IDX_MSH_VERTS, "vertices", PyObject *, 1)                                  \
  X(IDX_MSH_INDICES, "indices", PyObject *, 1)                                 \
  X(IDX_MSH_USER_DATA, "user_data", uint64_t, 0)                               \
  X(IDX_MSH_CAT, "category", uint32_t, 0)                                      \
  X(IDX_MSH_MASK, "mask", uint32_t, 0)

#define SCHEMA_HANDLE_ONLY(X) X(IDX_H_H, "handle", BodyHandle, 1)

#define SCHEMA_SET_TRNS(X)                                                     \
  X(IDX_ST_HANDLE, "handle", BodyHandle, 1)                                    \
  X(IDX_ST_POS, "pos", PyObject *, 1)                                          \
  X(IDX_ST_ROT, "rot", PyObject *, 1)

#define SCHEMA_CCD(X)                                                          \
  X(IDX_CCD_H, "handle", BodyHandle, 1)                                        \
  X(IDX_CCD_E, "enabled", bool, 1)

#define SCHEMA_XYZ(X)                                                          \
  X(IDX_XYZ_X, "x", float, 1)                                                  \
  X(IDX_XYZ_Y, "y", float, 1)                                                  \
  X(IDX_XYZ_Z, "z", float, 1)

#define SCHEMA_BATCH_BUOYANCY(X)                                               \
  X(IDX_BBUOY_HANDLES, "handles", PyObject *, 1)                               \
  X(IDX_BBUOY_SURFACE_Y, "surface_y", double, 1)                               \
  X(IDX_BBUOY_BUOYANCY, "buoyancy", float, 0)                                  \
  X(IDX_BBUOY_LIN_DRAG, "linear_drag", float, 0)                               \
  X(IDX_BBUOY_ANG_DRAG, "angular_drag", float, 0)                              \
  X(IDX_BBUOY_DT, "dt", float, 0)                                              \
  X(IDX_BBUOY_VEL, "fluid_velocity", PyObject *, 0)

// This is the "Master" structural schema for Hull/Compound/Body-like objects
#define SCHEMA_HC_MASTER(X)                                                    \
  X(IDX_HC_POS, "pos", PyObject *, 1)                                          \
  X(IDX_HC_ROT, "rot", PyObject *, 1)                                          \
  X(IDX_HC_DATA, "data", PyObject *, 1)                                        \
  X(IDX_HC_MOTION, "motion", int, 0)                                           \
  X(IDX_HC_MASS, "mass", float, 0)                                             \
  X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                                \
  X(IDX_HC_SENSOR, "is_sensor", bool, 0)                                       \
  X(IDX_HC_CAT, "category", uint32_t, 0)                                       \
  X(IDX_HC_MASK, "mask", uint32_t, 0)                                          \
  X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)                                 \
  X(IDX_HC_FRIC, "friction", float, 0)                                         \
  X(IDX_HC_REST, "restitution", float, 0)                                      \
  X(IDX_HC_CCD, "ccd", bool, 0)

// These are specific overlays just to change the Python keyword strings
#define SCHEMA_HC_HULL(X)                                                      \
  X(IDX_HC_POS, "pos", PyObject *, 1)                                          \
  X(IDX_HC_ROT, "rot", PyObject *, 1)                                          \
      X(IDX_HC_DATA, "points", PyObject *, 1) /* Change name to "points" */    \
      X(IDX_HC_MOTION, "motion", int, 0) X(IDX_HC_MASS, "mass", float, 0)      \
          X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                        \
              X(IDX_HC_SENSOR, "is_sensor", bool, 0)                           \
                  X(IDX_HC_CAT, "category", uint32_t, 0)                       \
                      X(IDX_HC_MASK, "mask", uint32_t, 0)                      \
                          X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)         \
                              X(IDX_HC_FRIC, "friction", float, 0)             \
                                  X(IDX_HC_REST, "restitution", float, 0)      \
                                      X(IDX_HC_CCD, "ccd", bool, 0)

#define SCHEMA_HC_COMP(X)                                                      \
  X(IDX_HC_POS, "pos", PyObject *, 1)                                          \
  X(IDX_HC_ROT, "rot", PyObject *, 1)                                          \
      X(IDX_HC_DATA, "parts", PyObject *, 1) /* Change name to "parts" */      \
      X(IDX_HC_MOTION, "motion", int, 0) X(IDX_HC_MASS, "mass", float, 0)      \
          X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                        \
              X(IDX_HC_SENSOR, "is_sensor", bool, 0)                           \
                  X(IDX_HC_CAT, "category", uint32_t, 0)                       \
                      X(IDX_HC_MASK, "mask", uint32_t, 0)                      \
                          X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)         \
                              X(IDX_HC_FRIC, "friction", float, 0)             \
                                  X(IDX_HC_REST, "restitution", float, 0)      \
                                      X(IDX_HC_CCD, "ccd", bool, 0)

#define SCHEMA_BATCH_CREATE(X)                                                 \
  X(IDX_BC_POSITIONS, "positions", PyObject *, 1)                              \
  X(IDX_BC_SIZES, "sizes", PyObject *, 1)                                      \
  X(IDX_BC_SHAPE, "shape_type", int, 0)                                        \
  X(IDX_BC_MOTION, "motion_type", int, 0)

#define SCHEMA_BATCH_DESTROY(X) X(IDX_BD_HANDLES, "handles", PyObject *, 1)

#define SCHEMA_SET_ROT(X)                                                      \
  X(IDX_SETROT_H, "handle", BodyHandle, 1)                                     \
  X(IDX_SETROT_X, "x", float, 1)                                               \
  X(IDX_SETROT_Y, "y", float, 1)                                               \
  X(IDX_SETROT_Z, "z", float, 1)                                               \
  X(IDX_SETROT_W, "w", float, 1)

#define SCHEMA_RENDER(X) \
    X(IDX_RND_ALPHA, "alpha", float, 1)

#define SCHEMA_RAYCAST(X) \
    X(IDX_RAY_START, "start",     PyObject*, 1) \
    X(IDX_RAY_DIR,   "direction", PyObject*, 1) \
    X(IDX_RAY_DIST,  "max_dist",  float,     0) \
    X(IDX_RAY_IGN,   "ignore",    BodyHandle,0)

#define SCHEMA_RAYCAST_BATCH(X) \
    X(IDX_RB_STARTS, "starts",     PyObject*, 1) \
    X(IDX_RB_DIRS,   "directions", PyObject*, 1) \
    X(IDX_RB_DIST,   "max_dist",   float,     0)

#define SCHEMA_SHAPECAST(X) \
    X(IDX_SC_SHAPE,  "shape",  int,        1) \
    X(IDX_SC_POS,    "pos",    PyObject*,  1) \
    X(IDX_SC_ROT,    "rot",    PyObject*,  1) \
    X(IDX_SC_DIR,    "dir",    PyObject*,  1) \
    X(IDX_SC_SIZE,   "size",   PyObject*,  0) \
    X(IDX_SC_IGNORE, "ignore", BodyHandle, 0)

/** --- THE GENERATOR ENGINE --- **/

#define GEN_ENUM(ID, NAME, TYPE, REQ) ID,

// Defines the Enum and the Count for a signature type
#define DEFINE_INDEX_GROUP(GroupName, Schema)                                  \
  typedef enum { Schema(GEN_ENUM) GroupName##_COUNT } GroupName##_Idx;

// Declares a specific parser that uses an Index Group
#define DECLARE_PARSER(ParserName, GroupName)                                  \
  extern FastParser ParserName##Parser;                                        \
  extern FastArgSpec ParserName##Specs[GroupName##_COUNT];

// A. Define the Index Groups (One per unique signature)
DEFINE_INDEX_GROUP(Body, SCHEMA_BODY)
DEFINE_INDEX_GROUP(Vec3, SCHEMA_VEC3)
DEFINE_INDEX_GROUP(ImpAt, SCHEMA_IMPULSE_AT)
DEFINE_INDEX_GROUP(HOnly, SCHEMA_HANDLE_ONLY)
DEFINE_INDEX_GROUP(XYZ, SCHEMA_XYZ)
DEFINE_INDEX_GROUP(SetPos, SCHEMA_SET_POS)
DEFINE_INDEX_GROUP(Buoy, SCHEMA_BUOYANCY)
DEFINE_INDEX_GROUP(BatchBuoy, SCHEMA_BATCH_BUOYANCY)
DEFINE_INDEX_GROUP(Mesh, SCHEMA_MESH)
DEFINE_INDEX_GROUP(SetTrns, SCHEMA_SET_TRNS)
DEFINE_INDEX_GROUP(CCD, SCHEMA_CCD)
DEFINE_INDEX_GROUP(HC, SCHEMA_HC_MASTER) // HC group defined ONLY once
DEFINE_INDEX_GROUP(BatchCreate, SCHEMA_BATCH_CREATE)
DEFINE_INDEX_GROUP(BatchDestroy, SCHEMA_BATCH_DESTROY)
DEFINE_INDEX_GROUP(SetRot, SCHEMA_SET_ROT)
DEFINE_INDEX_GROUP(Render, SCHEMA_RENDER)
DEFINE_INDEX_GROUP(Raycast, SCHEMA_RAYCAST)
DEFINE_INDEX_GROUP(RayBatch, SCHEMA_RAYCAST_BATCH)
DEFINE_INDEX_GROUP(Shapecast, SCHEMA_SHAPECAST)

// B. Declare the Parsers
DECLARE_PARSER(Body, Body)
DECLARE_PARSER(Impulse, Vec3)
DECLARE_PARSER(AngImpulse, Vec3)
DECLARE_PARSER(Force, Vec3)
DECLARE_PARSER(Torque, Vec3)
DECLARE_PARSER(SetLinVel, Vec3)
DECLARE_PARSER(SetAngVel, Vec3)
DECLARE_PARSER(ImpulseAt, ImpAt)
DECLARE_PARSER(HOnly, HOnly)
DECLARE_PARSER(Destroy, HOnly)
DECLARE_PARSER(Activate, HOnly)
DECLARE_PARSER(Gravity, XYZ)
DECLARE_PARSER(SetPos, SetPos)
DECLARE_PARSER(Buoy, Buoy)
DECLARE_PARSER(BatchBuoy, BatchBuoy)
DECLARE_PARSER(Mesh, Mesh)
DECLARE_PARSER(SetTrns, SetTrns)
DECLARE_PARSER(CCD, CCD)
DECLARE_PARSER(ConvexHull, HC)
DECLARE_PARSER(Compound, HC)
DECLARE_PARSER(BatchCreate, BatchCreate)
DECLARE_PARSER(BatchDestroy, BatchDestroy)
DECLARE_PARSER(SetRot, SetRot)
DECLARE_PARSER(Render, Render)
DECLARE_PARSER(Raycast, Raycast)
DECLARE_PARSER(RayBatch, RayBatch)
DECLARE_PARSER(Shapecast, Shapecast)

void culverin_init_all_parsers(void);