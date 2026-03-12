#pragma once

#include "culverin_fast_parse.h"
#include "culverin_types.h"

/**
 * SCHEMA DEFINITIONS
 * Format: X(IndexName, PythonName, C-Type, IsRequired)
 */

#define SCHEMA_BODY(X)                                                                             \
    X(IDX_POS, "pos", PyObject *, 0)                                                               \
    X(IDX_ROT, "rot", PyObject *, 0)                                                               \
    X(IDX_SIZE, "size", PyObject *, 0)                                                             \
    X(IDX_SHAPE, "shape", int, 0)                                                                  \
    X(IDX_MOTION, "motion", int, 0)                                                                \
    X(IDX_USER_DATA, "user_data", uint64_t, 0)                                                     \
    X(IDX_SENSOR, "is_sensor", bool, 0)                                                            \
    X(IDX_MASS, "mass", float, 0)                                                                  \
    X(IDX_CAT, "category", uint32_t, 0)                                                            \
    X(IDX_MASK, "mask", uint32_t, 0)                                                               \
    X(IDX_FRIC, "friction", float, 0)                                                              \
    X(IDX_REST, "restitution", float, 0)                                                           \
    X(IDX_MAT, "material_id", uint32_t, 0)                                                         \
    X(IDX_CCD, "ccd", bool, 0)

#define SCHEMA_SET_POS(X)                                                                          \
    X(IDX_SETPOS_HANDLE, "handle", BodyHandle, 1)                                                  \
    X(IDX_SETPOS_X, "x", JPH_Real, 1)                                                              \
    X(IDX_SETPOS_Y, "y", JPH_Real, 1)                                                              \
    X(IDX_SETPOS_Z, "z", JPH_Real, 1)

// Shared by Impulse, AngImpulse, Force, and Torque
#define SCHEMA_VEC3(X)                                                                             \
    X(IDX_V3_H, "handle", BodyHandle, 1)                                                           \
    X(IDX_V3_X, "x", float, 1)                                                                     \
    X(IDX_V3_Y, "y", float, 1)                                                                     \
    X(IDX_V3_Z, "z", float, 1)

#define SCHEMA_IMPULSE_AT(X)                                                                       \
    X(IDX_IMPAT_H, "handle", BodyHandle, 1)                                                        \
    X(IDX_IMPAT_IX, "ix", float, 1)                                                                \
    X(IDX_IMPAT_IY, "iy", float, 1)                                                                \
    X(IDX_IMPAT_IZ, "iz", float, 1)                                                                \
    X(IDX_IMPAT_PX, "px", JPH_Real, 1)                                                             \
    X(IDX_IMPAT_PY, "py", JPH_Real, 1)                                                             \
    X(IDX_IMPAT_PZ, "pz", JPH_Real, 1)

#define SCHEMA_BUOYANCY(X)                                                                         \
    X(IDX_BUOY_HANDLE, "handle", BodyHandle, 1)                                                    \
    X(IDX_BUOY_SURFACE_Y, "surface_y", double, 1)                                                  \
    X(IDX_BUOY_BUOYANCY, "buoyancy", float, 0)                                                     \
    X(IDX_BUOY_LIN_DRAG, "linear_drag", float, 0)                                                  \
    X(IDX_BUOY_ANG_DRAG, "angular_drag", float, 0)                                                 \
    X(IDX_BUOY_DT, "dt", float, 0)                                                                 \
    X(IDX_BUOY_VEL, "fluid_velocity", PyObject *, 0)

#define SCHEMA_HULL_OR_COMP(X)                                                                     \
    X(IDX_HC_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_HC_ROT, "rot", PyObject *, 1)                                                            \
    X(IDX_HC_DATA, "points_or_parts", PyObject *, 1)                                               \
    X(IDX_HC_MOTION, "motion", int, 0)                                                             \
    X(IDX_HC_MASS, "mass", float, 0)                                                               \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                                                  \
    X(IDX_HC_SENSOR, "is_sensor", bool, 0)                                                         \
    X(IDX_HC_CAT, "category", uint32_t, 0)                                                         \
    X(IDX_HC_MASK, "mask", uint32_t, 0)                                                            \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)                                                   \
    X(IDX_HC_FRIC, "friction", float, 0)                                                           \
    X(IDX_HC_REST, "restitution", float, 0)                                                        \
    X(IDX_HC_CCD, "ccd", bool, 0)

#define SCHEMA_MESH(X)                                                                             \
    X(IDX_MSH_POS, "pos", PyObject *, 1)                                                           \
    X(IDX_MSH_ROT, "rot", PyObject *, 1)                                                           \
    X(IDX_MSH_VERTS, "vertices", PyObject *, 1)                                                    \
    X(IDX_MSH_INDICES, "indices", PyObject *, 1)                                                   \
    X(IDX_MSH_USER_DATA, "user_data", uint64_t, 0)                                                 \
    X(IDX_MSH_CAT, "category", uint32_t, 0)                                                        \
    X(IDX_MSH_MASK, "mask", uint32_t, 0)

#define SCHEMA_HANDLE_ONLY(X) X(IDX_H_H, "handle", BodyHandle, 1)

#define SCHEMA_SET_TRNS(X)                                                                         \
    X(IDX_ST_HANDLE, "handle", BodyHandle, 1)                                                      \
    X(IDX_ST_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_ST_ROT, "rot", PyObject *, 1)

#define SCHEMA_CCD(X)                                                                              \
    X(IDX_CCD_H, "handle", BodyHandle, 1)                                                          \
    X(IDX_CCD_E, "enabled", bool, 1)

#define SCHEMA_XYZ(X)                                                                              \
    X(IDX_XYZ_X, "x", float, 1)                                                                    \
    X(IDX_XYZ_Y, "y", float, 1)                                                                    \
    X(IDX_XYZ_Z, "z", float, 1)

#define SCHEMA_BATCH_BUOYANCY(X)                                                                   \
    X(IDX_BBUOY_HANDLES, "handles", PyObject *, 1)                                                 \
    X(IDX_BBUOY_SURFACE_Y, "surface_y", double, 1)                                                 \
    X(IDX_BBUOY_BUOYANCY, "buoyancy", float, 0)                                                    \
    X(IDX_BBUOY_LIN_DRAG, "linear_drag", float, 0)                                                 \
    X(IDX_BBUOY_ANG_DRAG, "angular_drag", float, 0)                                                \
    X(IDX_BBUOY_DT, "dt", float, 0)                                                                \
    X(IDX_BBUOY_VEL, "fluid_velocity", PyObject *, 0)

// This is the "Master" structural schema for Hull/Compound/Body-like objects
#define SCHEMA_HC_MASTER(X)                                                                        \
    X(IDX_HC_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_HC_ROT, "rot", PyObject *, 1)                                                            \
    X(IDX_HC_DATA, "data", PyObject *, 1)                                                          \
    X(IDX_HC_MOTION, "motion", int, 0)                                                             \
    X(IDX_HC_MASS, "mass", float, 0)                                                               \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                                                  \
    X(IDX_HC_SENSOR, "is_sensor", bool, 0)                                                         \
    X(IDX_HC_CAT, "category", uint32_t, 0)                                                         \
    X(IDX_HC_MASK, "mask", uint32_t, 0)                                                            \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)                                                   \
    X(IDX_HC_FRIC, "friction", float, 0)                                                           \
    X(IDX_HC_REST, "restitution", float, 0)                                                        \
    X(IDX_HC_CCD, "ccd", bool, 0)

// These are specific overlays just to change the Python keyword strings
#define SCHEMA_HC_HULL(X)                                                                          \
    X(IDX_HC_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_HC_ROT, "rot", PyObject *, 1)                                                            \
    X(IDX_HC_DATA, "points", PyObject *, 1) /* Change name to "points" */                          \
    X(IDX_HC_MOTION, "motion", int, 0)                                                             \
    X(IDX_HC_MASS, "mass", float, 0)                                                               \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                                                  \
    X(IDX_HC_SENSOR, "is_sensor", bool, 0)                                                         \
    X(IDX_HC_CAT, "category", uint32_t, 0)                                                         \
    X(IDX_HC_MASK, "mask", uint32_t, 0)                                                            \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)                                                   \
    X(IDX_HC_FRIC, "friction", float, 0)                                                           \
    X(IDX_HC_REST, "restitution", float, 0)                                                        \
    X(IDX_HC_CCD, "ccd", bool, 0)

#define SCHEMA_HC_COMP(X)                                                                          \
    X(IDX_HC_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_HC_ROT, "rot", PyObject *, 1)                                                            \
    X(IDX_HC_DATA, "parts", PyObject *, 1) /* Change name to "parts" */                            \
    X(IDX_HC_MOTION, "motion", int, 0)                                                             \
    X(IDX_HC_MASS, "mass", float, 0)                                                               \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, 0)                                                  \
    X(IDX_HC_SENSOR, "is_sensor", bool, 0)                                                         \
    X(IDX_HC_CAT, "category", uint32_t, 0)                                                         \
    X(IDX_HC_MASK, "mask", uint32_t, 0)                                                            \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, 0)                                                   \
    X(IDX_HC_FRIC, "friction", float, 0)                                                           \
    X(IDX_HC_REST, "restitution", float, 0)                                                        \
    X(IDX_HC_CCD, "ccd", bool, 0)

#define SCHEMA_BATCH_CREATE(X)                                                                     \
    X(IDX_BC_POSITIONS, "positions", PyObject *, 1)                                                \
    X(IDX_BC_SIZES, "sizes", PyObject *, 1)                                                        \
    X(IDX_BC_SHAPE, "shape_type", int, 0)                                                          \
    X(IDX_BC_MOTION, "motion_type", int, 0)

#define SCHEMA_BATCH_DESTROY(X) X(IDX_BD_HANDLES, "handles", PyObject *, 1)

#define SCHEMA_SET_ROT(X)                                                                          \
    X(IDX_SETROT_H, "handle", BodyHandle, 1)                                                       \
    X(IDX_SETROT_X, "x", float, 1)                                                                 \
    X(IDX_SETROT_Y, "y", float, 1)                                                                 \
    X(IDX_SETROT_Z, "z", float, 1)                                                                 \
    X(IDX_SETROT_W, "w", float, 1)

#define SCHEMA_RENDER(X) X(IDX_RND_ALPHA, "alpha", float, 1)

#define SCHEMA_RAYCAST(X)                                                                          \
    X(IDX_RAY_START, "start", PyObject *, 1)                                                       \
    X(IDX_RAY_DIR, "direction", PyObject *, 1)                                                     \
    X(IDX_RAY_DIST, "max_dist", float, 0)                                                          \
    X(IDX_RAY_IGN, "ignore", BodyHandle, 0)

#define SCHEMA_RAYCAST_BATCH(X)                                                                    \
    X(IDX_RB_STARTS, "starts", PyObject *, 1)                                                      \
    X(IDX_RB_DIRS, "directions", PyObject *, 1)                                                    \
    X(IDX_RB_DIST, "max_dist", float, 0)

#define SCHEMA_SHAPECAST(X)                                                                        \
    X(IDX_SC_SHAPE, "shape", int, 1)                                                               \
    X(IDX_SC_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_SC_ROT, "rot", PyObject *, 1)                                                            \
    X(IDX_SC_DIR, "dir", PyObject *, 1)                                                            \
    X(IDX_SC_SIZE, "size", PyObject *, 0)                                                          \
    X(IDX_SC_IGNORE, "ignore", BodyHandle, 0)

#define SCHEMA_OVERLAP_SPHERE(X)                                                                   \
    X(IDX_OS_CENTER, "center", PyObject *, 1)                                                      \
    X(IDX_OS_RADIUS, "radius", float, 1)

#define SCHEMA_OVERLAP_AABB(X)                                                                     \
    X(IDX_OA_MIN, "min", PyObject *, 1)                                                            \
    X(IDX_OA_MAX, "max", PyObject *, 1)

#define SCHEMA_SET_USER_DATA(X)                                                                    \
    X(IDX_SUD_H, "handle", BodyHandle, 1)                                                          \
    X(IDX_SUD_D, "data", uint64_t, 1)

#define SCHEMA_SET_MOTION(X)                                                                       \
    X(IDX_SM_H, "handle", BodyHandle, 1)                                                           \
    X(IDX_SM_M, "motion", int, 1)

#define SCHEMA_COL_FILTER(X)                                                                       \
    X(IDX_CF_H, "handle", BodyHandle, 1)                                                           \
    X(IDX_CF_C, "category", uint32_t, 1)                                                           \
    X(IDX_CF_M, "mask", uint32_t, 1)

#define SCHEMA_REG_MAT(X)                                                                          \
    X(IDX_RM_ID, "id", uint32_t, 1)                                                                \
    X(IDX_RM_FRIC, "friction", float, 0)                                                           \
    X(IDX_RM_REST, "restitution", float, 0)

#define SCHEMA_SET_CONSTR_TARGET(X)                                                                \
    X(IDX_SCT_H, "handle", uint64_t, 1)                                                            \
    X(IDX_SCT_T, "target", float, 1)

#define SCHEMA_HEIGHTFIELD(X)                                                                      \
    X(IDX_HF_POS, "pos", PyObject *, 1)                                                            \
    X(IDX_HF_ROT, "rot", PyObject *, 1)                                                            \
    X(IDX_HF_SCALE, "scale", PyObject *, 1)                                                        \
    X(IDX_HF_HEIGHTS, "heights", PyObject *, 1)                                                    \
    X(IDX_HF_GRID_SIZE, "grid_size", int, 1)                                                       \
    X(IDX_HF_USER_DATA, "user_data", uint64_t, 0)                                                  \
    X(IDX_HF_CAT, "category", uint32_t, 0)                                                         \
    X(IDX_HF_MASK, "mask", uint32_t, 0)                                                            \
    X(IDX_HF_MAT_ID, "material_id", uint32_t, 0)                                                   \
    X(IDX_HF_FRIC, "friction", float, 0)                                                           \
    X(IDX_HF_REST, "restitution", float, 0)

#define SCHEMA_DEBUG_DATA(X)                                                                       \
    X(IDX_DD_SHAPES, "shapes", bool, 0)                                                            \
    X(IDX_DD_CONSTRAINTS, "constraints", bool, 0)                                                  \
    X(IDX_DD_BBOX, "bbox", bool, 0)                                                                \
    X(IDX_DD_CENTERS, "centers", bool, 0)                                                          \
    X(IDX_DD_WIREFRAME, "wireframe", bool, 0)

#define SCHEMA_CREATE_CONSTR(X)                                                                    \
    X(IDX_CC_TYPE, "type", int, 1)                                                                 \
    X(IDX_CC_BODY1, "body1", BodyHandle, 1)                                                        \
    X(IDX_CC_BODY2, "body2", BodyHandle, 1)                                                        \
    X(IDX_CC_PARAMS, "params", PyObject *, 0)                                                      \
    X(IDX_CC_MOTOR, "motor", PyObject *, 0)

#define SCHEMA_STEP(X) X(IDX_STEP_DT, "dt", float, 0) // Optional (0)

#define SCHEMA_CHAR_MOVE(X)                                                                        \
    X(IDX_CM_VEL, "velocity", Vec3f, 1)                                                            \
    X(IDX_CM_DT, "dt", float, 1)

#define SCHEMA_LOAD_STATE(X) X(IDX_LS_STATE, "state", PyObject *, 1)

#define SCHEMA_CREATE_CHAR(X)                                                                      \
    X(IDX_CCHAR_POS, "pos", PosStride, 1)                                                          \
    X(IDX_CCHAR_H, "height", float, 0)                                                             \
    X(IDX_CCHAR_R, "radius", float, 0)                                                             \
    X(IDX_CCHAR_STEP, "step_height", float, 0)                                                     \
    X(IDX_CCHAR_SLOPE, "max_slope", float, 0)

#define SCHEMA_SET_POS_CHAR(X) X(IDX_SPC_POS, "pos", PosStride, 1)

#define SCHEMA_SET_ROT_CHAR(X) X(IDX_SRC_ROT, "rot", AuxStride, 1)

#define SCHEMA_SET_STRENGTH_CHAR(X) X(IDX_SSC_STRENGTH, "strength", float, 1)

#define SCHEMA_VEHICLE_INPUT(X)                                                                    \
    X(IDX_VI_FWD, "forward", float, 0)                                                             \
    X(IDX_VI_RIGHT, "right", float, 0)                                                             \
    X(IDX_VI_BRAKE, "brake", float, 0)                                                             \
    X(IDX_VI_HAND, "handbrake", float, 0)

#define SCHEMA_WHEEL_IDX(X) X(IDX_WH_INDEX, "index", uint32_t, 1)

#define SCHEMA_TANK_INPUT(X)                                                                       \
    X(IDX_TI_LEFT, "left", float, 1)                                                               \
    X(IDX_TI_RIGHT, "right", float, 1)                                                             \
    X(IDX_TI_BRAKE, "brake", float, 0)

#define SCHEMA_CREATE_VEHICLE(X)                                                                   \
    X(IDX_CV_CHASSIS, "chassis", uint64_t, 1)                                                      \
    X(IDX_CV_WHEELS, "wheels", PyObject *, 1)                                                      \
    X(IDX_CV_DRIVE, "drive", PyObject *, 0)                                                        \
    X(IDX_CV_ENGINE, "engine", PyObject *, 0)                                                      \
    X(IDX_CV_TRANS, "transmission", PyObject *, 0)

#define SCHEMA_CREATE_TRACKED(X)                                                                   \
    X(IDX_CT_CHASSIS, "chassis", uint64_t, 1)                                                      \
    X(IDX_CT_WHEELS, "wheels", PyObject *, 1)                                                      \
    X(IDX_CT_TRACKS, "tracks", PyObject *, 1)                                                      \
    X(IDX_CT_TORQUE, "max_torque", float, 0)                                                       \
    X(IDX_CT_RPM, "max_rpm", float, 0)

#define SCHEMA_CREATE_RAGDOLL(X)                                                                   \
    X(IDX_CR_SETTINGS, "settings", PyObject *, 1)                                                  \
    X(IDX_CR_POS, "pos", PosStride, 1)                                                             \
    X(IDX_CR_ROT, "rot", AuxStride, 0)                                                             \
    X(IDX_CR_USER, "user_data", uint64_t, 0)                                                       \
    X(IDX_CR_CAT, "category", uint32_t, 0)                                                         \
    X(IDX_CR_MASK, "mask", uint32_t, 0)                                                            \
    X(IDX_CR_MAT, "material_id", uint32_t, 0)

#define SCHEMA_RAGDOLL_SETTINGS(X) X(IDX_RS_SKELETON, "skeleton", PyObject *, 1)

#define SCHEMA_RAGDOLL_ADD_PART(X)                                                                 \
    X(IDX_RAP_JOINT, "joint_index", int, 1)                                                        \
    X(IDX_RAP_SHAPE, "shape_type", int, 1)                                                         \
    X(IDX_RAP_SIZE, "size", PyObject *, 1)                                                         \
    X(IDX_RAP_MASS, "mass", float, 0)                                                              \
    X(IDX_RAP_PARENT, "parent_index", int, 0)                                                      \
    X(IDX_RAP_TWIST_MIN, "twist_min", float, 0)                                                    \
    X(IDX_RAP_TWIST_MAX, "twist_max", float, 0)                                                    \
    X(IDX_RAP_CONE, "cone_angle", float, 0)                                                        \
    X(IDX_RAP_AXIS, "axis", Vec3f, 0)                                                              \
    X(IDX_RAP_NORMAL, "normal", Vec3f, 0)                                                          \
    X(IDX_RAP_POS, "pos", PyObject *, 0)

#define SCHEMA_ADD_JOINT(X)                                                                        \
    X(IDX_AJ_NAME, "name", PyObject *, 1)                                                          \
    X(IDX_AJ_PARENT, "parent_index", int, 0)

#define SCHEMA_GET_JOINT_IDX(X) X(IDX_GJI_NAME, "name", PyObject *, 1)

#define SCHEMA_RAGDOLL_DRIVE(X)                                                                    \
    X(IDX_RD_POS, "root_pos", PosStride, 1)                                                        \
    X(IDX_RD_ROT, "root_rot", AuxStride, 1)                                                        \
    X(IDX_RD_MATS, "matrices", PyObject *, 1)

/** --- THE GENERATOR ENGINE --- **/

#define GEN_ENUM(ID, NAME, TYPE, REQ) ID,

// Defines the Enum and the Count for a signature type
#define DEFINE_INDEX_GROUP(GroupName, Schema)                                                      \
    typedef enum { Schema(GEN_ENUM) GroupName##_COUNT } GroupName##_Idx;

// Declares a specific parser that uses an Index Group
#define DECLARE_PARSER(ParserName, GroupName)                                                      \
    extern FastParser ParserName##Parser;                                                          \
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
DEFINE_INDEX_GROUP(OverlapSphere, SCHEMA_OVERLAP_SPHERE)
DEFINE_INDEX_GROUP(OverlapAABB, SCHEMA_OVERLAP_AABB)
DEFINE_INDEX_GROUP(SetUserData, SCHEMA_SET_USER_DATA)
DEFINE_INDEX_GROUP(SetMotion, SCHEMA_SET_MOTION)
DEFINE_INDEX_GROUP(ColFilter, SCHEMA_COL_FILTER)
DEFINE_INDEX_GROUP(RegMat, SCHEMA_REG_MAT)
DEFINE_INDEX_GROUP(SetConstr, SCHEMA_SET_CONSTR_TARGET)
DEFINE_INDEX_GROUP(Heightfield, SCHEMA_HEIGHTFIELD)
DEFINE_INDEX_GROUP(DebugData, SCHEMA_DEBUG_DATA)
DEFINE_INDEX_GROUP(CreateConstr, SCHEMA_CREATE_CONSTR)
DEFINE_INDEX_GROUP(Step, SCHEMA_STEP)
DEFINE_INDEX_GROUP(CharMove, SCHEMA_CHAR_MOVE)
DEFINE_INDEX_GROUP(LoadState, SCHEMA_LOAD_STATE)
DEFINE_INDEX_GROUP(CreateChar, SCHEMA_CREATE_CHAR)
DEFINE_INDEX_GROUP(SetPosChar, SCHEMA_SET_POS_CHAR)
DEFINE_INDEX_GROUP(SetRotChar, SCHEMA_SET_ROT_CHAR)
DEFINE_INDEX_GROUP(SetStrengthChar, SCHEMA_SET_STRENGTH_CHAR)
DEFINE_INDEX_GROUP(VehicleInput, SCHEMA_VEHICLE_INPUT)
DEFINE_INDEX_GROUP(WheelIdx, SCHEMA_WHEEL_IDX)
DEFINE_INDEX_GROUP(TankInput, SCHEMA_TANK_INPUT)
DEFINE_INDEX_GROUP(CreateVehicle, SCHEMA_CREATE_VEHICLE)
DEFINE_INDEX_GROUP(CreateTracked, SCHEMA_CREATE_TRACKED)
DEFINE_INDEX_GROUP(CreateRagdoll, SCHEMA_CREATE_RAGDOLL)
DEFINE_INDEX_GROUP(RagdollSettings, SCHEMA_RAGDOLL_SETTINGS)
DEFINE_INDEX_GROUP(RagdollAddPart, SCHEMA_RAGDOLL_ADD_PART)
DEFINE_INDEX_GROUP(AddJoint, SCHEMA_ADD_JOINT)
DEFINE_INDEX_GROUP(GetJointIdx, SCHEMA_GET_JOINT_IDX)
DEFINE_INDEX_GROUP(RagdollDrive, SCHEMA_RAGDOLL_DRIVE)

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
DECLARE_PARSER(OverlapSphere, OverlapSphere)
DECLARE_PARSER(OverlapAABB, OverlapAABB)
DECLARE_PARSER(SetUserData, SetUserData)
DECLARE_PARSER(GetUserData, HOnly) // Reuses Handle-Only group
DECLARE_PARSER(GetMotion, HOnly)   // Reuses Handle-Only group
DECLARE_PARSER(SetMotion, SetMotion)
DECLARE_PARSER(ColFilter, ColFilter)
DECLARE_PARSER(RegMat, RegMat)
DECLARE_PARSER(SetConstrTarget, SetConstr)
DECLARE_PARSER(Heightfield, Heightfield)
DECLARE_PARSER(DebugData, DebugData)
DECLARE_PARSER(CreateConstr, CreateConstr)
DECLARE_PARSER(DestroyConstr, HOnly)
DECLARE_PARSER(Step, Step)
DECLARE_PARSER(CharMove, CharMove)
DECLARE_PARSER(LoadState, LoadState)
DECLARE_PARSER(CreateChar, CreateChar)
DECLARE_PARSER(SetPosChar, SetPosChar)
DECLARE_PARSER(SetRotChar, SetRotChar)
DECLARE_PARSER(SetStrengthChar, SetStrengthChar)
DECLARE_PARSER(VehicleInput, VehicleInput)
DECLARE_PARSER(WheelIdx, WheelIdx)
DECLARE_PARSER(TankInput, TankInput)
DECLARE_PARSER(CreateVehicle, CreateVehicle)
DECLARE_PARSER(CreateTracked, CreateTracked)
DECLARE_PARSER(CreateRagdoll, CreateRagdoll)
DECLARE_PARSER(RagdollSettings, RagdollSettings)
DECLARE_PARSER(RagdollAddPart, RagdollAddPart)
DECLARE_PARSER(AddJoint, AddJoint)
DECLARE_PARSER(GetJointIdx, GetJointIdx)
DECLARE_PARSER(RagdollDrive, RagdollDrive)

void culverin_init_all_parsers(void);
void culverin_free_all_parsers(void);
