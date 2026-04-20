#pragma once

#include "culverin_fast_parse.h"
#include "culverin_types.h"

typedef uint64_t ParseBodyHandle;
static constexpr size_t PARSER_REGISTRY_SIZE = 128;

/**
 * SCHEMA DEFINITIONS
 * Format: X(IndexName, PythonName, C-Type, IsRequired)
 */

#define SCHEMA_BODY(X)                                                                             \
    X(IDX_POS, "pos", PyObject *, false)                                                           \
    X(IDX_ROT, "rot", PyObject *, false)                                                           \
    X(IDX_SIZE, "size", PyObject *, false)                                                         \
    X(IDX_SHAPE, "shape", int, false)                                                              \
    X(IDX_MOTION, "motion", int, false)                                                            \
    X(IDX_USER_DATA, "user_data", uint64_t, false)                                                 \
    X(IDX_SENSOR, "is_sensor", bool, false)                                                        \
    X(IDX_MASS, "mass", float, false)                                                              \
    X(IDX_CAT, "category", uint32_t, false)                                                        \
    X(IDX_MASK, "mask", uint32_t, false)                                                           \
    X(IDX_FRIC, "friction", float, false)                                                          \
    X(IDX_REST, "restitution", float, false)                                                       \
    X(IDX_MAT, "material_id", uint32_t, false)                                                     \
    X(IDX_CCD, "ccd", bool, false)

#define SCHEMA_SET_POS(X)                                                                          \
    X(IDX_SETPOS_HANDLE, "handle", ParseBodyHandle, true)                                          \
    X(IDX_SETPOS_X, "x", JPH_Real, true)                                                           \
    X(IDX_SETPOS_Y, "y", JPH_Real, true)                                                           \
    X(IDX_SETPOS_Z, "z", JPH_Real, true)

// Shared by Impulse, AngImpulse, Force, and Torque
#define SCHEMA_VEC3(X)                                                                             \
    X(IDX_V3_H, "handle", ParseBodyHandle, true)                                                   \
    X(IDX_V3_X, "x", float, true)                                                                  \
    X(IDX_V3_Y, "y", float, true)                                                                  \
    X(IDX_V3_Z, "z", float, true)

#define SCHEMA_IMPULSE_AT(X)                                                                       \
    X(IDX_IMPAT_H, "handle", ParseBodyHandle, true)                                                \
    X(IDX_IMPAT_IX, "ix", float, true)                                                             \
    X(IDX_IMPAT_IY, "iy", float, true)                                                             \
    X(IDX_IMPAT_IZ, "iz", float, true)                                                             \
    X(IDX_IMPAT_PX, "px", JPH_Real, true)                                                          \
    X(IDX_IMPAT_PY, "py", JPH_Real, true)                                                          \
    X(IDX_IMPAT_PZ, "pz", JPH_Real, true)

#define SCHEMA_BUOYANCY(X)                                                                         \
    X(IDX_BUOY_HANDLE, "handle", ParseBodyHandle, true)                                            \
    X(IDX_BUOY_SURFACE_Y, "surface_y", double, true)                                               \
    X(IDX_BUOY_BUOYANCY, "buoyancy", float, false)                                                 \
    X(IDX_BUOY_LIN_DRAG, "linear_drag", float, false)                                              \
    X(IDX_BUOY_ANG_DRAG, "angular_drag", float, false)                                             \
    X(IDX_BUOY_DT, "dt", float, false)                                                             \
    X(IDX_BUOY_VEL, "fluid_velocity", PyObject *, false)

#define SCHEMA_HULL_OR_COMP(X)                                                                     \
    X(IDX_HC_POS, "pos", PyObject *, true)                                                         \
    X(IDX_HC_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_HC_DATA, "points_or_parts", PyObject *, true)                                            \
    X(IDX_HC_MOTION, "motion", int, false)                                                         \
    X(IDX_HC_MASS, "mass", float, false)                                                           \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, false)                                              \
    X(IDX_HC_SENSOR, "is_sensor", bool, false)                                                     \
    X(IDX_HC_CAT, "category", uint32_t, false)                                                     \
    X(IDX_HC_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, false)                                               \
    X(IDX_HC_FRIC, "friction", float, false)                                                       \
    X(IDX_HC_REST, "restitution", float, false)                                                    \
    X(IDX_HC_CCD, "ccd", bool, false)

#define SCHEMA_MESH(X)                                                                             \
    X(IDX_MSH_POS, "pos", PyObject *, true)                                                        \
    X(IDX_MSH_ROT, "rot", PyObject *, true)                                                        \
    X(IDX_MSH_VERTS, "vertices", PyObject *, true)                                                 \
    X(IDX_MSH_INDICES, "indices", PyObject *, true)                                                \
    X(IDX_MSH_USER_DATA, "user_data", uint64_t, false)                                             \
    X(IDX_MSH_CAT, "category", uint32_t, false)                                                    \
    X(IDX_MSH_MASK, "mask", uint32_t, false)

#define SCHEMA_HANDLE_ONLY(X) X(IDX_H_H, "handle", ParseBodyHandle, true)

#define SCHEMA_SET_TRNS(X)                                                                         \
    X(IDX_ST_HANDLE, "handle", ParseBodyHandle, true)                                              \
    X(IDX_ST_POS, "pos", PyObject *, true)                                                         \
    X(IDX_ST_ROT, "rot", PyObject *, true)

#define SCHEMA_CCD(X)                                                                              \
    X(IDX_CCD_H, "handle", ParseBodyHandle, true)                                                  \
    X(IDX_CCD_E, "enabled", bool, true)

#define SCHEMA_XYZ(X)                                                                              \
    X(IDX_XYZ_X, "x", float, true)                                                                 \
    X(IDX_XYZ_Y, "y", float, true)                                                                 \
    X(IDX_XYZ_Z, "z", float, true)

#define SCHEMA_BATCH_BUOYANCY(X)                                                                   \
    X(IDX_BBUOY_HANDLES, "handles", PyObject *, true)                                              \
    X(IDX_BBUOY_SURFACE_Y, "surface_y", double, true)                                              \
    X(IDX_BBUOY_BUOYANCY, "buoyancy", float, false)                                                \
    X(IDX_BBUOY_LIN_DRAG, "linear_drag", float, false)                                             \
    X(IDX_BBUOY_ANG_DRAG, "angular_drag", float, false)                                            \
    X(IDX_BBUOY_DT, "dt", float, false)                                                            \
    X(IDX_BBUOY_VEL, "fluid_velocity", PyObject *, false)

// This is the "Master" structural schema for Hull/Compound/Body-like objects
#define SCHEMA_HC_MASTER(X)                                                                        \
    X(IDX_HC_POS, "pos", PyObject *, true)                                                         \
    X(IDX_HC_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_HC_DATA, "data", PyObject *, true)                                                       \
    X(IDX_HC_MOTION, "motion", int, false)                                                         \
    X(IDX_HC_MASS, "mass", float, false)                                                           \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, false)                                              \
    X(IDX_HC_SENSOR, "is_sensor", bool, false)                                                     \
    X(IDX_HC_CAT, "category", uint32_t, false)                                                     \
    X(IDX_HC_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, false)                                               \
    X(IDX_HC_FRIC, "friction", float, false)                                                       \
    X(IDX_HC_REST, "restitution", float, false)                                                    \
    X(IDX_HC_CCD, "ccd", bool, false)

// These are specific overlays just to change the Python keyword strings
#define SCHEMA_HC_HULL(X)                                                                          \
    X(IDX_HC_POS, "pos", PyObject *, true)                                                         \
    X(IDX_HC_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_HC_DATA, "points", PyObject *, true) /* Change name to "points" */                       \
    X(IDX_HC_MOTION, "motion", int, false)                                                         \
    X(IDX_HC_MASS, "mass", float, false)                                                           \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, false)                                              \
    X(IDX_HC_SENSOR, "is_sensor", bool, false)                                                     \
    X(IDX_HC_CAT, "category", uint32_t, false)                                                     \
    X(IDX_HC_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, false)                                               \
    X(IDX_HC_FRIC, "friction", float, false)                                                       \
    X(IDX_HC_REST, "restitution", float, false)                                                    \
    X(IDX_HC_CCD, "ccd", bool, false)

#define SCHEMA_HC_COMP(X)                                                                          \
    X(IDX_HC_POS, "pos", PyObject *, true)                                                         \
    X(IDX_HC_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_HC_DATA, "parts", PyObject *, true) /* Change name to "parts" */                         \
    X(IDX_HC_MOTION, "motion", int, false)                                                         \
    X(IDX_HC_MASS, "mass", float, false)                                                           \
    X(IDX_HC_USER_DATA, "user_data", uint64_t, false)                                              \
    X(IDX_HC_SENSOR, "is_sensor", bool, false)                                                     \
    X(IDX_HC_CAT, "category", uint32_t, false)                                                     \
    X(IDX_HC_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_HC_MAT_ID, "material_id", uint32_t, false)                                               \
    X(IDX_HC_FRIC, "friction", float, false)                                                       \
    X(IDX_HC_REST, "restitution", float, false)                                                    \
    X(IDX_HC_CCD, "ccd", bool, false)

#define SCHEMA_BATCH_CREATE(X)                                                                     \
    X(IDX_BC_POSITIONS, "positions", PyObject *, true)                                             \
    X(IDX_BC_SIZES, "sizes", PyObject *, true)                                                     \
    X(IDX_BC_SHAPE, "shape_type", int, false)                                                      \
    X(IDX_BC_MOTION, "motion_type", int, false)

#define SCHEMA_BATCH_DESTROY(X) X(IDX_BD_HANDLES, "handles", PyObject *, true)

#define SCHEMA_SET_ROT(X)                                                                          \
    X(IDX_SETROT_H, "handle", ParseBodyHandle, true)                                               \
    X(IDX_SETROT_X, "x", float, true)                                                              \
    X(IDX_SETROT_Y, "y", float, true)                                                              \
    X(IDX_SETROT_Z, "z", float, true)                                                              \
    X(IDX_SETROT_W, "w", float, true)

#define SCHEMA_RENDER(X) X(IDX_RND_ALPHA, "alpha", float, true)

#define SCHEMA_RAYCAST(X)                                                                          \
    X(IDX_RAY_START, "start", PyObject *, true)                                                    \
    X(IDX_RAY_DIR, "direction", PyObject *, true)                                                  \
    X(IDX_RAY_DIST, "max_dist", float, false)                                                      \
    X(IDX_RAY_IGN, "ignore", ParseBodyHandle, false)

#define SCHEMA_RAYCAST_BATCH(X)                                                                    \
    X(IDX_RB_STARTS, "starts", PyObject *, true)                                                   \
    X(IDX_RB_DIRS, "directions", PyObject *, true)                                                 \
    X(IDX_RB_DIST, "max_dist", float, false)

#define SCHEMA_SHAPECAST(X)                                                                        \
    X(IDX_SC_SHAPE, "shape", int, true)                                                            \
    X(IDX_SC_POS, "pos", PyObject *, true)                                                         \
    X(IDX_SC_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_SC_DIR, "dir", PyObject *, true)                                                         \
    X(IDX_SC_SIZE, "size", PyObject *, false)                                                      \
    X(IDX_SC_IGNORE, "ignore", ParseBodyHandle, false)

#define SCHEMA_OVERLAP_SPHERE(X)                                                                   \
    X(IDX_OS_CENTER, "center", PyObject *, true)                                                   \
    X(IDX_OS_RADIUS, "radius", float, true)

#define SCHEMA_OVERLAP_AABB(X)                                                                     \
    X(IDX_OA_MIN, "min", PyObject *, true)                                                         \
    X(IDX_OA_MAX, "max", PyObject *, true)

#define SCHEMA_SET_USER_DATA(X)                                                                    \
    X(IDX_SUD_H, "handle", ParseBodyHandle, true)                                                  \
    X(IDX_SUD_D, "data", uint64_t, true)

#define SCHEMA_SET_MOTION(X)                                                                       \
    X(IDX_SM_H, "handle", ParseBodyHandle, true)                                                   \
    X(IDX_SM_M, "motion", int, true)

#define SCHEMA_COL_FILTER(X)                                                                       \
    X(IDX_CF_H, "handle", ParseBodyHandle, true)                                                   \
    X(IDX_CF_C, "category", uint32_t, true)                                                        \
    X(IDX_CF_M, "mask", uint32_t, true)

#define SCHEMA_REG_MAT(X)                                                                          \
    X(IDX_RM_ID, "id", uint32_t, true)                                                             \
    X(IDX_RM_FRIC, "friction", float, false)                                                       \
    X(IDX_RM_REST, "restitution", float, false)

#define SCHEMA_SET_CONSTR_TARGET(X)                                                                \
    X(IDX_SCT_H, "handle", uint64_t, true)                                                         \
    X(IDX_SCT_T, "target", float, true)

#define SCHEMA_HEIGHTFIELD(X)                                                                      \
    X(IDX_HF_POS, "pos", PyObject *, true)                                                         \
    X(IDX_HF_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_HF_SCALE, "scale", PyObject *, true)                                                     \
    X(IDX_HF_HEIGHTS, "heights", PyObject *, true)                                                 \
    X(IDX_HF_GRID_SIZE, "grid_size", int, true)                                                    \
    X(IDX_HF_USER_DATA, "user_data", uint64_t, false)                                              \
    X(IDX_HF_CAT, "category", uint32_t, false)                                                     \
    X(IDX_HF_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_HF_MAT_ID, "material_id", uint32_t, false)                                               \
    X(IDX_HF_FRIC, "friction", float, false)                                                       \
    X(IDX_HF_REST, "restitution", float, false)

#define SCHEMA_DEBUG_DATA(X)                                                                       \
    X(IDX_DD_SHAPES, "shapes", bool, false)                                                        \
    X(IDX_DD_CONSTRAINTS, "constraints", bool, false)                                              \
    X(IDX_DD_BBOX, "bbox", bool, false)                                                            \
    X(IDX_DD_CENTERS, "centers", bool, false)                                                      \
    X(IDX_DD_WIREFRAME, "wireframe", bool, false)

#define SCHEMA_CREATE_CONSTR(X)                                                                    \
    X(IDX_CC_TYPE, "type", int, true)                                                              \
    X(IDX_CC_BODY1, "body1", ParseBodyHandle, true)                                                \
    X(IDX_CC_BODY2, "body2", ParseBodyHandle, true)                                                \
    X(IDX_CC_PARAMS, "params", PyObject *, false)                                                  \
    X(IDX_CC_MOTOR, "motor", PyObject *, false)

#define SCHEMA_STEP(X) X(IDX_STEP_DT, "dt", float, false) // Optional (0)

#define SCHEMA_CHAR_MOVE(X)                                                                        \
    X(IDX_CM_VEL, "velocity", Vec3f, true)                                                         \
    X(IDX_CM_DT, "dt", float, true)

#define SCHEMA_LOAD_STATE(X) X(IDX_LS_STATE, "state", PyObject *, true)

#define SCHEMA_CREATE_CHAR(X)                                                                      \
    X(IDX_CCHAR_POS, "pos", PosStride, true)                                                       \
    X(IDX_CCHAR_H, "height", float, false)                                                         \
    X(IDX_CCHAR_R, "radius", float, false)                                                         \
    X(IDX_CCHAR_STEP, "step_height", float, false)                                                 \
    X(IDX_CCHAR_SLOPE, "max_slope", float, false)

#define SCHEMA_SET_POS_CHAR(X) X(IDX_SPC_POS, "pos", PosStride, true)

#define SCHEMA_SET_ROT_CHAR(X) X(IDX_SRC_ROT, "rot", AuxStride, true)

#define SCHEMA_SET_STRENGTH_CHAR(X) X(IDX_SSC_STRENGTH, "strength", float, true)

#define SCHEMA_VEHICLE_INPUT(X)                                                                    \
    X(IDX_VI_FWD, "forward", float, false)                                                         \
    X(IDX_VI_RIGHT, "right", float, false)                                                         \
    X(IDX_VI_BRAKE, "brake", float, false)                                                         \
    X(IDX_VI_HAND, "handbrake", float, false)

#define SCHEMA_WHEEL_IDX(X) X(IDX_WH_INDEX, "index", uint32_t, true)

#define SCHEMA_TANK_INPUT(X)                                                                       \
    X(IDX_TI_LEFT, "left", float, true)                                                            \
    X(IDX_TI_RIGHT, "right", float, true)                                                          \
    X(IDX_TI_BRAKE, "brake", float, false)

#define SCHEMA_CREATE_VEHICLE(X)                                                                   \
    X(IDX_CV_CHASSIS, "chassis", uint64_t, true)                                                   \
    X(IDX_CV_WHEELS, "wheels", PyObject *, true)                                                   \
    X(IDX_CV_DRIVE, "drive", PyObject *, false)                                                    \
    X(IDX_CV_ENGINE, "engine", PyObject *, false)                                                  \
    X(IDX_CV_TRANS, "transmission", PyObject *, false)

#define SCHEMA_CREATE_TRACKED(X)                                                                   \
    X(IDX_CT_CHASSIS, "chassis", uint64_t, true)                                                   \
    X(IDX_CT_WHEELS, "wheels", PyObject *, true)                                                   \
    X(IDX_CT_TRACKS, "tracks", PyObject *, true)                                                   \
    X(IDX_CT_TORQUE, "max_torque", float, false)                                                   \
    X(IDX_CT_RPM, "max_rpm", float, false)

#define SCHEMA_CREATE_RAGDOLL(X)                                                                   \
    X(IDX_CR_SETTINGS, "settings", PyObject *, true)                                               \
    X(IDX_CR_POS, "pos", PosStride, true)                                                          \
    X(IDX_CR_ROT, "rot", AuxStride, false)                                                         \
    X(IDX_CR_USER, "user_data", uint64_t, false)                                                   \
    X(IDX_CR_CAT, "category", uint32_t, false)                                                     \
    X(IDX_CR_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_CR_MAT, "material_id", uint32_t, false)

#define SCHEMA_RAGDOLL_SETTINGS(X) X(IDX_RS_SKELETON, "skeleton", PyObject *, true)

#define SCHEMA_RAGDOLL_ADD_PART(X)                                                                 \
    X(IDX_RAP_JOINT, "joint_index", int, true)                                                     \
    X(IDX_RAP_SHAPE, "shape_type", int, true)                                                      \
    X(IDX_RAP_SIZE, "size", PyObject *, true)                                                      \
    X(IDX_RAP_MASS, "mass", float, false)                                                          \
    X(IDX_RAP_PARENT, "parent_index", int, false)                                                  \
    X(IDX_RAP_TWIST_MIN, "twist_min", float, false)                                                \
    X(IDX_RAP_TWIST_MAX, "twist_max", float, false)                                                \
    X(IDX_RAP_CONE, "cone_angle", float, false)                                                    \
    X(IDX_RAP_AXIS, "axis", Vec3f, false)                                                          \
    X(IDX_RAP_NORMAL, "normal", Vec3f, false)                                                      \
    X(IDX_RAP_POS, "pos", PyObject *, false)

#define SCHEMA_ADD_JOINT(X)                                                                        \
    X(IDX_AJ_NAME, "name", PyObject *, true)                                                       \
    X(IDX_AJ_PARENT, "parent_index", int, false)

#define SCHEMA_GET_JOINT_IDX(X) X(IDX_GJI_NAME, "name", PyObject *, true)

#define SCHEMA_RAGDOLL_DRIVE(X)                                                                    \
    X(IDX_RD_POS, "root_pos", PosStride, true)                                                     \
    X(IDX_RD_ROT, "root_rot", AuxStride, true)                                                     \
    X(IDX_RD_MATS, "matrices", PyObject *, true)

#define SCHEMA_SBSS_CREATE_CONSTRAINTS(X)                                                          \
    X(IDX_SCC_COMP, "compliance", float, true)                                                     \
    X(IDX_SCC_BEND, "bend_type", int, false)

#define SCHEMA_SBSS_ADD_VERTICES(X)                                                                \
    X(IDX_SAVS_POS, "positions", PyObject *, true)                                                 \
    X(IDX_SAVS_MASS, "inv_masses", PyObject *, false)

#define SCHEMA_SBSS_ADD_FACES(X) X(IDX_SAFS_IND, "indices", PyObject *, true)

#define SCHEMA_GET_SB_VERTEX(X)                                                                    \
    X(IDX_GSBV_H, "handle", ParseBodyHandle, true)                                                 \
    X(IDX_GSBV_I, "index", uint32_t, true)

#define SCHEMA_CREATE_SOFT_BODY(X)                                                                 \
    X(IDX_CSB_SHARED, "shared_settings", PyObject *, true)                                         \
    X(IDX_CSB_POS, "pos", PyObject *, true)                                                        \
    X(IDX_CSB_ROT, "rot", PyObject *, true)                                                        \
    X(IDX_CSB_USER_DATA, "user_data", uint64_t, false)                                             \
    X(IDX_CSB_CAT, "category", uint32_t, false)                                                    \
    X(IDX_CSB_MASK, "mask", uint32_t, false)                                                       \
    X(IDX_CSB_PRESSURE, "pressure", float, false)                                                  \
    X(IDX_CSB_V_RADIUS, "vertex_radius", float, false)                                             \
    X(IDX_CSB_LIN_DAMP, "linear_damping", float, false)                                            \
    X(IDX_CSB_ITER, "num_iterations", uint32_t, false)                                             \
    X(IDX_CSB_MAX_VEL, "max_linear_velocity", float, false)                                        \
    X(IDX_CSB_GRAV, "gravity_factor", float, false)                                                \
    X(IDX_CSB_FRIC, "friction", float, false)                                                      \
    X(IDX_CSB_REST, "restitution", float, false)                                                   \
    X(IDX_CSB_ROT_ID, "make_rotation_identity", bool, false)                                       \
    X(IDX_CSB_UPDATE_POS, "update_position", bool, false)                                          \
    X(IDX_CSB_FACE_DS, "faces_double_sided", bool, false)

#define SCHEMA_SBSS_ADD_VERTEX(X)                                                                  \
    X(IDX_SAV_POS, "pos", PyObject *, true)                                                        \
    X(IDX_SAV_MASS, "inv_mass", float, true)

#define SCHEMA_SBSS_ADD_FACE(X)                                                                    \
    X(IDX_SAF_V1, "v1", uint32_t, true)                                                            \
    X(IDX_SAF_V2, "v2", uint32_t, true)                                                            \
    X(IDX_SAF_V3, "v3", uint32_t, true)

#define SCHEMA_REG_ENTITY_ONLY(X) X(IDX_REO_ENT, "entity", uint64_t, true)

#define SCHEMA_REG_COMP_ONLY(X) X(IDX_RCO_COMP, "comp_id", uint32_t, true)

#define SCHEMA_REG_REG_COMP(X) X(IDX_RRC_SIZE, "size_bytes", uint32_t, true)

#define SCHEMA_REG_ADD(X)                                                                          \
    X(IDX_RA_ENT, "entity", uint64_t, true)                                                        \
    X(IDX_RA_COMP, "comp_id", uint32_t, true)                                                      \
    X(IDX_RA_DATA, "data", PyObject *, false)

#define SCHEMA_REG_ENT_COMP(X)                                                                     \
    X(IDX_REC_ENT, "entity", uint64_t, true)                                                       \
    X(IDX_REC_COMP, "comp_id", uint32_t, true)

#define SCHEMA_REG_SYNC_PHYS(X)                                                                    \
    X(IDX_RSP_WORLD, "world", PyObject *, true)                                                    \
    X(IDX_RSP_H_COMP, "handle_comp_id", uint32_t, true)                                            \
    X(IDX_RSP_T_COMP, "transform_comp_id", uint32_t, true)                                         \
    X(IDX_RSP_R_COMP, "rot_comp_id", int, false)

#define SCHEMA_MATH_PERSPECTIVE(X)                                                                 \
    X(IDX_MP_FOVY, "fovy", float, true)                                                            \
    X(IDX_MP_ASPECT, "aspect", float, true)                                                        \
    X(IDX_MP_NEAR, "near", float, true)                                                            \
    X(IDX_MP_FAR, "far", float, true)

#define SCHEMA_MATH_ORTHO(X)                                                                       \
    X(IDX_MO_LEFT, "left", float, true)                                                            \
    X(IDX_MO_RIGHT, "right", float, true)                                                          \
    X(IDX_MO_BOTTOM, "bottom", float, true)                                                        \
    X(IDX_MO_TOP, "top", float, true)                                                              \
    X(IDX_MO_NEAR, "near", float, true)                                                            \
    X(IDX_MO_FAR, "far", float, true)

#define SCHEMA_MATH_TRIO(X)                                                                        \
    X(IDX_MT_0, "arg0", PyObject *, true)                                                          \
    X(IDX_MT_1, "arg1", PyObject *, true)                                                          \
    X(IDX_MT_2, "arg2", PyObject *, true)

#define SCHEMA_MATH_MAT(X) X(IDX_MMM_MAT, "mat", PyObject *, true)
#define SCHEMA_MATH_MAT_PAIR(X)                                                                    \
    X(IDX_MMP_A, "a", PyObject *, true) X(IDX_MMP_B, "b", PyObject *, true)
#define SCHEMA_MATH_MAT_VEC(X)                                                                     \
    X(IDX_MMV_MAT, "mat", PyObject *, true) X(IDX_MMV_VEC, "vec", PyObject *, true)
#define SCHEMA_MATH_MAT_BATCH(X)                                                                   \
    X(IDX_MMB_MAT, "mat", PyObject *, true) X(IDX_MMB_BATCH, "batch", PyObject *, true)
#define SCHEMA_MATH_CULL(X)                                                                        \
    X(IDX_MC_VP, "vp", PyObject *, true)                                                           \
    X(IDX_MC_MIN, "min", PyObject *, true)                                                         \
    X(IDX_MC_MAX, "max", PyObject *, true)

#define SCHEMA_MATH_CULL_BATCH(X)                                                                  \
    X(IDX_MCB_VP, "vp", PyObject *, true)                                                          \
    X(IDX_MCB_AABBS, "aabbs", PyObject *, true)

#define SCHEMA_MATH_VEC3_BATCH(X) X(IDX_MVB_VECS, "vecs", PyObject *, true)

#define SCHEMA_MATH_EULER(X)                                                                       \
    X(IDX_ME_X, "x", float, true)                                                                  \
    X(IDX_ME_Y, "y", float, true)                                                                  \
    X(IDX_ME_Z, "z", float, true)

#define SCHEMA_MATH_QUAT(X)                                                                        \
    X(IDX_MQ_X, "x", float, true)                                                                  \
    X(IDX_MQ_Y, "y", float, true)                                                                  \
    X(IDX_MQ_Z, "z", float, true)                                                                  \
    X(IDX_MQ_W, "w", float, true)

#define SCHEMA_MATH_SLERP(X)                                                                       \
    X(IDX_MS_Q1, "q1", PyObject *, true)                                                           \
    X(IDX_MS_Q2, "q2", PyObject *, true)                                                           \
    X(IDX_MS_T, "t", float, true)

#define SCHEMA_MATH_QUAT_PAIR(X)                                                                   \
    X(IDX_MQP_A, "a", PyObject *, true)                                                            \
    X(IDX_MQP_B, "b", PyObject *, true)

#define SCHEMA_MATH_LERP_BATCH(X)                                                                  \
    X(IDX_MLB_VECS_A, "vecs_a", PyObject *, true)                                                  \
    X(IDX_MLB_VECS_B, "vecs_b", PyObject *, true)                                                  \
    X(IDX_MLB_ALPHA, "alpha", float, true)

#define SCHEMA_MATH_QUAT_VEC(X)                                                                    \
    X(IDX_MQV_Q, "q", PyObject *, true)                                                            \
    X(IDX_MQV_V, "v", PyObject *, true)

#define SCHEMA_MATH_QUAT_VEC_BATCH(X)                                                              \
    X(IDX_MQVB_Q, "q", PyObject *, true)                                                           \
    X(IDX_MQVB_VECS, "vecs", PyObject *, true)

#define SCHEMA_MATH_QUAT_OP(X) X(IDX_MQO_Q, "q", PyObject *, true)

#define SCHEMA_MATH_PROJECT(X)                                                                     \
    X(IDX_MPR_V, "v", PyObject *, true)                                                            \
    X(IDX_MPR_MVP, "mvp", PyObject *, true)                                                        \
    X(IDX_MPR_VP, "viewport", PyObject *, true)

#define SCHEMA_MATH_UNPROJECT(X)                                                                   \
    X(IDX_MUP_V, "v", PyObject *, true)                                                            \
    X(IDX_MUP_MVP, "mvp", PyObject *, true)                                                        \
    X(IDX_MUP_VP, "viewport", PyObject *, true)

#define SCHEMA_MATH_VEC_PAIR(X)                                                                    \
    X(IDX_MVP_V1, "v1", PyObject *, true)                                                          \
    X(IDX_MVP_V2, "v2", PyObject *, true)

#define SCHEMA_MATH_RAY_PLANE(X)                                                                   \
    X(IDX_RP_RO, "ray_origin", PyObject *, true)                                                   \
    X(IDX_RP_RD, "ray_dir", PyObject *, true)                                                      \
    X(IDX_RP_PO, "plane_pos", PyObject *, true)                                                    \
    X(IDX_RP_PN, "plane_norm", PyObject *, true)

#define SCHEMA_MATH_AXIS_ANGLE(X)                                                                  \
    X(IDX_MAA_AXIS, "axis", PyObject *, true)                                                      \
    X(IDX_MAA_ANGLE, "angle", float, true)

#define SCHEMA_MATH_DIST_BATCH(X)                                                                  \
    X(IDX_MDB_VECS_A, "vecs_a", PyObject *, true)                                                  \
    X(IDX_MDB_VECS_B, "vecs_b", PyObject *, true)

#define SCHEMA_MATH_VEC_OP(X) X(IDX_MVO_V, "v", PyObject *, true)

#define SCHEMA_MATH_REFLECT(X)                                                                     \
    X(IDX_MRF_V, "v", PyObject *, true)                                                            \
    X(IDX_MRF_N, "normal", PyObject *, true)

#define SCHEMA_CREATE_SHIP(X)                                                                      \
    X(IDX_CS_SLED, "sled", ParseBodyHandle, true)                                                  \
    X(IDX_CS_KP, "kp", float, true)                                                                \
    X(IDX_CS_KD, "kd", float, true)                                                                \
    X(IDX_CS_THROTTLE, "throttle_force", float, true)                                              \
    X(IDX_CS_STEER, "steer_speed", float, true)                                                    \
    X(IDX_CS_BANKING, "banking", float, false)                                                     \
    X(IDX_CS_GRIP, "lateral_grip", float, false)                                                   \
    X(IDX_CS_DRAG, "linear_drag", float, false)

#define SCHEMA_SHIP_INPUT(X)                                                                       \
    X(IDX_SI_FWD, "forward", float, true)                                                          \
    X(IDX_SI_RIGHT, "right", float, true)

#define SCHEMA_STRESS_TEST(X)                                                                      \
    X(IDX_0, "a0", uint64_t, false)                                                                \
    X(IDX_1, "a1", uint64_t, false)                                                                \
    X(IDX_2, "a2", uint64_t, false)                                                                \
    X(IDX_3, "a3", uint64_t, false)                                                                \
    X(IDX_4, "a4", uint64_t, false)                                                                \
    X(IDX_5, "a5", uint64_t, false)                                                                \
    X(IDX_6, "a6", uint64_t, false)                                                                \
    X(IDX_7, "a7", uint64_t, false)                                                                \
    X(IDX_8, "a8", uint64_t, false)                                                                \
    X(IDX_9, "a9", uint64_t, false)                                                                \
    X(IDX_10, "a10", uint64_t, false)                                                              \
    X(IDX_11, "a11", uint64_t, false)                                                              \
    X(IDX_12, "a12", uint64_t, false)                                                              \
    X(IDX_13, "a13", uint64_t, false)                                                              \
    X(IDX_14, "a14", uint64_t, false)                                                              \
    X(IDX_15, "a15", uint64_t, false)                                                              \
    X(IDX_16, "a16", uint64_t, false)                                                              \
    X(IDX_17, "a17", uint64_t, false)                                                              \
    X(IDX_18, "a18", uint64_t, false)                                                              \
    X(IDX_19, "a19", uint64_t, false)                                                              \
    X(IDX_20, "a20", uint64_t, false)                                                              \
    X(IDX_21, "a21", uint64_t, 0)                                                                  \
    X(IDX_22, "a22", uint64_t, false)                                                              \
    X(IDX_23, "a23", uint64_t, false)                                                              \
    X(IDX_24, "a24", uint64_t, false)                                                              \
    X(IDX_25, "a25", uint64_t, false)                                                              \
    X(IDX_26, "a26", uint64_t, false)                                                              \
    X(IDX_27, "a27", uint64_t, false)                                                              \
    X(IDX_28, "a28", uint64_t, false)                                                              \
    X(IDX_29, "a29", uint64_t, false)                                                              \
    X(IDX_30, "a30", uint64_t, false)                                                              \
    X(IDX_31, "a31", uint64_t, false)                                                              \
    X(IDX_32, "a32", uint64_t, false)                                                              \
    X(IDX_33, "a33", uint64_t, false)                                                              \
    X(IDX_34, "a34", uint64_t, false)                                                              \
    X(IDX_35, "a35", uint64_t, false)                                                              \
    X(IDX_36, "a36", uint64_t, false)                                                              \
    X(IDX_37, "a37", uint64_t, 0)                                                                  \
    X(IDX_38, "a38", uint64_t, false) X(IDX_39, "a39", uint64_t, false)                            \
        X(IDX_40, "a40", uint64_t, false) X(IDX_41, "a41", uint64_t, 0)                            \
            X(IDX_42, "a42", uint64_t, false) X(IDX_43, "a43", uint64_t, false)                    \
                X(IDX_44, "a44", uint64_t, false) X(IDX_45, "a45", uint64_t, 0)                    \
                    X(IDX_46, "a46", uint64_t, false) X(IDX_47, "a47", uint64_t, 0)                \
                        X(IDX_48, "a48", uint64_t, 0) X(IDX_49, "a49", uint64_t, 0)                \
                            X(IDX_50, "a50", uint64_t, 0) X(IDX_51, "a51", uint64_t, false)        \
                                X(IDX_52, "a52", uint64_t, 0) X(IDX_53, "a53", uint64_t, 0)        \
                                    X(IDX_54, "a54", uint64_t, 0) X(IDX_55, "a55", uint64_t, 0)    \
                                        X(IDX_56, "a56", uint64_t, false)                          \
                                            X(IDX_57, "a57", uint64_t, 0) X(IDX_58, "a58",         \
                                                                            uint64_t, false)       \
                                                X(IDX_59, "a59", uint64_t, false)                  \
                                                    X(IDX_60, "a60", uint64_t, false)              \
                                                        X(IDX_61, "a61", uint64_t, false)          \
                                                            X(IDX_62, "a62", uint64_t, false)      \
                                                                X(IDX_63, "a63", uint64_t, false)

/** --- THE GENERATOR ENGINE --- **/

#define GEN_ENUM(ID, NAME, TYPE, REQ) ID,

// Defines the Enum and the Count for a signature type
#define DEFINE_INDEX_GROUP(GroupName, Schema)                                                      \
    typedef enum { Schema(GEN_ENUM) GroupName##_COUNT } GroupName##_Idx;

// Declares a specific parser that uses an Index Group
#define DECLARE_PARSER(ParserName, GroupName)                                                      \
    FastParser ParserName##Parser;                                                                 \
    FastArgSpec ParserName##Specs[GroupName##_COUNT];

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
DEFINE_INDEX_GROUP(SbssCreateConstraints, SCHEMA_SBSS_CREATE_CONSTRAINTS)
DEFINE_INDEX_GROUP(CreateSoftBody, SCHEMA_CREATE_SOFT_BODY)
DEFINE_INDEX_GROUP(SbssAddVertex, SCHEMA_SBSS_ADD_VERTEX)
DEFINE_INDEX_GROUP(SbssAddFace, SCHEMA_SBSS_ADD_FACE)
DEFINE_INDEX_GROUP(SbssAddVertices, SCHEMA_SBSS_ADD_VERTICES)
DEFINE_INDEX_GROUP(SbssAddFaces, SCHEMA_SBSS_ADD_FACES)
DEFINE_INDEX_GROUP(GetSbVertex, SCHEMA_GET_SB_VERTEX)
DEFINE_INDEX_GROUP(RegEntityOnly, SCHEMA_REG_ENTITY_ONLY)
DEFINE_INDEX_GROUP(RegCompOnly, SCHEMA_REG_COMP_ONLY)
DEFINE_INDEX_GROUP(RegRegComp, SCHEMA_REG_REG_COMP)
DEFINE_INDEX_GROUP(RegAdd, SCHEMA_REG_ADD)
DEFINE_INDEX_GROUP(RegEntComp, SCHEMA_REG_ENT_COMP)
DEFINE_INDEX_GROUP(RegSyncPhys, SCHEMA_REG_SYNC_PHYS)
DEFINE_INDEX_GROUP(MathPersp, SCHEMA_MATH_PERSPECTIVE)
DEFINE_INDEX_GROUP(MathOrtho, SCHEMA_MATH_ORTHO)
DEFINE_INDEX_GROUP(MathTrio, SCHEMA_MATH_TRIO)
DEFINE_INDEX_GROUP(MathMat, SCHEMA_MATH_MAT)
DEFINE_INDEX_GROUP(MathMatPair, SCHEMA_MATH_MAT_PAIR)
DEFINE_INDEX_GROUP(MathMatVec, SCHEMA_MATH_MAT_VEC)
DEFINE_INDEX_GROUP(MathMatBatch, SCHEMA_MATH_MAT_BATCH)
DEFINE_INDEX_GROUP(MathCull, SCHEMA_MATH_CULL)
DEFINE_INDEX_GROUP(MathCullBatch, SCHEMA_MATH_CULL_BATCH)
DEFINE_INDEX_GROUP(MathVec3Batch, SCHEMA_MATH_VEC3_BATCH)
DEFINE_INDEX_GROUP(MathEuler, SCHEMA_MATH_EULER)
DEFINE_INDEX_GROUP(MathQuat, SCHEMA_MATH_QUAT)
DEFINE_INDEX_GROUP(MathSlerp, SCHEMA_MATH_SLERP)
DEFINE_INDEX_GROUP(MathQuatPair, SCHEMA_MATH_QUAT_PAIR)
DEFINE_INDEX_GROUP(MathLerpBatch, SCHEMA_MATH_LERP_BATCH)
DEFINE_INDEX_GROUP(MathQuatVec, SCHEMA_MATH_QUAT_VEC)
DEFINE_INDEX_GROUP(MathQuatVecBatch, SCHEMA_MATH_QUAT_VEC_BATCH)
DEFINE_INDEX_GROUP(MathQuatOp, SCHEMA_MATH_QUAT_OP)
DEFINE_INDEX_GROUP(MathProject, SCHEMA_MATH_PROJECT)
DEFINE_INDEX_GROUP(MathUnproject, SCHEMA_MATH_UNPROJECT)
DEFINE_INDEX_GROUP(MathVecPair, SCHEMA_MATH_VEC_PAIR)
DEFINE_INDEX_GROUP(MathRayPlane, SCHEMA_MATH_RAY_PLANE)
DEFINE_INDEX_GROUP(MathAxisAngle, SCHEMA_MATH_AXIS_ANGLE)
DEFINE_INDEX_GROUP(MathDistBatch, SCHEMA_MATH_DIST_BATCH)
DEFINE_INDEX_GROUP(MathVecOp, SCHEMA_MATH_VEC_OP)
DEFINE_INDEX_GROUP(MathReflect, SCHEMA_MATH_REFLECT)
DEFINE_INDEX_GROUP(CreateShip, SCHEMA_CREATE_SHIP)
DEFINE_INDEX_GROUP(ShipInput, SCHEMA_SHIP_INPUT)
DEFINE_INDEX_GROUP(StressTest, SCHEMA_STRESS_TEST)

#define FOR_ALL_PARSERS(X)                                                                         \
    X(Body, Body, SCHEMA_BODY)                                                                     \
    X(Impulse, Vec3, SCHEMA_VEC3)                                                                  \
    X(WheelIdx, WheelIdx, SCHEMA_WHEEL_IDX)                                                        \
    X(AngImpulse, Vec3, SCHEMA_VEC3)                                                               \
    X(Force, Vec3, SCHEMA_VEC3)                                                                    \
    X(Torque, Vec3, SCHEMA_VEC3)                                                                   \
    X(SetLinVel, Vec3, SCHEMA_VEC3)                                                                \
    X(SetAngVel, Vec3, SCHEMA_VEC3)                                                                \
    X(ImpulseAt, ImpAt, SCHEMA_IMPULSE_AT)                                                         \
    X(HOnly, HOnly, SCHEMA_HANDLE_ONLY)                                                            \
    X(Destroy, HOnly, SCHEMA_HANDLE_ONLY)                                                          \
    X(Activate, HOnly, SCHEMA_HANDLE_ONLY)                                                         \
    X(Gravity, XYZ, SCHEMA_XYZ)                                                                    \
    X(SetPos, SetPos, SCHEMA_SET_POS)                                                              \
    X(Buoy, Buoy, SCHEMA_BUOYANCY)                                                                 \
    X(BatchBuoy, BatchBuoy, SCHEMA_BATCH_BUOYANCY)                                                 \
    X(Mesh, Mesh, SCHEMA_MESH)                                                                     \
    X(SetTrns, SetTrns, SCHEMA_SET_TRNS)                                                           \
    X(CCD, CCD, SCHEMA_CCD)                                                                        \
    X(ConvexHull, HC, SCHEMA_HC_HULL)                                                              \
    X(Compound, HC, SCHEMA_HC_COMP)                                                                \
    X(BatchCreate, BatchCreate, SCHEMA_BATCH_CREATE)                                               \
    X(BatchDestroy, BatchDestroy, SCHEMA_BATCH_DESTROY)                                            \
    X(SetRot, SetRot, SCHEMA_SET_ROT)                                                              \
    X(Render, Render, SCHEMA_RENDER)                                                               \
    X(Raycast, Raycast, SCHEMA_RAYCAST)                                                            \
    X(RayBatch, RayBatch, SCHEMA_RAYCAST_BATCH)                                                    \
    X(Shapecast, Shapecast, SCHEMA_SHAPECAST)                                                      \
    X(OverlapSphere, OverlapSphere, SCHEMA_OVERLAP_SPHERE)                                         \
    X(OverlapAABB, OverlapAABB, SCHEMA_OVERLAP_AABB)                                               \
    X(SetUserData, SetUserData, SCHEMA_SET_USER_DATA)                                              \
    X(GetUserData, HOnly, SCHEMA_HANDLE_ONLY)                                                      \
    X(GetMotion, HOnly, SCHEMA_HANDLE_ONLY)                                                        \
    X(SetMotion, SetMotion, SCHEMA_SET_MOTION)                                                     \
    X(ColFilter, ColFilter, SCHEMA_COL_FILTER)                                                     \
    X(RegMat, RegMat, SCHEMA_REG_MAT)                                                              \
    X(SetConstrTarget, SetConstr, SCHEMA_SET_CONSTR_TARGET)                                        \
    X(Heightfield, Heightfield, SCHEMA_HEIGHTFIELD)                                                \
    X(DebugData, DebugData, SCHEMA_DEBUG_DATA)                                                     \
    X(CreateConstr, CreateConstr, SCHEMA_CREATE_CONSTR)                                            \
    X(DestroyConstr, HOnly, SCHEMA_HANDLE_ONLY)                                                    \
    X(Step, Step, SCHEMA_STEP)                                                                     \
    X(CharMove, CharMove, SCHEMA_CHAR_MOVE)                                                        \
    X(LoadState, LoadState, SCHEMA_LOAD_STATE)                                                     \
    X(CreateChar, CreateChar, SCHEMA_CREATE_CHAR)                                                  \
    X(SetPosChar, SetPosChar, SCHEMA_SET_POS_CHAR)                                                 \
    X(SetRotChar, SetRotChar, SCHEMA_SET_ROT_CHAR)                                                 \
    X(SetStrengthChar, SetStrengthChar, SCHEMA_SET_STRENGTH_CHAR)                                  \
    X(VehicleInput, VehicleInput, SCHEMA_VEHICLE_INPUT)                                            \
    X(TankInput, TankInput, SCHEMA_TANK_INPUT)                                                     \
    X(CreateVehicle, CreateVehicle, SCHEMA_CREATE_VEHICLE)                                         \
    X(CreateTracked, CreateTracked, SCHEMA_CREATE_TRACKED)                                         \
    X(CreateRagdoll, CreateRagdoll, SCHEMA_CREATE_RAGDOLL)                                         \
    X(RagdollSettings, RagdollSettings, SCHEMA_RAGDOLL_SETTINGS)                                   \
    X(RagdollAddPart, RagdollAddPart, SCHEMA_RAGDOLL_ADD_PART)                                     \
    X(AddJoint, AddJoint, SCHEMA_ADD_JOINT)                                                        \
    X(GetJointIdx, GetJointIdx, SCHEMA_GET_JOINT_IDX)                                              \
    X(RagdollDrive, RagdollDrive, SCHEMA_RAGDOLL_DRIVE)                                            \
    X(SbssCreateConstraints, SbssCreateConstraints, SCHEMA_SBSS_CREATE_CONSTRAINTS)                \
    X(CreateSoftBody, CreateSoftBody, SCHEMA_CREATE_SOFT_BODY)                                     \
    X(SbssAddVertex, SbssAddVertex, SCHEMA_SBSS_ADD_VERTEX)                                        \
    X(SbssAddFace, SbssAddFace, SCHEMA_SBSS_ADD_FACE)                                              \
    X(SbssAddVertices, SbssAddVertices, SCHEMA_SBSS_ADD_VERTICES)                                  \
    X(SbssAddFaces, SbssAddFaces, SCHEMA_SBSS_ADD_FACES)                                           \
    X(GetSbVertex, GetSbVertex, SCHEMA_GET_SB_VERTEX)                                              \
    X(RegEntityOnly, RegEntityOnly, SCHEMA_REG_ENTITY_ONLY)                                        \
    X(RegCompOnly, RegCompOnly, SCHEMA_REG_COMP_ONLY)                                              \
    X(RegRegComp, RegRegComp, SCHEMA_REG_REG_COMP)                                                 \
    X(RegAdd, RegAdd, SCHEMA_REG_ADD)                                                              \
    X(RegEntComp, RegEntComp, SCHEMA_REG_ENT_COMP)                                                 \
    X(RegSyncPhys, RegSyncPhys, SCHEMA_REG_SYNC_PHYS)                                              \
    X(StressTest, StressTest, SCHEMA_STRESS_TEST)                                                  \
    X(CreateShip, CreateShip, SCHEMA_CREATE_SHIP)                                                  \
    X(ShipInput, ShipInput, SCHEMA_SHIP_INPUT)

#define FOR_ALL_MATH_PARSERS(X)                                                                    \
    X(MathPersp, MathPersp, SCHEMA_MATH_PERSPECTIVE)                                               \
    X(MathOrtho, MathOrtho, SCHEMA_MATH_ORTHO)                                                     \
    X(MathTrio, MathTrio, SCHEMA_MATH_TRIO)                                                        \
    X(MathMat, MathMat, SCHEMA_MATH_MAT)                                                           \
    X(MathMatPair, MathMatPair, SCHEMA_MATH_MAT_PAIR)                                              \
    X(MathMatVec, MathMatVec, SCHEMA_MATH_MAT_VEC)                                                 \
    X(MathMatBatch, MathMatBatch, SCHEMA_MATH_MAT_BATCH)                                           \
    X(MathCull, MathCull, SCHEMA_MATH_CULL)                                                        \
    X(MathCullBatch, MathCullBatch, SCHEMA_MATH_CULL_BATCH)                                        \
    X(MathVec3Batch, MathVec3Batch, SCHEMA_MATH_VEC3_BATCH)                                        \
    X(MathEuler, MathEuler, SCHEMA_MATH_EULER)                                                     \
    X(MathQuat, MathQuat, SCHEMA_MATH_QUAT)                                                        \
    X(MathSlerp, MathSlerp, SCHEMA_MATH_SLERP)                                                     \
    X(MathQuatPair, MathQuatPair, SCHEMA_MATH_QUAT_PAIR)                                           \
    X(MathLerpBatch, MathLerpBatch, SCHEMA_MATH_LERP_BATCH)                                        \
    X(MathQuatVec, MathQuatVec, SCHEMA_MATH_QUAT_VEC)                                              \
    X(MathQuatVecBatch, MathQuatVecBatch, SCHEMA_MATH_QUAT_VEC_BATCH)                              \
    X(MathQuatOp, MathQuatOp, SCHEMA_MATH_QUAT_OP)                                                 \
    X(MathProject, MathProject, SCHEMA_MATH_PROJECT)                                               \
    X(MathUnproject, MathUnproject, SCHEMA_MATH_UNPROJECT)                                         \
    X(MathVecPair, MathVecPair, SCHEMA_MATH_VEC_PAIR)                                              \
    X(MathRayPlane, MathRayPlane, SCHEMA_MATH_RAY_PLANE)                                           \
    X(MathAxisAngle, MathAxisAngle, SCHEMA_MATH_AXIS_ANGLE)                                        \
    X(MathDistBatch, MathDistBatch, SCHEMA_MATH_DIST_BATCH)                                        \
    X(MathVecOp, MathVecOp, SCHEMA_MATH_VEC_OP)                                                    \
    X(MathReflect, MathReflect, SCHEMA_MATH_REFLECT)

#define MAP_TO_DECLARE(P, G, S) DECLARE_PARSER(P, G)

// B. Declare the Parsers
typedef struct CulverinParsers {
    FOR_ALL_PARSERS(MAP_TO_DECLARE)
    FastParser *registry[PARSER_REGISTRY_SIZE];
    size_t registry_count;
} CulverinParsers;

typedef struct MathParsers {
    FOR_ALL_MATH_PARSERS(MAP_TO_DECLARE)
    FastParser *registry[PARSER_REGISTRY_SIZE];
    size_t registry_count;
} MathParsers;

void fp_dump_schemas_json(CulverinParsers *cp, FILE *out);
void culverin_init_all_parsers(CulverinParsers *cp);
void culverin_free_all_parsers(CulverinParsers *cp);