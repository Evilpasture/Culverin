#pragma once

#include "bindings/parsing/FastParse.h"
#include "core/Types.h"

typedef uint64_t ParseBodyHandle;
static constexpr size_t kParserRegistrySize = 128;

/**
 * SCHEMA DEFINITIONS
 * Format: X(IndexName, PythonName, C-Type, IsRequired)
 */

#define CPH_SCHEMA_WORLD_INIT(X)                                                                       \
    X(IDX_SETTINGS, "settings", PyObject *, false)                                                 \
    X(IDX_BODIES, "bodies", PyObject *, false)

#define CPH_SCHEMA_BODY(X)                                                                             \
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
    X(IDX_CCD, "ccd", bool, false)                                                                 \
    X(IDX_ALLOWED_DOFS, "allowed_dofs", uint32_t, false)

#define CPH_SCHEMA_SET_POS(X)                                                                          \
    X(IDX_SETPOS_HANDLE, "handle", ParseBodyHandle, true)                                          \
    X(IDX_SETPOS_X, "x", JPH_Real, true)                                                           \
    X(IDX_SETPOS_Y, "y", JPH_Real, true)                                                           \
    X(IDX_SETPOS_Z, "z", JPH_Real, true)

// Shared by Impulse, AngImpulse, Force, and Torque
#define CPH_SCHEMA_VEC3(X)                                                                             \
    X(IDX_V3_H, "handle", ParseBodyHandle, true)                                                   \
    X(IDX_V3_X, "x", float, true)                                                                  \
    X(IDX_V3_Y, "y", float, true)                                                                  \
    X(IDX_V3_Z, "z", float, true)

#define CPH_SCHEMA_IMPULSE_AT(X)                                                                       \
    X(IDX_IMPAT_H, "handle", ParseBodyHandle, true)                                                \
    X(IDX_IMPAT_IX, "ix", float, true)                                                             \
    X(IDX_IMPAT_IY, "iy", float, true)                                                             \
    X(IDX_IMPAT_IZ, "iz", float, true)                                                             \
    X(IDX_IMPAT_PX, "px", JPH_Real, true)                                                          \
    X(IDX_IMPAT_PY, "py", JPH_Real, true)                                                          \
    X(IDX_IMPAT_PZ, "pz", JPH_Real, true)

#define CPH_SCHEMA_BUOYANCY(X)                                                                         \
    X(IDX_BUOY_HANDLE, "handle", ParseBodyHandle, true)                                            \
    X(IDX_BUOY_SURFACE_Y, "surface_y", double, true)                                               \
    X(IDX_BUOY_BUOYANCY, "buoyancy", float, false)                                                 \
    X(IDX_BUOY_LIN_DRAG, "linear_drag", float, false)                                              \
    X(IDX_BUOY_ANG_DRAG, "angular_drag", float, false)                                             \
    X(IDX_BUOY_DT, "dt", float, false)                                                             \
    X(IDX_BUOY_VEL, "fluid_velocity", PyObject *, false)

#define CPH_SCHEMA_HULL_OR_COMP(X)                                                                     \
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

#define CPH_SCHEMA_MESH(X)                                                                             \
    X(IDX_MSH_POS, "pos", PyObject *, true)                                                        \
    X(IDX_MSH_ROT, "rot", PyObject *, true)                                                        \
    X(IDX_MSH_VERTS, "vertices", PyObject *, true)                                                 \
    X(IDX_MSH_INDICES, "indices", PyObject *, true)                                                \
    X(IDX_MSH_USER_DATA, "user_data", uint64_t, false)                                             \
    X(IDX_MSH_CAT, "category", uint32_t, false)                                                    \
    X(IDX_MSH_MASK, "mask", uint32_t, false)

#define CPH_SCHEMA_HANDLE_ONLY(X) X(IDX_H_H, "handle", ParseBodyHandle, true)

#define CPH_SCHEMA_SET_TRNS(X)                                                                         \
    X(IDX_ST_HANDLE, "handle", ParseBodyHandle, true)                                              \
    X(IDX_ST_POS, "pos", PyObject *, true)                                                         \
    X(IDX_ST_ROT, "rot", PyObject *, true)

#define CPH_SCHEMA_CCD(X)                                                                              \
    X(IDX_CCD_H, "handle", ParseBodyHandle, true)                                                  \
    X(IDX_CCD_E, "enabled", bool, true)

#define CPH_SCHEMA_XYZ(X)                                                                              \
    X(IDX_XYZ_X, "x", float, true)                                                                 \
    X(IDX_XYZ_Y, "y", float, true)                                                                 \
    X(IDX_XYZ_Z, "z", float, true)

#define CPH_SCHEMA_BATCH_BUOYANCY(X)                                                                   \
    X(IDX_BBUOY_HANDLES, "handles", PyObject *, true)                                              \
    X(IDX_BBUOY_SURFACE_Y, "surface_y", double, true)                                              \
    X(IDX_BBUOY_BUOYANCY, "buoyancy", float, false)                                                \
    X(IDX_BBUOY_LIN_DRAG, "linear_drag", float, false)                                             \
    X(IDX_BBUOY_ANG_DRAG, "angular_drag", float, false)                                            \
    X(IDX_BBUOY_DT, "dt", float, false)                                                            \
    X(IDX_BBUOY_VEL, "fluid_velocity", PyObject *, false)

// This is the "Master" structural schema for Hull/Compound/Body-like objects
#define CPH_SCHEMA_HC_MASTER(X)                                                                        \
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
#define CPH_SCHEMA_HC_HULL(X)                                                                          \
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

#define CPH_SCHEMA_HC_COMP(X)                                                                          \
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

#define CPH_SCHEMA_BATCH_CREATE(X)                                                                     \
    X(IDX_BC_POSITIONS, "positions", PyObject *, true)                                             \
    X(IDX_BC_SIZES, "sizes", PyObject *, true)                                                     \
    X(IDX_BC_SHAPE, "shape_type", int, false)                                                      \
    X(IDX_BC_MOTION, "motion_type", int, false)

#define CPH_SCHEMA_BATCH_DESTROY(X) X(IDX_BD_HANDLES, "handles", PyObject *, true)

#define CPH_SCHEMA_SET_ROT(X)                                                                          \
    X(IDX_SETROT_H, "handle", ParseBodyHandle, true)                                               \
    X(IDX_SETROT_X, "x", float, true)                                                              \
    X(IDX_SETROT_Y, "y", float, true)                                                              \
    X(IDX_SETROT_Z, "z", float, true)                                                              \
    X(IDX_SETROT_W, "w", float, true)

#define CPH_SCHEMA_RENDER(X) X(IDX_RND_ALPHA, "alpha", float, true)

#define CPH_SCHEMA_RAYCAST(X)                                                                          \
    X(IDX_RAY_START, "start", PyObject *, true)                                                    \
    X(IDX_RAY_DIR, "direction", PyObject *, true)                                                  \
    X(IDX_RAY_DIST, "max_dist", float, false)                                                      \
    X(IDX_RAY_IGN, "ignore", ParseBodyHandle, false)

#define CPH_SCHEMA_RAYCAST_BATCH(X)                                                                    \
    X(IDX_RB_STARTS, "starts", PyObject *, true)                                                   \
    X(IDX_RB_DIRS, "directions", PyObject *, true)                                                 \
    X(IDX_RB_DIST, "max_dist", float, false)

#define CPH_SCHEMA_SHAPECAST(X)                                                                        \
    X(IDX_SC_SHAPE, "shape", int, true)                                                            \
    X(IDX_SC_POS, "pos", PyObject *, true)                                                         \
    X(IDX_SC_ROT, "rot", PyObject *, true)                                                         \
    X(IDX_SC_DIR, "dir", PyObject *, true)                                                         \
    X(IDX_SC_SIZE, "size", PyObject *, false)                                                      \
    X(IDX_SC_IGNORE, "ignore", ParseBodyHandle, false)

#define CPH_SCHEMA_OVERLAP_SPHERE(X)                                                                   \
    X(IDX_OS_CENTER, "center", PyObject *, true)                                                   \
    X(IDX_OS_RADIUS, "radius", float, true)

#define CPH_SCHEMA_OVERLAP_AABB(X)                                                                     \
    X(IDX_OA_MIN, "min", PyObject *, true)                                                         \
    X(IDX_OA_MAX, "max", PyObject *, true)

#define CPH_SCHEMA_SET_USER_DATA(X)                                                                    \
    X(IDX_SUD_H, "handle", ParseBodyHandle, true)                                                  \
    X(IDX_SUD_D, "data", uint64_t, true)

#define CPH_SCHEMA_SET_MOTION(X)                                                                       \
    X(IDX_SM_H, "handle", ParseBodyHandle, true)                                                   \
    X(IDX_SM_M, "motion", int, true)

#define CPH_SCHEMA_COL_FILTER(X)                                                                       \
    X(IDX_CF_H, "handle", ParseBodyHandle, true)                                                   \
    X(IDX_CF_C, "category", uint32_t, true)                                                        \
    X(IDX_CF_M, "mask", uint32_t, true)

#define CPH_SCHEMA_REG_MAT(X)                                                                          \
    X(IDX_RM_ID, "id", uint32_t, true)                                                             \
    X(IDX_RM_FRIC, "friction", float, false)                                                       \
    X(IDX_RM_REST, "restitution", float, false)

#define CPH_SCHEMA_SET_CONSTR_TARGET(X)                                                                \
    X(IDX_SCT_H, "handle", uint64_t, true)                                                         \
    X(IDX_SCT_T, "target", float, true)

#define CPH_SCHEMA_HEIGHTFIELD(X)                                                                      \
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

#define CPH_SCHEMA_DEBUG_DATA(X)                                                                       \
    X(IDX_DD_SHAPES, "shapes", bool, false)                                                        \
    X(IDX_DD_CONSTRAINTS, "constraints", bool, false)                                              \
    X(IDX_DD_BBOX, "bbox", bool, false)                                                            \
    X(IDX_DD_CENTERS, "centers", bool, false)                                                      \
    X(IDX_DD_WIREFRAME, "wireframe", bool, false)

#define CPH_SCHEMA_CREATE_CONSTR(X)                                                                    \
    X(IDX_CC_TYPE, "type", int, true)                                                              \
    X(IDX_CC_BODY1, "body1", ParseBodyHandle, true)                                                \
    X(IDX_CC_BODY2, "body2", ParseBodyHandle, true)                                                \
    X(IDX_CC_PARAMS, "params", PyObject *, false)                                                  \
    X(IDX_CC_MOTOR, "motor", PyObject *, false)

#define CPH_SCHEMA_STEP(X) X(IDX_STEP_DT, "dt", float, false) // Optional (0)

#define CPH_SCHEMA_CHAR_MOVE(X)                                                                        \
    X(IDX_CM_VEL, "velocity", CPH_Vec3f, true)                                                         \
    X(IDX_CM_DT, "dt", float, true)

#define CPH_SCHEMA_LOAD_STATE(X) X(IDX_LS_STATE, "state", PyObject *, true)

#define CPH_SCHEMA_CREATE_CHAR(X)                                                                      \
    X(IDX_CCHAR_POS, "pos", CPH_PosStride, true)                                                       \
    X(IDX_CCHAR_H, "height", float, false)                                                         \
    X(IDX_CCHAR_R, "radius", float, false)                                                         \
    X(IDX_CCHAR_STEP, "step_height", float, false)                                                 \
    X(IDX_CCHAR_SLOPE, "max_slope", float, false)

#define CPH_SCHEMA_SET_POS_CHAR(X) X(IDX_SPC_POS, "pos", CPH_PosStride, true)

#define CPH_SCHEMA_SET_ROT_CHAR(X) X(IDX_SRC_ROT, "rot", CPH_AuxStride, true)

#define CPH_SCHEMA_SET_STRENGTH_CHAR(X) X(IDX_SSC_STRENGTH, "strength", float, true)

#define CPH_SCHEMA_VEHICLE_INPUT(X)                                                                    \
    X(IDX_VI_FWD, "forward", float, false)                                                         \
    X(IDX_VI_RIGHT, "right", float, false)                                                         \
    X(IDX_VI_BRAKE, "brake", float, false)                                                         \
    X(IDX_VI_HAND, "handbrake", float, false)

#define CPH_SCHEMA_WHEEL_IDX(X) X(IDX_WH_INDEX, "index", uint32_t, true)

#define CPH_SCHEMA_TANK_INPUT(X)                                                                       \
    X(IDX_TI_LEFT, "left", float, true)                                                            \
    X(IDX_TI_RIGHT, "right", float, true)                                                          \
    X(IDX_TI_BRAKE, "brake", float, false)

#define CPH_SCHEMA_CREATE_VEHICLE(X)                                                                   \
    X(IDX_CV_CHASSIS, "chassis", uint64_t, true)                                                   \
    X(IDX_CV_WHEELS, "wheels", PyObject *, true)                                                   \
    X(IDX_CV_DRIVE, "drive", PyObject *, false)                                                    \
    X(IDX_CV_ENGINE, "engine", PyObject *, false)                                                  \
    X(IDX_CV_TRANS, "transmission", PyObject *, false)

#define CPH_SCHEMA_CREATE_TRACKED(X)                                                                   \
    X(IDX_CT_CHASSIS, "chassis", uint64_t, true)                                                   \
    X(IDX_CT_WHEELS, "wheels", PyObject *, true)                                                   \
    X(IDX_CT_TRACKS, "tracks", PyObject *, true)                                                   \
    X(IDX_CT_TORQUE, "max_torque", float, false)                                                   \
    X(IDX_CT_RPM, "max_rpm", float, false)

#define CPH_SCHEMA_CREATE_RAGDOLL(X)                                                                   \
    X(IDX_CR_SETTINGS, "settings", PyObject *, true)                                               \
    X(IDX_CR_POS, "pos", CPH_PosStride, true)                                                          \
    X(IDX_CR_ROT, "rot", CPH_AuxStride, false)                                                         \
    X(IDX_CR_USER, "user_data", uint64_t, false)                                                   \
    X(IDX_CR_CAT, "category", uint32_t, false)                                                     \
    X(IDX_CR_MASK, "mask", uint32_t, false)                                                        \
    X(IDX_CR_MAT, "material_id", uint32_t, false)

#define CPH_SCHEMA_RAGDOLL_SETTINGS(X) X(IDX_RS_SKELETON, "skeleton", PyObject *, true)

#define CPH_SCHEMA_RAGDOLL_ADD_PART(X)                                                                 \
    X(IDX_RAP_JOINT, "joint_index", int, true)                                                     \
    X(IDX_RAP_SHAPE, "shape_type", int, true)                                                      \
    X(IDX_RAP_SIZE, "size", PyObject *, true)                                                      \
    X(IDX_RAP_MASS, "mass", float, false)                                                          \
    X(IDX_RAP_PARENT, "parent_index", int, false)                                                  \
    X(IDX_RAP_TWIST_MIN, "twist_min", float, false)                                                \
    X(IDX_RAP_TWIST_MAX, "twist_max", float, false)                                                \
    X(IDX_RAP_CONE, "cone_angle", float, false)                                                    \
    X(IDX_RAP_AXIS, "axis", CPH_Vec3f, false)                                                          \
    X(IDX_RAP_NORMAL, "normal", CPH_Vec3f, false)                                                      \
    X(IDX_RAP_POS, "pos", PyObject *, false)

#define CPH_SCHEMA_ADD_JOINT(X)                                                                        \
    X(IDX_AJ_NAME, "name", PyObject *, true)                                                       \
    X(IDX_AJ_PARENT, "parent_index", int, false)

#define CPH_SCHEMA_GET_JOINT_IDX(X) X(IDX_GJI_NAME, "name", PyObject *, true)

#define CPH_SCHEMA_RAGDOLL_DRIVE(X)                                                                    \
    X(IDX_RD_POS, "root_pos", CPH_PosStride, true)                                                     \
    X(IDX_RD_ROT, "root_rot", CPH_AuxStride, true)                                                     \
    X(IDX_RD_MATS, "matrices", PyObject *, true)

#define CPH_SCHEMA_SBSS_CREATE_CONSTRAINTS(X)                                                          \
    X(IDX_SCC_COMP, "compliance", float, true)                                                     \
    X(IDX_SCC_BEND, "bend_type", int, false)

#define CPH_SCHEMA_SBSS_ADD_VERTICES(X)                                                                \
    X(IDX_SAVS_POS, "positions", PyObject *, true)                                                 \
    X(IDX_SAVS_MASS, "inv_masses", PyObject *, false)                                              \
    X(IDX_SAVS_VEL, "velocities", PyObject *, false)

#define CPH_SCHEMA_SBSS_ADD_FACES(X)                                                                   \
    X(IDX_SAFS_IND, "indices", PyObject *, true)                                                   \
    X(IDX_SAFS_MAT, "materials", PyObject *, false)

#define CPH_SCHEMA_GET_SB_VERTEX(X)                                                                    \
    X(IDX_GSBV_H, "handle", ParseBodyHandle, true)                                                 \
    X(IDX_GSBV_I, "index", uint32_t, true)

#define CPH_SCHEMA_CREATE_SOFT_BODY(X)                                                                 \
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

#define CPH_SCHEMA_SBSS_ADD_VERTEX(X)                                                                  \
    X(IDX_SAV_POS, "pos", PyObject *, true)                                                        \
    X(IDX_SAV_MASS, "inv_mass", float, false)                                                      \
    X(IDX_SAV_VEL, "velocity", PyObject *, false)

#define CPH_SCHEMA_SBSS_ADD_FACE(X)                                                                    \
    X(IDX_SAF_V1, "v1", uint32_t, true)                                                            \
    X(IDX_SAF_V2, "v2", uint32_t, true)                                                            \
    X(IDX_SAF_V3, "v3", uint32_t, true)                                                            \
    X(IDX_SAF_MAT, "material_index", uint32_t, false) // Add this line

#define CPH_SCHEMA_REG_ENTITY_ONLY(X) X(IDX_REO_ENT, "entity", uint64_t, true)

#define CPH_SCHEMA_REG_COMP_ONLY(X) X(IDX_RCO_COMP, "comp_id", uint32_t, true)

#define CPH_SCHEMA_REG_REG_COMP(X) X(IDX_RRC_SIZE, "size_bytes", uint32_t, true)

#define CPH_SCHEMA_REG_ADD(X)                                                                          \
    X(IDX_RA_ENT, "entity", uint64_t, true)                                                        \
    X(IDX_RA_COMP, "comp_id", uint32_t, true)                                                      \
    X(IDX_RA_DATA, "data", PyObject *, false)

#define CPH_SCHEMA_REG_ENT_COMP(X)                                                                     \
    X(IDX_REC_ENT, "entity", uint64_t, true)                                                       \
    X(IDX_REC_COMP, "comp_id", uint32_t, true)

#define CPH_SCHEMA_REG_SYNC_PHYS(X)                                                                    \
    X(IDX_RSP_WORLD, "world", PyObject *, true)                                                    \
    X(IDX_RSP_H_COMP, "handle_comp_id", uint32_t, true)                                            \
    X(IDX_RSP_T_COMP, "transform_comp_id", uint32_t, true)                                         \
    X(IDX_RSP_R_COMP, "rot_comp_id", int, false)

#define CPH_SCHEMA_MATH_PERSPECTIVE(X)                                                                 \
    X(IDX_MP_FOVY, "fovy", float, true)                                                            \
    X(IDX_MP_ASPECT, "aspect", float, true)                                                        \
    X(IDX_MP_NEAR, "near", float, true)                                                            \
    X(IDX_MP_FAR, "far", float, true)

#define CPH_SCHEMA_MATH_ORTHO(X)                                                                       \
    X(IDX_MO_LEFT, "left", float, true)                                                            \
    X(IDX_MO_RIGHT, "right", float, true)                                                          \
    X(IDX_MO_BOTTOM, "bottom", float, true)                                                        \
    X(IDX_MO_TOP, "top", float, true)                                                              \
    X(IDX_MO_NEAR, "near", float, true)                                                            \
    X(IDX_MO_FAR, "far", float, true)

#define CPH_SCHEMA_MATH_LOOKAT(X)                                                                      \
    X(IDX_ML_EYE, "eye", PyObject *, true)                                                         \
    X(IDX_ML_TARGET, "target", PyObject *, true)                                                   \
    X(IDX_ML_UP, "up", PyObject *, true)

#define CPH_SCHEMA_MATH_TRS(X)                                                                         \
    X(IDX_MTRS_T, "translation", PyObject *, true)                                                 \
    X(IDX_MTRS_R, "rotation", PyObject *, true)                                                    \
    X(IDX_MTRS_S, "scale", PyObject *, true)

#define CPH_SCHEMA_MATH_TRS_BATCH(X)                                                                   \
    X(IDX_MTRSB_T, "translations", PyObject *, true)                                               \
    X(IDX_MTRSB_R, "rotations", PyObject *, true)                                                  \
    X(IDX_MTRSB_S, "scales", PyObject *, true)

#define CPH_SCHEMA_MATH_MAT(X) X(IDX_MMM_MAT, "mat", PyObject *, true)
#define CPH_SCHEMA_MATH_MAT_PAIR(X)                                                                    \
    X(IDX_MMP_A, "a", PyObject *, true) X(IDX_MMP_B, "b", PyObject *, true)
#define CPH_SCHEMA_MATH_MAT_VEC(X)                                                                     \
    X(IDX_MMV_MAT, "mat", PyObject *, true) X(IDX_MMV_VEC, "vec", PyObject *, true)
#define CPH_SCHEMA_MATH_MAT_BATCH(X)                                                                   \
    X(IDX_MMB_MAT, "mat", PyObject *, true) X(IDX_MMB_BATCH, "batch", PyObject *, true)
#define CPH_SCHEMA_MATH_CULL(X)                                                                        \
    X(IDX_MC_VP, "vp", PyObject *, true)                                                           \
    X(IDX_MC_MIN, "min", PyObject *, true)                                                         \
    X(IDX_MC_MAX, "max", PyObject *, true)

#define CPH_SCHEMA_MATH_CULL_BATCH(X)                                                                  \
    X(IDX_MCB_VP, "vp", PyObject *, true)                                                          \
    X(IDX_MCB_AABBS, "aabbs", PyObject *, true)

#define CPH_SCHEMA_MATH_VEC3_BATCH(X) X(IDX_MVB_VECS, "vecs", PyObject *, true)

#define CPH_SCHEMA_MATH_EULER(X)                                                                       \
    X(IDX_ME_X, "x", float, true)                                                                  \
    X(IDX_ME_Y, "y", float, true)                                                                  \
    X(IDX_ME_Z, "z", float, true)

#define CPH_SCHEMA_MATH_QUAT(X)                                                                        \
    X(IDX_MQ_X, "x", float, true)                                                                  \
    X(IDX_MQ_Y, "y", float, true)                                                                  \
    X(IDX_MQ_Z, "z", float, true)                                                                  \
    X(IDX_MQ_W, "w", float, true)

#define CPH_SCHEMA_MATH_SLERP(X)                                                                       \
    X(IDX_MS_Q1, "q1", PyObject *, true)                                                           \
    X(IDX_MS_Q2, "q2", PyObject *, true)                                                           \
    X(IDX_MS_T, "t", float, true)

#define CPH_SCHEMA_MATH_QUAT_PAIR(X)                                                                   \
    X(IDX_MQP_A, "a", PyObject *, true)                                                            \
    X(IDX_MQP_B, "b", PyObject *, true)

#define CPH_SCHEMA_MATH_LERP_BATCH(X)                                                                  \
    X(IDX_MLB_VECS_A, "vecs_a", PyObject *, true)                                                  \
    X(IDX_MLB_VECS_B, "vecs_b", PyObject *, true)                                                  \
    X(IDX_MLB_ALPHA, "alpha", float, true)

#define CPH_SCHEMA_MATH_QUAT_VEC(X)                                                                    \
    X(IDX_MQV_Q, "q", PyObject *, true)                                                            \
    X(IDX_MQV_V, "v", PyObject *, true)

#define CPH_SCHEMA_MATH_QUAT_VEC_BATCH(X)                                                              \
    X(IDX_MQVB_Q, "q", PyObject *, true)                                                           \
    X(IDX_MQVB_VECS, "vecs", PyObject *, true)

#define CPH_SCHEMA_MATH_QUAT_OP(X) X(IDX_MQO_Q, "q", PyObject *, true)

#define CPH_SCHEMA_MATH_PROJECT(X)                                                                     \
    X(IDX_MPR_V, "v", PyObject *, true)                                                            \
    X(IDX_MPR_MVP, "mvp", PyObject *, true)                                                        \
    X(IDX_MPR_VP, "viewport", PyObject *, true)

#define CPH_SCHEMA_MATH_UNPROJECT(X)                                                                   \
    X(IDX_MUP_V, "v", PyObject *, true)                                                            \
    X(IDX_MUP_MVP, "mvp", PyObject *, true)                                                        \
    X(IDX_MUP_VP, "viewport", PyObject *, true)

#define CPH_SCHEMA_MATH_VEC_PAIR(X)                                                                    \
    X(IDX_MVP_V1, "v1", PyObject *, true)                                                          \
    X(IDX_MVP_V2, "v2", PyObject *, true)

#define CPH_SCHEMA_MATH_RAY_PLANE(X)                                                                   \
    X(IDX_RP_RO, "ray_origin", PyObject *, true)                                                   \
    X(IDX_RP_RD, "ray_dir", PyObject *, true)                                                      \
    X(IDX_RP_PO, "plane_pos", PyObject *, true)                                                    \
    X(IDX_RP_PN, "plane_norm", PyObject *, true)

#define CPH_SCHEMA_MATH_AXIS_ANGLE(X)                                                                  \
    X(IDX_MAA_AXIS, "axis", PyObject *, true)                                                      \
    X(IDX_MAA_ANGLE, "angle", float, true)

#define CPH_SCHEMA_MATH_DIST_BATCH(X)                                                                  \
    X(IDX_MDB_VECS_A, "vecs_a", PyObject *, true)                                                  \
    X(IDX_MDB_VECS_B, "vecs_b", PyObject *, true)

#define CPH_SCHEMA_MATH_VEC_OP(X) X(IDX_MVO_V, "v", PyObject *, true)

#define CPH_SCHEMA_MATH_REFLECT(X)                                                                     \
    X(IDX_MRF_V, "v", PyObject *, true)                                                            \
    X(IDX_MRF_N, "normal", PyObject *, true)

#define CPH_SCHEMA_MATH_EULER_VEC(X) X(IDX_MEV_V, "euler", PyObject *, true)
#define CPH_SCHEMA_MATH_EULER_BATCH(X) X(IDX_MEB_VECS, "eulers", PyObject *, true)

#define CPH_SCHEMA_CREATE_SHIP(X)                                                                      \
    X(IDX_CS_SLED, "sled", ParseBodyHandle, true)                                                  \
    X(IDX_CS_KP, "kp", float, true)                                                                \
    X(IDX_CS_KD, "kd", float, true)                                                                \
    X(IDX_CS_THROTTLE, "throttle_force", float, true)                                              \
    X(IDX_CS_STEER, "steer_speed", float, true)                                                    \
    X(IDX_CS_BANKING, "banking", float, false)                                                     \
    X(IDX_CS_GRIP, "lateral_grip", float, false)                                                   \
    X(IDX_CS_DRAG, "linear_drag", float, false)

#define CPH_SCHEMA_SHIP_INPUT(X)                                                                       \
    X(IDX_SI_FWD, "forward", float, true)                                                          \
    X(IDX_SI_RIGHT, "right", float, true)

#define CPH_SCHEMA_STRESS_TEST(X)                                                                      \
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
    X(IDX_21, "a21", uint64_t, false)                                                              \
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
    X(IDX_37, "a37", uint64_t, false)                                                              \
    X(IDX_38, "a38", uint64_t, false)                                                              \
    X(IDX_39, "a39", uint64_t, false)                                                              \
    X(IDX_40, "a40", uint64_t, false)                                                              \
    X(IDX_41, "a41", uint64_t, false)                                                              \
    X(IDX_42, "a42", uint64_t, false)                                                              \
    X(IDX_43, "a43", uint64_t, false)                                                              \
    X(IDX_44, "a44", uint64_t, false)                                                              \
    X(IDX_45, "a45", uint64_t, false)                                                              \
    X(IDX_46, "a46", uint64_t, false)                                                              \
    X(IDX_47, "a47", uint64_t, false)                                                              \
    X(IDX_48, "a48", uint64_t, false)                                                              \
    X(IDX_49, "a49", uint64_t, false)                                                              \
    X(IDX_50, "a50", uint64_t, false)                                                              \
    X(IDX_51, "a51", uint64_t, false)                                                              \
    X(IDX_52, "a52", uint64_t, false)                                                              \
    X(IDX_53, "a53", uint64_t, false)                                                              \
    X(IDX_54, "a54", uint64_t, false)                                                              \
    X(IDX_55, "a55", uint64_t, false)                                                              \
    X(IDX_56, "a56", uint64_t, false)                                                              \
    X(IDX_57, "a57", uint64_t, false)                                                              \
    X(IDX_58, "a58", uint64_t, false)                                                              \
    X(IDX_59, "a59", uint64_t, false)                                                              \
    X(IDX_60, "a60", uint64_t, false)                                                              \
    X(IDX_61, "a61", uint64_t, false)                                                              \
    X(IDX_62, "a62", uint64_t, false)                                                              \
    X(IDX_63, "a63", uint64_t, false)

#define CPH_SCHEMA_DUMP_SCHEMA(X) X(IDX_DS_PATH, "path", const char *, false)

/** --- THE GENERATOR ENGINE --- **/

#define CPH_GEN_ENUM(ID, NAME, TYPE, REQ) ID,

// Defines the Enum and the Count for a signature type
#define CPH_DEFINE_INDEX_GROUP(GroupName, Schema)                                                      \
    typedef enum : uint8_t { Schema(CPH_GEN_ENUM) GroupName##_COUNT } GroupName##_Idx;

// Declares a specific parser that uses an Index Group
#define CPH_DECLARE_PARSER(ParserName, GroupName) CPH_FastParser ParserName##Parser;

// A. Define the Index Groups (One per unique signature)
CPH_DEFINE_INDEX_GROUP(WorldInit, CPH_SCHEMA_WORLD_INIT)
CPH_DEFINE_INDEX_GROUP(Body, CPH_SCHEMA_BODY)
CPH_DEFINE_INDEX_GROUP(Vec3, CPH_SCHEMA_VEC3)
CPH_DEFINE_INDEX_GROUP(ImpAt, CPH_SCHEMA_IMPULSE_AT)
CPH_DEFINE_INDEX_GROUP(HOnly, CPH_SCHEMA_HANDLE_ONLY)
CPH_DEFINE_INDEX_GROUP(XYZ, CPH_SCHEMA_XYZ)
CPH_DEFINE_INDEX_GROUP(SetPos, CPH_SCHEMA_SET_POS)
CPH_DEFINE_INDEX_GROUP(Buoy, CPH_SCHEMA_BUOYANCY)
CPH_DEFINE_INDEX_GROUP(BatchBuoy, CPH_SCHEMA_BATCH_BUOYANCY)
CPH_DEFINE_INDEX_GROUP(Mesh, CPH_SCHEMA_MESH)
CPH_DEFINE_INDEX_GROUP(SetTrns, CPH_SCHEMA_SET_TRNS)
CPH_DEFINE_INDEX_GROUP(CCD, CPH_SCHEMA_CCD)
CPH_DEFINE_INDEX_GROUP(HC, CPH_SCHEMA_HC_MASTER) // HC group defined ONLY once
CPH_DEFINE_INDEX_GROUP(BatchCreate, CPH_SCHEMA_BATCH_CREATE)
CPH_DEFINE_INDEX_GROUP(BatchDestroy, CPH_SCHEMA_BATCH_DESTROY)
CPH_DEFINE_INDEX_GROUP(SetRot, CPH_SCHEMA_SET_ROT)
CPH_DEFINE_INDEX_GROUP(Render, CPH_SCHEMA_RENDER)
CPH_DEFINE_INDEX_GROUP(Raycast, CPH_SCHEMA_RAYCAST)
CPH_DEFINE_INDEX_GROUP(RayBatch, CPH_SCHEMA_RAYCAST_BATCH)
CPH_DEFINE_INDEX_GROUP(Shapecast, CPH_SCHEMA_SHAPECAST)
CPH_DEFINE_INDEX_GROUP(OverlapSphere, CPH_SCHEMA_OVERLAP_SPHERE)
CPH_DEFINE_INDEX_GROUP(OverlapAABB, CPH_SCHEMA_OVERLAP_AABB)
CPH_DEFINE_INDEX_GROUP(SetUserData, CPH_SCHEMA_SET_USER_DATA)
CPH_DEFINE_INDEX_GROUP(SetMotion, CPH_SCHEMA_SET_MOTION)
CPH_DEFINE_INDEX_GROUP(ColFilter, CPH_SCHEMA_COL_FILTER)
CPH_DEFINE_INDEX_GROUP(RegMat, CPH_SCHEMA_REG_MAT)
CPH_DEFINE_INDEX_GROUP(SetConstr, CPH_SCHEMA_SET_CONSTR_TARGET)
CPH_DEFINE_INDEX_GROUP(Heightfield, CPH_SCHEMA_HEIGHTFIELD)
CPH_DEFINE_INDEX_GROUP(DebugData, CPH_SCHEMA_DEBUG_DATA)
CPH_DEFINE_INDEX_GROUP(CreateConstr, CPH_SCHEMA_CREATE_CONSTR)
CPH_DEFINE_INDEX_GROUP(Step, CPH_SCHEMA_STEP)
CPH_DEFINE_INDEX_GROUP(CharMove, CPH_SCHEMA_CHAR_MOVE)
CPH_DEFINE_INDEX_GROUP(LoadState, CPH_SCHEMA_LOAD_STATE)
CPH_DEFINE_INDEX_GROUP(CreateChar, CPH_SCHEMA_CREATE_CHAR)
CPH_DEFINE_INDEX_GROUP(SetPosChar, CPH_SCHEMA_SET_POS_CHAR)
CPH_DEFINE_INDEX_GROUP(SetRotChar, CPH_SCHEMA_SET_ROT_CHAR)
CPH_DEFINE_INDEX_GROUP(SetStrengthChar, CPH_SCHEMA_SET_STRENGTH_CHAR)
CPH_DEFINE_INDEX_GROUP(VehicleInput, CPH_SCHEMA_VEHICLE_INPUT)
CPH_DEFINE_INDEX_GROUP(WheelIdx, CPH_SCHEMA_WHEEL_IDX)
CPH_DEFINE_INDEX_GROUP(TankInput, CPH_SCHEMA_TANK_INPUT)
CPH_DEFINE_INDEX_GROUP(CreateVehicle, CPH_SCHEMA_CREATE_VEHICLE)
CPH_DEFINE_INDEX_GROUP(CreateTracked, CPH_SCHEMA_CREATE_TRACKED)
CPH_DEFINE_INDEX_GROUP(CreateRagdoll, CPH_SCHEMA_CREATE_RAGDOLL)
CPH_DEFINE_INDEX_GROUP(RagdollSettings, CPH_SCHEMA_RAGDOLL_SETTINGS)
CPH_DEFINE_INDEX_GROUP(RagdollAddPart, CPH_SCHEMA_RAGDOLL_ADD_PART)
CPH_DEFINE_INDEX_GROUP(AddJoint, CPH_SCHEMA_ADD_JOINT)
CPH_DEFINE_INDEX_GROUP(GetJointIdx, CPH_SCHEMA_GET_JOINT_IDX)
CPH_DEFINE_INDEX_GROUP(RagdollDrive, CPH_SCHEMA_RAGDOLL_DRIVE)
CPH_DEFINE_INDEX_GROUP(SbssCreateConstraints, CPH_SCHEMA_SBSS_CREATE_CONSTRAINTS)
CPH_DEFINE_INDEX_GROUP(CreateSoftBody, CPH_SCHEMA_CREATE_SOFT_BODY)
CPH_DEFINE_INDEX_GROUP(SbssAddVertex, CPH_SCHEMA_SBSS_ADD_VERTEX)
CPH_DEFINE_INDEX_GROUP(SbssAddFace, CPH_SCHEMA_SBSS_ADD_FACE)
CPH_DEFINE_INDEX_GROUP(SbssAddVertices, CPH_SCHEMA_SBSS_ADD_VERTICES)
CPH_DEFINE_INDEX_GROUP(SbssAddFaces, CPH_SCHEMA_SBSS_ADD_FACES)
CPH_DEFINE_INDEX_GROUP(GetSbVertex, CPH_SCHEMA_GET_SB_VERTEX)
CPH_DEFINE_INDEX_GROUP(RegEntityOnly, CPH_SCHEMA_REG_ENTITY_ONLY)
CPH_DEFINE_INDEX_GROUP(RegCompOnly, CPH_SCHEMA_REG_COMP_ONLY)
CPH_DEFINE_INDEX_GROUP(RegRegComp, CPH_SCHEMA_REG_REG_COMP)
CPH_DEFINE_INDEX_GROUP(RegAdd, CPH_SCHEMA_REG_ADD)
CPH_DEFINE_INDEX_GROUP(RegEntComp, CPH_SCHEMA_REG_ENT_COMP)
CPH_DEFINE_INDEX_GROUP(RegSyncPhys, CPH_SCHEMA_REG_SYNC_PHYS)
CPH_DEFINE_INDEX_GROUP(MathPersp, CPH_SCHEMA_MATH_PERSPECTIVE)
CPH_DEFINE_INDEX_GROUP(MathOrtho, CPH_SCHEMA_MATH_ORTHO)
CPH_DEFINE_INDEX_GROUP(MathLookAt, CPH_SCHEMA_MATH_LOOKAT)
CPH_DEFINE_INDEX_GROUP(MathTRS, CPH_SCHEMA_MATH_TRS)
CPH_DEFINE_INDEX_GROUP(MathTRSBatch, CPH_SCHEMA_MATH_TRS_BATCH)
CPH_DEFINE_INDEX_GROUP(MathMat, CPH_SCHEMA_MATH_MAT)
CPH_DEFINE_INDEX_GROUP(MathMatPair, CPH_SCHEMA_MATH_MAT_PAIR)
CPH_DEFINE_INDEX_GROUP(MathMatVec, CPH_SCHEMA_MATH_MAT_VEC)
CPH_DEFINE_INDEX_GROUP(MathMatBatch, CPH_SCHEMA_MATH_MAT_BATCH)
CPH_DEFINE_INDEX_GROUP(MathCull, CPH_SCHEMA_MATH_CULL)
CPH_DEFINE_INDEX_GROUP(MathCullBatch, CPH_SCHEMA_MATH_CULL_BATCH)
CPH_DEFINE_INDEX_GROUP(MathVec3Batch, CPH_SCHEMA_MATH_VEC3_BATCH)
CPH_DEFINE_INDEX_GROUP(MathEuler, CPH_SCHEMA_MATH_EULER)
CPH_DEFINE_INDEX_GROUP(MathQuat, CPH_SCHEMA_MATH_QUAT)
CPH_DEFINE_INDEX_GROUP(MathSlerp, CPH_SCHEMA_MATH_SLERP)
CPH_DEFINE_INDEX_GROUP(MathQuatPair, CPH_SCHEMA_MATH_QUAT_PAIR)
CPH_DEFINE_INDEX_GROUP(MathLerpBatch, CPH_SCHEMA_MATH_LERP_BATCH)
CPH_DEFINE_INDEX_GROUP(MathQuatVec, CPH_SCHEMA_MATH_QUAT_VEC)
CPH_DEFINE_INDEX_GROUP(MathQuatVecBatch, CPH_SCHEMA_MATH_QUAT_VEC_BATCH)
CPH_DEFINE_INDEX_GROUP(MathQuatOp, CPH_SCHEMA_MATH_QUAT_OP)
CPH_DEFINE_INDEX_GROUP(MathProject, CPH_SCHEMA_MATH_PROJECT)
CPH_DEFINE_INDEX_GROUP(MathUnproject, CPH_SCHEMA_MATH_UNPROJECT)
CPH_DEFINE_INDEX_GROUP(MathVecPair, CPH_SCHEMA_MATH_VEC_PAIR)
CPH_DEFINE_INDEX_GROUP(MathRayPlane, CPH_SCHEMA_MATH_RAY_PLANE)
CPH_DEFINE_INDEX_GROUP(MathAxisAngle, CPH_SCHEMA_MATH_AXIS_ANGLE)
CPH_DEFINE_INDEX_GROUP(MathDistBatch, CPH_SCHEMA_MATH_DIST_BATCH)
CPH_DEFINE_INDEX_GROUP(MathVecOp, CPH_SCHEMA_MATH_VEC_OP)
CPH_DEFINE_INDEX_GROUP(MathReflect, CPH_SCHEMA_MATH_REFLECT)
CPH_DEFINE_INDEX_GROUP(MathEulerVec, CPH_SCHEMA_MATH_EULER_VEC)
CPH_DEFINE_INDEX_GROUP(MathEulerBatch, CPH_SCHEMA_MATH_EULER_BATCH)
CPH_DEFINE_INDEX_GROUP(CreateShip, CPH_SCHEMA_CREATE_SHIP)
CPH_DEFINE_INDEX_GROUP(ShipInput, CPH_SCHEMA_SHIP_INPUT)
CPH_DEFINE_INDEX_GROUP(StressTest, CPH_SCHEMA_STRESS_TEST)
CPH_DEFINE_INDEX_GROUP(DumpSchema, CPH_SCHEMA_DUMP_SCHEMA)

#define CPH_FOR_ALL_PARSERS(X)                                                                         \
    X(WorldInit, WorldInit, CPH_SCHEMA_WORLD_INIT)                                                     \
    X(Body, Body, CPH_SCHEMA_BODY)                                                                     \
    X(Impulse, Vec3, CPH_SCHEMA_VEC3)                                                                  \
    X(WheelIdx, WheelIdx, CPH_SCHEMA_WHEEL_IDX)                                                        \
    X(AngImpulse, Vec3, CPH_SCHEMA_VEC3)                                                               \
    X(Force, Vec3, CPH_SCHEMA_VEC3)                                                                    \
    X(Torque, Vec3, CPH_SCHEMA_VEC3)                                                                   \
    X(SetLinVel, Vec3, CPH_SCHEMA_VEC3)                                                                \
    X(SetAngVel, Vec3, CPH_SCHEMA_VEC3)                                                                \
    X(ImpulseAt, ImpAt, CPH_SCHEMA_IMPULSE_AT)                                                         \
    X(HOnly, HOnly, CPH_SCHEMA_HANDLE_ONLY)                                                            \
    X(Destroy, HOnly, CPH_SCHEMA_HANDLE_ONLY)                                                          \
    X(Activate, HOnly, CPH_SCHEMA_HANDLE_ONLY)                                                         \
    X(Gravity, XYZ, CPH_SCHEMA_XYZ)                                                                    \
    X(SetPos, SetPos, CPH_SCHEMA_SET_POS)                                                              \
    X(Buoy, Buoy, CPH_SCHEMA_BUOYANCY)                                                                 \
    X(BatchBuoy, BatchBuoy, CPH_SCHEMA_BATCH_BUOYANCY)                                                 \
    X(Mesh, Mesh, CPH_SCHEMA_MESH)                                                                     \
    X(SetTrns, SetTrns, CPH_SCHEMA_SET_TRNS)                                                           \
    X(CCD, CCD, CPH_SCHEMA_CCD)                                                                        \
    X(ConvexHull, HC, CPH_SCHEMA_HC_HULL)                                                              \
    X(Compound, HC, CPH_SCHEMA_HC_COMP)                                                                \
    X(BatchCreate, BatchCreate, CPH_SCHEMA_BATCH_CREATE)                                               \
    X(BatchDestroy, BatchDestroy, CPH_SCHEMA_BATCH_DESTROY)                                            \
    X(SetRot, SetRot, CPH_SCHEMA_SET_ROT)                                                              \
    X(Render, Render, CPH_SCHEMA_RENDER)                                                               \
    X(Raycast, Raycast, CPH_SCHEMA_RAYCAST)                                                            \
    X(RayBatch, RayBatch, CPH_SCHEMA_RAYCAST_BATCH)                                                    \
    X(Shapecast, Shapecast, CPH_SCHEMA_SHAPECAST)                                                      \
    X(OverlapSphere, OverlapSphere, CPH_SCHEMA_OVERLAP_SPHERE)                                         \
    X(OverlapAABB, OverlapAABB, CPH_SCHEMA_OVERLAP_AABB)                                               \
    X(SetUserData, SetUserData, CPH_SCHEMA_SET_USER_DATA)                                              \
    X(GetUserData, HOnly, CPH_SCHEMA_HANDLE_ONLY)                                                      \
    X(GetMotion, HOnly, CPH_SCHEMA_HANDLE_ONLY)                                                        \
    X(SetMotion, SetMotion, CPH_SCHEMA_SET_MOTION)                                                     \
    X(ColFilter, ColFilter, CPH_SCHEMA_COL_FILTER)                                                     \
    X(RegMat, RegMat, CPH_SCHEMA_REG_MAT)                                                              \
    X(SetConstrTarget, SetConstr, CPH_SCHEMA_SET_CONSTR_TARGET)                                        \
    X(Heightfield, Heightfield, CPH_SCHEMA_HEIGHTFIELD)                                                \
    X(DebugData, DebugData, CPH_SCHEMA_DEBUG_DATA)                                                     \
    X(CreateConstr, CreateConstr, CPH_SCHEMA_CREATE_CONSTR)                                            \
    X(DestroyConstr, HOnly, CPH_SCHEMA_HANDLE_ONLY)                                                    \
    X(Step, Step, CPH_SCHEMA_STEP)                                                                     \
    X(CreateChar, CreateChar, CPH_SCHEMA_CREATE_CHAR)                                                  \
    X(LoadState, LoadState, CPH_SCHEMA_LOAD_STATE)                                                     \
    X(CreateVehicle, CreateVehicle, CPH_SCHEMA_CREATE_VEHICLE)                                         \
    X(CreateTracked, CreateTracked, CPH_SCHEMA_CREATE_TRACKED)                                         \
    X(CreateRagdoll, CreateRagdoll, CPH_SCHEMA_CREATE_RAGDOLL)                                         \
    X(RagdollSettings, RagdollSettings, CPH_SCHEMA_RAGDOLL_SETTINGS)                                   \
    X(CreateSoftBody, CreateSoftBody, CPH_SCHEMA_CREATE_SOFT_BODY)                                     \
    X(GetSbVertex, GetSbVertex, CPH_SCHEMA_GET_SB_VERTEX)                                              \
    X(StressTest, StressTest, CPH_SCHEMA_STRESS_TEST)                                                  \
    X(CreateShip, CreateShip, CPH_SCHEMA_CREATE_SHIP)

#define CPH_FOR_ALL_CHAR_PARSERS(X)                                                                    \
    X(CharMove, CharMove, CPH_SCHEMA_CHAR_MOVE)                                                        \
    X(SetPosChar, SetPosChar, CPH_SCHEMA_SET_POS_CHAR)                                                 \
    X(SetRotChar, SetRotChar, CPH_SCHEMA_SET_ROT_CHAR)                                                 \
    X(SetStrengthChar, SetStrengthChar, CPH_SCHEMA_SET_STRENGTH_CHAR)

#define CPH_FOR_ALL_VEHICLE_PARSERS(X)                                                                 \
    X(VehicleInput, VehicleInput, CPH_SCHEMA_VEHICLE_INPUT)                                            \
    X(TankInput, TankInput, CPH_SCHEMA_TANK_INPUT)

#define CPH_FOR_ALL_ECS_PARSERS(X)                                                                     \
    X(RegEntityOnly, RegEntityOnly, CPH_SCHEMA_REG_ENTITY_ONLY)                                        \
    X(RegCompOnly, RegCompOnly, CPH_SCHEMA_REG_COMP_ONLY)                                              \
    X(RegRegComp, RegRegComp, CPH_SCHEMA_REG_REG_COMP)                                                 \
    X(RegAdd, RegAdd, CPH_SCHEMA_REG_ADD)                                                              \
    X(RegEntComp, RegEntComp, CPH_SCHEMA_REG_ENT_COMP)                                                 \
    X(RegSyncPhys, RegSyncPhys, CPH_SCHEMA_REG_SYNC_PHYS)

#define CPH_FOR_ALL_SKELETON_PARSERS(X)                                                                \
    X(AddJoint, AddJoint, CPH_SCHEMA_ADD_JOINT)                                                        \
    X(GetJointIdx, GetJointIdx, CPH_SCHEMA_GET_JOINT_IDX)

#define CPH_FOR_ALL_RAGDOLL_PARSERS(X) X(RagdollDrive, RagdollDrive, CPH_SCHEMA_RAGDOLL_DRIVE)

#define CPH_FOR_ALL_RAGDOLL_SETTINGS_PARSERS(X)                                                        \
    X(RagdollAddPart, RagdollAddPart, CPH_SCHEMA_RAGDOLL_ADD_PART)

#define CPH_FOR_ALL_SHIP_PARSERS(X) X(ShipInput, ShipInput, CPH_SCHEMA_SHIP_INPUT)

#define CPH_FOR_ALL_SBSS_PARSERS(X)                                                                    \
    X(SbssAddVertex, SbssAddVertex, CPH_SCHEMA_SBSS_ADD_VERTEX)                                        \
    X(SbssAddFace, SbssAddFace, CPH_SCHEMA_SBSS_ADD_FACE)                                              \
    X(SbssAddVertices, SbssAddVertices, CPH_SCHEMA_SBSS_ADD_VERTICES)                                  \
    X(SbssAddFaces, SbssAddFaces, CPH_SCHEMA_SBSS_ADD_FACES)                                           \
    X(SbssCreateConstraints, SbssCreateConstraints, CPH_SCHEMA_SBSS_CREATE_CONSTRAINTS)

#define CPH_FOR_ALL_MATH_PARSERS(X)                                                                    \
    X(MathPersp, MathPersp, CPH_SCHEMA_MATH_PERSPECTIVE)                                               \
    X(MathOrtho, MathOrtho, CPH_SCHEMA_MATH_ORTHO)                                                     \
    X(MathLookAt, MathLookAt, CPH_SCHEMA_MATH_LOOKAT)                                                  \
    X(MathTRS, MathTRS, CPH_SCHEMA_MATH_TRS)                                                           \
    X(MathTRSBatch, MathTRSBatch, CPH_SCHEMA_MATH_TRS_BATCH)                                           \
    X(MathMat, MathMat, CPH_SCHEMA_MATH_MAT)                                                           \
    X(MathMatPair, MathMatPair, CPH_SCHEMA_MATH_MAT_PAIR)                                              \
    X(MathMatVec, MathMatVec, CPH_SCHEMA_MATH_MAT_VEC)                                                 \
    X(MathMatBatch, MathMatBatch, CPH_SCHEMA_MATH_MAT_BATCH)                                           \
    X(MathCull, MathCull, CPH_SCHEMA_MATH_CULL)                                                        \
    X(MathCullBatch, MathCullBatch, CPH_SCHEMA_MATH_CULL_BATCH)                                        \
    X(MathVec3Batch, MathVec3Batch, CPH_SCHEMA_MATH_VEC3_BATCH)                                        \
    X(MathEuler, MathEuler, CPH_SCHEMA_MATH_EULER)                                                     \
    X(MathQuat, MathQuat, CPH_SCHEMA_MATH_QUAT)                                                        \
    X(MathSlerp, MathSlerp, CPH_SCHEMA_MATH_SLERP)                                                     \
    X(MathQuatPair, MathQuatPair, CPH_SCHEMA_MATH_QUAT_PAIR)                                           \
    X(MathLerpBatch, MathLerpBatch, CPH_SCHEMA_MATH_LERP_BATCH)                                        \
    X(MathQuatVec, MathQuatVec, CPH_SCHEMA_MATH_QUAT_VEC)                                              \
    X(MathQuatVecBatch, MathQuatVecBatch, CPH_SCHEMA_MATH_QUAT_VEC_BATCH)                              \
    X(MathQuatOp, MathQuatOp, CPH_SCHEMA_MATH_QUAT_OP)                                                 \
    X(MathProject, MathProject, CPH_SCHEMA_MATH_PROJECT)                                               \
    X(MathUnproject, MathUnproject, CPH_SCHEMA_MATH_UNPROJECT)                                         \
    X(MathVecPair, MathVecPair, CPH_SCHEMA_MATH_VEC_PAIR)                                              \
    X(MathRayPlane, MathRayPlane, CPH_SCHEMA_MATH_RAY_PLANE)                                           \
    X(MathAxisAngle, MathAxisAngle, CPH_SCHEMA_MATH_AXIS_ANGLE)                                        \
    X(MathDistBatch, MathDistBatch, CPH_SCHEMA_MATH_DIST_BATCH)                                        \
    X(MathVecOp, MathVecOp, CPH_SCHEMA_MATH_VEC_OP)                                                    \
    X(MathReflect, MathReflect, CPH_SCHEMA_MATH_REFLECT)                                               \
    X(MathEulerVec, MathEulerVec, CPH_SCHEMA_MATH_EULER_VEC)                                           \
    X(MathEulerBatch, MathEulerBatch, CPH_SCHEMA_MATH_EULER_BATCH)

#define CPH_FOR_ALL_MODULE_PARSERS(X) X(DumpSchema, DumpSchema, CPH_SCHEMA_DUMP_SCHEMA)

#define CPH_MAP_TO_DECLARE(P, G, S) CPH_DECLARE_PARSER(P, G)

// B. Declare the Parsers
typedef struct CPH_WorldParsers {
    CPH_FOR_ALL_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_WorldParsers;

typedef struct CPH_CharacterParsers {
    CPH_FOR_ALL_CHAR_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_CharacterParsers;

typedef struct CPH_VehicleParsers {
    CPH_FOR_ALL_VEHICLE_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_VehicleParsers;

typedef struct CPH_ECSParsers {
    CPH_FOR_ALL_ECS_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_ECSParsers;

typedef struct CPH_SkeletonParsers {
    CPH_FOR_ALL_SKELETON_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_SkeletonParsers;

typedef struct CPH_RagdollParsers {
    CPH_FOR_ALL_RAGDOLL_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_RagdollParsers;

typedef struct CPH_RagdollSettingsParsers {
    CPH_FOR_ALL_RAGDOLL_SETTINGS_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_RagdollSettingsParsers;

typedef struct CPH_ShipParsers {
    CPH_FOR_ALL_SHIP_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_ShipParsers;

typedef struct CPH_SoftBodySharedSettingsParsers {
    CPH_FOR_ALL_SBSS_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_SoftBodySharedSettingsParsers;

typedef struct CPH_MathParsers {
    CPH_FOR_ALL_MATH_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_MathParsers;

typedef struct CPH_ModuleParsers {
    CPH_FOR_ALL_MODULE_PARSERS(CPH_MAP_TO_DECLARE)
    CPH_FastParser *registry[kParserRegistrySize];
    size_t registry_count;
} CPH_ModuleParsers;

// --- World ---
void CPH_InitWorldParsers(CPH_WorldParsers *wp);
void CPH_FreeWorldParsers(CPH_WorldParsers *wp);

// --- Character ---
void CPH_InitCharParsers(CPH_CharacterParsers *cp);
void CPH_FreeCharParsers(CPH_CharacterParsers *cp);

// --- Vehicle ---
void CPH_InitVehicleParserGroup(CPH_VehicleParsers *vp);
void CPH_FreeVehicleParserGroup(CPH_VehicleParsers *vp);

// --- ECS Registry ---
void CPH_InitEcsParsers(CPH_ECSParsers *ep);
void CPH_FreeEcsParsers(CPH_ECSParsers *ep);

// --- Skeleton ---
void CPH_InitSkeletonParsers(CPH_SkeletonParsers *sp);
void CPH_FreeSkeletonParsers(CPH_SkeletonParsers *sp);

// --- Ragdoll Runtime ---
void CPH_InitRagdollParsers(CPH_RagdollParsers *rp);
void CPH_FreeRagdollParsers(CPH_RagdollParsers *rp);

// --- Ragdoll Settings ---
void CPH_InitRagdollSettingsParsers(CPH_RagdollSettingsParsers *rsp);
void CPH_FreeRagdollSettingsParsers(CPH_RagdollSettingsParsers *rsp);

// --- Ship ---
void CPH_InitShipParsers(CPH_ShipParsers *sp);
void CPH_FreeShipParsers(CPH_ShipParsers *sp);

// --- Soft Body Shared Settings (SBSS) ---
void CPH_InitSbssParsers(CPH_SoftBodySharedSettingsParsers *sbssp);
void CPH_FreeSbssParsers(CPH_SoftBodySharedSettingsParsers *sbssp);

// --- Math Service ---
void CPH_MathInitAllParsers(CPH_MathParsers *mp);
void CPH_MathFreeAllParsers(CPH_MathParsers *mp);

// --- Module ---
void CPH_InitModuleParsers(CPH_ModuleParsers *mp);
void CPH_FreeModuleParsers(CPH_ModuleParsers *mp);

void CPH_FastParseDumpSchemasJson(CPH_WorldParsers *wp, FILE *out);
