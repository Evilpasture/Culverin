// culverin_arg_indices.h
#pragma once
#include "culverin_fast_parse.h"

void culverin_init_all_parsers(void);

typedef enum {
    IDX_POS, IDX_ROT, IDX_SIZE, IDX_SHAPE, IDX_MOTION,
    IDX_USER_DATA, IDX_SENSOR, IDX_MASS, IDX_CAT, IDX_MASK,
    IDX_FRIC, IDX_REST, IDX_MAT, IDX_CCD,
    IDX_BODY_COUNT
} BodyArgIdx;

typedef enum {
    IDX_BATCH_POSITIONS,
    IDX_BATCH_SIZES,
    IDX_BATCH_SHAPE_TYPE,
    IDX_BATCH_MOTION_TYPE,
    IDX_BATCH_COUNT 
} BatchCreateArgIdx;

typedef enum {
    IDX_SETPOS_HANDLE, IDX_SETPOS_X, IDX_SETPOS_Y, IDX_SETPOS_Z,
    IDX_SETPOS_COUNT
} SetPosArgIdx;

typedef enum {
    IDX_SETVEL_HANDLE, IDX_SETVEL_VX, IDX_SETVEL_VY, IDX_SETVEL_VZ,
    IDX_SETVEL_COUNT
} SetVelArgIdx;

typedef enum {
    IDX_BATCH_DESTROY_HANDLES,
    IDX_BATCH_DESTROY_COUNT
} BatchDestroyArgIdx;

typedef enum {
    IDX_SETROT_HANDLE,
    IDX_SETROT_X,
    IDX_SETROT_Y,
    IDX_SETROT_Z,
    IDX_SETROT_W,
    IDX_SETROT_COUNT
} SetRotArgIdx;

typedef enum {
    IDX_SETLINVEL_HANDLE,
    IDX_SETLINVEL_X,
    IDX_SETLINVEL_Y,
    IDX_SETLINVEL_Z,
    IDX_SETLINVEL_COUNT
} SetLinVelArgIdx;

typedef enum {
    IDX_SETANGVEL_HANDLE,
    IDX_SETANGVEL_X,
    IDX_SETANGVEL_Y,
    IDX_SETANGVEL_Z,
    IDX_SETANGVEL_COUNT
} SetAngVelArgIdx;

typedef enum {
    IDX_IMPULSE_HANDLE,
    IDX_IMPULSE_X,
    IDX_IMPULSE_Y,
    IDX_IMPULSE_Z,
    IDX_IMPULSE_COUNT
} ImpulseArgIdx;

typedef enum {
    IDX_IMPULSE_AT_HANDLE,
    IDX_IMPULSE_AT_IX,
    IDX_IMPULSE_AT_IY,
    IDX_IMPULSE_AT_IZ,
    IDX_IMPULSE_AT_PX,
    IDX_IMPULSE_AT_PY,
    IDX_IMPULSE_AT_PZ,
    IDX_IMPULSE_AT_COUNT
} ImpulseAtArgIdx;

typedef enum {
    IDX_GRAV_X, IDX_GRAV_Y, IDX_GRAV_Z,
    IDX_GRAV_COUNT
} GravityArgIdx;

typedef enum {
    IDX_BUOY_HANDLE,
    IDX_BUOY_SURFACE_Y,
    IDX_BUOY_BUOYANCY,
    IDX_BUOY_LIN_DRAG,
    IDX_BUOY_ANG_DRAG,
    IDX_BUOY_DT,
    IDX_BUOY_VEL, // We'll take this as a PyObject* to handle the (fff)
    IDX_BUOY_COUNT
} BuoyancyArgIdx;

typedef enum {
    IDX_BBUOY_HANDLES,
    IDX_BBUOY_SURFACE_Y,
    IDX_BBUOY_BUOYANCY,
    IDX_BBUOY_LIN_DRAG,
    IDX_BBUOY_ANG_DRAG,
    IDX_BBUOY_DT,
    IDX_BBUOY_VEL,
    IDX_BBUOY_COUNT
} BatchBuoyancyArgIdx;

typedef enum {
    IDX_HULL_POS,
    IDX_HULL_ROT,
    IDX_HULL_POINTS,
    IDX_HULL_MOTION,
    IDX_HULL_MASS,
    IDX_HULL_USER_DATA,
    IDX_HULL_CAT,
    IDX_HULL_MASK,
    IDX_HULL_MAT_ID,
    IDX_HULL_FRIC,
    IDX_HULL_REST,
    IDX_HULL_CCD,
    IDX_HULL_COUNT
} ConvexHullArgIdx;

typedef enum {
    IDX_TORQUE_HANDLE,
    IDX_TORQUE_X,
    IDX_TORQUE_Y,
    IDX_TORQUE_Z,
    IDX_TORQUE_COUNT
} TorqueArgIdx;

typedef enum {
    IDX_FORCE_HANDLE,
    IDX_FORCE_X,
    IDX_FORCE_Y,
    IDX_FORCE_Z,
    IDX_FORCE_COUNT
} ForceArgIdx;

typedef enum {
    IDX_ANGIMP_HANDLE,
    IDX_ANGIMP_X,
    IDX_ANGIMP_Y,
    IDX_ANGIMP_Z,
    IDX_ANGIMP_COUNT
} AngImpArgIdx;

typedef enum {
    IDX_CMP_POS,
    IDX_CMP_ROT,
    IDX_CMP_PARTS,
    IDX_CMP_MOTION,
    IDX_CMP_MASS,
    IDX_CMP_USER_DATA,
    IDX_CMP_SENSOR,
    IDX_CMP_CAT,
    IDX_CMP_MASK,
    IDX_CMP_MAT_ID,
    IDX_CMP_FRIC,
    IDX_CMP_REST,
    IDX_CMP_CCD,
    IDX_CMP_COUNT
} CompoundArgIdx;

typedef enum {
    IDX_MESH_POS,
    IDX_MESH_ROT,
    IDX_MESH_VERTS,
    IDX_MESH_INDICES,
    IDX_MESH_USER_DATA,
    IDX_MESH_CAT,
    IDX_MESH_MASK,
    IDX_MESH_COUNT
} MeshArgIdx;

typedef enum {
    IDX_DESTROY_HANDLE,
    IDX_DESTROY_COUNT
} DestroyArgIdx;

typedef enum {
    IDX_SETTRNS_HANDLE,
    IDX_SETTRNS_POS,
    IDX_SETTRNS_ROT,
    IDX_SETTRNS_COUNT
} SetTransformArgIdx;

typedef enum {
    IDX_HANDLE_ONLY,
    IDX_HANDLE_ONLY_COUNT
} HandleOnlyArgIdx;

typedef enum {
    IDX_CCD_HANDLE,
    IDX_CCD_ENABLED,
    IDX_CCD_COUNT
} CCDArgIdx;

static FastParser BodyParser;
static FastArgSpec BodySpecs[IDX_BODY_COUNT];

static FastParser SetPosParser;
static FastArgSpec SetPosSpecs[IDX_SETPOS_COUNT];

static FastParser SetVelParser;
static FastArgSpec SetVelSpecs[IDX_SETVEL_COUNT];

static FastParser BatchCreateParser;
static FastArgSpec BatchCreateSpecs[IDX_BATCH_COUNT];

static FastParser BatchDestroyParser;
static FastArgSpec BatchDestroySpecs[IDX_BATCH_DESTROY_COUNT];

static FastParser SetRotParser;
static FastArgSpec SetRotSpecs[IDX_SETROT_COUNT];

static FastParser SetLinVelParser;
static FastArgSpec SetLinVelSpecs[IDX_SETLINVEL_COUNT];

static FastParser SetAngVelParser;
static FastArgSpec SetAngVelSpecs[IDX_SETANGVEL_COUNT];

static FastParser ImpulseParser;
static FastArgSpec ImpulseSpecs[IDX_IMPULSE_COUNT];

static FastParser ImpulseAtParser;
static FastArgSpec ImpulseAtSpecs[IDX_IMPULSE_AT_COUNT];

static FastParser GravityParser;
static FastArgSpec GravitySpecs[IDX_GRAV_COUNT];

static FastParser BuoyancyParser;
static FastArgSpec BuoyancySpecs[IDX_BUOY_COUNT];

static FastParser BatchBuoyancyParser;
static FastArgSpec BatchBuoyancySpecs[IDX_BBUOY_COUNT];

static FastParser ConvexHullParser;
static FastArgSpec ConvexHullSpecs[IDX_HULL_COUNT];

static FastParser TorqueParser;
static FastArgSpec TorqueSpecs[IDX_TORQUE_COUNT];

static FastParser ForceParser;
static FastArgSpec ForceSpecs[IDX_FORCE_COUNT];

static FastParser AngImpParser;
static FastArgSpec AngImpSpecs[IDX_ANGIMP_COUNT];

static FastParser CompoundParser;
static FastArgSpec CompoundSpecs[IDX_CMP_COUNT];

static FastParser MeshParser;
static FastArgSpec MeshSpecs[IDX_MESH_COUNT];

static FastParser DestroyParser;
static FastArgSpec DestroySpecs[IDX_DESTROY_COUNT];

static FastParser SetTransformParser;
static FastArgSpec SetTransformSpecs[IDX_SETTRNS_COUNT];

static FastParser HandleOnlyParser;
static FastArgSpec HandleOnlySpecs[IDX_HANDLE_ONLY_COUNT];

static FastParser CCDParser;
static FastArgSpec CCDSpecs[IDX_CCD_COUNT];