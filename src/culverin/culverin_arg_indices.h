// culverin_arg_indices.h
#pragma once
#include "culverin_fast_parse.h"

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