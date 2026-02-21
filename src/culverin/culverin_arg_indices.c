#include "culverin_arg_indices.h"

// Allocate memory
#define ALLOC_PARSER(ParserName, GroupName) \
    FastParser ParserName##Parser; \
    FastArgSpec ParserName##Specs[GroupName##_COUNT];

#define GEN_SPEC(ID, NAME, TYPE, REQ) \
    [ID] = { .name = NAME, .required = (bool)REQ, \
             .convert = FP_GET_CONVERTER((TYPE){0}) },

#define INIT_PARSER(ParserName, GroupName, Schema) do { \
    FastArgSpec temp[] = { Schema(GEN_SPEC) }; \
    memcpy(ParserName##Specs, temp, sizeof(temp)); \
    fp_init_impl(&ParserName##Parser, ParserName##Specs, GroupName##_COUNT); \
} while(0)

// Actual Allocations
ALLOC_PARSER(Body,       Body)
ALLOC_PARSER(Impulse,    Vec3)
ALLOC_PARSER(AngImpulse, Vec3)
ALLOC_PARSER(Force,      Vec3)
ALLOC_PARSER(Torque,     Vec3)
ALLOC_PARSER(SetLinVel, Vec3)
ALLOC_PARSER(SetAngVel, Vec3)
ALLOC_PARSER(ImpulseAt,  ImpAt)
ALLOC_PARSER(HOnly,      HOnly)
ALLOC_PARSER(Destroy,    HOnly)
ALLOC_PARSER(Activate,   HOnly)
ALLOC_PARSER(Gravity,    XYZ)
ALLOC_PARSER(SetPos,     SetPos)
ALLOC_PARSER(Buoy,       Buoy)
ALLOC_PARSER(BatchBuoy,  BatchBuoy)
ALLOC_PARSER(Mesh,       Mesh)
ALLOC_PARSER(SetTrns,    SetTrns)
ALLOC_PARSER(CCD,        CCD)
ALLOC_PARSER(ConvexHull, HC)
ALLOC_PARSER(Compound,   HC)
ALLOC_PARSER(BatchCreate, BatchCreate)
ALLOC_PARSER(BatchDestroy, BatchDestroy)
ALLOC_PARSER(SetRot, SetRot)

void culverin_init_all_parsers(void) {
    INIT_PARSER(Body,       Body,      SCHEMA_BODY);
    INIT_PARSER(Impulse,    Vec3,      SCHEMA_VEC3);
    INIT_PARSER(AngImpulse, Vec3,      SCHEMA_VEC3);
    INIT_PARSER(Force,      Vec3,      SCHEMA_VEC3);
    INIT_PARSER(Torque,     Vec3,      SCHEMA_VEC3);
    INIT_PARSER(SetLinVel, Vec3, SCHEMA_VEC3);
    INIT_PARSER(SetAngVel, Vec3, SCHEMA_VEC3);
    INIT_PARSER(ImpulseAt,  ImpAt,     SCHEMA_IMPULSE_AT);
    INIT_PARSER(HOnly,      HOnly,     SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Destroy,    HOnly,     SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Activate,   HOnly,     SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Gravity,    XYZ,       SCHEMA_XYZ);
    INIT_PARSER(SetPos,     SetPos,    SCHEMA_SET_POS);
    INIT_PARSER(Buoy,       Buoy,      SCHEMA_BUOYANCY);
    INIT_PARSER(BatchBuoy,  BatchBuoy, SCHEMA_BATCH_BUOYANCY);
    INIT_PARSER(Mesh,       Mesh,      SCHEMA_MESH);
    INIT_PARSER(SetTrns,    SetTrns,   SCHEMA_SET_TRNS);
    INIT_PARSER(CCD,        CCD,       SCHEMA_CCD);
    INIT_PARSER(ConvexHull, HC,   SCHEMA_HC_HULL);
    INIT_PARSER(Compound,   HC,   SCHEMA_HC_COMP);
    INIT_PARSER(BatchCreate, BatchCreate, SCHEMA_BATCH_CREATE);
    INIT_PARSER(BatchDestroy, BatchDestroy, SCHEMA_BATCH_DESTROY);
    INIT_PARSER(SetRot, SetRot, SCHEMA_SET_ROT);
}