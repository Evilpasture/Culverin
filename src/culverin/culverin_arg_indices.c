#include "culverin_arg_indices.h"

// --- 1. THE SAFETY ENGINE ---

// Helper macro to count schema entries at compile time
#define COUNT_X(ID, NAME, TYPE, REQ) +1

/**
 * INIT_PARSER
 * 1. Counts the elements in the Schema macro.
 * 2. Static Asserts that the count matches the GroupName_COUNT enum.
 * 3. Builds the spec array and initializes the parser.
 */
#define INIT_PARSER(ParserName, GroupName, Schema)                                                 \
    do {                                                                                           \
        /* Safety Check: Ensure Schema length matches Group Count */                               \
        static_assert((0 Schema(COUNT_X)) == GroupName##_COUNT,                                    \
                      "FastParse: Schema length mismatch for " #ParserName);                       \
                                                                                                   \
        FastArgSpec temp[] = {Schema(GEN_SPEC)};                                                   \
        memcpy(ParserName##Specs, temp, sizeof(temp));                                             \
        fp_init_impl(&ParserName##Parser, ParserName##Specs, GroupName##_COUNT);                   \
    } while (0)

// --- 2. ALLOCATIONS ---

#define ALLOC_PARSER(ParserName, GroupName)                                                        \
    static FastParser ParserName##Parser;                                                          \
    static FastArgSpec ParserName##Specs[GroupName##_COUNT];

// Signature-grouped allocations
ALLOC_PARSER(Body, Body)
ALLOC_PARSER(Impulse, Vec3)
ALLOC_PARSER(AngImpulse, Vec3)
ALLOC_PARSER(Force, Vec3)
ALLOC_PARSER(Torque, Vec3)
ALLOC_PARSER(SetLinVel, Vec3)
ALLOC_PARSER(SetAngVel, Vec3)
ALLOC_PARSER(ImpulseAt, ImpAt)
ALLOC_PARSER(HOnly, HOnly)
ALLOC_PARSER(Destroy, HOnly)
ALLOC_PARSER(Activate, HOnly)
ALLOC_PARSER(Gravity, XYZ)
ALLOC_PARSER(SetPos, SetPos)
ALLOC_PARSER(Buoy, Buoy)
ALLOC_PARSER(BatchBuoy, BatchBuoy)
ALLOC_PARSER(Mesh, Mesh)
ALLOC_PARSER(SetTrns, SetTrns)
ALLOC_PARSER(CCD, CCD)
ALLOC_PARSER(ConvexHull, HC)
ALLOC_PARSER(Compound, HC)
ALLOC_PARSER(BatchCreate, BatchCreate)
ALLOC_PARSER(BatchDestroy, BatchDestroy)
ALLOC_PARSER(SetRot, SetRot)
ALLOC_PARSER(Render, Render)
ALLOC_PARSER(Raycast, Raycast)
ALLOC_PARSER(RayBatch, RayBatch)
ALLOC_PARSER(Shapecast, Shapecast)
ALLOC_PARSER(OverlapSphere, OverlapSphere)
ALLOC_PARSER(OverlapAABB, OverlapAABB)
ALLOC_PARSER(SetUserData, SetUserData)
ALLOC_PARSER(GetUserData, HOnly)
ALLOC_PARSER(GetMotion, HOnly)
ALLOC_PARSER(SetMotion, SetMotion)
ALLOC_PARSER(ColFilter, ColFilter)
ALLOC_PARSER(RegMat, RegMat)
ALLOC_PARSER(SetConstrTarget, SetConstr)
ALLOC_PARSER(Heightfield, Heightfield)
ALLOC_PARSER(DebugData, DebugData)
ALLOC_PARSER(CreateConstr, CreateConstr)
ALLOC_PARSER(DestroyConstr, HOnly)
ALLOC_PARSER(Step, Step)
ALLOC_PARSER(CharMove, CharMove)

// --- 3. INITIALIZATION ---

#define GEN_SPEC(ID, NAME, TYPE, REQ)                                                              \
    [ID] = {.name = (NAME), .required = (bool)(REQ), .convert = FP_GET_CONVERTER((TYPE){0})},

void culverin_init_all_parsers(void) {
    INIT_PARSER(Body, Body, SCHEMA_BODY);
    INIT_PARSER(Impulse, Vec3, SCHEMA_VEC3);
    INIT_PARSER(AngImpulse, Vec3, SCHEMA_VEC3);
    INIT_PARSER(Force, Vec3, SCHEMA_VEC3);
    INIT_PARSER(Torque, Vec3, SCHEMA_VEC3);
    INIT_PARSER(SetLinVel, Vec3, SCHEMA_VEC3);
    INIT_PARSER(SetAngVel, Vec3, SCHEMA_VEC3);
    INIT_PARSER(ImpulseAt, ImpAt, SCHEMA_IMPULSE_AT);
    INIT_PARSER(HOnly, HOnly, SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Destroy, HOnly, SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Activate, HOnly, SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Gravity, XYZ, SCHEMA_XYZ);
    INIT_PARSER(SetPos, SetPos, SCHEMA_SET_POS);
    INIT_PARSER(Buoy, Buoy, SCHEMA_BUOYANCY);
    INIT_PARSER(BatchBuoy, BatchBuoy, SCHEMA_BATCH_BUOYANCY);
    INIT_PARSER(Mesh, Mesh, SCHEMA_MESH);
    INIT_PARSER(SetTrns, SetTrns, SCHEMA_SET_TRNS);
    INIT_PARSER(CCD, CCD, SCHEMA_CCD);
    INIT_PARSER(BatchCreate, BatchCreate, SCHEMA_BATCH_CREATE);
    INIT_PARSER(BatchDestroy, BatchDestroy, SCHEMA_BATCH_DESTROY);
    INIT_PARSER(SetRot, SetRot, SCHEMA_SET_ROT);

    // Structural Overlays (Sharing HC index group)
    INIT_PARSER(ConvexHull, HC, SCHEMA_HC_HULL);
    INIT_PARSER(Compound, HC, SCHEMA_HC_COMP);
    INIT_PARSER(Render, Render, SCHEMA_RENDER);
    INIT_PARSER(Raycast, Raycast, SCHEMA_RAYCAST);
    INIT_PARSER(RayBatch, RayBatch, SCHEMA_RAYCAST_BATCH);
    INIT_PARSER(Shapecast, Shapecast, SCHEMA_SHAPECAST);
    INIT_PARSER(OverlapSphere, OverlapSphere, SCHEMA_OVERLAP_SPHERE);
    INIT_PARSER(OverlapAABB, OverlapAABB, SCHEMA_OVERLAP_AABB);
    INIT_PARSER(SetUserData, SetUserData, SCHEMA_SET_USER_DATA);
    INIT_PARSER(GetUserData, HOnly, SCHEMA_HANDLE_ONLY);
    INIT_PARSER(GetMotion, HOnly, SCHEMA_HANDLE_ONLY);
    INIT_PARSER(SetMotion, SetMotion, SCHEMA_SET_MOTION);
    INIT_PARSER(ColFilter, ColFilter, SCHEMA_COL_FILTER);
    INIT_PARSER(RegMat, RegMat, SCHEMA_REG_MAT);
    INIT_PARSER(SetConstrTarget, SetConstr, SCHEMA_SET_CONSTR_TARGET);
    INIT_PARSER(Heightfield, Heightfield, SCHEMA_HEIGHTFIELD);
    INIT_PARSER(DebugData, DebugData, SCHEMA_DEBUG_DATA);
    INIT_PARSER(CreateConstr, CreateConstr, SCHEMA_CREATE_CONSTR);
    INIT_PARSER(DestroyConstr, HOnly, SCHEMA_HANDLE_ONLY);
    INIT_PARSER(Step, Step, SCHEMA_STEP);
    INIT_PARSER(CharMove, CharMove, SCHEMA_CHAR_MOVE);
}

#define DEINIT_PARSER(ParserName, GroupName) \
    fp_deinit(&ParserName##Parser);

void culverin_free_all_parsers(void) {
    DEINIT_PARSER(Body, Body)
    DEINIT_PARSER(Impulse, Vec3)
    DEINIT_PARSER(AngImpulse, Vec3)
    DEINIT_PARSER(Force, Vec3)
    DEINIT_PARSER(Torque, Vec3)
    DEINIT_PARSER(SetLinVel, Vec3)
    DEINIT_PARSER(SetAngVel, Vec3)
    DEINIT_PARSER(ImpulseAt, ImpAt)
    DEINIT_PARSER(HOnly, HOnly)
    DEINIT_PARSER(Destroy, HOnly)
    DEINIT_PARSER(Activate, HOnly)
    DEINIT_PARSER(Gravity, XYZ)
    DEINIT_PARSER(SetPos, SetPos)
    DEINIT_PARSER(Buoy, Buoy)
    DEINIT_PARSER(BatchBuoy, BatchBuoy)
    DEINIT_PARSER(Mesh, Mesh)
    DEINIT_PARSER(SetTrns, SetTrns)
    DEINIT_PARSER(CCD, CCD)
    DEINIT_PARSER(ConvexHull, HC)
    DEINIT_PARSER(Compound, HC)
    DEINIT_PARSER(BatchCreate, BatchCreate)
    DEINIT_PARSER(BatchDestroy, BatchDestroy)
    DEINIT_PARSER(SetRot, SetRot)
    DEINIT_PARSER(Render, Render)
    DEINIT_PARSER(Raycast, Raycast)
    DEINIT_PARSER(RayBatch, RayBatch)
    DEINIT_PARSER(Shapecast, Shapecast)
    DEINIT_PARSER(OverlapSphere, OverlapSphere)
    DEINIT_PARSER(OverlapAABB, OverlapAABB)
    DEINIT_PARSER(SetUserData, SetUserData)
    DEINIT_PARSER(GetUserData, HOnly)
    DEINIT_PARSER(GetMotion, HOnly)
    DEINIT_PARSER(SetMotion, SetMotion)
    DEINIT_PARSER(ColFilter, ColFilter)
    DEINIT_PARSER(RegMat, RegMat)
    DEINIT_PARSER(SetConstrTarget, SetConstr)
    DEINIT_PARSER(Heightfield, Heightfield)
    DEINIT_PARSER(DebugData, DebugData)
    DEINIT_PARSER(CreateConstr, CreateConstr)
    DEINIT_PARSER(DestroyConstr, HOnly)
    DEINIT_PARSER(Step, Step)
    DEINIT_PARSER(CharMove, CharMove)
}