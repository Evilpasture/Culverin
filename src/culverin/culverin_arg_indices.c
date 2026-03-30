#include "culverin_arg_indices.h"

// --- 1. THE SAFETY ENGINE ---

// Helper macro to count schema entries at compile time
#define COUNT_X(ID, NAME, TYPE, REQ) +1

/**
 * INIT_PARSER
 * Builds the spec array and initializes the parser.
 */
#define INIT_PARSER(ParserName, GroupName, Schema)                                                 \
    do {                                                                                           \
        static_assert((0 Schema(COUNT_X)) == GroupName##_COUNT,                                    \
                      "FastParse: Schema length mismatch for " #ParserName);                       \
        FastArgSpec temp[] = {Schema(GEN_SPEC)};                                                   \
        memcpy(ParserName##Specs, temp, sizeof(temp));                                             \
        fp_init_impl(&ParserName##Parser, ParserName##Specs, GroupName##_COUNT);                   \
    } while (0)

// --- 2. REGISTRATION & SETUP MACROS ---

#define REGISTER_PARSER(ParserName)                                                                \
    do {                                                                                           \
        ParserName##Parser.parser_name           = #ParserName;                                    \
        parser_registry[parser_registry_count++] = &ParserName##Parser;                            \
    } while (0)

#define SETUP_PARSER(ParserName, GroupName, Schema)                                                \
    do {                                                                                           \
        INIT_PARSER(ParserName, GroupName, Schema);                                                \
        REGISTER_PARSER(ParserName);                                                               \
    } while (0)

#define TEARDOWN_PARSER(ParserName, GroupName)                                                     \
    fp_deinit(&ParserName##Parser);

// --- 3. ALLOCATIONS ---

#define ALLOC_PARSER(ParserName, GroupName)                                                        \
    FastParser ParserName##Parser;                                                                 \
    FastArgSpec ParserName##Specs[GroupName##_COUNT];

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
ALLOC_PARSER(LoadState, LoadState)
ALLOC_PARSER(CreateChar, CreateChar)
ALLOC_PARSER(SetPosChar, SetPosChar)
ALLOC_PARSER(SetRotChar, SetRotChar)
ALLOC_PARSER(SetStrengthChar, SetStrengthChar)
ALLOC_PARSER(VehicleInput, VehicleInput)
ALLOC_PARSER(WheelIdx, WheelIdx)
ALLOC_PARSER(TankInput, TankInput)
ALLOC_PARSER(CreateVehicle, CreateVehicle)
ALLOC_PARSER(CreateTracked, CreateTracked)
ALLOC_PARSER(CreateRagdoll, CreateRagdoll)
ALLOC_PARSER(RagdollSettings, RagdollSettings)
ALLOC_PARSER(RagdollAddPart, RagdollAddPart)
ALLOC_PARSER(AddJoint, AddJoint)
ALLOC_PARSER(GetJointIdx, GetJointIdx)
ALLOC_PARSER(RagdollDrive, RagdollDrive)
ALLOC_PARSER(StressTest, StressTest)

// --- 4. INITIALIZATION & CLEANUP ---

#define GEN_SPEC(ID, NAME, TYPE, REQ)                                                              \
    [ID] = {.name      = (NAME),                                                                   \
            .type_name = #TYPE,                                                                    \
            .required  = (bool)(REQ),                                                              \
            .convert   = FP_GET_CONVERTER((TYPE){0})},

CULV_MAYBE_UNUSED static FastParser *parser_registry[128];
CULV_MAYBE_UNUSED static int parser_registry_count = 0;

void culverin_init_all_parsers(void) {
    SETUP_PARSER(Body, Body, SCHEMA_BODY);
    SETUP_PARSER(Impulse, Vec3, SCHEMA_VEC3);
    SETUP_PARSER(WheelIdx, WheelIdx, SCHEMA_WHEEL_IDX);
    SETUP_PARSER(AngImpulse, Vec3, SCHEMA_VEC3);
    SETUP_PARSER(Force, Vec3, SCHEMA_VEC3);
    SETUP_PARSER(Torque, Vec3, SCHEMA_VEC3);
    SETUP_PARSER(SetLinVel, Vec3, SCHEMA_VEC3);
    SETUP_PARSER(SetAngVel, Vec3, SCHEMA_VEC3);
    SETUP_PARSER(ImpulseAt, ImpAt, SCHEMA_IMPULSE_AT);
    SETUP_PARSER(HOnly, HOnly, SCHEMA_HANDLE_ONLY);
    SETUP_PARSER(Destroy, HOnly, SCHEMA_HANDLE_ONLY);
    SETUP_PARSER(Activate, HOnly, SCHEMA_HANDLE_ONLY);
    SETUP_PARSER(Gravity, XYZ, SCHEMA_XYZ);
    SETUP_PARSER(SetPos, SetPos, SCHEMA_SET_POS);
    SETUP_PARSER(Buoy, Buoy, SCHEMA_BUOYANCY);
    SETUP_PARSER(BatchBuoy, BatchBuoy, SCHEMA_BATCH_BUOYANCY);
    SETUP_PARSER(Mesh, Mesh, SCHEMA_MESH);
    SETUP_PARSER(SetTrns, SetTrns, SCHEMA_SET_TRNS);
    SETUP_PARSER(CCD, CCD, SCHEMA_CCD);
    SETUP_PARSER(ConvexHull, HC, SCHEMA_HC_HULL);
    SETUP_PARSER(Compound, HC, SCHEMA_HC_COMP);
    SETUP_PARSER(BatchCreate, BatchCreate, SCHEMA_BATCH_CREATE);
    SETUP_PARSER(BatchDestroy, BatchDestroy, SCHEMA_BATCH_DESTROY);
    SETUP_PARSER(Render, Render, SCHEMA_RENDER);
    SETUP_PARSER(Raycast, Raycast, SCHEMA_RAYCAST);
    SETUP_PARSER(RayBatch, RayBatch, SCHEMA_RAYCAST_BATCH);
    SETUP_PARSER(Shapecast, Shapecast, SCHEMA_SHAPECAST);
    SETUP_PARSER(OverlapSphere, OverlapSphere, SCHEMA_OVERLAP_SPHERE);
    SETUP_PARSER(OverlapAABB, OverlapAABB, SCHEMA_OVERLAP_AABB);
    SETUP_PARSER(SetUserData, SetUserData, SCHEMA_SET_USER_DATA);
    SETUP_PARSER(GetUserData, HOnly, SCHEMA_HANDLE_ONLY);
    SETUP_PARSER(GetMotion, HOnly, SCHEMA_HANDLE_ONLY);
    SETUP_PARSER(SetMotion, SetMotion, SCHEMA_SET_MOTION);
    SETUP_PARSER(ColFilter, ColFilter, SCHEMA_COL_FILTER);
    SETUP_PARSER(RegMat, RegMat, SCHEMA_REG_MAT);
    SETUP_PARSER(SetConstrTarget, SetConstr, SCHEMA_SET_CONSTR_TARGET);
    SETUP_PARSER(Heightfield, Heightfield, SCHEMA_HEIGHTFIELD);
    SETUP_PARSER(DebugData, DebugData, SCHEMA_DEBUG_DATA);
    SETUP_PARSER(CreateConstr, CreateConstr, SCHEMA_CREATE_CONSTR);
    SETUP_PARSER(DestroyConstr, HOnly, SCHEMA_HANDLE_ONLY);
    SETUP_PARSER(Step, Step, SCHEMA_STEP);
    SETUP_PARSER(CharMove, CharMove, SCHEMA_CHAR_MOVE);
    SETUP_PARSER(LoadState, LoadState, SCHEMA_LOAD_STATE);
    SETUP_PARSER(CreateChar, CreateChar, SCHEMA_CREATE_CHAR);
    SETUP_PARSER(SetPosChar, SetPosChar, SCHEMA_SET_POS_CHAR);
    SETUP_PARSER(SetRotChar, SetRotChar, SCHEMA_SET_ROT_CHAR);
    SETUP_PARSER(SetStrengthChar, SetStrengthChar, SCHEMA_SET_STRENGTH_CHAR);
    SETUP_PARSER(VehicleInput, VehicleInput, SCHEMA_VEHICLE_INPUT);
    SETUP_PARSER(TankInput, TankInput, SCHEMA_TANK_INPUT);
    SETUP_PARSER(CreateVehicle, CreateVehicle, SCHEMA_CREATE_VEHICLE);
    SETUP_PARSER(CreateTracked, CreateTracked, SCHEMA_CREATE_TRACKED);
    SETUP_PARSER(CreateRagdoll, CreateRagdoll, SCHEMA_CREATE_RAGDOLL);
    SETUP_PARSER(RagdollSettings, RagdollSettings, SCHEMA_RAGDOLL_SETTINGS);
    SETUP_PARSER(RagdollAddPart, RagdollAddPart, SCHEMA_RAGDOLL_ADD_PART);
    SETUP_PARSER(AddJoint, AddJoint, SCHEMA_ADD_JOINT);
    SETUP_PARSER(GetJointIdx, GetJointIdx, SCHEMA_GET_JOINT_IDX);
    SETUP_PARSER(RagdollDrive, RagdollDrive, SCHEMA_RAGDOLL_DRIVE);
    SETUP_PARSER(StressTest, StressTest, SCHEMA_STRESS_TEST);
}

void culverin_free_all_parsers(void) {
    TEARDOWN_PARSER(Body, Body);
    TEARDOWN_PARSER(Impulse, Vec3);
    TEARDOWN_PARSER(AngImpulse, Vec3);
    TEARDOWN_PARSER(Force, Vec3);
    TEARDOWN_PARSER(Torque, Vec3);
    TEARDOWN_PARSER(SetLinVel, Vec3);
    TEARDOWN_PARSER(SetAngVel, Vec3);
    TEARDOWN_PARSER(ImpulseAt, ImpAt);
    TEARDOWN_PARSER(HOnly, HOnly);
    TEARDOWN_PARSER(Destroy, HOnly);
    TEARDOWN_PARSER(Activate, HOnly);
    TEARDOWN_PARSER(Gravity, XYZ);
    TEARDOWN_PARSER(SetPos, SetPos);
    TEARDOWN_PARSER(Buoy, Buoy);
    TEARDOWN_PARSER(BatchBuoy, BatchBuoy);
    TEARDOWN_PARSER(Mesh, Mesh);
    TEARDOWN_PARSER(SetTrns, SetTrns);
    TEARDOWN_PARSER(CCD, CCD);
    TEARDOWN_PARSER(ConvexHull, HC);
    TEARDOWN_PARSER(Compound, HC);
    TEARDOWN_PARSER(BatchCreate, BatchCreate);
    TEARDOWN_PARSER(BatchDestroy, BatchDestroy);
    TEARDOWN_PARSER(SetRot, SetRot);
    TEARDOWN_PARSER(Render, Render);
    TEARDOWN_PARSER(Raycast, Raycast);
    TEARDOWN_PARSER(RayBatch, RayBatch);
    TEARDOWN_PARSER(Shapecast, Shapecast);
    TEARDOWN_PARSER(OverlapSphere, OverlapSphere);
    TEARDOWN_PARSER(OverlapAABB, OverlapAABB);
    TEARDOWN_PARSER(SetUserData, SetUserData);
    TEARDOWN_PARSER(GetUserData, HOnly);
    TEARDOWN_PARSER(GetMotion, HOnly);
    TEARDOWN_PARSER(SetMotion, SetMotion);
    TEARDOWN_PARSER(ColFilter, ColFilter);
    TEARDOWN_PARSER(RegMat, RegMat);
    TEARDOWN_PARSER(SetConstrTarget, SetConstr);
    TEARDOWN_PARSER(Heightfield, Heightfield);
    TEARDOWN_PARSER(DebugData, DebugData);
    TEARDOWN_PARSER(CreateConstr, CreateConstr);
    TEARDOWN_PARSER(DestroyConstr, HOnly);
    TEARDOWN_PARSER(Step, Step);
    TEARDOWN_PARSER(CharMove, CharMove);
    TEARDOWN_PARSER(LoadState, LoadState);
    TEARDOWN_PARSER(CreateChar, CreateChar);
    TEARDOWN_PARSER(SetPosChar, SetPosChar);
    TEARDOWN_PARSER(SetRotChar, SetRotChar);
    TEARDOWN_PARSER(SetStrengthChar, SetStrengthChar);
    TEARDOWN_PARSER(VehicleInput, VehicleInput);
    TEARDOWN_PARSER(WheelIdx, WheelIdx);
    TEARDOWN_PARSER(TankInput, TankInput);
    TEARDOWN_PARSER(CreateVehicle, CreateVehicle);
    TEARDOWN_PARSER(CreateTracked, CreateTracked);
    TEARDOWN_PARSER(CreateRagdoll, CreateRagdoll);
    TEARDOWN_PARSER(RagdollSettings, RagdollSettings);
    TEARDOWN_PARSER(RagdollAddPart, RagdollAddPart);
    TEARDOWN_PARSER(AddJoint, AddJoint);
    TEARDOWN_PARSER(GetJointIdx, GetJointIdx);
    TEARDOWN_PARSER(RagdollDrive, RagdollDrive);
    TEARDOWN_PARSER(StressTest, StressTest);
}

void fp_dump_schemas_json(FILE *out) {
    fprintf(out, "{\n");
    for (int i = 0; i < parser_registry_count; i++) {
        FastParser *fp = parser_registry[i];
        fprintf(out, "  \"%s\": [\n", fp->parser_name);
        for (size_t j = 0; j < fp->count; j++) {
            fprintf(out, "    {\"name\": \"%s\", \"type\": \"%s\", \"required\": %s}%s\n",
                    fp->specs[j].name, fp->specs[j].type_name,
                    (int)fp->specs[j].required ? "true" : "false", (j == fp->count - 1) ? "" : ",");
        }
        fprintf(out, "  ]%s\n", (i == parser_registry_count - 1) ? "" : ",");
    }
    fprintf(out, "}\n");
}