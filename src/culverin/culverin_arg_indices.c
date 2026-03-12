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

// --- 3. INITIALIZATION ---

#define GEN_SPEC(ID, NAME, TYPE, REQ)                                                              \
    [ID] = {.name      = (NAME),                                                                   \
            .type_name = #TYPE, /* Stringify the C type (e.g., "JPH_Real") */                      \
            .required  = (bool)(REQ),                                                              \
            .convert   = FP_GET_CONVERTER((TYPE){0})},

// We'll keep a list of all parsers we want to export to Python stubs
CULV_MAYBE_UNUSED
static FastParser *parser_registry[128];
CULV_MAYBE_UNUSED
static int parser_registry_count = 0;

#define REGISTER_PARSER(ParserName)                                                                \
    ParserName##Parser.parser_name           = #ParserName;                                        \
    parser_registry[parser_registry_count++] = &ParserName##Parser;

void culverin_init_all_parsers(void) {
    INIT_PARSER(Body, Body, SCHEMA_BODY);
    REGISTER_PARSER(Body);
    INIT_PARSER(Impulse, Vec3, SCHEMA_VEC3);
    REGISTER_PARSER(Impulse);
    INIT_PARSER(WheelIdx, WheelIdx, SCHEMA_WHEEL_IDX);
    REGISTER_PARSER(WheelIdx);
    INIT_PARSER(AngImpulse, Vec3, SCHEMA_VEC3);
    REGISTER_PARSER(AngImpulse);
    INIT_PARSER(Force, Vec3, SCHEMA_VEC3);
    REGISTER_PARSER(Force);
    INIT_PARSER(Torque, Vec3, SCHEMA_VEC3);
    REGISTER_PARSER(Torque);
    INIT_PARSER(SetLinVel, Vec3, SCHEMA_VEC3);
    REGISTER_PARSER(SetLinVel);
    INIT_PARSER(SetAngVel, Vec3, SCHEMA_VEC3);
    REGISTER_PARSER(SetAngVel);
    INIT_PARSER(ImpulseAt, ImpAt, SCHEMA_IMPULSE_AT);
    REGISTER_PARSER(ImpulseAt);
    INIT_PARSER(HOnly, HOnly, SCHEMA_HANDLE_ONLY);
    REGISTER_PARSER(HOnly);
    INIT_PARSER(Destroy, HOnly, SCHEMA_HANDLE_ONLY);
    REGISTER_PARSER(Destroy);
    INIT_PARSER(Activate, HOnly, SCHEMA_HANDLE_ONLY);
    REGISTER_PARSER(Activate);
    INIT_PARSER(Gravity, XYZ, SCHEMA_XYZ);
    REGISTER_PARSER(Gravity);
    INIT_PARSER(SetPos, SetPos, SCHEMA_SET_POS);
    REGISTER_PARSER(SetPos);
    INIT_PARSER(Buoy, Buoy, SCHEMA_BUOYANCY);
    REGISTER_PARSER(Buoy);
    INIT_PARSER(BatchBuoy, BatchBuoy, SCHEMA_BATCH_BUOYANCY);
    REGISTER_PARSER(BatchBuoy);
    INIT_PARSER(Mesh, Mesh, SCHEMA_MESH);
    REGISTER_PARSER(Mesh);
    INIT_PARSER(SetTrns, SetTrns, SCHEMA_SET_TRNS);
    REGISTER_PARSER(SetTrns);
    INIT_PARSER(CCD, CCD, SCHEMA_CCD);
    REGISTER_PARSER(CCD);
    INIT_PARSER(BatchCreate, BatchCreate, SCHEMA_BATCH_CREATE);
    REGISTER_PARSER(BatchCreate);
    INIT_PARSER(BatchDestroy, BatchDestroy, SCHEMA_BATCH_DESTROY);
    REGISTER_PARSER(BatchDestroy);
    INIT_PARSER(SetRot, SetRot, SCHEMA_SET_ROT);
    REGISTER_PARSER(SetRot);

    // Structural Overlays (Sharing HC index group)
    INIT_PARSER(ConvexHull, HC, SCHEMA_HC_HULL);
    REGISTER_PARSER(ConvexHull);
    INIT_PARSER(Compound, HC, SCHEMA_HC_COMP);
    REGISTER_PARSER(Compound);
    INIT_PARSER(Render, Render, SCHEMA_RENDER);
    REGISTER_PARSER(Render);
    INIT_PARSER(Raycast, Raycast, SCHEMA_RAYCAST);
    REGISTER_PARSER(Raycast);
    INIT_PARSER(RayBatch, RayBatch, SCHEMA_RAYCAST_BATCH);
    REGISTER_PARSER(RayBatch);
    INIT_PARSER(Shapecast, Shapecast, SCHEMA_SHAPECAST);
    REGISTER_PARSER(Shapecast);
    INIT_PARSER(OverlapSphere, OverlapSphere, SCHEMA_OVERLAP_SPHERE);
    REGISTER_PARSER(OverlapSphere);
    INIT_PARSER(OverlapAABB, OverlapAABB, SCHEMA_OVERLAP_AABB);
    REGISTER_PARSER(OverlapAABB);
    INIT_PARSER(SetUserData, SetUserData, SCHEMA_SET_USER_DATA);
    REGISTER_PARSER(SetUserData);
    INIT_PARSER(GetUserData, HOnly, SCHEMA_HANDLE_ONLY);
    REGISTER_PARSER(GetUserData);
    INIT_PARSER(GetMotion, HOnly, SCHEMA_HANDLE_ONLY);
    REGISTER_PARSER(GetMotion);
    INIT_PARSER(SetMotion, SetMotion, SCHEMA_SET_MOTION);
    REGISTER_PARSER(SetMotion);
    INIT_PARSER(ColFilter, ColFilter, SCHEMA_COL_FILTER);
    REGISTER_PARSER(ColFilter);
    INIT_PARSER(RegMat, RegMat, SCHEMA_REG_MAT);
    REGISTER_PARSER(RegMat);
    INIT_PARSER(SetConstrTarget, SetConstr, SCHEMA_SET_CONSTR_TARGET);
    REGISTER_PARSER(SetConstrTarget);
    INIT_PARSER(Heightfield, Heightfield, SCHEMA_HEIGHTFIELD);
    REGISTER_PARSER(Heightfield);
    INIT_PARSER(DebugData, DebugData, SCHEMA_DEBUG_DATA);
    REGISTER_PARSER(DebugData);
    INIT_PARSER(CreateConstr, CreateConstr, SCHEMA_CREATE_CONSTR);
    REGISTER_PARSER(CreateConstr);
    INIT_PARSER(DestroyConstr, HOnly, SCHEMA_HANDLE_ONLY);
    REGISTER_PARSER(DestroyConstr);
    INIT_PARSER(Step, Step, SCHEMA_STEP);
    REGISTER_PARSER(Step);
    INIT_PARSER(CharMove, CharMove, SCHEMA_CHAR_MOVE);
    REGISTER_PARSER(CharMove);
    INIT_PARSER(LoadState, LoadState, SCHEMA_LOAD_STATE);
    REGISTER_PARSER(LoadState);
    INIT_PARSER(CreateChar, CreateChar, SCHEMA_CREATE_CHAR);
    REGISTER_PARSER(CreateChar);
    INIT_PARSER(SetPosChar, SetPosChar, SCHEMA_SET_POS_CHAR);
    REGISTER_PARSER(SetPosChar);
    INIT_PARSER(SetRotChar, SetRotChar, SCHEMA_SET_ROT_CHAR);
    REGISTER_PARSER(SetRotChar);
    INIT_PARSER(SetStrengthChar, SetStrengthChar, SCHEMA_SET_STRENGTH_CHAR);
    REGISTER_PARSER(SetStrengthChar);
    INIT_PARSER(VehicleInput, VehicleInput, SCHEMA_VEHICLE_INPUT);
    REGISTER_PARSER(VehicleInput);
    INIT_PARSER(TankInput, TankInput, SCHEMA_TANK_INPUT);
    REGISTER_PARSER(TankInput);
    INIT_PARSER(CreateVehicle, CreateVehicle, SCHEMA_CREATE_VEHICLE);
    REGISTER_PARSER(CreateVehicle);
    INIT_PARSER(CreateTracked, CreateTracked, SCHEMA_CREATE_TRACKED);
    REGISTER_PARSER(CreateTracked);
    INIT_PARSER(CreateRagdoll, CreateRagdoll, SCHEMA_CREATE_RAGDOLL);
    REGISTER_PARSER(CreateRagdoll);
    INIT_PARSER(RagdollSettings, RagdollSettings, SCHEMA_RAGDOLL_SETTINGS);
    REGISTER_PARSER(RagdollSettings);
    INIT_PARSER(RagdollAddPart, RagdollAddPart, SCHEMA_RAGDOLL_ADD_PART);
    REGISTER_PARSER(RagdollAddPart);
    INIT_PARSER(AddJoint, AddJoint, SCHEMA_ADD_JOINT);
    REGISTER_PARSER(AddJoint);
    INIT_PARSER(GetJointIdx, GetJointIdx, SCHEMA_GET_JOINT_IDX);
    REGISTER_PARSER(GetJointIdx);
    INIT_PARSER(RagdollDrive, RagdollDrive, SCHEMA_RAGDOLL_DRIVE);
    REGISTER_PARSER(RagdollDrive);
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

#define DEINIT_PARSER(ParserName, GroupName) fp_deinit(&ParserName##Parser);

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
    DEINIT_PARSER(LoadState, LoadState)
    DEINIT_PARSER(CreateChar, CreateChar)
    DEINIT_PARSER(SetPosChar, SetPosChar)
    DEINIT_PARSER(SetRotChar, SetRotChar)
    DEINIT_PARSER(SetStrengthChar, SetStrengthChar)
    DEINIT_PARSER(VehicleInput, VehicleInput)
    DEINIT_PARSER(WheelIdx, WheelIdx)
    DEINIT_PARSER(TankInput, TankInput)
    DEINIT_PARSER(CreateVehicle, CreateVehicle)
    DEINIT_PARSER(CreateTracked, CreateTracked)
    DEINIT_PARSER(CreateRagdoll, CreateRagdoll)
    DEINIT_PARSER(RagdollSettings, RagdollSettings)
    DEINIT_PARSER(RagdollAddPart, RagdollAddPart)
    DEINIT_PARSER(AddJoint, AddJoint)
    DEINIT_PARSER(GetJointIdx, GetJointIdx)
    DEINIT_PARSER(RagdollDrive, RagdollDrive)
}
