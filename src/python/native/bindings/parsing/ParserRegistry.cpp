#include "bindings/parsing/ParserRegistry.hpp"

namespace CPH::detail {
template <typename... Parsers>
void RegisterParsers(CPH_FastParser **registry, std::size_t &registry_count,
                     Parsers &...parsers) noexcept {
    ((registry[registry_count++] = parsers.Core()), ...);
}
} // namespace CPH::detail


void CPH_WorldParsers::Init() {
    registry_count = 0;
    WorldInit.Init("WorldInit", CPH::FastArg<PyObject *>{"settings", false}, CPH::FastArg<PyObject *>{"bodies", false});
    Body.Init("Body", CPH::FastArg<PyObject *>{"pos", false}, CPH::FastArg<PyObject *>{"rot", false}, CPH::FastArg<PyObject *>{"size", false}, CPH::FastArg<int>{"shape", false}, CPH::FastArg<int>{"motion", false}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<bool>{"is_sensor", false}, CPH::FastArg<float>{"mass", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false}, CPH::FastArg<float>{"friction", false}, CPH::FastArg<float>{"restitution", false}, CPH::FastArg<uint32_t>{"material_id", false}, CPH::FastArg<bool>{"ccd", false}, CPH::FastArg<uint32_t>{"allowed_dofs", false});
    Impulse.Init("Impulse", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    WheelIdx.Init("WheelIdx", CPH::FastArg<uint32_t>{"index", true});
    AngImpulse.Init("AngImpulse", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    Force.Init("Force", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    Torque.Init("Torque", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    SetLinVel.Init("SetLinVel", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    SetAngVel.Init("SetAngVel", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    ImpulseAt.Init("ImpulseAt", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"ix", true}, CPH::FastArg<float>{"iy", true}, CPH::FastArg<float>{"iz", true}, CPH::FastArg<JPH_Real>{"px", true}, CPH::FastArg<JPH_Real>{"py", true}, CPH::FastArg<JPH_Real>{"pz", true});
    HOnly.Init("HOnly", CPH::FastArg<ParseBodyHandle>{"handle", true});
    Destroy.Init("Destroy", CPH::FastArg<ParseBodyHandle>{"handle", true});
    Activate.Init("Activate", CPH::FastArg<ParseBodyHandle>{"handle", true});
    Gravity.Init("Gravity", CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    SetPos.Init("SetPos", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<JPH_Real>{"x", true}, CPH::FastArg<JPH_Real>{"y", true}, CPH::FastArg<JPH_Real>{"z", true});
    Buoy.Init("Buoy", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<double>{"surface_y", true}, CPH::FastArg<float>{"buoyancy", false}, CPH::FastArg<float>{"linear_drag", false}, CPH::FastArg<float>{"angular_drag", false}, CPH::FastArg<float>{"dt", false}, CPH::FastArg<PyObject *>{"fluid_velocity", false});
    BatchBuoy.Init("BatchBuoy", CPH::FastArg<PyObject *>{"handles", true}, CPH::FastArg<double>{"surface_y", true}, CPH::FastArg<float>{"buoyancy", false}, CPH::FastArg<float>{"linear_drag", false}, CPH::FastArg<float>{"angular_drag", false}, CPH::FastArg<float>{"dt", false}, CPH::FastArg<PyObject *>{"fluid_velocity", false});
    Mesh.Init("Mesh", CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true}, CPH::FastArg<PyObject *>{"vertices", true}, CPH::FastArg<PyObject *>{"indices", true}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false});
    SetTrns.Init("SetTrns", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true});
    CCD.Init("CCD", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<bool>{"enabled", true});
    ConvexHull.Init("ConvexHull", CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true}, CPH::FastArg<PyObject *>{"points", true}, CPH::FastArg<int>{"motion", false}, CPH::FastArg<float>{"mass", false}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<bool>{"is_sensor", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false}, CPH::FastArg<uint32_t>{"material_id", false}, CPH::FastArg<float>{"friction", false}, CPH::FastArg<float>{"restitution", false}, CPH::FastArg<bool>{"ccd", false});
    Compound.Init("Compound", CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true}, CPH::FastArg<PyObject *>{"parts", true}, CPH::FastArg<int>{"motion", false}, CPH::FastArg<float>{"mass", false}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<bool>{"is_sensor", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false}, CPH::FastArg<uint32_t>{"material_id", false}, CPH::FastArg<float>{"friction", false}, CPH::FastArg<float>{"restitution", false}, CPH::FastArg<bool>{"ccd", false});
    BatchCreate.Init("BatchCreate", CPH::FastArg<PyObject *>{"positions", true}, CPH::FastArg<PyObject *>{"sizes", true}, CPH::FastArg<int>{"shape_type", false}, CPH::FastArg<int>{"motion_type", false});
    BatchDestroy.Init("BatchDestroy", CPH::FastArg<PyObject *>{"handles", true});
    SetRot.Init("SetRot", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true}, CPH::FastArg<float>{"w", true});
    Render.Init("Render", CPH::FastArg<float>{"alpha", true});
    Raycast.Init("Raycast", CPH::FastArg<PyObject *>{"start", true}, CPH::FastArg<PyObject *>{"direction", true}, CPH::FastArg<float>{"max_dist", false}, CPH::FastArg<ParseBodyHandle>{"ignore", false});
    RayBatch.Init("RayBatch", CPH::FastArg<PyObject *>{"starts", true}, CPH::FastArg<PyObject *>{"directions", true}, CPH::FastArg<float>{"max_dist", false});
    Shapecast.Init("Shapecast", CPH::FastArg<int>{"shape", true}, CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true}, CPH::FastArg<PyObject *>{"dir", true}, CPH::FastArg<PyObject *>{"size", false}, CPH::FastArg<ParseBodyHandle>{"ignore", false});
    OverlapSphere.Init("OverlapSphere", CPH::FastArg<PyObject *>{"center", true}, CPH::FastArg<float>{"radius", true});
    OverlapAABB.Init("OverlapAABB", CPH::FastArg<PyObject *>{"min", true}, CPH::FastArg<PyObject *>{"max", true});
    SetUserData.Init("SetUserData", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<uint64_t>{"data", true});
    GetUserData.Init("GetUserData", CPH::FastArg<ParseBodyHandle>{"handle", true});
    GetMotion.Init("GetMotion", CPH::FastArg<ParseBodyHandle>{"handle", true});
    SetMotion.Init("SetMotion", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<int>{"motion", true});
    ColFilter.Init("ColFilter", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<uint32_t>{"category", true}, CPH::FastArg<uint32_t>{"mask", true});
    RegMat.Init("RegMat", CPH::FastArg<uint32_t>{"id", true}, CPH::FastArg<float>{"friction", false}, CPH::FastArg<float>{"restitution", false});
    SetConstrTarget.Init("SetConstrTarget", CPH::FastArg<uint64_t>{"handle", true}, CPH::FastArg<float>{"target", true});
    Heightfield.Init("Heightfield", CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true}, CPH::FastArg<PyObject *>{"scale", true}, CPH::FastArg<PyObject *>{"heights", true}, CPH::FastArg<int>{"grid_size", true}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false}, CPH::FastArg<uint32_t>{"material_id", false}, CPH::FastArg<float>{"friction", false}, CPH::FastArg<float>{"restitution", false});
    DebugData.Init("DebugData", CPH::FastArg<bool>{"shapes", false}, CPH::FastArg<bool>{"constraints", false}, CPH::FastArg<bool>{"bbox", false}, CPH::FastArg<bool>{"centers", false}, CPH::FastArg<bool>{"wireframe", false});
    CreateConstr.Init("CreateConstr", CPH::FastArg<int>{"type", true}, CPH::FastArg<ParseBodyHandle>{"body1", true}, CPH::FastArg<ParseBodyHandle>{"body2", true}, CPH::FastArg<PyObject *>{"params", false}, CPH::FastArg<PyObject *>{"motor", false});
    DestroyConstr.Init("DestroyConstr", CPH::FastArg<ParseBodyHandle>{"handle", true});
    Step.Init("Step", CPH::FastArg<float>{"dt", false});
    CreateChar.Init("CreateChar", CPH::FastArg<CPH_PosStride>{"pos", true}, CPH::FastArg<float>{"height", false}, CPH::FastArg<float>{"radius", false}, CPH::FastArg<float>{"step_height", false}, CPH::FastArg<float>{"max_slope", false});
    LoadState.Init("LoadState", CPH::FastArg<PyObject *>{"state", true});
    CreateVehicle.Init("CreateVehicle", CPH::FastArg<uint64_t>{"chassis", true}, CPH::FastArg<PyObject *>{"wheels", true}, CPH::FastArg<PyObject *>{"drive", false}, CPH::FastArg<PyObject *>{"engine", false}, CPH::FastArg<PyObject *>{"transmission", false});
    CreateTracked.Init("CreateTracked", CPH::FastArg<uint64_t>{"chassis", true}, CPH::FastArg<PyObject *>{"wheels", true}, CPH::FastArg<PyObject *>{"tracks", true}, CPH::FastArg<float>{"max_torque", false}, CPH::FastArg<float>{"max_rpm", false});
    CreateRagdoll.Init("CreateRagdoll", CPH::FastArg<PyObject *>{"settings", true}, CPH::FastArg<CPH_PosStride>{"pos", true}, CPH::FastArg<CPH_AuxStride>{"rot", false}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false}, CPH::FastArg<uint32_t>{"material_id", false});
    RagdollSettings.Init("RagdollSettings", CPH::FastArg<PyObject *>{"skeleton", true});
    CreateSoftBody.Init("CreateSoftBody", CPH::FastArg<PyObject *>{"shared_settings", true}, CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<PyObject *>{"rot", true}, CPH::FastArg<uint64_t>{"user_data", false}, CPH::FastArg<uint32_t>{"category", false}, CPH::FastArg<uint32_t>{"mask", false}, CPH::FastArg<float>{"pressure", false}, CPH::FastArg<float>{"vertex_radius", false}, CPH::FastArg<float>{"linear_damping", false}, CPH::FastArg<uint32_t>{"num_iterations", false}, CPH::FastArg<float>{"max_linear_velocity", false}, CPH::FastArg<float>{"gravity_factor", false}, CPH::FastArg<float>{"friction", false}, CPH::FastArg<float>{"restitution", false}, CPH::FastArg<bool>{"make_rotation_identity", false}, CPH::FastArg<bool>{"update_position", false}, CPH::FastArg<bool>{"faces_double_sided", false});
    GetSbVertex.Init("GetSbVertex", CPH::FastArg<ParseBodyHandle>{"handle", true}, CPH::FastArg<uint32_t>{"index", true});
    StressTest.Init("StressTest", CPH::FastArg<uint64_t>{"a0", false}, CPH::FastArg<uint64_t>{"a1", false}, CPH::FastArg<uint64_t>{"a2", false}, CPH::FastArg<uint64_t>{"a3", false}, CPH::FastArg<uint64_t>{"a4", false}, CPH::FastArg<uint64_t>{"a5", false}, CPH::FastArg<uint64_t>{"a6", false}, CPH::FastArg<uint64_t>{"a7", false}, CPH::FastArg<uint64_t>{"a8", false}, CPH::FastArg<uint64_t>{"a9", false}, CPH::FastArg<uint64_t>{"a10", false}, CPH::FastArg<uint64_t>{"a11", false}, CPH::FastArg<uint64_t>{"a12", false}, CPH::FastArg<uint64_t>{"a13", false}, CPH::FastArg<uint64_t>{"a14", false}, CPH::FastArg<uint64_t>{"a15", false}, CPH::FastArg<uint64_t>{"a16", false}, CPH::FastArg<uint64_t>{"a17", false}, CPH::FastArg<uint64_t>{"a18", false}, CPH::FastArg<uint64_t>{"a19", false}, CPH::FastArg<uint64_t>{"a20", false}, CPH::FastArg<uint64_t>{"a21", false}, CPH::FastArg<uint64_t>{"a22", false}, CPH::FastArg<uint64_t>{"a23", false}, CPH::FastArg<uint64_t>{"a24", false}, CPH::FastArg<uint64_t>{"a25", false}, CPH::FastArg<uint64_t>{"a26", false}, CPH::FastArg<uint64_t>{"a27", false}, CPH::FastArg<uint64_t>{"a28", false}, CPH::FastArg<uint64_t>{"a29", false}, CPH::FastArg<uint64_t>{"a30", false}, CPH::FastArg<uint64_t>{"a31", false}, CPH::FastArg<uint64_t>{"a32", false}, CPH::FastArg<uint64_t>{"a33", false}, CPH::FastArg<uint64_t>{"a34", false}, CPH::FastArg<uint64_t>{"a35", false}, CPH::FastArg<uint64_t>{"a36", false}, CPH::FastArg<uint64_t>{"a37", false}, CPH::FastArg<uint64_t>{"a38", false}, CPH::FastArg<uint64_t>{"a39", false}, CPH::FastArg<uint64_t>{"a40", false}, CPH::FastArg<uint64_t>{"a41", false}, CPH::FastArg<uint64_t>{"a42", false}, CPH::FastArg<uint64_t>{"a43", false}, CPH::FastArg<uint64_t>{"a44", false}, CPH::FastArg<uint64_t>{"a45", false}, CPH::FastArg<uint64_t>{"a46", false}, CPH::FastArg<uint64_t>{"a47", false}, CPH::FastArg<uint64_t>{"a48", false}, CPH::FastArg<uint64_t>{"a49", false}, CPH::FastArg<uint64_t>{"a50", false}, CPH::FastArg<uint64_t>{"a51", false}, CPH::FastArg<uint64_t>{"a52", false}, CPH::FastArg<uint64_t>{"a53", false}, CPH::FastArg<uint64_t>{"a54", false}, CPH::FastArg<uint64_t>{"a55", false}, CPH::FastArg<uint64_t>{"a56", false}, CPH::FastArg<uint64_t>{"a57", false}, CPH::FastArg<uint64_t>{"a58", false}, CPH::FastArg<uint64_t>{"a59", false}, CPH::FastArg<uint64_t>{"a60", false}, CPH::FastArg<uint64_t>{"a61", false}, CPH::FastArg<uint64_t>{"a62", false}, CPH::FastArg<uint64_t>{"a63", false});
    CreateShip.Init("CreateShip", CPH::FastArg<ParseBodyHandle>{"sled", true}, CPH::FastArg<float>{"kp", true}, CPH::FastArg<float>{"kd", true}, CPH::FastArg<float>{"throttle_force", true}, CPH::FastArg<float>{"steer_speed", true}, CPH::FastArg<float>{"banking", false}, CPH::FastArg<float>{"lateral_grip", false}, CPH::FastArg<float>{"linear_drag", false});
    CPH::detail::RegisterParsers(registry, registry_count, WorldInit, Body, Impulse, WheelIdx, AngImpulse, Force, Torque, SetLinVel, SetAngVel, ImpulseAt, HOnly, Destroy, Activate, Gravity, SetPos, Buoy, BatchBuoy, Mesh, SetTrns, CCD, ConvexHull, Compound, BatchCreate, BatchDestroy, SetRot, Render, Raycast, RayBatch, Shapecast, OverlapSphere, OverlapAABB, SetUserData, GetUserData, GetMotion, SetMotion, ColFilter, RegMat, SetConstrTarget, Heightfield, DebugData, CreateConstr, DestroyConstr, Step, CreateChar, LoadState, CreateVehicle, CreateTracked, CreateRagdoll, RagdollSettings, CreateSoftBody, GetSbVertex, StressTest, CreateShip);
}

void CPH_CharacterParsers::Init() {
    registry_count = 0;
    CharMove.Init("CharMove", CPH::FastArg<CPH_Vec3f>{"velocity", true}, CPH::FastArg<float>{"dt", true});
    SetPosChar.Init("SetPosChar", CPH::FastArg<CPH_PosStride>{"pos", true});
    SetRotChar.Init("SetRotChar", CPH::FastArg<CPH_AuxStride>{"rot", true});
    SetStrengthChar.Init("SetStrengthChar", CPH::FastArg<float>{"strength", true});
    CPH::detail::RegisterParsers(registry, registry_count, CharMove, SetPosChar, SetRotChar, SetStrengthChar);
}

void CPH_VehicleParsers::Init() {
    registry_count = 0;
    VehicleInput.Init("VehicleInput", CPH::FastArg<float>{"forward", false}, CPH::FastArg<float>{"right", false}, CPH::FastArg<float>{"brake", false}, CPH::FastArg<float>{"handbrake", false});
    TankInput.Init("TankInput", CPH::FastArg<float>{"left", true}, CPH::FastArg<float>{"right", true}, CPH::FastArg<float>{"brake", false});
    CPH::detail::RegisterParsers(registry, registry_count, VehicleInput, TankInput);
}

void CPH_ECSParsers::Init() {
    registry_count = 0;
    RegEntityOnly.Init("RegEntityOnly", CPH::FastArg<uint64_t>{"entity", true});
    RegCompOnly.Init("RegCompOnly", CPH::FastArg<uint32_t>{"comp_id", true});
    RegRegComp.Init("RegRegComp", CPH::FastArg<uint32_t>{"size_bytes", true});
    RegAdd.Init("RegAdd", CPH::FastArg<uint64_t>{"entity", true}, CPH::FastArg<uint32_t>{"comp_id", true}, CPH::FastArg<PyObject *>{"data", false});
    RegEntComp.Init("RegEntComp", CPH::FastArg<uint64_t>{"entity", true}, CPH::FastArg<uint32_t>{"comp_id", true});
    RegSyncPhys.Init("RegSyncPhys", CPH::FastArg<PyObject *>{"world", true}, CPH::FastArg<uint32_t>{"handle_comp_id", true}, CPH::FastArg<int>{"transform_comp_id", true}, CPH::FastArg<int>{"rot_comp_id", false});
    CPH::detail::RegisterParsers(registry, registry_count, RegEntityOnly, RegCompOnly, RegRegComp, RegAdd, RegEntComp, RegSyncPhys);
}

void CPH_SkeletonParsers::Init() {
    registry_count = 0;
    AddJoint.Init("AddJoint", CPH::FastArg<PyObject *>{"name", true}, CPH::FastArg<int>{"parent_index", false});
    GetJointIdx.Init("GetJointIdx", CPH::FastArg<PyObject *>{"name", true});
    CPH::detail::RegisterParsers(registry, registry_count, AddJoint, GetJointIdx);
}

void CPH_RagdollParsers::Init() {
    registry_count = 0;
    RagdollDrive.Init("RagdollDrive", CPH::FastArg<CPH_PosStride>{"root_pos", true}, CPH::FastArg<CPH_AuxStride>{"root_rot", true}, CPH::FastArg<PyObject *>{"matrices", true});
    CPH::detail::RegisterParsers(registry, registry_count, RagdollDrive);
}

void CPH_RagdollSettingsParsers::Init() {
    registry_count = 0;
    RagdollAddPart.Init("RagdollAddPart", CPH::FastArg<int>{"joint_index", true}, CPH::FastArg<int>{"shape_type", true}, CPH::FastArg<PyObject *>{"size", true}, CPH::FastArg<float>{"mass", false}, CPH::FastArg<int>{"parent_index", false}, CPH::FastArg<float>{"twist_min", false}, CPH::FastArg<float>{"twist_max", false}, CPH::FastArg<float>{"cone_angle", false}, CPH::FastArg<CPH_Vec3f>{"axis", false}, CPH::FastArg<CPH_Vec3f>{"normal", false}, CPH::FastArg<PyObject *>{"pos", false});
    CPH::detail::RegisterParsers(registry, registry_count, RagdollAddPart);
}

void CPH_ShipParsers::Init() {
    registry_count = 0;
    ShipInput.Init("ShipInput", CPH::FastArg<float>{"forward", true}, CPH::FastArg<float>{"right", true});
    CPH::detail::RegisterParsers(registry, registry_count, ShipInput);
}

void CPH_SoftBodySharedSettingsParsers::Init() {
    registry_count = 0;
    SbssAddVertex.Init("SbssAddVertex", CPH::FastArg<PyObject *>{"pos", true}, CPH::FastArg<float>{"inv_mass", false}, CPH::FastArg<PyObject *>{"velocity", false});
    SbssAddFace.Init("SbssAddFace", CPH::FastArg<uint32_t>{"v1", true}, CPH::FastArg<uint32_t>{"v2", true}, CPH::FastArg<uint32_t>{"v3", true}, CPH::FastArg<uint32_t>{"material_index", false});
    SbssAddVertices.Init("SbssAddVertices", CPH::FastArg<PyObject *>{"positions", true}, CPH::FastArg<PyObject *>{"inv_masses", false}, CPH::FastArg<PyObject *>{"velocities", false});
    SbssAddFaces.Init("SbssAddFaces", CPH::FastArg<PyObject *>{"indices", true}, CPH::FastArg<PyObject *>{"materials", false});
    SbssCreateConstraints.Init("SbssCreateConstraints", CPH::FastArg<float>{"compliance", true}, CPH::FastArg<int>{"bend_type", false});
    CPH::detail::RegisterParsers(registry, registry_count, SbssAddVertex, SbssAddFace, SbssAddVertices, SbssAddFaces, SbssCreateConstraints);
}

void CPH_MathParsers::Init() {
    registry_count = 0;
    MathPersp.Init("MathPersp", CPH::FastArg<float>{"fovy", true}, CPH::FastArg<float>{"aspect", true}, CPH::FastArg<float>{"near", true}, CPH::FastArg<float>{"far", true});
    MathOrtho.Init("MathOrtho", CPH::FastArg<float>{"left", true}, CPH::FastArg<float>{"right", true}, CPH::FastArg<float>{"bottom", true}, CPH::FastArg<float>{"top", true}, CPH::FastArg<float>{"near", true}, CPH::FastArg<float>{"far", true});
    MathLookAt.Init("MathLookAt", CPH::FastArg<PyObject *>{"eye", true}, CPH::FastArg<PyObject *>{"target", true}, CPH::FastArg<PyObject *>{"up", true});
    MathTRS.Init("MathTRS", CPH::FastArg<PyObject *>{"translation", true}, CPH::FastArg<PyObject *>{"rotation", true}, CPH::FastArg<PyObject *>{"scale", true});
    MathTRSBatch.Init("MathTRSBatch", CPH::FastArg<PyObject *>{"translations", true}, CPH::FastArg<PyObject *>{"rotations", true}, CPH::FastArg<PyObject *>{"scales", true});
    MathMat.Init("MathMat", CPH::FastArg<PyObject *>{"mat", true});
    MathMatPair.Init("MathMatPair", CPH::FastArg<PyObject *>{"a", true}, CPH::FastArg<PyObject *>{"b", true});
    MathMatVec.Init("MathMatVec", CPH::FastArg<PyObject *>{"mat", true}, CPH::FastArg<PyObject *>{"vec", true});
    MathMatBatch.Init("MathMatBatch", CPH::FastArg<PyObject *>{"mat", true}, CPH::FastArg<PyObject *>{"batch", true});
    MathCull.Init("MathCull", CPH::FastArg<PyObject *>{"vp", true}, CPH::FastArg<PyObject *>{"min", true}, CPH::FastArg<PyObject *>{"max", true});
    MathCullBatch.Init("MathCullBatch", CPH::FastArg<PyObject *>{"vp", true}, CPH::FastArg<PyObject *>{"aabbs", true});
    MathVec3Batch.Init("MathVec3Batch", CPH::FastArg<PyObject *>{"vecs", true});
    MathEuler.Init("MathEuler", CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true});
    MathQuat.Init("MathQuat", CPH::FastArg<float>{"x", true}, CPH::FastArg<float>{"y", true}, CPH::FastArg<float>{"z", true}, CPH::FastArg<float>{"w", true});
    MathSlerp.Init("MathSlerp", CPH::FastArg<PyObject *>{"q1", true}, CPH::FastArg<PyObject *>{"q2", true}, CPH::FastArg<float>{"t", true});
    MathQuatPair.Init("MathQuatPair", CPH::FastArg<PyObject *>{"a", true}, CPH::FastArg<PyObject *>{"b", true});
    MathLerpBatch.Init("MathLerpBatch", CPH::FastArg<PyObject *>{"vecs_a", true}, CPH::FastArg<PyObject *>{"vecs_b", true}, CPH::FastArg<float>{"alpha", true});
    MathQuatVec.Init("MathQuatVec", CPH::FastArg<PyObject *>{"q", true}, CPH::FastArg<PyObject *>{"v", true});
    MathQuatVecBatch.Init("MathQuatVecBatch", CPH::FastArg<PyObject *>{"q", true}, CPH::FastArg<PyObject *>{"vecs", true});
    MathQuatOp.Init("MathQuatOp", CPH::FastArg<PyObject *>{"q", true});
    MathProject.Init("MathProject", CPH::FastArg<PyObject *>{"v", true}, CPH::FastArg<PyObject *>{"mvp", true}, CPH::FastArg<PyObject *>{"viewport", true});
    MathUnproject.Init("MathUnproject", CPH::FastArg<PyObject *>{"v", true}, CPH::FastArg<PyObject *>{"mvp", true}, CPH::FastArg<PyObject *>{"viewport", true});
    MathVecPair.Init("MathVecPair", CPH::FastArg<PyObject *>{"v1", true}, CPH::FastArg<PyObject *>{"v2", true});
    MathRayPlane.Init("MathRayPlane", CPH::FastArg<PyObject *>{"ray_origin", true}, CPH::FastArg<PyObject *>{"ray_dir", true}, CPH::FastArg<PyObject *>{"plane_pos", true}, CPH::FastArg<PyObject *>{"plane_norm", true});
    MathAxisAngle.Init("MathAxisAngle", CPH::FastArg<PyObject *>{"axis", true}, CPH::FastArg<float>{"angle", true});
    MathDistBatch.Init("MathDistBatch", CPH::FastArg<PyObject *>{"vecs_a", true}, CPH::FastArg<PyObject *>{"vecs_b", true});
    MathVecOp.Init("MathVecOp", CPH::FastArg<PyObject *>{"v", true});
    MathReflect.Init("MathReflect", CPH::FastArg<PyObject *>{"v", true}, CPH::FastArg<PyObject *>{"normal", true});
    MathEulerVec.Init("MathEulerVec", CPH::FastArg<PyObject *>{"euler", true});
    MathEulerBatch.Init("MathEulerBatch", CPH::FastArg<PyObject *>{"eulers", true});
    CPH::detail::RegisterParsers(registry, registry_count, MathPersp, MathOrtho, MathLookAt, MathTRS, MathTRSBatch, MathMat, MathMatPair, MathMatVec, MathMatBatch, MathCull, MathCullBatch, MathVec3Batch, MathEuler, MathQuat, MathSlerp, MathQuatPair, MathLerpBatch, MathQuatVec, MathQuatVecBatch, MathQuatOp, MathProject, MathUnproject, MathVecPair, MathRayPlane, MathAxisAngle, MathDistBatch, MathVecOp, MathReflect, MathEulerVec, MathEulerBatch);
}

void CPH_ModuleParsers::Init() {
    registry_count = 0;
    DumpSchema.Init("DumpSchema", CPH::FastArg<const char *>{"path", false});
    CPH::detail::RegisterParsers(registry, registry_count, DumpSchema);
}
