#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct WorldParsers {
    ParserCatalog catalog{};

    Parser<"WorldInit", Arg<"settings", PyObject *, false>, Arg<"bodies", PyObject *, false>>
        WorldInit{catalog};
    Parser<"Body", Arg<"pos", PyObject *, false>, Arg<"rot", PyObject *, false>,
           Arg<"size", PyObject *, false>, Arg<"shape", int, false>, Arg<"motion", int, false>,
           Arg<"user_data", uint64_t, false>, Arg<"is_sensor", bool, false>,
           Arg<"mass", float, false>, Arg<"category", uint32_t, false>,
           Arg<"mask", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>, Arg<"material_id", uint32_t, false>,
           Arg<"ccd", bool, false>, Arg<"allowed_dofs", uint32_t, false>>
        Body{catalog};
    Parser<"Impulse", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        Impulse{catalog};
    Parser<"WheelIdx", Arg<"index", uint32_t, true>> WheelIdx{catalog};
    Parser<"AngImpulse", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        AngImpulse{catalog};
    Parser<"Force", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        Force{catalog};
    Parser<"Torque", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        Torque{catalog};
    Parser<"SetLinVel", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        SetLinVel{catalog};
    Parser<"SetAngVel", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        SetAngVel{catalog};
    Parser<"ImpulseAt", Arg<"handle", ParseBodyHandle, true>, Arg<"ix", float, true>,
           Arg<"iy", float, true>, Arg<"iz", float, true>, Arg<"px", JPH_Real, true>,
           Arg<"py", JPH_Real, true>, Arg<"pz", JPH_Real, true>>
        ImpulseAt{catalog};
    Parser<"HOnly", Arg<"handle", ParseBodyHandle, true>> HOnly{catalog};
    Parser<"Destroy", Arg<"handle", ParseBodyHandle, true>> Destroy{catalog};
    Parser<"Activate", Arg<"handle", ParseBodyHandle, true>> Activate{catalog};
    Parser<"Gravity", Arg<"x", float, true>, Arg<"y", float, true>, Arg<"z", float, true>> Gravity{
        catalog};
    Parser<"SetPos", Arg<"handle", ParseBodyHandle, true>, Arg<"x", JPH_Real, true>,
           Arg<"y", JPH_Real, true>, Arg<"z", JPH_Real, true>>
        SetPos{catalog};
    Parser<"Buoy", Arg<"handle", ParseBodyHandle, true>, Arg<"surface_y", double, true>,
           Arg<"buoyancy", float, false>, Arg<"linear_drag", float, false>,
           Arg<"angular_drag", float, false>, Arg<"dt", float, false>,
           Arg<"fluid_velocity", PyObject *, false>>
        Buoy{catalog};
    Parser<"BatchBuoy", Arg<"handles", PyObject *, true>, Arg<"surface_y", double, true>,
           Arg<"buoyancy", float, false>, Arg<"linear_drag", float, false>,
           Arg<"angular_drag", float, false>, Arg<"dt", float, false>,
           Arg<"fluid_velocity", PyObject *, false>>
        BatchBuoy{catalog};
    Parser<"Mesh", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"vertices", PyObject *, true>, Arg<"indices", PyObject *, true>,
           Arg<"user_data", uint64_t, false>, Arg<"category", uint32_t, false>,
           Arg<"mask", uint32_t, false>>
        Mesh{catalog};
    Parser<"SetTrns", Arg<"handle", ParseBodyHandle, true>, Arg<"pos", PyObject *, true>,
           Arg<"rot", PyObject *, true>>
        SetTrns{catalog};
    Parser<"CCD", Arg<"handle", ParseBodyHandle, true>, Arg<"enabled", bool, true>> CCD{catalog};
    Parser<"ConvexHull", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"points", PyObject *, true>, Arg<"motion", int, false>, Arg<"mass", float, false>,
           Arg<"user_data", uint64_t, false>, Arg<"is_sensor", bool, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>, Arg<"ccd", bool, false>>
        ConvexHull{catalog};
    Parser<"Compound", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"parts", PyObject *, true>, Arg<"motion", int, false>, Arg<"mass", float, false>,
           Arg<"user_data", uint64_t, false>, Arg<"is_sensor", bool, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>, Arg<"ccd", bool, false>>
        Compound{catalog};
    Parser<"BatchCreate", Arg<"positions", PyObject *, true>, Arg<"sizes", PyObject *, true>,
           Arg<"shape_type", int, false>, Arg<"motion_type", int, false>>
        BatchCreate{catalog};
    Parser<"BatchDestroy", Arg<"handles", PyObject *, true>> BatchDestroy{catalog};
    Parser<"SetRot", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>, Arg<"w", float, true>>
        SetRot{catalog};
    Parser<"Render", Arg<"alpha", float, true>> Render{catalog};
    Parser<"Raycast", Arg<"start", PyObject *, true>, Arg<"direction", PyObject *, true>,
           Arg<"max_dist", float, false>, Arg<"ignore", ParseBodyHandle, false>>
        Raycast{catalog};
    Parser<"RayBatch", Arg<"starts", PyObject *, true>, Arg<"directions", PyObject *, true>,
           Arg<"max_dist", float, false>>
        RayBatch{catalog};
    Parser<"Shapecast", Arg<"shape", int, true>, Arg<"pos", PyObject *, true>,
           Arg<"rot", PyObject *, true>, Arg<"dir", PyObject *, true>,
           Arg<"size", PyObject *, false>, Arg<"ignore", ParseBodyHandle, false>>
        Shapecast{catalog};
    Parser<"OverlapSphere", Arg<"center", PyObject *, true>, Arg<"radius", float, true>>
        OverlapSphere{catalog};
    Parser<"OverlapAABB", Arg<"min", PyObject *, true>, Arg<"max", PyObject *, true>> OverlapAABB{
        catalog};
    Parser<"SetUserData", Arg<"handle", ParseBodyHandle, true>, Arg<"data", uint64_t, true>>
        SetUserData{catalog};
    Parser<"GetUserData", Arg<"handle", ParseBodyHandle, true>> GetUserData{catalog};
    Parser<"GetMotion", Arg<"handle", ParseBodyHandle, true>> GetMotion{catalog};
    Parser<"SetMotion", Arg<"handle", ParseBodyHandle, true>, Arg<"motion", int, true>> SetMotion{
        catalog};
    Parser<"ColFilter", Arg<"handle", ParseBodyHandle, true>, Arg<"category", uint32_t, true>,
           Arg<"mask", uint32_t, true>>
        ColFilter{catalog};
    Parser<"RegMat", Arg<"id", uint32_t, true>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>>
        RegMat{catalog};
    Parser<"SetConstrTarget", Arg<"handle", uint64_t, true>, Arg<"target", float, true>>
        SetConstrTarget{catalog};
    Parser<"Heightfield", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"scale", PyObject *, true>, Arg<"heights", PyObject *, true>,
           Arg<"grid_size", int, true>, Arg<"user_data", uint64_t, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>>
        Heightfield{catalog};
    Parser<"DebugData", Arg<"shapes", bool, false>, Arg<"constraints", bool, false>,
           Arg<"bbox", bool, false>, Arg<"centers", bool, false>, Arg<"wireframe", bool, false>>
        DebugData{catalog};
    Parser<"CreateConstr", Arg<"type", int, true>, Arg<"body1", ParseBodyHandle, true>,
           Arg<"body2", ParseBodyHandle, true>, Arg<"params", PyObject *, false>,
           Arg<"motor", PyObject *, false>>
        CreateConstr{catalog};
    Parser<"DestroyConstr", Arg<"handle", ParseBodyHandle, true>> DestroyConstr{catalog};
    Parser<"Step", Arg<"dt", float, false>> Step{catalog};
    Parser<"CreateChar", Arg<"pos", PosStride, true>, Arg<"height", float, false>,
           Arg<"radius", float, false>, Arg<"step_height", float, false>,
           Arg<"max_slope", float, false>>
        CreateChar{catalog};
    Parser<"LoadState", Arg<"state", PyObject *, true>> LoadState{catalog};
    Parser<"CreateVehicle", Arg<"chassis", uint64_t, true>, Arg<"wheels", PyObject *, true>,
           Arg<"drive", PyObject *, false>, Arg<"engine", PyObject *, false>,
           Arg<"transmission", PyObject *, false>>
        CreateVehicle{catalog};
    Parser<"CreateTracked", Arg<"chassis", uint64_t, true>, Arg<"wheels", PyObject *, true>,
           Arg<"tracks", PyObject *, true>, Arg<"max_torque", float, false>,
           Arg<"max_rpm", float, false>>
        CreateTracked{catalog};
    Parser<"CreateRagdoll", Arg<"settings", PyObject *, true>, Arg<"pos", PosStride, true>,
           Arg<"rot", AuxStride, false>, Arg<"user_data", uint64_t, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>>
        CreateRagdoll{catalog};
    Parser<"RagdollSettings", Arg<"skeleton", PyObject *, true>> RagdollSettings{catalog};
    Parser<"CreateSoftBody", Arg<"shared_settings", PyObject *, true>, Arg<"pos", PyObject *, true>,
           Arg<"rot", PyObject *, true>, Arg<"user_data", uint64_t, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"pressure", float, false>, Arg<"vertex_radius", float, false>,
           Arg<"linear_damping", float, false>, Arg<"num_iterations", uint32_t, false>,
           Arg<"max_linear_velocity", float, false>, Arg<"gravity_factor", float, false>,
           Arg<"friction", float, false>, Arg<"restitution", float, false>,
           Arg<"make_rotation_identity", bool, false>, Arg<"update_position", bool, false>,
           Arg<"faces_double_sided", bool, false>>
        CreateSoftBody{catalog};
    Parser<"GetSbVertex", Arg<"handle", ParseBodyHandle, true>, Arg<"index", uint32_t, true>>
        GetSbVertex{catalog};
    Parser<"StressTest", Arg<"a0", uint64_t, false>, Arg<"a1", uint64_t, false>,
           Arg<"a2", uint64_t, false>, Arg<"a3", uint64_t, false>, Arg<"a4", uint64_t, false>,
           Arg<"a5", uint64_t, false>, Arg<"a6", uint64_t, false>, Arg<"a7", uint64_t, false>,
           Arg<"a8", uint64_t, false>, Arg<"a9", uint64_t, false>, Arg<"a10", uint64_t, false>,
           Arg<"a11", uint64_t, false>, Arg<"a12", uint64_t, false>, Arg<"a13", uint64_t, false>,
           Arg<"a14", uint64_t, false>, Arg<"a15", uint64_t, false>, Arg<"a16", uint64_t, false>,
           Arg<"a17", uint64_t, false>, Arg<"a18", uint64_t, false>, Arg<"a19", uint64_t, false>,
           Arg<"a20", uint64_t, false>, Arg<"a21", uint64_t, false>, Arg<"a22", uint64_t, false>,
           Arg<"a23", uint64_t, false>, Arg<"a24", uint64_t, false>, Arg<"a25", uint64_t, false>,
           Arg<"a26", uint64_t, false>, Arg<"a27", uint64_t, false>, Arg<"a28", uint64_t, false>,
           Arg<"a29", uint64_t, false>, Arg<"a30", uint64_t, false>, Arg<"a31", uint64_t, false>,
           Arg<"a32", uint64_t, false>, Arg<"a33", uint64_t, false>, Arg<"a34", uint64_t, false>,
           Arg<"a35", uint64_t, false>, Arg<"a36", uint64_t, false>, Arg<"a37", uint64_t, false>,
           Arg<"a38", uint64_t, false>, Arg<"a39", uint64_t, false>, Arg<"a40", uint64_t, false>,
           Arg<"a41", uint64_t, false>, Arg<"a42", uint64_t, false>, Arg<"a43", uint64_t, false>,
           Arg<"a44", uint64_t, false>, Arg<"a45", uint64_t, false>, Arg<"a46", uint64_t, false>,
           Arg<"a47", uint64_t, false>, Arg<"a48", uint64_t, false>, Arg<"a49", uint64_t, false>,
           Arg<"a50", uint64_t, false>, Arg<"a51", uint64_t, false>, Arg<"a52", uint64_t, false>,
           Arg<"a53", uint64_t, false>, Arg<"a54", uint64_t, false>, Arg<"a55", uint64_t, false>,
           Arg<"a56", uint64_t, false>, Arg<"a57", uint64_t, false>, Arg<"a58", uint64_t, false>,
           Arg<"a59", uint64_t, false>, Arg<"a60", uint64_t, false>, Arg<"a61", uint64_t, false>,
           Arg<"a62", uint64_t, false>, Arg<"a63", uint64_t, false>>
        StressTest{catalog};
    Parser<"CreateShip", Arg<"sled", ParseBodyHandle, true>, Arg<"kp", float, true>,
           Arg<"kd", float, true>, Arg<"throttle_force", float, true>,
           Arg<"steer_speed", float, true>, Arg<"banking", float, false>,
           Arg<"lateral_grip", float, false>, Arg<"linear_drag", float, false>>
        CreateShip{catalog};
};

} // namespace CPH::Python
