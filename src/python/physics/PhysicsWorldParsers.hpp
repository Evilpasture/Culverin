#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct WorldParsers {
    ParserCatalog catalog{};

    Parser<"WorldInit", Arg<"settings", PyObject *, false>, Arg<"bodies", PyObject *, false>>
        worldInit{catalog};
    Parser<"Body", Arg<"pos", PyObject *, false>, Arg<"rot", PyObject *, false>,
           Arg<"size", PyObject *, false>, Arg<"shape", int, false>, Arg<"motion", int, false>,
           Arg<"user_data", uint64_t, false>, Arg<"is_sensor", bool, false>,
           Arg<"mass", float, false>, Arg<"category", uint32_t, false>,
           Arg<"mask", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>, Arg<"material_id", uint32_t, false>,
           Arg<"ccd", bool, false>, Arg<"allowed_dofs", uint32_t, false>>
        body{catalog};
    Parser<"Impulse", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        impulse{catalog};
    Parser<"WheelIdx", Arg<"index", uint32_t, true>> wheelIdx{catalog};
    Parser<"AngImpulse", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        angImpulse{catalog};
    Parser<"Force", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        force{catalog};
    Parser<"Torque", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        torque{catalog};
    Parser<"SetLinVel", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        setLinVel{catalog};
    Parser<"SetAngVel", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>>
        setAngVel{catalog};
    Parser<"ImpulseAt", Arg<"handle", ParseBodyHandle, true>, Arg<"ix", float, true>,
           Arg<"iy", float, true>, Arg<"iz", float, true>, Arg<"px", JPH::Real, true>,
           Arg<"py", JPH::Real, true>, Arg<"pz", JPH::Real, true>>
        impulseAt{catalog};
    Parser<"HOnly", Arg<"handle", ParseBodyHandle, true>> hOnly{catalog};
    Parser<"Destroy", Arg<"handle", ParseBodyHandle, true>> destroy{catalog};
    Parser<"Activate", Arg<"handle", ParseBodyHandle, true>> activate{catalog};
    Parser<"Gravity", Arg<"x", float, true>, Arg<"y", float, true>, Arg<"z", float, true>> gravity{
        catalog};
    Parser<"SetPos", Arg<"handle", ParseBodyHandle, true>, Arg<"x", JPH::Real, true>,
           Arg<"y", JPH::Real, true>, Arg<"z", JPH::Real, true>>
        setPos{catalog};
    Parser<"Buoy", Arg<"handle", ParseBodyHandle, true>, Arg<"surface_y", double, true>,
           Arg<"buoyancy", float, false>, Arg<"linear_drag", float, false>,
           Arg<"angular_drag", float, false>, Arg<"dt", float, false>,
           Arg<"fluid_velocity", PyObject *, false>>
        buoy{catalog};
    Parser<"BatchBuoy", Arg<"handles", PyObject *, true>, Arg<"surface_y", double, true>,
           Arg<"buoyancy", float, false>, Arg<"linear_drag", float, false>,
           Arg<"angular_drag", float, false>, Arg<"dt", float, false>,
           Arg<"fluid_velocity", PyObject *, false>>
        batchBuoy{catalog};
    Parser<"Mesh", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"vertices", PyObject *, true>, Arg<"indices", PyObject *, true>,
           Arg<"user_data", uint64_t, false>, Arg<"category", uint32_t, false>,
           Arg<"mask", uint32_t, false>>
        mesh{catalog};
    Parser<"SetTrns", Arg<"handle", ParseBodyHandle, true>, Arg<"pos", PyObject *, true>,
           Arg<"rot", PyObject *, true>>
        setTrns{catalog};
    Parser<"CCD", Arg<"handle", ParseBodyHandle, true>, Arg<"enabled", bool, true>> ccd{catalog};
    Parser<"ConvexHull", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"points", PyObject *, true>, Arg<"motion", int, false>, Arg<"mass", float, false>,
           Arg<"user_data", uint64_t, false>, Arg<"is_sensor", bool, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>, Arg<"ccd", bool, false>>
        convexHull{catalog};
    Parser<"Compound", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"parts", PyObject *, true>, Arg<"motion", int, false>, Arg<"mass", float, false>,
           Arg<"user_data", uint64_t, false>, Arg<"is_sensor", bool, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>, Arg<"ccd", bool, false>>
        compound{catalog};
    Parser<"BatchCreate", Arg<"positions", PyObject *, true>, Arg<"sizes", PyObject *, true>,
           Arg<"shape_type", int, false>, Arg<"motion_type", int, false>>
        batchCreate{catalog};
    Parser<"BatchDestroy", Arg<"handles", PyObject *, true>> batchDestroy{catalog};
    Parser<"SetRot", Arg<"handle", ParseBodyHandle, true>, Arg<"x", float, true>,
           Arg<"y", float, true>, Arg<"z", float, true>, Arg<"w", float, true>>
        setRot{catalog};
    Parser<"Render", Arg<"alpha", float, true>> render{catalog};
    Parser<"Raycast", Arg<"start", PyObject *, true>, Arg<"direction", PyObject *, true>,
           Arg<"max_dist", float, false>, Arg<"ignore", ParseBodyHandle, false>>
        raycast{catalog};
    Parser<"RayBatch", Arg<"starts", PyObject *, true>, Arg<"directions", PyObject *, true>,
           Arg<"max_dist", float, false>>
        rayBatch{catalog};
    Parser<"Shapecast", Arg<"shape", int, true>, Arg<"pos", PyObject *, true>,
           Arg<"rot", PyObject *, true>, Arg<"dir", PyObject *, true>,
           Arg<"size", PyObject *, false>, Arg<"ignore", ParseBodyHandle, false>>
        shapecast{catalog};
    Parser<"OverlapSphere", Arg<"center", PyObject *, true>, Arg<"radius", float, true>>
        overlapSphere{catalog};
    Parser<"OverlapAABB", Arg<"min", PyObject *, true>, Arg<"max", PyObject *, true>> overlapAabb{
        catalog};
    Parser<"SetUserData", Arg<"handle", ParseBodyHandle, true>, Arg<"data", uint64_t, true>>
        setUserData{catalog};
    Parser<"GetUserData", Arg<"handle", ParseBodyHandle, true>> getUserData{catalog};
    Parser<"GetMotion", Arg<"handle", ParseBodyHandle, true>> getMotion{catalog};
    Parser<"SetMotion", Arg<"handle", ParseBodyHandle, true>, Arg<"motion", int, true>> setMotion{
        catalog};
    Parser<"ColFilter", Arg<"handle", ParseBodyHandle, true>, Arg<"category", uint32_t, true>,
           Arg<"mask", uint32_t, true>>
        colFilter{catalog};
    Parser<"RegMat", Arg<"id", uint32_t, true>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>>
        regMat{catalog};
    Parser<"SetConstrTarget", Arg<"handle", uint64_t, true>, Arg<"target", float, true>>
        setConstrTarget{catalog};
    Parser<"Heightfield", Arg<"pos", PyObject *, true>, Arg<"rot", PyObject *, true>,
           Arg<"scale", PyObject *, true>, Arg<"heights", PyObject *, true>,
           Arg<"grid_size", int, true>, Arg<"user_data", uint64_t, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>, Arg<"friction", float, false>,
           Arg<"restitution", float, false>>
        heightfield{catalog};
    Parser<"DebugData", Arg<"shapes", bool, false>, Arg<"constraints", bool, false>,
           Arg<"bbox", bool, false>, Arg<"centers", bool, false>, Arg<"wireframe", bool, false>>
        debugData{catalog};
    Parser<"CreateConstr", Arg<"type", int, true>, Arg<"body1", ParseBodyHandle, true>,
           Arg<"body2", ParseBodyHandle, true>, Arg<"params", PyObject *, false>,
           Arg<"motor", PyObject *, false>>
        createConstr{catalog};
    Parser<"DestroyConstr", Arg<"handle", ParseBodyHandle, true>> destroyConstr{catalog};
    Parser<"Step", Arg<"dt", float, false>> step{catalog};
    Parser<"CreateChar", Arg<"pos", PosStride, true>, Arg<"height", float, false>,
           Arg<"radius", float, false>, Arg<"step_height", float, false>,
           Arg<"max_slope", float, false>>
        createChar{catalog};
    Parser<"LoadState", Arg<"state", PyObject *, true>> loadState{catalog};
    Parser<"CreateVehicle", Arg<"chassis", uint64_t, true>, Arg<"wheels", PyObject *, true>,
           Arg<"drive", PyObject *, false>, Arg<"engine", PyObject *, false>,
           Arg<"transmission", PyObject *, false>>
        createVehicle{catalog};
    Parser<"CreateTracked", Arg<"chassis", uint64_t, true>, Arg<"wheels", PyObject *, true>,
           Arg<"tracks", PyObject *, true>, Arg<"max_torque", float, false>,
           Arg<"max_rpm", float, false>>
        createTracked{catalog};
    Parser<"CreateRagdoll", Arg<"settings", PyObject *, true>, Arg<"pos", PosStride, true>,
           Arg<"rot", AuxStride, false>, Arg<"user_data", uint64_t, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"material_id", uint32_t, false>>
        createRagdoll{catalog};
    Parser<"RagdollSettings", Arg<"skeleton", PyObject *, true>> ragdollSettings{catalog};
    Parser<"CreateSoftBody", Arg<"shared_settings", PyObject *, true>, Arg<"pos", PyObject *, true>,
           Arg<"rot", PyObject *, true>, Arg<"user_data", uint64_t, false>,
           Arg<"category", uint32_t, false>, Arg<"mask", uint32_t, false>,
           Arg<"pressure", float, false>, Arg<"vertex_radius", float, false>,
           Arg<"linear_damping", float, false>, Arg<"num_iterations", uint32_t, false>,
           Arg<"max_linear_velocity", float, false>, Arg<"gravity_factor", float, false>,
           Arg<"friction", float, false>, Arg<"restitution", float, false>,
           Arg<"make_rotation_identity", bool, false>, Arg<"update_position", bool, false>,
           Arg<"faces_double_sided", bool, false>>
        createSoftBody{catalog};
    Parser<"GetSbVertex", Arg<"handle", ParseBodyHandle, true>, Arg<"index", uint32_t, true>>
        getSbVertex{catalog};
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
        stressTest{catalog};
    Parser<"CreateShip", Arg<"sled", ParseBodyHandle, true>, Arg<"kp", float, true>,
           Arg<"kd", float, true>, Arg<"throttle_force", float, true>,
           Arg<"steer_speed", float, true>, Arg<"banking", float, false>,
           Arg<"lateral_grip", float, false>, Arg<"linear_drag", float, false>>
        createShip{catalog};
};

} // namespace CPH::Python
