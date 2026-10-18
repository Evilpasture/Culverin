#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct MathParsers {
    ParserCatalog catalog{};

    Parser<"MathPersp", Arg<"fovy", float, true>, Arg<"aspect", float, true>,
           Arg<"near", float, true>, Arg<"far", float, true>>
        MathPersp{catalog};
    Parser<"MathOrtho", Arg<"left", float, true>, Arg<"right", float, true>,
           Arg<"bottom", float, true>, Arg<"top", float, true>, Arg<"near", float, true>,
           Arg<"far", float, true>>
        MathOrtho{catalog};
    Parser<"MathLookAt", Arg<"eye", PyObject *, true>, Arg<"target", PyObject *, true>,
           Arg<"up", PyObject *, true>>
        MathLookAt{catalog};
    Parser<"MathTRS", Arg<"translation", PyObject *, true>, Arg<"rotation", PyObject *, true>,
           Arg<"scale", PyObject *, true>>
        MathTRS{catalog};
    Parser<"MathTRSBatch", Arg<"translations", PyObject *, true>,
           Arg<"rotations", PyObject *, true>, Arg<"scales", PyObject *, true>>
        MathTRSBatch{catalog};
    Parser<"MathMat", Arg<"mat", PyObject *, true>> MathMat{catalog};
    Parser<"MathMatPair", Arg<"a", PyObject *, true>, Arg<"b", PyObject *, true>> MathMatPair{
        catalog};
    Parser<"MathMatVec", Arg<"mat", PyObject *, true>, Arg<"vec", PyObject *, true>> MathMatVec{
        catalog};
    Parser<"MathMatBatch", Arg<"mat", PyObject *, true>, Arg<"batch", PyObject *, true>>
        MathMatBatch{catalog};
    Parser<"MathCull", Arg<"vp", PyObject *, true>, Arg<"min", PyObject *, true>,
           Arg<"max", PyObject *, true>>
        MathCull{catalog};
    Parser<"MathCullBatch", Arg<"vp", PyObject *, true>, Arg<"aabbs", PyObject *, true>>
        MathCullBatch{catalog};
    Parser<"MathVec3Batch", Arg<"vecs", PyObject *, true>> MathVec3Batch{catalog};
    Parser<"MathEuler", Arg<"x", float, true>, Arg<"y", float, true>, Arg<"z", float, true>>
        MathEuler{catalog};
    Parser<"MathQuat", Arg<"x", float, true>, Arg<"y", float, true>, Arg<"z", float, true>,
           Arg<"w", float, true>>
        MathQuat{catalog};
    Parser<"MathSlerp", Arg<"q1", PyObject *, true>, Arg<"q2", PyObject *, true>,
           Arg<"t", float, true>>
        MathSlerp{catalog};
    Parser<"MathQuatPair", Arg<"a", PyObject *, true>, Arg<"b", PyObject *, true>> MathQuatPair{
        catalog};
    Parser<"MathLerpBatch", Arg<"vecs_a", PyObject *, true>, Arg<"vecs_b", PyObject *, true>,
           Arg<"alpha", float, true>>
        MathLerpBatch{catalog};
    Parser<"MathQuatVec", Arg<"q", PyObject *, true>, Arg<"v", PyObject *, true>> MathQuatVec{
        catalog};
    Parser<"MathQuatVecBatch", Arg<"q", PyObject *, true>, Arg<"vecs", PyObject *, true>>
        MathQuatVecBatch{catalog};
    Parser<"MathQuatOp", Arg<"q", PyObject *, true>> MathQuatOp{catalog};
    Parser<"MathProject", Arg<"v", PyObject *, true>, Arg<"mvp", PyObject *, true>,
           Arg<"viewport", PyObject *, true>>
        MathProject{catalog};
    Parser<"MathUnproject", Arg<"v", PyObject *, true>, Arg<"mvp", PyObject *, true>,
           Arg<"viewport", PyObject *, true>>
        MathUnproject{catalog};
    Parser<"MathVecPair", Arg<"v1", PyObject *, true>, Arg<"v2", PyObject *, true>> MathVecPair{
        catalog};
    Parser<"MathRayPlane", Arg<"ray_origin", PyObject *, true>, Arg<"ray_dir", PyObject *, true>,
           Arg<"plane_pos", PyObject *, true>, Arg<"plane_norm", PyObject *, true>>
        MathRayPlane{catalog};
    Parser<"MathAxisAngle", Arg<"axis", PyObject *, true>, Arg<"angle", float, true>> MathAxisAngle{
        catalog};
    Parser<"MathDistBatch", Arg<"vecs_a", PyObject *, true>, Arg<"vecs_b", PyObject *, true>>
        MathDistBatch{catalog};
    Parser<"MathVecOp", Arg<"v", PyObject *, true>> MathVecOp{catalog};
    Parser<"MathReflect", Arg<"v", PyObject *, true>, Arg<"normal", PyObject *, true>> MathReflect{
        catalog};
    Parser<"MathEulerVec", Arg<"euler", PyObject *, true>> MathEulerVec{catalog};
    Parser<"MathEulerBatch", Arg<"eulers", PyObject *, true>> MathEulerBatch{catalog};
};

} // namespace CPH::Python
