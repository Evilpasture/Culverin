#pragma once

#include "bindings/parsing/Parser.hpp"

namespace CPH::Python {

struct MathParsers {
    ParserCatalog catalog{};

    Parser<"MathPersp", Arg<"fovy", float, true>, Arg<"aspect", float, true>,
           Arg<"near", float, true>, Arg<"far", float, true>>
        mathPersp{catalog};
    Parser<"MathOrtho", Arg<"left", float, true>, Arg<"right", float, true>,
           Arg<"bottom", float, true>, Arg<"top", float, true>, Arg<"near", float, true>,
           Arg<"far", float, true>>
        mathOrtho{catalog};
    Parser<"MathLookAt", Arg<"eye", PyObject *, true>, Arg<"target", PyObject *, true>,
           Arg<"up", PyObject *, true>>
        mathLookAt{catalog};
    Parser<"MathTRS", Arg<"translation", PyObject *, true>, Arg<"rotation", PyObject *, true>,
           Arg<"scale", PyObject *, true>>
        mathTrs{catalog};
    Parser<"MathTRSBatch", Arg<"translations", PyObject *, true>,
           Arg<"rotations", PyObject *, true>, Arg<"scales", PyObject *, true>>
        mathTrsBatch{catalog};
    Parser<"MathMat", Arg<"mat", PyObject *, true>> mathMat{catalog};
    Parser<"MathMatPair", Arg<"a", PyObject *, true>, Arg<"b", PyObject *, true>> mathMatPair{
        catalog};
    Parser<"MathMatVec", Arg<"mat", PyObject *, true>, Arg<"vec", PyObject *, true>> mathMatVec{
        catalog};
    Parser<"MathMatBatch", Arg<"mat", PyObject *, true>, Arg<"batch", PyObject *, true>>
        mathMatBatch{catalog};
    Parser<"MathCull", Arg<"vp", PyObject *, true>, Arg<"min", PyObject *, true>,
           Arg<"max", PyObject *, true>>
        mathCull{catalog};
    Parser<"MathCullBatch", Arg<"vp", PyObject *, true>, Arg<"aabbs", PyObject *, true>>
        mathCullBatch{catalog};
    Parser<"MathVec3Batch", Arg<"vecs", PyObject *, true>> mathVec3Batch{catalog};
    Parser<"MathEuler", Arg<"x", float, true>, Arg<"y", float, true>, Arg<"z", float, true>>
        mathEuler{catalog};
    Parser<"MathQuat", Arg<"x", float, true>, Arg<"y", float, true>, Arg<"z", float, true>,
           Arg<"w", float, true>>
        mathQuat{catalog};
    Parser<"MathSlerp", Arg<"q1", PyObject *, true>, Arg<"q2", PyObject *, true>,
           Arg<"t", float, true>>
        mathSlerp{catalog};
    Parser<"MathQuatPair", Arg<"a", PyObject *, true>, Arg<"b", PyObject *, true>> mathQuatPair{
        catalog};
    Parser<"MathLerpBatch", Arg<"vecs_a", PyObject *, true>, Arg<"vecs_b", PyObject *, true>,
           Arg<"alpha", float, true>>
        mathLerpBatch{catalog};
    Parser<"MathQuatVec", Arg<"q", PyObject *, true>, Arg<"v", PyObject *, true>> mathQuatVec{
        catalog};
    Parser<"MathQuatVecBatch", Arg<"q", PyObject *, true>, Arg<"vecs", PyObject *, true>>
        mathQuatVecBatch{catalog};
    Parser<"MathQuatOp", Arg<"q", PyObject *, true>> mathQuatOp{catalog};
    Parser<"MathProject", Arg<"v", PyObject *, true>, Arg<"mvp", PyObject *, true>,
           Arg<"viewport", PyObject *, true>>
        mathProject{catalog};
    Parser<"MathUnproject", Arg<"v", PyObject *, true>, Arg<"mvp", PyObject *, true>,
           Arg<"viewport", PyObject *, true>>
        mathUnproject{catalog};
    Parser<"MathVecPair", Arg<"v1", PyObject *, true>, Arg<"v2", PyObject *, true>> mathVecPair{
        catalog};
    Parser<"MathRayPlane", Arg<"ray_origin", PyObject *, true>, Arg<"ray_dir", PyObject *, true>,
           Arg<"plane_pos", PyObject *, true>, Arg<"plane_norm", PyObject *, true>>
        mathRayPlane{catalog};
    Parser<"MathAxisAngle", Arg<"axis", PyObject *, true>, Arg<"angle", float, true>> mathAxisAngle{
        catalog};
    Parser<"MathDistBatch", Arg<"vecs_a", PyObject *, true>, Arg<"vecs_b", PyObject *, true>>
        mathDistBatch{catalog};
    Parser<"MathVecOp", Arg<"v", PyObject *, true>> mathVecOp{catalog};
    Parser<"MathReflect", Arg<"v", PyObject *, true>, Arg<"normal", PyObject *, true>> mathReflect{
        catalog};
    Parser<"MathEulerVec", Arg<"euler", PyObject *, true>> mathEulerVec{catalog};
    Parser<"MathEulerBatch", Arg<"eulers", PyObject *, true>> mathEulerBatch{catalog};
};

} // namespace CPH::Python
