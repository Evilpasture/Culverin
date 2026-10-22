#pragma once

#include "bindings/parsing/Parser.hpp"

namespace CPH::Python {

struct SkeletonParsers {
    ParserCatalog catalog{};

    Parser<"AddJoint", Arg<"name", PyObject *, true>, Arg<"parent_index", int, false>> addJoint{
        catalog};
    Parser<"GetJointIdx", Arg<"name", PyObject *, true>> getJointIdx{catalog};
};

struct RagdollParsers {
    ParserCatalog catalog{};

    Parser<"RagdollDrive", Arg<"root_pos", PosStride, true>,
           Arg<"root_rot", AuxStride, true>, Arg<"matrices", PyObject *, true>>
        ragdollDrive{catalog};
};

struct RagdollSettingsParsers {
    ParserCatalog catalog{};

    Parser<"RagdollAddPart", Arg<"joint_index", int, true>, Arg<"shape_type", int, true>,
           Arg<"size", PyObject *, true>, Arg<"mass", float, false>,
           Arg<"parent_index", int, false>, Arg<"twist_min", float, false>,
           Arg<"twist_max", float, false>, Arg<"cone_angle", float, false>,
           Arg<"axis", JPH::Float3, false>, Arg<"normal", JPH::Float3, false>,
           Arg<"pos", PyObject *, false>>
        ragdollAddPart{catalog};
};

} // namespace CPH::Python
