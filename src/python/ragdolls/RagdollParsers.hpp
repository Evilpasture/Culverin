#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct SkeletonParsers {
    ParserCatalog catalog{};

    Parser<"AddJoint", Arg<"name", PyObject *, true>, Arg<"parent_index", int, false>> AddJoint{
        catalog};
    Parser<"GetJointIdx", Arg<"name", PyObject *, true>> GetJointIdx{catalog};
};

struct RagdollParsers {
    ParserCatalog catalog{};

    Parser<"RagdollDrive", Arg<"root_pos", CPH_PosStride, true>,
           Arg<"root_rot", CPH_AuxStride, true>, Arg<"matrices", PyObject *, true>>
        RagdollDrive{catalog};
};

struct RagdollSettingsParsers {
    ParserCatalog catalog{};

    Parser<"RagdollAddPart", Arg<"joint_index", int, true>, Arg<"shape_type", int, true>,
           Arg<"size", PyObject *, true>, Arg<"mass", float, false>,
           Arg<"parent_index", int, false>, Arg<"twist_min", float, false>,
           Arg<"twist_max", float, false>, Arg<"cone_angle", float, false>,
           Arg<"axis", CPH_Vec3f, false>, Arg<"normal", CPH_Vec3f, false>,
           Arg<"pos", PyObject *, false>>
        RagdollAddPart{catalog};
};

} // namespace CPH::Python
