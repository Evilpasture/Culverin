#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct SoftBodySharedSettingsParsers {
    ParserCatalog catalog{};

    Parser<"SbssAddVertex", Arg<"pos", PyObject *, true>, Arg<"inv_mass", float, false>,
           Arg<"velocity", PyObject *, false>>
        SbssAddVertex{catalog};
    Parser<"SbssAddFace", Arg<"v1", uint32_t, true>, Arg<"v2", uint32_t, true>,
           Arg<"v3", uint32_t, true>, Arg<"material_index", uint32_t, false>>
        SbssAddFace{catalog};
    Parser<"SbssAddVertices", Arg<"positions", PyObject *, true>,
           Arg<"inv_masses", PyObject *, false>, Arg<"velocities", PyObject *, false>>
        SbssAddVertices{catalog};
    Parser<"SbssAddFaces", Arg<"indices", PyObject *, true>, Arg<"materials", PyObject *, false>>
        SbssAddFaces{catalog};
    Parser<"SbssCreateConstraints", Arg<"compliance", float, true>, Arg<"bend_type", int, false>>
        SbssCreateConstraints{catalog};
};

} // namespace CPH::Python
