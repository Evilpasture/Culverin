#pragma once

#include "bindings/parsing/Parser.hpp"

namespace CPH::Python {

struct EcsParsers {
    ParserCatalog catalog{};

    Parser<"RegEntityOnly", Arg<"entity", uint64_t, true>> regEntityOnly{catalog};
    Parser<"RegCompOnly", Arg<"comp_id", uint32_t, true>> regCompOnly{catalog};
    Parser<"RegRegComp", Arg<"size_bytes", uint32_t, true>> regRegComp{catalog};
    Parser<"RegAdd", Arg<"entity", uint64_t, true>, Arg<"comp_id", uint32_t, true>,
           Arg<"data", PyObject *, false>>
        regAdd{catalog};
    Parser<"RegEntComp", Arg<"entity", uint64_t, true>, Arg<"comp_id", uint32_t, true>> regEntComp{
        catalog};
    Parser<"RegSyncPhys", Arg<"world", PyObject *, true>, Arg<"handle_comp_id", uint32_t, true>,
           Arg<"transform_comp_id", int, true>, Arg<"rot_comp_id", int, false>>
        regSyncPhys{catalog};
};

} // namespace CPH::Python
