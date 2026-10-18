#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct EcsParsers {
    ParserCatalog catalog{};

    Parser<"RegEntityOnly", Arg<"entity", uint64_t, true>> RegEntityOnly{catalog};
    Parser<"RegCompOnly", Arg<"comp_id", uint32_t, true>> RegCompOnly{catalog};
    Parser<"RegRegComp", Arg<"size_bytes", uint32_t, true>> RegRegComp{catalog};
    Parser<"RegAdd", Arg<"entity", uint64_t, true>, Arg<"comp_id", uint32_t, true>,
           Arg<"data", PyObject *, false>>
        RegAdd{catalog};
    Parser<"RegEntComp", Arg<"entity", uint64_t, true>, Arg<"comp_id", uint32_t, true>> RegEntComp{
        catalog};
    Parser<"RegSyncPhys", Arg<"world", PyObject *, true>, Arg<"handle_comp_id", uint32_t, true>,
           Arg<"transform_comp_id", int, true>, Arg<"rot_comp_id", int, false>>
        RegSyncPhys{catalog};
};

} // namespace CPH::Python
