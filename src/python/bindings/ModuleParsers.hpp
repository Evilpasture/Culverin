#pragma once

#include "bindings/parsing/Parser.hpp"

namespace CPH::Python {

struct ModuleParsers {
    ParserCatalog catalog{};

    Parser<"DumpSchema", Arg<"path", const char *, false>> dumpSchema{catalog};
};

} // namespace CPH::Python
