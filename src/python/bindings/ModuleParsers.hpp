#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct ModuleParsers {
    ParserCatalog catalog{};

    Parser<"DumpSchema", Arg<"path", const char *, false>> DumpSchema{catalog};
};

} // namespace CPH::Python
