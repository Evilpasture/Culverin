#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct ShipParsers {
    ParserCatalog catalog{};

    Parser<"ShipInput", Arg<"forward", float, true>, Arg<"right", float, true>> shipInput{catalog};
};

} // namespace CPH::Python
