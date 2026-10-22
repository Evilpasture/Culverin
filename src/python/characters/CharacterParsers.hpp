#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct CharacterParsers {
    ParserCatalog catalog{};

    Parser<"CharMove", Arg<"velocity", JPH::Float3, true>, Arg<"dt", float, true>> charMove{catalog};
    Parser<"SetPosChar", Arg<"pos", PosStride, true>> setPosChar{catalog};
    Parser<"SetRotChar", Arg<"rot", AuxStride, true>> setRotChar{catalog};
    Parser<"SetStrengthChar", Arg<"strength", float, true>> setStrengthChar{catalog};
};

} // namespace CPH::Python
