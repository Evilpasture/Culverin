#pragma once

#include "bindings/parsing/FastParser.hpp"

namespace CPH::Python {

struct CharacterParsers {
    ParserCatalog catalog{};

    Parser<"CharMove", Arg<"velocity", Vec3f, true>, Arg<"dt", float, true>> CharMove{catalog};
    Parser<"SetPosChar", Arg<"pos", PosStride, true>> SetPosChar{catalog};
    Parser<"SetRotChar", Arg<"rot", AuxStride, true>> SetRotChar{catalog};
    Parser<"SetStrengthChar", Arg<"strength", float, true>> SetStrengthChar{catalog};
};

} // namespace CPH::Python
