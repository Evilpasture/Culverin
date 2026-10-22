#pragma once

#include "bindings/parsing/Parser.hpp"

namespace CPH::Python {

struct VehicleParsers {
    ParserCatalog catalog{};

    Parser<"VehicleInput", Arg<"forward", float, false>, Arg<"right", float, false>,
           Arg<"brake", float, false>, Arg<"handbrake", float, false>>
        vehicleInput{catalog};
    Parser<"TankInput", Arg<"left", float, true>, Arg<"right", float, true>,
           Arg<"brake", float, false>>
        tankInput{catalog};
};

} // namespace CPH::Python
