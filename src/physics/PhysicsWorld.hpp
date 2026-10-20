#pragma once

// Free helpers whose subject is a settings value rather than a world,
// plus the module-init ABI self-check. World operations are declared
// as PhysicsWorld members in PhysicsWorldState.hpp.

#include "core/Types.hpp"
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <joltc.h>


namespace CPH {

void ConfigureBodySettings(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                               BodyConfig config);
void ApplyBodyCreationProps(JPH::BodyCreationSettings &settings, const JPH::Shape *shape,
                                BodyCreationProps props);

int VerifyAbiAlignment(JPH_BodyInterface *body_interface);

} // namespace CPH
