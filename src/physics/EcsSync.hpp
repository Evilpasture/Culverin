#pragma once

// The one place where the ECS and the physics world meet: copying body
// transforms into components. It lives in src/physics/ because it is the only
// code that needs to see both halves; src/ecs/ stays free of Jolt entirely,
// which is what lets the registry be reused without a physics world at all.

#include "core/Error.hpp"
#include "ecs/EcsTypes.hpp"
#include <cstdint>
#include <expected>

namespace CPH {

struct PhysicsWorld;
class Registry;

struct EcsSyncConfig {
    std::uint32_t handle_component   = 0;
    int           position_component = -1; // <0 means "do not sync positions"
    int           rotation_component = -1; // <0 means "do not sync rotations"
};

// Walks the handle component, resolves each body handle and writes the live
// position and/or rotation into the matching components. Bodies whose handles
// have gone stale are skipped rather than removed: pruning is the caller's
// policy, not this function's.
[[nodiscard]] std::expected<void, Error> SyncEcsTransforms(PhysicsWorld      &world,
                                                            Registry         &registry,
                                                            const EcsSyncConfig &config);

} // namespace CPH
