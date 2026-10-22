#pragma once

#include "physics/PhysicsWorldState.hpp"

// Mirrors active Jolt bodies into the dense shadow buffers (previous <=
// current <= Jolt). Called under jph_trampoline_lock during a step.

namespace CPH {

[[gnu::flatten, gnu::hot]]
void SyncShadowBuffers(const PhysicsWorld *world) noexcept;

} // namespace CPH
