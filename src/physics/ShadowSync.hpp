#pragma once

#include "physics/PhysicsWorldState.hpp"

// Mirrors active Jolt bodies into the dense shadow buffers (previous <=
// current <= Jolt). Called under jph_trampoline_lock during a step.
[[gnu::flatten, gnu::hot]]
void CPH_SyncShadowBuffers(const CPH_PhysicsWorld *world) noexcept;
