#pragma once

#include "physics/PhysicsWorldState.hpp"

// Mirrors active Jolt bodies into the dense shadow buffers (previous <=
// current <= Jolt). Called under jph_trampoline_lock during a step.

namespace CPH {

// Mirrors active Jolt bodies into the canonical dense shadow buffers. Runs on
// the step thread during a step with no shadowLock held -- precisely why
// zero-copy readers must NOT point at these arrays (see TransformBuffer).
[[gnu::flatten, gnu::hot]]
void SyncShadowBuffers(const PhysicsWorld *world) noexcept;

// Snapshots the canonical transform channels into the back frame of the
// world's TransformBuffer and publishes it (one atomic index flip). Called on
// the step thread immediately after SyncShadowBuffers, so every external
// reader sees a single coherent frame and never a torn write.
void PublishTransforms(PhysicsWorld *world) noexcept;

} // namespace CPH
