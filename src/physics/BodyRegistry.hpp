#pragma once

// Fixed-capacity, lock-free body registry.
//
// The legacy design grew the per-body slot tables with Resize() and kept a
// "retired" copy alive for one tick so raw pointers into the previous epoch
// would not dangle. Jolt already fixes max_bodies at PhysicsSystem::Init, so
// growing past that cap is impossible; the tables can be allocated once, up
// front, and never move. Once they never move, resolving a generational
// handle no longer needs a lock: it is a bounds check plus one acquire load
// of the slot's generation.
//
// BodyRegistry is a non-owning *view* over the slot tables that BodyBuffers
// owns. That is what makes it cheap and composable: it introduces no second
// source of truth (the same arrays the SIMD shadow sync walks are the arrays
// Resolve consults), and it is only safe because the fixed-capacity guarantee
// means those arrays are never reallocated for the world's lifetime.
//
// Dependency-light (core/Handle.hpp only); generation recycling and
// fixed-capacity handle validity are exercised through the Python API in
// tests/test_subsystem_ownership.py and tests/test_core.py.

#include "core/Handle.hpp"

#include <atomic>
#include <cstdint>
#include <optional>

namespace CPH {

class BodyRegistry {
  public:
    // Bind to externally-owned, fixed-capacity slot tables. All pointers must
    // remain valid for the registry's lifetime (they do: capacity is frozen).
    void Bind(const std::atomic<std::uint32_t> *generations,
              const std::atomic<std::uint8_t> *states, const std::uint32_t *slotToDense,
              std::uint32_t capacity) noexcept {
        generations_ = generations;
        states_      = states;
        slotToDense_ = slotToDense;
        capacity_    = capacity;
    }

    [[nodiscard]] bool IsBound() const noexcept { return generations_ != nullptr; }
    [[nodiscard]] std::uint32_t Capacity() const noexcept { return capacity_; }

    // Zero-lock handle resolution. A handle to a recycled slot carries a stale
    // generation and resolves to nullopt even though its index is in range.
    // acquire pairs with the release-store that retires a slot in
    // RemoveBodySlot(), so a freshly reused slot is never mistaken for the old
    // body.
    [[nodiscard]] std::optional<std::uint32_t> Resolve(BodyHandle handle) const noexcept {
        const std::uint32_t slot = handle.Index();
        const std::uint32_t gen  = handle.Generation();
        if (slot >= capacity_) [[unlikely]] {
            return std::nullopt;
        }
        if (generations_[slot].load(std::memory_order::acquire) != gen) {
            return std::nullopt;
        }
        return slot;
    }

    [[nodiscard]] std::uint8_t State(std::uint32_t slot) const noexcept {
        return states_[slot].load(std::memory_order::acquire);
    }
    [[nodiscard]] std::uint32_t Generation(std::uint32_t slot) const noexcept {
        return generations_[slot].load(std::memory_order::acquire);
    }
    [[nodiscard]] std::uint32_t Dense(std::uint32_t slot) const noexcept {
        return slotToDense_[slot];
    }

  private:
    const std::atomic<std::uint32_t> *generations_ = nullptr;
    const std::atomic<std::uint8_t> *states_       = nullptr;
    const std::uint32_t *slotToDense_              = nullptr;
    std::uint32_t capacity_                        = 0;
};

} // namespace CPH
