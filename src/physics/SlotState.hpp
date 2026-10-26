#pragma once

// Slot lifecycle state machine. A slot is the sparse, stable identity of a
// body; its state decides which operations are admissible against a body.
// Since the pure-async refactor, ALL admissible operations are deferred to
// the CommandMailbox (no "immediate" Jolt application from a producer), so
// these masks gate admission/execution-at-flush only -- they no longer select
// between an immediate and a deferred execution path.

#include <cstdint>
#include <utility>


namespace CPH {

enum class SlotState : std::uint8_t {
    Empty = 0,
    PendingCreate,
    Alive,
    PendingDestroy,
    Character,
    SoftBody,
    Count,
};

[[nodiscard]] constexpr std::uint32_t SlotStateBit(SlotState state) noexcept {
    return 1U << std::to_underlying(state);
}

// Bit masks over SlotState values, one per admission policy.
// States in which a deferred command may resolve a body ID (flush guard).
inline constexpr std::uint32_t kValidBidMask =
    SlotStateBit(SlotState::Alive) | SlotStateBit(SlotState::PendingCreate) |
    SlotStateBit(SlotState::PendingDestroy) | SlotStateBit(SlotState::Character) |
    SlotStateBit(SlotState::SoftBody);
inline constexpr std::uint32_t kMaskImmStandard =
    SlotStateBit(SlotState::Alive) | SlotStateBit(SlotState::Character) |
    SlotStateBit(SlotState::SoftBody);
inline constexpr std::uint32_t kMaskImmStrict =
    SlotStateBit(SlotState::Alive) | SlotStateBit(SlotState::SoftBody);
inline constexpr std::uint32_t kMaskDeferred = SlotStateBit(SlotState::PendingCreate);
inline constexpr std::uint32_t kMaskDestructible =
    SlotStateBit(SlotState::Alive) | SlotStateBit(SlotState::PendingCreate) |
    SlotStateBit(SlotState::Character) | SlotStateBit(SlotState::SoftBody);

struct SlotPredicate {
    bool isImmediate; // Body exists in Jolt: act now.
    bool isDeferred;  // Body pending creation: queue for the step flush.

    [[nodiscard]] constexpr bool is_executable() const noexcept {
        return isImmediate || isDeferred;
    }
};

[[nodiscard]] constexpr SlotPredicate
GetSlotPredicate(std::uint8_t state, std::uint32_t imm_mask) noexcept {
    const std::uint32_t state_bit = 1U << (state & 7);
    return {.isImmediate = (state_bit & imm_mask) != 0,
            .isDeferred  = (state_bit & kMaskDeferred) != 0};
}

} // namespace CPH
