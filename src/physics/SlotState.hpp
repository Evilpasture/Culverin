#pragma once

// Slot lifecycle state machine. A slot is the sparse, stable identity of a
// body; its state decides which operations may run immediately against Jolt
// (body exists), must be deferred to the command queue (body is being
// created), or must be rejected.

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

[[nodiscard]] constexpr std::uint32_t slot_state_bit(SlotState state) noexcept {
    return 1U << std::to_underlying(state);
}

// Bit masks over SlotState values, one per admission policy.
// States in which a deferred command may resolve a body ID (flush guard).
inline constexpr std::uint32_t kValidBidMask =
    slot_state_bit(SlotState::Alive) | slot_state_bit(SlotState::PendingCreate) |
    slot_state_bit(SlotState::PendingDestroy) | slot_state_bit(SlotState::Character) |
    slot_state_bit(SlotState::SoftBody);
inline constexpr std::uint32_t kMaskImmStandard =
    slot_state_bit(SlotState::Alive) | slot_state_bit(SlotState::Character) |
    slot_state_bit(SlotState::SoftBody);
inline constexpr std::uint32_t kMaskImmStrict =
    slot_state_bit(SlotState::Alive) | slot_state_bit(SlotState::SoftBody);
inline constexpr std::uint32_t kMaskDeferred = slot_state_bit(SlotState::PendingCreate);
inline constexpr std::uint32_t kMaskDestructible =
    slot_state_bit(SlotState::Alive) | slot_state_bit(SlotState::PendingCreate) |
    slot_state_bit(SlotState::Character) | slot_state_bit(SlotState::SoftBody);

struct SlotPredicate {
    bool is_immediate; // Body exists in Jolt: act now.
    bool is_deferred;  // Body pending creation: queue for the step flush.

    [[nodiscard]] constexpr bool is_executable() const noexcept {
        return is_immediate || is_deferred;
    }
};

[[nodiscard]] constexpr SlotPredicate
get_slot_predicate(std::uint8_t state, std::uint32_t imm_mask) noexcept {
    const std::uint32_t state_bit = 1U << (state & 7);
    return {.is_immediate = (state_bit & imm_mask) != 0,
            .is_deferred  = (state_bit & kMaskDeferred) != 0};
}

} // namespace CPH
