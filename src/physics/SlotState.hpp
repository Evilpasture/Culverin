#pragma once

// Slot lifecycle state machine. A slot is the sparse, stable identity of a
// body; its state decides which operations may run immediately against Jolt
// (body exists), must be deferred to the command queue (body is being
// created), or must be rejected.

#include <cstdint>

enum SlotState : std::uint8_t {
    SLOT_EMPTY = 0,
    SLOT_PENDING_CREATE,
    SLOT_ALIVE,
    SLOT_PENDING_DESTROY,
    SLOT_CHARACTER,
    SLOT_SOFT_BODY,
    SLOT_COUNT,
};

// Bit masks over SlotState values, one per admission policy.
// States in which a deferred command may resolve a body ID (flush guard).
inline constexpr std::uint32_t kValidBidMask =
    (1U << SLOT_ALIVE) | (1U << SLOT_PENDING_CREATE) | (1U << SLOT_PENDING_DESTROY) |
    (1U << SLOT_CHARACTER) | (1U << SLOT_SOFT_BODY);
inline constexpr std::uint32_t kMaskImmStandard =
    (1U << SLOT_ALIVE) | (1U << SLOT_CHARACTER) | (1U << SLOT_SOFT_BODY);
inline constexpr std::uint32_t kMaskImmStrict = (1U << SLOT_ALIVE) | (1U << SLOT_SOFT_BODY);
inline constexpr std::uint32_t kMaskDeferred  = (1U << SLOT_PENDING_CREATE);
inline constexpr std::uint32_t kMaskDestructible =
    (1U << SLOT_ALIVE) | (1U << SLOT_PENDING_CREATE) | (1U << SLOT_CHARACTER) |
    (1U << SLOT_SOFT_BODY);

struct CPH_SlotPredicate {
    bool is_immediate; // Body exists in Jolt: act now.
    bool is_deferred;  // Body pending creation: queue for the step flush.

    [[nodiscard]] constexpr bool is_executable() const noexcept {
        return is_immediate || is_deferred;
    }
};

[[nodiscard]] constexpr CPH_SlotPredicate
get_slot_predicate(std::uint8_t state, std::uint32_t imm_mask) noexcept {
    const std::uint32_t state_bit = 1U << (state & 7);
    return {.is_immediate = (state_bit & imm_mask) != 0,
            .is_deferred  = (state_bit & kMaskDeferred) != 0};
}
