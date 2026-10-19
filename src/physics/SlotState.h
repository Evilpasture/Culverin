#pragma once

#include "physics/CommandBuffer.h"
#include <stdint.h>

[[nodiscard]] [[gnu::const]]
static inline bool is_state_valid(uint8_t state, uint8_t mask) {
    return (bool)((1U << state) & mask);
}

typedef struct {
    CPH_u1 is_immediate : 1;
    CPH_u1 is_deferred : 1;
    CPH_u1 is_executable : 1;
    CPH_u5 _unused : 5;
} CPH_SlotPredicate;

static_assert(sizeof(CPH_SlotPredicate) == sizeof(uint8_t));

static constexpr uint32_t kMaskImmStandard =
    (1U << SLOT_ALIVE) | (1U << SLOT_CHARACTER) | (1U << SLOT_SOFT_BODY);
static constexpr uint32_t kMaskImmStrict = (1U << SLOT_ALIVE) | (1U << SLOT_SOFT_BODY);
static constexpr uint32_t kMaskDeferred = (1U << SLOT_PENDING_CREATE);
static constexpr uint32_t kMaskDestructible =
    (1U << SLOT_ALIVE) | (1U << SLOT_PENDING_CREATE) | (1U << SLOT_CHARACTER) |
    (1U << SLOT_SOFT_BODY);

[[gnu::const]] [[nodiscard]] static CPH_FORCE_INLINE CPH_SlotPredicate
get_slot_predicate(uint8_t state, uint32_t imm_mask) {
    const uint32_t state_bit = 1U << (state & 7);

    const CPH_u1 imm = (CPH_u1)!!(bool)(state_bit & imm_mask);
    const CPH_u1 def = (CPH_u1)!!(bool)(state_bit & kMaskDeferred);

    return (CPH_SlotPredicate){.is_immediate = imm,
                           .is_deferred = def,
                           .is_executable = (CPH_u1)(imm | def),
                           ._unused = 0};
}
