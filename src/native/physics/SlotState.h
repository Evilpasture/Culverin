#pragma once

#include "physics/CommandBuffer.h"
#include <stdint.h>

CULV_NODISCARD [[gnu::const]]
static inline bool is_state_valid(uint8_t state, uint8_t mask) {
    return (bool)((1U << state) & mask);
}

typedef struct {
    culv_u1 is_immediate : 1;
    culv_u1 is_deferred : 1;
    culv_u1 is_executable : 1;
    culv_u5 _unused : 5;
} SlotPredicate;

static_assert(sizeof(SlotPredicate) == sizeof(uint8_t));

static constexpr uint32_t MASK_IMM_STANDARD =
    (1U << SLOT_ALIVE) | (1U << SLOT_CHARACTER) | (1U << SLOT_SOFT_BODY);
static constexpr uint32_t MASK_IMM_STRICT = (1U << SLOT_ALIVE) | (1U << SLOT_SOFT_BODY);
static constexpr uint32_t MASK_DEFERRED = (1U << SLOT_PENDING_CREATE);
static constexpr uint32_t MASK_DESTRUCTIBLE =
    (1U << SLOT_ALIVE) | (1U << SLOT_PENDING_CREATE) | (1U << SLOT_CHARACTER) |
    (1U << SLOT_SOFT_BODY);

[[gnu::const]] CULV_NODISCARD static CULV_FORCE_INLINE SlotPredicate
get_slot_predicate(uint8_t state, uint32_t imm_mask) {
    const uint32_t state_bit = 1U << (state & 7);

    const culv_u1 imm = (culv_u1)!!(bool)(state_bit & imm_mask);
    const culv_u1 def = (culv_u1)!!(bool)(state_bit & MASK_DEFERRED);

    return (SlotPredicate){.is_immediate = imm,
                           .is_deferred = def,
                           .is_executable = (culv_u1)(imm | def),
                           ._unused = 0};
}
