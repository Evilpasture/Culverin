#pragma once

#include "jolt/JoltConfig.h"
#include "core/CompilerSpecifics.h"
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>

#ifndef JPH_INVALID_BODY_ID
CULV_MAYBE_UNUSED static constexpr size_t JPH_INVALID_BODY_ID = 0xFFFFFFFF;
#endif

// Jolt BodyID layout: [8 bits sequence | 24 bits index]. The low 23 bits hold
// the index because the top bit of the low 24-bit portion is used for flags.
#ifndef JPH_BODY_ID_INDEX_MASK
CULV_MAYBE_UNUSED static constexpr size_t JPH_BODY_ID_INDEX_MASK = 0x00FFFFFF;
#endif

CULV_NODISCARD [[gnu::const]]
static inline uint32_t JPH_ID_TO_INDEX(uint32_t id) {
    static constexpr uint32_t ID_TO_INDEX_MASK = 0x007FFFFF;
    return id & ID_TO_INDEX_MASK;
}

// Allocate a Jolt value type on the stack with guaranteed 32-byte alignment.
// USAGE: JPH_STACK_ALLOC(JPH_RVec3, my_vec);
#if defined(__clang__) || defined(__GNUC__)
#    define JPH_ALIGNED_STORAGE(Type, Name, Align) Type Name __attribute__((aligned(Align)))
#elif defined(_MSC_VER)
#    define JPH_ALIGNED_STORAGE(Type, Name, Align) __declspec(align(Align)) Type Name
#else
#    include <stdalign.h>
#    define JPH_ALIGNED_STORAGE(Type, Name, Align) alignas(Align) Type Name
#endif

#define JPH_STACK_ALLOC(Type, Name)                                                                \
    JPH_ALIGNED_STORAGE(Type, Name##_storage, 32);                                                 \
    Type *Name = &Name##_storage
