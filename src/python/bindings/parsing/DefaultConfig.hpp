#pragma once
#include "core/Compiler.hpp"
#include <stddef.h>

namespace CPH::Python {

// Parser constants
//
// The hard ceiling on a parser's arity; Init() refuses anything wider. It is
// also the width of the provided/required bitmasks, so raising it means
// widening those too.
[[maybe_unused]] static constexpr size_t kParserMaxArgumentCount = 64;

// Cache line the core is aligned (and padded) to. Diagnostics walk
// coldSpecs while the hot path walks hotSpecs; keeping the two halves on
// whole lines is why this is not just a struct default.
[[maybe_unused]] static constexpr size_t kParserAlignment = 64;

/**
 * The number of arguments at which the parser switches from O(N) linear search
 * to O(1) hash table lookup. Below this value, the overhead of hashing and
 * table allocation exceeds the cost of a tight linear scan.
 */
[[maybe_unused]] static constexpr size_t kParserKeywordHashThreshold = 10;

} // namespace CPH::Python
