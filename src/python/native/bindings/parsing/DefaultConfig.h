#pragma once
#include <stddef.h>
#include "core/CompilerSpecifics.h"

// Parser constants
[[maybe_unused]] static constexpr size_t kMaxArgLimit = 64;
/**
 * The number of arguments at which the parser switches from O(N) linear search
 * to O(1) hash table lookup. Below this value, the overhead of hashing and
 * table allocation exceeds the cost of a tight linear scan.
 */
[[maybe_unused]] static constexpr size_t kFastParseHashThreshold = 10;
