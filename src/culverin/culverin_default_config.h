#pragma once

// Parser constants
static constexpr size_t MAX_ARG_LIMIT = 64;
/**
 * The number of arguments at which the parser switches from O(N) linear search
 * to O(1) hash table lookup. Below this value, the overhead of hashing and
 * table allocation exceeds the cost of a tight linear scan.
 */
static constexpr size_t FP_HASH_THRESHOLD = 10;
