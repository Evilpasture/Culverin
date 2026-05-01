#pragma once
#include "CulverinCPP"

#if defined(_MSC_VER)
#    include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#    include <xmmintrin.h>
#endif

namespace CPH {
enum class CacheLevel : Unsigned8 { L1 = 3, L2 = 2, L3 = 1, stream = 0 };
enum class AccessType : Unsigned8 { Read = 0, Write = 1 };

template <AccessType Access = AccessType::Read, CacheLevel Level = CacheLevel::L1>
[[gnu::always_inline]] inline void Prefetch(const void *addr) noexcept {
#if defined(__clang__) || defined(__GNUC__)
    __builtin_prefetch(addr, static_cast<int>(Access), static_cast<Integer32>(Level));
#elif defined(_MSC_VER)
    // MSVC doesn't have a direct 1:1 for __builtin_prefetch's rw param
    if constexpr (Access == AccessType::Write) {
#    if defined(_M_X64) || defined(_M_IX86)
        _m_prefetchw(const_cast<void *>(addr)); // Included via <intrin.h>
#    else
        _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
#    endif
    } else {
        if constexpr (Level == CacheLevel::L1)
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T0);
        else if constexpr (Level == CacheLevel::L2)
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_T1);
        else
            _mm_prefetch(static_cast<const char *>(addr), _MM_HINT_NTA);
    }
#endif
}
} // namespace CPH