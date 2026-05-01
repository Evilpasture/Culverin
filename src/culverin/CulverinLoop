#pragma once
#include "CulverinCPP"
#include <type_traits>
#include <utility>

namespace CPH {
template <SizeType N, typename F> constexpr void Unroll(F &&f) {
    [&f]<SizeType... Is>(std::index_sequence<Is...>) -> auto {
        (f(std::integral_constant<SizeType, Is>{}), ...);
    }(std::make_index_sequence<N>{});
}

template <typename T, SizeType N, typename F> constexpr void Unroll(F &&f) {
    constexpr SizeType MAX_UNROLL = (sizeof(T) > 32) ? 4 : 8;
    constexpr SizeType ActualN    = (N > MAX_UNROLL) ? MAX_UNROLL : N;

    [&f]<SizeType... Is>(std::index_sequence<Is...>) -> auto {
        (f(std::integral_constant<SizeType, Is>{}), ...);
    }(std::make_index_sequence<ActualN>{});
}

template <SizeType Factor, typename F> constexpr void UnrollLoop(SizeType total, F &&f) {
    SizeType i = 0;
    for (; i + Factor <= total; i += Factor) {
        Unroll<Factor>([&](auto index) -> auto { f(i + index); });
    }
    for (; i < total; ++i) {
        f(i);
    }
}

template <SizeType N, typename F> constexpr void Repeat(F &&f) {
    [&f]<SizeType... Is>(std::index_sequence<Is...>) -> auto {
        ((static_cast<void>(Is), f()), ...);
    }(std::make_index_sequence<N>{});
}
} // namespace CPH
