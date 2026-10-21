#pragma once

// CPH::Error — the one error type for native fallible APIs, built to sit in
// std::expected<T, CPH::Error>:
//
//     std::expected<BodyHandle, CPH::Error> create_body(...) {
//         if (...) return std::unexpected(WorldError::WorldFull); // enum -> CPH::Error
//         return handle;
//     }
//
// Any enum converts to CPH::Error implicitly; the enum *type* names its
// category. Rules for error enums:
//   * a fixed underlying type that fits in 32 bits, values in [1, 256);
//   * NO 0 enumerator — 0 means "no error" in every category
//     (an expected<T, Error> holding value 0 is the success state).
// The category registers itself the first time a value materializes at
// runtime and resolves enumerator names/messages from the enumerators
// themselves; define `std::string_view ErrorMessage(E)` next to an enum
// (found by ADL) to override its messages, otherwise the enumerator name
// is the message. Foreign (non-enum) C error domains can register a
// category manually via CPH::RegisterErrorCategory.
//
// Reflection is __PRETTY_FUNCTION__ parsing (GCC/Clang). ErrorCode is an
// 8-byte trivially-copyable pair (category hash, value), so CPH::Error
// costs nothing to pass around or store in std::expected.

#include <array>
#include <atomic>
#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <utility>

namespace CPH {

class Error;

struct ErrorCategory {
    std::string_view name;
    std::string_view (*to_string)(std::uint32_t value); // human message
    std::string_view (*to_name)(std::uint32_t value);   // enumerator-style name
};

namespace detail {

// FNV-1a. Category identity is Hash32(TypeName<E>()).
constexpr std::uint32_t Hash32(std::string_view str) noexcept {
    std::uint32_t hash = 2166136261U;
    for (const char c : str) {
        hash ^= static_cast<std::uint32_t>(static_cast<unsigned char>(c));
        hash *= 16777619U;
    }
    return hash;
}

// --- type/enumerator names from __PRETTY_FUNCTION__ -------------------
// GCC:   "... [with T = CPH::Foo; std::string_view = ...]"
// Clang: "... [T = CPH::Foo]"

template <typename T>
constexpr std::string_view TypeNameProbe() noexcept {
    return __PRETTY_FUNCTION__;
}

template <typename T>
constexpr std::string_view TypeName() noexcept {
    const std::string_view sig   = TypeNameProbe<T>();
    const auto             start = sig.find("T = ");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto name_start = start + 4;
    const auto end        = sig.find_first_of(";]", name_start);
    return sig.substr(name_start, end - name_start);
}

template <auto V>
constexpr std::string_view EnumeratorProbe() noexcept {
    return __PRETTY_FUNCTION__;
}

// The enumerator name for `static_cast<E>(V)`, or "" if V is not an
// enumerator of E (the probe then shows a cast expression like "(E)7",
// whose tail is not an identifier).
template <typename E, int V>
constexpr std::string_view EnumeratorName() noexcept {
    const std::string_view sig   = EnumeratorProbe<static_cast<E>(V)>();
    const auto             start = sig.find("V = ");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto value_start = start + 4;
    const auto end         = sig.find_first_of(";]", value_start);
    std::string_view name  = sig.substr(value_start, end - value_start);
    if (const auto scope = name.rfind("::"); scope != std::string_view::npos) {
        name = name.substr(scope + 2);
    }
    if (name.empty() || !(name.front() == '_' || (name.front() >= 'A' && name.front() <= 'Z') ||
                          (name.front() >= 'a' && name.front() <= 'z'))) {
        return {};
    }
    for (const char c : name) {
        if (!(c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9'))) {
            return {};
        }
    }
    return name;
}

// Enumerator names are probed in [kEnumProbeMin, kEnumProbeMax); error
// enums keep their values inside it.
inline constexpr int kEnumProbeMin = 0;
inline constexpr int kEnumProbeMax = 256;

struct EnumValueName {
    int              value;
    std::string_view name;
};

template <typename E>
constexpr std::size_t EnumValueCount() noexcept {
    return []<std::size_t... Is>(std::index_sequence<Is...>) {
        return (std::size_t{0} + ... +
                (EnumeratorName<E, int(Is) + kEnumProbeMin>().empty() ? std::size_t{0}
                                                                       : std::size_t{1}));
    }(std::make_index_sequence<kEnumProbeMax - kEnumProbeMin>{});
}

template <typename E>
constexpr auto MakeEnumNames() noexcept {
    std::array<EnumValueName, EnumValueCount<E>()> table{};
    std::size_t                                    out = 0;
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        const auto add = [&](int value, std::string_view name) {
            if (!name.empty()) {
                table[out++] = {value, name};
            }
        };
        (add(int(Is) + kEnumProbeMin, EnumeratorName<E, int(Is) + kEnumProbeMin>()), ...);
    }(std::make_index_sequence<kEnumProbeMax - kEnumProbeMin>{});
    return table;
}

template <typename E>
inline constexpr auto kEnumNames = MakeEnumNames<E>();

template <typename E>
std::string_view EnumToName(std::uint32_t value) noexcept {
    for (const auto &entry : kEnumNames<E>) {
        if (static_cast<std::uint32_t>(entry.value) == value) {
            return entry.name;
        }
    }
    return "UNKNOWN";
}

template <typename E>
concept HasErrorMessage =
    requires(E value) { { ErrorMessage(value) } -> std::convertible_to<std::string_view>; };

template <typename E>
std::string_view EnumToMessage(std::uint32_t value) {
    if constexpr (HasErrorMessage<E>) {
        return ErrorMessage(static_cast<E>(value));
    } else {
        return EnumToName<E>(value);
    }
}

// --- category registry -------------------------------------------------
// A lock-free singly-linked list of categories, prepended once per enum
// type (plus manual foreign registrations). Culverin is a single
// extension module, so a header-local list head is the whole registry —
// no cross-binary sharing needed. Entries live for the process duration.

struct CategoryNode {
    std::uint32_t                 category_hash;
    const ErrorCategory          *category;
    std::atomic<CategoryNode *>   next;
};

inline std::atomic<CategoryNode *> g_category_head{nullptr};

inline void RegisterCategoryNode(CategoryNode &node) noexcept {
    CategoryNode *head = g_category_head.load(std::memory_order_relaxed);
    do {
        node.next.store(head, std::memory_order_relaxed);
    } while (!g_category_head.compare_exchange_weak(head, &node, std::memory_order_release,
                                                    std::memory_order_relaxed));
}

inline const ErrorCategory *FindCategory(std::uint32_t category_hash) noexcept {
    for (const CategoryNode *node = g_category_head.load(std::memory_order_acquire); node;
         node                      = node->next.load(std::memory_order_acquire)) {
        if (node->category_hash == category_hash) {
            return node->category;
        }
    }
    return nullptr;
}

template <typename E>
void RegisterCategory() {
    static const ErrorCategory category{TypeName<E>(), &EnumToMessage<E>, &EnumToName<E>};
    static CategoryNode        node{Hash32(TypeName<E>()), &category, nullptr};
    static std::atomic<bool>   once{false};
    if (!once.load(std::memory_order_acquire) &&
        !once.exchange(true, std::memory_order_acq_rel)) {
        RegisterCategoryNode(node);
    }
}

// Never defined: calling it makes a constant evaluation (hence a
// consteval/constexpr ErrorCode from a zero enum value) ill-formed, and
// any accidental runtime reach a link error.
void ErrorCodeCannotBeZero() noexcept;

} // namespace detail

// (category hash, value) — implicitly constructible from any error enum.
// Default-constructed, and value 0 in general, means "no error".
class ErrorCode {
    std::uint32_t m_category = 0;
    std::uint32_t m_value    = 0;

public:
    constexpr ErrorCode() noexcept = default;

    constexpr ErrorCode(std::uint32_t category_hash, std::uint32_t value) noexcept
        : m_category(category_hash), m_value(value) {}

    template <typename E>
        requires std::is_enum_v<E>
    constexpr ErrorCode(E value) noexcept
        : m_category(detail::Hash32(detail::TypeName<E>())),
          m_value(static_cast<std::uint32_t>(static_cast<std::underlying_type_t<E>>(value))) {
        static_assert(sizeof(std::underlying_type_t<E>) <= sizeof(std::uint32_t),
                      "CPH::Error enums must fit in 32 bits (use a fixed underlying type)");
        static_assert(detail::EnumeratorName<E, 0>().empty(),
                      "CPH::Error enums must not declare a 0 enumerator (0 means no error)");
        if consteval {
            if (m_value == 0) {
                detail::ErrorCodeCannotBeZero();
            }
        } else {
#if defined(CULVERIN_DEBUG)
            if (m_value == 0) {
                std::abort(); // a zero value leaked in through a cast or deserialization
            }
#endif
            detail::RegisterCategory<E>();
        }
    }

    template <typename E>
        requires std::is_enum_v<E>
    [[nodiscard]] constexpr bool Is() const noexcept {
        return m_category == detail::Hash32(detail::TypeName<E>());
    }

    template <typename E>
        requires std::is_enum_v<E>
    [[nodiscard]] constexpr bool Is(E value) const noexcept {
        return Is<E>() && m_value == static_cast<std::uint32_t>(
                                         static_cast<std::underlying_type_t<E>>(value));
    }

    // Precondition: Is<E>().
    template <typename E>
        requires std::is_enum_v<E>
    [[nodiscard]] constexpr E As() const noexcept {
        return static_cast<E>(static_cast<std::underlying_type_t<E>>(m_value));
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept { return m_value != 0; }
    [[nodiscard]] constexpr std::uint32_t CategoryHash() const noexcept { return m_category; }
    [[nodiscard]] constexpr std::uint32_t Value() const noexcept { return m_value; }
    [[nodiscard]] Error                   ToError() const noexcept;

    [[nodiscard]] friend constexpr bool operator==(const ErrorCode &,
                                                   const ErrorCode &) noexcept = default;
};

static_assert(sizeof(ErrorCode) == 8 && std::is_standard_layout_v<ErrorCode> &&
              std::is_trivially_copyable_v<ErrorCode>);

// The error itself: an ErrorCode plus category-resolved descriptions.
// Default-constructed means no error.
class Error {
    ErrorCode m_code{};

public:
    constexpr Error() noexcept = default;
    constexpr Error(ErrorCode code) noexcept : m_code(code) {}

    template <typename E>
        requires std::is_enum_v<E>
    constexpr Error(E value) noexcept : m_code(value) {}

    [[nodiscard]] constexpr ErrorCode  Code() const noexcept { return m_code; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return static_cast<bool>(m_code);
    }

    [[nodiscard]] std::string_view Category() const noexcept {
        if (!m_code) {
            return "None";
        }
        const ErrorCategory *category = detail::FindCategory(m_code.CategoryHash());
        return category ? category->name : std::string_view{"Unknown"};
    }

    [[nodiscard]] std::string_view Name() const noexcept {
        if (!m_code) {
            return "None";
        }
        const ErrorCategory *category = detail::FindCategory(m_code.CategoryHash());
        return (category && category->to_name) ? category->to_name(m_code.Value())
                                                : std::string_view{"Unknown"};
    }

    [[nodiscard]] std::string_view Message() const {
        if (!m_code) {
            return "None";
        }
        const ErrorCategory *category = detail::FindCategory(m_code.CategoryHash());
        return (category && category->to_string) ? category->to_string(m_code.Value())
                                                  : std::string_view{"Unknown"};
    }

    [[nodiscard]] friend bool operator==(const Error &, const Error &) = default;
};

inline Error ErrorCode::ToError() const noexcept {
    return Error{*this};
}

// Registers a foreign (non-enum) error domain and returns its category
// hash, for building ErrorCodes from C-API result values. `name` must
// outlive the process (a string literal). The registration deliberately
// leaks: the registry is process-lifetime.
inline std::uint32_t RegisterErrorCategory(std::string_view                name,
                                           std::string_view (*to_string)(std::uint32_t),
                                           std::string_view (*to_name)(std::uint32_t)) {
    const std::uint32_t hash     = detail::Hash32(name);
    auto               *category = new ErrorCategory{name, to_string, to_name};
    auto               *node     = new detail::CategoryNode{hash, category, nullptr};
    detail::RegisterCategoryNode(*node);
    return hash;
}

} // namespace CPH
