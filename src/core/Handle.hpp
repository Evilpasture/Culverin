#pragma once

// Strongly-typed generational handles.
//
// A handle is a 64-bit value laid out as (generation << 32) | slot index.
// That layout is the wire format: handles cross the Python boundary as plain
// integers, and the ECS mirror of a body handle stores the same number, so
// the representation must not change. What changes here is the C++ side.
// BodyHandle and ConstraintHandle used to be `using = std::uint64_t`, which
// made a body handle and a constraint handle the same type: one could be
// passed where the other was expected, and any integer -- including a slot
// index or a generation on its own -- would silently convert to a handle.
// They are now distinct class types, so mixing them is a compile error and
// building one from a raw integer requires saying so.
//
// Generation 0 is never issued: every slot table initialises its generations
// to 1, and a retired slot bumps to a new non-zero generation. A
// default-constructed handle is therefore raw 0 and genuinely invalid, which
// is why IsValid() can be a plain != 0 test.

#include <compare>
#include <cstdint>
#include <type_traits>

namespace CPH {

template <class Tag>
class GenerationalHandle {
public:
    using ValueType = std::uint64_t;

    static constexpr std::uint32_t kIndexBits = 32;
    static constexpr ValueType kIndexMask     = 0xFFFFFFFFULL;

    // Raw 0: never issued, so this is the invalid handle.
    constexpr GenerationalHandle() noexcept = default;

    // Explicit on purpose. Raw 64-bit values arrive from Python and from
    // stored buffers; wrapping one must be a visible act, not an implicit
    // conversion (which is what the old uint64_t alias allowed).
    constexpr explicit GenerationalHandle(ValueType raw) noexcept : raw_(raw) {}

    // The only supported way to build a live handle.
    [[nodiscard]] static constexpr GenerationalHandle
    FromParts(std::uint32_t slot, std::uint32_t gen) noexcept {
        return GenerationalHandle((static_cast<ValueType>(gen) << kIndexBits) | slot);
    }

    // Slot-table index.
    [[nodiscard]] constexpr std::uint32_t Index() const noexcept {
        return static_cast<std::uint32_t>(raw_ & kIndexMask);
    }

    // Generation counter. 0 means "never issued".
    [[nodiscard]] constexpr std::uint32_t Generation() const noexcept {
        return static_cast<std::uint32_t>(raw_ >> kIndexBits);
    }

    // The 64-bit value Python sees and buffers store.
    [[nodiscard]] constexpr ValueType Raw() const noexcept { return raw_; }

    [[nodiscard]] constexpr bool IsValid() const noexcept { return raw_ != 0; }
    constexpr explicit operator bool() const noexcept { return IsValid(); }

    friend constexpr bool operator==(GenerationalHandle lhs, GenerationalHandle rhs) noexcept {
        return lhs.raw_ == rhs.raw_;
    }
    friend constexpr bool operator!=(GenerationalHandle lhs, GenerationalHandle rhs) noexcept {
        return lhs.raw_ != rhs.raw_;
    }
    friend constexpr auto operator<=>(GenerationalHandle lhs, GenerationalHandle rhs) noexcept {
        return lhs.raw_ <=> rhs.raw_;
    }

private:
    ValueType raw_ = 0;
};

// Only the tag distinguishes the instantiations; tags are never constructed.
struct BodyHandleTag {};
struct ConstraintHandleTag {};
struct EntityTag {};

using BodyHandle       = GenerationalHandle<BodyHandleTag>;
using ConstraintHandle = GenerationalHandle<ConstraintHandleTag>;

// Recognises any instantiation. Generic code (the Python value builder) needs
// this to handle a handle without naming every tag.
template <class T> struct IsGenerationalHandle : std::false_type {};
template <class Tag> struct IsGenerationalHandle<GenerationalHandle<Tag>> : std::true_type {};
template <class T> inline constexpr bool kIsGenerationalHandle = IsGenerationalHandle<T>::value;

} // namespace CPH
