#pragma once

#include <Python.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

namespace CPH::Python {

/// Owning, typed, RAII view over a PEP 3118 buffer exported by a Python object.
///
/// Tell, don't ask. Every call site used to acquire a Py_buffer into a local,
/// interrogate `.len` and `.buf` by hand, cast the void *, and then remember to
/// call PyBuffer_Release on each of the three or four exit paths -- with the
/// releases hand-duplicated after every failed acquisition and every rejected
/// length. Handing the raw struct back would only move that work elsewhere, so
/// this type does it:
///
///   * `Acquire` owns the acquire-and-report step: on failure it returns
///     nothing, with the exception already set by the exporter's slot;
///   * the destructor releases the buffer on every path, including the early
///     `return nullptr` after a rejected shape, so the hand-written Release
///     chains are gone;
///   * `Require...` is how a call site states its shape contract. The view
///     checks it and raises the caller's message verbatim, so the text stays
///     the caller's diagnostic rather than something this type invented;
///   * `Bytes()` / `As<T>()` (and the `Float32()` / `Uint32()` /
///     `Uint64()` shorthands) are the only accessors, and they are typed: no
///     call site sees a Py_buffer, a `buf`, a `len`, or a `void *`.
///
/// The data is contiguous (PyBUF_SIMPLE) for as long as the exporter holds it;
/// the spans die with this view, so they must not outlive it.
class BufferView final {
  public:
    /// Acquires `owner`'s buffer through its `getbuffer` slot.
    [[nodiscard]] static std::optional<BufferView> Acquire(PyObject *owner) noexcept;

    ~BufferView() { Reset(); }

    BufferView(BufferView &&donor) noexcept : view_(donor.view_) { donor.view_.obj = nullptr; }

    BufferView &operator=(BufferView &&donor) noexcept {
        if (this != &donor) {
            Reset();
            view_           = donor.view_;
            donor.view_.obj = nullptr;
        }
        return *this;
    }

    BufferView(const BufferView &)            = delete;
    BufferView &operator=(const BufferView &) = delete;

    /// The whole buffer, in bytes.
    [[nodiscard]] std::span<const std::byte> Bytes() const noexcept {
        if (view_.buf == nullptr) {
            return {};
        }
        return {static_cast<const std::byte *>(view_.buf), SizeBytes()};
    }

    /// The buffer as `T`. Like any reinterpretation of a foreign buffer this
    /// carries the exporter's alignment with it, exactly as the
    /// `static_cast<T *>(view.buf)` it replaces did: a buffer that is not
    /// aligned for `T` is still the exporter's bug, not one this view can fix
    /// without copying.
    template <typename T> [[nodiscard]] std::span<const T> As() const noexcept {
        static_assert(std::is_trivially_copyable_v<T>,
                      "BufferView::As<T> reinterprets; T must be trivially copyable");
        const std::span<const std::byte> bytes = Bytes();
        return {reinterpret_cast<const T *>(bytes.data()), bytes.size() / sizeof(T)};
    }

    /// The whole buffer as float32. Only meaningful where the exporter really
    /// holds floats -- that is the contract of every packed-float entry point
    /// in this layer, and it is the caller's to uphold.
    [[nodiscard]] std::span<const float> Float32() const noexcept { return As<float>(); }

    [[nodiscard]] std::span<const std::uint32_t> Uint32() const noexcept {
        return As<std::uint32_t>();
    }

    [[nodiscard]] std::span<const std::uint64_t> Uint64() const noexcept {
        return As<std::uint64_t>();
    }

    /// Length in bytes. The one thing a caller genuinely cannot be told, since
    /// every count and every "expected N, got M" diagnostic is derived from it.
    [[nodiscard]] std::size_t SizeBytes() const noexcept {
        return view_.len > 0 ? static_cast<std::size_t>(view_.len) : 0;
    }

    /// Tells the view its length must be a whole number of `element_bytes`
    /// (12 for packed vec3s). Raises `ValueError(message)` and returns false
    /// otherwise; the buffer is released by this object either way.
    [[nodiscard]] bool RequireMultipleOf(std::size_t                   element_bytes,
                                         const char                   *message) const noexcept {
        if (element_bytes == 0 || SizeBytes() % element_bytes == 0) {
            return true;
        }
        PyErr_SetString(PyExc_ValueError, message);
        return false;
    }

    /// Tells the view it must be exactly `expected_bytes` long, for inputs
    /// whose length is fully determined by another argument (a heightfield
    /// grid, say).
    [[nodiscard]] bool RequireExactly(std::size_t expected_bytes, const char *message)
        const noexcept {
        if (SizeBytes() == expected_bytes) {
            return true;
        }
        PyErr_SetString(PyExc_ValueError, message);
        return false;
    }

    /// Tells the two views they must describe the same number of bytes, so a
    /// paired-input entry point can walk them in lockstep.
    [[nodiscard]] bool RequireSameSizeAs(const BufferView &other,
                                         const char       *message) const noexcept {
        if (SizeBytes() == other.SizeBytes()) {
            return true;
        }
        PyErr_SetString(PyExc_ValueError, message);
        return false;
    }

    /// Tells the view it must supply at least `minimum_bytes`; otherwise it
    /// raises `ValueError` through the caller's printf format, so a diagnostic
    /// that has to name the requirement, a count and the shortfall stays
    /// word-for-word at the call site. The caller supplies every argument,
    /// including this buffer's length -- `SizeBytes()` is right there.
    [[nodiscard]] [[gnu::format(printf, 3, 4)]] bool RequireAtLeast(std::size_t minimum_bytes,
                                                                   const char *message, ...)
        const noexcept;

    /// Tells the view the caller reads it as `element_bytes`-wide elements.
    ///
    /// The predicate is "the exporter does not declare that item size *and*
    /// the length is not a whole number of them", which is what the hand
    /// written handle check did. TODO(smell): the itemsize half is close to
    /// dead -- an exporter that declares 8-byte items and a length that is not
    /// a multiple of 8 is malformed -- so this is effectively a multiple-of-N
    /// check wearing a costume. Preserved as-is because loosening it changes
    /// which buffers are rejected.
    [[nodiscard]] bool RequireElementsOf(std::size_t element_bytes, const char *message)
        const noexcept;

  private:
    BufferView() = default;

    void Reset() noexcept {
        if (view_.obj != nullptr) {
            PyBuffer_Release(&view_);
        }
        view_.obj = nullptr;
    }

    Py_buffer view_{};
};

inline bool BufferView::RequireAtLeast(std::size_t minimum_bytes, const char *message, ...)
    const noexcept {
    if (SizeBytes() >= minimum_bytes) {
        return true;
    }
    // Rebuild the caller's message and append this buffer's length. PyErr_Format
    // takes varargs, so forwarding the caller's is not portable; instead the
    // caller's format must end at the point where the length goes, which is how
    // every call site uses it.
    va_list arguments;
    va_start(arguments, message);
    PyErr_FormatV(PyExc_ValueError, message, arguments);
    va_end(arguments);
    return false;
}

inline bool BufferView::RequireElementsOf(std::size_t element_bytes, const char *message)
    const noexcept {
    if (element_bytes == 0) {
        return true;
    }
    // Accept if the exporter declares this item size, *or* the length divides
    // evenly -- the disjunction, not the conjunction. Kept verbatim from the
    // hand written check this replaces.
    const std::size_t item_size = view_.itemsize > 0 ? static_cast<std::size_t>(view_.itemsize) : 0;
    if (item_size == element_bytes || SizeBytes() % element_bytes == 0) {
        return true;
    }
    PyErr_SetString(PyExc_ValueError, message);
    return false;
}

inline std::optional<BufferView> BufferView::Acquire(PyObject *owner) noexcept {
    BufferView acquired;
    if (owner == nullptr || PyObject_GetBuffer(owner, &acquired.view_, PyBUF_SIMPLE) < 0) {
        return std::nullopt;
    }
    return std::optional<BufferView>(std::move(acquired));
}

} // namespace CPH::Python
