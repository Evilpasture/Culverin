#pragma once

#include <Python.h>

#include <cstddef>
#include <optional>
#include <span>

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
///   * `Bytes()` / `Float32()` are the only accessors, and they are typed: no
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

    /// The whole buffer as float32. Only meaningful where the exporter really
    /// holds floats -- that is the contract of every packed-float entry point
    /// in this layer, and it is the caller's to uphold.
    [[nodiscard]] std::span<const float> Float32() const noexcept {
        const std::span<const std::byte> bytes = Bytes();
        return {reinterpret_cast<const float *>(bytes.data()), bytes.size() / sizeof(float)};
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

  private:
    BufferView() = default;

    void Reset() noexcept {
        if (view_.obj != nullptr) {
            PyBuffer_Release(&view_);
        }
        view_.obj = nullptr;
    }

    [[nodiscard]] std::size_t SizeBytes() const noexcept {
        return view_.len > 0 ? static_cast<std::size_t>(view_.len) : 0;
    }

    Py_buffer view_{};
};

inline std::optional<BufferView> BufferView::Acquire(PyObject *owner) noexcept {
    BufferView acquired;
    if (owner == nullptr || PyObject_GetBuffer(owner, &acquired.view_, PyBUF_SIMPLE) < 0) {
        return std::nullopt;
    }
    return std::optional<BufferView>(std::move(acquired));
}

} // namespace CPH::Python
