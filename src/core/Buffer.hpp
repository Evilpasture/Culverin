#pragma once

// Buffer<T>: the one owning array type for native storage.
//
// Why not std::vector? The extension is built -fno-exceptions and the
// native layer reports allocation failure as status codes that the Python
// layer turns into MemoryError. std::vector cannot do that (its allocation
// failure path terminates under -fno-exceptions); Buffer allocates with
// nothrow aligned new and reports failure as `false`. It also owns the
// over-alignment (AVX 32-byte shadow buffers, 64-byte contact events)
// without a custom-allocator type infecting every signature.
//
// A Buffer is fixed-capacity; growth is an explicit reallocate() so the
// failure case stays visible at the call site. Like std::unique_ptr,
// a const Buffer still yields mutable elements through data().
//
// Why not JPH::Array<T> either? Same two problems, and they were measured
// rather than assumed. JPH::Array is NOT std::vector here -- Jolt only
// aliases it to std::vector under JPH_USE_STD_VECTOR, which this build does
// not define, so it is Jolt's hand-rolled container ("No exception safety"
// per its own header).
//
//   1. It has no failure channel at all. Array::reallocate() assigns
//      mElements = ptr with no null check, so a failed resize() RETURNS
//      SUCCESSFULLY with size() == the requested count and data() == nullptr.
//      Measured with the allocation hooks installed: resize(1 << 48) reports
//      size() 281474976710656 backed by a null pointer. Every later element
//      access is a null dereference and nothing upstream can detect it.
//      Buffer::allocate() returns false and leaves the buffer empty.
//   2. It cannot express over-alignment. STLAllocator only aligns to
//      alignof(T), and needs_aligned_allocate is false unless
//      alignof(T) > JPH_DEFAULT_ALLOCATE_ALIGNMENT (16). PosStride has
//      alignof 8, so JPH::Array<PosStride> comes back 16-byte aligned --
//      measured -- while the seven Buffer<T, kAvxAlignment> shadow buffers
//      (positions, prev_positions, rotations, prev_rotations,
//      linear_velocities, angular_velocities, soft-body vertices) need 32.
//
// JPH::Array stays correct at the Jolt API boundary, where Jolt hands us one
// or expects one back (SoftBodySharedSettings, GetVertices, mGearRatios).
// What it cannot be is this project's owning storage.

#include <cstddef>
#include <memory>
#include <new>
#include <span>
#include <utility>


namespace CPH {

template <class T, std::size_t Alignment = alignof(T)>
class Buffer {
    static constexpr std::size_t kAlignment =
        Alignment > alignof(T) ? Alignment : alignof(T);

    T *data_          = nullptr;
    std::size_t size_ = 0;

  public:
    Buffer() = default;

    Buffer(const Buffer &)            = delete;
    Buffer &operator=(const Buffer &) = delete;

    Buffer(Buffer &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {}

    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            reset();
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    ~Buffer() { reset(); }

    void reset() noexcept {
        std::destroy_n(data_, size_);
        ::operator delete[](data_, std::align_val_t(kAlignment));
        data_ = nullptr;
        size_ = 0;
    }

    // Allocates `count` value-initialized elements, replacing current
    // contents. Returns false (leaving the buffer empty) on OOM/overflow.
    [[nodiscard]] bool allocate(std::size_t count) noexcept {
        return allocate_impl(count, true);
    }

    // Like allocate(), but elements are only default-constructed: scalars
    // and trivial members are left indeterminate (the caller overwrites
    // every element it reads). This is the malloc-parity fast path for
    // large scratch storage such as resize targets; class types still run
    // their default constructors and member initializers.
    [[nodiscard]] bool allocate_for_overwrite(std::size_t count) noexcept {
        return allocate_impl(count, false);
    }

    // Grows/shrinks to `count`, preserving the common prefix. On failure
    // the buffer is unchanged.
    [[nodiscard]] bool reallocate(std::size_t count) noexcept {
        return reallocate_impl(count, true);
    }

    // reallocate() with allocate_for_overwrite() semantics for the
    // non-preserved tail.
    [[nodiscard]] bool reallocate_for_overwrite(std::size_t count) noexcept {
        return reallocate_impl(count, false);
    }

    [[nodiscard]] T *data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

    [[nodiscard]] T &operator[](std::size_t index) const noexcept { return data_[index]; }

    [[nodiscard]] T *begin() const noexcept { return data_; }
    [[nodiscard]] T *end() const noexcept { return data_ + size_; }

    [[nodiscard]] std::span<T> span() const noexcept { return {data_, size_}; }
    [[nodiscard]] std::span<const T> const_span() const noexcept { return {data_, size_}; }

  private:
    [[nodiscard]] bool allocate_impl(std::size_t count, bool value_initialize) noexcept {
        reset();
        if (count == 0) {
            return true;
        }
        if (count > static_cast<std::size_t>(-1) / sizeof(T)) {
            return false;
        }
        auto *raw = static_cast<T *>(::operator new[](
            count * sizeof(T), std::align_val_t(kAlignment), std::nothrow));
        if (raw == nullptr) {
            return false;
        }
        if (value_initialize) {
            std::uninitialized_value_construct_n(raw, count);
        } else {
            std::uninitialized_default_construct_n(raw, count);
        }
        data_ = raw;
        size_ = count;
        return true;
    }

    [[nodiscard]] bool reallocate_impl(std::size_t count, bool value_initialize) noexcept {
        if (count == size_) {
            return true;
        }
        Buffer next;
        if (!next.allocate_impl(count, value_initialize)) {
            return false;
        }
        const std::size_t common = count < size_ ? count : size_;
        for (std::size_t i = 0; i < common; ++i) {
            next.data_[i] = std::move(data_[i]);
        }
        *this = std::move(next);
        return true;
    }
};

// Owning nothrow array for short-lived temporaries whose size is not part
// of a struct contract. `buf.get() == nullptr` signals allocation failure.
template <class T>
[[nodiscard]] inline std::unique_ptr<T[]> make_array(std::size_t count) noexcept {
    return std::unique_ptr<T[]>(new (std::nothrow) T[count]());
}

} // namespace CPH
