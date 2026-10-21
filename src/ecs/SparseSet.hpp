#pragma once

// Sparse-set component storage: a sparse array indexed by entity slot maps to
// a dense, contiguous payload array, so iteration over the live components is
// linear and cache-friendly while lookup stays O(1).
//
// The payload is opaque bytes (`element_size` per component) rather than a C++
// type, because components are registered at runtime by their size rather than
// generated as C++ structs. Everything owning lives in CPH::Buffer, so there is
// no manual allocate/reallocate/free here.

#include "core/Buffer.hpp"
#include "ecs/EcsTypes.hpp"
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <expected>
#include <optional>
#include <span>

namespace CPH {

class SparseSet {
  public:
    static constexpr std::uint32_t kInvalidDenseIndex = 0xFFFFFFFFu;

    SparseSet() = default;
    explicit SparseSet(std::uint32_t element_size) noexcept : element_size_(element_size) {}

    SparseSet(const SparseSet &)                    = delete;
    SparseSet &operator=(const SparseSet &)         = delete;
    SparseSet(SparseSet &&) noexcept                = default;
    SparseSet &operator=(SparseSet &&) noexcept     = default;

    [[nodiscard]] std::uint32_t element_size() const noexcept { return element_size_; }

    // Number of live components.
    [[nodiscard]] std::uint32_t count() const noexcept { return count_; }

    // Pointers handed to the binding's buffer proxies. Stable only while
    // nothing grows the set, which is what Registry's exported-view counter
    // exists to enforce.
    [[nodiscard]] Entity    *dense_entities() const noexcept { return dense_.data(); }
    [[nodiscard]] std::byte *element_data() const noexcept { return data_.data(); }

    [[nodiscard]] std::optional<std::uint32_t> DenseIndexOf(Entity entity) const noexcept {
        const std::uint32_t index = entity.Index();
        if (index >= sparse_.size()) {
            return std::nullopt;
        }
        const std::uint32_t dense = sparse_[index];
        return dense == kInvalidDenseIndex ? std::nullopt : std::optional{dense};
    }

    [[nodiscard]] bool Has(Entity entity) const noexcept {
        return DenseIndexOf(entity).has_value();
    }

    // Payload of the dense slot `dense`, or an empty span if out of range.
    [[nodiscard]] std::span<const std::byte> ElementAt(std::uint32_t dense) const noexcept {
        if (dense >= count_) {
            return {};
        }
        return {data_.data() + static_cast<std::size_t>(dense) * element_size_, element_size_};
    }

    // Inserts `entity`, or overwrites its payload if it is already present.
    // An empty `payload` zero-fills the component, matching the Python
    // convention of `add(entity, comp_id)` with no data argument.
    //
    // `live_view_exports` is the registry-wide count of exported memoryviews.
    // Growing the dense arrays reallocates the very memory a proxy points at,
    // so growth is refused while any view is live -- the refusal only happens
    // when growth is actually required, so adding into spare capacity keeps
    // working with views outstanding.
    [[nodiscard]] std::expected<void, Error>
    Insert(Entity entity, std::span<const std::byte> payload, std::uint32_t live_view_exports) {
        const std::uint32_t index = entity.Index();

        if (!EnsureSparseCapacity(index + 1)) {
            return std::unexpected(EcsError::OutOfMemory);
        }

        std::uint32_t dense = sparse_[index];
        if (dense == kInvalidDenseIndex) {
            if (count_ >= dense_.size() && live_view_exports > 0) {
                return std::unexpected(EcsError::ComponentResizeWhileViewed);
            }
            if (!EnsureDenseCapacity()) {
                return std::unexpected(EcsError::OutOfMemory);
            }
            dense          = count_++;
            dense_[dense]  = entity;
            sparse_[index] = dense;
        }

        std::byte *dst = data_.data() + static_cast<std::size_t>(dense) * element_size_;
        if (payload.empty()) {
            std::memset(dst, 0, element_size_);
        } else {
            std::memcpy(dst, payload.data(), element_size_);
        }
        return {};
    }

    // Swap-and-pop removal. Returns false when the entity is not present.
    bool Erase(Entity entity) noexcept {
        const std::uint32_t index = entity.Index();
        if (index >= sparse_.size()) {
            return false;
        }
        const std::uint32_t dense = sparse_[index];
        if (dense == kInvalidDenseIndex) {
            return false;
        }

        const std::uint32_t last = count_ - 1;
        if (dense != last) {
            const Entity last_entity = dense_[last];
            dense_[dense]                 = last_entity;
            sparse_[last_entity.Index()]  = dense;
            std::memcpy(data_.data() + static_cast<std::size_t>(dense) * element_size_,
                        data_.data() + static_cast<std::size_t>(last) * element_size_,
                        element_size_);
        }
        sparse_[index] = kInvalidDenseIndex;
        count_--;
        return true;
    }

    // Drops every element without releasing storage; entity handles stay
    // valid as handles, the caller decides whether they are still meaningful.
    void EraseAll() noexcept {
        count_ = 0;
        if (data_.data() != nullptr) {
            // kInvalidDenseIndex is all-ones, so byte-filling is exact.
            std::memset(sparse_.data(), 0xFF, sparse_.size() * sizeof(std::uint32_t));
        }
    }

  private:
    [[nodiscard]] bool EnsureSparseCapacity(std::uint32_t required) {
        if (sparse_.size() >= required) {
            return true;
        }
        std::size_t new_cap = sparse_.size() == 0 ? kInitialSparseCapacity : sparse_.size() * 2;
        while (new_cap < required) {
            new_cap *= 2;
        }
        const std::size_t previous = sparse_.size();
        // reallocate() value-initializes the whole new array, then copies the
        // common prefix back, so the tail arrives as 0 and is rewritten here.
        if (!sparse_.reallocate(new_cap)) {
            return false;
        }
        for (std::size_t i = previous; i < new_cap; ++i) {
            sparse_[i] = kInvalidDenseIndex;
        }
        return true;
    }

    [[nodiscard]] bool EnsureDenseCapacity() {
        if (count_ < dense_.size()) {
            return true;
        }
        const std::size_t new_cap = dense_.size() == 0 ? kFirstDenseCapacity : dense_.size() * 2;
        if (!dense_.reallocate(new_cap)) {
            return false;
        }
        return data_.reallocate(new_cap * element_size_);
    }

    static constexpr std::size_t kInitialSparseCapacity = 1024;
    static constexpr std::size_t kFirstDenseCapacity    = 64;

    Buffer<std::uint32_t> sparse_;
    Buffer<Entity>        dense_;
    Buffer<std::byte>     data_;
    std::uint32_t         element_size_ = 0;
    std::uint32_t         count_        = 0;
};

} // namespace CPH
