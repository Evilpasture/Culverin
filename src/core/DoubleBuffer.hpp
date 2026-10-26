#pragma once

// A tiny single-producer / multi-reader double (or N-) buffer.
//
// The refactor's central rule is "readers never wait on writers". A writer
// fills Back(), then calls Publish() to rotate the front index with one
// release-store; readers take Front() with one acquire-load and see a frame
// that is never being written. No mutex, no condition variable, no spin.
//
// This is the primitive behind physics/TransformBuffer.hpp and the per-entity
// state caches in characters/ and vehicles/. It is deliberately a template
// with no project dependencies (only libstdc++ headers), and its observable
// guarantees are covered by tests/test_subsystem_ownership.py through the
// public Python buffer API.
//
// Memory order: Publish() is release, Front() is acquire, so every write the
// producer made to Back() before Publish() is visible to any reader that
// subsequently observes the new front index. The index is a single atomic,
// so a reader can never see a half-rotated frame.
//
// FrameCount >= 2. Two frames are enough for readers that re-acquire a view
// every step (the common memoryview/NumPy pattern); a held view is guaranteed
// stable for FrameCount-1 further publishes. Increase it (e.g. 3, triple
// buffering) to widen the window for consumers that keep a raw pointer across
// many steps.

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace CPH {

template <class T, std::size_t FrameCount = 2> class DoubleBuffer {
    static_assert(FrameCount >= 2, "double buffering needs at least two frames");

  public:
    using value_type = T;

    // Mutable back frame: only the single writer may touch this.
    [[nodiscard]] T &Back() noexcept {
        return frames_[NextIndex(front_.load(std::memory_order::relaxed))];
    }

    // Stable, fully-published front frame. Safe to read from any thread
    // without a lock; the reference stays valid for this object's lifetime
    // (frames are value members, never reallocated).
    [[nodiscard]] const T &Front() const noexcept {
        return frames_[front_.load(std::memory_order::acquire)];
    }
    [[nodiscard]] T &Front() noexcept { return frames_[front_.load(std::memory_order::acquire)]; }

    [[nodiscard]] std::size_t FrontIndex() const noexcept {
        return front_.load(std::memory_order::acquire);
    }

    // Direct frame access for one-time, single-threaded initialization (e.g.
    // allocating every frame's arrays before any reader exists). Never use
    // Slot() to publish data -- use Back()/Publish() so the release/acquire
    // handshake is preserved.
    [[nodiscard]] T &Slot(std::size_t i) noexcept { return frames_[i]; }
    [[nodiscard]] const T &Slot(std::size_t i) const noexcept { return frames_[i]; }
    void Publish() noexcept {
        const std::size_t current = front_.load(std::memory_order::relaxed);
        front_.store(NextIndex(current), std::memory_order::release);
    }

    static constexpr std::size_t kFrameCount = FrameCount;

  private:
    [[nodiscard]] static constexpr std::size_t NextIndex(std::size_t index) noexcept {
        return (index + 1) % FrameCount;
    }

    T frames_[FrameCount]{};
    std::atomic<std::size_t> front_{0};
};

} // namespace CPH
