#pragma once

// Pure-asynchronous command mailbox (the ownership domain that replaces the
// immediate/deferred split and makes Jolt mutations step-thread-only).
//
// Every Python-visible mutation -- creates, destroys, transforms, impulses,
// forces, torques -- becomes a command pushed here. Producers only ever touch
// this mailbox; they never reach into JPH::PhysicsSystem. The step thread
// Detach()es the whole queue with a pointer swap and FlushCommands() it, in
// order, before physicsSystem.Update(). Because there is exactly one consumer
// and it runs on the step thread, the jph_trampoline_lock that used to guard
// "immediate" impulses applied from arbitrary threads is no longer needed.
//
// Synchronization: a single lightweight mutex protects the active buffer and
// its count. A Push is one lock + one store (the analysis's ~5ns mailbox);
// Detach is one lock + one swap. Command payloads own their Jolt settings, so
// Clear() must assign a default Command to destroy them (no leaks).
//
// Templated on the command type (PhysicsCommand in the engine, a trivial
// struct in the unit test) so this header has no Jolt dependency and can be
// compiled standalone.

#include "core/Buffer.hpp"

#include <cstddef>
#include <limits>
#include <mutex>
#include <span>
#include <utility>

namespace CPH {

template <class Command> class CommandMailbox {
  public:
    static constexpr std::size_t kDefaultCapacity = 64;

    // Allocates the active and spare buffers at `capacity`. Called once at
    // world init; growth, if ever needed, is handled by Reserve/Push.
    [[nodiscard]] bool Init(std::size_t capacity = kDefaultCapacity) noexcept {
        std::scoped_lock lock(mutex_);
        if (!active_.allocate(capacity) || !spare_.allocate(capacity)) {
            return false;
        }
        count_ = 0;
        return true;
    }

    // Producer side. Constructs the command in place at the tail; grows the
    // active buffer on demand (all-or-nothing, see Buffer::reallocate).
    void Push(Command command) noexcept {
        std::scoped_lock lock(mutex_);
        if (count_ >= active_.size() && !GrowLocked(count_ + 1)) {
            return; // OOM: the command is dropped; callers pre-Reserve for
                    // bulk paths so this cannot lose a batch.
        }
        active_[count_++] = std::move(command);
    }

    // Ensure room for `extra` more commands (bulk create/destroy paths).
    [[nodiscard]] bool Reserve(std::size_t extra) noexcept {
        std::scoped_lock lock(mutex_);
        if (extra > std::numeric_limits<std::size_t>::max() - count_) {
            return false;
        }
        return GrowLocked(count_ + extra);
    }

    // Consumer side (step thread). Swaps active/spare under the lock and
    // returns the captured commands as a span into the (now) spare buffer.
    // Flush them before the next Detach. A producer pushing after this call
    // writes into the fresh active buffer and cannot disturb the returned span.
    [[nodiscard]] std::span<Command> Detach() noexcept {
        std::scoped_lock lock(mutex_);
        const std::size_t captured = count_;
        std::swap(active_, spare_);
        count_ = 0;
        return {spare_.data(), captured};
    }

    // Destroy pending payloads (world teardown).
    void Clear() noexcept {
        std::scoped_lock lock(mutex_);
        for (std::size_t i = 0; i < count_; ++i) {
            active_[i] = Command{};
        }
        count_ = 0;
    }

    [[nodiscard]] bool Empty() const noexcept {
        std::scoped_lock lock(mutex_);
        return count_ == 0;
    }
    [[nodiscard]] std::size_t Size() const noexcept {
        std::scoped_lock lock(mutex_);
        return count_;
    }

  private:
    // mutex_ is already held by the caller.
    [[nodiscard]] bool GrowLocked(std::size_t required) noexcept {
        if (required <= active_.size()) {
            return true;
        }
        std::size_t capacity = active_.size() == 0 ? kDefaultCapacity : active_.size();
        while (capacity < required) {
            if (capacity > std::numeric_limits<std::size_t>::max() / 2) {
                return false;
            }
            capacity *= 2;
        }
        return active_.reallocate_for_overwrite(capacity);
    }

    mutable std::mutex mutex_;
    Buffer<Command> active_;
    Buffer<Command> spare_;
    std::size_t count_ = 0;
};

} // namespace CPH
