#pragma once

// Double-buffered transform output (the ownership domain that fixes data
// tearing on zero-copy memoryviews).
//
// Before this refactor, memoryview(world.positions) pointed straight at
// bodies.positions, which SyncShadowBuffers overwrote during Phase 2 of a step
// with shadowLock *released*. A Python/NumPy/render thread reading that view
// concurrently could observe X from frame N and Y,Z from frame N-1 -- a
// sheared transform -- and no lock in the world could stop it because the
// buffer protocol bypasses locks entirely.
//
// TransformBuffer owns FrameCount complete frames of the four per-step SoA
// channels. The step writes exclusively into Back() and calls Publish() (one
// atomic release-store) at FinalizeStep; Python always exports Front(). A
// reader therefore never observes memory that is being written.
//
// The canonical bodies.* arrays remain the single internal source of truth
// (interpolation via prevPositions, MirrorToShadow on set ops, Jolt sync all
// still target them). Publish() snapshots the current frame from those arrays
// into Back() before rotating, which keeps Jolt sync numerics byte-identical
// while decoupling every external reader from the writer. That snapshot is a
// handful of contiguous memcpys -- negligible next to physicsSystem.Update.
//
// Template parameters are the stride types (PosStride/AuxStride in the engine,
// trivial PODs in the unit test) so this header stays free of Jolt includes
// and can be compiled standalone.

#include "core/Buffer.hpp"
#include "core/DoubleBuffer.hpp"

#include <cstddef>
#include <cstring>

namespace CPH {

template <class PosT, class AuxT, std::size_t FrameCount = 2> class TransformBuffer {
  public:
    struct Frame {
        Buffer<PosT> positions;
        Buffer<AuxT> rotations;
        Buffer<AuxT> linearVelocities;
        Buffer<AuxT> angularVelocities;

        [[nodiscard]] bool allocate(std::size_t capacity) noexcept {
            return positions.allocate(capacity) && rotations.allocate(capacity) &&
                   linearVelocities.allocate(capacity) && angularVelocities.allocate(capacity);
        }

        void clear() noexcept {
            positions.reset();
            rotations.reset();
            linearVelocities.reset();
            angularVelocities.reset();
        }
    };

    // Allocate every frame at the fixed capacity, once. No resize path: the
    // world's max_bodies is the ceiling and is known at init.
    [[nodiscard]] bool Init(std::size_t capacity) noexcept {
        for (std::size_t i = 0; i < FrameCount; ++i) {
            if (!frames_.Slot(i).allocate(capacity)) {
                return false;
            }
        }
        capacity_ = capacity;
        return true;
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

    // Stable published frame -- what memoryview/NumPy/render must read.
    [[nodiscard]] const Frame &Front() const noexcept { return frames_.Front(); }
    [[nodiscard]] Frame &Front() noexcept { return frames_.Front(); }
    // Writable frame -- only the step thread may touch this between a copy and
    // the matching Publish().
    [[nodiscard]] Frame &Back() noexcept { return frames_.Back(); }

    // Snapshot `count` records from the canonical arrays into Back() and make
    // that frame the new Front(). Runs on the step thread with the GIL
    // released; it never writes a buffer a reader can currently see.
    void Publish(const PosT *pos, const AuxT *rot, const AuxT *lvel, const AuxT *avel,
                 std::size_t count) noexcept {
        Frame &back         = frames_.Back();
        const std::size_t n = count > capacity_ ? capacity_ : count;
        if (n > 0) {
            std::memcpy(back.positions.data(), pos, n * sizeof(PosT));
            std::memcpy(back.rotations.data(), rot, n * sizeof(AuxT));
            std::memcpy(back.linearVelocities.data(), lvel, n * sizeof(AuxT));
            std::memcpy(back.angularVelocities.data(), avel, n * sizeof(AuxT));
        }
        frames_.Publish();
    }

    // Rotate without copying (used when the back frame was filled in place).
    void Flip() noexcept { frames_.Publish(); }

    // Mirror a single dense record from the canonical shadow buffers into
    // EVERY frame, so an out-of-step mutation (creation, teleport, character
    // move) is immediately visible through a zero-copy view regardless of
    // which frame is currently published or will be published after the next
    // flip. This is the O(1) counterpart of Publish(): the step republishes
    // all slots from canonical, while mutators publish just the slot they
    // touch. All mutators hold shadowLock and run with the world not stepping,
    // so no step is concurrently writing the back frame.
    void WriteSlot(std::size_t dense, const PosT &pos, const AuxT &rot, const AuxT &lvel,
                   const AuxT &avel) noexcept {
        for (std::size_t f = 0; f < FrameCount; ++f) {
            Frame &frame                   = frames_.Slot(f);
            frame.positions[dense]         = pos;
            frame.rotations[dense]         = rot;
            frame.linearVelocities[dense]  = lvel;
            frame.angularVelocities[dense] = avel;
        }
    }

    // Bulk variant of WriteSlot for out-of-step operations that rewrite many
    // slots at once (snapshot restore, batch creation, baked-scene load).
    // Copies `count` records from the canonical arrays into EVERY frame so a
    // reader holding any published frame sees the restored state. Unlike
    // Publish() this does not flip: these operations happen with the world
    // quiesced (no step racing), so an atomic swap is unnecessary.
    void WriteAll(const PosT *pos, const AuxT *rot, const AuxT *lvel, const AuxT *avel,
                  std::size_t count) noexcept {
        const std::size_t n = count > capacity_ ? capacity_ : count;
        for (std::size_t f = 0; f < FrameCount; ++f) {
            Frame &frame = frames_.Slot(f);
            if (n > 0) {
                std::memcpy(frame.positions.data(), pos, n * sizeof(PosT));
                std::memcpy(frame.rotations.data(), rot, n * sizeof(AuxT));
                std::memcpy(frame.linearVelocities.data(), lvel, n * sizeof(AuxT));
                std::memcpy(frame.angularVelocities.data(), avel, n * sizeof(AuxT));
            }
        }
    }

  private:
    DoubleBuffer<Frame, FrameCount> frames_;
    std::size_t capacity_ = 0;
};

} // namespace CPH
