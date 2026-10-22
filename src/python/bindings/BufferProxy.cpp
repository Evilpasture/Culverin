#include "bindings/BufferProxy.hpp"

#include "core/Types.hpp"
#include "ecs/Registry.hpp"
#include "physics/PhysicsSync.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstdint>

namespace CPH::Python {

namespace {

// Everything that is fixed per channel: how the consumer should read the
// bytes, how many of them belong to one body, and whether it may write.
struct ChannelSpec {
    const char *format;
    std::size_t itemSize;
    std::size_t stride; ///< Items per body: 4 for a packed vec4, 1 otherwise.
};

// The per-body strides the old MakeProxy() calls passed by hand. Positions,
// rotations and velocities are vec4-shaped regardless of precision.
constexpr std::size_t kVec4Stride = 4;
constexpr std::size_t kScalarStride = 1;

[[nodiscard]] constexpr ChannelSpec SpecFor(WorldChannel channel) noexcept {
    switch (channel) {
    case WorldChannel::Positions:
        return {CPH::kRealFormatString, sizeof(JPH::Real), kVec4Stride};
    case WorldChannel::Rotations:
        return {"f", sizeof(float), kVec4Stride};
    case WorldChannel::LinearVelocities:
        return {"f", sizeof(float), kVec4Stride};
    case WorldChannel::AngularVelocities:
        return {"f", sizeof(float), kVec4Stride};
    case WorldChannel::UserData:
        return {"Q", sizeof(std::uint64_t), kScalarStride};
    case WorldChannel::SoftBodyVertices:
        return {CPH::kRealFormatString, sizeof(JPH::Real), kVec4Stride};
    }
    return {"B", 1, 1};
}

// The SoA array behind a channel. `SoftBodyVertices` is not one: its address
// is pinned per proxy, so it is answered separately.
[[nodiscard]] void *SoaDataFor(CPH::PhysicsWorld &world, WorldChannel channel) noexcept {
    switch (channel) {
    case WorldChannel::Positions:
        return world.bodies.positions.data();
    case WorldChannel::Rotations:
        return world.bodies.rotations.data();
    case WorldChannel::LinearVelocities:
        return world.bodies.linearVelocities.data();
    case WorldChannel::AngularVelocities:
        return world.bodies.angularVelocities.data();
    case WorldChannel::UserData:
        return world.bodies.userData.data();
    case WorldChannel::SoftBodyVertices:
        break;
    }
    return nullptr;
}

} // namespace

// --- WorldBuffer ---------------------------------------------------------

WorldBuffer::WorldBuffer(CPH::PhysicsWorld &world, WorldChannel channel) noexcept
    : world_(&world), channel_(channel) {}

WorldBuffer::WorldBuffer(CPH::PhysicsWorld &world, WorldChannel channel, void *pinned,
                         std::size_t pinnedBytes) noexcept
    : world_(&world), channel_(channel), pinned_(pinned),
      pinnedElements_(SpecFor(channel).itemSize == 0 ? 0 : pinnedBytes / SpecFor(channel).itemSize) {}

std::size_t WorldBuffer::ElementCount() const noexcept {
    if (channel_ == WorldChannel::SoftBodyVertices) {
        return pinnedElements_;
    }
    const ChannelSpec spec = SpecFor(channel_);
    return world_->count.load(std::memory_order::acquire) * spec.stride;
}

std::expected<BufferDescriptor, const char *> WorldBuffer::Describe() {
    const ChannelSpec spec = SpecFor(channel_);

    // Read under the shadow lock, with any in-flight step waited out: body
    // creation can reallocate the arrays, and a step writes them from another
    // thread. `BlockUntilNotStepping` parks on the step condition and drops the
    // GIL while it waits; its own unlock/lock of the shadow mutex is balanced,
    // so the unique_lock below still owns it when this scope ends -- including
    // on the failure return.
    std::unique_lock shadow(world_->shadowLock);
    BlockUntilNotStepping(*world_);

    void        *data  = nullptr;
    std::size_t length = 0;
    if (channel_ == WorldChannel::SoftBodyVertices) {
        // Pinned when the proxy was made: one body's block, which cannot move
        // while the export is outstanding.
        data   = pinned_;
        length = pinnedElements_;
    } else {
        data   = SoaDataFor(*world_, channel_);
        length = world_->count.load(std::memory_order::acquire) * spec.stride;
    }

    if (data == nullptr) {
        return std::unexpected("Physics buffer not allocated");
    }

    return BufferDescriptor{.data     = data,
                            .bytes    = length * spec.itemSize,
                            .itemSize = spec.itemSize,
                            .format   = spec.format,
                            .readonly = false};
}

void WorldBuffer::AcquireExport() noexcept {
    world_->viewExportCount.fetch_add(1, std::memory_order::relaxed);
}

void WorldBuffer::ReleaseExport() noexcept {
    world_->viewExportCount.fetch_sub(1, std::memory_order::relaxed);
}

void WorldBuffer::OnBufferReleased() noexcept {
#if defined(STRICT_THREAD_SAFETY)
    // TODO(smell): nothing increments `activeQueries` for a proxy read, so this
    // decrement is unbalanced -- it exists because the getbuffer path used to
    // count a query, and the branch it lives in has never been compiled. Kept
    // verbatim (it now at least compiles) rather than deleted, so whoever
    // re-enables STRICT_THREAD_SAFETY sees the accounting they must fix.
    if (world_->activeQueries.fetch_sub(1, std::memory_order::release) == 1) {
        const std::scoped_lock notify(world_->stepSync.mutex);
        world_->stepSync.cond.notify_all();
    }
#endif
}

// --- RegistryBuffer ------------------------------------------------------

RegistryBuffer::RegistryBuffer(CPH::Registry &registry, RegistryChannel channel, void *data,
                               std::size_t count, std::size_t elementSize) noexcept
    : registry_(&registry), channel_(channel), data_(data),
      // Component data is exported as opaque bytes ("B", 1 byte per item), so
      // its item count is its byte count; entity handles are one item each.
      // `elementSize` therefore sizes the *data* channel and is not itself an
      // item size.
      elements_(channel == RegistryChannel::EntityHandles ? count : count * elementSize),
      itemSize_(channel == RegistryChannel::EntityHandles ? sizeof(std::uint64_t) : 1) {}

std::expected<BufferDescriptor, const char *> RegistryBuffer::Describe() const noexcept {
    return BufferDescriptor{
        .data     = data_,
        .bytes    = elements_ * itemSize_,
        .itemSize = itemSize_,
        .format   = channel_ == RegistryChannel::EntityHandles ? "Q" : "B",
        // Entity handles are the registry's own storage; component bytes are
        // the consumer's to write.
        .readonly = channel_ == RegistryChannel::EntityHandles,
    };
}

void RegistryBuffer::AcquireExport() noexcept { registry_->OnViewExported(); }

void RegistryBuffer::ReleaseExport() noexcept { registry_->OnViewReleased(); }

// --- BufferProxy ---------------------------------------------------------

BufferProxy::BufferProxy(Payload payload) noexcept : payload_(std::move(payload)) {
    // The construction-time length, without taking the shadow lock: the old
    // path read the body count as an atomic and nothing else. len(proxy) works
    // from here on, before any buffer has been requested.
    length_ = std::visit([](const auto &buffer) { return buffer.ElementCount(); }, payload_);
}

std::expected<BufferDescriptor, const char *> BufferProxy::Describe() {
    auto described = std::visit([](auto &buffer) { return buffer.Describe(); }, payload_);
    if (described) {
        const std::size_t itemSize = described->itemSize == 0 ? 1 : described->itemSize;
        length_                    = described->bytes / itemSize;
    }
    return described;
}

void BufferProxy::AcquireExport() noexcept {
    std::visit([](auto &buffer) { buffer.AcquireExport(); }, payload_);
}

void BufferProxy::ReleaseExport() noexcept {
    std::visit([](auto &buffer) { buffer.ReleaseExport(); }, payload_);
}

void BufferProxy::OnBufferReleased() noexcept {
    std::visit([](auto &buffer) { buffer.OnBufferReleased(); }, payload_);
}

} // namespace CPH::Python
