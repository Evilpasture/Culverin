#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <variant>

namespace CPH {
// The tag has to match the definition: `PhysicsWorld` is a struct, `Registry`
// is a class. Mismatching is only a warning, but under the MSVC ABI it can
// pick a different mangled name for members that mention the type.
struct PhysicsWorld;
class Registry;
} // namespace CPH

namespace CPH::Python {

/// One exported buffer, as the buffer protocol needs to see it.
struct BufferDescriptor {
    void       *data     = nullptr;
    std::size_t bytes    = 0;
    std::size_t itemSize = 1;
    const char *format   = "B";
    bool        readonly = false;
};

/// Which per-body array of the physics world a proxy exports.
enum class WorldChannel : std::uint8_t {
    Positions,
    Rotations,
    LinearVelocities,
    AngularVelocities,
    UserData,
    SoftBodyVertices, ///< Not an SoA channel: one soft body's private block.
};

/// A physics-world buffer.
///
/// The SoA channels are re-read on every Describe(), because creating a body
/// can reallocate the arrays; the soft-body block is pinned when the proxy is
/// made, since it belongs to one body and cannot move while the view is held.
///
/// The locking policy lives here rather than at the boundary on purpose. Every
/// channel is described under the shadow lock with any in-flight step waited
/// out -- that is what keeps a half-written frame out of NumPy -- and the wait
/// drops the GIL while it parks, so the stepping thread can finish and wake it.
/// None of that is Python's business; it is this buffer's.
class WorldBuffer {
  public:
    WorldBuffer(CPH::PhysicsWorld &world, WorldChannel channel) noexcept;

    /// `pinnedBytes` is the size of the block, not an element count.
    WorldBuffer(CPH::PhysicsWorld &world, WorldChannel channel, void *pinned,
                std::size_t pinnedBytes) noexcept;

    [[nodiscard]] std::expected<BufferDescriptor, const char *> Describe();

    /// Elements as of "now", read without taking the lock (an atomic snapshot),
    /// which is what the old construction path did.
    [[nodiscard]] std::size_t ElementCount() const noexcept;

    void AcquireExport() noexcept;
    void ReleaseExport() noexcept;

    /// Hook for the buffer-protocol release slot. Physics exports account for
    /// an in-flight query there; the guarded branch below has never been
    /// compiled with the increment it pairs with, so it stays exactly as it
    /// was -- reviewed, unexercised, and flagged.
    void OnBufferReleased() noexcept;

  private:
    CPH::PhysicsWorld *world_;
    WorldChannel       channel_;
    void              *pinned_         = nullptr;
    std::size_t        pinnedElements_ = 0;
};

/// Which array of the ECS registry a proxy exports.
enum class RegistryChannel : std::uint8_t {
    ComponentData, ///< Dense component bytes; the consumer casts by component.
    EntityHandles, ///< Dense entity handles; read-only.
};

/// An ECS registry buffer.
///
/// Nothing here can move while the proxy is alive: the registry refuses to grow
/// a component set that has an outstanding export, so the address and length
/// are pinned at construction and Describe() only re-states them.
class RegistryBuffer {
  public:
    RegistryBuffer(CPH::Registry &registry, RegistryChannel channel, void *data, std::size_t count,
                   std::size_t elementSize) noexcept;

    /// Never fails, but shares the return type of the physics arm so the
    /// variant visit has one result type to deduce.
    [[nodiscard]] std::expected<BufferDescriptor, const char *> Describe() const noexcept;

    [[nodiscard]] std::size_t ElementCount() const noexcept { return elements_; }

    void AcquireExport() noexcept;
    void ReleaseExport() noexcept;

    /// Nothing to unwind on release: this domain contributes only the export
    /// count that ReleaseExport() already gives back.
    void OnBufferReleased() noexcept {}

  private:
    CPH::Registry *registry_;
    RegistryChannel channel_;
    void           *data_;
    std::size_t     elements_;
    std::size_t     itemSize_;
};

/// What a `culverin.BufferProxy` hands out: one of the two domains above.
///
/// The domain is the *type*, not a tag: the old design carried a
/// `ProxyBufferType` enum, switched on it in five places, and used "less than
/// EcsData" as a way of asking "is this the physics world?" -- so adding a
/// channel meant finding every such comparison. `std::variant` makes each
/// operation one visit, and the compiler rejects a payload nobody handles.
///
/// Deliberately free of CPython: no PyObject, no Py_buffer, no refcounts. The
/// adapter in Getters.cpp owns all of that, and this class is the part that
/// would survive unchanged if the bindings were regenerated for another
/// language.
class BufferProxy final {
  public:
    using Payload = std::variant<WorldBuffer, RegistryBuffer>;

    explicit BufferProxy(Payload payload) noexcept;

    /// Acquires the buffer (locking as the payload requires) and caches its
    /// length. On failure the message is the caller's to raise, unchanged.
    [[nodiscard]] std::expected<BufferDescriptor, const char *> Describe();

    /// Element count as of the last Describe(). `len(proxy)` before any
    /// acquisition reports the count the proxy was built with, as before.
    [[nodiscard]] std::size_t Length() const noexcept { return length_; }

    void AcquireExport() noexcept;
    void ReleaseExport() noexcept;
    void OnBufferReleased() noexcept;

  private:
    Payload     payload_;
    std::size_t length_ = 0;
};

} // namespace CPH::Python
