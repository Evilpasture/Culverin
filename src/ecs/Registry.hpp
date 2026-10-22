#pragma once

// The ECS registry: entity generation bookkeeping plus a bank of sparse-set
// component stores. Pure C++23 -- no <Python.h>, no Jolt, no Python error
// state. Every fallible operation reports through std::expected<T, Error>
// with an EcsError code, and the binding layer is what turns those into
// exceptions.
//
// Locking: the registry owns its mutex and takes it inside each operation, so
// a caller cannot forget it. The one exception is the physics transform
// bridge, which must hold the registry lock and the physics shadow lock at
// the same time; it is a friend and documented at its definition.

#include "core/Buffer.hpp"
#include "ecs/EcsTypes.hpp"
#include "ecs/SparseSet.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <mutex>
#include <optional>
#include <span>

namespace CPH {

struct PhysicsWorld;
class Registry;
struct EcsSyncConfig;

// Copies body transforms from a PhysicsWorld into ECS components. Declared
// here so the registry can befriend it; defined in src/physics/EcsSync.cpp,
// which is the only translation unit that sees both halves.
[[nodiscard]] std::expected<void, Error> SyncEcsTransforms(PhysicsWorld      &world,
                                                            Registry         &registry,
                                                            const EcsSyncConfig &config);

class Registry {
  public:
    using ComponentId = std::uint32_t;

    Registry() = default;
    ~Registry() = default;

    Registry(const Registry &)            = delete;
    Registry &operator=(const Registry &) = delete;
    Registry(Registry &&)                 = delete;
    Registry &operator=(Registry &&)      = delete;

    // --- Entities ---------------------------------------------------------

    // Hands out a fresh handle, growing the entity arrays when the free list
    // runs dry. The order handles are issued in is observable -- it decides
    // every entity id a program ever sees -- so the free-list layout is kept
    // exactly as it was before the migration.
    [[nodiscard]] Entity Create();

    // Invalidates `entity` and detaches it from every component. Silent when
    // the handle is already dead: destroying twice is not an error.
    void Destroy(Entity entity);

    [[nodiscard]] bool IsAlive(Entity entity) const;

    // Drops every component and invalidates every entity by bumping all
    // generations. Refused while memoryviews are exported.
    [[nodiscard]] std::expected<void, Error> Clear();

    [[nodiscard]] std::uint32_t active_entity_count() const noexcept {
        return activeEntities_;
    }

    // --- Components -------------------------------------------------------

    // Registers a component type of `element_size` bytes and returns its id.
    [[nodiscard]] std::expected<ComponentId, Error> RegisterComponent(std::uint32_t element_size);

    // Inserts or overwrites `entity`'s component. An absent payload zero-fills
    // it; a present payload must match the registered element size exactly.
    [[nodiscard]] std::expected<void, Error>
    Add(Entity entity, ComponentId component, std::optional<std::span<const std::byte>> payload);

    [[nodiscard]] std::expected<void, Error> Remove(Entity entity, ComponentId component);

    [[nodiscard]] std::expected<bool, Error> Has(Entity entity, ComponentId component) const;

    [[nodiscard]] std::expected<std::optional<std::span<const std::byte>>, Error>
    Get(Entity entity, ComponentId component) const;

    // Live component count, or 0 for an unknown id (not an error: the Python
    // API has always answered 0 here rather than raising).
    [[nodiscard]] std::uint32_t ComponentCount(ComponentId component) const noexcept;

    [[nodiscard]] std::uint32_t ComponentElementSize(ComponentId component) const noexcept;

    [[nodiscard]] std::uint32_t component_count() const noexcept { return componentCount_; }

    // --- View bookkeeping -------------------------------------------------

    // Storage a buffer proxy needs, captured under the registry lock.
    struct ComponentView {
        Entity        *entities;
        std::byte     *data;
        std::uint32_t  count;
        std::uint32_t  elementSize;
    };

    [[nodiscard]] std::expected<ComponentView, Error> View(ComponentId component);

    // Exported memoryviews pin the dense arrays: growth would reallocate the
    // memory a proxy hands out. The binding increments on export and
    // decrements on release.
    void OnViewExported() noexcept {
        viewExports_.fetch_add(1, std::memory_order_relaxed);
    }
    void OnViewReleased() noexcept {
        viewExports_.fetch_sub(1, std::memory_order_relaxed);
    }

  private:
    friend std::expected<void, Error> SyncEcsTransforms(PhysicsWorld &, Registry &,
                                                         const EcsSyncConfig &);

    [[nodiscard]] bool GrowEntities();
    [[nodiscard]] const SparseSet *SetAt(ComponentId component) const noexcept;
    [[nodiscard]] SparseSet *SetAt(ComponentId component) noexcept;

    static constexpr std::uint32_t kInitialEntityCapacity    = 1024;
    static constexpr std::uint32_t kInitialComponentCapacity = 16;

    mutable std::mutex    mutex_;
    Buffer<std::uint32_t> generations_;
    Buffer<std::uint32_t> freeIndices_;
    std::uint32_t         freeCount_      = 0;
    std::uint32_t         activeEntities_ = 0;

    Buffer<SparseSet> components_;
    std::uint32_t     componentCount_ = 0;

    std::atomic<std::uint32_t> viewExports_{0};
};

} // namespace CPH
