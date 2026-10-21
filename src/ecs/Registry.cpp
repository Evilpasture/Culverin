#include "ecs/Registry.hpp"

namespace CPH {

// Entity indices come from a free list filled in descending order and popped
// from the back, so a fresh registry issues 0, 1, 2, ... in order. Generation
// 1 is the initial one, matching GenerationalHandle's convention that the raw
// value 0 means "invalid" and anything else is valid.

namespace {

std::uint32_t LiveViewExports(const std::atomic<std::uint32_t> &counter) noexcept {
    return counter.load(std::memory_order::acquire);
}

} // namespace

const SparseSet *Registry::SetAt(ComponentId component) const noexcept {
    return component < component_count_ ? &components_[component] : nullptr;
}

SparseSet *Registry::SetAt(ComponentId component) noexcept {
    return component < component_count_ ? &components_[component] : nullptr;
}

// --- Entities -------------------------------------------------------------

bool Registry::GrowEntities() {
    const std::uint32_t old_cap = static_cast<std::uint32_t>(generations_.size());
    const std::uint32_t new_cap = old_cap == 0 ? kInitialEntityCapacity : old_cap * 2;

    Buffer<std::uint32_t> next_generations;
    Buffer<std::uint32_t> next_free;
    if (!next_generations.reallocate(new_cap) || !next_free.reallocate(new_cap)) {
        return false;
    }
    for (std::uint32_t i = 0; i < old_cap; ++i) {
        next_generations[i] = generations_[i];
        next_free[i]        = free_indices_[i];
    }
    // New slots start at generation 1 and are pushed onto the free list in
    // descending order, so popping from the back yields ascending indices.
    for (std::uint32_t i = old_cap; i < new_cap; ++i) {
        next_generations[i]           = 1;
        next_free[free_count_++]      = (new_cap - 1) - (i - old_cap);
    }

    generations_  = std::move(next_generations);
    free_indices_ = std::move(next_free);
    return true;
}

Entity Registry::Create() {
    const std::scoped_lock lock(mutex_);

    if (free_count_ == 0 && !GrowEntities()) {
        return Entity{}; // caller reports OOM; an invalid handle is the signal
    }

    const std::uint32_t index = free_indices_[--free_count_];
    const std::uint32_t generation = generations_[index];
    active_entities_++;
    return Entity::FromParts(index, generation);
}

void Registry::Destroy(Entity entity) {
    const std::scoped_lock lock(mutex_);

    const std::uint32_t index = entity.Index();
    if (index >= generations_.size() || generations_[index] != entity.Generation()) {
        return; // already dead
    }

    for (std::uint32_t i = 0; i < component_count_; ++i) {
        components_[i].Erase(entity);
    }

    generations_[index]++;
    free_indices_[free_count_++] = index;
    active_entities_--;
}

bool Registry::IsAlive(Entity entity) const {
    const std::scoped_lock lock(mutex_);
    const std::uint32_t index = entity.Index();
    return index < generations_.size() && generations_[index] == entity.Generation();
}

std::expected<void, Error> Registry::Clear() {
    const std::scoped_lock lock(mutex_);

    if (LiveViewExports(view_exports_) > 0) {
        return std::unexpected(EcsError::ClearWhileViewed);
    }

    for (std::uint32_t i = 0; i < component_count_; ++i) {
        components_[i].EraseAll();
    }

    const std::uint32_t capacity = static_cast<std::uint32_t>(generations_.size());
    for (std::uint32_t i = 0; i < capacity; ++i) {
        free_indices_[i] = (capacity - 1) - i;
        generations_[i]++;
    }
    free_count_      = capacity;
    active_entities_ = 0;
    return {};
}

// --- Components -----------------------------------------------------------

std::expected<Registry::ComponentId, Error>
Registry::RegisterComponent(std::uint32_t element_size) {
    const std::scoped_lock lock(mutex_);

    if (component_count_ >= components_.size()) {
        // Registering reallocates the component bank, which would move the
        // sets a live proxy is pointing into.
        if (LiveViewExports(view_exports_) > 0) {
            return std::unexpected(EcsError::RegisterWhileViewed);
        }
        const std::uint32_t old_cap = static_cast<std::uint32_t>(components_.size());
        const std::uint32_t new_cap =
            old_cap == 0 ? kInitialComponentCapacity : old_cap * 2;

        Buffer<SparseSet> next;
        if (!next.allocate(new_cap)) {
            return std::unexpected(EcsError::OutOfMemory);
        }
        for (std::uint32_t i = 0; i < component_count_; ++i) {
            next[i] = std::move(components_[i]);
        }
        components_ = std::move(next);
    }

    const ComponentId id = component_count_++;
    components_[id]      = SparseSet(element_size);
    return id;
}

std::expected<void, Error>
Registry::Add(Entity entity, ComponentId component,
              std::optional<std::span<const std::byte>> payload) {
    // Validation order matches the pre-migration binding: entity first, then
    // component id, then payload size.
    const std::scoped_lock lock(mutex_);
    const std::uint32_t index = entity.Index();
    if (index >= generations_.size() || generations_[index] != entity.Generation()) {
        return std::unexpected(EcsError::InvalidEntity);
    }
    if (component >= component_count_) {
        return std::unexpected(EcsError::InvalidComponent);
    }

    SparseSet &set = components_[component];

    std::span<const std::byte> bytes;
    if (payload) {
        if (payload->size() != set.element_size()) {
            return std::unexpected(EcsError::ComponentSizeMismatch);
        }
        bytes = *payload;
    }
    return set.Insert(entity, bytes, LiveViewExports(view_exports_));
}

std::expected<void, Error> Registry::Remove(Entity entity, ComponentId component) {
    const std::scoped_lock lock(mutex_);
    if (component >= component_count_) {
        return std::unexpected(EcsError::InvalidComponent);
    }
    // A stale entity simply has nothing to remove; this has never raised.
    components_[component].Erase(entity);
    return {};
}

std::expected<bool, Error> Registry::Has(Entity entity, ComponentId component) const {
    const std::scoped_lock lock(mutex_);
    if (component >= component_count_) {
        return std::unexpected(EcsError::InvalidComponent);
    }
    return components_[component].Has(entity);
}

std::expected<std::optional<std::span<const std::byte>>, Error>
Registry::Get(Entity entity, ComponentId component) const {
    const std::scoped_lock lock(mutex_);
    if (component >= component_count_) {
        return std::unexpected(EcsError::InvalidComponent);
    }
    const SparseSet &set = components_[component];
    const std::optional<std::uint32_t> dense = set.DenseIndexOf(entity);
    if (!dense) {
        return std::optional<std::span<const std::byte>>{};
    }
    return std::optional{set.ElementAt(*dense)};
}

std::uint32_t Registry::ComponentCount(ComponentId component) const noexcept {
    const std::scoped_lock lock(mutex_);
    const SparseSet *set = SetAt(component);
    return set != nullptr ? set->count() : 0;
}

std::uint32_t Registry::ComponentElementSize(ComponentId component) const noexcept {
    const std::scoped_lock lock(mutex_);
    const SparseSet *set = SetAt(component);
    return set != nullptr ? set->element_size() : 0;
}

std::expected<Registry::ComponentView, Error> Registry::View(ComponentId component) {
    const std::scoped_lock lock(mutex_);
    SparseSet *set = SetAt(component);
    if (set == nullptr) {
        return std::unexpected(EcsError::InvalidComponent);
    }
    return ComponentView{.entities     = set->dense_entities(),
                         .data         = set->element_data(),
                         .count        = set->count(),
                         .element_size = set->element_size()};
}

} // namespace CPH
