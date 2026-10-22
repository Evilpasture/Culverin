#include "physics/EcsSync.hpp"
#include "ecs/Registry.hpp"
#include "physics/PhysicsWorldState.hpp"

#include <cstring>

namespace CPH {

namespace {

// Component payload layouts the transform bridge understands. They are plain
// float triples and quads so that any language binding -- or a headless C++
// caller -- can read them without a schema.
constexpr std::uint32_t kPositionElementSize = sizeof(float) * 3;
constexpr std::uint32_t kRotationElementSize = sizeof(float) * 4;
constexpr std::uint32_t kHandleElementSize   = sizeof(std::uint64_t);

} // namespace

std::expected<void, Error> SyncEcsTransforms(PhysicsWorld      &world,
                                              Registry         &registry,
                                              const EcsSyncConfig &config) {
    // Cheap validation first, while holding only the registry lock.
    {
        const std::scoped_lock registry_lock(registry.mutex_);

        if (config.handleComponent >= registry.componentCount_) {
            return std::unexpected(EcsError::InvalidHandleComponent);
        }
        const SparseSet &handle_set = registry.components_[config.handleComponent];
        if (handle_set.element_size() != kHandleElementSize) {
            return std::unexpected(EcsError::HandleComponentSize);
        }

        const auto valid_index = [&](int component) {
            return component >= 0 &&
                   static_cast<std::uint32_t>(component) < registry.componentCount_;
        };
        if (valid_index(config.positionComponent) &&
            registry.components_[config.positionComponent].element_size() !=
                kPositionElementSize) {
            return std::unexpected(EcsError::PositionComponentSize);
        }
        if (valid_index(config.rotationComponent) &&
            registry.components_[config.rotationComponent].element_size() !=
                kRotationElementSize) {
            return std::unexpected(EcsError::RotationComponentSize);
        }
    }

    // Acquisition order is part of the engine's contract: shadow_lock before
    // the registry mutex, never the other way round.
    const std::scoped_lock shadow_lock(world.shadowLock);
    const std::scoped_lock registry_lock(registry.mutex_);

    const auto valid_index = [&](int component) {
        return component >= 0 &&
               static_cast<std::uint32_t>(component) < registry.componentCount_;
    };
    const std::uint32_t handle_component = config.handleComponent;
    const int position_component = config.positionComponent;
    const int rotation_component = config.rotationComponent;

    const SparseSet &handle_set = registry.components_[handle_component];
    const SparseSet *position_set =
        valid_index(position_component) ? &registry.components_[position_component] : nullptr;
    const SparseSet *rotation_set =
        valid_index(rotation_component) ? &registry.components_[rotation_component] : nullptr;

    for (std::uint32_t i = 0; i < handle_set.count(); ++i) {
        const Entity entity = handle_set.dense_entities()[i];

        BodyHandle body_handle;
        std::memcpy(&body_handle, handle_set.ElementAt(i).data(), kHandleElementSize);

        const std::optional<std::uint32_t> resolved = world.Resolve(body_handle);
        if (!resolved) {
            continue; // stale handle: the body is gone, the component stays
        }
        const std::uint32_t dense = world.bodies.slotToDense[*resolved];

        if (position_set != nullptr) {
            const std::optional<std::uint32_t> target = position_set->DenseIndexOf(entity);
            if (target) {
                const PosStride *in = &world.bodies.positions[dense];
                float *out = reinterpret_cast<float *>(
                    position_set->element_data() +
                    static_cast<std::size_t>(*target) * kPositionElementSize);
                out[0] = static_cast<float>(in->x);
                out[1] = static_cast<float>(in->y);
                out[2] = static_cast<float>(in->z);
            }
        }

        if (rotation_set != nullptr) {
            const std::optional<std::uint32_t> target = rotation_set->DenseIndexOf(entity);
            if (target) {
                const float *in =
                    reinterpret_cast<const float *>(&world.bodies.rotations[dense]);
                float *out = reinterpret_cast<float *>(
                    rotation_set->element_data() +
                    static_cast<std::size_t>(*target) * kRotationElementSize);
                std::memcpy(out, in, kRotationElementSize);
            }
        }
    }
    return {};
}

} // namespace CPH
