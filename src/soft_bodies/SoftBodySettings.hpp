#pragma once

// The topology half of a soft body: vertices, faces, the generated edge/shear
// constraints and the optimization pass. This is Jolt data-structure work, so
// it lives next to the shadow storage rather than in the binding.
//
// The vertex/face builders take spans of already-parsed floats. Deciding
// whether a Python buffer is a float32 triplet is the binding's job; packing
// the elements into Jolt's AoS layout and validating the indices is this one.

#include "core/Error.hpp"
#include "jolt/Jolt.hpp"

#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>

#include <cstdint>
#include <expected>
#include <memory>
#include <span>

namespace CPH {

enum class SoftBodySettingsError : std::uint32_t {
    DegenerateFace = 1,
    FaceVertexOutOfRange,
    FaceOutOfRange,
    VertexOutOfRange,
    ConstraintsAlreadyCreated,
    ConstraintsMissing,
};

class SoftBodySettings {
  public:
    // Allocates the settings and pushes the default material Jolt needs to
    // resolve face collisions; the object holds one reference, matching what
    // joltc's JPH_SoftBodySharedSettings_Create left behind.
    [[nodiscard]] static std::unique_ptr<SoftBodySettings> Create();

    ~SoftBodySettings();

    SoftBodySettings(const SoftBodySettings &)            = delete;
    SoftBodySettings &operator=(const SoftBodySettings &) = delete;
    SoftBodySettings(SoftBodySettings &&)                 = delete;
    SoftBodySettings &operator=(SoftBodySettings &&)      = delete;

    [[nodiscard]] std::uint32_t vertex_count() const noexcept;
    [[nodiscard]] bool optimized() const noexcept { return optimized_; }
    [[nodiscard]] bool constraints_created() const noexcept { return constraints_created_; }

    void AddVertex(JPH::Float3 position, JPH::Float3 velocity, float inv_mass);
    // `inv_masses` and `velocities` may be empty, meaning "1.0 for every
    // vertex" and "zero", exactly as the optional buffers did.
    void AddVertices(std::span<const float> positions, std::span<const float> inv_masses,
                     std::span<const float> velocities);

    [[nodiscard]] std::expected<void, Error> AddFace(std::uint32_t v1, std::uint32_t v2,
                                                     std::uint32_t v3, std::uint32_t material);
    // `out_bad_face` receives the index of the offending face on
    // FaceOutOfRange; it may be null.
    [[nodiscard]] std::expected<void, Error> AddFaces(std::span<const std::uint32_t> indices,
                                                      std::span<const std::uint32_t> materials,
                                                      std::uint32_t *out_bad_face);

    // `bend_type` is a JPH::SoftBodySharedSettings::EBendType value; the range
    // check and its diagnostic belong to the caller.
    [[nodiscard]] std::expected<void, Error> CreateConstraints(float compliance, int bend_type);
    [[nodiscard]] std::expected<void, Error> Optimize();

    [[nodiscard]] std::expected<JPH::Float3, Error> GetVertexPosition(std::uint32_t index) const;

    [[nodiscard]] JPH::SoftBodySharedSettings *raw() const noexcept { return settings_; }

  private:
    SoftBodySettings() = default;

    JPH::SoftBodySharedSettings *settings_       = nullptr;
    bool                         constraints_created_ = false;
    bool                         optimized_           = false;
};

} // namespace CPH
