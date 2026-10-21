#include "soft_bodies/SoftBodySettings.hpp"

#include <cstddef>

namespace CPH {

std::unique_ptr<SoftBodySettings> SoftBodySettings::Create() {
    std::unique_ptr<SoftBodySettings> result(new SoftBodySettings());
    result->settings_ = new JPH::SoftBodySharedSettings();
    // Jolt requires at least one material to resolve face collisions.
    result->settings_->mMaterials.push_back(JPH::PhysicsMaterial::sDefault);
    // joltc's JPH_SoftBodySharedSettings_Create returned an already-AddRef'd
    // object; match it so the refcount is unchanged.
    result->settings_->AddRef();
    return result;
}

SoftBodySettings::~SoftBodySettings() {
    if (settings_ != nullptr) {
        settings_->Release();
    }
}

std::uint32_t SoftBodySettings::vertex_count() const noexcept {
    return settings_ != nullptr ? static_cast<std::uint32_t>(settings_->mVertices.size()) : 0;
}

void SoftBodySettings::AddVertex(JPH::Float3 position, JPH::Float3 velocity, float inv_mass) {
    settings_->mVertices.push_back(JPH::SoftBodySharedSettings::Vertex(position, velocity, inv_mass));
}

void SoftBodySettings::AddVertices(std::span<const float> positions,
                                   std::span<const float> inv_masses,
                                   std::span<const float> velocities) {
    constexpr std::size_t kStride = 3;
    const std::size_t     count   = positions.size() / kStride;

    // joltc's AddVertices early-returned on a zero count and reserved once for
    // the batch; both preserved. Positions already arrive interleaved as one
    // Vertex per element, so the copy is a straight push per vertex.
    if (count != 0) {
        settings_->mVertices.reserve(settings_->mVertices.size() + count);
        for (std::size_t i = 0; i < count; ++i) {
            const std::size_t v = i * kStride;
            settings_->mVertices.push_back(JPH::SoftBodySharedSettings::Vertex(
                JPH::Float3(positions[v], positions[v + 1], positions[v + 2]),
                velocities.empty()
                    ? JPH::Float3(0, 0, 0)
                    : JPH::Float3(velocities[v], velocities[v + 1], velocities[v + 2]),
                inv_masses.empty() ? 1.0F : inv_masses[i]));
        }
    }
}

std::expected<void, Error> SoftBodySettings::AddFace(std::uint32_t v1, std::uint32_t v2,
                                                     std::uint32_t v3, std::uint32_t material) {
    if (v1 == v2 || v2 == v3 || v1 == v3) {
        return std::unexpected(SoftBodySettingsError::DegenerateFace);
    }

    const std::uint32_t count = vertex_count();
    if (v1 >= count || v2 >= count || v3 >= count) {
        return std::unexpected(SoftBodySettingsError::FaceVertexOutOfRange);
    }

    settings_->mFaces.push_back(JPH::SoftBodySharedSettings::Face(v1, v2, v3, material));
    return {};
}

std::expected<void, Error> SoftBodySettings::AddFaces(std::span<const std::uint32_t> indices,
                                                      std::span<const std::uint32_t> materials,
                                                      std::uint32_t *out_bad_face) {
    if (out_bad_face != nullptr) {
        *out_bad_face = 0;
    }

    // TODO(bug): `materials` is indexed by face, so a non-empty span shorter
    // than face_count reads past its end. The Python binding rejects a
    // mismatch and is the only caller, so this is unreachable today; it is a
    // precondition this API states nowhere and checks nowhere.
    constexpr std::size_t kStride     = 3;
    const std::size_t     face_count  = indices.size() / kStride;
    const std::uint32_t   vertex_count_now = vertex_count();

    // joltc's AddFaces reserved once and pushed each face; the zero-count early
    // return is part of that behaviour.
    if (face_count != 0) {
        settings_->mFaces.reserve(settings_->mFaces.size() + face_count);
    }

    for (std::size_t i = 0; i < face_count; ++i) {
        const std::size_t v = i * kStride;
        const std::uint32_t v1 = indices[v];
        const std::uint32_t v2 = indices[v + 1];
        const std::uint32_t v3 = indices[v + 2];

        if (v1 >= vertex_count_now || v2 >= vertex_count_now || v3 >= vertex_count_now) {
            if (out_bad_face != nullptr) {
                *out_bad_face = static_cast<std::uint32_t>(i);
            }
            return std::unexpected(SoftBodySettingsError::FaceOutOfRange);
        }

        settings_->mFaces.push_back(JPH::SoftBodySharedSettings::Face(
            v1, v2, v3, materials.empty() ? 0U : materials[i]));
    }
    return {};
}

std::expected<void, Error> SoftBodySettings::CreateConstraints(float compliance, int bend_type) {
    if (constraints_created_) {
        return std::unexpected(SoftBodySettingsError::ConstraintsAlreadyCreated);
    }
    constraints_created_ = true;

    // The binder handles both Edge compliance and Shear compliance. joltc built
    // a VertexAttributes array sized to the vertex count with only those two
    // fields set, leaving mBendCompliance at its FLT_MAX default and the LRA
    // fields at theirs; same thing here.
    JPH::Array<JPH::SoftBodySharedSettings::VertexAttributes> attributes;
    attributes.resize(settings_->mVertices.size());
    for (auto &attribute : attributes) {
        attribute.mCompliance      = compliance;
        attribute.mShearCompliance = compliance;
    }

    settings_->CreateConstraints(attributes.data(), static_cast<JPH::uint>(attributes.size()),
                                 static_cast<JPH::SoftBodySharedSettings::EBendType>(bend_type));
    return {};
}

std::expected<void, Error> SoftBodySettings::Optimize() {
    if (!constraints_created_) {
        return std::unexpected(SoftBodySettingsError::ConstraintsMissing);
    }
    settings_->Optimize();
    optimized_ = true;
    return {};
}

std::expected<JPH::Float3, Error> SoftBodySettings::GetVertexPosition(std::uint32_t index) const {
    // The caller has already range-checked against vertex_count(); the check is
    // kept because the diagnostic has to stay reachable.
    if (index >= settings_->mVertices.size()) {
        return std::unexpected(SoftBodySettingsError::VertexOutOfRange);
    }
    return settings_->mVertices[index].mPosition;
}

} // namespace CPH
