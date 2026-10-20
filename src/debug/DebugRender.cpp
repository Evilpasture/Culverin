#include "debug/DebugRender.hpp"

#include "physics/PhysicsWorldState.hpp"


namespace CPH {

void WorldDebugRenderer::DrawLine(JPH::RVec3Arg inFrom, JPH::RVec3Arg inTo,
                                  JPH::ColorArg inColor) {
    if (!world_->debug_lines.ensure(2)) {
        return;
    }
    const std::uint32_t color = inColor.GetUInt32();
    world_->debug_lines.push(static_cast<float>(inFrom.GetX()), static_cast<float>(inFrom.GetY()),
                             static_cast<float>(inFrom.GetZ()), color);
    world_->debug_lines.push(static_cast<float>(inTo.GetX()), static_cast<float>(inTo.GetY()),
                             static_cast<float>(inTo.GetZ()), color);
}

void WorldDebugRenderer::DrawTriangle(JPH::RVec3Arg inV1, JPH::RVec3Arg inV2, JPH::RVec3Arg inV3,
                                      JPH::ColorArg inColor,
                                      [[maybe_unused]] ECastShadow inCastShadow) {
    if (!world_->debug_triangles.ensure(3)) {
        return;
    }
    const std::uint32_t color = inColor.GetUInt32();
    const JPH::RVec3 vertices[3] = {inV1, inV2, inV3};
    for (const JPH::RVec3 &v : vertices) {
        world_->debug_triangles.push(static_cast<float>(v.GetX()), static_cast<float>(v.GetY()),
                                     static_cast<float>(v.GetZ()), color);
    }
}

void WorldDebugRenderer::DrawText3D([[maybe_unused]] JPH::RVec3Arg inPosition,
                                    [[maybe_unused]] const JPH::string_view &inString,
                                    [[maybe_unused]] JPH::ColorArg inColor,
                                    [[maybe_unused]] float inHeight) {
    // Text is not batched to Python bytes.
}

} // namespace CPH
