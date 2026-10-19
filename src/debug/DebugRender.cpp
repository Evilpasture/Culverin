#include "debug/DebugRender.hpp"

#include "physics/PhysicsWorldState.hpp"


namespace CPH {

namespace {

void JPH_API_CALL OnDebugDrawLine(void *userData, const JPH_RVec3 *from, const JPH_RVec3 *to,
                                  JPH_Color color) {
    auto *world = static_cast<PhysicsWorld *>(userData);
    if (!world->debug_lines.ensure(2)) {
        return;
    }
    world->debug_lines.push(static_cast<float>(from->x), static_cast<float>(from->y),
                            static_cast<float>(from->z), color);
    world->debug_lines.push(static_cast<float>(to->x), static_cast<float>(to->y),
                            static_cast<float>(to->z), color);
}

void JPH_API_CALL OnDebugDrawTriangle(void *userData, const JPH_RVec3 *v1, const JPH_RVec3 *v2,
                                      const JPH_RVec3 *v3, JPH_Color color,
                                      [[maybe_unused]] JPH_DebugRenderer_CastShadow castShadow) {
    auto *world = static_cast<PhysicsWorld *>(userData);
    if (!world->debug_triangles.ensure(3)) {
        return;
    }
    const JPH_RVec3 *verts[3] = {v1, v2, v3};
    for (const JPH_RVec3 *v : verts) {
        world->debug_triangles.push(static_cast<float>(v->x), static_cast<float>(v->y),
                                    static_cast<float>(v->z), color);
    }
}

void JPH_API_CALL OnDebugDrawText([[maybe_unused]] void *userData,
                                  [[maybe_unused]] const JPH_RVec3 *position,
                                  [[maybe_unused]] const char *str,
                                  [[maybe_unused]] JPH_Color color,
                                  [[maybe_unused]] float height) {
    // Text is not batched to Python bytes.
}

} // namespace

const JPH_DebugRenderer_Procs kDebugProcs = {.DrawLine     = OnDebugDrawLine,
                                             .DrawTriangle = OnDebugDrawTriangle,
                                             .DrawText3D   = OnDebugDrawText};

} // namespace CPH
