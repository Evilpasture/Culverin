#include "debug/DebugRender.h"
#include "core/NativeAlloc.h"
#include "physics/PhysicsWorldState.h"

// --- Debug Buffer Helpers ---
void debug_buffer_ensure(DebugBuffer *buf, size_t count_needed) {
    if (buf->count + count_needed > buf->capacity) {
        constexpr size_t NEW_INITIAL_CAPACITY = 4096;
        size_t new_cap = (buf->capacity == 0) ? NEW_INITIAL_CAPACITY : buf->capacity * 2;
        while (buf->count + count_needed > new_cap) {
            new_cap *= 2;
        }

        void *new_ptr = CulvNative_Realloc(buf->data, new_cap * sizeof(DebugVertex));
        if (!new_ptr) {
            return; // Silent fail on OOM for debug info
        }

        buf->data     = (DebugVertex *)new_ptr;
        buf->capacity = new_cap;
    }
}

void debug_buffer_push(DebugBuffer *buf, DebugCoordinates pos, uint32_t color) {
    if (buf->count >= buf->capacity) {
        return; // Safety
    }
    buf->data[buf->count].x     = pos.x;
    buf->data[buf->count].y     = pos.y;
    buf->data[buf->count].z     = pos.z;
    buf->data[buf->count].color = color;
    buf->count++;
}

void debug_buffer_free(DebugBuffer *buf) {
    if (buf->data) {
        CulvNative_Free(buf->data);
    }
    buf->data     = nullptr;
    buf->count    = 0;
    buf->capacity = 0;
}

// --- Jolt Debug Callbacks ---
static void JPH_API_CALL OnDebugDrawLine(void *userData, const JPH_RVec3 *from, const JPH_RVec3 *to,
                                         JPH_Color color) {
    auto *world = (PhysicsWorld *)userData;
    debug_buffer_ensure(&world->debug_lines, 2);
    debug_buffer_push(&world->debug_lines,
                      (DebugCoordinates){(float)from->x, (float)from->y, (float)from->z}, color);
    debug_buffer_push(&world->debug_lines,
                      (DebugCoordinates){(float)to->x, (float)to->y, (float)to->z}, color);
}

static void JPH_API_CALL
OnDebugDrawTriangle(void *userData, const JPH_RVec3 *v1, const JPH_RVec3 *v2,
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    const JPH_RVec3 *v3, JPH_Color color,
                    [[maybe_unused]] JPH_DebugRenderer_CastShadow castShadow) {
    auto *world = (PhysicsWorld *)userData;
    constexpr int VERTS_PER_TRIANGLE = 3;
    debug_buffer_ensure(&world->debug_triangles, VERTS_PER_TRIANGLE);
    debug_buffer_push(&world->debug_triangles,
                      (DebugCoordinates){(float)v1->x, (float)v1->y, (float)v1->z}, color);
    debug_buffer_push(&world->debug_triangles,
                      (DebugCoordinates){(float)v2->x, (float)v2->y, (float)v2->z}, color);
    debug_buffer_push(&world->debug_triangles,
                      (DebugCoordinates){(float)v3->x, (float)v3->y, (float)v3->z}, color);
}

static void JPH_API_CALL OnDebugDrawText([[maybe_unused]] void *userData,
                                         [[maybe_unused]] const JPH_RVec3 *position,
                                         // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                                         [[maybe_unused]] const char *str,
                                         [[maybe_unused]] JPH_Color color,
                                         [[maybe_unused]] float height) {
    // Text is hard to batch efficiently to Python bytes.
    // Usually ignored or printed to stdout.
}

const JPH_DebugRenderer_Procs debug_procs = {.DrawLine     = OnDebugDrawLine,
                                             .DrawTriangle = OnDebugDrawTriangle,
                                             .DrawText3D   = OnDebugDrawText};
