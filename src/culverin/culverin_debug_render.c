#include "culverin_debug_render.h"
#include "culverin.h"
#include <Python.h>


// --- Debug Buffer Helpers ---
void debug_buffer_ensure(DebugBuffer *buf, size_t count_needed) {
  if (buf->count + count_needed > buf->capacity) {
    size_t new_cap = (buf->capacity == 0) ? 4096 : buf->capacity * 2;
    while (buf->count + count_needed > new_cap) {
      new_cap *= 2;
    }

    void *new_ptr = PyMem_RawRealloc(buf->data, new_cap * sizeof(DebugVertex));
    if (!new_ptr) {
      return; // Silent fail on OOM for debug info
    }

    buf->data = (DebugVertex *)new_ptr;
    buf->capacity = new_cap;
  }
}

void debug_buffer_push(DebugBuffer *buf, DebugCoordinates pos, uint32_t color) {
  if (buf->count >= buf->capacity) {
    return; // Safety
  }
  buf->data[buf->count].x = pos.x;
  buf->data[buf->count].y = pos.y;
  buf->data[buf->count].z = pos.z;
  buf->data[buf->count].color = color;
  buf->count++;
}

void debug_buffer_free(DebugBuffer *buf) {
  if (buf->data) {
    PyMem_RawFree(buf->data);
  }
  buf->data = NULL;
  buf->count = 0;
  buf->capacity = 0;
}

// --- Jolt Debug Callbacks ---
static void JPH_API_CALL OnDebugDrawLine(void *userData, const JPH_RVec3 *from,
                                         const JPH_RVec3 *to, JPH_Color color) {
  auto *self = (PhysicsWorldObject *)userData;
  debug_buffer_ensure(&self->debug_lines, 2);
  debug_buffer_push(
      &self->debug_lines,
      (DebugCoordinates){(float)from->x, (float)from->y, (float)from->z},
      color);
  debug_buffer_push(
      &self->debug_lines,
      (DebugCoordinates){(float)to->x, (float)to->y, (float)to->z}, color);
}

static void JPH_API_CALL
OnDebugDrawTriangle(void *userData, const JPH_RVec3 *v1, const JPH_RVec3 *v2,
                    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                    const JPH_RVec3 *v3, JPH_Color color,
                    JPH_DebugRenderer_CastShadow castShadow) {
  PhysicsWorldObject *self = (PhysicsWorldObject *)userData;
  debug_buffer_ensure(&self->debug_triangles, 3);
  debug_buffer_push(
      &self->debug_triangles,
      (DebugCoordinates){(float)v1->x, (float)v1->y, (float)v1->z}, color);
  debug_buffer_push(
      &self->debug_triangles,
      (DebugCoordinates){(float)v2->x, (float)v2->y, (float)v2->z}, color);
  debug_buffer_push(
      &self->debug_triangles,
      (DebugCoordinates){(float)v3->x, (float)v3->y, (float)v3->z}, color);
}

static void JPH_API_CALL
OnDebugDrawText(void *userData, const JPH_RVec3 *position,
                // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                const char *str, JPH_Color color, float height) {
  // Text is hard to batch efficiently to Python bytes.
  // Usually ignored or printed to stdout.
}

const JPH_DebugRenderer_Procs debug_procs = {.DrawLine = OnDebugDrawLine,
                                             .DrawTriangle =
                                                 OnDebugDrawTriangle,
                                             .DrawText3D = OnDebugDrawText};
