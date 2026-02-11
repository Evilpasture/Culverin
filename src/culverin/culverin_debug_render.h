#pragma once
#include "joltc.h"
#include <stdint.h>

// 16-byte strided vertex for easy GPU upload: [X, Y, Z, Color(u32)]
typedef struct {
  float x, y, z;
  uint32_t color;
} DebugVertex;

// A simple dynamic array container
typedef struct {
  DebugVertex *data;
  size_t count;
  size_t capacity;
} DebugBuffer;

typedef struct {
  float x;
  float y;
  float z;
} DebugCoordinates;

void debug_buffer_ensure(DebugBuffer *buf, size_t count_needed);
void debug_buffer_push(DebugBuffer *buf, DebugCoordinates pos, uint32_t color);
void debug_buffer_free(DebugBuffer *buf);

extern const JPH_DebugRenderer_Procs debug_procs;