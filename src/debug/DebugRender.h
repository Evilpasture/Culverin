#pragma once
#include "jolt/JoltConfig.h"
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>

// 16-byte strided vertex for easy GPU upload: [X, Y, Z, Color(u32)]
typedef struct {
    float x, y, z;
    uint32_t color;
} CPH_DebugVertex;

// A simple dynamic array container
typedef struct {
    CPH_DebugVertex *data;
    size_t count;
    size_t capacity;
} CPH_DebugBuffer;

typedef struct {
    float x;
    float y;
    float z;
} CPH_DebugCoordinates;

void CPH_DebugBufferEnsure(CPH_DebugBuffer *buf, size_t count_needed);
void CPH_DebugBufferPush(CPH_DebugBuffer *buf, CPH_DebugCoordinates pos, uint32_t color);
void CPH_DebugBufferFree(CPH_DebugBuffer *buf);

extern const JPH_DebugRenderer_Procs kDebugProcs;
