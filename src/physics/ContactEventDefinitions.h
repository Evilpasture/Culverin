#pragma once
#include "core/Types.h"
#include <stddef.h>

// 1. Remove alignas from the inner struct
typedef struct {
    CPH_ATOMIC(BodyHandle) body1; // 8
    CPH_ATOMIC(BodyHandle) body2; // 8
    JPH_Real px, py, pz;           // 24/12
    float nx, ny, nz;              // 12
    float impulse;                 // 4
    float sliding_speed;           // 4
    uint32_t flags;                // 4
#if !defined(JPH_DOUBLE_PRECISION)
    uint32_t padding_magic[3];
#endif
} CPH_ContactEventSlim; // Just a 64-byte data block now

static_assert(sizeof(CPH_ContactEventSlim) == kMemoryAlignmentSize);

// 2. The Tail (Already 64 bytes)
typedef struct {
    // --- 8-byte Alignment Block (Offset 0 to 16) ---
    uint64_t udata1; // 8
    uint64_t udata2; // 8

    // --- 4-byte Alignment Block (Offset 16 to 64) ---
    float rvx, rvy, rvz; // 12 (Total 28)
    float toi;           // 4  (Total 32)
    float penetration;   // 4  (Total 36)

    uint32_t mat1; // 4  (Total 40)
    uint32_t mat2; // 4  (Total 44)

    uint32_t sub1; // 4  (Total 48)
    uint32_t sub2; // 4  (Total 52)

    uint32_t padding[3]; // 12 (Total 64!)
} CPH_ContactEventFatExt;

static_assert(sizeof(CPH_ContactEventFatExt) == kMemoryAlignmentSize);

// 3. The Outer Container (Where the alignment lives)
typedef struct CPH_ContactEvent {
    alignas(kMemoryAlignmentSize) CPH_ContactEventSlim slim; // Offset 0
    CPH_ContactEventFatExt fat;                               // Offset 64
} CPH_ContactEvent;

static_assert(sizeof(CPH_ContactEvent) == kMemoryAlignmentSize * 2);
static_assert(alignof(CPH_ContactEvent) == kMemoryAlignmentSize);
static_assert(offsetof(CPH_ContactEvent, fat) == kMemoryAlignmentSize);

static inline CPH_ContactEvent *GetEventAt(void *buffer, size_t index) {
    // Standard pointer arithmetic on ContactEvent* handles the 128-byte stride
    return &((CPH_ContactEvent *)buffer)[index];
}

static inline CPH_ContactEventSlim *GetSlimHeader(CPH_ContactEvent *event) {
    // Offset 0
    return &event->slim;
}

static inline CPH_ContactEventFatExt *GetFatExtension(CPH_ContactEvent *event) {
    // Offset 64 - guaranteed by our alignment wizardry
    return &event->fat;
}