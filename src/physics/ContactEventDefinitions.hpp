#pragma once

// Contact event wire layout. This is a public contract: Python decodes
// these records directly (get_contact_events_raw/ex), so sizes and field
// offsets must not change. One event = 128 bytes: a 64-byte "slim" header
// with the hot fields plus a 64-byte fat extension.

#include "core/Compiler.hpp"
#include "core/Types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>

struct CPH_ContactEventSlim {
    std::atomic<BodyHandle> body1; // 8
    std::atomic<BodyHandle> body2; // 8
    JPH_Real px, py, pz;           // 24/12
    float nx, ny, nz;              // 12
    float impulse;                 // 4
    float sliding_speed;           // 4
    std::uint32_t flags;           // 4
#if !defined(JPH_DOUBLE_PRECISION)
    std::uint32_t padding_magic[3];
#endif
};

static_assert(sizeof(CPH_ContactEventSlim) == kMemoryAlignmentSize);

struct CPH_ContactEventFatExt {
    std::uint64_t udata1; // 8
    std::uint64_t udata2; // 8

    float rvx, rvy, rvz; // 12 (Total 28)
    float toi;           // 4  (Total 32)
    float penetration;   // 4  (Total 36)

    std::uint32_t mat1; // 4  (Total 40)
    std::uint32_t mat2; // 4  (Total 44)

    std::uint32_t sub1; // 4  (Total 48)
    std::uint32_t sub2; // 4  (Total 52)

    std::uint32_t padding[3]; // 12 (Total 64)
};

static_assert(sizeof(CPH_ContactEventFatExt) == kMemoryAlignmentSize);

struct CPH_ContactEvent {
    alignas(kMemoryAlignmentSize) CPH_ContactEventSlim slim; // Offset 0
    CPH_ContactEventFatExt fat;                              // Offset 64
};

static_assert(sizeof(CPH_ContactEvent) == kMemoryAlignmentSize * 2);
static_assert(alignof(CPH_ContactEvent) == kMemoryAlignmentSize);
static_assert(offsetof(CPH_ContactEvent, fat) == kMemoryAlignmentSize);
