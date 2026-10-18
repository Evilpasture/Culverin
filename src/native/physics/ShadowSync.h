#pragma once
#include "physics/PhysicsWorldState.h"

#ifdef __cplusplus
#    define CPH_NOEXCEPT noexcept
#else
#    define CPH_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif
[[gnu::flatten, gnu::hot]]
void culverin_sync_shadow_buffers(const CPH_PhysicsWorld *world) CPH_NOEXCEPT;

#ifdef __cplusplus
}
#endif