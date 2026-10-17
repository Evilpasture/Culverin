#pragma once
#include "physics/PhysicsWorldState.h"

#ifdef __cplusplus
#    define CULV_NOEXCEPT noexcept
#else
#    define CULV_NOEXCEPT
#endif

#ifdef __cplusplus
extern "C" {
#endif
[[gnu::flatten, gnu::hot]]
void culverin_sync_shadow_buffers(const PhysicsWorld *world) CULV_NOEXCEPT;

#ifdef __cplusplus
}
#endif