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
void CPH_SyncShadowBuffers(const CPH_PhysicsWorld *world) CPH_NOEXCEPT;

#ifdef __cplusplus
}
#endif