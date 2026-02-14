#pragma once
#include "culverin.h"

typedef struct {
    const JPH_Body* body;
    uint32_t dense_idx;
} SyncWorkItem;

void culverin_sync_shadow_buffers(PhysicsWorldObject *self);