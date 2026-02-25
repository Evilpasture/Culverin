#pragma once
#include "culverin.h"

int PhysicsWorld_resize(struct PhysicsWorldObject *self, size_t new_capacity);

void free_constraints(PhysicsWorldObject *self);

void free_shadow_buffers(PhysicsWorldObject *self);

void PhysicsWorld_free_members(PhysicsWorldObject *self);

int init_settings(PhysicsWorldObject *self, PyObject *settings_dict, float *gx, float *gy,
                  float *gz, int *max_bodies, int *max_pairs);

int init_jolt_core(PhysicsWorldObject *self, WorldLimits limits, GravityVector gravity);

int allocate_buffers(PhysicsWorldObject *self, int max_bodies);

int load_baked_scene(PhysicsWorldObject *self, PyObject *baked);

int verify_abi_alignment(JPH_BodyInterface *bi);

void PhysicsWorld_releasebuffer(PhysicsWorldObject *self, Py_buffer *view);

void free_new_buffers(NewBuffers *nb);
// --- Allocator wrappers ---

// Helper to align a pointer to the next N-byte boundary
static inline void *culv_align_ptr(void *ptr, size_t alignment) {
    return (void *)(((uintptr_t)ptr + (alignment - 1)) & ~(uintptr_t)(alignment - 1));
}

// We allocate (Size + Alignment) so we always have room to 'nudge'
// We store the original pointer just before the aligned one to recover it for Free
static inline void *CulvMem_RawMallocAligned(size_t size, size_t alignment) {
    // Total allocation: size + alignment + space to store the original pointer
    size_t total_size = size + alignment + sizeof(void *);
    void *raw         = PyMem_RawMalloc(total_size);
    if (!(bool)raw)
        return NULL;

    // Calculate the aligned address, leaving space for the header
    void *aligned = culv_align_ptr((char *)raw + sizeof(void *), alignment);

    // Verify alignment (debug builds only)
    assert(((uintptr_t)aligned & (alignment - 1)) == 0);

    // Store the original 'raw' pointer immediately before the 'aligned' pointer
    ((void **)aligned)[-1] = raw;

    return aligned;
}

static inline void CulvMem_RawFreeAligned(void *aligned) {
    if (!(bool)aligned)
        return;
    // Retrieve the original pointer stored in the header
    void *raw = ((void **)aligned)[-1];
    PyMem_RawFree(raw);
}