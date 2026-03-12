#pragma once
#include "culverin.h"
#include <stdlib.h>
#ifdef _WIN32
#    include <malloc.h>
#endif

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

PyType_DeclareSlot_VoidFromModule PhysicsWorld_releasebuffer(PhysicsWorldObject *self,
                                                             Py_buffer *view);

void free_new_buffers(NewBuffers *nb);
// --- Allocator wrappers ---

// Helper to align a pointer to the next N-byte boundary
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline void *culv_align_ptr(void *ptr, size_t alignment) {
    // Basic runtime check: alignment must be power of 2 and not 0
    // (In C23, you could also use a macro for this)
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        return ptr; // Or handle error
    }

    uintptr_t addr         = (uintptr_t)ptr;
    uintptr_t mask         = alignment - 1;
    uintptr_t aligned_addr = (addr + mask) & ~mask;

    size_t offset = (size_t)(aligned_addr - addr);
    return (unsigned char *)ptr + offset;
}

// We allocate (Size + Alignment) so we always have room to 'nudge'
// We store the original pointer just before the aligned one to recover it for Free
CULV_NODISCARD
CULV_MAYBE_UNUSED
static inline void *CulvMem_RawMallocAligned(size_t size, size_t alignment) {
    // Basic sanity: posix_memalign requires alignment to be a
    // multiple of sizeof(void*) and a power of two.
    if (alignment < sizeof(void *)) {
        alignment = sizeof(void *);
    }

#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    void *ptr = nullptr; // C23 nullptr
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}
CULV_MAYBE_UNUSED
static inline void CulvMem_RawFreeAligned(void *aligned) {
#if defined(_WIN32)
    _aligned_free(aligned);
#else
    free(aligned);
#endif
}
