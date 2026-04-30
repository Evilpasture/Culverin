#pragma once

#include "culverin.h"
#include "culverin_soft_body.h"
#include "culverin_types.h"
#include <Python.h>
#ifdef _WIN32
#    include <malloc.h>
#endif

// Temporary container for resize
typedef struct {
    JPH_Real *pos, *ppos;
    float *rot, *prot, *lvel, *avel;
    JPH_BodyID *bids;
    uint64_t *udat;
    CULV_ATOMIC(uint32_t) * gens;
    uint32_t *s2d, *d2s, *free, *cats, *masks, *mats;
    CULV_ATOMIC(uint8_t) * stat;
    SoftBodyShadow *softs;
} NewBuffers;

typedef struct {
    int max_bodies;
    int max_pairs;
    int max_contact_constraints;
    int temp_allocator_size;
    int max_physics_jobs;
    int max_physics_barriers;
    int num_threads;
    float penetration_slop;
} WorldSettings;

typedef struct {
    float gx;
    float gy;
    float gz;
} GravityVector;

struct PhysicsWorldObject;

int PhysicsWorld_resize(struct PhysicsWorldObject *self, size_t new_capacity);

void free_constraints(struct PhysicsWorldObject *self);

void free_shadow_buffers(struct PhysicsWorldObject *self);

void PhysicsWorld_free_members(struct PhysicsWorldObject *self);

int init_settings(struct PhysicsWorldObject *self, PyObject *settings_dict, GravityVector *gravity, 
                  WorldSettings *settings);

int init_jolt_core(struct PhysicsWorldObject *self, WorldSettings settings, GravityVector gravity);

int allocate_buffers(struct PhysicsWorldObject *self, int max_bodies);

int load_baked_scene(struct PhysicsWorldObject *self, PyObject *baked);

int verify_abi_alignment(JPH_BodyInterface *bi);

PyType_DeclareSlot_StatusFromModule PhysicsWorld_getbuffer(struct PhysicsWorldObject *self,
                                                           Py_buffer *view,
                                                           CULV_MAYBE_UNUSED int flags);

PyType_DeclareSlot_VoidFromModule PhysicsWorld_releasebuffer(struct PhysicsWorldObject *self,
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
    void *ptr = nullptr;
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

[[gnu::always_inline, maybe_unused]]
static inline void internal_culverin_safe_free(void **ptr) {
    if ((ptr != nullptr) && ((*ptr) != nullptr)) {
        CULV_RAW_FREE(*ptr);
        *ptr = nullptr;
    }
}

[[gnu::always_inline, maybe_unused]]
static inline void internal_culverin_safe_free_aligned(void **ptr) {
    if ((ptr != nullptr) && ((*ptr) != nullptr)) {
        CulvMem_RawFreeAligned(*ptr);
        *ptr = nullptr;
    }
}

#define CULVERIN_SAFE_FREE(p) internal_culverin_safe_free((void **)&(p))
#define CULVERIN_SAFE_FREE_ALIGNED(p) internal_culverin_safe_free_aligned((void **)&(p))