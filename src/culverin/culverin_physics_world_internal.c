#include "culverin_physics_world_internal.h"

static void free_new_buffers(NewBuffers *nb) {
    PyMem_RawFree(nb->pos);  PyMem_RawFree(nb->rot);
    PyMem_RawFree(nb->ppos); PyMem_RawFree(nb->prot);
    PyMem_RawFree(nb->lvel); PyMem_RawFree(nb->avel);
    PyMem_RawFree(nb->bids); PyMem_RawFree(nb->udat);
    PyMem_RawFree(nb->gens); PyMem_RawFree(nb->s2d);
    PyMem_RawFree(nb->d2s);  PyMem_RawFree(nb->stat);
    PyMem_RawFree(nb->free); PyMem_RawFree(nb->cats);
    PyMem_RawFree(nb->masks); PyMem_RawFree(nb->mats);
}

static int alloc_new_buffers(NewBuffers *nb, size_t cap) {
    memset(nb, 0, sizeof(NewBuffers));
    size_t f4 = cap * 4 * sizeof(float);
    
    nb->pos = PyMem_RawMalloc(f4);  nb->rot = PyMem_RawMalloc(f4);
    nb->ppos = PyMem_RawMalloc(f4); nb->prot = PyMem_RawMalloc(f4);
    nb->lvel = PyMem_RawMalloc(f4); nb->avel = PyMem_RawMalloc(f4);

    nb->bids  = PyMem_RawMalloc(cap * sizeof(JPH_BodyID));
    nb->udat  = PyMem_RawMalloc(cap * sizeof(uint64_t));
    nb->gens  = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->s2d   = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->d2s   = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->stat  = PyMem_RawMalloc(cap * sizeof(uint8_t));
    nb->free  = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->cats  = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->masks = PyMem_RawMalloc(cap * sizeof(uint32_t));
    nb->mats  = PyMem_RawMalloc(cap * sizeof(uint32_t));

    if (!nb->pos || !nb->rot || !nb->ppos || !nb->prot || !nb->lvel || !nb->avel ||
        !nb->bids || !nb->udat || !nb->gens || !nb->s2d || !nb->d2s || !nb->stat ||
        !nb->free || !nb->cats || !nb->masks || !nb->mats) {
        free_new_buffers(nb);
        return -1;
    }
    return 0;
}

static void migrate_and_init(PhysicsWorldObject *self, NewBuffers *nb, size_t new_cap) {
    size_t stride = 4 * sizeof(float);
    if (self->count > 0) {
        memcpy(nb->pos,  self->positions,         self->count * stride);
        memcpy(nb->rot,  self->rotations,         self->count * stride);
        memcpy(nb->ppos, self->prev_positions,    self->count * stride);
        memcpy(nb->prot, self->prev_rotations,    self->count * stride);
        memcpy(nb->lvel, self->linear_velocities, self->count * stride);
        memcpy(nb->avel, self->angular_velocities,self->count * stride);
        memcpy(nb->bids, self->body_ids,          self->count * sizeof(JPH_BodyID));
        memcpy(nb->udat, self->user_data,         self->count * sizeof(uint64_t));
        memcpy(nb->cats, self->categories,        self->count * sizeof(uint32_t));
        memcpy(nb->masks,self->masks,             self->count * sizeof(uint32_t));
        memcpy(nb->mats, self->material_ids,      self->count * sizeof(uint32_t));
    }

    memcpy(nb->gens, self->generations, self->slot_capacity * sizeof(uint32_t));
    memcpy(nb->s2d,  self->slot_to_dense, self->slot_capacity * sizeof(uint32_t));
    memcpy(nb->d2s,  self->dense_to_slot, self->slot_capacity * sizeof(uint32_t));
    memcpy(nb->stat, self->slot_states,   self->slot_capacity * sizeof(uint8_t));
    memcpy(nb->free, self->free_slots,    self->free_count * sizeof(uint32_t));

    for (size_t i = self->slot_capacity; i < new_cap; i++) {
        nb->gens[i] = 1;
        nb->stat[i] = SLOT_EMPTY;
        nb->free[self->free_count++] = (uint32_t)i;
    }
}

int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
    // 1. Validation
    if (self->view_export_count > 0) {
        PyErr_SetString(PyExc_BufferError, "Cannot resize while views are exported.");
        return -1;
    }
    BLOCK_UNTIL_NOT_QUERYING(self);
    if (new_capacity <= self->capacity) return 0;

    // 2. Transactional Allocation
    NewBuffers nb;
    if (alloc_new_buffers(&nb, new_capacity) < 0) {
        PyErr_NoMemory();
        return -1;
    }

    // 3. Data Migration
    migrate_and_init(self, &nb, new_capacity);

    // 4. Commit: Free OLD, assign NEW
    PyMem_RawFree(self->positions);          self->positions = nb.pos;
    PyMem_RawFree(self->rotations);          self->rotations = nb.rot;
    PyMem_RawFree(self->prev_positions);     self->prev_positions = nb.ppos;
    PyMem_RawFree(self->prev_rotations);     self->prev_rotations = nb.prot;
    PyMem_RawFree(self->linear_velocities);  self->linear_velocities = nb.lvel;
    PyMem_RawFree(self->angular_velocities); self->angular_velocities = nb.avel;
    
    PyMem_RawFree(self->body_ids);           self->body_ids = nb.bids;
    PyMem_RawFree(self->user_data);          self->user_data = nb.udat;
    PyMem_RawFree(self->generations);        self->generations = nb.gens;
    PyMem_RawFree(self->slot_to_dense);      self->slot_to_dense = nb.s2d;
    PyMem_RawFree(self->dense_to_slot);      self->dense_to_slot = nb.d2s;
    PyMem_RawFree(self->slot_states);        self->slot_states = nb.stat;
    PyMem_RawFree(self->free_slots);         self->free_slots = nb.free;
    PyMem_RawFree(self->categories);         self->categories = nb.cats;
    PyMem_RawFree(self->masks);              self->masks = nb.masks;
    PyMem_RawFree(self->material_ids);       self->material_ids = nb.mats;

    self->capacity = new_capacity;
    self->slot_capacity = new_capacity;
    return 0;
}