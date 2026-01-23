#include "culverin.h"

// --- Helper: Shape Caching (Internal) ---
static JPH_Shape* find_or_create_shape(PhysicsWorldObject* self, int type, const float* params) {
    // 1. Construct Key
    ShapeKey key;
    memset(&key, 0, sizeof(ShapeKey));
    key.type = (uint32_t)type;
    
    // Normalize params based on type for cleaner cache hits
    if (type == 0) { // BOX
        key.p1 = params[0]; key.p2 = params[1]; key.p3 = params[2];
    } else if (type == 1) { // SPHERE
        key.p1 = params[0]; key.p2 = 0.0f; key.p3 = 0.0f;
    } else if (type == 2) { // CAPSULE
        key.p1 = params[0]; key.p2 = params[1]; key.p3 = 0.0f;
    }

    // 2. Search Cache (Linear Search is sufficient for < 100 unique shape types)
    // If you have thousands of *unique* shapes, this should be a hash map.
    for (size_t i = 0; i < self->shape_cache_count; i++) {
        ShapeKey* k = &self->shape_cache[i].key;
        // Exact float comparison is acceptable here because the input source
        // is the same Python byte buffer.
        if (k->type == key.type && k->p1 == key.p1 && k->p2 == key.p2 && k->p3 == key.p3) {
            return self->shape_cache[i].shape;
        }
    }

    // 3. Not Found -> Create New Jolt Shape
    JPH_Shape* shape = NULL;
    
    if (type == 0) { // BOX
        // Using {0} guarantees alignment safety on stack
        JPH_Vec3 he = {0}; 
        he.x = key.p1; he.y = key.p2; he.z = key.p3;
        
        JPH_BoxShapeSettings* s = JPH_BoxShapeSettings_Create(&he, 0.05f);
        shape = (JPH_Shape*)JPH_BoxShapeSettings_CreateShape(s);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)s);
    } 
    else if (type == 1) { // SPHERE
        JPH_SphereShapeSettings* s = JPH_SphereShapeSettings_Create(key.p1);
        shape = (JPH_Shape*)JPH_SphereShapeSettings_CreateShape(s);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)s);
    } 
    else if (type == 2) { // CAPSULE
        JPH_CapsuleShapeSettings* s = JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
        shape = (JPH_Shape*)JPH_CapsuleShapeSettings_CreateShape(s);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)s);
    }

    if (!shape) return NULL;

    // 4. Store in Cache
    // Grow capacity if needed
    if (self->shape_cache_count >= self->shape_cache_capacity) {
        size_t new_cap = (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
        void* new_ptr = PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
        if (!new_ptr) {
            JPH_Shape_Destroy(shape);
            return NULL; // OOM
        }
        self->shape_cache = (ShapeEntry*)new_ptr;
        self->shape_cache_capacity = new_cap;
    }

    // Insert new entry
    self->shape_cache[self->shape_cache_count].key = key;
    self->shape_cache[self->shape_cache_count].shape = shape;
    self->shape_cache_count++;

    return shape;
}

// --- Handle Helper ---
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
    return ((uint64_t)gen << 32) | (uint64_t)slot;
}

static inline bool unpack_handle(PhysicsWorldObject* self, BodyHandle h, uint32_t* slot) {
    *slot = (uint32_t)(h & 0xFFFFFFFF);
    uint32_t gen = (uint32_t)(h >> 32);
    
    if (*slot >= self->slot_capacity) return false;
    return self->generations[*slot] == gen;
}

// --- Lifecycle: Deallocation ---
static void PhysicsWorld_dealloc(PhysicsWorldObject* self) {
    // 1. Destroy Physics System FIRST
    // This releases all Jolt Body references to the shapes.
    // If we destroyed shapes before this, the system would segfault on shutdown.
    if (self->system) JPH_PhysicsSystem_Destroy(self->system);
    
    // 2. Destroy Unique Shapes (The Cache)
    // Now that the System is gone, the RefCount on these shapes should be 1 (ours).
    // Calling Destroy now frees the underlying C++ memory safely.
    if (self->shape_cache) {
        for (size_t i = 0; i < self->shape_cache_count; i++) {
            if (self->shape_cache[i].shape) {
                JPH_Shape_Destroy(self->shape_cache[i].shape);
            }
        }
        PyMem_RawFree(self->shape_cache);
    }

    // 3. Destroy Interfaces & Filters
    // TODO: no C API for these. Jolt can handle, but we should expose them later.
    // if (self->bp_filter) JPH_ObjectVsBroadPhaseLayerFilter_Destroy(self->bp_filter);
    // if (self->pair_filter) JPH_ObjectLayerPairFilter_Destroy(self->pair_filter);
    // if (self->bp_interface) JPH_BroadPhaseLayerInterface_Destroy(self->bp_interface);

    // 4. Destroy Job System (Last)
    if (self->job_system) JPH_JobSystem_Destroy(self->job_system);

    // 5. Free Shadow Buffers
    if (self->positions) PyMem_RawFree(self->positions);
    if (self->rotations) PyMem_RawFree(self->rotations);
    if (self->linear_velocities) PyMem_RawFree(self->linear_velocities);
    if (self->angular_velocities) PyMem_RawFree(self->angular_velocities);
    if (self->body_ids) PyMem_RawFree(self->body_ids);
    if (self->generations) PyMem_RawFree(self->generations);
    if (self->slot_to_dense) PyMem_RawFree(self->slot_to_dense);
    if (self->dense_to_slot) PyMem_RawFree(self->dense_to_slot);
    if (self->free_slots) PyMem_RawFree(self->free_slots);

    #if PY_VERSION_HEX < 0x030D0000
    if (self->shadow_lock) PyThread_free_lock(self->shadow_lock);
    #endif

    Py_TYPE(self)->tp_free((PyObject*)self);
}

// --- Lifecycle: Initialization ---
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"settings", "bodies", NULL};
    PyObject *settings_dict = NULL;
    PyObject *bodies_list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &settings_dict, &bodies_list)) {
        return -1;
    }

    // 1. Module State & Python Helpers
    PyObject *module = PyType_GetModule(Py_TYPE(self));
    if (!module) return -1;
    CulverinState *st = get_culverin_state(module);

    PyObject *val_func = PyObject_GetAttrString(st->helper, "validate_settings");
    if (!val_func) return -1;
    PyObject *norm_settings = PyObject_CallFunctionObjArgs(val_func, settings_dict ? settings_dict : Py_None, NULL);
    Py_DECREF(val_func);
    if (!norm_settings) return -1;

    float gx, gy, gz, slop;
    int max_bodies, max_pairs;
    PyArg_ParseTuple(norm_settings, "ffffii", &gx, &gy, &gz, &slop, &max_bodies, &max_pairs);
    Py_DECREF(norm_settings);

    // 2. Initialize Jolt Systems
    JobSystemThreadPoolConfig job_cfg = { .maxJobs = 1024, .maxBarriers = 8, .numThreads = -1 };
    self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
    
    // Configure Layers (0=Static, 1=Dynamic)
    self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(2, 2);
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 0, 0); 
    JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(self->bp_interface, 1, 1); 

    self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(2);
    JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 0); 
    JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1); 
    JPH_ObjectLayerPairFilterTable_DisableCollision(self->pair_filter, 0, 0); // Static ignores Static

    self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(self->bp_interface, 2, self->pair_filter, 2);

    JPH_PhysicsSystemSettings phys_settings = {
        .maxBodies = (uint32_t)max_bodies,
        .maxBodyPairs = (uint32_t)max_pairs,
        .maxContactConstraints = 10240,
        .broadPhaseLayerInterface = self->bp_interface,
        .objectLayerPairFilter = self->pair_filter,
        .objectVsBroadPhaseLayerFilter = self->bp_filter
    };
    self->system = JPH_PhysicsSystem_Create(&phys_settings);
    
    // Safe Gravity Init
    JPH_Vec3 gravity = {0}; 
    gravity.x = gx; gravity.y = gy; gravity.z = gz;
    JPH_PhysicsSystem_SetGravity(self->system, &gravity);
    
    self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);

    #if PY_VERSION_HEX < 0x030D0000
    self->shadow_lock = PyThread_allocate_lock();
    #endif

    // Initialize Shape Cache
    self->shape_cache = NULL;
    self->shape_cache_count = 0;
    self->shape_cache_capacity = 0;

    // 3. SCENE BAKING & MEMORY ALLOCATION
    PyObject *baked = NULL;
    size_t baked_count = 0;

    if (bodies_list && bodies_list != Py_None) {
        PyObject *bake_func = PyObject_GetAttrString(st->helper, "bake_scene");
        baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
        Py_DECREF(bake_func);
        if (!baked) return -1;
    }

    // Determine Capacity (Must accommodate max_bodies from settings)
    if (baked) {
        PyObject* o_cnt = PyTuple_GetItem(baked, 0);
        baked_count = (size_t)PyLong_AsSize_t(o_cnt);
    }

    self->count = baked_count;
    // Capacity must be at least count, but preferably max_bodies to avoid reallocs
    self->capacity = (size_t)max_bodies;
    if (self->capacity < self->count) self->capacity = self->count + 1024;

    // --- ALLOCATE DENSE SHADOW BUFFERS ---
    self->positions = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
    self->rotations = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
    self->linear_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
    self->angular_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
    self->body_ids = PyMem_RawMalloc(self->capacity * sizeof(JPH_BodyID));

    // --- ALLOCATE SPARSE SET / HANDLES ---
    self->slot_capacity = self->capacity;
    self->generations = PyMem_RawCalloc(self->slot_capacity, sizeof(uint32_t));
    self->slot_to_dense = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
    self->dense_to_slot = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
    self->free_slots = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
    self->free_count = 0;

    // Check Allocations
    if (!self->positions || !self->rotations || !self->body_ids ||
        !self->linear_velocities || !self->angular_velocities ||
        !self->generations || !self->slot_to_dense || !self->dense_to_slot || !self->free_slots) {
        Py_XDECREF(baked);
        PyErr_NoMemory();
        return -1;
    }

    // 4. BODY CREATION LOOP
    if (baked) {
        PyObject *o_pos = PyTuple_GetItem(baked, 1);
        PyObject *o_rot = PyTuple_GetItem(baked, 2);
        PyObject *o_shape = PyTuple_GetItem(baked, 3);
        PyObject *o_mot = PyTuple_GetItem(baked, 4);
        PyObject *o_layer = PyTuple_GetItem(baked, 5);

        float *f_pos = (float*)PyBytes_AsString(o_pos);
        float *f_rot = (float*)PyBytes_AsString(o_rot);
        float *f_shape = (float*)PyBytes_AsString(o_shape);
        unsigned char *u_mot = (unsigned char*)PyBytes_AsString(o_mot);
        unsigned char *u_layer = (unsigned char*)PyBytes_AsString(o_layer);

        JPH_BodyInterface* bi = self->body_interface;

        for (size_t i = 0; i < self->count; i++) {
            // Stack Alignment Safety
            #ifdef _MSC_VER
                __declspec(align(32)) JPH_RVec3 body_pos = {0};
                __declspec(align(16)) JPH_Quat body_rot = {0};
            #else
                JPH_RVec3 body_pos __attribute__((aligned(32))) = {0};
                JPH_Quat body_rot __attribute__((aligned(16))) = {0};
            #endif

            body_pos.x = (double)f_pos[i*4]; 
            body_pos.y = (double)f_pos[i*4+1]; 
            body_pos.z = (double)f_pos[i*4+2];

            body_rot.x = f_rot[i*4]; 
            body_rot.y = f_rot[i*4+1]; 
            body_rot.z = f_rot[i*4+2]; 
            body_rot.w = f_rot[i*4+3];

            float params[3] = {f_shape[i*4+1], f_shape[i*4+2], f_shape[i*4+3]};
            JPH_Shape* shape = find_or_create_shape(self, (int)f_shape[i*4], params);

            if (shape) {
                JPH_BodyCreationSettings* creation = JPH_BodyCreationSettings_Create3(
                    shape, &body_pos, &body_rot, 
                    (JPH_MotionType)u_mot[i], (JPH_ObjectLayer)u_layer[i]
                );

                // --- KEY HANDLE LOGIC ---
                // For initial bodies, Slot Index == Dense Index == i
                // We set Jolt UserData to the SLOT index (i)
                JPH_BodyCreationSettings_SetUserData(creation, (uint64_t)i);

                if (u_mot[i] == 2) JPH_BodyCreationSettings_SetAllowSleeping(creation, true);

                self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(bi, creation, JPH_Activation_Activate);
                JPH_BodyCreationSettings_Destroy(creation);
                
                // Initialize Indirection Maps
                self->generations[i] = 1;      // Start at Generation 1
                self->slot_to_dense[i] = (uint32_t)i;
                self->dense_to_slot[i] = (uint32_t)i;
            } else {
                self->body_ids[i] = JPH_INVALID_BODY_ID; 
            }
        }
        Py_DECREF(baked);
    }

    // 5. Initialize Free Slots
    // All slots beyond 'count' are available
    for (uint32_t i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity; i++) {
        self->generations[i] = 1;
        self->free_slots[self->free_count++] = i;
    }

    culverin_sync_shadow_buffers(self);
    return 0;
}

static PyObject* PhysicsWorld_apply_impulse(PhysicsWorldObject* self, PyObject* args) {
    uint64_t handle_raw;
    float ix, iy, iz;
    
    // Change 'i' to 'K' to accept the 64-bit Handle
    if (!PyArg_ParseTuple(args, "Kfff", &handle_raw, &ix, &iy, &iz)) return NULL;

    // 1. Unpack Handle
    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        // Option A: Silent fail (robust for games)
        // Py_RETURN_NONE; 
        
        // Option B: Raise error (better for debugging)
        PyErr_SetString(PyExc_ValueError, "Invalid or stale body handle");
        return NULL;
    }

    // 2. Convert Slot -> Dense Index
    uint32_t dense_idx = self->slot_to_dense[slot];

    // 3. Apply to Jolt Body
    JPH_BodyID bid = self->body_ids[dense_idx];
    
    // Note: Cast raw float args to JPH_Vec3. 
    // We use a compound literal for the pointer.
    JPH_Vec3 impulse = {ix, iy, iz};
    JPH_BodyInterface_AddImpulse(self->body_interface, bid, &impulse);
    
    Py_RETURN_NONE;
}

// ABI BYPASS. 
// WILL REPLACE WITH THE COMMENTED FUNCTION DEFINITION BELOW 
// IF I FOUND A WAY TO FIX ABI MISMATCH.
static PyObject* PhysicsWorld_raycast(PhysicsWorldObject* self, PyObject* args) {
    float sx, sy, sz, dx, dy, dz;
    float max_dist = 1000.0f; // Default max distance

    // Parse: raycast((sx, sy, sz), (dx, dy, dz), max_dist=1000.0)
    if (!PyArg_ParseTuple(args, "(fff)(fff)|f", &sx, &sy, &sz, &dx, &dy, &dz, &max_dist)) {
        return NULL;
    }

    // --- THE "MANUAL ALIGNMENT" HACK ---
    // We allocate enough bytes on the stack + 32 bytes of padding.
    // Then we calculate a pointer address that is guaranteed to be 32-byte aligned.
    // This solves the Segfault for AVX2/SSE without needing compiler-specific attributes.
    JPH_STACK_ALLOC(JPH_RVec3, origin);
    JPH_STACK_ALLOC(JPH_Vec3, direction);
    // -----------------------------------

    // 1. Fill Origin
    origin->x = sx; origin->y = sy; origin->z = sz;

    // 2. Fix Ray Length Logic
    // Normalize input direction and scale by max_dist
    float mag = sqrtf(dx*dx + dy*dy + dz*dz);
    if (mag < 1e-6f) Py_RETURN_NONE;
    
    float scale = max_dist / mag;
    direction->x = dx * scale;
    direction->y = dy * scale;
    direction->z = dz * scale;

    // 3. Initialize Hit Result
    // Jolt expects this to be clean, and 'fraction' helps early-out optimizations
    JPH_RayCastResult hit; 
    memset(&hit, 0, sizeof(JPH_RayCastResult));
    hit.fraction = 1.0f + 1e-4f; // Ideally slightly > 1.0

    const JPH_NarrowPhaseQuery* query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    
    // Pass our manually aligned pointers
    bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, &hit, NULL, NULL, NULL);

    if (!has_hit) Py_RETURN_NONE;

    // 1. Get Slot from UserData
    // In our new system, UserData IS the slot index.
    uint64_t slot_idx = JPH_BodyInterface_GetUserData(self->body_interface, hit.bodyID);

    // 2. Validate
    if (slot_idx >= self->slot_capacity) Py_RETURN_NONE;

    // 3. Construct Handle
    // We combine the fixed slot with the current generation
    uint32_t gen = self->generations[slot_idx];
    BodyHandle handle = make_handle((uint32_t)slot_idx, gen);

    return Py_BuildValue("Kf", handle, hit.fraction); // 'K' for uint64
}

// ==============================================NO UNCOMMENT. AND LEAVE THIS ALONE.==============================================================
// static PyObject* PhysicsWorld_raycast(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
//     float sx, sy, sz, dx, dy, dz;
//     float max_dist = 1000.0f;
//     static char *kwlist[] = {"start", "direction", "max_dist", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|f", kwlist, 
//                                      &sx, &sy, &sz, &dx, &dy, &dz, &max_dist)) {
//         return NULL;
//     }

//     float mag = sqrtf(dx*dx + dy*dy + dz*dz);
//     if (mag < 1e-6f) {
//         Py_RETURN_NONE;
//     }

//     // --- FIX START ---
//     // Enforce alignment! 
//     // JPH_Vec3 needs 16-byte alignment. JPH_RVec3 (Double) needs 32-byte (if AVX2).
//     #ifdef _MSC_VER
//         __declspec(align(32)) JPH_RVec3 ray_origin = {0};
//         __declspec(align(16)) JPH_Vec3 ray_dir = {0};
//     #else
//         JPH_RVec3 ray_origin __attribute__((aligned(32))) = {0};
//         JPH_Vec3 ray_dir __attribute__((aligned(16))) = {0};
//     #endif
//     // --- FIX END ---


//     // Stack Allocation: Zero-Init is key to prevent garbage padding
//     // JPH_RVec3 ray_origin = {0};
//     ray_origin.x = (double)sx;
//     ray_origin.y = (double)sy;
//     ray_origin.z = (double)sz;

//     // JPH_Vec3 ray_dir = {0};
//     float scale = max_dist / mag;
//     ray_dir.x = dx * scale;
//     ray_dir.y = dy * scale;
//     ray_dir.z = dz * scale;
//     JPH_RayCastResult hit = {0};
//     hit.fraction = 1.0f;

//     const JPH_NarrowPhaseQuery* query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    
//     // NarrowPhaseQuery is thread-safe for reads, so no locks needed here 
//     // IF we assume Update isn't mutating the tree concurrently (which it shouldn't be in your loop)
//     bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, &ray_origin, &ray_dir, &hit, NULL, NULL, NULL);

//     if (!has_hit) Py_RETURN_NONE;

//     uint64_t udata = JPH_BodyInterface_GetUserData(self->body_interface, hit.bodyID);
//     int hit_index = (int)udata;

//     if (hit_index < 0 || (size_t)hit_index >= self->count) {
//         Py_RETURN_NONE;
//     }

//     return Py_BuildValue("if", hit_index, hit.fraction);
// }
// ==============================================================================================================================================

  // --- Methods & Getters (Standard) ---

  static PyObject *PhysicsWorld_step(PhysicsWorldObject * self,
                                     PyObject * args) {
    float dt = 1.0f/60.0f;
    if (!PyArg_ParseTuple(args, "|f", &dt)) return NULL;

    Py_BEGIN_ALLOW_THREADS
    JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);
    SHADOW_LOCK(&self->shadow_lock);
    culverin_sync_shadow_buffers(self);
    SHADOW_UNLOCK(&self->shadow_lock);
    self->time += (double)dt;
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_remove_body(PhysicsWorldObject* self, PyObject* args) {
    uint64_t handle_raw;
    if (!PyArg_ParseTuple(args, "K", &handle_raw)) return NULL; // 'K' = unsigned long long

    SHADOW_LOCK(&self->shadow_lock);

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale body handle");
        return NULL;
    }

    // 1. Get Dense Index from Slot
    uint32_t dense_idx = self->slot_to_dense[slot];

    // 2. Jolt Cleanup
    JPH_BodyInterface* bi = self->body_interface;
    JPH_BodyID id_to_remove = self->body_ids[dense_idx];
    JPH_BodyInterface_RemoveBody(bi, id_to_remove);
    JPH_BodyInterface_DestroyBody(bi, id_to_remove);

    // 3. Swap and Pop (Dense Arrays)
    size_t last_dense = self->count - 1;

    if (dense_idx != last_dense) {
        // A. Swap Shadow Data
        memcpy(&self->positions[dense_idx*4], &self->positions[last_dense*4], 4 * sizeof(float));
        memcpy(&self->rotations[dense_idx*4], &self->rotations[last_dense*4], 4 * sizeof(float));
        memcpy(&self->linear_velocities[dense_idx*4], &self->linear_velocities[last_dense*4], 4 * sizeof(float));
        memcpy(&self->angular_velocities[dense_idx*4], &self->angular_velocities[last_dense*4], 4 * sizeof(float));
        self->body_ids[dense_idx] = self->body_ids[last_dense];

        // B. Update Maps
        // Who lived at the end?
        uint32_t mover_slot = self->dense_to_slot[last_dense];
        
        // Point that slot to the new location (dense_idx)
        self->slot_to_dense[mover_slot] = dense_idx;
        
        // Update reverse map
        self->dense_to_slot[dense_idx] = mover_slot;
        
        // NOTE: We do NOT need to update Jolt UserData! 
        // Jolt holds 'mover_slot', which is still valid and uniquely identifies that body.
    }

    // 4. Invalidate Handle
    self->generations[slot]++; // Increment generation implies old handle is dead
    self->free_slots[self->free_count++] = slot; // Recycle slot

    self->count--;
    self->view_shape[0] = (Py_ssize_t)self->count;

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_get_index(PhysicsWorldObject* self, PyObject* args) {
    uint64_t handle_raw;
    if (!PyArg_ParseTuple(args, "K", &handle_raw)) return NULL;

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        Py_RETURN_NONE; // Or -1
    }

    return PyLong_FromUnsignedLong(self->slot_to_dense[slot]);
}

static PyObject* make_view(PhysicsWorldObject* self, void* ptr) {
    if (!ptr) Py_RETURN_NONE;

    // 1. Update the persistent metadata on the object
    // This MUST stay alive as long as the memoryview exists
    self->view_shape[0] = (Py_ssize_t)self->count;
    self->view_shape[1] = 4;
    self->view_strides[0] = (Py_ssize_t)(4 * sizeof(float));
    self->view_strides[1] = (Py_ssize_t)sizeof(float);

    // 2. Manually fill the Py_buffer
    Py_buffer buf;
    memset(&buf, 0, sizeof(Py_buffer));
    
    buf.buf = ptr;
    buf.obj = (PyObject*)self;
    Py_INCREF(self); // The memoryview now owns a reference to 'self'
    
    buf.len = (Py_ssize_t)(self->count * 4 * sizeof(float));
    buf.readonly = 1;         // Read-only for safety
    buf.itemsize = sizeof(float);
    buf.format = "f";         // Signal that these are C-floats
    buf.ndim = 2;
    buf.shape = self->view_shape;     // Points to the array in our struct
    buf.strides = self->view_strides; // Points to the array in our struct

    // 3. Create the memoryview from our hand-crafted buffer
    PyObject* mv = PyMemoryView_FromBuffer(&buf);
    if (!mv) {
        Py_DECREF(self);
        return NULL;
    }
    return mv;
}

static PyObject* get_positions(PhysicsWorldObject* self, void* c) { return make_view(self, self->positions); }
static PyObject* get_rotations(PhysicsWorldObject* self, void* c) { return make_view(self, self->rotations); }
static PyObject* get_velocities(PhysicsWorldObject* self, void* c) { return make_view(self, self->linear_velocities); }
static PyObject* get_angular_velocities(PhysicsWorldObject* self, void* c) { return make_view(self, self->angular_velocities); }
static PyObject* get_count(PhysicsWorldObject* self, void* c) { return PyLong_FromSize_t(self->count); }
static PyObject* get_time(PhysicsWorldObject* self, void* c) { return PyFloat_FromDouble(self->time); }

// --- Type Definition ---

static PyGetSetDef PhysicsWorld_getset[] = {
    {"positions", (getter)get_positions, NULL, NULL, NULL},
    {"rotations", (getter)get_rotations, NULL, NULL, NULL},
    {"velocities", (getter)get_velocities, NULL, NULL, NULL},
    {"angular_velocities", (getter)get_angular_velocities, NULL, NULL, NULL},
    {"count", (getter)get_count, NULL, NULL, NULL},
    {"time", (getter)get_time, NULL, NULL, NULL},
    {NULL}
};

static PyMethodDef PhysicsWorld_methods[] = {
    {"step", (PyCFunction)PhysicsWorld_step, METH_VARARGS, NULL},
    {"apply_impulse", (PyCFunction)PhysicsWorld_apply_impulse, METH_VARARGS, NULL},
    {"raycast", (PyCFunction)PhysicsWorld_raycast, METH_VARARGS, NULL},
    {"remove_body", (PyCFunction)PhysicsWorld_remove_body, METH_VARARGS, NULL},
    {"get_index", (PyCFunction)PhysicsWorld_get_index, METH_VARARGS, NULL},
    {NULL}
};

static PyType_Slot PhysicsWorld_slots[] = {
    {Py_tp_new, PyType_GenericNew},
    {Py_tp_init, PhysicsWorld_init},
    {Py_tp_dealloc, PhysicsWorld_dealloc},
    {Py_tp_methods, PhysicsWorld_methods},
    {Py_tp_getset, PhysicsWorld_getset},
    {0, NULL},
};

static PyType_Spec PhysicsWorld_spec = {
    .name = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(PhysicsWorldObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = PhysicsWorld_slots,
};

// --- Module Initialization ---

static int culverin_exec(PyObject *m) {
    CulverinState *st = get_culverin_state(m);

    // 1. Initialize Jolt Globally (Safe)
    JPH_Init();

    // 2. Import Helper Module
    // We assume culverin._culverin is available in the python path
    st->helper = PyImport_ImportModule("culverin._culverin");
    if (!st->helper) return -1; // ImportError

    // 3. Create Heap Type
    st->PhysicsWorldType = PyType_FromModuleAndSpec(m, &PhysicsWorld_spec, NULL);
    if (!st->PhysicsWorldType) return -1;

    // 4. Add to Module
    Py_INCREF(st->PhysicsWorldType);
    if (PyModule_AddObject(m, "PhysicsWorld", st->PhysicsWorldType) < 0) {
        Py_DECREF(st->PhysicsWorldType);
        return -1;
    }

    return 0;
}

static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
    CulverinState *st = get_culverin_state(m);
    Py_VISIT(st->helper);
    Py_VISIT(st->PhysicsWorldType);
    return 0;
}

static int culverin_clear(PyObject *m) {
    CulverinState *st = get_culverin_state(m);
    Py_CLEAR(st->helper);
    Py_CLEAR(st->PhysicsWorldType);
    return 0;
}

static PyModuleDef_Slot culverin_slots[] = {
    {Py_mod_exec, culverin_exec},
    #if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
    #endif
    {0, NULL}
};

static PyModuleDef culverin_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_culverin_c",           // Internal binary name
    .m_doc = "Culverin Physics Engine Core",
    .m_size = sizeof(CulverinState),
    .m_slots = culverin_slots,
    .m_traverse = culverin_traverse,
    .m_clear = culverin_clear,
};

PyMODINIT_FUNC PyInit__culverin_c(void) {
    return PyModuleDef_Init(&culverin_module);
}
