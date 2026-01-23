#include "culverin.h"

// --- Helper: Shape Caching (Internal) ---
static JPH_Shape* find_or_create_shape(PhysicsWorldObject* self, int type, const float* params) {
    ShapeKey key = { (uint32_t)type, params[0], params[1], params[2], params[3] };

    for (size_t i = 0; i < self->shape_cache_count; i++) {
        if (memcmp(&self->shape_cache[i].key, &key, sizeof(ShapeKey)) == 0) {
            return self->shape_cache[i].shape;
        }
    }

    JPH_Shape* shape = NULL;
    if (type == 0) { // BOX
        JPH_Vec3 he = {key.p1, key.p2, key.p3};
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
    else if (type == 3) { // CYLINDER
        // Cylinder parameters: half_height, radius, convex_radius (defaulting to 0.05)
        JPH_CylinderShapeSettings* s = JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
        shape = (JPH_Shape*)JPH_CylinderShapeSettings_CreateShape(s);
        JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)s);
    }
    else if (type == 4) { // PLANE
        JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
        // Plane needs a "half-extent" for the broadphase (usually very large like 1000m)
        JPH_PlaneShapeSettings* s = JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
        shape = (JPH_Shape*)JPH_PlaneShapeSettings_CreateShape(s);
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

    float gx;
    float gy;
    float gz;
    float slop;
    int max_bodies;
    int max_pairs;
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
        baked_count = PyLong_AsSize_t(o_cnt);
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
    self->slot_states = PyMem_RawCalloc(self->slot_capacity, sizeof(uint8_t));
    self->free_count = 0;
    // Command Queue (Start small, grow as needed)
    self->command_capacity = 64;
    self->command_count = 0;
    self->command_queue = PyMem_RawMalloc(self->command_capacity * sizeof(PhysicsCommand));

    // Check Allocations
    if (!self->positions || !self->rotations || !self->body_ids ||
        !self->linear_velocities || !self->angular_velocities ||
        !self->generations || !self->slot_to_dense || !self->dense_to_slot || !self->free_slots ||
        !self->slot_states || !self->command_queue) {
        Py_XDECREF(baked);
        PyErr_NoMemory();
        return -1;
    }

    // Initialize Slot States
    for (uint32_t i = 0; i < (uint32_t)self->count; i++) {
        self->slot_states[i] = SLOT_ALIVE;
    }
    for (uint32_t i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity; i++) {
        self->slot_states[i] = SLOT_EMPTY;
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

static PyObject* PhysicsWorld_apply_impulse(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};

    // 1. Parse Arguments with Keywords
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x, &y, &z)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);

    // 2. Unpack and Validate Handle
    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale body handle");
        return NULL;
    }

    // 3. Verify object is currently in the simulation
    if (self->slot_states[slot] != SLOT_ALIVE) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Body is not active or is pending destruction");
        return NULL;
    }

    // 4. Resolve Body ID
    uint32_t dense_idx = self->slot_to_dense[slot];
    JPH_BodyID bid = self->body_ids[dense_idx];

    // We can release the shadow lock now. The JPH_BodyID is just an integer,
    // and Jolt's internal BodyInterface handles its own locking for the impulse.
    SHADOW_UNLOCK(&self->shadow_lock);

    // 5. Apply to Jolt
    JPH_Vec3 impulse = {x, y, z};
    JPH_BodyInterface_AddImpulse(self->body_interface, bid, &impulse);

    Py_RETURN_NONE;
}

// ABI BYPASS. 
// WILL REPLACE WITH THE COMMENTED FUNCTION DEFINITION BELOW 
// IF I FIND A WAY TO FIX ABI MISMATCH.
static PyObject* PhysicsWorld_raycast(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    float sx, sy, sz;
    float dx, dy, dz;
    float max_dist = 1000.0f;
    static char *kwlist[] = {"start", "direction", "max_dist", NULL};

    // 1. Parse Arguments with Keywords
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|f", kwlist, 
                                     &sx, &sy, &sz, 
                                     &dx, &dy, &dz, 
                                     &max_dist)) {
        return NULL;
    }

    // 2. ABI Alignment Hack
    JPH_STACK_ALLOC(JPH_RVec3, origin);
    JPH_STACK_ALLOC(JPH_Vec3, direction);

    origin->x = (double)sx; 
    origin->y = (double)sy; 
    origin->z = (double)sz;

    // 3. Fix Ray Length Logic
    float mag = sqrtf(dx*dx + dy*dy + dz*dz);
    if (mag < 1e-6f) Py_RETURN_NONE;
    
    float scale = max_dist / mag;
    direction->x = dx * scale;
    direction->y = dy * scale;
    direction->z = dz * scale;

    // 4. Initialize Hit Result
    JPH_RayCastResult hit; 
    memset(&hit, 0, sizeof(JPH_RayCastResult));
    hit.fraction = 1.0f + 1e-4f; 

    // 5. Execute Query
    const JPH_NarrowPhaseQuery* query = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);
    bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, &hit, NULL, NULL, NULL);

    if (!has_hit) Py_RETURN_NONE;

    // 6. Construct Stable Handle with Locking
    SHADOW_LOCK(&self->shadow_lock);
    
    // In Culverin, UserData IS the slot index
    uint64_t slot_idx = JPH_BodyInterface_GetUserData(self->body_interface, hit.bodyID);

    if (slot_idx >= self->slot_capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE;
    }

    uint32_t gen = self->generations[slot_idx];
    BodyHandle handle = make_handle((uint32_t)slot_idx, gen);
    
    SHADOW_UNLOCK(&self->shadow_lock);

    return Py_BuildValue("Kf", handle, hit.fraction);
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


// Helper to grow queue
static bool ensure_command_capacity(PhysicsWorldObject* self) {
    if (self->command_count >= self->command_capacity) {
        size_t new_cap = self->command_capacity * 2;
        void* new_ptr = PyMem_RawRealloc(self->command_queue, new_cap * sizeof(PhysicsCommand));
        if (!new_ptr) return false;
        self->command_queue = (PhysicsCommand*)new_ptr;
        self->command_capacity = new_cap;
    }
    return true;
}

static void flush_commands(PhysicsWorldObject* self) {
    if (self->command_count == 0) return;

    JPH_BodyInterface* bi = self->body_interface;

    for (size_t i = 0; i < self->command_count; i++) {
        PhysicsCommand* cmd = &self->command_queue[i];
        uint32_t slot = cmd->slot;

        // For all commands EXCEPT Create, we need the dense index and Jolt ID
        uint32_t dense_idx = 0;
        JPH_BodyID bid = 0;
        if (cmd->type != CMD_CREATE_BODY) {
            dense_idx = self->slot_to_dense[slot];
            bid = self->body_ids[dense_idx];
        }

        switch (cmd->type) {
            case CMD_CREATE_BODY: {
                JPH_BodyCreationSettings* s = cmd->data.create.settings;
                
                // 1. Create in Jolt
                JPH_BodyID new_bid = JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);
                
                // 2. Add to Dense Arrays
                size_t new_dense = self->count;
                self->body_ids[new_dense] = new_bid;
                self->slot_to_dense[slot] = (uint32_t)new_dense;
                self->dense_to_slot[new_dense] = slot;
                
                // 3. Force immediate Shadow Buffer sync for this new body
                JPH_STACK_ALLOC(JPH_RVec3, p);
                JPH_STACK_ALLOC(JPH_Quat, q);
                JPH_BodyInterface_GetPosition(bi, new_bid, p);
                JPH_BodyInterface_GetRotation(bi, new_bid, q);
                
                self->positions[new_dense*4+0] = (float)p->x;
                self->positions[new_dense*4+1] = (float)p->y;
                self->positions[new_dense*4+2] = (float)p->z;
                self->positions[new_dense*4+3] = 0.0f;
                
                self->rotations[new_dense*4+0] = q->x;
                self->rotations[new_dense*4+1] = q->y;
                self->rotations[new_dense*4+2] = q->z;
                self->rotations[new_dense*4+3] = q->w;

                self->count++;
                self->slot_states[slot] = SLOT_ALIVE;
                
                // 4. Cleanup settings wrapper
                JPH_BodyCreationSettings_Destroy(s);
                break;
            }

            case CMD_DESTROY_BODY: {
                // 1. Jolt Cleanup
                JPH_BodyInterface_RemoveBody(bi, bid);
                JPH_BodyInterface_DestroyBody(bi, bid);
                
                // 2. Swap and Pop Dense Arrays
                size_t last_dense = self->count - 1;
                if (dense_idx != last_dense) {
                    // Move Shadow Data
                    memcpy(&self->positions[dense_idx*4], &self->positions[last_dense*4], 16);
                    memcpy(&self->rotations[dense_idx*4], &self->rotations[last_dense*4], 16);
                    memcpy(&self->linear_velocities[dense_idx*4], &self->linear_velocities[last_dense*4], 16);
                    memcpy(&self->angular_velocities[dense_idx*4], &self->angular_velocities[last_dense*4], 16);
                    self->body_ids[dense_idx] = self->body_ids[last_dense];
                    
                    // Update Map for the body that moved
                    uint32_t mover_slot = self->dense_to_slot[last_dense];
                    self->slot_to_dense[mover_slot] = dense_idx;
                    self->dense_to_slot[dense_idx] = mover_slot;
                }
                
                // 3. Recycle Slot
                self->generations[slot]++;
                self->free_slots[self->free_count++] = slot;
                self->slot_states[slot] = SLOT_EMPTY;
                self->count--;
                break;
            }

            case CMD_SET_POS: {
                JPH_STACK_ALLOC(JPH_RVec3, p);
                p->x = cmd->data.vec.x; p->y = cmd->data.vec.y; p->z = cmd->data.vec.z;
                JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);
                
                // Sync Shadow
                self->positions[dense_idx*4+0] = cmd->data.vec.x;
                self->positions[dense_idx*4+1] = cmd->data.vec.y;
                self->positions[dense_idx*4+2] = cmd->data.vec.z;
                break;
            }

            case CMD_SET_ROT: {
                JPH_STACK_ALLOC(JPH_Quat, q);
                q->x = cmd->data.vec.x; q->y = cmd->data.vec.y; q->z = cmd->data.vec.z; q->w = cmd->data.vec.w;
                JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);
                
                // Sync Shadow
                self->rotations[dense_idx*4+0] = q->x;
                self->rotations[dense_idx*4+1] = q->y;
                self->rotations[dense_idx*4+2] = q->z;
                self->rotations[dense_idx*4+3] = q->w;
                break;
            }

            case CMD_SET_TRNS: {
                JPH_STACK_ALLOC(JPH_RVec3, p);
                p->x = cmd->data.transform.px; p->y = cmd->data.transform.py; p->z = cmd->data.transform.pz;
                JPH_STACK_ALLOC(JPH_Quat, q);
                q->x = cmd->data.transform.rx; q->y = cmd->data.transform.ry; q->z = cmd->data.transform.rz; q->w = cmd->data.transform.rw;
                
                JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q, JPH_Activation_Activate);
                
                self->positions[dense_idx*4+0] = (float)p->x;
                self->positions[dense_idx*4+1] = (float)p->y;
                self->positions[dense_idx*4+2] = (float)p->z;
                self->rotations[dense_idx*4+0] = q->x;
                self->rotations[dense_idx*4+1] = q->y;
                self->rotations[dense_idx*4+2] = q->z;
                self->rotations[dense_idx*4+3] = q->w;
                break;
            }

            case CMD_SET_LINVEL: {
                JPH_Vec3 v = {cmd->data.vec.x, cmd->data.vec.y, cmd->data.vec.z};
                JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);
                
                self->linear_velocities[dense_idx*4+0] = v.x;
                self->linear_velocities[dense_idx*4+1] = v.y;
                self->linear_velocities[dense_idx*4+2] = v.z;
                break;
            }

            case CMD_SET_ANGVEL: {
                JPH_Vec3 v = {cmd->data.vec.x, cmd->data.vec.y, cmd->data.vec.z};
                JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);
                
                self->angular_velocities[dense_idx*4+0] = v.x;
                self->angular_velocities[dense_idx*4+1] = v.y;
                self->angular_velocities[dense_idx*4+2] = v.z;
                break;
            }

            case CMD_SET_MOTION: {
                JPH_BodyInterface_SetMotionType(bi, bid, (JPH_MotionType)cmd->data.motion_type, JPH_Activation_Activate);
                break;
            }

            case CMD_ACTIVATE: {
                JPH_BodyInterface_ActivateBody(bi, bid);
                break;
            }

            case CMD_DEACTIVATE: {
                JPH_BodyInterface_DeactivateBody(bi, bid);
                break;
            }
            default: {
                continue;
            }
        }
    }

    self->command_count = 0;
    self->view_shape[0] = (Py_ssize_t)self->count;
}

static PyObject* PhysicsWorld_save_state(PhysicsWorldObject* self, PyObject* Py_UNUSED(unused)) {
    SHADOW_LOCK(&self->shadow_lock);

    // Header: count, capacity, slot_capacity, time
    size_t meta_size = sizeof(size_t) * 3 + sizeof(double);
    size_t dense_size = self->count * 4 * sizeof(float) * 4; // pos, rot, lin, ang
    size_t mapping_size = self->slot_capacity * (sizeof(uint32_t) * 4 + sizeof(uint8_t));
    
    size_t total_size = meta_size + dense_size + mapping_size;
    
    PyObject* bytes = PyBytes_FromStringAndSize(NULL, total_size);
    char* ptr = PyBytes_AsString(bytes);

    // 1. Meta
    memcpy(ptr, &self->count, sizeof(size_t)); ptr += sizeof(size_t);
    memcpy(ptr, &self->time, sizeof(double)); ptr += sizeof(double);

    // 2. Dense Buffers
    memcpy(ptr, self->positions, self->count * 16); ptr += self->count * 16;
    memcpy(ptr, self->rotations, self->count * 16); ptr += self->count * 16;
    memcpy(ptr, self->linear_velocities, self->count * 16); ptr += self->count * 16;
    memcpy(ptr, self->angular_velocities, self->count * 16); ptr += self->count * 16;

    // 3. Mapping State (Handles/Slots)
    memcpy(ptr, self->generations, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(ptr, self->slot_to_dense, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(ptr, self->dense_to_slot, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(ptr, self->slot_states, self->slot_capacity);

    SHADOW_UNLOCK(&self->shadow_lock);
    return bytes;
}

static PyObject* PhysicsWorld_load_state(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    Py_buffer view;
    static char *kwlist[] = {"state", NULL};

    // 1. Parse Arguments with Keywords
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*", kwlist, &view)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);
    
    char* ptr = (char*)view.buf;
    size_t saved_count;
    
    // Safety check: Ensure buffer is at least large enough for the count header
    if (view.len < (Py_ssize_t)sizeof(size_t)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Invalid snapshot: buffer too small");
        return NULL;
    }

    memcpy(&saved_count, ptr, sizeof(size_t)); 
    ptr += sizeof(size_t);
    
    if (saved_count > self->capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyBuffer_Release(&view);
        PyErr_SetString(PyExc_ValueError, "Snapshot exceeds world capacity");
        return NULL;
    }

    // 1. Restore Meta
    self->count = saved_count;
    // Update the view shape so Python's count and future memoryviews are correct
    self->view_shape[0] = (Py_ssize_t)self->count;

    memcpy(&self->time, ptr, sizeof(double)); 
    ptr += sizeof(double);

    // 2. Restore Dense Buffers
    memcpy(self->positions, ptr, self->count * 16); ptr += self->count * 16;
    memcpy(self->rotations, ptr, self->count * 16); ptr += self->count * 16;
    memcpy(self->linear_velocities, ptr, self->count * 16); ptr += self->count * 16;
    memcpy(self->angular_velocities, ptr, self->count * 16); ptr += self->count * 16;

    // 3. Restore Mappings
    memcpy(self->generations, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(self->slot_to_dense, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(self->dense_to_slot, ptr, self->slot_capacity * 4); ptr += self->slot_capacity * 4;
    memcpy(self->slot_states, ptr, self->slot_capacity); ptr += self->slot_capacity;

    // 4. CRITICAL: Re-Sync Jolt Source of Truth
    // We iterate through the restored dense array and tell Jolt to teleport
    // the existing BodyIDs to the saved coordinates.
    for (size_t i = 0; i < self->count; i++) {
        JPH_BodyID bid = self->body_ids[i];
        if (bid == JPH_INVALID_BODY_ID) continue;

        JPH_STACK_ALLOC(JPH_RVec3, p);
        p->x = (double)self->positions[i*4]; 
        p->y = (double)self->positions[i*4+1]; 
        p->z = (double)self->positions[i*4+2];
        
        JPH_STACK_ALLOC(JPH_Quat, q);
        q->x = self->rotations[i*4]; 
        q->y = self->rotations[i*4+1]; 
        q->z = self->rotations[i*4+2]; 
        q->w = self->rotations[i*4+3];

        JPH_BodyInterface_SetPositionAndRotation(self->body_interface, bid, p, q, JPH_Activation_Activate);
        
        JPH_Vec3 lv = {
            self->linear_velocities[i*4], 
            self->linear_velocities[i*4+1], 
            self->linear_velocities[i*4+2]
        };
        JPH_BodyInterface_SetLinearVelocity(self->body_interface, bid, &lv);
        
        JPH_Vec3 av = {
            self->angular_velocities[i*4], 
            self->angular_velocities[i*4+1], 
            self->angular_velocities[i*4+2]
        };
        JPH_BodyInterface_SetAngularVelocity(self->body_interface, bid, &av);
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    Py_RETURN_NONE;
}
// --- Methods & Getters (Standard) ---

static PyObject* PhysicsWorld_step(PhysicsWorldObject* self, PyObject* args) {
    float dt = 1.0f/60.0f;
    if (!PyArg_ParseTuple(args, "|f", &dt)) return NULL;

    Py_BEGIN_ALLOW_THREADS
    
    SHADOW_LOCK(&self->shadow_lock);
    flush_commands(self); // <--- FLUSH BEFORE STEP
    SHADOW_UNLOCK(&self->shadow_lock);

    JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);
    
    SHADOW_LOCK(&self->shadow_lock);
    culverin_sync_shadow_buffers(self);
    SHADOW_UNLOCK(&self->shadow_lock);
    
    self->time += (double)dt;
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_create_body(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    // Default values
    float px = 0.0f, py = 0.0f, pz = 0.0f;
    float rx = 0.0f, ry = 0.0f, rz = 0.0f, rw = 1.0f;
    float s[4] = {1.0f, 1.0f, 1.0f, 0.0f}; // Default size params
    int shape_type = 0;  // SHAPE_BOX
    int motion_type = 2; // MOTION_DYNAMIC
    
    PyObject* py_size = NULL;
    static char *kwlist[] = {"pos", "rot", "size", "shape", "motion", NULL};

    // 1. Parse Arguments
    // Note: size is parsed as a generic Object 'O' for flexible tuple length
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|(fff)(ffff)Oii", kwlist, 
        &px, &py, &pz, &rx, &ry, &rz, &rw, &py_size, &shape_type, &motion_type)) {
        return NULL;
    }

    // 2. Extract Size Params from Tuple
    if (py_size && PyTuple_Check(py_size)) {
        Py_ssize_t sz_len = PyTuple_Size(py_size);
        for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
            PyObject* item = PyTuple_GetItem(py_size, i);
            if (PyNumber_Check(item)) {
                s[i] = (float)PyFloat_AsDouble(item);
            }
        }
    }

    // 3. Validation Logic
    if (shape_type == 4 && motion_type != 0) { // SHAPE_PLANE (4)
        PyErr_SetString(PyExc_ValueError, "SHAPE_PLANE must be MOTION_STATIC");
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);

    // 4. Capacity and Slot Check
    if (self->count + self->command_count >= self->capacity) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_MemoryError, "World capacity reached. Increase max_bodies in settings.");
        return NULL;
    }

    if (self->free_count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_MemoryError, "No free slots available.");
        return NULL;
    }

    // 5. Reserve Slot
    uint32_t slot = self->free_slots[--self->free_count];

    // 6. Find/Create Jolt Shape
    JPH_Shape* shape = find_or_create_shape(self, shape_type, s);
    if (!shape) {
        self->free_slots[self->free_count++] = slot;
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create or retrieve physics shape.");
        return NULL;
    }

    // 7. Prepare Jolt Creation Settings
    // Use ABI alignment hack for stack vectors
    JPH_STACK_ALLOC(JPH_RVec3, pos);
    pos->x = (double)px; pos->y = (double)py; pos->z = (double)pz;
    
    JPH_STACK_ALLOC(JPH_Quat, rot);
    rot->x = rx; rot->y = ry; rot->z = rz; rot->w = rw;

    // Layer selection: 0 for Static, 1 for Moving
    uint32_t layer = (motion_type == 0) ? 0 : 1;

    JPH_BodyCreationSettings* settings = JPH_BodyCreationSettings_Create3(
        shape, pos, rot, (JPH_MotionType)motion_type, (JPH_ObjectLayer)layer
    );

    // CRITICAL: Set UserData to the SLOT index for handle resolution
    JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)slot);

    if (motion_type == 2) {
        JPH_BodyCreationSettings_SetAllowSleeping(settings, true);
    }

    // 8. Queue the Creation Command
    if (!ensure_command_capacity(self)) {
        JPH_BodyCreationSettings_Destroy(settings);
        self->free_slots[self->free_count++] = slot;
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    PhysicsCommand* cmd = &self->command_queue[self->command_count++];
    cmd->type = CMD_CREATE_BODY;
    cmd->slot = slot;
    cmd->data.create.settings = settings; // Command now owns this pointer

    // 9. Update State Machine
    self->slot_states[slot] = SLOT_PENDING_CREATE;

    // 10. Generate and Return Handle
    uint32_t gen = self->generations[slot];
    BodyHandle handle = make_handle(slot, gen);

    SHADOW_UNLOCK(&self->shadow_lock);
    
    return PyLong_FromUnsignedLongLong(handle);
}

static PyObject* PhysicsWorld_destroy_body(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    static char *kwlist[] = {"handle", NULL};

    // 1. Parse Arguments (Now with Keywords)
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
        return NULL;
    }

    SHADOW_LOCK(&self->shadow_lock);

    // 2. Validate Handle
    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    // 3. Verify State
    // We only destroy if it is ALIVE or was just queued for creation (PENDING_CREATE)
    if (self->slot_states[slot] == SLOT_ALIVE || self->slot_states[slot] == SLOT_PENDING_CREATE) {
        
        // 4. Queue Command
        if (!ensure_command_capacity(self)) {
            SHADOW_UNLOCK(&self->shadow_lock);
            return PyErr_NoMemory();
        }

        PhysicsCommand* cmd = &self->command_queue[self->command_count++];
        cmd->type = CMD_DESTROY_BODY;
        cmd->slot = slot;

        // Mark as pending so logic knows it's "dead" even before the next step()
        self->slot_states[slot] = SLOT_PENDING_DESTROY;
    }

    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_set_position(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x, &y, &z)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock); // LOCK
    
    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    uint32_t dense_idx = self->slot_to_dense[slot];
    JPH_BodyID bid = self->body_ids[dense_idx];

    // 1. Update Jolt (Thread-safe, but needs ABI alignment)
    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = (double)x; p->y = (double)y; p->z = (double)z;
    
    // We can unlock before calling into Jolt to minimize contention, 
    // BUT we must write to shadow buffers while locked to stay consistent with the slot mapping.
    // To be safe and simple: keep lock held.
    JPH_BodyInterface_SetPosition(self->body_interface, bid, p, JPH_Activation_Activate);

    // 2. Update Shadow Buffer
    self->positions[dense_idx * 4 + 0] = x;
    self->positions[dense_idx * 4 + 1] = y;
    self->positions[dense_idx * 4 + 2] = z;

    SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_set_rotation(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    float x, y, z, w;
    static char *kwlist[] = {"handle", "x", "y", "z", "w", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffff", kwlist, &handle_raw, &x, &y, &z, &w)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock); // LOCK

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) { 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }
    
    uint32_t dense_idx = self->slot_to_dense[slot];
    JPH_BodyID bid = self->body_ids[dense_idx];

    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = x; q->y = y; q->z = z; q->w = w;
    JPH_BodyInterface_SetRotation(self->body_interface, bid, q, JPH_Activation_Activate);

    self->rotations[dense_idx * 4 + 0] = x;
    self->rotations[dense_idx * 4 + 1] = y;
    self->rotations[dense_idx * 4 + 2] = z;
    self->rotations[dense_idx * 4 + 3] = w;

    SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_set_linear_velocity(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x, &y, &z)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock); // LOCK

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) { 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    uint32_t dense_idx = self->slot_to_dense[slot];
    JPH_BodyID bid = self->body_ids[dense_idx];

    JPH_Vec3 v = {x, y, z};
    JPH_BodyInterface_SetLinearVelocity(self->body_interface, bid, &v);

    self->linear_velocities[dense_idx * 4 + 0] = x;
    self->linear_velocities[dense_idx * 4 + 1] = y;
    self->linear_velocities[dense_idx * 4 + 2] = z;

    SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_set_angular_velocity(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    float x, y, z;
    static char *kwlist[] = {"handle", "x", "y", "z", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x, &y, &z)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock); // LOCK

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) { 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    uint32_t dense_idx = self->slot_to_dense[slot];
    JPH_BodyID bid = self->body_ids[dense_idx];

    JPH_Vec3 v = {x, y, z};
    JPH_BodyInterface_SetAngularVelocity(self->body_interface, bid, &v);

    self->angular_velocities[dense_idx * 4 + 0] = x;
    self->angular_velocities[dense_idx * 4 + 1] = y;
    self->angular_velocities[dense_idx * 4 + 2] = z;

    SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_get_motion_type(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    static char *kwlist[] = {"handle", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock); // LOCK

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) { 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
    // GetMotionType is safe to call, but we hold lock to ensure 'bid' is valid
    long mt = (long)JPH_BodyInterface_GetMotionType(self->body_interface, bid);
    
    SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
    return PyLong_FromLong(mt);
}

static PyObject* PhysicsWorld_set_motion_type(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    int motion_type;
    static char *kwlist[] = {"handle", "motion", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Ki", kwlist, &handle_raw, &motion_type)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock); // LOCK

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) { 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
    JPH_BodyInterface_SetMotionType(self->body_interface, bid, (JPH_MotionType)motion_type, JPH_Activation_Activate);
    
    SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_activate(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)){ 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    JPH_BodyInterface_ActivateBody(self->body_interface, self->body_ids[self->slot_to_dense[slot]]);
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_deactivate(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    static char *kwlist[] = {"handle", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)){ 
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    JPH_BodyInterface_DeactivateBody(self->body_interface, self->body_ids[self->slot_to_dense[slot]]);
    Py_RETURN_NONE;
}

static PyObject* PhysicsWorld_set_transform(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    float px, py, pz, rx, ry, rz, rw;
    static char *kwlist[] = {"handle", "pos", "rot", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K(fff)(ffff)", kwlist, 
                                     &handle_raw, &px, &py, &pz, &rx, &ry, &rz, &rw)) 
        return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
        return NULL;
    }

    if (ensure_command_capacity(self)) {
        PhysicsCommand* cmd = &self->command_queue[self->command_count++];
        cmd->type = CMD_SET_TRNS;
        cmd->slot = slot;
        cmd->data.transform.px = px; cmd->data.transform.py = py; cmd->data.transform.pz = pz;
        cmd->data.transform.rx = rx; cmd->data.transform.ry = ry; cmd->data.transform.rz = rz; cmd->data.transform.rw = rw;
    }
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
}

// Helper to deduce handle and append to list
static void append_hit(QueryContext* ctx, JPH_BodyID bid) {
    // 1. Get Slot from Jolt UserData
    uint64_t slot = JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);
    
    // 2. Bound Check
    if (slot >= ctx->world->slot_capacity) return;
    
    // 3. Validate Generation
    // We only care about ALIVE bodies. If generation mismatches, it's a stale ID (rare in queries but possible)
    uint32_t gen = ctx->world->generations[slot];
    
    // 4. Create Handle
    BodyHandle h = make_handle((uint32_t)slot, gen);
    
    PyObject* py_h = PyLong_FromUnsignedLongLong(h);
    PyList_Append(ctx->result_list, py_h);
    Py_DECREF(py_h);
}

// Callback for NarrowPhase (Sphere Overlap)
// Signature: float (*)(void *, const JPH_CollideShapeResult *)
static float OverlapCallback_Narrow(void* context, const JPH_CollideShapeResult* result) {
    QueryContext* ctx = (QueryContext*)context;
    append_hit(ctx, result->bodyID2); // bodyID2 is the body in the world
    return 1.0f; // Continue query
}

// Callback for BroadPhase (AABB Overlap)
// Signature: float (*)(void *, const JPH_BodyID)
static float OverlapCallback_Broad(void* context, const JPH_BodyID result) {
    QueryContext* ctx = (QueryContext*)context;
    append_hit(ctx, result);
    return 1.0f; // Continue query
}


static PyObject* PhysicsWorld_overlap_sphere(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    float x, y, z, radius;
    static char *kwlist[] = {"center", "radius", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &x, &y, &z, &radius)) {
        return NULL;
    }

    // 1. Create Shape
    JPH_SphereShapeSettings* ss = JPH_SphereShapeSettings_Create(radius);
    JPH_Shape* shape = (JPH_Shape*)JPH_SphereShapeSettings_CreateShape(ss);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)ss);

    // 2. Prepare Result List with Safety Check
    PyObject* results = PyList_New(0);
    if (!results) {
        JPH_Shape_Destroy(shape);
        return PyErr_NoMemory();
    }

    // 3. ABI Aligned Setup
    JPH_STACK_ALLOC(JPH_RVec3, pos);
    pos->x = (double)x; pos->y = (double)y; pos->z = (double)z;
    
    JPH_STACK_ALLOC(JPH_Quat, rot);
    rot->x = 0; rot->y = 0; rot->z = 0; rot->w = 1;
    
    JPH_STACK_ALLOC(JPH_RMat4, com_transform);
    JPH_RMat4_RotationTranslation(com_transform, rot, pos);
    
    JPH_STACK_ALLOC(JPH_Vec3, scale);
    scale->x = 1; scale->y = 1; scale->z = 1;

    // Use a real aligned zero vector for baseOffset instead of NULL
    JPH_STACK_ALLOC(JPH_RVec3, base_offset);
    base_offset->x = 0; base_offset->y = 0; base_offset->z = 0;

    // 4. Initialize Settings (Strict Zero-Init)
    JPH_CollideShapeSettings collide_settings;
    memset(&collide_settings, 0, sizeof(JPH_CollideShapeSettings));
    JPH_CollideShapeSettings_Init(&collide_settings);

    // 5. Query with valid Filters
    QueryContext ctx = {self, results};
    const JPH_NarrowPhaseQuery* nq = JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

    SHADOW_LOCK(&self->shadow_lock);
    JPH_NarrowPhaseQuery_CollideShape(
        nq, 
        shape, 
        scale, 
        com_transform, 
        &collide_settings, 
        base_offset,             // Pass aligned zero vec
        OverlapCallback_Narrow, 
        &ctx, 
        NULL,                    // BroadPhaseLayerFilter (NULL is usually OK here if BP is skipped)
        NULL,                    // ObjectLayerFilter
        NULL,                    // BodyFilter
        NULL                     // ShapeFilter
    );
    SHADOW_UNLOCK(&self->shadow_lock);

    JPH_Shape_Destroy(shape);
    return results;
}

static PyObject* PhysicsWorld_overlap_aabb(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;
    static char *kwlist[] = {"min", "max", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, 
                                     &min_x, &min_y, &min_z, 
                                     &max_x, &max_y, &max_z)) {
        return NULL;
    }

    JPH_AABox box;
    box.min.x = min_x; box.min.y = min_y; box.min.z = min_z;
    box.max.x = max_x; box.max.y = max_y; box.max.z = max_z;

    PyObject* results = PyList_New(0);
    if (!results) return PyErr_NoMemory();

    QueryContext ctx = {self, results};
    const JPH_BroadPhaseQuery* bq = JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);

    SHADOW_LOCK(&self->shadow_lock);
    // Note: We leave filters as NULL for now. 
    // If it still crashes here, we need to create a 'DefaultAll' filter in Init.
    JPH_BroadPhaseQuery_CollideAABox(bq, &box, OverlapCallback_Broad, &ctx, NULL, NULL);
    SHADOW_UNLOCK(&self->shadow_lock);

    return results;
}

// Change signature to include PyObject* kwds
static PyObject* PhysicsWorld_get_index(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    static char *kwlist[] = {"handle", NULL};
    
    // Fix: Pass kwds and kwlist
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    uint32_t slot;
    if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        SHADOW_UNLOCK(&self->shadow_lock);
        Py_RETURN_NONE; 
    }

    uint32_t dense = self->slot_to_dense[slot];
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyLong_FromUnsignedLong(dense);
}

static PyObject* PhysicsWorld_is_alive(PhysicsWorldObject* self, PyObject* args, PyObject* kwds) {
    uint64_t handle_raw;
    static char *kwlist[] = {"handle", NULL};

    // Fix: Use PyArg_ParseTupleAndKeywords
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) return NULL;

    SHADOW_LOCK(&self->shadow_lock);
    uint32_t slot;
    bool alive = false;
    if (unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
        uint8_t state = self->slot_states[slot];
        if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
            alive = true;
        }
    }
    SHADOW_UNLOCK(&self->shadow_lock);

    if (alive) Py_RETURN_TRUE;
    Py_RETURN_FALSE;
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
    {"create_body", (PyCFunction)PhysicsWorld_create_body, METH_VARARGS | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)PhysicsWorld_destroy_body, METH_VARARGS | METH_KEYWORDS, NULL},
    {"apply_impulse", (PyCFunction)PhysicsWorld_apply_impulse, METH_VARARGS | METH_KEYWORDS, NULL},
    {"raycast", (PyCFunction)PhysicsWorld_raycast, METH_VARARGS | METH_KEYWORDS, NULL},
    {"overlap_sphere", (PyCFunction)PhysicsWorld_overlap_sphere, METH_VARARGS | METH_KEYWORDS, NULL},
    {"overlap_aabb", (PyCFunction)PhysicsWorld_overlap_aabb, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_index", (PyCFunction)PhysicsWorld_get_index, METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_alive", (PyCFunction)PhysicsWorld_is_alive, METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_position", (PyCFunction)PhysicsWorld_set_position, METH_VARARGS, NULL},
    {"set_rotation", (PyCFunction)PhysicsWorld_set_rotation, METH_VARARGS, NULL},
    {"set_linear_velocity", (PyCFunction)PhysicsWorld_set_linear_velocity, METH_VARARGS, NULL},
    {"set_angular_velocity", (PyCFunction)PhysicsWorld_set_angular_velocity, METH_VARARGS, NULL},
    {"get_motion_type", (PyCFunction)PhysicsWorld_get_motion_type, METH_VARARGS, NULL},
    {"set_motion_type", (PyCFunction)PhysicsWorld_set_motion_type, METH_VARARGS, NULL},
    {"set_transform", (PyCFunction)PhysicsWorld_set_transform, METH_VARARGS | METH_KEYWORDS, NULL},
    {"activate", (PyCFunction)PhysicsWorld_activate, METH_VARARGS | METH_KEYWORDS, NULL},
    {"deactivate", (PyCFunction)PhysicsWorld_deactivate, METH_VARARGS | METH_KEYWORDS, NULL},
    {"save_state", (PyCFunction)PhysicsWorld_save_state, METH_NOARGS, NULL},
    {"load_state", (PyCFunction)PhysicsWorld_load_state, METH_VARARGS | METH_KEYWORDS, NULL},
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
    // --- Add Constants ---
    PyModule_AddIntConstant(m, "SHAPE_BOX", 0);
    PyModule_AddIntConstant(m, "SHAPE_SPHERE", 1);
    PyModule_AddIntConstant(m, "SHAPE_CAPSULE", 2);

    PyModule_AddIntConstant(m, "MOTION_STATIC", 0);
    PyModule_AddIntConstant(m, "MOTION_KINEMATIC", 1);
    PyModule_AddIntConstant(m, "MOTION_DYNAMIC", 2);
    PyModule_AddIntConstant(m, "SHAPE_CYLINDER", 3);
    PyModule_AddIntConstant(m, "SHAPE_PLANE", 4);

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
