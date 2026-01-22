#include "culverin.h"
#include <structmember.h>

// --- Memory Cleanup ---
static void PhysicsWorld_dealloc(PhysicsWorldObject* self) {
    if (self->system) JPH_PhysicsSystem_Destroy(self->system);
    if (self->job_system) JPH_JobSystem_Destroy(self->job_system);
    
    // TODO: we would destroy with these calls below but API is unavailable. Jolt can handle during shutdown, for now.
    // if (self->bp_filter) JPH_ObjectVsBroadPhaseLayerFilter_Destroy(self->bp_filter);
    // if (self->pair_filter) JPH_ObjectLayerPairFilter_Destroy(self->pair_filter);
    // if (self->bp_interface) JPH_BroadPhaseLayerInterface_Destroy(self->bp_interface);

    if (self->positions) PyMem_RawFree(self->positions);
    if (self->rotations) PyMem_RawFree(self->rotations);
    if (self->linear_velocities) PyMem_RawFree(self->linear_velocities);
    if (self->angular_velocities) PyMem_RawFree(self->angular_velocities);
    if (self->body_ids) PyMem_RawFree(self->body_ids);

    #if PY_VERSION_HEX < 0x030D0000
    if (self->shadow_lock) PyThread_free_lock(self->shadow_lock);
    #endif

    // Decrement the heap type
    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free((PyObject*)self);
    Py_DECREF(tp);
}

// --- The Core Initialization Logic ---
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"settings", "bodies", NULL};
    PyObject *settings_dict = NULL;
    PyObject *bodies_list = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &settings_dict, &bodies_list)) {
        return -1;
    }

    // 1. Retrieve Module State to get the Helper
    // For Heap Types, Py_TYPE(self) is the class, PyType_GetModule gets the defining module
    PyObject *module = PyType_GetModule(Py_TYPE(self));
    if (!module) {
        PyErr_SetString(PyExc_RuntimeError, "Could not locate defining module.");
        return -1;
    }
    CulverinState *st = get_culverin_state(module);
    if (!st || !st->helper) {
        PyErr_SetString(PyExc_RuntimeError, "Module state not initialized (helper missing).");
        return -1;
    }

    // 2. Validate Settings (Python Call)
    PyObject *val_func = PyObject_GetAttrString(st->helper, "validate_settings");
    if (!val_func) return -1;

    PyObject *norm_settings = PyObject_CallFunctionObjArgs(val_func, settings_dict ? settings_dict : Py_None, NULL);
    Py_DECREF(val_func);
    if (!norm_settings) return -1;

    // Unpack Settings Tuple: (gx, gy, gz, slop, max_bodies, max_pairs)
    float gx, gy, gz, slop;
    int max_bodies, max_pairs;
    if (!PyArg_ParseTuple(norm_settings, "ffffii", &gx, &gy, &gz, &slop, &max_bodies, &max_pairs)) {
        Py_DECREF(norm_settings);
        return -1;
    }
    Py_DECREF(norm_settings);

    // 3. Initialize Jolt Systems
    JobSystemThreadPoolConfig job_cfg = { .maxJobs = 1024, .maxBarriers = 8, .numThreads = -1 };
    self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
    
    self->bp_interface = JPH_BroadPhaseLayerInterfaceMask_Create(1);
    self->pair_filter = JPH_ObjectLayerPairFilterMask_Create();
    self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterMask_Create(self->bp_interface);

    JPH_PhysicsSystemSettings phys_settings = {
        .maxBodies = (uint32_t)max_bodies,
        .maxBodyPairs = (uint32_t)max_pairs,
        .maxContactConstraints = 10240, // Could also expose this
        .broadPhaseLayerInterface = self->bp_interface,
        .objectLayerPairFilter = self->pair_filter,
        .objectVsBroadPhaseLayerFilter = self->bp_filter
    };
    self->system = JPH_PhysicsSystem_Create(&phys_settings);
    if (!self->system) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create JPH_PhysicsSystem");
        return -1;
    }
    JPH_PhysicsSystem_SetGravity(self->system, &(JPH_Vec3){gx, gy, gz});
    self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);

    #if PY_VERSION_HEX < 0x030D0000
    self->shadow_lock = PyThread_allocate_lock();
    #endif

    // 4. Bake Scene (Python Call) if bodies exist
    if (bodies_list && bodies_list != Py_None) {
        PyObject *bake_func = PyObject_GetAttrString(st->helper, "bake_scene");
        if (!bake_func) return -1;

        PyObject *baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
        Py_DECREF(bake_func);
        if (!baked) return -1;

        // Unpack: (count, b_pos, b_rot, b_ext, b_mot)
        Py_ssize_t count_arg = 0;
        PyObject *o_pos, *o_rot, *o_ext, *o_mot;
        if (!PyArg_ParseTuple(baked, "nOOOO", &count_arg, &o_pos, &o_rot, &o_ext, &o_mot)) {
            Py_DECREF(baked);
            return -1;
        }

        size_t count = (size_t)count_arg;
        self->capacity = (count < 1024) ? 1024 : count;
        self->count = count;

        // Allocate Shadow Buffers
        self->positions = PyMem_RawMalloc(self->capacity * 4 * sizeof(float));
        self->rotations = PyMem_RawMalloc(self->capacity * 4 * sizeof(float));
        self->linear_velocities = PyMem_RawMalloc(self->capacity * 4 * sizeof(float));
        self->angular_velocities = PyMem_RawMalloc(self->capacity * 4 * sizeof(float));
        self->body_ids = PyMem_RawMalloc(self->capacity * sizeof(JPH_BodyID));

        if (!self->positions || !self->rotations || !self->body_ids) {
            Py_DECREF(baked);
            PyErr_NoMemory();
            return -1;
        }

        // Access Raw Bytes safely
        float *f_pos = (float*)PyBytes_AsString(o_pos);
        float *f_rot = (float*)PyBytes_AsString(o_rot);
        float *f_ext = (float*)PyBytes_AsString(o_ext);
        unsigned char *u_mot = (unsigned char*)PyBytes_AsString(o_mot);

        if (!f_pos || !f_rot || !f_ext || !u_mot) {
            Py_DECREF(baked);
            return -1;
        }

        // Create Bodies Loop
        JPH_BodyInterface* bi = self->body_interface;
        for (size_t i = 0; i < count; i++) {
            JPH_BoxShapeSettings* box = JPH_BoxShapeSettings_Create(
                &(JPH_Vec3){f_ext[i*4+0], f_ext[i*4+1], f_ext[i*4+2]}, 
                0.05f
            );
            JPH_Shape* shape = JPH_BoxShapeSettings_CreateShape(box);

            JPH_BodyCreationSettings* creation = JPH_BodyCreationSettings_Create3(
                shape,
                &(JPH_RVec3){f_pos[i*4+0], f_pos[i*4+1], f_pos[i*4+2]},
                &(JPH_Quat){f_rot[i*4+0], f_rot[i*4+1], f_rot[i*4+2], f_rot[i*4+3]},
                (JPH_MotionType)u_mot[i],
                0 // Layer
            );
            
            if (u_mot[i] == 2) { // Dynamic
                JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
            }

            JPH_BodyID bid = JPH_BodyInterface_CreateAndAddBody(bi, creation, JPH_Activation_Activate);
            // TODO: usually we would check bid. but i don't see a convenient helper in joltc.h to do so.
            self->body_ids[i] = bid;

            JPH_Shape_Destroy(shape);
            JPH_ShapeSettings_Destroy((JPH_ShapeSettings*)box);
            JPH_BodyCreationSettings_Destroy(creation);
        }

        // Initial Sync
        culverin_sync_shadow_buffers(self);
        
        Py_DECREF(baked); // Done with python buffers
    } else {
        // Empty init
        self->capacity = 1024;
        self->count = 0;
        self->positions = PyMem_RawMalloc(1024 * 4 * sizeof(float));
        self->rotations = PyMem_RawMalloc(1024 * 4 * sizeof(float));
        self->linear_velocities = PyMem_RawMalloc(1024 * 4 * sizeof(float));
        self->angular_velocities = PyMem_RawMalloc(1024 * 4 * sizeof(float));
        self->body_ids = PyMem_RawMalloc(1024 * sizeof(JPH_BodyID));
    }

    return 0;
}

// --- Methods & Getters (Standard) ---

static PyObject* PhysicsWorld_step(PhysicsWorldObject* self, PyObject* args) {
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
