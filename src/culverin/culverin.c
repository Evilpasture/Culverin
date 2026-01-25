#include "culverin.h"

#include <math.h>
#include <stddef.h>

// Character helpers
// Callback: Can the character collide with this object?
static bool JPH_API_CALL
char_on_contact_validate(void *userData, const JPH_CharacterVirtual *character,
                         JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
  return true; // Usually true, unless you want to walk through certain bodies
}

// Callback: Handle the collision settings AND Apply Impulse
static void JPH_API_CALL char_on_contact_added(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {
  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  CharacterObject *self = (CharacterObject *)userData;
  JPH_BodyInterface *bi = self->world->body_interface;

  if (JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
    return;
  }

  float vx = self->last_vx;
  float vy = self->last_vy;
  float vz = self->last_vz;

  // Normal points FROM Character TO Body.
  // If we are moving towards the body, Velocity and Normal align -> Dot is
  // Positive.
  float dot =
      vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

  // FIX: Check for Positive Dot (Moving towards object)
  if (dot > 0.01f) {

    // FIX: Positive factor
    float factor = dot * self->push_strength;

    JPH_Vec3 impulse;
    // FIX: Push ALONG the normal (Towards body)
    impulse.x = contactNormal->x * factor;

    // Flatten Y: Only allow upward pushes (lifting), ignore downward (friction)
    float y_push = contactNormal->y * factor;
    impulse.y = (y_push > 0.0f) ? y_push : 0.0f;

    impulse.z = contactNormal->z * factor;

    JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
    JPH_BodyInterface_ActivateBody(bi, bodyID2);
  }
}

// Map the procs
static const JPH_CharacterContactListener_Procs char_listener_procs = {
    .OnContactValidate = char_on_contact_validate,
    .OnContactAdded = char_on_contact_added,
    .OnAdjustBodyVelocity = NULL, // Use defaults
    .OnContactPersisted = char_on_contact_added,
    .OnContactRemoved = NULL,
    .OnCharacterContactValidate = NULL,
    .OnCharacterContactAdded = NULL,
    .OnCharacterContactPersisted = NULL,
    .OnCharacterContactRemoved = NULL,
    .OnContactSolve = NULL};

// --- Helper: Shape Caching (Internal) ---
static JPH_Shape *find_or_create_shape(PhysicsWorldObject *self, int type,
                                       const float *params) {
  // 1. Construct Key
  ShapeKey key = {(uint32_t)type, params[0], params[1], params[2], params[3]};

  // 2. Search Cache (Manual Compare)
  for (size_t i = 0; i < self->shape_cache_count; i++) {
    ShapeKey *k = &self->shape_cache[i].key;

    // Compare members directly to avoid structure padding issues
    // Note: Exact float equality is intended here. We only want a cache hit
    // if the requested size is bit-identical to the cached size.
    if (k->type == key.type && k->p1 == key.p1 && k->p2 == key.p2 &&
        k->p3 == key.p3 && k->p4 == key.p4) {

      return self->shape_cache[i].shape;
    }
  }

  // 3. Not Found -> Create New Jolt Shape
  JPH_Shape *shape = NULL;

  if (type == 0) { // BOX
    JPH_Vec3 he = {key.p1, key.p2, key.p3};
    JPH_BoxShapeSettings *s = JPH_BoxShapeSettings_Create(&he, 0.05f);
    shape = (JPH_Shape *)JPH_BoxShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 1) { // SPHERE
    JPH_SphereShapeSettings *s = JPH_SphereShapeSettings_Create(key.p1);
    shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 2) { // CAPSULE
    JPH_CapsuleShapeSettings *s =
        JPH_CapsuleShapeSettings_Create(key.p1, key.p2);
    shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 3) { // CYLINDER
    JPH_CylinderShapeSettings *s =
        JPH_CylinderShapeSettings_Create(key.p1, key.p2, 0.05f);
    shape = (JPH_Shape *)JPH_CylinderShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  } else if (type == 4) { // PLANE
    JPH_Plane p = {{key.p1, key.p2, key.p3}, key.p4};
    JPH_PlaneShapeSettings *s =
        JPH_PlaneShapeSettings_Create(&p, NULL, 1000.0f);
    shape = (JPH_Shape *)JPH_PlaneShapeSettings_CreateShape(s);
    JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)s);
  }

  if (!shape) {
    return NULL;
  }

  // 4. Store in Cache
  if (self->shape_cache_count >= self->shape_cache_capacity) {
    size_t new_cap =
        (self->shape_cache_capacity == 0) ? 16 : self->shape_cache_capacity * 2;
    void *new_ptr =
        PyMem_RawRealloc(self->shape_cache, new_cap * sizeof(ShapeEntry));
    if (!new_ptr) {
      JPH_Shape_Destroy(shape);
      PyErr_NoMemory();
      return NULL; // OOM
    }
    self->shape_cache = (ShapeEntry *)new_ptr;
    self->shape_cache_capacity = new_cap;
  }

  self->shape_cache[self->shape_cache_count].key = key;
  self->shape_cache[self->shape_cache_count].shape = shape;
  self->shape_cache_count++;

  return shape;
}

// --- Handle Helper ---
static inline BodyHandle make_handle(uint32_t slot, uint32_t gen) {
  return ((uint64_t)gen << 32) | (uint64_t)slot;
}

static inline bool unpack_handle(PhysicsWorldObject *self, BodyHandle h,
                                 uint32_t *slot) {
  *slot = (uint32_t)(h & 0xFFFFFFFF);
  uint32_t gen = (uint32_t)(h >> 32);

  if (*slot >= self->slot_capacity) {
    return false;
  }
  return self->generations[*slot] == gen;
}

// --- Lifecycle: Deallocation ---
static void PhysicsWorld_dealloc(PhysicsWorldObject *self) {
  if (self->system) {
    JPH_PhysicsSystem_Destroy(self->system);
  }
  if (self->shape_cache) {
    for (size_t i = 0; i < self->shape_cache_count; i++) {
      if (self->shape_cache[i].shape) {
        JPH_Shape_Destroy(self->shape_cache[i].shape);
      }
    }
    PyMem_RawFree(self->shape_cache);
  }
  if (self->job_system) {
    JPH_JobSystem_Destroy(self->job_system);
  }

  PyMem_RawFree(self->positions);
  PyMem_RawFree(self->rotations);
  PyMem_RawFree(self->linear_velocities);
  PyMem_RawFree(self->angular_velocities);
  PyMem_RawFree(self->body_ids);
  PyMem_RawFree(self->generations);
  PyMem_RawFree(self->slot_to_dense);
  PyMem_RawFree(self->dense_to_slot);
  PyMem_RawFree(self->free_slots);
  PyMem_RawFree(self->slot_states);
  PyMem_RawFree(self->command_queue);
  PyMem_RawFree(self->user_data);

#if PY_VERSION_HEX < 0x030D0000
  if (self->shadow_lock)
    PyThread_free_lock(self->shadow_lock);
#endif
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// --- Lifecycle: Initialization ---
static int PhysicsWorld_init(PhysicsWorldObject *self, PyObject *args,
                             PyObject *kwds) {
  static char *kwlist[] = {"settings", "bodies", NULL};
  PyObject *settings_dict = NULL;
  PyObject *bodies_list = NULL;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &settings_dict,
                                   &bodies_list)) {
    return -1;
  }

  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);

  PyObject *val_func = PyObject_GetAttrString(st->helper, "validate_settings");
  PyObject *norm_settings = PyObject_CallFunctionObjArgs(
      val_func, settings_dict ? settings_dict : Py_None, NULL);
  Py_DECREF(val_func);
  if (!norm_settings) {
    return -1;
  }

  float gx = NAN;
  float gy = NAN;
  float gz = NAN;
  float slop = NAN;
  int max_bodies = 0;
  int max_pairs = 0;
  PyArg_ParseTuple(norm_settings, "ffffii", &gx, &gy, &gz, &slop, &max_bodies,
                   &max_pairs);
  Py_DECREF(norm_settings);

  JobSystemThreadPoolConfig job_cfg = {
      .maxJobs = 1024, .maxBarriers = 8, .numThreads = -1};
  self->job_system = JPH_JobSystemThreadPool_Create(&job_cfg);
  self->bp_interface = JPH_BroadPhaseLayerInterfaceTable_Create(2, 2);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 0, 0);
  JPH_BroadPhaseLayerInterfaceTable_MapObjectToBroadPhaseLayer(
      self->bp_interface, 1, 1);
  self->pair_filter = JPH_ObjectLayerPairFilterTable_Create(2);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 0);
  JPH_ObjectLayerPairFilterTable_EnableCollision(self->pair_filter, 1, 1);
  self->bp_filter = JPH_ObjectVsBroadPhaseLayerFilterTable_Create(
      self->bp_interface, 2, self->pair_filter, 2);

  JPH_PhysicsSystemSettings phys_settings = {
      .maxBodies = 1000000,
      .maxBodyPairs = 1000000,
      .maxContactConstraints = 102400,
      .broadPhaseLayerInterface = self->bp_interface,
      .objectLayerPairFilter = self->pair_filter,
      .objectVsBroadPhaseLayerFilter = self->bp_filter};
  self->system = JPH_PhysicsSystem_Create(&phys_settings);
  JPH_Vec3 gravity = {gx, gy, gz};
  JPH_PhysicsSystem_SetGravity(self->system, &gravity);
  self->body_interface = JPH_PhysicsSystem_GetBodyInterface(self->system);

#if PY_VERSION_HEX < 0x030D0000
  self->shadow_lock = PyThread_allocate_lock();
#endif

  PyObject *baked = NULL;
  size_t baked_count = 0;
  if (bodies_list && bodies_list != Py_None) {
    PyObject *bake_func = PyObject_GetAttrString(st->helper, "bake_scene");
    baked = PyObject_CallFunctionObjArgs(bake_func, bodies_list, NULL);
    Py_DECREF(bake_func);
    if (!baked) {
      return -1;
    }
    baked_count = PyLong_AsSize_t(PyTuple_GetItem(baked, 0));
  }

  self->count = baked_count;
  self->capacity = (size_t)max_bodies;
  if (self->capacity < 128) {
    self->capacity = 128; // Sanity minimum
  }
  if (self->capacity < self->count) {
    self->capacity = self->count + 1024;
  }

  self->positions = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->rotations = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->linear_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->angular_velocities = PyMem_RawCalloc(self->capacity * 4, sizeof(float));
  self->body_ids = PyMem_RawMalloc(self->capacity * sizeof(JPH_BodyID));
  // Allocate User Data Buffer (Initialized to 0)
  self->user_data = PyMem_RawCalloc(self->capacity, sizeof(uint64_t));
  self->slot_capacity = self->capacity;
  self->generations = PyMem_RawCalloc(self->slot_capacity, sizeof(uint32_t));
  self->slot_to_dense = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->dense_to_slot = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->free_slots = PyMem_RawMalloc(self->slot_capacity * sizeof(uint32_t));
  self->slot_states = PyMem_RawCalloc(self->slot_capacity, sizeof(uint8_t));
  self->command_queue = PyMem_RawMalloc(64 * sizeof(PhysicsCommand));
  self->command_capacity = 64;
  self->command_count = 0;

  if (!self->positions || !self->rotations || !self->body_ids ||
      !self->linear_velocities || !self->angular_velocities ||
      !self->user_data || !self->generations || !self->slot_to_dense ||
      !self->dense_to_slot || !self->free_slots || !self->slot_states ||
      !self->command_queue) {

    Py_XDECREF(baked);
    PyErr_NoMemory();
    return -1;
  }

  if (baked) {
    float *f_pos = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 1));
    float *f_rot = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 2));
    float *f_shape = (float *)PyBytes_AsString(PyTuple_GetItem(baked, 3));
    unsigned char *u_mot =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 4));
    unsigned char *u_layer =
        (unsigned char *)PyBytes_AsString(PyTuple_GetItem(baked, 5));

    // NEW: Extract User Data (Entity IDs) from the new 7th element in the tuple
    uint64_t *u_data = (uint64_t *)PyBytes_AsString(PyTuple_GetItem(baked, 6));

    JPH_BodyInterface *bi = self->body_interface;

    for (size_t i = 0; i < self->count; i++) {
      JPH_STACK_ALLOC(JPH_RVec3, body_pos);
      body_pos->x = (double)f_pos[i * 4];
      body_pos->y = (double)f_pos[i * 4 + 1];
      body_pos->z = (double)f_pos[i * 4 + 2];

      JPH_STACK_ALLOC(JPH_Quat, body_rot);
      body_rot->x = f_rot[i * 4];
      body_rot->y = f_rot[i * 4 + 1];
      body_rot->z = f_rot[i * 4 + 2];
      body_rot->w = f_rot[i * 4 + 3];

      // Correct stride is 5: [type, p1, p2, p3, p4]
      float params[4] = {f_shape[i * 5 + 1], f_shape[i * 5 + 2],
                         f_shape[i * 5 + 3], f_shape[i * 5 + 4]};
      JPH_Shape *shape =
          find_or_create_shape(self, (int)f_shape[i * 5], params);

      if (shape) {
        JPH_BodyCreationSettings *creation = JPH_BodyCreationSettings_Create3(
            shape, body_pos, body_rot, (JPH_MotionType)u_mot[i],
            (JPH_ObjectLayer)u_layer[i]);

        // UserData in Jolt holds the SLOT index
        JPH_BodyCreationSettings_SetUserData(creation, (uint64_t)i);

        if (u_mot[i] == 2) {
          JPH_BodyCreationSettings_SetAllowSleeping(creation, true);
        }

        self->body_ids[i] = JPH_BodyInterface_CreateAndAddBody(
            bi, creation, JPH_Activation_Activate);
        JPH_BodyCreationSettings_Destroy(creation);

        // Init Mappings
        self->generations[i] = 1;
        self->slot_to_dense[i] = (uint32_t)i;
        self->dense_to_slot[i] = (uint32_t)i;
        self->slot_states[i] = SLOT_ALIVE;

        // NEW: Init the Parallel User Data Array (Entity ID)
        self->user_data[i] = u_data[i];
      } else {
        self->body_ids[i] = JPH_INVALID_BODY_ID;
      }
    }
    Py_DECREF(baked);
  }
  for (uint32_t i = (uint32_t)self->count; i < (uint32_t)self->slot_capacity;
       i++) {
    self->generations[i] = 1;
    self->free_slots[self->free_count++] = i;
  }
  culverin_sync_shadow_buffers(self);
  return 0;
}

static PyObject *PhysicsWorld_apply_impulse(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &h, &x, &y,
                                   &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
  JPH_Vec3 imp = {x, y, z};
  JPH_BodyInterface_AddImpulse(self->body_interface,
                               self->body_ids[self->slot_to_dense[slot]], &imp);
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// ABI BYPASS.
// WILL REPLACE WITH THE COMMENTED FUNCTION DEFINITION BELOW
// IF I FIND A WAY TO FIX ABI MISMATCH.
static PyObject *PhysicsWorld_raycast(PhysicsWorldObject *self, PyObject *args,
                                      PyObject *kwds) {
  float sx = NAN;
  float sy = NAN;
  float sz = NAN;
  float dx = NAN;
  float dy = NAN;
  float dz = NAN;
  float max_dist = 1000.0f;
  static char *kwlist[] = {"start", "direction", "max_dist", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|f", kwlist, &sx, &sy,
                                   &sz, &dx, &dy, &dz, &max_dist)) {
    return NULL;
  }

  // 1. Calculate Magnitude & Scaling
  float mag_sq = dx * dx + dy * dy + dz * dz;
  if (mag_sq < 1e-9f) {
    Py_RETURN_NONE; // Safety: Zero length ray
  }
  float mag = sqrtf(mag_sq);
  float scale = max_dist / mag;

  // 2. Stack Alloc with Alignment (CRITICAL for your build)
  JPH_STACK_ALLOC(JPH_RVec3, origin);
  origin->x = sx;
  origin->y = sy;
  origin->z = sz;
  // Safety against uninitialized padding if macro mismatch exists
  memset(origin, 0, sizeof(double) * 4);
  origin->x = sx;
  origin->y = sy;
  origin->z = sz;

  JPH_STACK_ALLOC(JPH_Vec3, direction);
  direction->x = dx * scale;
  direction->y = dy * scale;
  direction->z = dz * scale;

  JPH_STACK_ALLOC(JPH_RayCastResult, hit);
  memset(hit, 0, sizeof(JPH_RayCastResult));

  // 3. Execute
  const JPH_NarrowPhaseQuery *query =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  // Pass pointers to the aligned stack variables
  bool has_hit = JPH_NarrowPhaseQuery_CastRay(query, origin, direction, hit,
                                              NULL, NULL, NULL);

  if (!has_hit) {
    Py_RETURN_NONE;
  }

  // 4. Resolve Handle
  SHADOW_LOCK(&self->shadow_lock);

  // In Culverin, UserData IS the slot index
  uint64_t slot_idx =
      JPH_BodyInterface_GetUserData(self->body_interface, hit->bodyID);

  if (slot_idx >= self->slot_capacity) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
  }

  uint32_t gen = self->generations[slot_idx];
  BodyHandle handle = make_handle((uint32_t)slot_idx, gen);

  SHADOW_UNLOCK(&self->shadow_lock);

  return Py_BuildValue("Kf", handle, hit->fraction);
}

// ==============================================NO UNCOMMENT. AND LEAVE THIS
// ALONE.============================================================== static
// PyObject* PhysicsWorld_raycast(PhysicsWorldObject* self, PyObject* args,
// PyObject* kwds) {
//     float sx, sy, sz, dx, dy, dz;
//     float max_dist = 1000.0f;
//     static char *kwlist[] = {"start", "direction", "max_dist", NULL};

//     if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)|f", kwlist,
//                                      &sx, &sy, &sz, &dx, &dy, &dz,
//                                      &max_dist)) {
//         return NULL;
//     }

//     float mag = sqrtf(dx*dx + dy*dy + dz*dz);
//     if (mag < 1e-6f) {
//         Py_RETURN_NONE;
//     }

//     // --- FIX START ---
//     // Enforce alignment!
//     // JPH_Vec3 needs 16-byte alignment. JPH_RVec3 (Double) needs 32-byte (if
//     AVX2). #ifdef _MSC_VER
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

//     const JPH_NarrowPhaseQuery* query =
//     JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

//     // NarrowPhaseQuery is thread-safe for reads, so no locks needed here
//     // IF we assume Update isn't mutating the tree concurrently (which it
//     shouldn't be in your loop) bool has_hit =
//     JPH_NarrowPhaseQuery_CastRay(query, &ray_origin, &ray_dir, &hit, NULL,
//     NULL, NULL);

//     if (!has_hit) Py_RETURN_NONE;

//     uint64_t udata = JPH_BodyInterface_GetUserData(self->body_interface,
//     hit.bodyID); int hit_index = (int)udata;

//     if (hit_index < 0 || (size_t)hit_index >= self->count) {
//         Py_RETURN_NONE;
//     }

//     return Py_BuildValue("if", hit_index, hit.fraction);
// }
// ==============================================================================================================================================

// Helper to grow queue
static bool ensure_command_capacity(PhysicsWorldObject *self) {
  if (self->command_count >= self->command_capacity) {
    size_t new_cap = self->command_capacity * 2;
    void *new_ptr =
        PyMem_RawRealloc(self->command_queue, new_cap * sizeof(PhysicsCommand));
    if (!new_ptr) {
      return false;
    }
    self->command_queue = (PhysicsCommand *)new_ptr;
    self->command_capacity = new_cap;
  }
  return true;
}

static void flush_commands(PhysicsWorldObject *self) {
  if (self->command_count == 0) {
    return;
  }

  JPH_BodyInterface *bi = self->body_interface;

  for (size_t i = 0; i < self->command_count; i++) {
    PhysicsCommand *cmd = &self->command_queue[i];
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
      JPH_BodyCreationSettings *s = cmd->data.create.settings;

      // 1. Create in Jolt
      JPH_BodyID new_bid =
          JPH_BodyInterface_CreateAndAddBody(bi, s, JPH_Activation_Activate);

      // 2. Add to Dense Arrays
      size_t new_dense = self->count;
      self->body_ids[new_dense] = new_bid;
      self->slot_to_dense[slot] = (uint32_t)new_dense;
      self->dense_to_slot[new_dense] = slot;
      self->user_data[new_dense] = cmd->data.create.user_data;

      // 3. Force immediate Shadow Buffer sync for this new body
      JPH_STACK_ALLOC(JPH_RVec3, p);
      JPH_STACK_ALLOC(JPH_Quat, q);
      JPH_BodyInterface_GetPosition(bi, new_bid, p);
      JPH_BodyInterface_GetRotation(bi, new_bid, q);

      self->positions[new_dense * 4 + 0] = (float)p->x;
      self->positions[new_dense * 4 + 1] = (float)p->y;
      self->positions[new_dense * 4 + 2] = (float)p->z;
      self->positions[new_dense * 4 + 3] = 0.0f;

      self->rotations[new_dense * 4 + 0] = q->x;
      self->rotations[new_dense * 4 + 1] = q->y;
      self->rotations[new_dense * 4 + 2] = q->z;
      self->rotations[new_dense * 4 + 3] = q->w;

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
        memcpy(&self->positions[(size_t)dense_idx * 4],
               &self->positions[last_dense * 4], 16);
        memcpy(&self->rotations[(size_t)dense_idx * 4],
               &self->rotations[last_dense * 4], 16);
        memcpy(&self->linear_velocities[(size_t)dense_idx * 4],
               &self->linear_velocities[last_dense * 4], 16);
        memcpy(&self->angular_velocities[(size_t)dense_idx * 4],
               &self->angular_velocities[last_dense * 4], 16);
        self->body_ids[dense_idx] = self->body_ids[last_dense];

        // Update Map for the body that moved
        uint32_t mover_slot = self->dense_to_slot[last_dense];
        self->slot_to_dense[mover_slot] = dense_idx;
        self->dense_to_slot[dense_idx] = mover_slot;
        self->user_data[dense_idx] = self->user_data[last_dense];
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
      p->x = cmd->data.vec.x;
      p->y = cmd->data.vec.y;
      p->z = cmd->data.vec.z;
      JPH_BodyInterface_SetPosition(bi, bid, p, JPH_Activation_Activate);

      // Sync Shadow
      self->positions[dense_idx * 4 + 0] = cmd->data.vec.x;
      self->positions[dense_idx * 4 + 1] = cmd->data.vec.y;
      self->positions[dense_idx * 4 + 2] = cmd->data.vec.z;
      break;
    }

    case CMD_SET_ROT: {
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->data.vec.x;
      q->y = cmd->data.vec.y;
      q->z = cmd->data.vec.z;
      q->w = cmd->data.vec.w;
      JPH_BodyInterface_SetRotation(bi, bid, q, JPH_Activation_Activate);

      // Sync Shadow
      self->rotations[dense_idx * 4 + 0] = q->x;
      self->rotations[dense_idx * 4 + 1] = q->y;
      self->rotations[dense_idx * 4 + 2] = q->z;
      self->rotations[dense_idx * 4 + 3] = q->w;
      break;
    }

    case CMD_SET_TRNS: {
      JPH_STACK_ALLOC(JPH_RVec3, p);
      p->x = cmd->data.transform.px;
      p->y = cmd->data.transform.py;
      p->z = cmd->data.transform.pz;
      JPH_STACK_ALLOC(JPH_Quat, q);
      q->x = cmd->data.transform.rx;
      q->y = cmd->data.transform.ry;
      q->z = cmd->data.transform.rz;
      q->w = cmd->data.transform.rw;

      JPH_BodyInterface_SetPositionAndRotation(bi, bid, p, q,
                                               JPH_Activation_Activate);

      self->positions[dense_idx * 4 + 0] = (float)p->x;
      self->positions[dense_idx * 4 + 1] = (float)p->y;
      self->positions[dense_idx * 4 + 2] = (float)p->z;
      self->rotations[dense_idx * 4 + 0] = q->x;
      self->rotations[dense_idx * 4 + 1] = q->y;
      self->rotations[dense_idx * 4 + 2] = q->z;
      self->rotations[dense_idx * 4 + 3] = q->w;
      break;
    }

    case CMD_SET_LINVEL: {
      JPH_Vec3 v = {cmd->data.vec.x, cmd->data.vec.y, cmd->data.vec.z};
      JPH_BodyInterface_SetLinearVelocity(bi, bid, &v);

      self->linear_velocities[dense_idx * 4 + 0] = v.x;
      self->linear_velocities[dense_idx * 4 + 1] = v.y;
      self->linear_velocities[dense_idx * 4 + 2] = v.z;
      break;
    }

    case CMD_SET_ANGVEL: {
      JPH_Vec3 v = {cmd->data.vec.x, cmd->data.vec.y, cmd->data.vec.z};
      JPH_BodyInterface_SetAngularVelocity(bi, bid, &v);

      self->angular_velocities[dense_idx * 4 + 0] = v.x;
      self->angular_velocities[dense_idx * 4 + 1] = v.y;
      self->angular_velocities[dense_idx * 4 + 2] = v.z;
      break;
    }

    case CMD_SET_MOTION: {
      JPH_BodyInterface_SetMotionType(bi, bid,
                                      (JPH_MotionType)cmd->data.motion_type,
                                      JPH_Activation_Activate);
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
    case CMD_SET_USER_DATA: {
      uint32_t dense_idx = self->slot_to_dense[slot];
      self->user_data[dense_idx] = cmd->data.user_data_val;
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

static PyObject *PhysicsWorld_save_state(PhysicsWorldObject *self,
                                         PyObject *Py_UNUSED(unused)) {
  SHADOW_LOCK(&self->shadow_lock);

  // Header: count, capacity, slot_capacity, time
  size_t meta_size = sizeof(size_t) * 3 + sizeof(double);
  size_t dense_size = self->count * 4 * sizeof(float) * 4; // pos, rot, lin, ang
  size_t mapping_size =
      self->slot_capacity * (sizeof(uint32_t) * 4 + sizeof(uint8_t));

  size_t total_size = meta_size + dense_size + mapping_size;

  PyObject *bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)total_size);
  char *ptr = PyBytes_AsString(bytes);

  // 1. Meta
  memcpy(ptr, &self->count, sizeof(size_t));
  ptr += sizeof(size_t);
  memcpy(ptr, &self->time, sizeof(double));
  ptr += sizeof(double);

  // 2. Dense Buffers
  memcpy(ptr, self->positions, self->count * 16);
  ptr += self->count * 16;
  memcpy(ptr, self->rotations, self->count * 16);
  ptr += self->count * 16;
  memcpy(ptr, self->linear_velocities, self->count * 16);
  ptr += self->count * 16;
  memcpy(ptr, self->angular_velocities, self->count * 16);
  ptr += self->count * 16;

  // 3. Mapping State (Handles/Slots)
  memcpy(ptr, self->generations, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(ptr, self->slot_to_dense, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(ptr, self->dense_to_slot, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(ptr, self->slot_states, self->slot_capacity);

  SHADOW_UNLOCK(&self->shadow_lock);
  return bytes;
}

static PyObject *PhysicsWorld_load_state(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  Py_buffer view;
  static char *kwlist[] = {"state", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "y*", kwlist, &view)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  char *ptr = (char *)view.buf;
  size_t saved_count = 0;

  // 1. Basic Header Check
  if (view.len < (Py_ssize_t)sizeof(size_t)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError, "Snapshot too small (header missing)");
    return NULL;
  }

  memcpy(&saved_count, ptr, sizeof(size_t));
  ptr += sizeof(size_t);

  if (saved_count > self->capacity) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError,
                    "Snapshot body count exceeds world capacity");
    return NULL;
  }

  // 2. CRITICAL: Calculate Required Size before reading
  // Header (remaining): time(double)
  // Arrays: 4 arrays * count * 4 floats * sizeof(float)
  // Mappings: 3 arrays * slot_cap * 4 bytes + 1 array * slot_cap * 1 byte
  size_t required_size = sizeof(size_t) + sizeof(double); // Meta
  required_size += saved_count * 16 * 4 * sizeof(float);  // Dense Buffers
  required_size += self->slot_capacity *
                   (3 * sizeof(uint32_t) + 1 * sizeof(uint8_t)); // Mappings

  if ((size_t)view.len < required_size) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&view);
    PyErr_SetString(PyExc_ValueError,
                    "Snapshot buffer is corrupted or truncated");
    return NULL;
  }

  // 3. Restore Meta
  self->count = saved_count;
  self->view_shape[0] = (Py_ssize_t)self->count;

  memcpy(&self->time, ptr, sizeof(double));
  ptr += sizeof(double);

  // 4. Restore Dense Buffers
  memcpy(self->positions, ptr, self->count * 16);
  ptr += self->count * 16;
  memcpy(self->rotations, ptr, self->count * 16);
  ptr += self->count * 16;
  memcpy(self->linear_velocities, ptr, self->count * 16);
  ptr += self->count * 16;
  memcpy(self->angular_velocities, ptr, self->count * 16);
  ptr += self->count * 16;

  // 5. Restore Mappings
  memcpy(self->generations, ptr, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(self->slot_to_dense, ptr, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(self->dense_to_slot, ptr, self->slot_capacity * 4);
  ptr += self->slot_capacity * 4;
  memcpy(self->slot_states, ptr, self->slot_capacity);

  // 6. Re-Sync Jolt
  for (size_t i = 0; i < self->count; i++) {
    JPH_BodyID bid = self->body_ids[i];
    if (bid == JPH_INVALID_BODY_ID) {
      continue;
    }

    JPH_STACK_ALLOC(JPH_RVec3, p);
    p->x = (double)self->positions[i * 4];
    p->y = (double)self->positions[i * 4 + 1];
    p->z = (double)self->positions[i * 4 + 2];

    JPH_STACK_ALLOC(JPH_Quat, q);
    q->x = self->rotations[i * 4];
    q->y = self->rotations[i * 4 + 1];
    q->z = self->rotations[i * 4 + 2];
    q->w = self->rotations[i * 4 + 3];

    JPH_BodyInterface_SetPositionAndRotation(self->body_interface, bid, p, q,
                                             JPH_Activation_Activate);

    JPH_Vec3 lv = {self->linear_velocities[i * 4],
                   self->linear_velocities[i * 4 + 1],
                   self->linear_velocities[i * 4 + 2]};
    JPH_BodyInterface_SetLinearVelocity(self->body_interface, bid, &lv);

    JPH_Vec3 av = {self->angular_velocities[i * 4],
                   self->angular_velocities[i * 4 + 1],
                   self->angular_velocities[i * 4 + 2]};
    JPH_BodyInterface_SetAngularVelocity(self->body_interface, bid, &av);
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  PyBuffer_Release(&view);
  Py_RETURN_NONE;
}
// --- Methods & Getters (Standard) ---

static PyObject *PhysicsWorld_step(PhysicsWorldObject *self, PyObject *args) {
  float dt = 1.0f / 60.0f;
  if (!PyArg_ParseTuple(args, "|f", &dt)) {
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS

      SHADOW_LOCK(&self->shadow_lock);
  flush_commands(self); // <--- FLUSH BEFORE STEP
  SHADOW_UNLOCK(&self->shadow_lock);

  JPH_PhysicsSystem_Update(self->system, dt, 1, self->job_system);

  SHADOW_LOCK(&self->shadow_lock);
  culverin_sync_shadow_buffers(self);
  SHADOW_UNLOCK(&self->shadow_lock);

  Py_END_ALLOW_THREADS self->time += (double)dt;

  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_create_character(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  float px = 0;
  float py = 0;
  float pz = 0;
  float height = 1.8f;
  float radius = 0.4f;
  float step_height = 0.4f;
  float max_slope = 45.0f;
  static char *kwlist[] = {"pos",         "height",    "radius",
                           "step_height", "max_slope", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)|ffff", kwlist, &px, &py,
                                   &pz, &height, &radius, &step_height,
                                   &max_slope)) {
    return NULL;
  }

  // 1. Create Shape (Capsule)
  float half_h = (height - 2.0f * radius) * 0.5f;
  if (half_h < 0.1f) {
    half_h = 0.1f;
  }

  JPH_CapsuleShapeSettings *ss =
      JPH_CapsuleShapeSettings_Create(half_h, radius);
  JPH_Shape *shape = (JPH_Shape *)JPH_CapsuleShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);

  if (!shape) {
    return PyErr_NoMemory();
  }

  // 2. Settings (Stack Allocation + Direct Access)
  JPH_CharacterVirtualSettings settings;
  memset(&settings, 0, sizeof(JPH_CharacterVirtualSettings));
  JPH_CharacterVirtualSettings_Init(&settings); // Important: Sets defaults

  // Direct Member Access (No _Set functions)
  settings.base.shape = shape;

  // Up Vector (Y-Up)
  settings.base.up.x = 0;
  settings.base.up.y = 1;
  settings.base.up.z = 0;

  // Infinite Plane Supporting Volume (Normal -Y, Distance huge)
  settings.base.supportingVolume.normal.x = 0;
  settings.base.supportingVolume.normal.y = 1;
  settings.base.supportingVolume.normal.z = 0;
  settings.base.supportingVolume.distance = -1.0e10f;

  float slope_rad = max_slope * (3.14159f / 180.0f);
  settings.base.maxSlopeAngle = slope_rad;
  settings.base.enhancedInternalEdgeRemoval = true;

  // Config
  settings.mass = 70.0f;
  settings.maxStrength = 500.0f;
  settings.characterPadding = 0.02f;
  settings.penetrationRecoverySpeed = 1.0f;
  settings.predictiveContactDistance = 0.1f;
  settings.maxCollisionIterations = 5;
  settings.maxConstraintIterations = 15;
  settings.minTimeRemaining = 0.0001f;
  settings.collisionTolerance = 0.001f;

  // 3. Create Jolt Character
  JPH_RVec3 pos = {(double)px, (double)py, (double)pz};
  JPH_Quat rot = {0, 0, 0, 1};

  // Pass address of stack settings
  JPH_CharacterVirtual *j_char =
      JPH_CharacterVirtual_Create(&settings, &pos, &rot, 0, self->system);

  // Jolt adds a ref to the shape, so we can release our pointer
  JPH_Shape_Destroy(shape);

  if (!j_char) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to create CharacterVirtual");
    return NULL;
  }

  // 4. Wrap in Python Object
  PyObject *module = PyType_GetModule(Py_TYPE(self));
  CulverinState *st = get_culverin_state(module);

  // Use the cached type from module state
  PyObject *char_type = st->CharacterType;
  if (!char_type) {
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)j_char);
    PyErr_SetString(PyExc_RuntimeError, "Character type not initialized");
    return NULL;
  }

  CharacterObject *obj = (CharacterObject *)PyObject_New(
      CharacterObject, (PyTypeObject *)char_type);
  // Note: PyObject_New does NOT consume a reference to char_type, so don't
  // DECREF it.

  obj->character = j_char;
  obj->world = self;
  obj->push_strength = 25.0f;
  obj->last_vx = 0;
  obj->last_vy = 0;
  obj->last_vz = 0;
  Py_INCREF(self);

  JPH_CharacterContactListener_SetProcs(&char_listener_procs);
  obj->listener =
      JPH_CharacterContactListener_Create(obj); // Pass 'obj' as userData
  JPH_CharacterVirtual_SetListener(j_char, obj->listener);

  obj->body_filter = JPH_BodyFilter_Create(NULL);
  obj->shape_filter = JPH_ShapeFilter_Create(NULL);
  obj->bp_filter = JPH_BroadPhaseLayerFilter_Create(NULL);
  obj->obj_filter = JPH_ObjectLayerFilter_Create(NULL);

  return (PyObject *)obj;
}

static int PhysicsWorld_resize(PhysicsWorldObject *self, size_t new_capacity) {
  if (new_capacity <= self->capacity) {
    return 0;
  }

  // Reallocate Shadow Buffers (SoA)
  self->positions = PyMem_RawRealloc(self->positions, new_capacity * 16);
  self->rotations = PyMem_RawRealloc(self->rotations, new_capacity * 16);
  self->linear_velocities =
      PyMem_RawRealloc(self->linear_velocities, new_capacity * 16);
  self->angular_velocities =
      PyMem_RawRealloc(self->angular_velocities, new_capacity * 16);
  self->body_ids =
      PyMem_RawRealloc(self->body_ids, new_capacity * sizeof(JPH_BodyID));
  self->user_data =
      PyMem_RawRealloc(self->user_data, new_capacity * sizeof(uint64_t));

  // Reallocate Mapping Arrays
  self->generations =
      PyMem_RawRealloc(self->generations, new_capacity * sizeof(uint32_t));
  self->slot_to_dense =
      PyMem_RawRealloc(self->slot_to_dense, new_capacity * sizeof(uint32_t));
  self->dense_to_slot =
      PyMem_RawRealloc(self->dense_to_slot, new_capacity * sizeof(uint32_t));
  self->slot_states =
      PyMem_RawRealloc(self->slot_states, new_capacity * sizeof(uint8_t));

  // The Free Slot stack needs to be expanded
  uint32_t *new_free_slots =
      PyMem_RawRealloc(self->free_slots, new_capacity * sizeof(uint32_t));

  if (!self->positions || !self->rotations || !new_free_slots) {
    PyErr_NoMemory();
    return -1;
  }
  self->free_slots = new_free_slots;

  // Initialize the new slots as free
  for (size_t i = self->slot_capacity; i < new_capacity; i++) {
    self->generations[i] = 1;
    self->slot_states[i] = SLOT_EMPTY;
    self->free_slots[self->free_count++] = (uint32_t)i;
  }

  self->capacity = new_capacity;
  self->slot_capacity = new_capacity;
  return 0;
}

static PyObject *PhysicsWorld_create_body(PhysicsWorldObject *self,
                                          PyObject *args, PyObject *kwds) {
  // Default values
  float px = 0.0f;
  float py = 0.0f;
  float pz = 0.0f;
  float rx = 0.0f;
  float ry = 0.0f;
  float rz = 0.0f;
  float rw = 1.0f;
  float s[4] = {1.0f, 1.0f, 1.0f, 0.0f}; // Default size params
  int shape_type = 0;                    // SHAPE_BOX
  int motion_type = 2;                   // MOTION_DYNAMIC
  unsigned long long user_data = 0;

  PyObject *py_size = NULL;
  static char *kwlist[] = {"pos",    "rot",       "size", "shape",
                           "motion", "user_data", NULL};

  // 1. Parse Arguments
  // Note: size is parsed as a generic Object 'O' for flexible tuple length
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|(fff)(ffff)OiiK", kwlist, &px,
                                   &py, &pz, &rx, &ry, &rz, &rw, &py_size,
                                   &shape_type, &motion_type, &user_data)) {
    return NULL;
  }

  // 2. Extract Size Params from Tuple
  if (py_size && PyTuple_Check(py_size)) {
    Py_ssize_t sz_len = PyTuple_Size(py_size);
    for (Py_ssize_t i = 0; i < sz_len && i < 4; i++) {
      PyObject *item = PyTuple_GetItem(py_size, i);
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
  // if (self->count + self->command_count >= self->capacity) {
  //     SHADOW_UNLOCK(&self->shadow_lock);
  //     PyErr_SetString(PyExc_MemoryError, "World capacity reached. Increase
  //     max_bodies in settings."); return NULL;
  // }

  // if (self->free_count == 0) {
  //     SHADOW_UNLOCK(&self->shadow_lock);
  //     PyErr_SetString(PyExc_MemoryError, "No free slots available.");
  //     return NULL;
  // }

  // If out of slots, auto-resize (Double the capacity)
  if (self->free_count == 0) {
    if (PhysicsWorld_resize(self, self->capacity * 2) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return NULL;
    }
  }

  // 5. Reserve Slot
  uint32_t slot = self->free_slots[--self->free_count];

  // 6. Find/Create Jolt Shape
  JPH_Shape *shape = find_or_create_shape(self, shape_type, s);
  if (!shape) {
    self->free_slots[self->free_count++] = slot;
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to create or retrieve physics shape.");
    return NULL;
  }

  // 7. Prepare Jolt Creation Settings
  // Use ABI alignment hack for stack vectors
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = (double)px;
  pos->y = (double)py;
  pos->z = (double)pz;

  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx;
  rot->y = ry;
  rot->z = rz;
  rot->w = rw;

  // Layer selection: 0 for Static, 1 for Moving
  uint32_t layer = (motion_type == 0) ? 0 : 1;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, (JPH_MotionType)motion_type, (JPH_ObjectLayer)layer);

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

  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->type = CMD_CREATE_BODY;
  cmd->slot = slot;
  cmd->data.create.settings = settings; // Command now owns this pointer
  cmd->data.create.user_data = (uint64_t)user_data;

  // 9. Update State Machine
  self->slot_states[slot] = SLOT_PENDING_CREATE;

  // 10. Generate and Return Handle
  uint32_t gen = self->generations[slot];
  BodyHandle handle = make_handle(slot, gen);

  SHADOW_UNLOCK(&self->shadow_lock);

  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject *PhysicsWorld_create_mesh_body(PhysicsWorldObject *self,
                                               PyObject *args, PyObject *kwds) {
  float px = NAN;
  float py = NAN;
  float pz = NAN;
  float rx = NAN;
  float ry = NAN;
  float rz = NAN;
  float rw = NAN;
  Py_buffer v_view;
  Py_buffer i_view;
  unsigned long long user_data = 0;
  static char *kwlist[] = {"pos",     "rot",       "vertices",
                           "indices", "user_data", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(ffff)y*y*|K", kwlist, &px,
                                   &py, &pz, &rx, &ry, &rz, &rw, &v_view,
                                   &i_view, &user_data)) {
    return NULL;
  }

  // 1. Calculate counts
  uint32_t vertex_count = (uint32_t)(v_view.len / (3 * sizeof(float)));
  uint32_t index_count = (uint32_t)(i_view.len / sizeof(uint32_t));
  uint32_t tri_count = index_count / 3;

  // 2. INFLATE INDICES
  // We create a temporary buffer of Jolt-sized structs
  JPH_IndexedTriangle *jolt_tris = (JPH_IndexedTriangle *)PyMem_RawMalloc(
      tri_count * sizeof(JPH_IndexedTriangle));
  if (!jolt_tris) {
    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    return PyErr_NoMemory();
  }

  uint32_t *raw_indices = (uint32_t *)i_view.buf;
  for (uint32_t t = 0; t < tri_count; t++) {
    jolt_tris[t].i1 = raw_indices[t * 3 + 0];
    jolt_tris[t].i2 = raw_indices[t * 3 + 1];
    jolt_tris[t].i3 = raw_indices[t * 3 + 2];
    jolt_tris[t].materialIndex = 0; // Default
    jolt_tris[t].userData = 0;      // Default
  }

  // 3. Create Jolt Mesh Shape
  // JPH_Vec3 is simple (3 floats), so we can pass v_view.buf directly
  JPH_MeshShapeSettings *mss = JPH_MeshShapeSettings_Create2(
      (JPH_Vec3 *)v_view.buf, vertex_count, jolt_tris, tri_count);

  if (!mss) {
    PyMem_RawFree(jolt_tris);
    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    PyErr_SetString(PyExc_RuntimeError,
                    "Jolt failed to create MeshShapeSettings");
    return NULL;
  }

  JPH_Shape *shape = (JPH_Shape *)JPH_MeshShapeSettings_CreateShape(mss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)mss);

  // We can free our inflated index buffer NOW because CreateShape
  // builds an internal BVH and no longer needs the original pointers.
  PyMem_RawFree(jolt_tris);

  if (!shape) {
    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    PyErr_SetString(PyExc_RuntimeError, "Jolt failed to build Mesh BVH");
    return NULL;
  }

  // 4. Reserve Slot and Queue Command (Normal Boilerplate)
  SHADOW_LOCK(&self->shadow_lock);
  if (self->free_count == 0) {
    // Double the capacity or grow by a fixed chunk
    if (PhysicsWorld_resize(self, self->capacity + 1024) < 0) {
      SHADOW_UNLOCK(&self->shadow_lock);
      JPH_Shape_Destroy(shape); // Cleanup local ref!
      PyBuffer_Release(&v_view);
      PyBuffer_Release(&i_view);
      return NULL; // PhysicsWorld_resize sets the PyErr_NoMemory
    }
  }

  // if (self->count + self->command_count >= self->capacity || self->free_count
  // == 0) {
  //     SHADOW_UNLOCK(&self->shadow_lock);
  //     JPH_Shape_Destroy(shape);
  //     PyBuffer_Release(&v_view); PyBuffer_Release(&i_view);
  //     PyErr_SetString(PyExc_MemoryError, "World capacity reached");
  //     return NULL;
  // }

  uint32_t slot = self->free_slots[--self->free_count];

  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = px;
  pos->y = py;
  pos->z = pz;
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = rx;
  rot->y = ry;
  rot->z = rz;
  rot->w = rw;

  JPH_BodyCreationSettings *settings = JPH_BodyCreationSettings_Create3(
      shape, pos, rot, JPH_MotionType_Static, 0 // Mesh usually Static, Layer 0
  );
  JPH_Shape_Destroy(shape);
  if (!settings) {
    // Handle error if settings creation failed (unlikely)
    SHADOW_UNLOCK(&self->shadow_lock);
    PyBuffer_Release(&v_view);
    PyBuffer_Release(&i_view);
    return PyErr_NoMemory();
  }
  JPH_BodyCreationSettings_SetUserData(settings, (uint64_t)slot);

  ensure_command_capacity(self);
  PhysicsCommand *cmd = &self->command_queue[self->command_count++];
  cmd->type = CMD_CREATE_BODY;
  cmd->slot = slot;
  cmd->data.create.settings = settings;
  cmd->data.create.user_data = (uint64_t)user_data;

  self->slot_states[slot] = SLOT_PENDING_CREATE;
  BodyHandle handle = make_handle(slot, self->generations[slot]);

  SHADOW_UNLOCK(&self->shadow_lock);

  PyBuffer_Release(&v_view);
  PyBuffer_Release(&i_view);

  return PyLong_FromUnsignedLongLong(handle);
}

static PyObject *PhysicsWorld_destroy_body(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  // 1. Parse Arguments (Now with Keywords)
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);

  // 2. Validate Handle
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  // 3. Verify State
  // We only destroy if it is ALIVE or was just queued for creation
  // (PENDING_CREATE)
  if (self->slot_states[slot] == SLOT_ALIVE ||
      self->slot_states[slot] == SLOT_PENDING_CREATE) {

    // 4. Queue Command
    if (!ensure_command_capacity(self)) {
      SHADOW_UNLOCK(&self->shadow_lock);
      return PyErr_NoMemory();
    }

    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->type = CMD_DESTROY_BODY;
    cmd->slot = slot;

    // Mark as pending so logic knows it's "dead" even before the next step()
    self->slot_states[slot] = SLOT_PENDING_DESTROY;
  }

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_position(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }
  uint32_t idx = self->slot_to_dense[slot];
  JPH_STACK_ALLOC(JPH_RVec3, p);
  p->x = x;
  p->y = y;
  p->z = z;
  JPH_BodyInterface_SetPosition(self->body_interface, self->body_ids[idx], p,
                                JPH_Activation_Activate);
  self->positions[idx * 4 + 0] = x;
  self->positions[idx * 4 + 1] = y;
  self->positions[idx * 4 + 2] = z;
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_rotation(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  float w = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", "w", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kffff", kwlist, &handle_raw, &x,
                                   &y, &z, &w)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock); // LOCK

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  uint32_t dense_idx = self->slot_to_dense[slot];
  JPH_BodyID bid = self->body_ids[dense_idx];

  JPH_STACK_ALLOC(JPH_Quat, q);
  q->x = x;
  q->y = y;
  q->z = z;
  q->w = w;
  JPH_BodyInterface_SetRotation(self->body_interface, bid, q,
                                JPH_Activation_Activate);

  self->rotations[dense_idx * 4 + 0] = x;
  self->rotations[dense_idx * 4 + 1] = y;
  self->rotations[dense_idx * 4 + 2] = z;
  self->rotations[dense_idx * 4 + 3] = w;

  SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_linear_velocity(PhysicsWorldObject *self,
                                                  PyObject *args,
                                                  PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock); // LOCK

  uint32_t slot = 0;
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

static PyObject *PhysicsWorld_set_angular_velocity(PhysicsWorldObject *self,
                                                   PyObject *args,
                                                   PyObject *kwds) {
  uint64_t handle_raw = 0;
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"handle", "x", "y", "z", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Kfff", kwlist, &handle_raw, &x,
                                   &y, &z)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock); // LOCK

  uint32_t slot = 0;
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

static PyObject *PhysicsWorld_get_motion_type(PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock); // LOCK

  uint32_t slot = 0;
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

static PyObject *PhysicsWorld_set_motion_type(PhysicsWorldObject *self,
                                              PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  int motion_type = 0;
  static char *kwlist[] = {"handle", "motion", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Ki", kwlist, &handle_raw,
                                   &motion_type)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock); // LOCK

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyID bid = self->body_ids[self->slot_to_dense[slot]];
  JPH_BodyInterface_SetMotionType(self->body_interface, bid,
                                  (JPH_MotionType)motion_type,
                                  JPH_Activation_Activate);

  SHADOW_UNLOCK(&self->shadow_lock); // UNLOCK
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_user_data(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  unsigned long long data = 0;
  static char *kwlist[] = {"handle", "data", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "KK", kwlist, &handle_raw,
                                   &data)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid handle");
    return NULL;
  }

  // Immediate update (Safe because we hold lock)
  uint32_t dense = self->slot_to_dense[slot];
  self->user_data[dense] = (uint64_t)data;

  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_get_user_data(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
  }

  uint64_t val = self->user_data[self->slot_to_dense[slot]];
  SHADOW_UNLOCK(&self->shadow_lock);

  return PyLong_FromUnsignedLongLong(val);
}

static PyObject *PhysicsWorld_activate(PhysicsWorldObject *self, PyObject *args,
                                       PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyInterface_ActivateBody(self->body_interface,
                                 self->body_ids[self->slot_to_dense[slot]]);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_deactivate(PhysicsWorldObject *self,
                                         PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  JPH_BodyInterface_DeactivateBody(self->body_interface,
                                   self->body_ids[self->slot_to_dense[slot]]);
  Py_RETURN_NONE;
}

static PyObject *PhysicsWorld_set_transform(PhysicsWorldObject *self,
                                            PyObject *args, PyObject *kwds) {
  uint64_t handle_raw = 0;
  float px = NAN;
  float py = NAN;
  float pz = NAN;
  float rx = NAN;
  float ry = NAN;
  float rz = NAN;
  float rw = NAN;
  static char *kwlist[] = {"handle", "pos", "rot", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K(fff)(ffff)", kwlist,
                                   &handle_raw, &px, &py, &pz, &rx, &ry, &rz,
                                   &rw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    SHADOW_UNLOCK(&self->shadow_lock);
    PyErr_SetString(PyExc_ValueError, "Invalid or stale handle");
    return NULL;
  }

  if (ensure_command_capacity(self)) {
    PhysicsCommand *cmd = &self->command_queue[self->command_count++];
    cmd->type = CMD_SET_TRNS;
    cmd->slot = slot;
    cmd->data.transform.px = px;
    cmd->data.transform.py = py;
    cmd->data.transform.pz = pz;
    cmd->data.transform.rx = rx;
    cmd->data.transform.ry = ry;
    cmd->data.transform.rz = rz;
    cmd->data.transform.rw = rw;
  }
  SHADOW_UNLOCK(&self->shadow_lock);
  Py_RETURN_NONE;
}

// Helper to deduce handle and append to list
static void append_hit(QueryContext *ctx, JPH_BodyID bid) {
  // 1. Get Slot from Jolt UserData
  uint64_t slot =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, bid);

  // 2. Bound Check
  if (slot >= ctx->world->slot_capacity) {
    return;
  }

  // 3. Validate Generation
  // We only care about ALIVE bodies. If generation mismatches, it's a stale ID
  // (rare in queries but possible)
  uint32_t gen = ctx->world->generations[slot];

  // 4. Create Handle
  BodyHandle h = make_handle((uint32_t)slot, gen);

  PyObject *py_h = PyLong_FromUnsignedLongLong(h);
  PyList_Append(ctx->result_list, py_h);
  Py_DECREF(py_h);
}

// JoltC Callback for CollideShape (Narrowphase)
// Signature: void (*)(void *, const JPH_CollideShapeResult *)
// Note: JoltC uses a void return for collecting hits in some versions,
// or a float return for casts. Checking joltc.h...
// typedef float JPH_CollideShapeCollectorCallback(void* context, const
// JPH_CollideShapeResult* result);
static float OverlapCallback_Narrow(void *context,
                                    const JPH_CollideShapeResult *result) {
  OverlapContext *ctx = (OverlapContext *)context;

  // Result contains bodyID2 (the body we hit)
  uint64_t slot = JPH_BodyInterface_GetUserData(ctx->world->body_interface,
                                                result->bodyID2);

  if (slot < ctx->world->slot_capacity) {
    uint32_t gen = ctx->world->generations[slot];
    BodyHandle h = make_handle((uint32_t)slot, gen);

    PyObject *py_h = PyLong_FromUnsignedLongLong(h);
    PyList_Append(ctx->result_list, py_h);
    Py_DECREF(py_h);
  }

  // Return fraction to continue query (usually 1.0 for "find all")
  // If we wanted to stop at first hit, we'd return 0.0.
  return 1.0f;
}

// JoltC Callback for BroadPhase
// typedef float JPH_CollideShapeBodyCollectorCallback(void* context, const
// JPH_BodyID result);
static float OverlapCallback_Broad(void *context, const JPH_BodyID result) {
  OverlapContext *ctx = (OverlapContext *)context;

  uint64_t slot =
      JPH_BodyInterface_GetUserData(ctx->world->body_interface, result);

  if (slot < ctx->world->slot_capacity) {
    uint32_t gen = ctx->world->generations[slot];
    BodyHandle h = make_handle((uint32_t)slot, gen);

    PyObject *py_h = PyLong_FromUnsignedLongLong(h);
    PyList_Append(ctx->result_list, py_h);
    Py_DECREF(py_h);
  }
  return 1.0f;
}

static PyObject *PhysicsWorld_overlap_sphere(PhysicsWorldObject *self,
                                             PyObject *args, PyObject *kwds) {
  float x = NAN;
  float y = NAN;
  float z = NAN;
  float radius = NAN;
  static char *kwlist[] = {"center", "radius", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &x, &y, &z,
                                   &radius)) {
    return NULL;
  }

  // 1. Create Temporary Jolt Shape
  // This requires heap allocation by Jolt, but we destroy it immediately after.
  JPH_SphereShapeSettings *ss = JPH_SphereShapeSettings_Create(radius);
  if (!ss) {
    return PyErr_NoMemory();
  }

  JPH_Shape *shape = (JPH_Shape *)JPH_SphereShapeSettings_CreateShape(ss);
  JPH_ShapeSettings_Destroy((JPH_ShapeSettings *)ss);

  if (!shape) {
    return PyErr_NoMemory();
  }

  // 2. Prepare Result List
  PyObject *results = PyList_New(0);
  if (!results) {
    JPH_Shape_Destroy(shape);
    return NULL;
  }

  // 3. Stack Allocation with Alignment (CRITICAL)

  // Position
  JPH_STACK_ALLOC(JPH_RVec3, pos);
  pos->x = x;
  pos->y = y;
  pos->z = z;

  // Rotation (Identity)
  JPH_STACK_ALLOC(JPH_Quat, rot);
  rot->x = 0;
  rot->y = 0;
  rot->z = 0;
  rot->w = 1;

  // Transform Matrix (Pos + Rot)
  JPH_STACK_ALLOC(JPH_RMat4, transform);
  JPH_RMat4_RotationTranslation(transform, rot, pos);

  // Scale (Identity)
  JPH_STACK_ALLOC(JPH_Vec3, scale);
  scale->x = 1.0f;
  scale->y = 1.0f;
  scale->z = 1.0f;

  // Base Offset (Zero)
  JPH_STACK_ALLOC(JPH_RVec3, base_offset);
  base_offset->x = 0;
  base_offset->y = 0;
  base_offset->z = 0;

  // Settings
  JPH_STACK_ALLOC(JPH_CollideShapeSettings, settings);
  JPH_CollideShapeSettings_Init(settings);

  // 4. Execute Query
  QueryContext ctx = {self, results};
  const JPH_NarrowPhaseQuery *nq =
      JPH_PhysicsSystem_GetNarrowPhaseQuery(self->system);

  SHADOW_LOCK(&self->shadow_lock);
  JPH_NarrowPhaseQuery_CollideShape(
      nq, shape, scale, transform, settings, base_offset,
      OverlapCallback_Narrow, &ctx,
      NULL, // BroadPhaseLayerFilter (NULL = collide with everything)
      NULL, // ObjectLayerFilter
      NULL, // BodyFilter
      NULL  // ShapeFilter
  );
  SHADOW_UNLOCK(&self->shadow_lock);

  // 5. Cleanup
  JPH_Shape_Destroy(shape);

  return results;
}

static PyObject *PhysicsWorld_overlap_aabb(PhysicsWorldObject *self,
                                           PyObject *args, PyObject *kwds) {
  float min_x = NAN;
  float min_y = NAN;
  float min_z = NAN;
  float max_x = NAN;
  float max_y = NAN;
  float max_z = NAN;
  static char *kwlist[] = {"min", "max", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)(fff)", kwlist, &min_x,
                                   &min_y, &min_z, &max_x, &max_y, &max_z)) {
    return NULL;
  }

  // 1. Prepare Result List
  PyObject *results = PyList_New(0);
  if (!results) {
    return NULL;
  }

  // 2. Stack Allocation with Alignment (CRITICAL)
  JPH_STACK_ALLOC(JPH_AABox, box);
  box->min.x = min_x;
  box->min.y = min_y;
  box->min.z = min_z;

  box->max.x = max_x;
  box->max.y = max_y;
  box->max.z = max_z;

  // 3. Execute Query
  QueryContext ctx = {self, results};
  const JPH_BroadPhaseQuery *bq =
      JPH_PhysicsSystem_GetBroadPhaseQuery(self->system);

  SHADOW_LOCK(&self->shadow_lock);
  JPH_BroadPhaseQuery_CollideAABox(bq, box, OverlapCallback_Broad, &ctx,
                                   NULL, // BroadPhaseLayerFilter
                                   NULL  // ObjectLayerFilter
  );
  SHADOW_UNLOCK(&self->shadow_lock);

  return results;
}

// Change signature to include PyObject* kwds
static PyObject *PhysicsWorld_get_index(PhysicsWorldObject *self,
                                        PyObject *args, PyObject *kwds) {
  uint64_t h = 0;
  static char *kwlist[] = {"handle", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &h)) {
    return NULL;
  }
  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  if (!unpack_handle(self, h, &slot) || self->slot_states[slot] != SLOT_ALIVE) {
    SHADOW_UNLOCK(&self->shadow_lock);
    Py_RETURN_NONE;
  }
  uint32_t idx = self->slot_to_dense[slot];
  SHADOW_UNLOCK(&self->shadow_lock);
  return PyLong_FromUnsignedLong(idx);
}

static PyObject *PhysicsWorld_is_alive(PhysicsWorldObject *self, PyObject *args,
                                       PyObject *kwds) {
  uint64_t handle_raw = 0;
  static char *kwlist[] = {"handle", NULL};

  // Fix: Use PyArg_ParseTupleAndKeywords
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "K", kwlist, &handle_raw)) {
    return NULL;
  }

  SHADOW_LOCK(&self->shadow_lock);
  uint32_t slot = 0;
  bool alive = false;
  if (unpack_handle(self, (BodyHandle)handle_raw, &slot)) {
    uint8_t state = self->slot_states[slot];
    if (state == SLOT_ALIVE || state == SLOT_PENDING_CREATE) {
      alive = true;
    }
  }
  SHADOW_UNLOCK(&self->shadow_lock);

  if (alive) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *make_view(PhysicsWorldObject *self, void *ptr) {
  if (!ptr) {
    Py_RETURN_NONE;
  }

  // We tell Python: "This is one long list of floats"
  self->view_shape[0] = (Py_ssize_t)(self->count * 4);
  self->view_strides[0] = (Py_ssize_t)sizeof(float);

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));

  buf.buf = ptr;
  buf.obj = (PyObject *)self;
  Py_INCREF(self);

  buf.len = (Py_ssize_t)(self->count * 4 * sizeof(float));
  buf.readonly = 1;
  buf.itemsize = sizeof(float);
  buf.format = "f";

  // THE CRITICAL CHANGES:
  buf.ndim = 1;                     // Set to 1 Dimension
  buf.shape = &self->view_shape[0]; // Point to the first element of shape
  buf.strides = &self->view_strides[0];

  return PyMemoryView_FromBuffer(&buf);
}

static PyObject *get_user_data_buffer(PhysicsWorldObject *self, void *c) {
  // Similar to make_view but for uint64
  if (!self->user_data) {
    Py_RETURN_NONE;
  }

  self->view_shape[0] = (Py_ssize_t)self->count;
  self->view_shape[1] = 1; // 1D array effectively

  Py_buffer buf;
  memset(&buf, 0, sizeof(Py_buffer));
  buf.buf = self->user_data;
  buf.obj = (PyObject *)self;
  Py_INCREF(self);
  buf.len = (Py_ssize_t)(self->count * sizeof(uint64_t));
  buf.readonly = 1;
  buf.itemsize = sizeof(uint64_t);
  buf.format = "Q"; // Unsigned Long Long
  buf.ndim = 1;
  buf.shape = &self->view_shape[0];
  buf.strides = &buf.itemsize;

  return PyMemoryView_FromBuffer(&buf);
}

// --- Character Methods ---

static void Character_dealloc(CharacterObject *self) {
  // FIX 1: Use Base Destructor (Virtual inherits Base)
  if (self->character) {
    JPH_CharacterBase_Destroy((JPH_CharacterBase *)self->character);
  }
  if (self->listener) {
    JPH_CharacterContactListener_Destroy(self->listener);
  }

  if (self->body_filter) {
    JPH_BodyFilter_Destroy(self->body_filter);
  }
  if (self->shape_filter) {
    JPH_ShapeFilter_Destroy(self->shape_filter);
  }
  if (self->bp_filter) {
    JPH_BroadPhaseLayerFilter_Destroy(self->bp_filter);
  }
  if (self->obj_filter) {
    JPH_ObjectLayerFilter_Destroy(self->obj_filter);
  }

  Py_XDECREF(self->world);
  Py_TYPE(self)->tp_free((PyObject *)self);
}

// Move the character (Apply Velocity + Gravity + Collision resolution)
static PyObject *Character_move(CharacterObject *self, PyObject *args,
                                PyObject *kwds) {
  float vx = NAN;
  float vy = NAN;
  float vz = NAN;
  float dt = NAN;
  static char *kwlist[] = {"velocity", "dt", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)f", kwlist, &vx, &vy, &vz,
                                   &dt)) {
    return NULL;
  }

  self->last_vx = vx;
  self->last_vy = vy;
  self->last_vz = vz;

  // 1. Update Linear Velocity
  JPH_Vec3 v = {vx, vy, vz};
  JPH_CharacterVirtual_SetLinearVelocity(self->character, &v);

  // 2. Extended Update Settings (Stack Allocated)
  JPH_ExtendedUpdateSettings update_settings;
  memset(&update_settings, 0, sizeof(JPH_ExtendedUpdateSettings));

  // Stick to floor (prevents bouncing down slopes)
  update_settings.stickToFloorStepDown.x = 0;
  update_settings.stickToFloorStepDown.y = -0.5f;
  update_settings.stickToFloorStepDown.z = 0;

  // Step Up (Stairs)
  update_settings.walkStairsStepUp.x = 0;
  update_settings.walkStairsStepUp.y = 0.4f;
  update_settings.walkStairsStepUp.z = 0;

  update_settings.walkStairsMinStepForward = 0.02f;
  update_settings.walkStairsStepForwardTest = 0.15f;
  update_settings.walkStairsCosAngleForwardContact = 0.996f; // ~5 degrees

  update_settings.walkStairsStepDownExtra.x = 0;
  update_settings.walkStairsStepDownExtra.y = 0;
  update_settings.walkStairsStepDownExtra.z = 0;

  // 3. Execute Update (FIX: Added missing &update_settings arg)
  JPH_CharacterVirtual_ExtendedUpdate(self->character, dt, &update_settings, 1,
                                      self->world->system, self->body_filter,
                                      self->shape_filter);

  Py_RETURN_NONE;
}

static PyObject *Character_get_position(CharacterObject *self, PyObject *args) {
  JPH_RVec3 pos;
  JPH_CharacterVirtual_GetPosition(self->character, &pos);
  return Py_BuildValue("fff", (float)pos.x, (float)pos.y, (float)pos.z);
}

static PyObject *Character_set_position(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = NAN;
  float y = NAN;
  float z = NAN;
  static char *kwlist[] = {"pos", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(fff)", kwlist, &x, &y, &z)) {
    return NULL;
  }

  JPH_RVec3 pos = {(double)x, (double)y, (double)z};
  JPH_CharacterVirtual_SetPosition(self->character, &pos);
  Py_RETURN_NONE;
}

static PyObject *Character_set_rotation(CharacterObject *self, PyObject *args,
                                        PyObject *kwds) {
  float x = NAN;
  float y = NAN;
  float z = NAN;
  float w = NAN;
  static char *kwlist[] = {"rot", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "(ffff)", kwlist, &x, &y, &z,
                                   &w)) {
    return NULL;
  }

  JPH_Quat q = {x, y, z, w};
  JPH_CharacterVirtual_SetRotation(self->character, &q);
  Py_RETURN_NONE;
}

static PyObject *Character_is_grounded(CharacterObject *self, PyObject *args) {
  JPH_GroundState state =
      JPH_CharacterBase_GetGroundState((JPH_CharacterBase *)self->character);
  if (state == JPH_GroundState_OnGround ||
      state == JPH_GroundState_OnSteepGround) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyObject *Character_set_strength(CharacterObject *self, PyObject *args) {
  float strength = NAN;
  if (!PyArg_ParseTuple(args, "f", &strength)) {
    return NULL;
  }

  // Store in our struct for the callback
  self->push_strength = strength;

  // Also update Jolt's internal max strength (for char-vs-char interaction)
  JPH_CharacterVirtual_SetMaxStrength(self->character, strength);

  Py_RETURN_NONE;
}

static PyObject *get_positions(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->positions);
}
static PyObject *get_rotations(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->rotations);
}
static PyObject *get_velocities(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->linear_velocities);
}
static PyObject *get_angular_velocities(PhysicsWorldObject *self, void *c) {
  return make_view(self, self->angular_velocities);
}
static PyObject *get_count(PhysicsWorldObject *self, void *c) {
  return PyLong_FromSize_t(self->count);
}
static PyObject *get_time(PhysicsWorldObject *self, void *c) {
  return PyFloat_FromDouble(self->time);
}

// --- Type Definition ---

static const PyGetSetDef PhysicsWorld_getset[] = {
    {"positions", (getter)get_positions, NULL, NULL, NULL},
    {"rotations", (getter)get_rotations, NULL, NULL, NULL},
    {"velocities", (getter)get_velocities, NULL, NULL, NULL},
    {"angular_velocities", (getter)get_angular_velocities, NULL, NULL, NULL},
    {"count", (getter)get_count, NULL, NULL, NULL},
    {"time", (getter)get_time, NULL, NULL, NULL},
    {"user_data", (getter)get_user_data_buffer, NULL, NULL, NULL},
    {NULL}};

static const PyMethodDef PhysicsWorld_methods[] = {
    // --- Lifecycle ---
    {"step", (PyCFunction)PhysicsWorld_step, METH_VARARGS, NULL},
    {"create_body", (PyCFunction)PhysicsWorld_create_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"destroy_body", (PyCFunction)PhysicsWorld_destroy_body,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_mesh_body", (PyCFunction)PhysicsWorld_create_mesh_body,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Interaction ---
    {"apply_impulse", (PyCFunction)PhysicsWorld_apply_impulse,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_position", (PyCFunction)PhysicsWorld_set_position,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_rotation", (PyCFunction)PhysicsWorld_set_rotation,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_linear_velocity", (PyCFunction)PhysicsWorld_set_linear_velocity,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_angular_velocity", (PyCFunction)PhysicsWorld_set_angular_velocity,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_transform", (PyCFunction)PhysicsWorld_set_transform,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Motion Control ---
    {"get_motion_type", (PyCFunction)PhysicsWorld_get_motion_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_motion_type", (PyCFunction)PhysicsWorld_set_motion_type,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"activate", (PyCFunction)PhysicsWorld_activate,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"deactivate", (PyCFunction)PhysicsWorld_deactivate,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Queries ---
    {"raycast", (PyCFunction)PhysicsWorld_raycast, METH_VARARGS | METH_KEYWORDS,
     NULL},
    {"overlap_sphere", (PyCFunction)PhysicsWorld_overlap_sphere,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"overlap_aabb", (PyCFunction)PhysicsWorld_overlap_aabb,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- Utilities ---
    {"get_index", (PyCFunction)PhysicsWorld_get_index,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_alive", (PyCFunction)PhysicsWorld_is_alive,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- User Data (New) ---
    {"get_user_data", (PyCFunction)PhysicsWorld_get_user_data,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_user_data", (PyCFunction)PhysicsWorld_set_user_data,
     METH_VARARGS | METH_KEYWORDS, NULL},

    // --- State & Advanced ---
    {"save_state", (PyCFunction)PhysicsWorld_save_state, METH_NOARGS, NULL},
    {"load_state", (PyCFunction)PhysicsWorld_load_state,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"create_character", (PyCFunction)PhysicsWorld_create_character,
     METH_VARARGS | METH_KEYWORDS, NULL},

    {NULL, NULL, 0, NULL}};

static const PyMethodDef Character_methods[] = {
    {"move", (PyCFunction)Character_move, METH_VARARGS | METH_KEYWORDS, NULL},
    {"get_position", (PyCFunction)Character_get_position, METH_NOARGS, NULL},
    {"set_position", (PyCFunction)Character_set_position,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"set_rotation", (PyCFunction)Character_set_rotation,
     METH_VARARGS | METH_KEYWORDS, NULL},
    {"is_grounded", (PyCFunction)Character_is_grounded, METH_NOARGS, NULL},
    {"set_strength", (PyCFunction)Character_set_strength, METH_VARARGS,
     "Set the max pushing force"},
    {NULL}};

static const PyType_Slot PhysicsWorld_slots[] = {
    {Py_tp_new, PyType_GenericNew},
    {Py_tp_init, PhysicsWorld_init},
    {Py_tp_dealloc, PhysicsWorld_dealloc},
    {Py_tp_methods, (PyMethodDef *)PhysicsWorld_methods},
    {Py_tp_getset, (PyGetSetDef *)PhysicsWorld_getset},
    {0, NULL},
};

static const PyType_Slot Character_slots[] = {
    {Py_tp_dealloc, Character_dealloc},
    {Py_tp_methods, (PyMethodDef *)Character_methods},
    {0, NULL},
};

static const PyType_Spec PhysicsWorld_spec = {
    .name = "culverin._culverin_c.PhysicsWorld",
    .basicsize = sizeof(PhysicsWorldObject),
    .flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .slots = (PyType_Slot *)PhysicsWorld_slots,
};

static const PyType_Spec Character_spec = {
    .name = "culverin._culverin_c.Character",
    .basicsize = sizeof(CharacterObject),
    .flags = Py_TPFLAGS_DEFAULT, // Not a base type
    .slots = (PyType_Slot *)Character_slots,
};

// --- Module Initialization ---

static int culverin_exec(PyObject *m) {
  CulverinState *st = get_culverin_state(m);

  // 1. Initialize Jolt
  JPH_Init();

  // 2. Import Helper
  st->helper = PyImport_ImportModule("culverin._culverin");
  if (!st->helper) {
    return -1;
  }

  // 3. Create PhysicsWorld Type
  st->PhysicsWorldType =
      PyType_FromModuleAndSpec(m, (PyType_Spec *)&PhysicsWorld_spec, NULL);
  if (!st->PhysicsWorldType) {
    return -1;
  }

  if (PyModule_AddObject(m, "PhysicsWorld", st->PhysicsWorldType) < 0) {
    Py_DECREF(st->PhysicsWorldType);
    return -1;
  }
  Py_INCREF(st->PhysicsWorldType); // AddObject steals reference, but we keep
                                   // one in struct

  // 4. Create Character Type (NEW)
  st->CharacterType =
      PyType_FromModuleAndSpec(m, (PyType_Spec *)&Character_spec, NULL);
  if (!st->CharacterType) {
    return -1;
  }

  if (PyModule_AddObject(m, "Character", st->CharacterType) < 0) {
    Py_DECREF(st->CharacterType);
    return -1;
  }
  Py_INCREF(st->CharacterType); // Keep ref in struct

  // 5. Add Constants
  PyModule_AddIntConstant(m, "SHAPE_BOX", 0);
  PyModule_AddIntConstant(m, "SHAPE_SPHERE", 1);
  PyModule_AddIntConstant(m, "SHAPE_CAPSULE", 2);
  PyModule_AddIntConstant(m, "SHAPE_CYLINDER", 3);
  PyModule_AddIntConstant(m, "SHAPE_PLANE", 4);
  PyModule_AddIntConstant(m, "SHAPE_MESH", 5);

  PyModule_AddIntConstant(m, "MOTION_STATIC", 0);
  PyModule_AddIntConstant(m, "MOTION_KINEMATIC", 1);
  PyModule_AddIntConstant(m, "MOTION_DYNAMIC", 2);

  return 0;
}

static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
  CulverinState *st = get_culverin_state(m);
  Py_VISIT(st->helper);
  Py_VISIT(st->PhysicsWorldType);
  Py_VISIT(st->CharacterType);
  return 0;
}

static int culverin_clear(PyObject *m) {
  CulverinState *st = get_culverin_state(m);
  Py_CLEAR(st->helper);
  Py_CLEAR(st->PhysicsWorldType);
  Py_CLEAR(st->CharacterType);
  return 0;
}

static const PyModuleDef_Slot culverin_slots[] = {
    {Py_mod_exec, culverin_exec},
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, NULL}};
/* NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables) */
static PyModuleDef culverin_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_culverin_c", // Internal binary name
    .m_doc = "Culverin Physics Engine Core",
    .m_size = sizeof(CulverinState),
    .m_slots = (PyModuleDef_Slot *)culverin_slots,
    .m_traverse = culverin_traverse,
    .m_clear = culverin_clear,
};

PyMODINIT_FUNC PyInit__culverin_c(void) {
  return PyModuleDef_Init(&culverin_module);
}
