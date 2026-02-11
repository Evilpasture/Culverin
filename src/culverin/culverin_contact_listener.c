#include "culverin_contact_listener.h"
#include "culverin.h"

// --- Internal Contact Helper ---
static void process_contact_manifold(PhysicsWorldObject *self, 
                                     const JPH_Body *body1, const JPH_Body *body2, 
                                     const JPH_ContactManifold *manifold,
                                     ContactEventType type) {
  
  // Fast Pointer-based UserData retrieval (No Jolt locks)
  BodyHandle h1 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body1);
  BodyHandle h2 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body2);

  uint32_t slot1 = (uint32_t)(h1 & 0xFFFFFFFF);
  uint32_t slot2 = (uint32_t)(h2 & 0xFFFFFFFF);

  // Safety: Ensure slot is within our shadow buffer range
  if (slot1 >= self->slot_capacity || slot2 >= self->slot_capacity) return;

  uint32_t idx1 = self->slot_to_dense[slot1];
  uint32_t idx2 = self->slot_to_dense[slot2];

  // Bitmask Filter
  if (!(self->categories[idx1] & self->masks[idx2]) || 
      !(self->categories[idx2] & self->masks[idx1])) return;

  size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);
  if (idx >= self->contact_max_capacity) return;

  ContactEvent *ev = &self->contact_buffer[idx];
  ev->type = (uint32_t)type;

  JPH_STACK_ALLOC(JPH_Vec3, n);
  JPH_ContactManifold_GetWorldSpaceNormal(manifold, n);

  bool swapped = (h1 > h2);
  if (!swapped) {
    ev->body1 = h1; ev->body2 = h2;
  } else {
    ev->body1 = h2; ev->body2 = h1;
    n->x = -n->x; n->y = -n->y; n->z = -n->z;
  }
  ev->nx = n->x; ev->ny = n->y; ev->nz = n->z;

  JPH_STACK_ALLOC(JPH_RVec3, p);
  JPH_ContactManifold_GetWorldSpaceContactPointOn1(manifold, 0, p);
  ev->px = (float)p->x; ev->py = (float)p->y; ev->pz = (float)p->z;

  // Impulse math skipped for sensors to prevent Static Body access violations
  if (JPH_Body_IsSensor((JPH_Body*)body1) || JPH_Body_IsSensor((JPH_Body*)body2)) {
      ev->impulse = 0.0f;
      ev->sliding_speed_sq = 0.0f;
  } else {
      JPH_Vec3 v1 = {0,0,0}, v2 = {0,0,0};
      if (JPH_Body_GetMotionType((JPH_Body*)body1) != JPH_MotionType_Static) 
          JPH_Body_GetLinearVelocity((JPH_Body*)body1, &v1);
      if (JPH_Body_GetMotionType((JPH_Body*)body2) != JPH_MotionType_Static) 
          JPH_Body_GetLinearVelocity((JPH_Body*)body2, &v2);

      float dvx = swapped ? (v2.x - v1.x) : (v1.x - v2.x);
      float dvy = swapped ? (v2.y - v1.y) : (v1.y - v2.y);
      float dvz = swapped ? (v2.z - v1.z) : (v1.z - v2.z);

      float dot = dvx * ev->nx + dvy * ev->ny + dvz * ev->nz;
      ev->impulse = fabsf(dot);
      ev->sliding_speed_sq = (dvx*dvx + dvy*dvy + dvz*dvz) - (dot*dot);
  }

  ev->mat1 = self->material_ids[idx1];
  ev->mat2 = self->material_ids[idx2];

  atomic_thread_fence(memory_order_release);
}

// --- Global Contact Listener ---
// 1. ADDED
static void JPH_API_CALL on_contact_added(void *userData, const JPH_Body *body1,
                                          const JPH_Body *body2,
                                          const JPH_ContactManifold *manifold,
                                          JPH_ContactSettings *settings) {
    process_contact_manifold((PhysicsWorldObject *)userData, body1, body2, manifold, EVENT_ADDED);
}

// 2. PERSISTED (Uses same helper, different type ID)
static void JPH_API_CALL on_contact_persisted(void *userData, const JPH_Body *body1,
                                              const JPH_Body *body2,
                                              const JPH_ContactManifold *manifold,
                                              JPH_ContactSettings *settings) {
    process_contact_manifold((PhysicsWorldObject *)userData, body1, body2, manifold, EVENT_PERSISTED);
}

// 3. REMOVED (Simpler logic, no manifold)
static void JPH_API_CALL on_contact_removed(void *userData, const JPH_SubShapeIDPair *pair) {
    PhysicsWorldObject *self = (PhysicsWorldObject *)userData;

    // Use indices from BodyIDs to look up handles in our private map
    uint32_t i1 = JPH_ID_TO_INDEX(pair->Body1ID);
    uint32_t i2 = JPH_ID_TO_INDEX(pair->Body2ID);

    BodyHandle h1 = 0, h2 = 0;
    if (self->id_to_handle_map) {
        if (i1 < self->max_jolt_bodies) h1 = self->id_to_handle_map[i1];
        if (i2 < self->max_jolt_bodies) h2 = self->id_to_handle_map[i2];
    }
    
    if (h1 == 0 || h2 == 0) return; // Ignore unmapped bodies

    size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx >= self->contact_max_capacity) return;

    ContactEvent *ev = &self->contact_buffer[idx];
    ev->type = EVENT_REMOVED;
    ev->body1 = (h1 < h2) ? h1 : h2;
    ev->body2 = (h1 < h2) ? h2 : h1;

    // Zero geometry for removal
    memset(&ev->px, 0, sizeof(float) * 8); 

    atomic_thread_fence(memory_order_release);
}

static JPH_ValidateResult JPH_API_CALL on_contact_validate(
    void *userData, const JPH_Body *body1, const JPH_Body *body2,
    const JPH_RVec3 *baseOffset, const JPH_CollideShapeResult *result) {
  PhysicsWorldObject *self = (PhysicsWorldObject *)userData;

  // 1. Extract Slots
  BodyHandle h1 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body1);
  BodyHandle h2 = (BodyHandle)JPH_Body_GetUserData((JPH_Body *)body2);
  uint32_t slot1 = (uint32_t)(h1 & 0xFFFFFFFF);
  uint32_t slot2 = (uint32_t)(h2 & 0xFFFFFFFF);

  // 2. Bitmask Filter
  uint32_t idx1 = self->slot_to_dense[slot1];
  uint32_t idx2 = self->slot_to_dense[slot2];

  uint32_t cat1 = self->categories[idx1];
  uint32_t mask1 = self->masks[idx1];
  uint32_t cat2 = self->categories[idx2];
  uint32_t mask2 = self->masks[idx2];

  // 3. Logic: If either doesn't want to hit the other's category, reject.
  if (!(cat1 & mask2) || !(cat2 & mask1)) {
    return JPH_ValidateResult_RejectContact; // <--- This stops the PHYSICS
                                             // solver
  }

  return JPH_ValidateResult_AcceptContact;
}

const JPH_ContactListener_Procs contact_procs = {
    .OnContactValidate = on_contact_validate,
    .OnContactAdded = on_contact_added,
    .OnContactPersisted = on_contact_persisted,
    .OnContactRemoved = on_contact_removed
};

// Fixed get_contact_events to be safer with locking
PyObject *PhysicsWorld_get_contact_events(PhysicsWorldObject *self,
                                                 PyObject *args) {
  // 1. Enter the lock to check the state machine
  SHADOW_LOCK(&self->shadow_lock);

  // GUARD: Ensure we aren't mid-step and prevent a new step from starting
  BLOCK_UNTIL_NOT_STEPPING(self);

  // 2. Acquire index (Memory Visibility)
  size_t count =
      atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
  if (count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyList_New(0);
  }

  if (count > self->contact_max_capacity) {
    count = self->contact_max_capacity;
  }

  // 3. Fast Copy (Hold lock for the shortest possible time)
  ContactEvent *scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
  if (!scratch) {
    SHADOW_UNLOCK(&self->shadow_lock);
    return PyErr_NoMemory();
  }

  // Snapshot the buffer data
  memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));

  // Clear the buffer index for the next frame
  atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

  // 4. EXIT the lock immediately
  SHADOW_UNLOCK(&self->shadow_lock);

  // 5. Slow Python Work (Done while the next physics step can run in parallel!)
  PyObject *list = PyList_New((Py_ssize_t)count);
  if (!list) {
    PyMem_RawFree(scratch);
    return NULL;
  }

  for (size_t i = 0; i < count; i++) {
    // Create a 4-item tuple: (ID1, ID2, ImpactStrength, SlidingStrengthSq)
    PyObject *item = PyTuple_New(4);
    if (!item) {
      Py_DECREF(list);
      PyMem_RawFree(scratch);
      return NULL;
    }

    // PyTuple_SET_ITEM "steals" the reference, so no extra DECREF needed on
    // these creators
    PyTuple_SET_ITEM(item, 0, PyLong_FromUnsignedLongLong(scratch[i].body1));
    PyTuple_SET_ITEM(item, 1, PyLong_FromUnsignedLongLong(scratch[i].body2));
    PyTuple_SET_ITEM(item, 2, PyFloat_FromDouble(scratch[i].impulse));
    PyTuple_SET_ITEM(item, 3, PyFloat_FromDouble(scratch[i].sliding_speed_sq));

    PyList_SET_ITEM(list, (Py_ssize_t)i, item);
  }

  PyMem_RawFree(scratch);
  return list;
}

PyObject *PhysicsWorld_get_contact_events_ex(PhysicsWorldObject *self, PyObject *args) {
    // 1. Acquire Lock & Copy Data
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);
    
    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyList_New(0);
    }

    if (count > self->contact_max_capacity) {
        count = self->contact_max_capacity;
    }

    // Allocate scratch buffer
    ContactEvent *scratch = PyMem_RawMalloc(count * sizeof(ContactEvent));
    if (!scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    // Copy data and release lock immediately
    memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // 2. Static Keys (Allocated ONCE)
    // We intentionally "leak" these references for the lifetime of the app.
    // This provides a massive speedup and stability fix.
    static PyObject *k_bodies = NULL;
    static PyObject *k_pos = NULL;
    static PyObject *k_norm = NULL;
    static PyObject *k_str = NULL;
    static PyObject *k_slide = NULL;
    static PyObject *k_mat = NULL;
    static PyObject *k_type = NULL;

    if (!k_bodies) {
        k_bodies = PyUnicode_InternFromString("bodies");
        k_pos = PyUnicode_InternFromString("position");
        k_norm = PyUnicode_InternFromString("normal");
        k_str = PyUnicode_InternFromString("strength");
        k_slide = PyUnicode_InternFromString("slide_sq");
        k_mat = PyUnicode_InternFromString("materials");
        k_type = PyUnicode_InternFromString("type");

        // Paranoid check: if any failed during init, clean up and fail
        if (!k_bodies || !k_pos || !k_norm || !k_str || !k_slide || !k_mat || !k_type) {
            PyMem_RawFree(scratch);
            return PyErr_NoMemory();
        }
    }

    // 3. Build Python List
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) {
        PyMem_RawFree(scratch);
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEvent *e = &scratch[i];

        PyObject *dict = PyDict_New();
        if (!dict) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, (Py_ssize_t)i, Py_None);
            continue;
        }

        // 1. Bodies (u64, u64)
        PyObject *b_tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(b_tuple, 0, PyLong_FromUnsignedLongLong(e->body1));
        PyTuple_SET_ITEM(b_tuple, 1, PyLong_FromUnsignedLongLong(e->body2));
        PyDict_SetItem(dict, k_bodies, b_tuple);
        Py_DECREF(b_tuple);

        // 2. Position (f, f, f)
        PyObject *p_tuple = PyTuple_New(3);
        PyTuple_SET_ITEM(p_tuple, 0, PyFloat_FromDouble(e->px));
        PyTuple_SET_ITEM(p_tuple, 1, PyFloat_FromDouble(e->py));
        PyTuple_SET_ITEM(p_tuple, 2, PyFloat_FromDouble(e->pz));
        PyDict_SetItem(dict, k_pos, p_tuple);
        Py_DECREF(p_tuple);

        // 3. Normal (f, f, f)
        PyObject *n_tuple = PyTuple_New(3);
        PyTuple_SET_ITEM(n_tuple, 0, PyFloat_FromDouble(e->nx));
        PyTuple_SET_ITEM(n_tuple, 1, PyFloat_FromDouble(e->ny));
        PyTuple_SET_ITEM(n_tuple, 2, PyFloat_FromDouble(e->nz));
        PyDict_SetItem(dict, k_norm, n_tuple);
        Py_DECREF(n_tuple);

        // 4. Materials (u32, u32)
        PyObject *m_tuple = PyTuple_New(2);
        PyTuple_SET_ITEM(m_tuple, 0, PyLong_FromUnsignedLong(e->mat1));
        PyTuple_SET_ITEM(m_tuple, 1, PyLong_FromUnsignedLong(e->mat2));
        PyDict_SetItem(dict, k_mat, m_tuple);
        Py_DECREF(m_tuple);

        // 5. Strength (float)
        PyObject *s_val = PyFloat_FromDouble(e->impulse);
        PyDict_SetItem(dict, k_str, s_val);
        Py_DECREF(s_val);

        // 6. Sliding Speed (float)
        PyObject *sl_val = PyFloat_FromDouble(e->sliding_speed_sq);
        PyDict_SetItem(dict, k_slide, sl_val);
        Py_DECREF(sl_val);

        // 7. Event Type (int)
        PyObject *t_val = PyLong_FromUnsignedLong(e->type);
        PyDict_SetItem(dict, k_type, t_val);
        Py_DECREF(t_val);

        // Steals ref to dict
        PyList_SET_ITEM(list, (Py_ssize_t)i, dict);
    }

    // REMOVED: Py_DECREF(keys) - we keep them alive statically now.
    
    PyMem_RawFree(scratch);
    return list;
}
// ContactEvent layout (packed, little-endian):
// uint64 body1, uint64 body2
// float32 px, py, pz
// float32 nx, ny, nz
// float32 impulse
PyObject *PhysicsWorld_get_contact_events_raw(PhysicsWorldObject *self,
                                                     PyObject *args) {
  // 1. Phase Guard
  SHADOW_LOCK(&self->shadow_lock);
  BLOCK_UNTIL_NOT_STEPPING(self);

  // 2. Atomic Acquire (Publication Visibility)
  size_t count =
      atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);

  if (count == 0) {
    SHADOW_UNLOCK(&self->shadow_lock);
    // Return empty view
    PyObject *empty = PyBytes_FromStringAndSize("", 0);
    PyObject *view = PyMemoryView_FromObject(empty);
    Py_DECREF(empty);
    return view;
  }

  if (count > self->contact_max_capacity) {
    count = self->contact_max_capacity;
  }

  // 3. Snapshot Data
  // We copy into a PyBytes object. This is fast (memcpy) and
  // ensures the data remains valid even after the next step() resets the
  // buffer.
  size_t bytes_size = count * sizeof(ContactEvent);
  PyObject *raw_bytes = PyBytes_FromStringAndSize((char *)self->contact_buffer,
                                                  (Py_ssize_t)bytes_size);

  // 4. Reset Index for next frame
  atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

  SHADOW_UNLOCK(&self->shadow_lock);

  if (!raw_bytes) {
    return NULL;
  }

  // 5. Wrap in MemoryView
  // This allows the user to use np.frombuffer(events, dtype=...) without extra
  // copies
  PyObject *view = PyMemoryView_FromObject(raw_bytes);
  Py_DECREF(raw_bytes);
  return view;
}