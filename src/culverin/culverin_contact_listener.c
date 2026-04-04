#include "culverin_contact_listener.h"
#include "culverin.h"
#include "culverin_fast_build.h"

// --- Internal Contact Helper ---
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
CULV_NO_TSAN
static void process_contact_manifold(PhysicsWorldObject *self, const JPH_Body *body1,
                                     const JPH_Body *body2, const JPH_ContactManifold *manifold,
                                     ContactEventType type) {

    // 1. Get raw handles from Jolt (Jolt UserData is uint64_t)
    uint64_t r1 = JPH_Body_GetUserData((JPH_Body *)body1);
    uint64_t r2 = JPH_Body_GetUserData((JPH_Body *)body2);

    // 2. Resolve slots using standard integer registers
    uint32_t slot1 = (uint32_t)(r1 & HANDLE_INDEX_MASK);
    uint32_t slot2 = (uint32_t)(r2 & HANDLE_INDEX_MASK);

    // Safety: Slot capacity is stable during step()
    if (slot1 >= self->slot_capacity || slot2 >= self->slot_capacity) {
        return;
    }

    uint32_t idx1 = self->slot_to_dense[slot1];
    uint32_t idx2 = self->slot_to_dense[slot2];

    // 3. Bitmask Filter
    if (!(self->categories[idx1] & self->masks[idx2]) ||
        !(self->categories[idx2] & self->masks[idx1])) {
        return;
    }

    // 4. Reserve Event Slot
    size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx >= self->contact_max_capacity) {
        return;
    }

    ContactEvent *ev = &self->contact_buffer[idx];
    ev->type         = (uint32_t)type;

    JPH_STACK_ALLOC(JPH_Vec3, n);
    JPH_ContactManifold_GetWorldSpaceNormal(manifold, n);

    // 5. Canonical Ordering (Atomic Refactor)
    bool swapped = (r1 > r2);
    if (!swapped) {
        // TSan Fix: Store using raw registers to avoid implicit seq_cst loads
        atomic_store_explicit(&ev->body1, r1, memory_order_relaxed);
        atomic_store_explicit(&ev->body2, r2, memory_order_relaxed);
    } else {
        atomic_store_explicit(&ev->body1, r2, memory_order_relaxed);
        atomic_store_explicit(&ev->body2, r1, memory_order_relaxed);
        n->x = -n->x;
        n->y = -n->y;
        n->z = -n->z;
    }
    ev->nx = n->x;
    ev->ny = n->y;
    ev->nz = n->z;

    JPH_STACK_ALLOC(JPH_RVec3, p);
    JPH_ContactManifold_GetWorldSpaceContactPointOn1(manifold, 0, p);
    ev->px = (float)p->x;
    ev->py = (float)p->y;
    ev->pz = (float)p->z;

    // 6. Impulse Math
    if ((int)JPH_Body_IsSensor(body1) || (int)JPH_Body_IsSensor(body2)) {
        ev->impulse          = 0.0f;
        ev->sliding_speed_sq = 0.0f;
    } else {
        JPH_Vec3 v1 = {0, 0, 0};
        JPH_Vec3 v2 = {0, 0, 0};
        if (JPH_Body_GetMotionType(body1) != JPH_MotionType_Static) {
            JPH_Body_GetLinearVelocity((JPH_Body *)body1, &v1);
        }
        if (JPH_Body_GetMotionType(body2) != JPH_MotionType_Static) {
            JPH_Body_GetLinearVelocity((JPH_Body *)body2, &v2);
        }

        float dvx = (int)swapped ? (v2.x - v1.x) : (v1.x - v2.x);
        float dvy = (int)swapped ? (v2.y - v1.y) : (v1.y - v2.y);
        float dvz = (int)swapped ? (v2.z - v1.z) : (v1.z - v2.z);

        float dot            = dvx * ev->nx + dvy * ev->ny + dvz * ev->nz;
        ev->impulse          = fabsf(dot);
        ev->sliding_speed_sq = (dvx * dvx + dvy * dvy + dvz * dvz) - (dot * dot);
    }

    ev->mat1 = self->material_ids[idx1];
    ev->mat2 = self->material_ids[idx2];

    // Release ensures all event data is visible when Python reads contact_atomic_idx
    atomic_thread_fence(memory_order_release);
}

// --- Global Contact Listener ---
// 1. ADDED
CULV_NO_TSAN
static void JPH_API_CALL on_contact_added(void *userData, const JPH_Body *body1,
                                          const JPH_Body *body2,
                                          const JPH_ContactManifold *manifold,
                                          JPH_ContactSettings *Py_UNUSED(settings)) {
    process_contact_manifold((PhysicsWorldObject *)userData, body1, body2, manifold, EVENT_ADDED);
}

// 2. PERSISTED (Uses same helper, different type ID)
CULV_NO_TSAN
static void JPH_API_CALL on_contact_persisted(void *userData, const JPH_Body *body1,
                                              const JPH_Body *body2,
                                              const JPH_ContactManifold *manifold,
                                              JPH_ContactSettings *Py_UNUSED(settings)) {
    process_contact_manifold((PhysicsWorldObject *)userData, body1, body2, manifold,
                             EVENT_PERSISTED);
}

// 3. REMOVED (Simpler logic, no manifold)
CULV_NO_TSAN
static void JPH_API_CALL on_contact_removed(void *userData, const JPH_SubShapeIDPair *pair) {
    PhysicsWorldObject *self = (PhysicsWorldObject *)userData;

    // SAFETY: Never call JPH_BodyInterface_... here.
    // It will deadlock because Jolt is holding internal locks.

    uint32_t i1 = JPH_ID_TO_INDEX(pair->Body1ID);
    uint32_t i2 = JPH_ID_TO_INDEX(pair->Body2ID);

    uint64_t r1 = 0;
    uint64_t r2 = 0;

    if (LIKELY(self->id_to_handle_map)) {
        // Change < to <= to include the max_jolt_bodies index (1-based)
        if (i1 <= self->max_jolt_bodies) {
            // TSan Fix: Atomic load from map with acquire semantics
            r1 = atomic_load_explicit(&self->id_to_handle_map[i1], memory_order_acquire);
        }
        if (i2 <= self->max_jolt_bodies) {
            // TSan Fix: Atomic load from map with acquire semantics
            r2 = atomic_load_explicit(&self->id_to_handle_map[i2], memory_order_acquire);
        }
    }

    // Filter out internal Jolt bodies or unmapped objects
    if (r1 == 0 || r2 == 0) {
        return;
    }

    size_t idx = atomic_fetch_add_explicit(&self->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx >= self->contact_max_capacity) {
        return;
    }

    ContactEvent *ev = &self->contact_buffer[idx];
    ev->type         = EVENT_REMOVED;

    // TSan Fix: Canonicalize using raw registers to avoid implicit seq_cst loads
    if (r1 < r2) {
        atomic_store_explicit(&ev->body1, r1, memory_order_relaxed);
        atomic_store_explicit(&ev->body2, r2, memory_order_relaxed);
    } else {
        atomic_store_explicit(&ev->body1, r2, memory_order_relaxed);
        atomic_store_explicit(&ev->body2, r1, memory_order_relaxed);
    }

    // Zero out geometry (Not used for removal events)
    memset(&ev->px, 0, sizeof(float) * 8);

    // Release creates a fence: Python sees the full event when it checks the atomic index
    atomic_thread_fence(memory_order_release);
}

static JPH_ValidateResult JPH_API_CALL on_contact_validate(
    void *userData, const JPH_Body *body1, const JPH_Body *body2,
    const JPH_RVec3 *Py_UNUSED(baseOffset), const JPH_CollideShapeResult *Py_UNUSED(result)) {
    PhysicsWorldObject *self = (PhysicsWorldObject *)userData;

    // 1. Extract Raw Handles (Jolt UserData is natively uint64_t)
    uint64_t r1 = JPH_Body_GetUserData((JPH_Body *)body1);
    uint64_t r2 = JPH_Body_GetUserData((JPH_Body *)body2);

    // 2. Resolve Slots using standard integer registers
    uint32_t slot1 = (uint32_t)(r1 & HANDLE_INDEX_MASK);
    uint32_t slot2 = (uint32_t)(r2 & HANDLE_INDEX_MASK);

    // Safety: Verify slots are within shadow buffer bounds
    if (slot1 >= self->slot_capacity || slot2 >= self->slot_capacity) {
        return JPH_ValidateResult_RejectContact;
    }

    uint32_t idx1 = self->slot_to_dense[slot1];
    uint32_t idx2 = self->slot_to_dense[slot2];

    // 3. Bitmask Filter
    // categories and masks are standard uint32_t arrays, stable during world.step()
    uint32_t cat1  = self->categories[idx1];
    uint32_t mask1 = self->masks[idx1];
    uint32_t cat2  = self->categories[idx2];
    uint32_t mask2 = self->masks[idx2];

    // 4. Logic: Bidirectional Rejection
    // If either body does not want to collide with the other's category, reject.
    if (!(cat1 & mask2) || !(cat2 & mask1)) {
        return JPH_ValidateResult_RejectContact;
    }

    return JPH_ValidateResult_AcceptContact;
}

const JPH_ContactListener_Procs contact_procs = {.OnContactValidate  = on_contact_validate,
                                                 .OnContactAdded     = on_contact_added,
                                                 .OnContactPersisted = on_contact_persisted,
                                                 .OnContactRemoved   = on_contact_removed};

// Fixed get_contact_events to be safer with locking
PyCFunction_DeclareMethodFromModule PhysicsWorld_get_contact_events(PhysicsWorldObject *self,
                                                                    PyObject *Py_UNUSED(args)) {
    // --- 1. SNAPSHOT PHASE (Locked) ---
    SHADOW_LOCK(&self->shadow_lock);

    // Guard: Ensure we aren't reading while Jolt is mid-step updating the buffer
    BLOCK_UNTIL_NOT_STEPPING(self);

    // Load atomic index (Acquire ensures we see all Listener stores)
    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);

    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyList_New(0);
    }

    if (count > self->contact_max_capacity) {
        count = self->contact_max_capacity;
    }

    // Fast copy into local memory so we can drop the lock immediately
    ContactEvent *scratch = CULV_RAW_MALLOC(count * sizeof(ContactEvent));
    if (!scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));

    // Reset the index for the next frame
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 2. BUILD PHASE (Unlocked & FastBuild Integrated) ---
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) {
        CULV_RAW_FREE(scratch);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        // TSan Fix: Explicitly load the handles from the atomic members in the struct.
        // We use relaxed because this 'scratch' copy is thread-local and synchronized.
        uint64_t b1_raw = atomic_load_explicit(&scratch[i].body1, memory_order_relaxed);
        uint64_t b2_raw = atomic_load_explicit(&scratch[i].body2, memory_order_relaxed);

        /**
         * OPTIMIZATION: FastBuild_Tuple
         * 1. fb_from_u64 converts b1_raw and b2_raw
         * 2. fb_from_float converts impulse and sliding_speed_sq
         * 3. fb_pack_tuple performs a single O(1) allocation
         */
        PyObject *item =
            FastBuild_Tuple(b1_raw, b2_raw, scratch[i].impulse, scratch[i].sliding_speed_sq);

        if (UNLIKELY(!item)) {
            Py_DECREF(list);
            CULV_RAW_FREE(scratch);
            return nullptr;
        }

        // PyList_SET_ITEM steals the reference from FastBuild_Tuple
        PyList_SET_ITEM(list, (Py_ssize_t)i, item);
    }

    CULV_RAW_FREE(scratch);
    return list;
}

PyCFunction_DeclareMethodFromModule PhysicsWorld_get_contact_events_ex(PhysicsWorldObject *self,
                                                                       PyObject *Py_UNUSED(args)) {
    // --- 1. SNAPSHOT PHASE ---
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

    ContactEvent *scratch = CULV_RAW_MALLOC(count * sizeof(ContactEvent));
    if (!scratch) {
        SHADOW_UNLOCK(&self->shadow_lock);
        return PyErr_NoMemory();
    }

    memcpy(scratch, self->contact_buffer, count * sizeof(ContactEvent));
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);
    SHADOW_UNLOCK(&self->shadow_lock);

    // --- 2. KEY INTERNING (Persistent) ---
    static PyObject *k_bodies = nullptr;
    static PyObject *k_pos    = nullptr;
    static PyObject *k_norm   = nullptr;
    static PyObject *k_str    = nullptr;
    static PyObject *k_slide  = nullptr;
    static PyObject *k_mat    = nullptr;
    static PyObject *k_type   = nullptr;

    if (UNLIKELY(!k_bodies)) {
        k_bodies = PyUnicode_InternFromString("bodies");
        k_pos    = PyUnicode_InternFromString("position");
        k_norm   = PyUnicode_InternFromString("normal");
        k_str    = PyUnicode_InternFromString("impulse");
        k_slide  = PyUnicode_InternFromString("slide_sq");
        k_mat    = PyUnicode_InternFromString("materials");
        k_type   = PyUnicode_InternFromString("type");
    }

    // --- 3. BUILD PHASE (FastBuild Engine) ---
    PyObject *list = PyList_New((Py_ssize_t)count);
    if (!list) {
        CULV_RAW_FREE(scratch);
        return nullptr;
    }

    for (size_t i = 0; i < count; i++) {
        ContactEvent *e = &scratch[i];

        // TSan Fix: Explicit relaxed loads for atomic handles
        uint64_t b1 = atomic_load_explicit(&e->body1, memory_order_relaxed);
        uint64_t b2 = atomic_load_explicit(&e->body2, memory_order_relaxed);

        /**
         * OPTIMIZATION: FastBuild_Dict
         * We compose the nested tuples (pos, normal, bodies, mats)
         * and the dictionary in a single, readable expression.
         */
        PyObject *dict = FastBuild_Dict(
            k_bodies, FastBuild_Tuple(b1, b2), k_pos, FastBuild_Tuple(e->px, e->py, e->pz), k_norm,
            FastBuild_Tuple(e->nx, e->ny, e->nz), k_mat, FastBuild_Tuple(e->mat1, e->mat2), k_str,
            e->impulse, k_slide, e->sliding_speed_sq, k_type, e->type);

        if (UNLIKELY(!dict)) {
            Py_INCREF(Py_None);
            PyList_SET_ITEM(list, (Py_ssize_t)i, Py_None);
            continue;
        }

        // Steals ref to dict created by FastBuild
        PyList_SET_ITEM(list, (Py_ssize_t)i, dict);
    }

    CULV_RAW_FREE(scratch);
    return list;
}
// ContactEvent layout (packed, little-endian):
// - body1 (uint64)
// - body2 (uint64)
// - px, py, pz (float32)
// - nx, ny, nz (float32)
// - impulse (float32)
// - sliding_speed_sq (float32)
// - mat1 (uint32)
// - mat2 (uint32)
// - type (uint32)
// - _pad (uint32)
PyCFunction_DeclareMethodFromModule PhysicsWorld_get_contact_events_raw(PhysicsWorldObject *self,
                                                                        PyObject *Py_UNUSED(args)) {
    // 1. Phase Guard
    SHADOW_LOCK(&self->shadow_lock);
    BLOCK_UNTIL_NOT_STEPPING(self);

    // 2. Atomic Acquire (Publication Visibility)
    size_t count = atomic_load_explicit(&self->contact_atomic_idx, memory_order_acquire);

    if (count == 0) {
        SHADOW_UNLOCK(&self->shadow_lock);
        // Return empty view
        PyObject *empty = PyBytes_FromStringAndSize("", 0);
        PyObject *view  = PyMemoryView_FromObject(empty);
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
    PyObject *raw_bytes =
        PyBytes_FromStringAndSize((char *)self->contact_buffer, (Py_ssize_t)bytes_size);

    // 4. Reset Index for next frame
    atomic_store_explicit(&self->contact_atomic_idx, 0, memory_order_relaxed);

    SHADOW_UNLOCK(&self->shadow_lock);

    if (!raw_bytes) {
        return nullptr;
    }

    // 5. Wrap in MemoryView
    // This allows the user to use np.frombuffer(events, dtype=...) without extra
    // copies
    PyObject *view = PyMemoryView_FromObject(raw_bytes);
    Py_DECREF(raw_bytes);
    return view;
}
