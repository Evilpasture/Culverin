#include "culverin_contact_listener.h"
#include "culverin_fast_build.h"
#include "culverin_physics_world.h"

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