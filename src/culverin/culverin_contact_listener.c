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