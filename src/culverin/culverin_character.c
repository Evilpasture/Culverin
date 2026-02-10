#include "culverin.h"

// Character helpers
// Callback: Can the character collide with this object?

static bool JPH_API_CALL
char_on_contact_validate(void *userData, const JPH_CharacterVirtual *character,
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                         JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
  return true; // Usually true, unless you want to walk through certain bodies
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

static void record_character_contact(CharacterObject *self, JPH_BodyID bodyID2, 
                                     const JPH_RVec3 *pos, const JPH_Vec3 *norm, 
                                     ContactEventType type) {
    PhysicsWorldObject *world = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    BodyHandle h2 = 0;
    
    if (world->id_to_handle_map && j_idx < world->max_jolt_bodies) {
        h2 = world->id_to_handle_map[j_idx];
    }
    if (h2 == 0) return; // Ignore unmapped bodies (like internal Jolt helpers)

    BodyHandle h1 = self->handle;
    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = (uint32_t)type;
        
        // Consistent ordering for Python set logic
        if (h1 < h2) { ev->body1 = h1; ev->body2 = h2; }
        else { ev->body1 = h2; ev->body2 = h1; }

        ev->px = (float)pos->x; ev->py = (float)pos->y; ev->pz = (float)pos->z;
        ev->nx = norm->x; ev->ny = norm->y; ev->nz = norm->z;
        ev->impulse = 1.0f; // Logical trigger strength
        ev->sliding_speed_sq = 0.0f;
        
        // Look up material of the object we hit
        uint32_t slot2 = (uint32_t)(h2 & 0xFFFFFFFF);
        uint32_t dense2 = world->slot_to_dense[slot2];
        ev->mat1 = 0; // Characters don't have materials yet
        ev->mat2 = world->material_ids[dense2];

        atomic_thread_fence(memory_order_release);
    }
}

static void report_char_vs_char(CharacterObject *self, const JPH_CharacterVirtual *other, 
                                const JPH_Vec3 *normal, const JPH_RVec3 *pos, 
                                ContactEventType type) {
    PhysicsWorldObject *world = self->world;
    BodyHandle h1 = self->handle;
    
    // 1. Get Inner Body ID
    JPH_BodyID other_bid = JPH_CharacterVirtual_GetInnerBodyID(other);
    
    // 2. Direct Jolt Lookup (Bypasses our map, which might be too small for Virtual IDs)
    uint64_t userdata = JPH_BodyInterface_GetUserData(world->body_interface, other_bid);
    BodyHandle h2 = (BodyHandle)userdata;
    
    if (h2 == 0) return; // Not a known Culverin object

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = (uint32_t)type;
        
        // Canonicalize Order
        if (h1 < h2) { ev->body1 = h1; ev->body2 = h2; } 
        else { ev->body1 = h2; ev->body2 = h1; }

        ev->sliding_speed_sq = 0.0f;
        ev->nx = normal->x; ev->ny = normal->y; ev->nz = normal->z;
        ev->px = (float)pos->x; ev->py = (float)pos->y; ev->pz = (float)pos->z;
        ev->impulse = 1.0f;
        ev->mat1 = 0; ev->mat2 = 0; 

        atomic_thread_fence(memory_order_release);
    }
}
static void JPH_API_CALL char_on_character_contact_added(
    void *userData, const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, JPH_SubShapeID subShapeID2,
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {
    
    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;
    
    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;
    
    report_char_vs_char(self, otherCharacter, contactNormal, contactPosition, EVENT_ADDED);
}

static void apply_character_impulse(CharacterObject *self, JPH_BodyID bodyID2, const JPH_Vec3 *contactNormal) {
    // 1. Thread-Safe Member Access
    float vx = atomic_load_explicit((&self->last_vx), memory_order_relaxed);
    float vy = atomic_load_explicit((&self->last_vy), memory_order_relaxed);
    float vz = atomic_load_explicit((&self->last_vz), memory_order_relaxed);
    float strength = atomic_load_explicit((&self->push_strength), memory_order_relaxed);

    JPH_BodyInterface *bi = self->world->body_interface;

    // 2. Ignore Sensors & Non-Dynamic Bodies
    if (JPH_BodyInterface_IsSensor(bi, bodyID2) ||
        JPH_BodyInterface_GetMotionType(bi, bodyID2) != JPH_MotionType_Dynamic) {
        return;
    }

    // 3. Calculate Pushing Force
    float dot = vx * contactNormal->x + vy * contactNormal->y + vz * contactNormal->z;

    if (dot > 0.1f) {
        float factor = dot * strength;
        const float max_impulse = 5000.0f;
        if (factor > max_impulse) factor = max_impulse;

        JPH_Vec3 impulse;
        impulse.x = contactNormal->x * factor;
        
        // Flatten Y Response (allow kicking up, suppress crushing down)
        float y_push = contactNormal->y * factor;
        impulse.y = (y_push > 0.0f) ? y_push : 0.0f;
        
        impulse.z = contactNormal->z * factor;

        JPH_BodyInterface_AddImpulse(bi, bodyID2, &impulse);
        JPH_BodyInterface_ActivateBody(bi, bodyID2);
    }
}

// --- Updated Added Callback ---
static void JPH_API_CALL char_on_contact_added(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

  ioSettings->canPushCharacter = true;
  ioSettings->canReceiveImpulses = true;

  CharacterObject *self = (CharacterObject *)userData;
  if (!self) return;

  // Record Event
  record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_ADDED);

  // Apply Impulse
  apply_character_impulse(self, bodyID2, contactNormal);
}


static void JPH_API_CALL char_on_contact_persisted(
    void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2,
    JPH_SubShapeID subShapeID2, const JPH_RVec3 *contactPosition,
    const JPH_Vec3 *contactNormal, JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;

    CharacterObject *self = (CharacterObject *)userData;
    if (!self) return;

    // Record Event
    record_character_contact(self, bodyID2, contactPosition, contactNormal, EVENT_PERSISTED);

    // Apply Impulse (CRITICAL FIX)
    apply_character_impulse(self, bodyID2, contactNormal);
}

static void JPH_API_CALL char_on_contact_removed(void *userData, const JPH_CharacterVirtual *character, JPH_BodyID bodyID2, JPH_SubShapeID subShapeID2) {
    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;

    PhysicsWorldObject *world = self->world;
    uint32_t j_idx = JPH_ID_TO_INDEX(bodyID2);
    
    BodyHandle h1 = self->handle;
    BodyHandle h2 = 0;
    if (world->id_to_handle_map && j_idx < world->max_jolt_bodies) {
        h2 = world->id_to_handle_map[j_idx];
    }
    if (h2 == 0) return;

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = EVENT_REMOVED;
        ev->body1 = (h1 < h2) ? h1 : h2;
        ev->body2 = (h1 < h2) ? h2 : h1;
        // Geometry is zeroed for removal
        memset(&ev->px, 0, sizeof(float) * 8); 
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_character_contact_persisted(
    void *userData, const JPH_CharacterVirtual *character,
    const JPH_CharacterVirtual *otherCharacter, JPH_SubShapeID subShapeID2,
    const JPH_RVec3 *contactPosition, const JPH_Vec3 *contactNormal,
    JPH_CharacterContactSettings *ioSettings) {

    ioSettings->canPushCharacter = true;
    ioSettings->canReceiveImpulses = true;

    CharacterObject *self = (CharacterObject *)userData;
    if (!self || !self->world) return;

    report_char_vs_char(self, otherCharacter, contactNormal, contactPosition, EVENT_PERSISTED);
}

static void JPH_API_CALL char_on_character_contact_removed(void *userData, const JPH_CharacterVirtual *character, const JPH_CharacterID otherCharacterID, JPH_SubShapeID subShapeID2) {
    CharacterObject *self = (CharacterObject *)userData;
    PhysicsWorldObject *world = self->world;

    BodyHandle h1 = self->handle;
    // We have to use the CharacterID to find the handle. 
    // Jolt CharacterIDs usually map to the inner BodyID.
    uint32_t j_idx = JPH_ID_TO_INDEX(otherCharacterID);
    BodyHandle h2 = world->id_to_handle_map[j_idx];

    size_t idx = atomic_fetch_add_explicit(&world->contact_atomic_idx, 1, memory_order_relaxed);
    if (idx < world->contact_max_capacity) {
        ContactEvent *ev = &world->contact_buffer[idx];
        ev->type = EVENT_REMOVED;
        ev->body1 = (h1 < h2) ? h1 : h2;
        ev->body2 = (h1 < h2) ? h2 : h1;
        atomic_thread_fence(memory_order_release);
    }
}

static void JPH_API_CALL char_on_adjust_velocity(
    void *userData, const JPH_CharacterVirtual *character, const JPH_Body *body2, 
    JPH_Vec3 *ioLinearVelocity, JPH_Vec3 *ioAngularVelocity) {
    
    // Usually, we want the default behavior (character follows the body).
    // TODO: add logic here if you want the character to "slip" on certain materials.
}

// Map the procs
const JPH_CharacterContactListener_Procs char_listener_procs = {
    .OnContactValidate = char_on_contact_validate,
    .OnContactAdded = char_on_contact_added,
    .OnAdjustBodyVelocity = char_on_adjust_velocity, // ADDED
    .OnContactPersisted = char_on_contact_persisted, // CHANGED from char_on_contact_added
    .OnContactRemoved = char_on_contact_removed,      // ADDED
    .OnCharacterContactValidate = NULL,               // Default True is fine
    .OnCharacterContactAdded = char_on_character_contact_added,
    .OnCharacterContactPersisted = char_on_character_contact_persisted, // ADDED
    .OnCharacterContactRemoved = char_on_character_contact_removed,     // ADDED
    .OnContactSolve = NULL                            // Advanced, keep NULL
};