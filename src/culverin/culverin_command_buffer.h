#pragma once
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>
#include "culverin_types.h"


struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

// --- Command Buffer Optimized Layout (32 Bytes) ---

// Bit-packing helper macros
#define CMD_HEADER(type, slot) ((uint32_t)((type) & 0xFF) | ((slot) << 8))
#define CMD_GET_TYPE(header) ((CommandType)((header) & 0xFF))
#define CMD_GET_SLOT(header) ((header) >> 8)

// --- Slot State Machine ---
typedef enum {
  SLOT_EMPTY = 0,
  SLOT_PENDING_CREATE = 1,
  SLOT_ALIVE = 2,
  SLOT_PENDING_DESTROY = 3
} SlotState;

typedef enum {
  CMD_CREATE_BODY,
  CMD_DESTROY_BODY,
  CMD_SET_POS,
  CMD_SET_ROT,
  CMD_SET_TRNS, // Position + Rotation
  CMD_SET_LINVEL,
  CMD_SET_ANGVEL,
  CMD_SET_MOTION,
  CMD_ACTIVATE,
  CMD_DEACTIVATE,
  CMD_SET_USER_DATA,
  CMD_SET_CCD
} CommandType;

// Internal helper to resolve slots to Jolt IDs safely
typedef struct {
    JPH_BodyID bid;
    uint32_t dense_idx;
    bool is_alive;
} ResolvedCmd;

// Force 8-byte alignment for the whole union to ensure 64-bit pointers align
// correctly wherever they fall inside the 32-byte block.
#if defined(_MSC_VER)
__declspec(align(8))
#else
__attribute__((aligned(8)))
#endif
typedef union {
  uint32_t header;

  // 1. Create Body (Matches current logic)
  struct {
    uint32_t header;
    uint32_t material_id;
    JPH_BodyCreationSettings *settings;
    uint64_t user_data;
    uint32_t category;
    uint32_t mask;
    uint8_t _pad[32]; // Padding to 64
  } create;

  // 2. Transform (Updated to JPH_Real for Position)
  struct {
    uint32_t header;
    uint32_t _pad_align;    // Ensure JPH_Real starts at 8-byte boundary
    JPH_Real px, py, pz;    // 24 bytes (Double precision safe)
    float rx, ry, rz, rw;   // 16 bytes (Rotations are floats in Jolt)
    uint8_t _tail[16];      // Padding to 64
  } transform;

  // 3. Vector and quat (Updated to JPH_Real for Position/Velocity consistency)
  struct {
    uint32_t header;
    float x, y, z;
    uint8_t _pad[48]; 
  } vec3f;

  struct {
    uint32_t header;
    uint32_t _pad;
    JPH_Real x, y, z; 
    uint8_t _tail[32]; 
  } pos;

  struct {
    uint32_t header;
    float x, y, z, w;
    uint8_t _tail[44]; 
  } quat;

  // 4. Motion / CCD
  struct {
    uint32_t header;
    int32_t motion_type;
    uint8_t _pad[56];
  } motion;

  // 5. User Data
  struct {
    uint32_t header;        
    uint32_t _align_pad;    
    uint64_t user_data_val; 
    uint8_t _tail_pad[48];  
  } user_data;

  struct {
    uint32_t header;
    uint32_t _pad;
    PosStride pos;      // 24 bytes
    AuxStride velocity; // 16 bytes (New!)
    uint8_t _tail[16];  // Padding reduced!
  } teleport; // TODO: unused, but interesting. will implement later

} PhysicsCommand;

_Static_assert(sizeof(PhysicsCommand) == 64, "PhysicsCommand must be 64 bytes");

// INCLUDE AFTER PHYSICSCOMMAND!
#include "culverin.h"

void world_remove_body_slot(struct PhysicsWorldObject *self, uint32_t slot);
bool ensure_command_capacity(struct PhysicsWorldObject *self);
void flush_commands_internal(struct PhysicsWorldObject *self, PhysicsCommand *queue, size_t count);
void sync_and_flush_internal(struct PhysicsWorldObject *self);
void clear_command_queue(struct PhysicsWorldObject *self);