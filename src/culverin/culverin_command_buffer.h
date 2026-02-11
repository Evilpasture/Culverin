#pragma once
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>

struct PhysicsWorldObject; 
typedef struct PhysicsWorldObject PhysicsWorldObject;

// --- Command Buffer Optimized Layout (32 Bytes) ---

// Bit-packing helper macros
#define CMD_HEADER(type, slot) ((uint32_t)((type) & 0xFF) | ((slot) << 8))
#define CMD_GET_TYPE(header)   ((CommandType)((header) & 0xFF))
#define CMD_GET_SLOT(header)   ((header) >> 8)

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

// Force 8-byte alignment for the whole union to ensure 64-bit pointers align
// correctly wherever they fall inside the 32-byte block.
#if defined(_MSC_VER)
__declspec(align(8))
#else
__attribute__((aligned(8)))
#endif
typedef union {
  uint32_t header;

  // 1. Create Body (32 Bytes - Full)
  struct {
    uint32_t header;
    uint32_t material_id;
    JPH_BodyCreationSettings *settings;
    uint64_t user_data;
    uint32_t category;
    uint32_t mask;
  } create;

  // 2. Transform (32 Bytes - Full)
  struct {
    uint32_t header;
    float px, py, pz;
    float rx, ry, rz, rw;
    // No padding needed: 4 + 12 + 16 = 32
  } transform;

  // 3. Vector (20 Bytes -> 32 Bytes)
  struct {
    uint32_t header;
    float x, y, z, w;
    uint8_t _pad[12]; // Explicitly fill the rest
  } vec;

  // 4. Motion (8 Bytes -> 32 Bytes)
  struct {
    uint32_t header;
    int32_t motion_type;
    uint8_t _pad[24];
  } motion;

  // 5. User Data (16 Bytes -> 32 Bytes)
  // Note: uint64 must align to 8 bytes.
  // Layout: [H:4][Pad:4][U64:8][Pad:16]
  struct {
    uint32_t header;        // 4 bytes (Offset 0)
    uint32_t _align_pad;    // 4 bytes (Offset 4) -> Aligns next u64 to 8
    uint64_t user_data_val; // 8 bytes (Offset 8)
    uint8_t _tail_pad[16];  // 16 bytes (Offset 16)
  } user_data;

} PhysicsCommand;

// Per-struct guards. If any individual struct grows > 32, build fails.
_Static_assert(sizeof(PhysicsCommand) == 32,
               "PhysicsCommand union must be 32 bytes");
_Static_assert(sizeof(((PhysicsCommand *)0)->create) == 32,
               "Create struct overflow");
_Static_assert(sizeof(((PhysicsCommand *)0)->transform) == 32,
               "Transform struct overflow");

// INCLUDE AFTER PHYSICSCOMMAND!
#include "culverin.h"

void world_remove_body_slot(struct PhysicsWorldObject *self, uint32_t slot);
bool ensure_command_capacity(struct PhysicsWorldObject *self);
void flush_commands(struct PhysicsWorldObject *self);
void clear_command_queue(struct PhysicsWorldObject *self);