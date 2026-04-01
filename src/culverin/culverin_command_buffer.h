#pragma once
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"
#include "joltc.h"
#include <stddef.h>
#include <stdint.h>

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

// --- Command Buffer Optimized Layout (64 Bytes per Cache Line) ---

// Bit-packing helper macros
#define CMD_HEADER(type, slot) ((uint32_t)((type) & 0xFF) | ((slot) << 8))
#define CMD_GET_TYPE(header) ((CommandType)((header) & 0xFF))
#define CMD_GET_SLOT(header) ((header) >> 8)

#if defined(JPH_DOUBLE_PRECISION)
CULV_MAYBE_UNUSED static constexpr int16_t CMD_ALIGN = 16;
#else
CULV_MAYBE_UNUSED static constexpr int16_t CMD_ALIGN = 8;
#endif

// --- Slot State Machine ---
typedef enum SlotState : uint8_t {
    SLOT_EMPTY           = 0,
    SLOT_PENDING_CREATE  = 1,
    SLOT_ALIVE           = 2,
    SLOT_PENDING_DESTROY = 3,
    SLOT_CHARACTER       = 4
} SlotState;

typedef enum CommandType : uint8_t {
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
    CMD_SET_CCD,
    CMD_TELEPORT,
    CMD_APPLY_IMPULSE,
    CMD_APPLY_FORCE,
    CMD_APPLY_TORQUE,
    CMD_APPLY_ANG_IMPULSE,
    CMD_APPLY_IMPULSE_AT
} CommandType;

// Internal helper to resolve slots to Jolt IDs safely
typedef struct {
    JPH_BodyID bid;
    uint32_t dense_idx;
    bool is_alive;
} ResolvedCmd;

// Force exactly 64-byte alignment and sizing. 
// This ensures exactly ONE command per CPU Cache Line, preventing false-sharing 
// and cache-straddling across thread boundaries.
#if defined(_MSC_VER)
__declspec(align(64))
#else
__attribute__((aligned(MEMORY_ALIGNMENT_SIZE)))
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
    } create;

    // 2. Transform (Updated to JPH_Real for Position)
    struct {
        uint32_t header;
        uint32_t _pad_align;  // Ensure JPH_Real starts at 8-byte boundary
        JPH_Real px, py, pz;  // 24 bytes (Double precision safe)
        float rx, ry, rz, rw; // 16 bytes (Rotations are floats in Jolt)
    } transform;

    // 3. Vector and quat (Updated to JPH_Real for Position/Velocity consistency)
    struct {
        uint32_t header;
        uint32_t _pad;
        float x, y, z;
    } vec3f;

    struct {
        uint32_t header;
        uint32_t _pad;
        JPH_Real x, y, z;
    } pos;

    struct {
        uint32_t header;
        uint32_t _pad;
        float x, y, z, w;
    } quat;

    // 4. Motion / CCD
    struct {
        uint32_t header;
        uint32_t _pad;
        int32_t motion_type;
    } motion;

    // 5. User Data
    struct {
        uint32_t header;
        uint32_t _align_pad;
        uint64_t user_data_val;
    } user_data;

    struct {
        uint32_t header;
        uint32_t _pad;
        JPH_Real px, py, pz;
        float ix, iy, iz;
    } teleport; // TODO: unused, but interesting. will implement later

    struct {
        uint32_t header;
        uint32_t _pad;       // Move padding UP to offset 4
        JPH_Real px, py, pz; // Starts at offset 8 (Clean 8-byte alignment)
        float ix, iy, iz;    // Force vector follows at offset 32
    } impulse_at;

    // Forces the entire union to be exactly 64 bytes
    uint8_t _cache_pad[MEMORY_ALIGNMENT_SIZE];

} PhysicsCommand;

// C23 native static_assert
static_assert(sizeof(PhysicsCommand) == MEMORY_ALIGNMENT_SIZE, "PhysicsCommand MUST be exactly 64 bytes for cache alignment");
static_assert(offsetof(PhysicsCommand, vec3f.x) == 8, "vec3f.x must start at offset 8");
static_assert(offsetof(PhysicsCommand, transform.px) == 8, "transform.px must start at offset 8");
static_assert(offsetof(PhysicsCommand, create.settings) == 8, "create.settings must start at offset 8");
static_assert(alignof(PhysicsCommand) == MEMORY_ALIGNMENT_SIZE, "PhysicsCommand must be 64-byte aligned");

// INCLUDE AFTER PHYSICSCOMMAND!
#include "culverin.h"

void world_remove_body_slot(struct PhysicsWorldObject *self, uint32_t slot);
bool ensure_command_capacity(struct PhysicsWorldObject *self);
void flush_commands_internal(struct PhysicsWorldObject *self, PhysicsCommand *queue, size_t count);
void sync_and_flush_internal(struct PhysicsWorldObject *self);
void clear_command_queue(struct PhysicsWorldObject *self);

// True Direct Threading: Evaluates the next command without returning to a while loop.
// Includes aggressive software prefetching for indirect lookups.
#define DISPATCH()                                                                                 \
    do {                                                                                           \
        if (UNLIKELY(i >= count))                                                                  \
            return;                                                                                \
        cmd    = &queue[i++];                                                                      \
        header = cmd->header;                                                                      \
        type   = CMD_GET_TYPE(header);                                                             \
        slot   = CMD_GET_SLOT(header);                                                             \
        /* Aggressive Prefetching of the NEXT loop's data dependencies */                          \
        if (LIKELY(i < count)) {                                                                   \
            CULV_PREFETCH(&queue[i]);                                                              \
            uint32_t next_slot = CMD_GET_SLOT(queue[i].header);                                    \
            CULV_PREFETCH(&self->slot_states[next_slot]);                                          \
            CULV_PREFETCH(&self->slot_to_dense[next_slot]);                                        \
        }                                                                                          \
        state = self->slot_states[slot];                                                           \
        bid   = JPH_INVALID_BODY_ID;                                                               \
        if (LIKELY(state == SLOT_ALIVE || state == SLOT_PENDING_CREATE ||                          \
                   state == SLOT_PENDING_DESTROY || state == SLOT_CHARACTER)) {                    \
            bid = self->body_ids[self->slot_to_dense[slot]];                                       \
        }                                                                                          \
        if (LIKELY(type == CMD_CREATE_BODY || bid != JPH_INVALID_BODY_ID)) {                       \
            goto *dispatch_table[type];                                                            \
        }                                                                                          \
        goto op_NEXT;                                                                              \
    } while (0)