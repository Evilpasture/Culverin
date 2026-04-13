#pragma once
#include "culverin_compiler_specifics.h"
#include "culverin_types.h"
#include "joltc.h"
#include <Python.h>
#include <stddef.h>
#include <stdint.h>

struct PhysicsWorldObject;
typedef struct PhysicsWorldObject PhysicsWorldObject;

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
    SLOT_CHARACTER       = 4,
    SLOT_SOFT_BODY       = 5
} SlotState;

typedef enum CommandType : uint8_t {
    CMD_CREATE_BODY,
    CMD_CREATE_SOFT_BODY,
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

/**
 * Packs a CommandType and a Slot ID into a single U32 header.
 * Layout: [ Slot (24 bits) | Type (8 bits) ]
 */
CULV_NODISCARD [[gnu::const]]
static inline uint32_t CMD_HEADER(CommandType type, uint32_t slot) {
    static constexpr uint8_t CMD_TYPE_BITS  = 8;
    static constexpr uint8_t CMD_TYPE_MASK  = ((1u << CMD_TYPE_BITS) - 1);
    static constexpr uint8_t CMD_SLOT_SHIFT = CMD_TYPE_BITS;
    return ((uint32_t)type & CMD_TYPE_MASK) | (slot << CMD_SLOT_SHIFT);
}

/** Extracts the CommandType (lowest 8 bits) */
CULV_NODISCARD [[gnu::const]]
static inline CommandType CMD_GET_TYPE(uint32_t header) {
    static constexpr uint8_t CMD_TYPE_BITS = 8;
    static constexpr uint8_t CMD_TYPE_MASK = ((1u << CMD_TYPE_BITS) - 1);
    return (CommandType)(header & CMD_TYPE_MASK);
}

/** Extracts the Slot ID (upper 24 bits) */
CULV_NODISCARD [[gnu::const]]
static inline uint32_t CMD_GET_SLOT(uint32_t header) {
    static constexpr uint8_t CMD_TYPE_BITS  = 8;
    static constexpr uint8_t CMD_SLOT_SHIFT = CMD_TYPE_BITS;
    return header >> CMD_SLOT_SHIFT;
}

// Internal helper to resolve slots to Jolt IDs safely
typedef struct {
    JPH_BodyID bid;
    uint32_t dense_idx;
    bool is_alive;
} ResolvedCmd;

// Force exactly 64-byte alignment and sizing.
// This ensures exactly ONE command per CPU Cache Line, preventing false-sharing
// and cache-straddling across thread boundaries.
typedef union {
    [[gnu::aligned(64)]]
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

    struct {
        uint32_t header;
        uint32_t category;
        JPH_SoftBodyCreationSettings *settings;
        union {
            uint64_t u64;
            PyObject *obj;
            void *ptr;
        } user_data; // Still 8 bytes
        uint32_t mask;
        uint32_t material_id;
        uint32_t num_vertices;
    } create_soft;

    // Forces the entire union to be exactly 64 bytes
    uint8_t _cache_pad[MEMORY_ALIGNMENT_SIZE];

} PhysicsCommand;

static constexpr auto OFFSET_START =
    8; // The offset where the actual command data starts (after the header and padding)

// C23 native static_assert
static_assert(sizeof(PhysicsCommand) == MEMORY_ALIGNMENT_SIZE,
              "PhysicsCommand MUST be exactly 64 bytes for cache alignment");
static_assert(offsetof(PhysicsCommand, vec3f.x) == OFFSET_START, "vec3f.x must start at offset 8");
static_assert(offsetof(PhysicsCommand, transform.px) == OFFSET_START,
              "transform.px must start at offset 8");
static_assert(offsetof(PhysicsCommand, create.settings) == OFFSET_START,
              "create.settings must start at offset 8");
static_assert(alignof(PhysicsCommand) == MEMORY_ALIGNMENT_SIZE,
              "PhysicsCommand must be 64-byte aligned");
static_assert(offsetof(PhysicsCommand, create_soft.settings) == OFFSET_START);

// INCLUDE AFTER PHYSICSCOMMAND!
#include "culverin.h"

void world_remove_body_slot(struct PhysicsWorldObject *self, uint32_t slot);
CULV_NODISCARD
bool ensure_command_capacity(struct PhysicsWorldObject *self);
CULV_NODISCARD
bool ensure_command_bulk_capacity(PhysicsWorldObject *self, size_t batch_size);
void flush_commands_internal(struct PhysicsWorldObject *self, PhysicsCommand *queue, size_t count);
void sync_and_flush_internal(struct PhysicsWorldObject *self);
void clear_command_queue(struct PhysicsWorldObject *self);

// True Direct Threading: Evaluates the next command without returning to a while loop.
// Includes aggressive software prefetching for indirect lookups.
static constexpr uint32_t VALID_BID_MASK = (1u << SLOT_ALIVE) | (1u << SLOT_PENDING_CREATE) |
                                           (1u << SLOT_PENDING_DESTROY) |
                                           (1u << SLOT_CHARACTER | (1u << SLOT_SOFT_BODY));
#define DISPATCH()                                                                                 \
    do {                                                                                           \
        if (UNLIKELY(i >= count)) {                                                                \
            return;                                                                                \
        }                                                                                          \
        cmd    = &queue[i++];                                                                      \
        header = cmd->header;                                                                      \
        type   = CMD_GET_TYPE(header);                                                             \
        slot   = CMD_GET_SLOT(header);                                                             \
                                                                                                   \
        if (LIKELY(i < count)) {                                                                   \
            CULV_PREFETCH_READ(&queue[i]);                                                         \
            uint32_t next_slot = CMD_GET_SLOT(queue[i].header);                                    \
            CULV_PREFETCH_READ(&self->slot_states[next_slot]);                                     \
            CULV_PREFETCH_READ(&self->slot_to_dense[next_slot]);                                   \
        }                                                                                          \
                                                                                                   \
        state = atomic_load_explicit(&self->slot_states[slot], memory_order_acquire);              \
                                                                                                   \
        /* Branchless State Validation (Guarded against UB shift) */                               \
        /* If state > 31, (state < 32) evaluates to 0, masking the entire result to 0 */           \
        uint32_t is_valid = (state < 32) & ((VALID_BID_MASK >> (state & 31)) & 1);                 \
                                                                                                   \
        /* Unconditional Read: dense_idx and bid are always within allocated bounds */             \
        uint32_t dense_idx = self->slot_to_dense[slot];                                            \
        bid                = self->body_ids[dense_idx];                                            \
                                                                                                   \
        /* Branchless Condition: Is it CREATE, or does it have a valid BID? */                     \
        uint32_t is_executable =                                                                   \
            is_valid & ((type == CMD_CREATE_BODY) | (type == CMD_CREATE_SOFT_BODY) |               \
                        (bid != JPH_INVALID_BODY_ID));                                             \
                                                                                                   \
        /* Branchless Target Selection via Ternary (Compiles to CMOV / CSEL) */                    \
        const void *target = is_executable ? dispatch_table[type] : &&op_NOP;                      \
        goto *target;                                                                              \
    } while (0)
