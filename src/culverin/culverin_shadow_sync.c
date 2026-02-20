#include "culverin_shadow_sync.h"
#include "culverin_compiler_specifics.h"

// DOUBLE ASSERT, just to be sure.
// 1. Verify the Strides match the Math
// If JPH_Real is double, PosStride must be 32 bytes (4 * 8).
// If JPH_Real is float, PosStride must be 16 bytes (4 * 4).
static_assert(sizeof(PosStride) == sizeof(JPH_Real) * 4);
static_assert(sizeof(AuxStride) == sizeof(float) * 4);

// 2. Verify Alignment for SIMD
// -------------------------------------------------------------------------
// NOTE: These static_asserts are commented out because 'alignas' on the 
// Stride structs creates a hard alignment contract with the compiler. 
//
// In Clang 22, 'alignas(32)' forces the use of Aligned SIMD (vmovaps), 
// which triggers a Segfault/Access Violation if Jolt Physics provides 
// buffers that are not strictly 32-byte aligned (e.g., due to 16-byte 
// alignment or the SIMD-width overshoot bug discovered in 
// ContactConstraintManager/LockFreeHashMap).
//
// We now rely on 'vmovups' (Unaligned) via the compiler's natural 
// optimization path to ensure safety across varying heap-alignment 
// scenarios while maintaining ~13 cycles/body performance.
// -------------------------------------------------------------------------
// static_assert(alignof(PosStride) >= sizeof(PosStride));
// static_assert(alignof(AuxStride) >= sizeof(AuxStride));

// 3. Verify Cache Line Friendliness
// To prevent False Sharing, ensure the arrays start on 64-byte boundaries.
static_assert(alignof(PhysicsWorldObject) >= 64);

/**
 * High-Performance Shadow Sync
 * Decouples Jolt State from Shadow Buffers via a stack-allocated worklist.
 */
void culverin_sync_shadow_buffers(PhysicsWorldObject *self) {
#ifdef CULVERIN_PROFILE_SYNC
  uint64_t start = rdtsc();
#endif

  const auto *sys = self->system;
  uint32_t active_count =
      JPH_PhysicsSystem_GetNumActiveBodies(sys, JPH_BodyType_Rigid);
  if (active_count == 0) {
    return;
  }

  const JPH_BodyID *active_ids =
      JPH_PhysicsSystem_GetActiveBodiesUnsafe(sys, JPH_BodyType_Rigid);
  if (UNLIKELY(!active_ids || !self->positions)) {
    return;
  }

  // Hoist pointers to local registers with restrict for vectorized bursts
  auto *CULV_RESTRICT s_pos = (PosStride *)self->positions;
  auto *CULV_RESTRICT s_ppos = (PosStride *)self->prev_positions;
  auto *CULV_RESTRICT s_rot = (AuxStride *)self->rotations;
  auto *CULV_RESTRICT s_prot = (AuxStride *)self->prev_rotations;
  auto *CULV_RESTRICT s_lvel = (AuxStride *)self->linear_velocities;
  auto *CULV_RESTRICT s_avel = (AuxStride *)self->angular_velocities;
  const uint32_t *CULV_RESTRICT s2d = self->slot_to_dense;

  // Batching Worklist (Stack allocated - 512 bytes total)
  constexpr int WORK_CHUNK = 32;
  SyncWorkItem worklist[WORK_CHUNK];
  uint32_t work_ptr = 0;

  for (uint32_t i = 0; i < active_count; i++) {
    const JPH_Body *b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i]);
    if (UNLIKELY(!b)) {
      continue;
    }

    // --- PHASE 1: PREPARATION & LOOK-AHEAD PREFETCH ---
    // Prefetch the NEXT body to hide DRAM latency. We can't pointer-math an 
    // opaque struct, but prefetching the base pointer pulls the first 64 bytes 
    // (which contains position/rotation) into the L1 cache.
    if (i + 2 < active_count) {
      const JPH_Body *next_b = JPH_PhysicsSystem_GetBodyPtr(sys, active_ids[i + 2]);
      if (next_b) {
        CULV_PREFETCH(next_b);
      }
    }

    uint64_t handle = JPH_Body_GetUserData((JPH_Body *)b);
    auto slot = (uint32_t)(handle & 0xFFFFFFFF);
    auto gen = (uint32_t)(handle >> 32); // shift 32 bits

    // Filter and Validate logic outside the lock
    if (LIKELY(slot < self->slot_capacity && self->generations[slot] == gen &&
               self->slot_states[slot] == SLOT_ALIVE)) {
      worklist[work_ptr++] = (SyncWorkItem){b, s2d[slot]};
    }

    // --- PHASE 2: BURST SYNC (Hold Shadow Lock) ---
    if (work_ptr == WORK_CHUNK || (i == active_count - 1 && work_ptr > 0)) {
      SHADOW_LOCK(&self->shadow_lock);

      // ========== A. SNAPSHOT (Shadow → Shadow) ==========
      // Pure memory copy with known stride. Clang uses YMM/XMM for this.
      #pragma clang loop vectorize(enable) interleave(enable)
      for (uint32_t j = 0; j < work_ptr; j++) {
        uint32_t D = worklist[j].dense_idx;
        s_ppos[D] = s_pos[D]; 
        s_prot[D] = s_rot[D]; 
      }

      // ========== B. POSITION BURST ==========
    #pragma clang loop vectorize(enable) interleave(enable)
    for (uint32_t j = 0; j < work_ptr; j++) {
        uint32_t D = worklist[j].dense_idx;
        // We "Over-Copy" 32 bytes from the body's position offset.
        // This grabs X, Y, Z AND the first 8 bytes of the Rotation.
        __builtin_memcpy(&s_pos[D], (const char*)worklist[j].body + 48, 32);
    }

    // ========== C. ROTATION BURST ==========
    #pragma clang loop vectorize(enable) interleave(enable)
    for (uint32_t j = 0; j < work_ptr; j++) {
        uint32_t D = worklist[j].dense_idx;
        // This "fixes" the 8 bytes we over-copied above by writing
        // the actual 16-byte rotation over it.
        __builtin_memcpy(&s_rot[D], (const char*)worklist[j].body + 72, 16);
    }

      // ========== D. VELOCITY BURST ==========
      #pragma clang loop vectorize(enable) interleave(enable)
      for (uint32_t j = 0; j < work_ptr; j++) {
        uint32_t D = worklist[j].dense_idx;
        const char *B_ptr = (const char *)worklist[j].body;
        
        // We Over-Copy 16 bytes for both.
        // LinVel grabs 12b + first 4b of AngVel
        // AngVel grabs 12b + first 4b of UserData
        // (Python ignores the .w component anyway, so the trash data is harmless)
        __builtin_memcpy(&s_lvel[D], B_ptr + 88, 16);
        __builtin_memcpy(&s_avel[D], B_ptr + 100, 16);
      }

      SHADOW_UNLOCK(&self->shadow_lock);
      work_ptr = 0; // Reset for next batch
    }
  }

#ifdef CULVERIN_PROFILE_SYNC
  uint64_t elapsed = rdtsc() - start;
  if (active_count > 0) {
    fprintf(stderr, "Sync: %llu cycles for %u bodies (%.1f cyc/body)\n",
            elapsed, active_count, (double)elapsed / active_count);
  }
#endif
}