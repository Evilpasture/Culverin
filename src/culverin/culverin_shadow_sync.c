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
// In Clang 23, 'alignas(32)' forces the use of Aligned SIMD (vmovaps), 
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
  constexpr size_t POSITION_SIZE = sizeof(PosStride); // sizeof(JPH_Real) * 4
  constexpr size_t AUXILLIARY_SIZE = sizeof(AuxStride); // sizeof(float) * 4

  // Hoist pointers to local registers (tells compiler they are stable)
  // Commented out to remove vmovaps -> vmovups
  auto *CULV_RESTRICT s_pos = (PosStride *)CULV_ASSUME_ALIGNED(self->positions, POSITION_SIZE);
  auto *CULV_RESTRICT s_ppos = (PosStride *)CULV_ASSUME_ALIGNED(self->prev_positions, POSITION_SIZE);
  auto *CULV_RESTRICT s_rot = (AuxStride *)CULV_ASSUME_ALIGNED(self->rotations, AUXILLIARY_SIZE);
  auto *CULV_RESTRICT s_prot = (AuxStride *)CULV_ASSUME_ALIGNED(self->prev_rotations, AUXILLIARY_SIZE);
  auto *CULV_RESTRICT s_lvel = (AuxStride *)CULV_ASSUME_ALIGNED(self->linear_velocities, AUXILLIARY_SIZE);
  auto *CULV_RESTRICT s_avel = (AuxStride *)CULV_ASSUME_ALIGNED(self->angular_velocities, AUXILLIARY_SIZE);
  // auto *CULV_RESTRICT s_pos = (PosStride *)self->positions;
  // auto *CULV_RESTRICT s_ppos = (PosStride *)self->prev_positions;
  // auto *CULV_RESTRICT s_rot = (AuxStride *)self->rotations;
  // auto *CULV_RESTRICT s_prot = (AuxStride *)self->prev_rotations;
  // auto *CULV_RESTRICT s_lvel = (AuxStride *)self->linear_velocities;
  // auto *CULV_RESTRICT s_avel = (AuxStride *)self->angular_velocities;
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

    // --- PHASE 1: PREPARATION (No Lock) ---
    uint64_t handle = JPH_Body_GetUserData((JPH_Body *)b);
    auto slot = (uint32_t)(handle & 0xFFFFFFFF);
    auto gen = (uint32_t)(handle >> 32); // shift 32 bits

    // Filter and Validate logic outside the lock
    if (LIKELY(slot < self->slot_capacity && self->generations[slot] == gen &&
               self->slot_states[slot] == SLOT_ALIVE)) {

      uint32_t dense = s2d[slot];

      // Prefetch Jolt body data into L1 cache for Phase 2
      CULV_PREFETCH(((const char *)b) + 48);

      worklist[work_ptr++] = (SyncWorkItem){b, dense};
    }

    // --- PHASE 2: BURST SYNC (Hold Shadow Lock) ---
    if (work_ptr == WORK_CHUNK || (i == active_count - 1 && work_ptr > 0)) {
      if (work_ptr > WORK_CHUNK) {
        unreachable();
      }
      SHADOW_LOCK(&self->shadow_lock);

// ========== PHASE A: SNAPSHOT (Shadow → Shadow) ==========
// This is a pure memory copy with known stride, easy to vectorize
#pragma clang loop unroll_count(8) interleave(enable)
// #pragma clang loop unroll(full) vectorize(enable)
      for (uint32_t j = 0; j < work_ptr; j++) {
        uint32_t D = worklist[j].dense_idx;
        s_ppos[D] = s_pos[D]; // 32-byte AVX move
        s_prot[D] = s_rot[D]; // 16-byte SSE move
      }

// ========== PHASE B: SYNC (Jolt → Shadow) ==========
#pragma clang loop unroll_count(WORK_CHUNK) vectorize(enable)// do we want to unroll? yes.
      for (uint32_t j = 0; j < work_ptr; j++) {
        const JPH_Body *B = worklist[j].body;
        uint32_t D = worklist[j].dense_idx;

        // Use stack locals as "landing zones" for the getters
        JPH_RVec3 p;
        JPH_Quat q;
        JPH_Vec3 lv;
        JPH_Vec3 av;

        // The compiler can only optimize these if it can see the source code
        JPH_Body_GetPosition(B, &p);
        JPH_Body_GetRotation(B, &q);
        JPH_Body_GetLinearVelocity((JPH_Body *)B, &lv);
        JPH_Body_GetAngularVelocity((JPH_Body *)B, &av);

        // Slam into Shadow
        s_pos[D] = (PosStride){p.x, p.y, p.z, 0.0};
        s_rot[D] = (AuxStride){q.x, q.y, q.z, q.w};
        s_lvel[D] = (AuxStride){lv.x, lv.y, lv.z, 0.0f};
        s_avel[D] = (AuxStride){av.x, av.y, av.z, 0.0f};
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