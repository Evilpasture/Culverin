#pragma once

#include "culverin_compiler_specifics.h"
#include "culverin_parsers.h"
#include "culverin_types.h"

/**
 * ============================================================================
 * CULVERIN FAST PARSE ENGINE (C23 "O(1)" / MULTI-INTERPRETER EDITION)
 * ============================================================================
 *
 * A high-performance, zero-allocation argument parsing system for Python C
 * extensions. Optimized for PEP 703 (Free-threading) and PEP 489 (Multi-phase
 * initialization).
 *
 * This version uses "Module State Isolation" to remain safe for multiple
 * subinterpreters. All parser metadata and interned keyword pointers are
 * stored within the interpreter-specific CulverinState.
 *
 * ----------------------------------------------------------------------------
 * 1. THE SETUP (Isolated Schema Bundling)
 * ----------------------------------------------------------------------------
 * Define your API in culverin_arg_indices.h using X-Macros. All parsers
 * are bundled into a struct instead of global variables.
 *
 *   #define FOR_ALL_PARSERS(X) \
 *       X(Force, Vec3, SCHEMA_VEC3) \
 *       X(Torque, Vec3, SCHEMA_VEC3)
 *
 *   typedef struct {
 *       FOR_ALL_PARSERS(DECLARE_PARSER) // Expands to FastParser members
 *   } CulverinParsers;
 *
 * ----------------------------------------------------------------------------
 * 2. INITIALIZATION (Interpreter Local)
 * ----------------------------------------------------------------------------
 * In your module's exec slot (Py_mod_exec), initialize the parsers for the
 * current interpreter instance:
 *
 *   PyType_DeclareSlot_Status culverin_exec(PyObject *m) {
 *       CulverinState *st = get_culverin_state(m);
 *       culverin_init_all_parsers(&st->parsers);
 *       return 0;
 *   }
 *
 * ----------------------------------------------------------------------------
 * 3. THE USAGE (Safe Function Level)
 * ----------------------------------------------------------------------------
 * Retrieve the state from the module and use the local parser struct.
 *
 *   static PyObject* my_func(PyObject* self, PyObject* const* args,
 *                            size_t nargsf, PyObject* kwnames) {
 *       // Fetch state specific to THIS interpreter
 *       CulverinState *st = get_culverin_state(PyType_GetModule(Py_TYPE(self)));
 *
 *       uint64_t h_raw;
 *       float ix;
 *       float iy;
 *       float iz;
 *       JPH_Real px;
 *       JPH_Real py;
 *       JPH_Real pz;
 *       const void *const restrict targets[ImpAt_COUNT] = {[IDX_IMPAT_H] = (const void *const
 * restrict)&h_raw, [IDX_IMPAT_IX] = (const void *const restrict)&ix, [IDX_IMPAT_IY] = (const void
 * *const restrict)&iy,   [IDX_IMPAT_IZ] = (const void *const restrict)&iz, [IDX_IMPAT_PX] = (const
 * void *const restrict)&px,   [IDX_IMPAT_PY] = (const void *const restrict)&py, [IDX_IMPAT_PZ] =
 * (const void *const restrict)&pz};
 *
 *       if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
 * &st->parsers.ImpulseAtParser, targets)) { return nullptr;
 *       }
 *
 *
 * ----------------------------------------------------------------------------
 * 4. CRITICAL INVARIANTS (The "Don't Crash" Rules)
 * ----------------------------------------------------------------------------
 *
 * INVARIANT A: POINTER ISOLATION
 * PyObject* addresses for interned keywords (e.g., "handle") are ONLY valid
 * in the interpreter that created them. NEVER share a FastParser struct
 * globally across C-memory.
 *
 * INVARIANT B: SCHEMA INTEGRITY
 * Target array indices must match the schema used for initialization.
 * Managed via the GroupName##_Idx enums in culverin_arg_indices.h.
 *
 * INVARIANT C: PRECISION SAFETY
 * Using 'JPH_Real' automatically selects float or double based on the
 * Jolt Physics build, preventing memory corruption in shadow buffers.
 *
 * ----------------------------------------------------------------------------
 * 5. PERFORMANCE CHARACTERISTICS
 * ----------------------------------------------------------------------------
 * - Zero Allocation: No temporary Python objects created on the hot path.
 * - Pointer Hashing: O(1) comparison of interned keyword addresses.
 * - Vectorcall Native: Directly supports PEP 590 fast-calling convention.
 * - Subinterpreter Friendly: Fully isolated; safe for concurrent interpreter
 *   initialization in Python 3.12+.
 * ============================================================================
 */

// Culverin-Specific Converters
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_vec3f(PyObject *o, const void *t) {
    Vec3f *v = (Vec3f *)t; // Cast away const to write to the struct
    return parse_vec3_f32(o, &v->x, &v->y, &v->z) != 0;
}

CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_pos_stride(PyObject *o, const void *t) {
    PosStride *v = (PosStride *)t;
    return parse_vec3_r64(o, &v->x, &v->y, &v->z) != 0;
}

CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_aux_stride(PyObject *o, const void *t) {
    AuxStride *v = (AuxStride *)t;
    return parse_quat_f32(o, &v->x, &v->y, &v->z, &v->w) != 0;
}

#define FP_CUSTOM_TYPE_NAMES , Vec3f : "Vector3", PosStride : "Vector3", AuxStride : "Quaternion"

#define FP_CUSTOM_CONVERTERS                                                                       \
    , Vec3f : fp_conv_vec3f, PosStride : fp_conv_pos_stride, AuxStride : fp_conv_aux_stride

// 4. Include the actual submodule
#include <fast_parse.h>