#pragma once

#include "culverin_compiler_specifics.h"
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>

/**
 * ============================================================================
 * CULVERIN FAST PARSE ENGINE (C23 "O(1)" EDITION)
 * ============================================================================
 *
 * A high-performance, zero-allocation argument parsing system for Python C
 * extensions. Replaces PyArg_ParseTupleAndKeywords with a hybrid system using 
 * X-Macro Schemas for "Lazy" maintenance and Pointer Hashing for O(1) lookups.
 *
 * ----------------------------------------------------------------------------
 * 1. THE SETUP (Lazy Schema Definition)
 * ----------------------------------------------------------------------------
 * Define your API once in culverin_arg_indices.h using X-Macros.
 * 
 *   #define SCHEMA_VEC3(X) \
 *       X(IDX_V3_H, "handle", uint64_t, 1) \  // REQUIRED (1)
 *       X(IDX_V3_X, "x",      float,    1) \
 *       X(IDX_V3_Y, "y",      float,    1) \
 *       X(IDX_V3_Z, "z",      float,    1)
 *
 *   // Generate Enum (IDX_V3_H...) and Count (Vec3_COUNT)
 *   DEFINE_INDEX_GROUP(Vec3, SCHEMA_VEC3)
 *
 *   // Declare specific Parser objects
 *   DECLARE_PARSER(Force, Vec3) 
 *   DECLARE_PARSER(Torque, Vec3)
 *
 * ----------------------------------------------------------------------------
 * 2. INITIALIZATION (Module Level)
 * ----------------------------------------------------------------------------
 * In culverin_arg_indices.c, allocate and initialize with one line:
 *
 *   ALLOC_PARSER(Force, Vec3)
 *
 *   void culverin_init_all_parsers() {
 *       INIT_PARSER(Force, Vec3, SCHEMA_VEC3);
 *   }
 *
 * ----------------------------------------------------------------------------
 * 3. THE USAGE (Function Level)
 * ----------------------------------------------------------------------------
 * Use the FastParse_Unified macro. It branches between Vectorcall and Legacy.
 *
 *   static PyObject* my_func(PyObject* self, PyObject* const* args,
 *                            size_t nargsf, PyObject* kwnames) {
 *       uint64_t h; float x, y, z;
 *
 *       // Target Array - Explicitly mapped using Schema IDs
 *       void *targets[Vec3_COUNT];
 *       targets[IDX_V3_H] = &h;
 *       targets[IDX_V3_X] = &x;
 *       targets[IDX_V3_Y] = &y;
 *       targets[IDX_V3_Z] = &z;
 *
 *       if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, 
 *                              &ForceParser, targets))
 *           return nullptr;
 *
 *       // ... physics logic ...
 *   }
 *
 * ----------------------------------------------------------------------------
 * 4. CRITICAL INVARIANTS (The "Don't Crash" Rules)
 * ----------------------------------------------------------------------------
 *
 * INVARIANT A: SCHEMA INTEGRITY
 * The ID used in the 'targets' array (e.g., IDX_V3_X) MUST match the schema 
 * used to initialize the parser. The X-Macro ensures this by generating the 
 * Enum and Parser Specs from the same source.
 *
 * INVARIANT B: PRECISION SAFETY
 * Using 'JPH_Real' in the Schema allows the engine to automatically dispatch 
 * to either 'float' or 'double' converters based on your Jolt build, 
 * preventing stack corruption.
 *
 * INVARIANT C: INITIALIZATION
 * culverin_init_all_parsers() MUST be called in the module exec phase. 
 * If interned pointers are uninitialized, O(1) address-hashing will fail.
 *
 * ----------------------------------------------------------------------------
 * 5. PERFORMANCE CHARACTERISTICS
 * ----------------------------------------------------------------------------
 * - Zero Allocation: No Python tuples/dicts created during the hot path.
 * - Pointer Hashing: Hashes interned string addresses (extremely fast O(1)).
 * - C23 Dispatch: Uses 'typeof_unqual' to select type-correct converters.
 * - Bitmask Validation: 'Required' args checked via 1 instruction (AND/CMP).
 * - Multi-Signature Reuse: One Schema can power many functions (Force/Torque).
 * ============================================================================
 */

#define FP_EMPTY_SLOT 0xFFFF

/** --- 1. TYPES & STRUCTS --- **/

typedef struct {
  const char *name;
  PyObject *interned;
  void (*convert)(PyObject *, void *);
  bool required;
} FastArgSpec;

typedef struct {
  FastArgSpec *specs;
  uint16_t *lookup_table;
  size_t count;
  size_t table_mask;
  uint64_t required_mask;
} FastParser;

/** --- 2. CONVERTER DISPATCH (Header-only for Inlining) --- **/

static inline void fp_conv_float(PyObject *o, void *t) {
  *(float *)t = (float)PyFloat_AsDouble(o);
}
static inline void fp_conv_double(PyObject *o, void *t) {
  *(double *)t = PyFloat_AsDouble(o);
}
static inline void fp_conv_int(PyObject *o, void *t) {
  *(int *)t = (int)PyLong_AsLong(o);
}
static inline void fp_conv_u32(PyObject *o, void *t) {
  *(uint32_t *)t = (uint32_t)PyLong_AsUnsignedLongMask(o);
}
static inline void fp_conv_u64(PyObject *o, void *t) {
  *(uint64_t *)t = PyLong_AsUnsignedLongLong(o);
}
static inline void fp_conv_bool(PyObject *o, void *t) {
  *(bool *)t = (bool)PyObject_IsTrue(o);
}
static inline void fp_conv_pyobj(PyObject *o, void *t) { *(PyObject **)t = o; }

#define FP_GET_CONVERTER(T)                                                    \
  _Generic((T),                                                                \
      float: fp_conv_float,                                                    \
      double: fp_conv_double,                                                  \
      int: fp_conv_int,                                                        \
      uint32_t: fp_conv_u32,                                                   \
      uint64_t: fp_conv_u64,                                                   \
      bool: fp_conv_bool,                                                      \
      PyObject *: fp_conv_pyobj)

#define FP_ARG(name_str, var)                                                  \
  {.name = (name_str),                                                         \
   .convert = FP_GET_CONVERTER((typeof_unqual(var)){0}),                       \
   .required = false}

#define FP_REQ_ARG(name_str, var)                                              \
  {.name = (name_str),                                                         \
   .convert = FP_GET_CONVERTER((typeof_unqual(var)){0}),                       \
   .required = true}

/** --- 3. EXTERN DECLARATIONS (Cold Paths in .c) --- **/

extern bool fp_report_missing(const FastParser *fp, uint64_t provided_mask);
extern bool fp_report_multiple(const FastParser *fp, size_t index);
extern bool fp_report_too_many(const FastParser *fp, Py_ssize_t nargs);
extern void fp_init_impl(FastParser *fp, FastArgSpec *specs, size_t count);
extern bool fp_parse_legacy(PyObject *args, PyObject *kwargs,
                            const FastParser *fp, void **targets, size_t dummy);

/** --- 4. THE HOT PATH (Inlined Vectorcall Engine) --- **/

static inline size_t fp_hash_ptr(PyObject *ptr, size_t mask) {
  uintptr_t v = (uintptr_t)ptr;
  return ((v >> 4) ^ (v >> 10)) & mask;
}

CULV_NODISCARD
static inline bool fp_parse_vector(PyObject *const *args, Py_ssize_t nargs,
                                   PyObject *kwnames, const FastParser *fp,
                                   void **targets) {
  uint64_t provided_mask = 0;
  const size_t count = fp->count;
  const FastArgSpec *specs = fp->specs;

  // 1. Check if we have more positional args than the spec allows
  if (UNLIKELY(nargs > (Py_ssize_t)count)) {
    return fp_report_too_many(fp, nargs);
  }

  // 2. Positional Logic
  for (Py_ssize_t i = 0; i < nargs; ++i) {
    provided_mask |= (1ULL << i);
    // Removed Py_None check: Let the converter (e.g. PyFloat_AsDouble)
    // handle None by raising a TypeError, matching standard Python behavior.
    specs[i].convert(args[i], targets[i]);
  }

  // 3. Keywords Logic
  if (kwnames) {
    Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
    PyObject *const *kw_vals = args + nargs;
    for (Py_ssize_t i = 0; i < nkw; ++i) {
      PyObject *key = PyTuple_GET_ITEM(kwnames, i);
      size_t idx = FP_EMPTY_SLOT;

      // Lookup logic (Table or Linear)
      if (fp->lookup_table) {
        size_t h = fp_hash_ptr(key, fp->table_mask);
        while (fp->lookup_table[h] != FP_EMPTY_SLOT) {
          size_t candidate = fp->lookup_table[h];
          if (specs[candidate].interned == key ||
              PyUnicode_Compare(key, specs[candidate].interned) == 0) {
            idx = candidate;
            break;
          }
          h = (h + 1) & fp->table_mask;
        }
      } else {
        for (size_t j = 0; j < count; ++j) {
          if (key == specs[j].interned ||
              PyUnicode_Compare(key, specs[j].interned) == 0) {
            idx = j;
            break;
          }
        }
      }

      if (idx == FP_EMPTY_SLOT) {
        PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
        return false;
      }

      // Robustness: Check if this keyword was already filled positionally
      if (UNLIKELY(provided_mask & (1ULL << idx))) {
        return fp_report_multiple(fp, idx);
      }

      provided_mask |= (1ULL << idx);
      specs[idx].convert(kw_vals[i], targets[idx]);
    }
  }

  // 4. Final Required Check
  if (UNLIKELY((provided_mask & fp->required_mask) != fp->required_mask)) {
    return fp_report_missing(fp, provided_mask);
  }
  return (PyErr_Occurred() == NULL);
}

/** --- 5. PUBLIC MACROS --- **/

#define FastParse_Unified(arg1, arg2, arg3, arg4, arg5)                        \
  _Generic((arg1),                                                             \
      PyObject *const *: fp_parse_vector,                                      \
      PyObject *: fp_parse_legacy)(arg1, arg2, arg3, arg4, arg5)

#define FastParse_Init(fp, specs, count)                                       \
  do {                                                                         \
    static_assert((count) <= 64,                                               \
                  "FastParse only supports up to 64 arguments");               \
    fp_init_impl(fp, specs, count);                                            \
  } while (0)