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
 * extensions. Replaces PyArg_ParseTupleAndKeywords with a hybrid system that
 * uses Linear Search for small arg counts and Pointer Hashing for large ones.
 *
 * ----------------------------------------------------------------------------
 * 1. THE SETUP (Module Level)
 * ----------------------------------------------------------------------------
 * Define a global Parser and its Specs.
 *
 *   static FastParser BodyParser;
 *   static FastArgSpec BodySpecs[14];
 *
 *   void init_my_parser() {
 *       float f; int i; PyObject *o; // Dummies for type deduction
 *       FastArgSpec temp[] = {
 *           FP_REQ_ARG("pos", o),   // REQUIRED (index 0)
 *           FP_ARG("mass", f),      // OPTIONAL (index 1)
 *           FP_REQ_ARG("shape", i)  // REQUIRED (index 2)
 *       };
 *       memcpy(BodySpecs, temp, sizeof(temp));
 *
 *       // FastParse_Init:
 *       // - Interns strings for pointer equality
 *       // - Calculates bitmask for REQUIRED arguments
 *       // - Builds O(1) Pointer-Hash-Table if count > 10
 *       FastParse_Init(&BodyParser, BodySpecs, 14);
 *   }
 *
 * ----------------------------------------------------------------------------
 * 2. THE USAGE (Function Level)
 * ----------------------------------------------------------------------------
 * Use the FastParse_Unified macro. It branches between Vectorcall and Legacy.
 *
 *   static PyObject* my_func(PyObject* self, PyObject* const* args,
 *                            size_t nargsf, PyObject* kwnames) {
 *       auto nargs = PyVectorcall_NARGS(nargsf);
 *
 *       // 1. Define C variables with default values
 *       PyObject *o_pos = nullptr; float mass = 1.0f; int shape = 0;
 *
 *       // 2. Target Array - MUST MATCH SPEC ORDER EXACTLY
 *       void *targets[] = { &o_pos, &mass, &shape };
 *       static_assert(sizeof(targets)/sizeof(void*) == 3);
 *
 *       // 3. Fast Parse (5-argument signature)
 *       if (!FastParse_Unified(args, nargs, kwnames, &BodyParser, targets))
 *           return nullptr;
 *
 *       // ... logic ...
 *   }
 *
 * ----------------------------------------------------------------------------
 * 3. CRITICAL INVARIANTS (The "Don't Crash" Rules)
 * ----------------------------------------------------------------------------
 *
 * INVARIANT A: ORDER SYNC
 * The index of a key in 'BodySpecs' MUST correspond to the index of its
 * pointer in the 'targets' array. Break this, and you corrupt C memory.
 *
 * INVARIANT B: TYPE SYNC
 * The type used in FP_ARG/FP_REQ_ARG MUST match the actual C type of the
 * target variable. The parser uses C23 typeof_unqual to select converters.
 *
 * INVARIANT C: INITIALIZATION
 * FastParse_Init() MUST be called in the module exec/init phase. If
 * interned pointers or the lookup table are uninitialized, lookups will fail.
 *
 * INVARIANT D: BITMASK LIMIT
 * This system supports up to 64 arguments per function (uint64_t mask).
 *
 * ----------------------------------------------------------------------------
 * 4. PERFORMANCE CHARACTERISTICS
 * ----------------------------------------------------------------------------
 * - Small Functions (<= 10 args): Uses cache-friendly linear pointer search.
 * - Large Functions (> 10 args): Uses O(1) Linear Probing Hash Table.
 * - Hashing: Hashes the memory address of interned PyObjects (extremely fast).
 * - Required Check: Validated via a single bitwise AND/CMP instruction.
 * - Allocations: Zero heap allocations during the function call.
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
extern void fp_init_impl(FastParser *fp, FastArgSpec *specs, size_t count);
extern bool fp_parse_legacy(PyObject *args, PyObject *kwargs,
                            const FastParser *fp, void **targets, size_t dummy);

/** --- 4. THE HOT PATH (Inlined Vectorcall Engine) --- **/

static inline size_t fp_hash_ptr(PyObject *ptr, size_t mask) {
  uintptr_t v = (uintptr_t)ptr;
  return ((v >> 4) ^ (v >> 10)) & mask;
}

[[nodiscard]]
static inline bool fp_parse_vector(PyObject *const *args, Py_ssize_t nargs,
                                   PyObject *kwnames, const FastParser *fp,
                                   void **targets) {
  uint64_t provided_mask = 0;
  const size_t count = fp->count;
  const FastArgSpec *specs = fp->specs;

  // Positional
  size_t pos_limit = ((size_t)nargs < count) ? (size_t)nargs : count;
  for (size_t i = 0; i < pos_limit; ++i) {
    provided_mask |= (1ULL << i);
    if (args[i] != Py_None) {
      specs[i].convert(args[i], targets[i]);
    } else if (specs[i].required) {
      return fp_report_missing(fp, provided_mask & ~(1ULL << i));
    }
  }

  // Keywords
  if (kwnames) {
    Py_ssize_t nkw = PyTuple_GET_SIZE(kwnames);
    PyObject *const *kw_vals = args + nargs;
    for (Py_ssize_t i = 0; i < nkw; ++i) {
      PyObject *key = PyTuple_GET_ITEM(kwnames, i);
      if (fp->lookup_table) {
        size_t h = fp_hash_ptr(key, fp->table_mask);
        while (fp->lookup_table[h] != FP_EMPTY_SLOT) {
          size_t idx = fp->lookup_table[h];
          // FIX: Check pointer first (fast), then fall back to string compare
          // (safe)
          if (specs[idx].interned == key ||
              PyUnicode_Compare(key, specs[idx].interned) == 0) {
            provided_mask |= (1ULL << idx);
            if (kw_vals[i] != Py_None) {
              specs[idx].convert(kw_vals[i], targets[idx]);
            } else if (specs[idx].required) {
              return fp_report_missing(fp, provided_mask & ~(1ULL << idx));
            }
            goto next_kw;
          }
          h = (h + 1) & fp->table_mask;
        }
      } else {
        for (size_t j = 0; j < count; ++j) {
          // FIX: Fallback here too
          if (key == specs[j].interned ||
              PyUnicode_Compare(key, specs[j].interned) == 0) {
            provided_mask |= (1ULL << j);
            if (kw_vals[i] != Py_None) {
              specs[j].convert(kw_vals[i], targets[j]);
            } else if (specs[j].required) {
              return fp_report_missing(fp, provided_mask & ~(1ULL << j));
            }
            goto next_kw;
          }
        }
      }
      PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
      return false;
    next_kw:;
    }
  }

  if (UNLIKELY((provided_mask & fp->required_mask) != fp->required_mask)) {
    return fp_report_missing(fp, provided_mask);
  }
  return true;
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