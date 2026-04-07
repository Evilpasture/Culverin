#pragma once

#include "culverin_compiler_specifics.h"
#include "culverin_types.h"
#include <Python.h>
#include <stdbool.h>
#include <stdint.h>

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
 *       uint64_t h; float x, y, z;
 *       void *targets[Vec3_COUNT];
 *       targets[IDX_V3_H] = &h;
 *       // ... map targets ...
 *
 *       // Pass the interpreter-local parser address
 *       if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames,
 *                              &st->parsers.ForceParser, targets))
 *           return nullptr;
 *   }
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

constexpr int FP_EMPTY_SLOT = 0xFFFF;

/** --- 1. TYPES & STRUCTS --- **/

typedef struct {
    const char *name;
    const char *type_name;
    PyObject *interned;
    bool (*convert)(PyObject *, void *);
    PyTypeObject *type_guard;
    bool required;
} FastArgSpec;

// 1. Forward declare the struct tag and the typedef simultaneously
typedef struct FastParser FastParser;

// 2. Now the compiler knows 'FastParser' is a valid type name.
// Even though it doesn't know the size of the struct yet,
// it knows how to handle a POINTER to it.
typedef bool (*FastParseFunc)(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                              PyObject *CULV_RESTRICT kwnames, const FastParser *CULV_RESTRICT fp,
                              void *CULV_RESTRICT *CULV_RESTRICT targets);

// 3. Define the actual body
struct FastParser {
    const char *parser_name;
    FastArgSpec *specs;
    uint16_t *lookup_table;
    FastParseFunc hot_path; // Validated: FastParseFunc is known
    size_t count;
    size_t table_mask;
    uint64_t required_mask;
    uint64_t type_guard_mask;
};

// 1. Forward declaration of the generic fallback
CULV_NODISCARD
static inline bool fp_parse_vector(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                                   PyObject *CULV_RESTRICT kwnames,
                                   const FastParser *CULV_RESTRICT fp,
                                   void *CULV_RESTRICT *CULV_RESTRICT targets);

// 2. The Speculative Stubs
CULV_NODISCARD static inline bool fp_speculate_p0(PyObject *const *CULV_RESTRICT args,
                                                  Py_ssize_t nargs, PyObject *CULV_RESTRICT kwnames,
                                                  const FastParser *CULV_RESTRICT fp,
                                                  void *CULV_RESTRICT *CULV_RESTRICT targets) {
    if (LIKELY(nargs == 0 && kwnames == nullptr)) {
        return true;
    }
    return fp_parse_vector(args, nargs, kwnames, fp, targets);
}

CULV_NODISCARD static inline bool
fp_speculate_p1_naked(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                      PyObject *CULV_RESTRICT kwnames, const FastParser *CULV_RESTRICT fp,
                      void *CULV_RESTRICT *CULV_RESTRICT targets) {
    if (LIKELY(nargs == 1 && kwnames == nullptr)) {
        return fp->specs[0].convert(args[0], targets[0]);
    }
    return fp_parse_vector(args, nargs, kwnames, fp, targets);
}

CULV_NODISCARD static inline bool
fp_speculate_p2_naked(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                      PyObject *CULV_RESTRICT kwnames, const FastParser *CULV_RESTRICT fp,
                      void *CULV_RESTRICT *CULV_RESTRICT targets) {
    if (LIKELY(nargs == 2 && kwnames == nullptr)) {
        return (fp->specs[0].convert(args[0], targets[0]) &&
                fp->specs[1].convert(args[1], targets[1])) != 0;
    }
    return fp_parse_vector(args, nargs, kwnames, fp, targets);
}

CULV_NODISCARD static inline bool
fp_speculate_p3_naked(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                      PyObject *CULV_RESTRICT kwnames, const FastParser *CULV_RESTRICT fp,
                      void *CULV_RESTRICT *CULV_RESTRICT targets) {
    if (LIKELY(nargs == 3 && kwnames == nullptr)) {
        return (fp->specs[0].convert(args[0], targets[0]) &&
                fp->specs[1].convert(args[1], targets[1]) &&
                fp->specs[2].convert(args[2], targets[2])) != 0;
    }
    return fp_parse_vector(args, nargs, kwnames, fp, targets);
}

CULV_NODISCARD static inline bool
fp_speculate_p4_naked(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                      PyObject *CULV_RESTRICT kwnames, const FastParser *CULV_RESTRICT fp,
                      void *CULV_RESTRICT *CULV_RESTRICT targets) {
    if (LIKELY(nargs == 4 && kwnames == nullptr)) {
        return (fp->specs[0].convert(args[0], targets[0]) &&
                fp->specs[1].convert(args[1], targets[1]) &&
                fp->specs[2].convert(args[2], targets[2]) &&
                fp->specs[3].convert(args[3], targets[3])) != 0;
    }
    return fp_parse_vector(args, nargs, kwnames, fp, targets);
}

#include "culverin_parsers.h"
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_vec3f(PyObject *o, void *t) {
    Vec3f *v = (Vec3f *)t;
    return parse_vec3_f32(o, &v->x, &v->y, &v->z) != 0;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_pos_stride(PyObject *o, void *t) {
    PosStride *v = (PosStride *)t;
    return parse_vec3_r64(o, &v->x, &v->y, &v->z) != 0;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_aux_stride(PyObject *o, void *t) {
    AuxStride *v = (AuxStride *)t;
    return parse_quat_f32(o, &v->x, &v->y, &v->z, &v->w) != 0;
}

/** --- 2. CONVERTER DISPATCH (Header-only for Inlining) --- **/
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_float(PyObject *o, void *t) {
    if (UNLIKELY(o == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "float argument cannot be None");
        return false;
    }
    double v = PyFloat_AsDouble(o);
    if (UNLIKELY(v == -1.0 && PyErr_Occurred())) {
        return false;
    }
    *(float *)t = (float)v;
    return true;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_double(PyObject *o, void *t) {
    if (UNLIKELY(o == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "double argument cannot be None");
        return false;
    }
    double v = PyFloat_AsDouble(o);
    if (UNLIKELY(v == -1.0 && PyErr_Occurred())) {
        return false;
    }
    *(double *)t = v;
    return true;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_int(PyObject *o, void *t) {
    long v = PyLong_AsLong(o);
    if (UNLIKELY(v == -1 && PyErr_Occurred())) {
        return false;
    }
    *(int *)t = (int)v;
    return true;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_u32(PyObject *o, void *t) {
    unsigned long v = PyLong_AsUnsignedLongMask(o);
    if (UNLIKELY(PyErr_Occurred())) {
        return false;
    }
    *(uint32_t *)t = (uint32_t)v;
    return true;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_u64(PyObject *o, void *t) {
    unsigned long long v = PyLong_AsUnsignedLongLong(o);
    if (UNLIKELY(PyErr_Occurred())) {
        return false;
    }
    *(uint64_t *)t = (uint64_t)v;
    return true;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_bool(PyObject *o, void *t) {
    int v = PyObject_IsTrue(o);
    if (UNLIKELY(v == -1)) {
        return false;
    }
    *(bool *)t = (bool)v;
    return true;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_pyobj(PyObject *o, void *t) {
    *(PyObject **)t = o;
    return true;
}

#define FP_GET_CONVERTER(T)                                                                        \
    _Generic((T),                                                                                  \
        float: fp_conv_float,                                                                      \
        double: fp_conv_double,                                                                    \
        int: fp_conv_int,                                                                          \
        uint32_t: fp_conv_u32,                                                                     \
        uint64_t: fp_conv_u64,                                                                     \
        bool: fp_conv_bool,                                                                        \
        PyObject *: fp_conv_pyobj,                                                                 \
        Vec3f: fp_conv_vec3f,          /* New */                                                   \
        PosStride: fp_conv_pos_stride, /* New */                                                   \
        AuxStride: fp_conv_aux_stride  /* New */                                                   \
    )

#define FP_ARG(name_str, var)                                                                      \
    {.name = (name_str), .convert = FP_GET_CONVERTER((typeof_unqual(var)){0}), .required = false}

#define FP_REQ_ARG(name_str, var)                                                                  \
    {.name = (name_str), .convert = FP_GET_CONVERTER((typeof_unqual(var)){0}), .required = true}

/** --- 3. EXTERN DECLARATIONS (Cold Paths in .c) --- **/

extern bool fp_report_type_error(const FastParser *fp, size_t index, PyObject *val);
extern bool fp_report_missing(const FastParser *fp, uint64_t provided_mask);
extern bool fp_report_multiple(const FastParser *fp, size_t index);
extern bool fp_report_too_many(const FastParser *fp, Py_ssize_t nargs);
extern void fp_init_impl(FastParser *fp, FastArgSpec *specs, size_t count);
extern void fp_deinit(FastParser *fp);
extern bool fp_parse_legacy(PyObject *args, PyObject *kwargs, PyObject *unused,
                            const FastParser *fp, void **targets);

/** --- 4. THE HOT PATH (Inlined Vectorcall Engine) --- **/

static inline size_t fp_hash_ptr(PyObject *ptr, size_t mask) {
    auto v = (uintptr_t)ptr;
    // Golden ratio multiplier spreads pointer bits more effectively than shifts alone
    return ((v * 11400714819323198485ULL) >> 32) & mask;
}

CULV_NODISCARD
static inline bool fp_parse_vector(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs,
                                   PyObject *CULV_RESTRICT kwnames,
                                   const FastParser *CULV_RESTRICT fp,
                                   void *CULV_RESTRICT *CULV_RESTRICT targets) {
    uint64_t provided_mask   = 0;
    const uint64_t tg_mask   = fp->type_guard_mask;
    const size_t count       = fp->count;
    const FastArgSpec *specs = fp->specs;

    // 1. Validate Positional Count
    if (UNLIKELY(nargs > (Py_ssize_t)count)) {
        return fp_report_too_many(fp, nargs);
    }

    // 2. Speculative Positional Logic
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject *val = args[i];

        if (UNLIKELY(tg_mask & (1ULL << i))) {
            if (UNLIKELY(!Py_IS_TYPE(val, specs[i].type_guard))) {
                if (!PyObject_TypeCheck(val, specs[i].type_guard)) {
                    return fp_report_type_error(fp, i, val);
                }
            }
        }

        if (UNLIKELY(!specs[i].convert(val, targets[i]))) {
            return false;
        }
    }

    // Bulk bitmask generation for positional arguments
    provided_mask = (nargs >= 64) ? ~(uint64_t)0 : ((1ULL << (nargs & 63)) - 1);

    // 3. Speculative Keywords Logic
    if (kwnames) {
        const Py_ssize_t nkw     = PyTuple_GET_SIZE(kwnames);
        PyObject *const *kw_vals = args + nargs;
        const uint16_t *ltable   = fp->lookup_table;
        const size_t t_mask      = fp->table_mask;

        for (Py_ssize_t i = 0; i < nkw; ++i) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            size_t idx    = FP_EMPTY_SLOT;

            // BRANCH 1: Fast Path (Hash Table exists)
            if (ltable) {
                size_t h         = fp_hash_ptr(key, t_mask);
                size_t candidate = ltable[h];

                if (LIKELY(candidate != FP_EMPTY_SLOT && specs[candidate].interned == key)) {
                    idx = candidate;
                } else {
                    // Collision resolution
                    while (ltable[h] != FP_EMPTY_SLOT) {
                        if (specs[ltable[h]].interned == key) {
                            idx = ltable[h];
                            break;
                        }
                        h = (h + 1) & t_mask;
                    }
                }
            }

            // BRANCH 2: Fallback (No table OR hash miss/un-interned key)
            if (UNLIKELY(idx == FP_EMPTY_SLOT)) {
                for (size_t j = 0; j < count; ++j) {
                    // Check positional-only boundary if you have them,
                    // otherwise linear scan all interned pointers.
                    if (specs[j].interned == key ||
                        PyUnicode_Compare(key, specs[j].interned) == 0) {
                        idx = j;
                        break;
                    }
                }
            }

            // Validation logic
            if (UNLIKELY(idx == FP_EMPTY_SLOT)) {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
                return false;
            }

            if (UNLIKELY(provided_mask & (1ULL << idx))) {
                return fp_report_multiple(fp, idx);
            }

            PyObject *val = kw_vals[i];

            if (UNLIKELY(tg_mask & (1ULL << idx))) {
                if (UNLIKELY(!Py_IS_TYPE(val, specs[idx].type_guard))) {
                    if (!PyObject_TypeCheck(val, specs[idx].type_guard)) {
                        return fp_report_type_error(fp, idx, val);
                    }
                }
            }

            if (UNLIKELY(!specs[idx].convert(val, targets[idx]))) {
                return false;
            }
            provided_mask |= (1ULL << idx);
        }
    }

    // 4. Final Required Check
    if (UNLIKELY((provided_mask & fp->required_mask) != fp->required_mask)) {
        return fp_report_missing(fp, provided_mask);
    }

    return true;
}

/** --- 5. PUBLIC MACROS --- **/

// Declare (but never define) a function with a name that explains the error
void ERROR_FastParse_First_Arg_Must_Be_PyObject_Ptr_Or_Vectorcall_Ptr(void);

#define FastParse_Unified(arg1, arg2, arg3, arg4, arg5)                                            \
    _Generic((arg1),                                                                               \
        PyObject *const *: (arg4)->hot_path,                                                       \
        PyObject **: (arg4)->hot_path,                                                             \
        PyObject *: fp_parse_legacy,                                                               \
        default: ERROR_FastParse_First_Arg_Must_Be_PyObject_Ptr_Or_Vectorcall_Ptr)(                \
        (arg1), (arg2), (arg3), (arg4), (arg5))

#define FastParse_Init(fp, specs, count)                                                           \
    do {                                                                                           \
        static_assert((count) <= 64, "FastParse only supports up to 64 arguments");                \
        fp_init_impl(fp, specs, count);                                                            \
    } while (0)
