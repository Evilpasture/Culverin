#pragma once

#include "culverin_compiler_specifics.h"
#include "culverin_types.h"
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

typedef struct {
    const char* parser_name;
    FastArgSpec *specs;
    uint16_t *lookup_table;
    size_t count;
    size_t table_mask;
    uint64_t required_mask;
    uint64_t type_guard_mask;
} FastParser;

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
    return ((v >> 4) ^ (v >> 10)) & mask;
}

CULV_NODISCARD
static inline bool fp_parse_vector(PyObject *const *CULV_RESTRICT args, Py_ssize_t nargs, PyObject *CULV_RESTRICT kwnames,
                                   const FastParser *CULV_RESTRICT fp, void *CULV_RESTRICT *CULV_RESTRICT targets) {
    uint64_t provided_mask   = 0;
    const uint64_t tg_mask   = fp->type_guard_mask; // Load once to register
    const size_t count       = fp->count;
    const FastArgSpec *specs = fp->specs;

    // 1. Validate Positional Count
    if (UNLIKELY(nargs > (Py_ssize_t)count)) {
        return fp_report_too_many(fp, nargs);
    }

    // 2. Positional Logic
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject *val = args[i];
        const FastArgSpec *spec = &specs[i];

        // O(1) Bitwise check. If tg_mask is 0, this is naturally skipped.
        if (UNLIKELY(tg_mask & (1ULL << i))) {
            if (UNLIKELY(!Py_IS_TYPE(val, spec->type_guard) && 
                         !PyObject_TypeCheck(val, spec->type_guard))) {
                return fp_report_type_error(fp, i, val);
            }
        }

        if (UNLIKELY(!spec->convert(val, targets[i]))) {
            return false; // Convert sets its own ValueError if needed
        }
        provided_mask |= (1ULL << i);
    }

    // 3. Keywords Logic
    if (kwnames) {
        Py_ssize_t nkw           = PyTuple_GET_SIZE(kwnames);
        PyObject *const *kw_vals = args + nargs;

        for (Py_ssize_t i = 0; i < nkw; ++i) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            size_t idx    = FP_EMPTY_SLOT;

            // Fast Path: O(1) Hash Table Lookup
            if (fp->lookup_table) {
                size_t h = fp_hash_ptr(key, fp->table_mask);
                while (fp->lookup_table[h] != FP_EMPTY_SLOT) {
                    size_t candidate = fp->lookup_table[h];
                    // Pointer comparison only (assumes interned strings)
                    if (LIKELY(specs[candidate].interned == key)) {
                        idx = candidate;
                        break;
                    }
                    h = (h + 1) & fp->table_mask;
                }
            }

            // Slow Path: Linear fallback for small schemas OR un-interned string keys
            if (UNLIKELY(idx == FP_EMPTY_SLOT)) {
                for (size_t j = 0; j < count; ++j) {
                    if (specs[j].interned == key || 
                        PyUnicode_Compare(key, specs[j].interned) == 0) {
                        idx = j;
                        break;
                    }
                }
            }

            // Keyword Validation
            if (UNLIKELY(idx == FP_EMPTY_SLOT)) {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument '%U'", key);
                return false;
            }

            if (UNLIKELY(provided_mask & (1ULL << idx))) {
                return fp_report_multiple(fp, idx);
            }

            // Type Guard & Conversion
            PyObject *val = kw_vals[i];
            const FastArgSpec *spec = &specs[idx];

            // Type Guard Validation
            if (UNLIKELY(tg_mask & (1ULL << idx))) {
                if (UNLIKELY(!Py_IS_TYPE(val, spec->type_guard) && 
                             !PyObject_TypeCheck(val, spec->type_guard))) {
                    return fp_report_type_error(fp, idx, val);
                }
            }

            if (UNLIKELY(!spec->convert(val, targets[idx]))) {
                return false;
            }
            provided_mask |= (1ULL << idx);
        }
    }

    // 4. Final Required Check
    if (UNLIKELY((provided_mask & fp->required_mask) != fp->required_mask)) {
        return fp_report_missing(fp, provided_mask);
    }

    // We trust our internal boolean returns. No need for the expensive TLS PyErr_Occurred()
    return true; 
}

/** --- 5. PUBLIC MACROS --- **/

// Declare (but never define) a function with a name that explains the error
void ERROR_FastParse_First_Arg_Must_Be_PyObject_Ptr_Or_Vectorcall_Ptr(void);

#define FastParse_Unified(arg1, arg2, arg3, arg4, arg5)                                            \
    _Generic((arg1),                                                                               \
        PyObject *const *: fp_parse_vector,                                                        \
        PyObject **: fp_parse_vector,                                                              \
        PyObject *: fp_parse_legacy,                                                               \
        default: ERROR_FastParse_First_Arg_Must_Be_PyObject_Ptr_Or_Vectorcall_Ptr)(                \
        (arg1), (arg2), (arg3), (arg4), (arg5))

#define FastParse_Init(fp, specs, count)                                                           \
    do {                                                                                           \
        static_assert((count) <= 64, "FastParse only supports up to 64 arguments");                \
        fp_init_impl(fp, specs, count);                                                            \
    } while (0)
