#pragma once

#include "bindings/PythonCommon.hpp"
#include "bindings/parsing/ParseError.hpp"

#include <cstddef>
#include <cstdint>

namespace CPH::Python {

/**
 * ============================================================================
 * FAST PARSE ENGINE
 * ============================================================================
 * A high-performance, zero-allocation argument parsing system for Python C
 * extensions. Optimized for PEP 703 (Free-threading) and PEP 489 (Multi-phase
 * initialization).
 * ============================================================================
 */

static constexpr int kFastParseEmptySlot = 0xFFFF;
/** --- 1. TYPES & STRUCTS --- **/

struct FastArgSpecHot {
    PyObject *interned;                        // For keyword matching
    PyTypeObject *type_guard;                  // For type checking
    bool (*convert)(PyObject *, const void *); // The workhorse
};                          // 24 bytes

struct FastArgSpecCold {
    const char *name;
    const char *type_name;
}; // 16 bytes

struct FastParserCore;

// The engine reports failures through ParseResult; the *converters* stay
// `bool (PyObject *, const void *)` because they write into the caller's
// target slot — the write-in-place contract is what keeps the hot path free
// of copies. A converter reports why it failed via the ambient PyErr, which
// the engine captures into a ParseError immediately (see CaptureError()).
using FastParseFunc = ParseResult (*)(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                                     const FastParserCore *fastparser, const void *const *targets);

static constexpr auto kFastParseAlignmentSize = 64;
struct FastArgDef {
    const char *name;
    const char *type_name;
    PyTypeObject *type_guard;
    bool (*convert)(PyObject *, const void *);
    bool required;
};

struct FastParserCore {
    alignas(kFastParseAlignmentSize) const char *parser_name;
    FastArgSpecHot *hot_specs;
    FastArgSpecCold *cold_specs;
    uint16_t *lookup_table;
    FastParseFunc hot_path;
    size_t count;
    size_t table_mask;
    uint64_t required_mask;
    uint64_t type_guard_mask;
    uint64_t warned_mask;
};

static_assert(sizeof(struct FastParserCore) == static_cast<size_t>(kFastParseAlignmentSize) * 2);
static_assert(alignof(struct FastParserCore) == kFastParseAlignmentSize);

// Forward declaration of the generic fallback
[[nodiscard]] static inline ParseResult fast_parse_vector(
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject *kwnames,
    const FastParserCore *fastparser,
    const void *const *targets);

/** --- 2. EXTERN DECLARATIONS --- **/
// Diagnostics build the message first (exactly as before B3) and then hand it
// back as an owned ParseError instead of leaving it in the thread state.
[[gnu::noinline, nodiscard]]
extern ParseError FastParseReportTypeError(const FastParserCore *fastparser, size_t index,
                                           PyObject *val);
[[gnu::noinline, nodiscard]]
extern ParseError FastParseReportMissing(const FastParserCore *fastparser, uint64_t provided_mask);
[[gnu::noinline, nodiscard]]
extern ParseError FastParseReportMultiple(const FastParserCore *fastparser, size_t index);
[[gnu::noinline, nodiscard]]
extern ParseError FastParseReportTooMany(const FastParserCore *fastparser, Py_ssize_t nargs);
[[gnu::noinline, nodiscard]]
extern ParseError FastParseReportUnknownKeyword(const FastParserCore *fastparser, PyObject *key);
[[gnu::noinline, nodiscard]]
extern ParseError FastParseReportConversionError(const FastParserCore *fastparser, size_t index,
                                                 PyObject *val);
extern void FastParseInitImpl(FastParserCore *fastparser, const FastArgDef *defs,
                                  size_t count);
extern void FastParseDeinit(FastParserCore *fastparser);
[[nodiscard]] extern ParseResult FastParseLegacy(
    PyObject *args,
    PyObject *kwargs,
    [[maybe_unused]] PyObject *unused,
    const FastParserCore *fastparser,
    const void *const *targets);

/** --- 3. SPECULATIVE STUBS (Monomorphic fast-paths) --- **/

[[gnu::always_inline]]
static inline bool fast_parse_dispatch_conv(bool (*conv)(PyObject *, const void *),
                                            PyObject *obj, const void *target);

[[nodiscard, gnu::always_inline]]
static inline ParseResult fast_parse_spec_check_and_conv(const FastArgSpecHot *spec,
                                                         std::size_t index, PyObject *value,
                                                         const void *target,
                                                         const FastParserCore *fastparser) {
    if (spec->type_guard) {
        if (Likely(Py_TYPE(value) != spec->type_guard)) {
            if (Unlikely(!PyObject_TypeCheck(value, spec->type_guard))) {
                return std::unexpected(FastParseReportTypeError(fastparser, index, value));
            }
        }
    }
    if (Unlikely(!fast_parse_dispatch_conv(spec->convert, value, target))) {
        // Deliberately undecorated: the monomorphic stubs have always returned
        // the converter's raw error, while only fast_parse_process_pos adds the
        // signature context. Unifying them would change the diagnostics these
        // hot-path calls produce, which is a behaviour change rather than a
        // refactor -- flagged for a follow-up, not done here.
        return CaptureFailure();
    }
    return {};
}

[[gnu::noinline]] static inline ParseResult
fast_parse_fallback(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                    const FastParserCore *fastparser, const void *const *targets) {
    return fast_parse_vector(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p0(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 0 && kwnames == nullptr)) {
        return {};
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p1(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 1 && kwnames == nullptr)) {
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[0], 0,
                                              args[0], targets[0], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p2(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 2 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[1], 1,
                                              args[1], targets[1], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p3(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 3 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[2], 2,
                                              args[2], targets[2], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p4(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 4 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[3], 3,
                                              args[3], targets[3], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p5(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 5 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[4], 4,
                                              args[4], targets[4], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p6(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 6 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[4], 4, args[4], targets[4], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[5], 5,
                                              args[5], targets[5], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p7(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 7 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[4], 4, args[4], targets[4], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[5], 5, args[5], targets[5], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[6], 6,
                                              args[6], targets[6], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
fast_parse_speculate_p8(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (Likely(nargs == 8 && kwnames == nullptr)) {
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[4], 4, args[4], targets[4], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[5], 5, args[5], targets[5], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = fast_parse_spec_check_and_conv(
                &fastparser->hot_specs[6], 6, args[6], targets[6], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return fast_parse_spec_check_and_conv(&fastparser->hot_specs[7], 7,
                                              args[7], targets[7], fastparser);
    }
    return fast_parse_fallback(args, nargs, kwnames, fastparser, targets);
}

/** --- 3. CONVERTER DISPATCH --- **/

template <typename T>
[[nodiscard]] static inline T *fast_parse_target(const void *target) noexcept {
    return static_cast<T *>(const_cast<void *>(target));
}

[[nodiscard, gnu::always_inline]] static inline bool
fast_parse_conv_float(PyObject *obj, const void *target) {
    // HOT PATH: Exact float match (Pointer comparison)
    if (PyFloat_CheckExact(obj)) {
        *fast_parse_target<float>(target) =
            static_cast<float>((reinterpret_cast<PyFloatObject *>(obj))->ob_fval);
        return true;
    }

    // COLD PATH: Handle ints or float subclasses
    double val = PyFloat_AsDouble(obj);
    if (Unlikely(val == -1.0 && PyErr_Occurred())) {
        return false;
    }
    *fast_parse_target<float>(target) = static_cast<float>(val);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool
fast_parse_conv_double(PyObject *obj, const void *target) {
    // HOT PATH: Exact float match
    // Bypasses the overhead of PyFloat_AsDouble (magic method checks, int conversion logic)
    if (PyFloat_CheckExact(obj)) {
        *fast_parse_target<double>(target) = (reinterpret_cast<PyFloatObject *>(obj))->ob_fval;
        return true;
    }

    // COLD PATH: Handle ints, None, or float subclasses
    if (Unlikely(obj == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "double argument cannot be None");
        return false;
    }

    double val = PyFloat_AsDouble(obj);
    if (Unlikely(val == -1.0 && PyErr_Occurred())) {
        return false;
    }
    *fast_parse_target<double>(target) = val;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool fast_parse_conv_int(PyObject *obj,
                                                                              const void *target) {
    // 1. Singletons (Pointer identity is the fastest possible check)
    if (obj == Py_True) {
        *fast_parse_target<int>(target) = 1;
        return true;
    }
    if (obj == Py_False) {
        *fast_parse_target<int>(target) = 0;
        return true;
    }

    // 2. None Check (Cold path, but keeps PyLong_AsLong clean)
    if (Unlikely(obj == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "int argument cannot be None");
        return false;
    }

    // 3. Single Conversion Call
    long val = PyLong_AsLong(obj);

    // 4. Hot Path: 99.9% of integers in physics/logic are not -1
    if (Likely(val != -1)) {
        *fast_parse_target<int>(target) = static_cast<int>(val);
        return true;
    }

    // 5. Cold Path: Was it actually -1 or a conversion error?
    if (PyErr_Occurred()) {
        return false;
    }

    *fast_parse_target<int>(target) = -1;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool fast_parse_conv_u32(PyObject *obj,
                                                                              const void *target) {
    if (obj == Py_True) {
        *fast_parse_target<uint32_t>(target) = 1;
        return true;
    }
    if (obj == Py_False) {
        *fast_parse_target<uint32_t>(target) = 0;
        return true;
    }
    if (Unlikely(obj == Py_None)) {
        PyErr_SetString(PyExc_TypeError, "uint32 argument cannot be None");
        return false;
    }

    unsigned long val = PyLong_AsUnsignedLong(obj);

    if (Likely(val != (unsigned long)-1)) {
        *fast_parse_target<uint32_t>(target) = static_cast<uint32_t>(val);
        return true;
    }

    if (PyErr_Occurred()) {
        return false;
    }

    *fast_parse_target<uint32_t>(target) = static_cast<uint32_t>(-1);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool fast_parse_conv_u64(PyObject *obj,
                                                                              const void *target) {
    if (obj == Py_True) {
        *fast_parse_target<uint64_t>(target) = 1;
        return true;
    }
    if (obj == Py_False) {
        *fast_parse_target<uint64_t>(target) = 0;
        return true;
    }
    // u64 often used for handles, which might be None
    if (obj == Py_None) {
        *fast_parse_target<uint64_t>(target) = 0;
        return true;
    }

    unsigned long long val = PyLong_AsUnsignedLongLong(obj);

    if (Likely(val != (unsigned long long)-1)) {
        *fast_parse_target<uint64_t>(target) = static_cast<uint64_t>(val);
        return true;
    }

    if (PyErr_Occurred()) {
        return false;
    }

    *fast_parse_target<uint64_t>(target) = static_cast<uint64_t>(-1);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool fast_parse_conv_bool(PyObject *obj,
                                                                               const void *target) {
    // Exact pointer check for the singletons
    if (obj == Py_True) {
        *fast_parse_target<bool>(target) = true;
        return true;
    }
    if (obj == Py_False) {
        *fast_parse_target<bool>(target) = false;
        return true;
    }

    // Fallback for truthiness of lists, strings, etc.
    int val = PyObject_IsTrue(obj);
    if (Unlikely(val == -1)) {
        return false;
    }
    *fast_parse_target<bool>(target) = static_cast<bool>(val);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool
fast_parse_conv_pyobj(PyObject *obj, const void *target) {
    *fast_parse_target<PyObject *>(target) = obj;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool fast_parse_conv_str(PyObject *obj,
                                                                              const void *target) {
    if (Unlikely(!PyUnicode_Check(obj))) {
        PyErr_Format(PyExc_TypeError, "expected str, got %s", Py_TYPE(obj)->tp_name);
        return false;
    }
    const char *str = PyUnicode_AsUTF8(obj);
    if (Unlikely(str == nullptr)) {
        return false;
    }
    *fast_parse_target<const char *>(target) = str;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool
fast_parse_conv_ssize(PyObject *obj, const void *target) {
    // HOT PATH: Exact integer match
    if (Likely(PyLong_CheckExact(obj))) {
        Py_ssize_t val = PyLong_AsSsize_t(obj);
        // Optimization: Values like buffer sizes and counts are rarely exactly -1.
        if (Likely(val != -1)) {
            *fast_parse_target<Py_ssize_t>(target) = val;
            return true;
        }
    }

    // COLD PATH: Handle None, subclasses, or the literal value -1
    if (obj == Py_None) {
        PyErr_SetString(PyExc_TypeError, "size/index argument cannot be None");
        return false;
    }

    Py_ssize_t val = PyLong_AsSsize_t(obj);
    if (val == -1 && PyErr_Occurred()) {
        return false;
    }
    *fast_parse_target<Py_ssize_t>(target) = val;
    return true;
}

[[gnu::always_inline, nodiscard]]
static inline bool fast_parse_dispatch_conv(bool (*conv)(PyObject *, const void *),
                                                 PyObject *obj, const void *target) {
    // These comparisons are just address checks (extremely fast)
    // If a match is found, the compiler inlines the specific converter.
    if (conv == fast_parse_conv_float) {
        return fast_parse_conv_float(obj, target);
    }
    if (conv == fast_parse_conv_int) {
        return fast_parse_conv_int(obj, target);
    }
    if (conv == fast_parse_conv_u64) {
        return fast_parse_conv_u64(obj, target);
    }
    if (conv == fast_parse_conv_ssize) {
        return fast_parse_conv_ssize(obj, target);
    }
    if (conv == fast_parse_conv_bool) {
        return fast_parse_conv_bool(obj, target);
    }
    if (conv == fast_parse_conv_pyobj) {
        return fast_parse_conv_pyobj(obj, target);
    }

    // Fallback for custom or less common converters
    return conv(obj, target);
}

/** --- 4. THE HOT PATH --- **/
[[gnu::always_inline, gnu::const, maybe_unused, nodiscard]]
static inline size_t fast_parse_hash_ptr(PyObject *ptr, size_t mask) {
    auto val = reinterpret_cast<uintptr_t>(ptr);
    // Golden ratio multiplier spreads pointer bits effectively
    constexpr auto gratio     = 11400714819323198485ULL;
    constexpr auto shift_bits = 32ULL;
    return ((val * gratio) >> shift_bits) & mask;
}
[[gnu::always_inline, nodiscard]]
static inline bool fast_parse_check_type_guard(
    const FastArgSpecHot *spec,
    PyObject *val) {
    PyTypeObject *guard = spec->type_guard;

    // If no guard is defined (NULL), this branch is perfectly predicted as "not taken"
    if (Likely(guard == nullptr)) {
        return true;
    }

    // HOT PATH: Exact pointer match (The most common case in Python)
    if (Likely(Py_IS_TYPE(val, guard))) {
        return true;
    }

    // COLD PATH: Inheritance check (Subclasses)
    return PyObject_TypeCheck(val, guard);
}
[[gnu::noinline, gnu::cold]]
extern bool FastParseWarnUninternedSlow(PyObject key[], const char pname[]);

/**
 * COLD FALLBACK: fast_parse_cmp_slow
 * Separated to keep the instruction cache lean for interned keys.
 */
[[gnu::noinline, gnu::cold]]
extern size_t FastParseCmpSlow(
    PyObject *key,
    const FastParserCore *fastparse);

[[gnu::always_inline, nodiscard]]
static inline size_t fast_parse_find_keyword_index(
    PyObject *key,
    const FastParserCore *fastparse) {
    const FastArgSpecHot *specs = fastparse->hot_specs;
    const uint16_t *ltable          = fastparse->lookup_table;

    if (Likely(ltable != nullptr)) {
        size_t hash = fast_parse_hash_ptr(key, fastparse->table_mask);

        // THE FIX: Probe until we hit an empty slot (table is 50% empty, so this is fast)
        // We use table_mask as a safe upper bound to prevent infinite loops.
        for (size_t i = 0; i <= fastparse->table_mask; i++) {
            size_t slot = ltable[(hash + i) & fastparse->table_mask];
            if (slot == kFastParseEmptySlot) {
                break;
            }
            if (specs[slot].interned == key) {
                return slot;
            }
        }
    } else {
        const size_t count = fastparse->count;
        for (size_t j = 0; j < count; ++j) {
            if (specs[j].interned == key) {
                return j;
            }
        }
    }

    return FastParseCmpSlow(key, fastparse);
}

[[gnu::always_inline]]
static inline ParseResult fast_parse_process_pos(
    const FastParserCore *fastparse,
    PyObject *const *args,
    Py_ssize_t nargs,
    const void *const *targets) {
#pragma unroll 4
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        // spec is 24 bytes, targets[i] is 8 bytes.
        // We are touching very little memory here.
        const FastArgSpecHot *spec = &fastparse->hot_specs[i];

        // 1. Safety check (Type Guard)
        // We use the helper to keep the "Happy Path" linear.
        if (Unlikely(!fast_parse_check_type_guard(spec, args[i]))) {
            return std::unexpected(
                FastParseReportTypeError(fastparse, static_cast<size_t>(i), args[i]));
        }

        // 2. Conversion
        if (Unlikely(!spec->convert(args[i], targets[i]))) {
            return std::unexpected(
                FastParseReportConversionError(fastparse, static_cast<size_t>(i), args[i]));
        }
    }
    return {};
}
[[gnu::always_inline, nodiscard]]
static inline ParseResult fast_parse_process_kw(
    const FastParserCore *fastparse,
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject *kwnames,
    uint64_t *mask,
    const void *const *targets) {
    if (!kwnames) {
        return {};
    }

    const Py_ssize_t nkw     = PyTuple_GET_SIZE(kwnames);
    PyObject *const *kw_vals = args + nargs;

#pragma unroll 4
    for (Py_ssize_t i = 0; i < nkw; ++i) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        size_t idx    = fast_parse_find_keyword_index(key, fastparse);

        if (Unlikely(idx == kFastParseEmptySlot)) {
            return std::unexpected(FastParseReportUnknownKeyword(fastparse, key));
        }

        if (Unlikely(*mask & (1ULL << idx))) {
            return std::unexpected(FastParseReportMultiple(fastparse, idx));
        }

        // UPDATED: Now only takes 2 arguments
        if (!fast_parse_check_type_guard(&fastparse->hot_specs[idx], kw_vals[i])) {
            return std::unexpected(FastParseReportTypeError(fastparse, idx, kw_vals[i]));
        }

        if (Unlikely(!fastparse->hot_specs[idx].convert(kw_vals[i], targets[idx]))) {
            return std::unexpected(FastParseReportConversionError(fastparse, idx, kw_vals[i]));
        }
        *mask |= (1ULL << idx);
    }
    return {};
}

[[gnu::always_inline, gnu::const, nodiscard]]
static inline uint64_t fast_parse_make_mask(size_t n) {
    if (n == 0) {
        return 0;
    }
    static constexpr uint64_t FULL_BITS = 64;
    if (n >= FULL_BITS) {
        return ~static_cast<uint64_t>(0);
    }
    return (n >= 64) ? ~static_cast<uint64_t>(0) : (1ULL << n) - 1;
}

// unadulterated speed!!!
[[nodiscard, gnu::always_inline, gnu::hot, gnu::no_stack_protector,
  gnu::nonnull(1, 4, 5)]] static inline ParseResult
fast_parse_vector(
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject *kwnames,
    const FastParserCore *fastparse,
    const void *const *targets) {
    // 1. Pre-flight Check
    if (Unlikely(nargs > static_cast<Py_ssize_t>(fastparse->count))) {
        return std::unexpected(FastParseReportTooMany(fastparse, nargs));
    }

    // 2. Handle Positionals
    if (ParseResult positionals = fast_parse_process_pos(fastparse, args, nargs, targets);
        Unlikely(!positionals)) {
        return positionals;
    }

    // 3. Setup Mask & Handle Keywords

    uint64_t mask = fast_parse_make_mask(nargs);
    if (ParseResult keywords =
            fast_parse_process_kw(fastparse, args, nargs, kwnames, &mask, targets);
        Unlikely(!keywords)) {
        return keywords;
    }

    // 4. Final Verification
    if (Unlikely((mask & fastparse->required_mask) != fastparse->required_mask)) {
        return std::unexpected(FastParseReportMissing(fastparse, mask));
    }

    return {};
}

} // namespace CPH::Python
