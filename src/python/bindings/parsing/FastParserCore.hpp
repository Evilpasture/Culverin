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
    PyTypeObject *typeGuard;                  // For type checking
    bool (*convert)(PyObject *, const void *); // The workhorse
};                          // 24 bytes

struct FastArgSpecCold {
    const char *name;
    const char *typeName;
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
    const char *typeName;
    PyTypeObject *typeGuard;
    bool (*convert)(PyObject *, const void *);
    bool required;
};

struct FastParserCore {
    alignas(kFastParseAlignmentSize) const char *parserName;
    FastArgSpecHot *hotSpecs;
    FastArgSpecCold *coldSpecs;
    uint16_t *lookupTable;
    FastParseFunc hotPath;
    size_t count;
    size_t tableMask;
    uint64_t requiredMask;
    uint64_t typeGuardMask;
    uint64_t warnedMask;
};

static_assert(sizeof(struct FastParserCore) == static_cast<size_t>(kFastParseAlignmentSize) * 2);
static_assert(alignof(struct FastParserCore) == kFastParseAlignmentSize);

// Forward declaration of the generic fallback
[[nodiscard]] static inline ParseResult FastParseVector(
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
static inline bool FastParseDispatchConv(bool (*conv)(PyObject *, const void *),
                                            PyObject *obj, const void *target);

[[nodiscard, gnu::always_inline]]
static inline ParseResult FastParseSpecCheckAndConv(const FastArgSpecHot *spec,
                                                         std::size_t index, PyObject *value,
                                                         const void *target,
                                                         const FastParserCore *fastparser) {
    if (spec->typeGuard) {
        if (Py_TYPE(value) != spec->typeGuard) [[likely]] {
            if (!PyObject_TypeCheck(value, spec->typeGuard)) [[unlikely]] {
                return std::unexpected(FastParseReportTypeError(fastparser, index, value));
            }
        }
    }
    if (!FastParseDispatchConv(spec->convert, value, target)) [[unlikely]] {
        // Deliberately undecorated: the monomorphic stubs have always returned
        // the converter's raw error, while only FastParseProcessPos adds the
        // signature context. Unifying them would change the diagnostics these
        // hot-path calls produce, which is a behaviour change rather than a
        // refactor -- flagged for a follow-up, not done here.
        return CaptureFailure();
    }
    return {};
}

[[gnu::noinline]] static inline ParseResult
FastParseFallback(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                    const FastParserCore *fastparser, const void *const *targets) {
    return FastParseVector(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP0(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 0 && kwnames == nullptr) [[likely]] {
        return {};
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP1(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 1 && kwnames == nullptr) [[likely]] {
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[0], 0,
                                              args[0], targets[0], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP2(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 2 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[1], 1,
                                              args[1], targets[1], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP3(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 3 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[2], 2,
                                              args[2], targets[2], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP4(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 4 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[3], 3,
                                              args[3], targets[3], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP5(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 5 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[4], 4,
                                              args[4], targets[4], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP6(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 6 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[4], 4, args[4], targets[4], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[5], 5,
                                              args[5], targets[5], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP7(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 7 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[4], 4, args[4], targets[4], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[5], 5, args[5], targets[5], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[6], 6,
                                              args[6], targets[6], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

[[nodiscard, gnu::always_inline]] static inline ParseResult
FastParseSpeculateP8(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
                        const FastParserCore *fastparser, const void *const *targets) {
    if (nargs == 8 && kwnames == nullptr) [[likely]] {
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[0], 0, args[0], targets[0], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[1], 1, args[1], targets[1], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[2], 2, args[2], targets[2], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[3], 3, args[3], targets[3], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[4], 4, args[4], targets[4], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[5], 5, args[5], targets[5], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        if (ParseResult step = FastParseSpecCheckAndConv(
                &fastparser->hotSpecs[6], 6, args[6], targets[6], fastparser);
            !step) {
            return std::unexpected(std::move(step.error()));
        }
        return FastParseSpecCheckAndConv(&fastparser->hotSpecs[7], 7,
                                              args[7], targets[7], fastparser);
    }
    return FastParseFallback(args, nargs, kwnames, fastparser, targets);
}

/** --- 3. CONVERTER DISPATCH --- **/

template <typename T>
[[nodiscard]] static inline T *FastParseTarget(const void *target) noexcept {
    return static_cast<T *>(const_cast<void *>(target));
}

[[nodiscard, gnu::always_inline]] static inline bool
FastParseConvFloat(PyObject *obj, const void *target) {
    // HOT PATH: Exact float match (Pointer comparison)
    if (PyFloat_CheckExact(obj)) {
        *FastParseTarget<float>(target) =
            static_cast<float>((reinterpret_cast<PyFloatObject *>(obj))->ob_fval);
        return true;
    }

    // COLD PATH: Handle ints or float subclasses
    double val = PyFloat_AsDouble(obj);
    if (val == -1.0 && PyErr_Occurred()) [[unlikely]] {
        return false;
    }
    *FastParseTarget<float>(target) = static_cast<float>(val);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool
FastParseConvDouble(PyObject *obj, const void *target) {
    // HOT PATH: Exact float match
    // Bypasses the overhead of PyFloat_AsDouble (magic method checks, int conversion logic)
    if (PyFloat_CheckExact(obj)) {
        *FastParseTarget<double>(target) = (reinterpret_cast<PyFloatObject *>(obj))->ob_fval;
        return true;
    }

    // COLD PATH: Handle ints, None, or float subclasses
    if (obj == Py_None) [[unlikely]] {
        PyErr_SetString(PyExc_TypeError, "double argument cannot be None");
        return false;
    }

    double val = PyFloat_AsDouble(obj);
    if (val == -1.0 && PyErr_Occurred()) [[unlikely]] {
        return false;
    }
    *FastParseTarget<double>(target) = val;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool FastParseConvInt(PyObject *obj,
                                                                              const void *target) {
    // 1. Singletons (Pointer identity is the fastest possible check)
    if (obj == Py_True) {
        *FastParseTarget<int>(target) = 1;
        return true;
    }
    if (obj == Py_False) {
        *FastParseTarget<int>(target) = 0;
        return true;
    }

    // 2. None Check (Cold path, but keeps PyLong_AsLong clean)
    if (obj == Py_None) [[unlikely]] {
        PyErr_SetString(PyExc_TypeError, "int argument cannot be None");
        return false;
    }

    // 3. Single Conversion Call
    long val = PyLong_AsLong(obj);

    // 4. Hot Path: 99.9% of integers in physics/logic are not -1
    if (val != -1) [[likely]] {
        *FastParseTarget<int>(target) = static_cast<int>(val);
        return true;
    }

    // 5. Cold Path: Was it actually -1 or a conversion error?
    if (PyErr_Occurred()) {
        return false;
    }

    *FastParseTarget<int>(target) = -1;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool FastParseConvU32(PyObject *obj,
                                                                              const void *target) {
    if (obj == Py_True) {
        *FastParseTarget<uint32_t>(target) = 1;
        return true;
    }
    if (obj == Py_False) {
        *FastParseTarget<uint32_t>(target) = 0;
        return true;
    }
    if (obj == Py_None) [[unlikely]] {
        PyErr_SetString(PyExc_TypeError, "uint32 argument cannot be None");
        return false;
    }

    unsigned long val = PyLong_AsUnsignedLong(obj);

    if (val != (unsigned long)-1) [[likely]] {
        *FastParseTarget<uint32_t>(target) = static_cast<uint32_t>(val);
        return true;
    }

    if (PyErr_Occurred()) {
        return false;
    }

    *FastParseTarget<uint32_t>(target) = static_cast<uint32_t>(-1);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool FastParseConvU64(PyObject *obj,
                                                                              const void *target) {
    if (obj == Py_True) {
        *FastParseTarget<uint64_t>(target) = 1;
        return true;
    }
    if (obj == Py_False) {
        *FastParseTarget<uint64_t>(target) = 0;
        return true;
    }
    // u64 often used for handles, which might be None
    if (obj == Py_None) {
        *FastParseTarget<uint64_t>(target) = 0;
        return true;
    }

    unsigned long long val = PyLong_AsUnsignedLongLong(obj);

    if (val != (unsigned long long)-1) [[likely]] {
        *FastParseTarget<uint64_t>(target) = static_cast<uint64_t>(val);
        return true;
    }

    if (PyErr_Occurred()) {
        return false;
    }

    *FastParseTarget<uint64_t>(target) = static_cast<uint64_t>(-1);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool FastParseConvBool(PyObject *obj,
                                                                               const void *target) {
    // Exact pointer check for the singletons
    if (obj == Py_True) {
        *FastParseTarget<bool>(target) = true;
        return true;
    }
    if (obj == Py_False) {
        *FastParseTarget<bool>(target) = false;
        return true;
    }

    // Fallback for truthiness of lists, strings, etc.
    int val = PyObject_IsTrue(obj);
    if (val == -1) [[unlikely]] {
        return false;
    }
    *FastParseTarget<bool>(target) = static_cast<bool>(val);
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool
FastParseConvPyobj(PyObject *obj, const void *target) {
    *FastParseTarget<PyObject *>(target) = obj;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool FastParseConvStr(PyObject *obj,
                                                                              const void *target) {
    if (!PyUnicode_Check(obj)) [[unlikely]] {
        PyErr_Format(PyExc_TypeError, "expected str, got %s", Py_TYPE(obj)->tp_name);
        return false;
    }
    const char *str = PyUnicode_AsUTF8(obj);
    if (str == nullptr) [[unlikely]] {
        return false;
    }
    *FastParseTarget<const char *>(target) = str;
    return true;
}

[[nodiscard, gnu::always_inline]] static inline bool
FastParseConvSsize(PyObject *obj, const void *target) {
    // HOT PATH: Exact integer match
    if (PyLong_CheckExact(obj)) [[likely]] {
        Py_ssize_t val = PyLong_AsSsize_t(obj);
        // Optimization: Values like buffer sizes and counts are rarely exactly -1.
        if (val != -1) [[likely]] {
            *FastParseTarget<Py_ssize_t>(target) = val;
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
    *FastParseTarget<Py_ssize_t>(target) = val;
    return true;
}

[[gnu::always_inline, nodiscard]]
static inline bool FastParseDispatchConv(bool (*conv)(PyObject *, const void *),
                                                 PyObject *obj, const void *target) {
    // These comparisons are just address checks (extremely fast)
    // If a match is found, the compiler inlines the specific converter.
    if (conv == FastParseConvFloat) {
        return FastParseConvFloat(obj, target);
    }
    if (conv == FastParseConvInt) {
        return FastParseConvInt(obj, target);
    }
    if (conv == FastParseConvU64) {
        return FastParseConvU64(obj, target);
    }
    if (conv == FastParseConvSsize) {
        return FastParseConvSsize(obj, target);
    }
    if (conv == FastParseConvBool) {
        return FastParseConvBool(obj, target);
    }
    if (conv == FastParseConvPyobj) {
        return FastParseConvPyobj(obj, target);
    }

    // Fallback for custom or less common converters
    return conv(obj, target);
}

/** --- 4. THE HOT PATH --- **/
[[gnu::always_inline, gnu::const, maybe_unused, nodiscard]]
static inline size_t FastParseHashPtr(PyObject *ptr, size_t mask) {
    auto val = reinterpret_cast<uintptr_t>(ptr);
    // Golden ratio multiplier spreads pointer bits effectively
    constexpr auto gratio     = 11400714819323198485ULL;
    constexpr auto shift_bits = 32ULL;
    return ((val * gratio) >> shift_bits) & mask;
}
[[gnu::always_inline, nodiscard]]
static inline bool FastParseCheckTypeGuard(
    const FastArgSpecHot *spec,
    PyObject *val) {
    PyTypeObject *guard = spec->typeGuard;

    // If no guard is defined (NULL), this branch is perfectly predicted as "not taken"
    if (guard == nullptr) [[likely]] {
        return true;
    }

    // HOT PATH: Exact pointer match (The most common case in Python)
    if (Py_IS_TYPE(val, guard)) [[likely]] {
        return true;
    }

    // COLD PATH: Inheritance check (Subclasses)
    return PyObject_TypeCheck(val, guard);
}
[[gnu::noinline, gnu::cold]]
extern bool FastParseWarnUninternedSlow(PyObject key[], const char pname[]);

/**
 * COLD FALLBACK: FastParseCmpSlow
 * Separated to keep the instruction cache lean for interned keys.
 */
[[gnu::noinline, gnu::cold]]
extern size_t FastParseCmpSlow(
    PyObject *key,
    const FastParserCore *fastparse);

[[gnu::always_inline, nodiscard]]
static inline size_t FastParseFindKeywordIndex(
    PyObject *key,
    const FastParserCore *fastparse) {
    const FastArgSpecHot *specs = fastparse->hotSpecs;
    const uint16_t *ltable          = fastparse->lookupTable;

    if (ltable != nullptr) [[likely]] {
        size_t hash = FastParseHashPtr(key, fastparse->tableMask);

        // THE FIX: Probe until we hit an empty slot (table is 50% empty, so this is fast)
        // We use table_mask as a safe upper bound to prevent infinite loops.
        for (size_t i = 0; i <= fastparse->tableMask; i++) {
            size_t slot = ltable[(hash + i) & fastparse->tableMask];
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
static inline ParseResult FastParseProcessPos(
    const FastParserCore *fastparse,
    PyObject *const *args,
    Py_ssize_t nargs,
    const void *const *targets) {
#pragma unroll 4
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        // spec is 24 bytes, targets[i] is 8 bytes.
        // We are touching very little memory here.
        const FastArgSpecHot *spec = &fastparse->hotSpecs[i];

        // 1. Safety check (Type Guard)
        // We use the helper to keep the "Happy Path" linear.
        if (!FastParseCheckTypeGuard(spec, args[i])) [[unlikely]] {
            return std::unexpected(
                FastParseReportTypeError(fastparse, static_cast<size_t>(i), args[i]));
        }

        // 2. Conversion
        if (!spec->convert(args[i], targets[i])) [[unlikely]] {
            return std::unexpected(
                FastParseReportConversionError(fastparse, static_cast<size_t>(i), args[i]));
        }
    }
    return {};
}
[[gnu::always_inline, nodiscard]]
static inline ParseResult FastParseProcessKw(
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
        size_t idx    = FastParseFindKeywordIndex(key, fastparse);

        if (idx == kFastParseEmptySlot) [[unlikely]] {
            return std::unexpected(FastParseReportUnknownKeyword(fastparse, key));
        }

        if (*mask & (1ULL << idx)) [[unlikely]] {
            return std::unexpected(FastParseReportMultiple(fastparse, idx));
        }

        // UPDATED: Now only takes 2 arguments
        if (!FastParseCheckTypeGuard(&fastparse->hotSpecs[idx], kw_vals[i])) {
            return std::unexpected(FastParseReportTypeError(fastparse, idx, kw_vals[i]));
        }

        if (!fastparse->hotSpecs[idx].convert(kw_vals[i], targets[idx])) [[unlikely]] {
            return std::unexpected(FastParseReportConversionError(fastparse, idx, kw_vals[i]));
        }
        *mask |= (1ULL << idx);
    }
    return {};
}

[[gnu::always_inline, gnu::const, nodiscard]]
static inline uint64_t FastParseMakeMask(size_t n) {
    if (n == 0) {
        return 0;
    }
    static constexpr uint64_t kFullBits = 64;
    if (n >= kFullBits) {
        return ~static_cast<uint64_t>(0);
    }
    return (n >= 64) ? ~static_cast<uint64_t>(0) : (1ULL << n) - 1;
}

// unadulterated speed!!!
[[nodiscard, gnu::always_inline, gnu::hot, gnu::no_stack_protector,
  gnu::nonnull(1, 4, 5)]] static inline ParseResult
FastParseVector(
    PyObject *const *args,
    Py_ssize_t nargs,
    PyObject *kwnames,
    const FastParserCore *fastparse,
    const void *const *targets) {
    // 1. Pre-flight Check
    if (nargs > static_cast<Py_ssize_t>(fastparse->count)) [[unlikely]] {
        return std::unexpected(FastParseReportTooMany(fastparse, nargs));
    }

    // 2. Handle Positionals
    if (ParseResult positionals = FastParseProcessPos(fastparse, args, nargs, targets);
        !positionals) [[unlikely]] {
        return positionals;
    }

    // 3. Setup Mask & Handle Keywords

    uint64_t mask = FastParseMakeMask(nargs);
    if (ParseResult keywords =
            FastParseProcessKw(fastparse, args, nargs, kwnames, &mask, targets);
        !keywords) [[unlikely]] {
        return keywords;
    }

    // 4. Final Verification
    if ((mask & fastparse->requiredMask) != fastparse->requiredMask) [[unlikely]] {
        return std::unexpected(FastParseReportMissing(fastparse, mask));
    }

    return {};
}

} // namespace CPH::Python
