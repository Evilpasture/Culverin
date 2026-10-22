#pragma once

// The type-erased half of the argument parser: the runtime metadata every
// parser owns, and the cold routines that turn a failure into a diagnostic.
//
// Nothing in this header knows the argument types. `Parser<Name, Args...>`
// (Parser.hpp) holds the compile-time half -- the fold that converts
// positionals, the binary dispatch that converts keywords -- and stores one
// `ParserCore` for keyword lookup, schema dumping and error rendering.
//
// What is *not* here any more is the reason this redesign happened: the old
// core kept a `FastParseFunc hotPath` member pointing at one of ten
// hand-written entry points (a generic vectorcall loop plus one speculative
// stub per arity 0..8), and every hot argument was converted through a
// `bool (*)(PyObject *, const void *)` stored in the spec table. Both are
// gone. The core now holds metadata only: no function pointers at all.

#include "bindings/PythonCommon.hpp"
#include "bindings/parsing/DefaultConfig.hpp"
#include "bindings/parsing/ParseError.hpp"

#include <cstddef>
#include <cstdint>

namespace CPH::Python {

// Returned by keyword lookup when no argument matches.
static constexpr std::size_t kParserEmptySlot = 0xFFFF;

// Touched on every keyword call: the interned spelling for pointer comparison
// and the optional type guard.
struct ArgHot {
    PyObject *interned;       // interned keyword spelling
    PyTypeObject *typeGuard;  // nullptr when the type has no exact check
}; // 16 bytes

// Touched only when a diagnostic is built.
struct ArgCold {
    const char *name;
    const char *typeName;
}; // 16 bytes

// Construction-time description of one argument. The converter is absent on
// purpose: it lives in the trait and is called directly by the fold, so it
// never has to be erased to a function pointer.
struct ArgDef {
    const char *name;
    const char *typeName;
    PyTypeObject *typeGuard;
    bool required;
};

[[nodiscard]] static inline bool CheckTypeGuard(const ArgHot *spec, PyObject *value) noexcept {
    PyTypeObject *guard = spec->typeGuard;

    // The common case is "this argument needs no type check", which the
    // branch predictor resolves for free.
    if (guard == nullptr) [[likely]] {
        return true;
    }

    // HOT PATH: exact pointer match
    if (Py_IS_TYPE(value, guard)) [[likely]] {
        return true;
    }

    // COLD PATH: subclasses
    return PyObject_TypeCheck(value, guard);
}

[[nodiscard]] static inline uint64_t MakeMask(std::size_t provided) noexcept {
    if (provided >= 64) {
        return ~static_cast<uint64_t>(0);
    }
    return (1ULL << provided) - 1;
}

struct alignas(kParserAlignment) ParserCore {
    const char *parserName;
    ArgHot *hotSpecs;
    ArgCold *coldSpecs;
    uint16_t *lookupTable;
    std::size_t count;
    std::size_t tableMask;
    uint64_t requiredMask;
    uint64_t typeGuardMask;
    uint64_t warnedMask;

    // Interns the keyword spellings, builds the optional hash table and
    // records the masks. Fatal on allocation failure, exactly as before.
    void Init(const ArgDef *defs, std::size_t count) noexcept;
    void Deinit() noexcept;

    // Interned-pointer lookup, then the binary probe with an uninterned-string
    // slow path. `warnedMask` memoises the RuntimeWarning, so this is
    // logically const despite the write -- see the definition.
    [[nodiscard]] std::size_t FindKeywordIndex(PyObject *key) const noexcept;

    // Diagnostics. Out of line and cold: the hot path never touches them.
    [[nodiscard, gnu::cold]] ParseError ReportMissing(uint64_t provided_mask) const;
    [[nodiscard, gnu::cold]] ParseError ReportTypeError(std::size_t index, PyObject *value) const;
    [[nodiscard, gnu::cold]] ParseError ReportMultiple(std::size_t index) const;
    [[nodiscard, gnu::cold]] ParseError ReportTooMany(Py_ssize_t nargs) const;
    [[nodiscard, gnu::cold]] ParseError ReportUnknownKeyword(PyObject *key) const;
    [[nodiscard, gnu::cold]] ParseError ReportConversionError(std::size_t index,
                                                              PyObject *value) const;
};

static_assert(sizeof(ParserCore) == kParserAlignment * 2,
              "ParserCore is cache-line sized and padded to two lines; adding a field "
              "changes the layout of every parser in the process");
static_assert(alignof(ParserCore) == kParserAlignment);

} // namespace CPH::Python
