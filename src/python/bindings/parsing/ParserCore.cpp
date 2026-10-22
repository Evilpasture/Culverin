#include "ParserCore.hpp"
#include "bindings/PythonCommon.hpp"
#include <stdlib.h>
#include <string.h>

namespace CPH::Python {

namespace {

/**
 * COLD PATH: WarnUninternedSlow
 *
 * This function handles the expensive string formatting and warning emission
 * for non-interned keywords. By marking it 'noinline' and 'cold', we ensure
 * that Clang keeps this code away from the high-speed keyword probe's
 * instruction cache zone.
 */
[[gnu::noinline, gnu::cold, nodiscard]] bool WarnUninternedSlow(PyObject *key,
                                                                const char *pname) {
    int result = PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                                  "Priority Alert: Keyword '%U' in parser '%s' is not interned.\n"
                                  "This forces a slow string comparison. To resolve this, use "
                                  "sys.intern() on keywords passed from Python logic.",
                                  key, pname ? pname : "unknown");
    return result >= 0;
}

/**
 * COLD PATH: CompareSlow
 *
 * Reached only when the pointer probe misses, i.e. for keywords that were not
 * interned. Separated from FindKeywordIndex so the probe stays linear.
 */
[[gnu::noinline, gnu::cold, nodiscard]] std::size_t CompareSlow(PyObject *key,
                                                                const ParserCore *parser) {
    const std::size_t count = parser->count;
    const ArgHot *specs     = parser->hotSpecs;

    for (std::size_t j = 0; j < count; ++j) {
        // 1. Is it the exact same pointer that just slipped past the probe?
        if (key == specs[j].interned) {
            return j; // Return silently, no warning needed!
        }

        // 2. Slow string content comparison
        if (PyUnicode_Compare(key, specs[j].interned) == 0) {
            // The warning is a memo, not observable parser state: it is set
            // once per argument for the lifetime of the process and never
            // read by the hot path. Writing it through a const core is what
            // keeps FindKeywordIndex callable from a const Parse().
            if ((parser->warnedMask & (1ULL << j)) == 0) {
                (const_cast<ParserCore *>(parser))->warnedMask |= (1ULL << j);
                if (!WarnUninternedSlow(key, parser->parserName)) {
                    return kParserEmptySlot;
                }
            }
            return j;
        }
    }
    return kParserEmptySlot;
}

// Golden-ratio multiplier spreads pointer bits effectively.
[[gnu::always_inline, gnu::const]] inline std::size_t HashPointer(PyObject *ptr,
                                                                  std::size_t mask) noexcept {
    auto value                = reinterpret_cast<uintptr_t>(ptr);
    constexpr auto gratio     = 11400714819323198485ULL;
    constexpr auto shift_bits = 32ULL;
    return ((value * gratio) >> shift_bits) & mask;
}

/**
 * Linear probe, or the hash table above eight arguments. Kept inside the
 * translation unit: one call per keyword argument, and the body is large
 * enough that inlining it into every parser instantiation would cost more
 * than the call saves.
 */
[[gnu::always_inline]] inline std::size_t FindKeyword(PyObject *key, const ParserCore *parser) {
    const ArgHot *specs    = parser->hotSpecs;
    const uint16_t *ltable = parser->lookupTable;

    if (ltable != nullptr) [[likely]] {
        std::size_t hash = HashPointer(key, parser->tableMask);

        // Probe until an empty slot; the table is 50% empty, so this is short.
        // tableMask is the safe upper bound against an infinite loop.
        for (std::size_t i = 0; i <= parser->tableMask; i++) {
            std::size_t slot = ltable[(hash + i) & parser->tableMask];
            if (slot == kParserEmptySlot) {
                break;
            }
            if (specs[slot].interned == key) {
                return slot;
            }
        }
    } else {
        const std::size_t count = parser->count;
        for (std::size_t j = 0; j < count; ++j) {
            if (specs[j].interned == key) {
                return j;
            }
        }
    }

    return CompareSlow(key, parser);
}

[[gnu::always_inline, gnu::const]] inline int Min3(int a, int b, int c) noexcept {
    int m = a < b ? a : b;
    return m < c ? m : c;
}

[[gnu::noinline, gnu::cold, nodiscard]] int Levenshtein(const char *s1, const char *s2) {
    const int len1         = static_cast<int>(strlen(s1));
    const int len2         = static_cast<int>(strlen(s2));
    constexpr auto max_len = 64;

    if (len1 >= max_len || len2 >= max_len) {
        return max_len;
    }

    uint8_t column[max_len + 1];
    for (int i = 0; i <= len2; i++) {
        column[i] = static_cast<uint8_t>(i);
    }
#if defined(__clang__)
#pragma unroll 4
#endif
    for (int x = 1; x <= len1; x++) {
        column[0]         = static_cast<uint8_t>(x);
        uint8_t last_diag = static_cast<uint8_t>(x - 1);
        for (int y = 1; y <= len2; y++) {
            uint8_t old_column = column[y];
            uint8_t cost       = (s1[x - 1] != s2[y - 1]);

            // Branchless min if your compiler doesn't do it automatically
            column[y] = static_cast<uint8_t>(Min3(column[y] + 1,     // Deletion
                                                  column[y - 1] + 1, // Insertion
                                                  last_diag + cost)); // Substitution
            last_diag = old_column;
        }
    }
    return static_cast<int>(column[len2]);
}

} // namespace

void ParserCore::Init(const ArgDef *defs, std::size_t count) noexcept {
    if (count > kParserMaxArgumentCount) {
        Py_FatalError("Culverin parser: argument count exceeds 64.");
    }

    this->count         = count;
    this->requiredMask  = 0;
    this->typeGuardMask = 0;
    this->lookupTable   = nullptr;
    this->warnedMask    = 0;

    // 1. Allocate Hot and Cold arrays (Data-Oriented Split)
    // Using PyMem_RawMalloc to avoid GIL dependencies during allocation
    this->hotSpecs  = static_cast<ArgHot *>(Allocate(count * sizeof(ArgHot)));
    this->coldSpecs = static_cast<ArgCold *>(Allocate(count * sizeof(ArgCold)));

    if (!this->hotSpecs || !this->coldSpecs) {
        Py_FatalError("Culverin parser: memory allocation failed during initialization.");
    }

    for (std::size_t i = 0; i < count; i++) {
        // Populate Hot (Accessed every keyword call)
        this->hotSpecs[i].interned =
            defs[i].name ? PyUnicode_InternFromString(defs[i].name) : nullptr;
        this->hotSpecs[i].typeGuard = defs[i].typeGuard;

        // Populate Cold (Accessed only on error)
        this->coldSpecs[i].name     = defs[i].name;
        this->coldSpecs[i].typeName = defs[i].typeName;

        // Setup masks for mask-based checks
        if (defs[i].required) {
            this->requiredMask |= (1ULL << i);
        }
        if (defs[i].typeGuard) {
            this->typeGuardMask |= (1ULL << i);
        }
    }

    // 2. Keyword Lookup Optimization
    // NOTE: the table is built above eight arguments while
    // kParserKeywordHashThreshold documents ten. Pre-existing inconsistency,
    // kept as-is because changing it changes which parsers allocate.
    if (count > 8) {
        std::size_t table_size = 1;
        while (table_size < (count * 2)) {
            table_size <<= 1UL;
        }

        this->tableMask = table_size - 1;
        this->lookupTable = static_cast<uint16_t *>(Allocate(table_size * sizeof(uint16_t)));
        if (!this->lookupTable) {
            Py_FatalError("Culverin parser: failed to allocate lookup table.");
        }

        for (std::size_t i = 0; i < table_size; i++) {
            this->lookupTable[i] = static_cast<uint16_t>(kParserEmptySlot);
        }

        for (std::size_t i = 0; i < count; i++) {
            std::size_t hash = HashPointer(this->hotSpecs[i].interned, this->tableMask);
            while (this->lookupTable[hash] != kParserEmptySlot) {
                hash = (hash + 1) & this->tableMask;
            }
            this->lookupTable[hash] = static_cast<uint16_t>(i);
        }
    }

#if defined(CULVERIN_DEBUG)
    PySys_WriteStderr("Culverin parser [%s]: arity %zu, required_mask=0x%016llx, "
                      "type_guard_mask=0x%016llx, keyword_table=%s\n",
                      this->parserName ? this->parserName : "?", this->count,
                      static_cast<unsigned long long>(this->requiredMask),
                      static_cast<unsigned long long>(this->typeGuardMask),
                      this->lookupTable ? "hash" : "linear");
#endif
}

void ParserCore::Deinit() noexcept {
    if (hotSpecs) {
#if defined(__clang__)
#pragma unroll 2
#endif
        for (std::size_t i = 0; i < count; i++) {
            Py_XDECREF(hotSpecs[i].interned);
            hotSpecs[i].interned = nullptr;
        }
        Deallocate(hotSpecs);
        hotSpecs = nullptr;
    }

    if (coldSpecs) {
        Deallocate(coldSpecs);
        coldSpecs = nullptr;
    }

    if (lookupTable) {
        Deallocate(lookupTable);
        lookupTable = nullptr;
    }
}

std::size_t ParserCore::FindKeywordIndex(PyObject *key) const noexcept {
    return FindKeyword(key, this);
}

[[gnu::noinline, gnu::cold, nodiscard]] ParseError ParserCore::ReportMissing(
    uint64_t provided_mask) const {
    const char *pname = parserName ? parserName : "function";

    Ref missing_details{PyList_New(0)};
    Ref signature_parts{PyList_New(0)};

    // If we can't allocate lists for the fancy error, fall back to a basic error
    if (!missing_details || !signature_parts) [[unlikely]] {
        PyErr_Format(PyExc_TypeError, "%s() missing required positional arguments", pname);
        return CaptureError();
    }

    for (std::size_t i = 0; i < count; i++) {
        // DATA-ORIENTED SPLIT:
        // cold_specs has the strings (name, type_name)
        // hot_specs has the runtime objects (type_guard)
        const ArgCold *cold = &coldSpecs[i];
        const ArgHot *hot   = &hotSpecs[i];

        bool is_required = (requiredMask & (1ULL << i)) != 0;
        bool is_provided = (provided_mask & (1ULL << i)) != 0;

        // Resolve type information
        const char *fallback_name = hot->typeGuard ? hot->typeGuard->tp_name : "object";
        const char *type_info     = cold->typeName ? cold->typeName : fallback_name;

        // 1. Log specifically which ones are missing
        if (is_required && !is_provided) {
            Ref detail{PyUnicode_FromFormat("    - %s (%s)", cold->name, type_info)};
            if (detail) {
                PyList_Append(missing_details.get(), detail.get());
            }
        }

        // 2. Build the full signature context [arg: type]
        Ref sig_part{static_cast<int>(is_required)
                         ? PyUnicode_FromFormat("%s: %s", cold->name, type_info)
                         : PyUnicode_FromFormat("[%s: %s]", cold->name, type_info)};

        if (sig_part) {
            PyList_Append(signature_parts.get(), sig_part.get());
        }
    }

    Py_ssize_t num_missing = PyList_Size(missing_details.get());
    Ref newline{PyUnicode_FromString("\n")};
    Ref comma{PyUnicode_FromString(", ")};

    // Check for NULL before joining (OOM protection)
    Ref joined_missing{newline ? PyUnicode_Join(newline.get(), missing_details.get()) : nullptr};
    Ref joined_sig{comma ? PyUnicode_Join(comma.get(), signature_parts.get()) : nullptr};

    if (joined_missing && joined_sig) {
        PyErr_Format(
            PyExc_TypeError,
            "%s() missing %zd required positional argument%s:\n%U\n\nExpected signature:\n  %s(%U)",
            pname, num_missing, num_missing == 1 ? "" : "s", joined_missing.get(), pname,
            joined_sig.get());
    } else {
        // Fallback if string joining failed due to memory pressure
        PyErr_Format(PyExc_TypeError, "%s() missing %zd required positional arguments", pname,
                     num_missing);
    }

    return CaptureError();
}

[[gnu::noinline, gnu::cold, nodiscard]] ParseError ParserCore::ReportTypeError(
    std::size_t index, PyObject *val) const {
    const ArgCold *cold_spec = &coldSpecs[index];
    const ArgHot *hot_spec   = &hotSpecs[index];
    const char *pname        = parserName ? parserName : "function";

    // Resolve the "expected" type name: use type_name string if available,
    // else peek into the hot type_guard's tp_name.
    const char *fallback_name = (hot_spec->typeGuard) ? hot_spec->typeGuard->tp_name : "object";
    const char *expected      = (cold_spec->typeName) ? cold_spec->typeName : fallback_name;

    // The compact message is both the OOM fallback and the last line of defence.
    const auto basic_error = [&]() {
        PyErr_Format(PyExc_TypeError, "%s() argument '%s' must be %s, not %s", pname,
                     cold_spec->name, expected, Py_TYPE(val)->tp_name);
        return CaptureError();
    };

    // 1. Safe Repr Acquisition (Truncated to avoid overwhelming the error message)
    Ref val_repr{PyObject_Repr(val)};
    if (!val_repr) [[unlikely]] {
        PyErr_Clear(); // Clear OOM from Repr to set our TypeError
        val_repr = PyUnicode_FromString("<repr unavailable>");
    } else if (PyUnicode_GetLength(val_repr.get()) > 100) {
        Ref truncated{PyUnicode_Substring(val_repr.get(), 0, 97)};
        if (truncated) {
            Ref formatted{PyUnicode_FromFormat("%U...", truncated.get())};
            if (formatted) {
                val_repr = std::move(formatted);
            }
        }
    }

    // 2. Diagnostic Signature Construction: (e.g. "func(arg1: int, !!! arg2: str !!!, [arg3:
    // float])")
    Ref signature_parts{PyList_New(0)};
    if (!signature_parts) [[unlikely]] {
        return basic_error();
    }

    for (std::size_t i = 0; i < count; i++) {
        const ArgCold *s_cold = &coldSpecs[i];
        const ArgHot *s_hot   = &hotSpecs[i];

        bool is_required = (requiredMask & (1ULL << i)) != 0;

        const char *fb = s_hot->typeGuard ? s_hot->typeGuard->tp_name : "object";
        const char *t  = s_cold->typeName ? s_cold->typeName : fb;

        static const char *FMT_MAP[] = {"[%s: %s]", "%s: %s"};
        // Highlight the offending argument with exclamation marks
        const char *fmt = (i == index) ? "!!! %s: %s !!!" : FMT_MAP[static_cast<int>(is_required)];

        Ref part{PyUnicode_FromFormat(fmt, s_cold->name, t)};
        if (part) {
            PyList_Append(signature_parts.get(), part.get());
        }
    }

    Ref comma{PyUnicode_FromString(", ")};
    if (!comma) [[unlikely]] {
        return basic_error();
    }

    Ref joined_sig{PyUnicode_Join(comma.get(), signature_parts.get())};
    if (!joined_sig) [[unlikely]] {
        return basic_error();
    }

    // 3. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "%s() argument '%s' must be %s, not %s.\n\n"
                 "Received value: %U\n"
                 "Signature context:\n  %s(%U)",
                 pname, cold_spec->name, expected, Py_TYPE(val)->tp_name,
                 val_repr ? val_repr.get() : Py_None, pname, joined_sig.get());
    return CaptureError();
}

[[gnu::noinline, gnu::cold, nodiscard]] ParseError ParserCore::ReportMultiple(
    std::size_t index) const {
    const char *pname    = parserName ? parserName : "function";
    const char *arg_name = coldSpecs[index].name;

    const auto basic_error = [&]() {
        // Fallback if we can't allocate the fancy message
        PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'", pname,
                     arg_name);
        return CaptureError();
    };

    // 1. Diagnostic Signature Construction
    Ref parts{PyList_New(0)};
    if (!parts) [[unlikely]] {
        return basic_error();
    }

    for (std::size_t i = 0; i < count; i++) {
        const char *name = coldSpecs[i].name;
        Ref part{(i == index) ? PyUnicode_FromFormat("!!! %s !!!", name)
                              : PyUnicode_FromString(name)};
        if (part) {
            PyList_Append(parts.get(), part.get());
        }
    }

    Ref comma{PyUnicode_FromString(", ")};
    if (!comma) [[unlikely]] {
        return basic_error();
    }

    Ref sig{PyUnicode_Join(comma.get(), parts.get())};
    if (!sig) [[unlikely]] {
        return basic_error();
    }

    // 2. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "%s() got multiple values for argument '%s'.\n"
                 "It was provided both as a positional argument and as a keyword argument.\n\n"
                 "Signature context:\n  %s(%U)",
                 pname, arg_name, pname, sig.get());
    return CaptureError();
}

[[gnu::noinline, gnu::cold, nodiscard]] ParseError ParserCore::ReportTooMany(
    Py_ssize_t nargs) const {
    const char *pname = parserName ? parserName : "function";

    // 1. Calculate counts using the bitmask
    // We count set bits in required_mask to find how many args are mandatory.
    // Count set bits in the 64-bit mask using hardware instruction
    std::size_t req_count   = static_cast<std::size_t>(__builtin_popcountll(requiredMask));
    std::size_t total_count = count;

    char count_str[64];
    if (req_count == total_count) {
        snprintf(count_str, sizeof(count_str), "%zu", total_count);
    } else {
        snprintf(count_str, sizeof(count_str), "from %zu to %zu", req_count, total_count);
    }

    const auto basic_error = [&]() {
        // Fallback if we can't allocate the fancy message strings
        PyErr_Format(PyExc_TypeError, "%s() takes %s positional arguments but %zd were given",
                     pname, count_str, nargs);
        return CaptureError();
    };

    // 2. Diagnostic Signature Construction (Using Cold Spec names)
    Ref parts{PyList_New(0)};
    if (!parts) [[unlikely]] {
        return basic_error();
    }

    for (std::size_t i = 0; i < total_count; i++) {
        // Names are stored in the Cold specs to keep the Hot path lean
        Ref part{PyUnicode_FromString(coldSpecs[i].name)};
        if (part) {
            PyList_Append(parts.get(), part.get());
        }
    }

    Ref comma{PyUnicode_FromString(", ")};
    if (!comma) [[unlikely]] {
        return basic_error();
    }

    Ref sig{PyUnicode_Join(comma.get(), parts.get())};
    if (!sig) [[unlikely]] {
        return basic_error();
    }

    // 3. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "%s() takes %s positional arguments but %zd were given.\n\n"
                 "This function does not accept variable positional arguments (*args).\n"
                 "Expected sequence:\n  %s(%U)",
                 pname, count_str, nargs, pname, sig.get());
    return CaptureError();
}

[[gnu::noinline, gnu::cold, nodiscard]] ParseError ParserCore::ReportUnknownKeyword(
    PyObject *key) const {
    const char *actual_name = PyUnicode_AsUTF8(key);
    if (!actual_name) {
        return CaptureError();
    }
    constexpr auto max_dist          = 3;
    const char *best_match           = nullptr;
    constexpr auto initial_best_dist = 100;
    int best_dist                    = initial_best_dist;
    const char *pname                = parserName ? parserName : "function";

    // 1. Find the best fuzzy match (Scanning Cold names)
    for (std::size_t i = 0; i < count; i++) {
        const char *cand = coldSpecs[i].name;
        if (!cand) {
            continue;
        }

        int dist = Levenshtein(actual_name, cand);
        if (dist < best_dist) {
            best_dist  = dist;
            best_match = cand;
        }
    }

    // Fallback logic for low-memory situations (and for the "no suggestion" case).
    const auto basic_error = [&]() {
        if (best_match && best_dist < max_dist) {
            PyErr_Format(PyExc_TypeError,
                         "'%s' is an invalid keyword argument for %s(); did you mean '%s'?",
                         actual_name, pname, best_match);
        } else {
            PyErr_Format(PyExc_TypeError, "'%s' is an invalid keyword argument for %s()",
                         actual_name, pname);
        }
        return CaptureError();
    };

    // 2. Diagnostic Message Construction (Building the list of valid options)
    Ref valid_args_list{PyList_New(0)};
    if (!valid_args_list) [[unlikely]] {
        return basic_error();
    }

    for (std::size_t i = 0; i < count; i++) {
        const ArgCold *cold = &coldSpecs[i];
        const ArgHot *hot   = &hotSpecs[i];

        if (!cold->name) {
            continue;
        }

        bool is_required = (requiredMask & (1ULL << i)) != 0;

        // Resolve type name: Cold string preferred, fallback to Hot type_guard info
        const char *fb = hot->typeGuard ? hot->typeGuard->tp_name : "object";
        const char *t  = cold->typeName ? cold->typeName : fb;

        Ref info{PyUnicode_FromFormat("    - %s: %s%s", cold->name, t,
                                      static_cast<int>(is_required) ? " [required]" : "")};
        if (info) {
            PyList_Append(valid_args_list.get(), info.get());
        }
    }

    Ref newline{PyUnicode_FromString("\n")};
    if (!newline) [[unlikely]] {
        return basic_error();
    }

    Ref joined{PyUnicode_Join(newline.get(), valid_args_list.get())};
    if (!joined) [[unlikely]] {
        return basic_error();
    }

    // 3. Format "Did you mean" suggestion
    static constexpr auto suggestion_buffer = 256;
    char suggestion[suggestion_buffer];
    suggestion[0] = '\0';
    if (best_match && best_dist < max_dist) {
        snprintf(suggestion, sizeof(suggestion), " Did you mean '%s'?", best_match);
    }

    // 4. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "'%s' is an invalid keyword argument for %s().%s\nValid arguments are:\n%U",
                 actual_name, pname, suggestion, joined.get());
    return CaptureError();
}

[[gnu::noinline, gnu::cold, nodiscard]] ParseError ParserCore::ReportConversionError(
    std::size_t index, PyObject *val) const {
    // 1. Capture the original "dry" error (e.g., OverflowError: "can't convert...")
    Ref exc{PyErr_GetRaisedException()};
    if (!exc) [[unlikely]] {
        return CaptureError();
    }

    const char *pname    = parserName ? parserName : "function";
    const char *arg_name = coldSpecs[index].name;

    // Safety valve: if any allocation below fails, re-raise the original
    // exception so the user still sees the dry error rather than an OOM.
    const auto oom_fallback = [&]() {
        PyErr_SetRaisedException(exc.release());
        return CaptureError();
    };

    // 2. Best-effort acquisition of Value Repr
    Ref val_repr{PyObject_Repr(val)};
    if (!val_repr) [[unlikely]] {
        PyErr_Clear(); // Clear OOM from repr attempt
        val_repr = PyUnicode_FromString("<repr unavailable>");
    }

    // 3. Setup diagnostics components
    Ref parts{PyList_New(0)};
    Ref comma{PyUnicode_FromString(", ")};
    Ref exc_str{PyObject_Str(exc.get())};

    // If we failed basic allocations, fall back to the original exception
    if (!parts || !comma || !exc_str) [[unlikely]] {
        return oom_fallback();
    }

    // Build the fancy signature context
    for (std::size_t i = 0; i < count; i++) {
        const char *name = coldSpecs[i].name;
        const char *type = coldSpecs[i].typeName;
        const char *fmt  = (i == index) ? "!!! %s: %s !!!" : "%s: %s";
        Ref part{PyUnicode_FromFormat(fmt, name, type ? type : "object")};
        if (part) {
            PyList_Append(parts.get(), part.get());
        }
    }

    Ref sig{PyUnicode_Join(comma.get(), parts.get())};
    if (!sig) [[unlikely]] {
        return oom_fallback();
    }

    // 4. Final assembly of the high-fidelity message
    Ref new_msg{PyUnicode_FromFormat("%s() argument '%s': %U\n"
                                     "Received value: %U\n\n"
                                     "Signature context:\n  %s(%U)",
                                     pname, arg_name, exc_str.get(), val_repr.get(), pname,
                                     sig.get())};

    if (!new_msg) [[unlikely]] {
        return oom_fallback();
    }

    PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(exc.get())), new_msg.get());
    return CaptureError();
}

} // namespace CPH::Python
