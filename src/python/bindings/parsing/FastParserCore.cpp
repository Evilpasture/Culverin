#include "FastParserCore.hpp"
#include "bindings/PythonCommon.hpp"
#include <stdlib.h>

namespace CPH::Python {

static constexpr FastParseFunc kFastParseMonoStubs[] = {
    fast_parse_speculate_p0, fast_parse_speculate_p1, fast_parse_speculate_p2,
    fast_parse_speculate_p3, fast_parse_speculate_p4, fast_parse_speculate_p5,
    fast_parse_speculate_p6, fast_parse_speculate_p7, fast_parse_speculate_p8};

static constexpr size_t kFastParseStubsSize = 9;
static_assert(sizeof(kFastParseMonoStubs) / sizeof(FastParseFunc) == kFastParseStubsSize,
              "FAST_PARSE_MONO_STUBS table must contain exactly 9 stubs (0-8 args)");

/**
 * COLD PATH: fast_parse_warn_uninterned_slow
 *
 * This function handles the expensive string formatting and warning emission
 * for non-interned keywords. By marking it 'noinline' and 'cold', we ensure
 * that Clang keeps this code away from the high-speed 'fast_parse_find_keyword_index'
 * instruction cache zone.
 */
[[gnu::noinline, gnu::cold, nodiscard]]
bool FastParseWarnUninternedSlow(PyObject *key, const char *pname) {
    int result = PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                                  "Priority Alert: Keyword '%U' in parser '%s' is not interned.\n"
                                  "This forces a slow string comparison. To resolve this, use "
                                  "sys.intern() on keywords passed from Python logic.",
                                  key, pname ? pname : "unknown");
    return result >= 0;
}

[[gnu::noinline, gnu::cold, nodiscard]]
size_t FastParseCmpSlow(PyObject *key, const FastParserCore *fastparse) {
    const size_t count              = fastparse->count;
    const FastArgSpecHot *specs = fastparse->hot_specs;

    for (size_t j = 0; j < count; ++j) {
        // 1. Is it the exact same pointer that just slipped past the hash table?
        if (key == specs[j].interned) {
            return j; // Return silently, no warning needed!
        }

        // 2. Slow string content comparison
        if (PyUnicode_Compare(key, specs[j].interned) == 0) {
            if ((fastparse->warned_mask & (1ULL << j)) == 0) {
                (const_cast<FastParserCore *>(fastparse))->warned_mask |= (1ULL << j);
                if (!FastParseWarnUninternedSlow(key, fastparse->parser_name)) {
                    return kFastParseEmptySlot;
                }
            }
            return j;
        }
    }
    return kFastParseEmptySlot;
}

[[gnu::noinline, gnu::cold, nodiscard]]
ParseError FastParseReportMissing(const FastParserCore *fastparser, uint64_t provided_mask) {
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    Ref missing_details{PyList_New(0)};
    Ref signature_parts{PyList_New(0)};

    // If we can't allocate lists for the fancy error, fall back to a basic error
    if (Unlikely(!missing_details || !signature_parts)) {
        PyErr_Format(PyExc_TypeError, "%s() missing required positional arguments", pname);
        return CaptureError();
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        // DATA-ORIENTED SPLIT:
        // cold_specs has the strings (name, type_name)
        // hot_specs has the runtime objects (type_guard)
        const FastArgSpecCold *cold = &fastparser->cold_specs[i];
        const FastArgSpecHot *hot   = &fastparser->hot_specs[i];

        bool is_required = (fastparser->required_mask & (1ULL << i)) != 0;
        bool is_provided = (provided_mask & (1ULL << i)) != 0;

        // Resolve type information
        const char *fallback_name = hot->type_guard ? hot->type_guard->tp_name : "object";
        const char *type_info     = cold->type_name ? cold->type_name : fallback_name;

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

[[gnu::noinline, nodiscard]]
ParseError FastParseReportTypeError(const FastParserCore *fastparser, size_t index, PyObject *val) {
    const FastArgSpecCold *cold_spec = &fastparser->cold_specs[index];
    const FastArgSpecHot *hot_spec   = &fastparser->hot_specs[index];
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    // Resolve the "expected" type name: use type_name string if available,
    // else peek into the hot type_guard's tp_name.
    const char *fallback_name = (hot_spec->type_guard) ? hot_spec->type_guard->tp_name : "object";
    const char *expected      = (cold_spec->type_name) ? cold_spec->type_name : fallback_name;

    // The compact message is both the OOM fallback and the last line of defence.
    const auto basic_error = [&]() {
        PyErr_Format(PyExc_TypeError, "%s() argument '%s' must be %s, not %s", pname,
                     cold_spec->name, expected, Py_TYPE(val)->tp_name);
        return CaptureError();
    };

    // 1. Safe Repr Acquisition (Truncated to avoid overwhelming the error message)
    Ref val_repr{PyObject_Repr(val)};
    if (Unlikely(!val_repr)) {
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
    if (Unlikely(!signature_parts)) {
        return basic_error();
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        const FastArgSpecCold *s_cold = &fastparser->cold_specs[i];
        const FastArgSpecHot *s_hot   = &fastparser->hot_specs[i];

        bool is_required = (fastparser->required_mask & (1ULL << i)) != 0;

        const char *fb = s_hot->type_guard ? s_hot->type_guard->tp_name : "object";
        const char *t  = s_cold->type_name ? s_cold->type_name : fb;

        static const char *FMT_MAP[] = {"[%s: %s]", "%s: %s"};
        // Highlight the offending argument with exclamation marks
        const char *fmt = (i == index) ? "!!! %s: %s !!!" : FMT_MAP[static_cast<int>(is_required)];

        Ref part{PyUnicode_FromFormat(fmt, s_cold->name, t)};
        if (part) {
            PyList_Append(signature_parts.get(), part.get());
        }
    }

    Ref comma{PyUnicode_FromString(", ")};
    if (Unlikely(!comma)) {
        return basic_error();
    }

    Ref joined_sig{PyUnicode_Join(comma.get(), signature_parts.get())};
    if (Unlikely(!joined_sig)) {
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

[[gnu::noinline, nodiscard]]
ParseError FastParseReportMultiple(const FastParserCore *fastparser, size_t index) {
    const char *pname    = fastparser->parser_name ? fastparser->parser_name : "function";
    const char *arg_name = fastparser->cold_specs[index].name;

    const auto basic_error = [&]() {
        // Fallback if we can't allocate the fancy message
        PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'", pname,
                     arg_name);
        return CaptureError();
    };

    // 1. Diagnostic Signature Construction
    Ref parts{PyList_New(0)};
    if (Unlikely(!parts)) {
        return basic_error();
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        const char *name = fastparser->cold_specs[i].name;
        Ref part{(i == index) ? PyUnicode_FromFormat("!!! %s !!!", name)
                              : PyUnicode_FromString(name)};
        if (part) {
            PyList_Append(parts.get(), part.get());
        }
    }

    Ref comma{PyUnicode_FromString(", ")};
    if (Unlikely(!comma)) {
        return basic_error();
    }

    Ref sig{PyUnicode_Join(comma.get(), parts.get())};
    if (Unlikely(!sig)) {
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

[[gnu::noinline]]
ParseError FastParseReportTooMany(const FastParserCore *fastparser, Py_ssize_t nargs) {
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    // 1. Calculate counts using the bitmask
    // We count set bits in required_mask to find how many args are mandatory.
    // Count set bits in the 64-bit mask using hardware instruction
    size_t req_count   = static_cast<size_t>(__builtin_popcountll(fastparser->required_mask));
    size_t total_count = fastparser->count;

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
    if (Unlikely(!parts)) {
        return basic_error();
    }

    for (size_t i = 0; i < total_count; i++) {
        // Names are stored in the Cold specs to keep the Hot path lean
        Ref part{PyUnicode_FromString(fastparser->cold_specs[i].name)};
        if (part) {
            PyList_Append(parts.get(), part.get());
        }
    }

    Ref comma{PyUnicode_FromString(", ")};
    if (Unlikely(!comma)) {
        return basic_error();
    }

    Ref sig{PyUnicode_Join(comma.get(), parts.get())};
    if (Unlikely(!sig)) {
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

[[gnu::always_inline, gnu::const]]
static inline int fast_parse_min3(int a, int b, int c) {
    int m = a < b ? a : b;
    return m < c ? m : c;
}

[[gnu::noinline, gnu::cold, nodiscard]]
static int fast_parse_levenshtein(const char *s1, const char *s2) {
    const int len1     = static_cast<int>(strlen(s1));
    const int len2     = static_cast<int>(strlen(s2));
    constexpr auto max_len = 64;

    if (len1 >= max_len || len2 >= max_len) {
        return max_len;
    }

    uint8_t column[max_len + 1];
    for (int i = 0; i <= len2; i++) {
        column[i] = static_cast<uint8_t>(i);
    }
#pragma unroll 4
    for (int x = 1; x <= len1; x++) {
        column[0]         = static_cast<uint8_t>(x);
        uint8_t last_diag = static_cast<uint8_t>(x - 1);
        for (int y = 1; y <= len2; y++) {
            uint8_t old_column = column[y];
            uint8_t cost       = (s1[x - 1] != s2[y - 1]);

            // Branchless min if your compiler doesn't do it automatically
            column[y] = fast_parse_min3(column[y] + 1,     // Deletion
                                        column[y - 1] + 1, // Insertion
                                        last_diag + cost); // Substitution
            last_diag = old_column;
        }
    }
    return static_cast<int>(column[len2]);
}

[[gnu::noinline]]
ParseError FastParseReportUnknownKeyword(const FastParserCore *fastparser, PyObject *key) {
    const char *actual_name = PyUnicode_AsUTF8(key);
    if (!actual_name) {
        return CaptureError();
    }
    constexpr auto max_dist          = 3;
    const char *best_match           = nullptr;
    constexpr auto initial_best_dist = 100;
    int best_dist                    = initial_best_dist;
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    // 1. Find the best fuzzy match (Scanning Cold names)
    for (size_t i = 0; i < fastparser->count; i++) {
        const char *cand = fastparser->cold_specs[i].name;
        if (!cand) {
            continue;
        }

        int dist = fast_parse_levenshtein(actual_name, cand);
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
    if (Unlikely(!valid_args_list)) {
        return basic_error();
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        const FastArgSpecCold *cold = &fastparser->cold_specs[i];
        const FastArgSpecHot *hot   = &fastparser->hot_specs[i];

        if (!cold->name) {
            continue;
        }

        bool is_required = (fastparser->required_mask & (1ULL << i)) != 0;

        // Resolve type name: Cold string preferred, fallback to Hot type_guard info
        const char *fb = hot->type_guard ? hot->type_guard->tp_name : "object";
        const char *t  = cold->type_name ? cold->type_name : fb;

        Ref info{PyUnicode_FromFormat("    - %s: %s%s", cold->name, t,
                                      static_cast<int>(is_required) ? " [required]" : "")};
        if (info) {
            PyList_Append(valid_args_list.get(), info.get());
        }
    }

    Ref newline{PyUnicode_FromString("\n")};
    if (Unlikely(!newline)) {
        return basic_error();
    }

    Ref joined{PyUnicode_Join(newline.get(), valid_args_list.get())};
    if (Unlikely(!joined)) {
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

[[gnu::noinline]]
ParseError FastParseReportConversionError(const FastParserCore *fastparser, size_t index,
                                          PyObject *val) {
    // 1. Capture the original "dry" error (e.g., OverflowError: "can't convert...")
    Ref exc{PyErr_GetRaisedException()};
    if (Unlikely(!exc)) {
        return CaptureError();
    }

    const char *pname    = fastparser->parser_name ? fastparser->parser_name : "function";
    const char *arg_name = fastparser->cold_specs[index].name;

    // Safety valve: if any allocation below fails, re-raise the original
    // exception so the user still sees the dry error rather than an OOM.
    const auto oom_fallback = [&]() {
        PyErr_SetRaisedException(exc.release());
        return CaptureError();
    };

    // 2. Best-effort acquisition of Value Repr
    Ref val_repr{PyObject_Repr(val)};
    if (Unlikely(!val_repr)) {
        PyErr_Clear(); // Clear OOM from repr attempt
        val_repr = PyUnicode_FromString("<repr unavailable>");
    }

    // 3. Setup diagnostics components
    Ref parts{PyList_New(0)};
    Ref comma{PyUnicode_FromString(", ")};
    Ref exc_str{PyObject_Str(exc.get())};

    // If we failed basic allocations, fall back to the original exception
    if (Unlikely(!parts || !comma || !exc_str)) {
        return oom_fallback();
    }

    // Build the fancy signature context
    for (size_t i = 0; i < fastparser->count; i++) {
        const char *name = fastparser->cold_specs[i].name;
        const char *type = fastparser->cold_specs[i].type_name;
        const char *fmt  = (i == index) ? "!!! %s: %s !!!" : "%s: %s";
        Ref part{PyUnicode_FromFormat(fmt, name, type ? type : "object")};
        if (part) {
            PyList_Append(parts.get(), part.get());
        }
    }

    Ref sig{PyUnicode_Join(comma.get(), parts.get())};
    if (Unlikely(!sig)) {
        return oom_fallback();
    }

    // 4. Final assembly of the high-fidelity message
    Ref new_msg{PyUnicode_FromFormat("%s() argument '%s': %U\n"
                                     "Received value: %U\n\n"
                                     "Signature context:\n  %s(%U)",
                                     pname, arg_name, exc_str.get(), val_repr.get(), pname,
                                     sig.get())};

    if (Unlikely(!new_msg)) {
        return oom_fallback();
    }

    PyErr_SetObject(reinterpret_cast<PyObject *>(Py_TYPE(exc.get())), new_msg.get());
    return CaptureError();
}

void FastParseInitImpl(FastParserCore *fastparser, const FastArgDef *defs, size_t count) {
    if (count > 64) {
        Py_FatalError("Culverin parser: argument count exceeds 64.");
    }

    fastparser->count           = count;
    fastparser->required_mask   = 0;
    fastparser->type_guard_mask = 0;
    fastparser->lookup_table    = nullptr;
    fastparser->warned_mask     = 0;

    // 1. Allocate Hot and Cold arrays (Data-Oriented Split)
    // Using PyMem_RawMalloc to avoid GIL dependencies during allocation
    fastparser->hot_specs =
        static_cast<FastArgSpecHot *>(Allocate(count * sizeof(FastArgSpecHot)));
    fastparser->cold_specs =
        static_cast<FastArgSpecCold *>(Allocate(count * sizeof(FastArgSpecCold)));

    if (!fastparser->hot_specs || !fastparser->cold_specs) {
        Py_FatalError("Culverin parser: memory allocation failed during initialization.");
    }

    for (size_t i = 0; i < count; i++) {
        // Populate Hot (Accessed every call)
        fastparser->hot_specs[i].interned =
            defs[i].name ? PyUnicode_InternFromString(defs[i].name) : nullptr;
        fastparser->hot_specs[i].convert    = defs[i].convert;
        fastparser->hot_specs[i].type_guard = defs[i].type_guard;

        // Populate Cold (Accessed only on error)
        fastparser->cold_specs[i].name      = defs[i].name;
        fastparser->cold_specs[i].type_name = defs[i].type_name;

        // Setup Masks for branchless/mask-based checks
        if (defs[i].required) {
            fastparser->required_mask |= (1ULL << i);
        }
        if (defs[i].type_guard) {
            fastparser->type_guard_mask |= (1ULL << i);
        }
    }

    // 2. Keyword Lookup Optimization
    if (count > 8) { // Threshold for using hash table
        size_t table_size = 1;
        while (table_size < (count * 2)) {
            table_size <<= 1UL;
        }

        fastparser->table_mask = table_size - 1;
        fastparser->lookup_table =
            static_cast<uint16_t *>(Allocate(table_size * sizeof(uint16_t)));
        if (!fastparser->lookup_table) {
            Py_FatalError("Culverin parser: failed to allocate lookup table.");
        }

        for (size_t i = 0; i < table_size; i++) {
            fastparser->lookup_table[i] = kFastParseEmptySlot;
        }

        for (size_t i = 0; i < count; i++) {
            size_t hash =
                fast_parse_hash_ptr(fastparser->hot_specs[i].interned, fastparser->table_mask);
            while (fastparser->lookup_table[hash] != kFastParseEmptySlot) {
                hash = (hash + 1) & fastparser->table_mask;
            }
            fastparser->lookup_table[hash] = static_cast<uint16_t>(i);
        }
    }

    // 3. Path Selection
    fastparser->hot_path = fast_parse_vector;

    // Optimization: If all args are required and no type guards are present,
    // we can use the specialized monomorphic stubs.
    uint64_t all_required = (count >= 64) ? ~0ULL : ((1ULL << count) - 1);
    if (fastparser->required_mask == all_required && all_required != 0) {
        if (count < (sizeof(kFastParseMonoStubs) / sizeof(FastParseFunc))) {
            fastparser->hot_path = kFastParseMonoStubs[count];
        }
    }
#if defined(CULVERIN_DEBUG)
    if (fastparser->hot_path == fast_parse_vector) {
        PySys_WriteStderr("Culverin parser [%s]: hot_path fell back to fast_parse_vector "
                          "(count=%zu, required_mask=0x%016llx, all_required=0x%016llx)\n",
                          fastparser->parser_name ? fastparser->parser_name : "?",
                          fastparser->count, (unsigned long long)fastparser->required_mask,
                          (unsigned long long)all_required);
    } else {
        PySys_WriteStderr("Culverin parser [%s]: hot_path -> MONO_STUB[%zu]\n",
                          fastparser->parser_name ? fastparser->parser_name : "?",
                          fastparser->count);
    }
#endif
}

void FastParseDeinit(FastParserCore *fastparser) {
    if (!fastparser) {
        return;
    }

    if (fastparser->hot_specs) {
#pragma unroll 2
        for (size_t i = 0; i < fastparser->count; i++) {
            Py_XDECREF(fastparser->hot_specs[i].interned);
            fastparser->hot_specs[i].interned = nullptr;
        }
        Deallocate(fastparser->hot_specs);
        fastparser->hot_specs = nullptr;
    }

    if (fastparser->cold_specs) {
        Deallocate(fastparser->cold_specs);
        fastparser->cold_specs = nullptr;
    }

    if (fastparser->lookup_table) {
        Deallocate(fastparser->lookup_table);
        fastparser->lookup_table = nullptr;
    }
}

/** --- LEGACY HELPERS --- **/

[[gnu::always_inline]]
static inline ParseResult fast_parse_process_pos_legacy(const FastParserCore *fastparse,
                                                        PyObject *args, Py_ssize_t nargs,
                                                        const void *const *targets) {
#pragma unroll 2
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject *val              = PyTuple_GET_ITEM(args, i);
        const FastArgSpecHot *spec = &fastparse->hot_specs[i];

        // 1. Type Guard Check (Hot path metadata)
        if (!fast_parse_check_type_guard(spec, val)) {
            // Error reporting handles the hop to Cold specs
            return std::unexpected(
                FastParseReportTypeError(fastparse, static_cast<size_t>(i), val));
        }

        // 2. Conversion (Hot path function pointer)
        // Undecorated on purpose -- matches the pre-B3 behaviour of this path.
        if (Unlikely(!spec->convert(val, targets[i]))) {
            return CaptureFailure();
        }
    }
    return {};
}

[[gnu::always_inline]]
static inline ParseResult fast_parse_process_kw_legacy(const FastParserCore *fastparse,
                                                       PyObject *kwargs, uint64_t *mask,
                                                       const void *const *targets) {
    PyObject *key;
    PyObject *val;
    Py_ssize_t pos = 0;

    // Standard dictionary iteration
    while (PyDict_Next(kwargs, &pos, &key, &val)) {
        // Find index using hash table or linear search (interned pointer match)
        size_t idx = fast_parse_find_keyword_index(key, fastparse);

        if (Unlikely(idx == kFastParseEmptySlot)) {
            return std::unexpected(FastParseReportUnknownKeyword(fastparse, key));
        }

        // Check if this argument was already provided via positionals
        if (Unlikely(*mask & (1ULL << idx))) {
            return std::unexpected(FastParseReportMultiple(fastparse, idx));
        }

        const FastArgSpecHot *spec = &fastparse->hot_specs[idx];

        // 1. Type Guard Check (Hot path metadata)
        if (!fast_parse_check_type_guard(spec, val)) {
            return std::unexpected(FastParseReportTypeError(fastparse, idx, val));
        }

        // 2. Conversion (Hot path function pointer)
        // Undecorated on purpose -- matches the pre-B3 behaviour of this path.
        if (Unlikely(!spec->convert(val, targets[idx]))) {
            return CaptureFailure();
        }

        // Mark as provided
        *mask |= (1ULL << idx);
    }
    return {};
}

/** --- ORCHESTRATOR --- **/

[[nodiscard, gnu::hot, gnu::nonnull(4, 5), gnu::no_stack_protector, gnu::flatten, gnu::noinline]]
ParseResult FastParseLegacy(PyObject *args, PyObject *kwargs, [[maybe_unused]] PyObject *unused,
                            const FastParserCore *fastparser, const void *const targets[1]) {

    Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;

    // 1. Pre-flight Check
    if (Unlikely(nargs > static_cast<Py_ssize_t>(fastparser->count))) {
        return std::unexpected(FastParseReportTooMany(fastparser, nargs));
    }

    // 2. Handle Positionals
    if (nargs > 0) {
        if (ParseResult positionals =
                fast_parse_process_pos_legacy(fastparser, args, nargs, targets);
            Unlikely(!positionals)) {
            return positionals;
        }
    }

    // 3. Setup Mask & Handle Keywords
    uint64_t mask = fast_parse_make_mask(static_cast<size_t>(nargs));

    if (kwargs && PyDict_GET_SIZE(kwargs) > 0) {
        if (ParseResult keywords = fast_parse_process_kw_legacy(fastparser, kwargs, &mask, targets);
            Unlikely(!keywords)) {
            return keywords;
        }
    }

    // 4. Final Verification
    if (Unlikely((mask & fastparser->required_mask) != fastparser->required_mask)) {
        return std::unexpected(FastParseReportMissing(fastparser, mask));
    }

    return {};
}

} // namespace CPH::Python
