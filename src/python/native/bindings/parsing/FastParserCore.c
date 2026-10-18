#include "FastParserCore.h"
#include "bindings/BindingCommon.h"
#include <stdlib.h>

static const CulvFastParseFunc kFastParseMonoStubs[] = {
    culv_fast_parse_speculate_p0,       culv_fast_parse_speculate_p1_naked, culv_fast_parse_speculate_p2_naked,
    culv_fast_parse_speculate_p3_naked, culv_fast_parse_speculate_p4_naked, culv_fast_parse_speculate_p5_naked,
    culv_fast_parse_speculate_p6_naked, culv_fast_parse_speculate_p7_naked, culv_fast_parse_speculate_p8_naked};

static constexpr size_t kFastParseStubsSize = 9;
static_assert(sizeof(kFastParseMonoStubs) / sizeof(CulvFastParseFunc) == kFastParseStubsSize,
              "CULV_FAST_PARSE_MONO_STUBS table must contain exactly 9 stubs (0-8 args)");

/**
 * COLD PATH: culv_fast_parse_warn_uninterned_slow
 *
 * This function handles the expensive string formatting and warning emission
 * for non-interned keywords. By marking it 'noinline' and 'cold', we ensure
 * that Clang keeps this code away from the high-speed 'culv_fast_parse_find_keyword_index'
 * instruction cache zone.
 */
[[gnu::noinline, gnu::cold, nodiscard]]
bool CPH_CulvFastParseWarnUninternedSlow(PyObject *key, const char *pname) {
    int result = PyErr_WarnFormat(PyExc_RuntimeWarning, 1,
                                  "Priority Alert: Keyword '%U' in parser '%s' is not interned.\n"
                                  "This forces a slow string comparison. To resolve this, use "
                                  "sys.intern() on keywords passed from Python logic.",
                                  key, pname ? pname : "unknown");
    return result >= 0;
}

[[gnu::noinline, gnu::cold, nodiscard]]
size_t CPH_CulvFastParseCmpSlow(PyObject key[static const restrict 1],
                   const CPH_CulvFastParser fastparse[static const restrict 1]) {
    const size_t count          = fastparse->count;
    const CPH_CulvFastArgSpecHot *specs = fastparse->hot_specs;

    for (size_t j = 0; j < count; ++j) {
        // 1. Is it the exact same pointer that just slipped past the hash table?
        if (key == specs[j].interned) {
            return j; // Return silently, no warning needed!
        }

        // 2. Slow string content comparison
        if (PyUnicode_Compare(key, specs[j].interned) == 0) {
            if ((fastparse->warned_mask & (1ULL << j)) == 0) {
                ((CPH_CulvFastParser *)fastparse)->warned_mask |= (1ULL << j);
                if (!CPH_CulvFastParseWarnUninternedSlow(key, fastparse->parser_name)) {
                    return kFastParseEmptySlot;
                }
            }
            return j;
        }
    }
    return kFastParseEmptySlot;
}

[[gnu::noinline, gnu::cold, nodiscard]]
bool CPH_CulvFastParseReportMissing(const CPH_CulvFastParser *fastparser, uint64_t provided_mask) {
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    PyObject *missing_details = PyList_New(0);
    PyObject *signature_parts = PyList_New(0);

    // If we can't allocate lists for the fancy error, fall back to a basic error
    if (CPH_FAST_PARSE_UNLIKELY(!missing_details || !signature_parts)) {
        Py_XDECREF(missing_details);
        Py_XDECREF(signature_parts);
        PyErr_Format(PyExc_TypeError, "%s() missing required positional arguments", pname);
        return false;
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        // DATA-ORIENTED SPLIT:
        // cold_specs has the strings (name, type_name)
        // hot_specs has the runtime objects (type_guard)
        const CPH_CulvFastArgSpecCold *cold = &fastparser->cold_specs[i];
        const CPH_CulvFastArgSpecHot *hot   = &fastparser->hot_specs[i];

        bool is_required = (fastparser->required_mask & (1ULL << i)) != 0;
        bool is_provided = (provided_mask & (1ULL << i)) != 0;

        // Resolve type information
        const char *fallback_name = hot->type_guard ? hot->type_guard->tp_name : "object";
        const char *type_info     = cold->type_name ? cold->type_name : fallback_name;

        // 1. Log specifically which ones are missing
        if (is_required && !is_provided) {
            PyObject *m = PyUnicode_FromFormat("    - %s (%s)", cold->name, type_info);
            if (m) {
                PyList_Append(missing_details, m);
                Py_DECREF(m);
            }
        }

        // 2. Build the full signature context [arg: type]
        PyObject *sig_part = (int)is_required
                                 ? PyUnicode_FromFormat("%s: %s", cold->name, type_info)
                                 : PyUnicode_FromFormat("[%s: %s]", cold->name, type_info);

        if (sig_part) {
            PyList_Append(signature_parts, sig_part);
            Py_DECREF(sig_part);
        }
    }

    Py_ssize_t num_missing = PyList_Size(missing_details);
    PyObject *newline      = PyUnicode_FromString("\n");
    PyObject *comma        = PyUnicode_FromString(", ");

    // Check for NULL before joining (OOM protection)
    PyObject *joined_missing = (newline) ? PyUnicode_Join(newline, missing_details) : nullptr;
    PyObject *joined_sig     = (comma) ? PyUnicode_Join(comma, signature_parts) : nullptr;

    if (joined_missing && joined_sig) {
        PyErr_Format(
            PyExc_TypeError,
            "%s() missing %zd required positional argument%s:\n%U\n\nExpected signature:\n  %s(%U)",
            pname, num_missing, num_missing == 1 ? "" : "s", joined_missing, pname, joined_sig);
    } else {
        // Fallback if string joining failed due to memory pressure
        PyErr_Format(PyExc_TypeError, "%s() missing %zd required positional arguments", pname,
                     num_missing);
    }

    // Cleanup: Py_XDECREF is safe for nullptrs
    Py_XDECREF(newline);
    Py_XDECREF(comma);
    Py_XDECREF(joined_missing);
    Py_XDECREF(joined_sig);
    Py_XDECREF(missing_details);
    Py_XDECREF(signature_parts);

    return false;
}
[[gnu::noinline, nodiscard]]
bool CPH_CulvFastParseReportTypeError(const CPH_CulvFastParser *fastparser, size_t index, PyObject *val) {
    const CPH_CulvFastArgSpecCold *cold_spec = &fastparser->cold_specs[index];
    const CPH_CulvFastArgSpecHot *hot_spec   = &fastparser->hot_specs[index];
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    // Resolve the "expected" type name: use type_name string if available,
    // else peek into the hot type_guard's tp_name.
    const char *fallback_name = (hot_spec->type_guard) ? hot_spec->type_guard->tp_name : "object";
    const char *expected      = (cold_spec->type_name) ? cold_spec->type_name : fallback_name;

    PyObject *val_repr        = nullptr;
    PyObject *signature_parts = nullptr;
    PyObject *comma           = nullptr;
    PyObject *joined_sig      = nullptr;

    // 1. Safe Repr Acquisition (Truncated to avoid overwhelming the error message)
    val_repr = PyObject_Repr(val);
    if (CPH_FAST_PARSE_UNLIKELY(!val_repr)) {
        PyErr_Clear(); // Clear OOM from Repr to set our TypeError
        val_repr = PyUnicode_FromString("<repr unavailable>");
    } else if (PyUnicode_GetLength(val_repr) > 100) {
        PyObject *truncated = PyUnicode_Substring(val_repr, 0, 97);
        if (truncated) {
            PyObject *fmt = PyUnicode_FromFormat("%U...", truncated);
            Py_DECREF(truncated);
            if (fmt) {
                Py_DECREF(val_repr);
                val_repr = fmt;
            }
        }
    }

    // 2. Diagnostic Signature Construction: (e.g. "func(arg1: int, !!! arg2: str !!!, [arg3:
    // float])")
    signature_parts = PyList_New(0);
    if (CPH_FAST_PARSE_UNLIKELY(!signature_parts)) {
        goto basic_error;
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        const CPH_CulvFastArgSpecCold *s_cold = &fastparser->cold_specs[i];
        const CPH_CulvFastArgSpecHot *s_hot   = &fastparser->hot_specs[i];

        bool is_required = (fastparser->required_mask & (1ULL << i)) != 0;

        const char *fb = s_hot->type_guard ? s_hot->type_guard->tp_name : "object";
        const char *t  = s_cold->type_name ? s_cold->type_name : fb;

        static const char *FMT_MAP[] = {"[%s: %s]", "%s: %s"};
        // Highlight the offending argument with exclamation marks
        const char *fmt = (i == index) ? "!!! %s: %s !!!" : FMT_MAP[(int)is_required];

        PyObject *part = PyUnicode_FromFormat(fmt, s_cold->name, t);
        if (part) {
            PyList_Append(signature_parts, part);
            Py_DECREF(part);
        }
    }

    comma = PyUnicode_FromString(", ");
    if (CPH_FAST_PARSE_UNLIKELY(!comma)) {
        goto basic_error;
    }

    joined_sig = PyUnicode_Join(comma, signature_parts);
    if (CPH_FAST_PARSE_UNLIKELY(!joined_sig)) {
        goto basic_error;
    }

    // 3. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "%s() argument '%s' must be %s, not %s.\n\n"
                 "Received value: %U\n"
                 "Signature context:\n  %s(%U)",
                 pname, cold_spec->name, expected, Py_TYPE(val)->tp_name,
                 val_repr ? val_repr : Py_None, pname, joined_sig);
    goto cleanup;

basic_error:
    // If we ran out of memory making the fancy message, give the standard Python-style one
    PyErr_Format(PyExc_TypeError, "%s() argument '%s' must be %s, not %s", pname, cold_spec->name,
                 expected, Py_TYPE(val)->tp_name);

cleanup:
    Py_XDECREF(comma);
    Py_XDECREF(joined_sig);
    Py_XDECREF(signature_parts);
    Py_XDECREF(val_repr);
    return false;
}
[[gnu::noinline, nodiscard]]
bool CPH_CulvFastParseReportMultiple(const CPH_CulvFastParser *fastparser, size_t index) {
    const char *pname    = fastparser->parser_name ? fastparser->parser_name : "function";
    const char *arg_name = fastparser->cold_specs[index].name;

    PyObject *parts = nullptr;
    PyObject *comma = nullptr;
    PyObject *sig   = nullptr;

    // 1. Diagnostic Signature Construction
    parts = PyList_New(0);
    if (CPH_FAST_PARSE_UNLIKELY(!parts)) {
        goto basic_error;
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        const char *name = fastparser->cold_specs[i].name;
        PyObject *p =
            (i == index) ? PyUnicode_FromFormat("!!! %s !!!", name) : PyUnicode_FromString(name);
        if (p) {
            PyList_Append(parts, p);
            Py_DECREF(p);
        }
    }

    comma = PyUnicode_FromString(", ");
    if (CPH_FAST_PARSE_UNLIKELY(!comma)) {
        goto basic_error;
    }

    sig = PyUnicode_Join(comma, parts);
    if (CPH_FAST_PARSE_UNLIKELY(!sig)) {
        goto basic_error;
    }

    // 2. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "%s() got multiple values for argument '%s'.\n"
                 "It was provided both as a positional argument and as a keyword argument.\n\n"
                 "Signature context:\n  %s(%U)",
                 pname, arg_name, pname, sig);
    goto cleanup;

basic_error:
    // Fallback if we can't allocate the fancy message
    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument '%s'", pname, arg_name);

cleanup:
    Py_XDECREF(comma);
    Py_XDECREF(sig);
    Py_XDECREF(parts);
    return false;
}
[[gnu::noinline]]
bool CPH_CulvFastParseReportTooMany(const CPH_CulvFastParser *fastparser, Py_ssize_t nargs) {
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    PyObject *parts = nullptr;
    PyObject *comma = nullptr;
    PyObject *sig   = nullptr;

    // 1. Calculate counts using the bitmask
    // We count set bits in required_mask to find how many args are mandatory.
    // Count set bits in the 64-bit mask using hardware instruction
    size_t req_count   = (size_t)__builtin_popcountll(fastparser->required_mask);
    size_t total_count = fastparser->count;

    char count_str[64];
    if (req_count == total_count) {
        snprintf(count_str, sizeof(count_str), "%zu", total_count);
    } else {
        snprintf(count_str, sizeof(count_str), "from %zu to %zu", req_count, total_count);
    }

    // 2. Diagnostic Signature Construction (Using Cold Spec names)
    parts = PyList_New(0);
    if (CPH_FAST_PARSE_UNLIKELY(!parts)) {
        goto basic_error;
    }

    for (size_t i = 0; i < total_count; i++) {
        // Names are stored in the Cold specs to keep the Hot path lean
        PyObject *p = PyUnicode_FromString(fastparser->cold_specs[i].name);
        if (p) {
            PyList_Append(parts, p);
            Py_DECREF(p);
        }
    }

    comma = PyUnicode_FromString(", ");
    if (CPH_FAST_PARSE_UNLIKELY(!comma)) {
        goto basic_error;
    }

    sig = PyUnicode_Join(comma, parts);
    if (CPH_FAST_PARSE_UNLIKELY(!sig)) {
        goto basic_error;
    }

    // 3. Final Fancy Format
    PyErr_Format(PyExc_TypeError,
                 "%s() takes %s positional arguments but %zd were given.\n\n"
                 "This function does not accept variable positional arguments (*args).\n"
                 "Expected sequence:\n  %s(%U)",
                 pname, count_str, nargs, pname, sig);
    goto cleanup;

basic_error:
    // Fallback if we can't allocate the fancy message strings
    PyErr_Format(PyExc_TypeError, "%s() takes %s positional arguments but %zd were given", pname,
                 count_str, nargs);

cleanup:
    Py_XDECREF(comma);
    Py_XDECREF(sig);
    Py_XDECREF(parts);
    return false;
}

[[gnu::always_inline, gnu::const]]
static inline int culv_fast_parse_min3(int a, int b, int c) {
    int m = a < b ? a : b;
    return m < c ? m : c;
}

static int culv_fast_parse_levenshtein(const char *s1, const char *s2) {
    // Assume s1 is the user input, s2 is the candidate from specs
    // Strip prefixes like 'ob_' or '_' if your interning logic needs it
    int len1 = (int)strlen(s1);
    int len2 = (int)strlen(s2);

    // Swap to ensure len1 >= len2 for smaller stack usage
    if (len1 < len2) {
        const char *tmp = s1;
        s1              = s2;
        s2              = tmp;
        int t           = len1;
        len1            = len2;
        len2            = t;
    }

    if (len2 == 0) {
        return len1;
    }
    if (len1 > 64) {
        len1 = 64; // Cap for safety
    }

    // uint8_t is enough for distance on 64-char strings
    alignas(16) uint8_t column[65];
#pragma unroll 4
    for (int i = 0; i <= len2; i++) {
        column[i] = (uint8_t)i;
    }
#pragma unroll 4
    for (int x = 1; x <= len1; x++) {
        column[0]         = (uint8_t)x;
        uint8_t last_diag = (uint8_t)(x - 1);
        for (int y = 1; y <= len2; y++) {
            uint8_t old_column = column[y];
            uint8_t cost       = (s1[x - 1] != s2[y - 1]);

            // Branchless min if your compiler doesn't do it automatically
            column[y] = culv_fast_parse_min3(column[y] + 1,     // Deletion
                                column[y - 1] + 1, // Insertion
                                last_diag + cost); // Substitution
            last_diag = old_column;
        }
    }
    return (int)column[len2];
}
[[gnu::noinline]]
bool CPH_CulvFastParseReportUnknownKeyword(const CPH_CulvFastParser *fastparser, PyObject *key) {
    const char *actual_name = PyUnicode_AsUTF8(key);
    if (!actual_name) {
        return false;
    }
    constexpr auto max_dist          = 3;
    const char *best_match           = nullptr;
    constexpr auto initial_best_dist = 100;
    int best_dist                    = initial_best_dist;
    const char *pname = fastparser->parser_name ? fastparser->parser_name : "function";

    PyObject *valid_args_list = nullptr;
    PyObject *newline         = nullptr;
    PyObject *joined          = nullptr;

    // 1. Find the best fuzzy match (Scanning Cold names)
    for (size_t i = 0; i < fastparser->count; i++) {
        const char *cand = fastparser->cold_specs[i].name;
        if (!cand) {
            continue;
        }

        int dist = culv_fast_parse_levenshtein(actual_name, cand);
        if (dist < best_dist) {
            best_dist  = dist;
            best_match = cand;
        }
    }

    // 2. Diagnostic Message Construction (Building the list of valid options)
    valid_args_list = PyList_New(0);
    if (CPH_FAST_PARSE_UNLIKELY(!valid_args_list)) {
        goto basic_error;
    }

    for (size_t i = 0; i < fastparser->count; i++) {
        const CPH_CulvFastArgSpecCold *cold = &fastparser->cold_specs[i];
        const CPH_CulvFastArgSpecHot *hot   = &fastparser->hot_specs[i];

        if (!cold->name) {
            continue;
        }

        bool is_required = (fastparser->required_mask & (1ULL << i)) != 0;

        // Resolve type name: Cold string preferred, fallback to Hot type_guard info
        const char *fb = hot->type_guard ? hot->type_guard->tp_name : "object";
        const char *t  = cold->type_name ? cold->type_name : fb;

        PyObject *info = PyUnicode_FromFormat("    - %s: %s%s", cold->name, t,
                                              (int)is_required ? " [required]" : "");
        if (info) {
            PyList_Append(valid_args_list, info);
            Py_DECREF(info);
        }
    }

    newline = PyUnicode_FromString("\n");
    if (CPH_FAST_PARSE_UNLIKELY(!newline)) {
        goto basic_error;
    }

    joined = PyUnicode_Join(newline, valid_args_list);
    if (CPH_FAST_PARSE_UNLIKELY(!joined)) {
        goto basic_error;
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
                 actual_name, pname, suggestion, joined);
    goto cleanup;

basic_error:
    // Fallback logic for low-memory situations
    if (best_match && best_dist < max_dist) {
        PyErr_Format(PyExc_TypeError,
                     "'%s' is an invalid keyword argument for %s(); did you mean '%s'?",
                     actual_name, pname, best_match);
    } else {
        PyErr_Format(PyExc_TypeError, "'%s' is an invalid keyword argument for %s()", actual_name,
                     pname);
    }

cleanup:
    Py_XDECREF(newline);
    Py_XDECREF(joined);
    Py_XDECREF(valid_args_list);
    return false;
}

[[gnu::noinline]]
bool CPH_CulvFastParseReportConversionError(const CPH_CulvFastParser *fastparser, size_t index, PyObject *val) {
    // 1. Capture the original "dry" error (e.g., OverflowError: "can't convert...")
    PyObject *exc = PyErr_GetRaisedException();
    if (CPH_FAST_PARSE_UNLIKELY(!exc)) {
        return false;
    }

    const char *pname    = fastparser->parser_name ? fastparser->parser_name : "function";
    const char *arg_name = fastparser->cold_specs[index].name;

    // 2. Best-effort acquisition of Value Repr
    PyObject *val_repr = PyObject_Repr(val);
    if (CPH_FAST_PARSE_UNLIKELY(!val_repr)) {
        PyErr_Clear(); // Clear OOM from repr attempt
        val_repr = PyUnicode_FromString("<repr unavailable>");
    }

    // 3. Setup diagnostics components (OOM-vulnerable block)
    PyObject *parts   = PyList_New(0);
    PyObject *comma   = PyUnicode_FromString(", ");
    PyObject *sig     = nullptr;
    PyObject *exc_str = PyObject_Str(exc);
    PyObject *new_msg = nullptr;

    // If we failed basic allocations, jump to the safety fallback
    if (CPH_FAST_PARSE_UNLIKELY(!parts || !comma || !exc_str)) {
        goto oom_fallback;
    }

    // Build the fancy signature context
    for (size_t i = 0; i < fastparser->count; i++) {
        const char *name = fastparser->cold_specs[i].name;
        const char *type = fastparser->cold_specs[i].type_name;
        const char *fmt  = (i == index) ? "!!! %s: %s !!!" : "%s: %s";
        PyObject *p      = PyUnicode_FromFormat(fmt, name, type ? type : "object");
        if (p) {
            PyList_Append(parts, p);
            Py_DECREF(p);
        }
    }

    sig = PyUnicode_Join(comma, parts);
    if (CPH_FAST_PARSE_UNLIKELY(!sig)) {
        goto oom_fallback;
    }

    // 4. Final assembly of the high-fidelity message
    new_msg = PyUnicode_FromFormat("%s() argument '%s': %U\n"
                                   "Received value: %U\n\n"
                                   "Signature context:\n  %s(%U)",
                                   pname, arg_name, exc_str, val_repr, pname, sig);

    if (CPH_FAST_PARSE_LIKELY(new_msg)) {
        PyErr_SetObject((PyObject *)Py_TYPE(exc), new_msg);
        Py_DECREF(exc); // Handled by SetObject creating a new instance
    } else {
        goto oom_fallback;
    }

    goto cleanup;

oom_fallback:
    // Safety Valve: If any allocation failed, re-raise the original exception.
    // This ensures the user still sees the dry error rather than an OOM error.
    PyErr_SetRaisedException(exc);

cleanup:
    Py_XDECREF(val_repr);
    Py_XDECREF(parts);
    Py_XDECREF(comma);
    Py_XDECREF(sig);
    Py_XDECREF(exc_str);
    Py_XDECREF(new_msg);
    return false;
}

void CPH_CulvFastParseInitImpl(CPH_CulvFastParser *fastparser, const CPH_CulvFastArgDef *defs, size_t count) {
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
    fastparser->hot_specs  = (CPH_CulvFastArgSpecHot *)CPH_PyAlloc(count * sizeof(CPH_CulvFastArgSpecHot));
    fastparser->cold_specs = (CPH_CulvFastArgSpecCold *)CPH_PyAlloc(count * sizeof(CPH_CulvFastArgSpecCold));

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

        fastparser->table_mask   = table_size - 1;
        fastparser->lookup_table = (uint16_t *)CPH_PyAlloc(table_size * sizeof(uint16_t));
        if (!fastparser->lookup_table) {
            Py_FatalError("Culverin parser: failed to allocate lookup table.");
        }

        for (size_t i = 0; i < table_size; i++) {
            fastparser->lookup_table[i] = kFastParseEmptySlot;
        }

        for (size_t i = 0; i < count; i++) {
            size_t hash = culv_fast_parse_hash_ptr(fastparser->hot_specs[i].interned, fastparser->table_mask);
            while (fastparser->lookup_table[hash] != kFastParseEmptySlot) {
                hash = (hash + 1) & fastparser->table_mask;
            }
            fastparser->lookup_table[hash] = (uint16_t)i;
        }
    }

    // 3. Path Selection
    fastparser->hot_path = culv_fast_parse_vector;

    // Optimization: If all args are required and no type guards are present,
    // we can use the specialized monomorphic stubs.
    uint64_t all_required = (count >= 64) ? ~0ULL : ((1ULL << count) - 1);
    if (fastparser->required_mask == all_required && all_required != 0) {
        if (count < (sizeof(kFastParseMonoStubs) / sizeof(CulvFastParseFunc))) {
            fastparser->hot_path = kFastParseMonoStubs[count];
        }
    }
#if defined(CULVERIN_DEBUG)
    if (fastparser->hot_path == culv_fast_parse_vector) {
        PySys_WriteStderr("Culverin parser [%s]: hot_path fell back to culv_fast_parse_vector "
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

void CPH_CulvFastParseDeinit(CPH_CulvFastParser *fastparser) {
    if (!fastparser) {
        return;
    }

    if (fastparser->hot_specs) {
#pragma unroll 2
        for (size_t i = 0; i < fastparser->count; i++) {
            Py_XDECREF(fastparser->hot_specs[i].interned);
            fastparser->hot_specs[i].interned = nullptr;
        }
        CPH_PyFree(fastparser->hot_specs);
        fastparser->hot_specs = nullptr;
    }

    if (fastparser->cold_specs) {
        CPH_PyFree(fastparser->cold_specs);
        fastparser->cold_specs = nullptr;
    }

    if (fastparser->lookup_table) {
        CPH_PyFree(fastparser->lookup_table);
        fastparser->lookup_table = nullptr;
    }
}

/** --- LEGACY HELPERS --- **/

[[gnu::always_inline]]
static inline bool
culv_fast_parse_process_pos_legacy(const CPH_CulvFastParser fastparse[static const restrict 1],
                      PyObject args[static const restrict 1], Py_ssize_t nargs,
                      const void *const CPH_FAST_PARSE_RESTRICT targets[static const restrict 1]) {
#pragma unroll 2
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        PyObject *val              = PyTuple_GET_ITEM(args, i);
        const CPH_CulvFastArgSpecHot *spec = &fastparse->hot_specs[i];

        // 1. Type Guard Check (Hot path metadata)
        if (!culv_fast_parse_check_type_guard(spec, val)) {
            // Error reporting handles the hop to Cold specs
            return CPH_CulvFastParseReportTypeError(fastparse, (size_t)i, val);
        }

        // 2. Conversion (Hot path function pointer)
        if (CPH_FAST_PARSE_UNLIKELY(!spec->convert(val, targets[i]))) {
            return false;
        }
    }
    return true;
}

[[gnu::always_inline]]
static inline bool
culv_fast_parse_process_kw_legacy(const CPH_CulvFastParser fastparse[static const restrict 1],
                     PyObject kwargs[static const restrict 1],
                     uint64_t mask[static const restrict 1],
                     const void *const CPH_FAST_PARSE_RESTRICT targets[static const restrict 1]) {
    PyObject *key;
    PyObject *val;
    Py_ssize_t pos = 0;

    // Standard dictionary iteration
    while (PyDict_Next(kwargs, &pos, &key, &val)) {
        // Find index using hash table or linear search (interned pointer match)
        size_t idx = culv_fast_parse_find_keyword_index(key, fastparse);

        if (CPH_FAST_PARSE_UNLIKELY(idx == kFastParseEmptySlot)) {
            return CPH_CulvFastParseReportUnknownKeyword(fastparse, key);
        }

        // Check if this argument was already provided via positionals
        if (CPH_FAST_PARSE_UNLIKELY(*mask & (1ULL << idx))) {
            return CPH_CulvFastParseReportMultiple(fastparse, idx);
        }

        const CPH_CulvFastArgSpecHot *spec = &fastparse->hot_specs[idx];

        // 1. Type Guard Check (Hot path metadata)
        if (!culv_fast_parse_check_type_guard(spec, val)) {
            return CPH_CulvFastParseReportTypeError(fastparse, idx, val);
        }

        // 2. Conversion (Hot path function pointer)
        if (CPH_FAST_PARSE_UNLIKELY(!spec->convert(val, targets[idx]))) {
            return false;
        }

        // Mark as provided
        *mask |= (1ULL << idx);
    }
    return true;
}

/** --- ORCHESTRATOR --- **/

[[nodiscard, gnu::hot, gnu::nonnull(4, 5), gnu::no_stack_protector, gnu::flatten, gnu::noinline]]
bool CPH_CulvFastParseLegacy(PyObject args[static const CPH_FAST_PARSE_RESTRICT 1], PyObject kwargs[const CPH_FAST_PARSE_RESTRICT],
                     [[maybe_unused]] PyObject *unused,
                     const CPH_CulvFastParser fastparser[static const CPH_FAST_PARSE_RESTRICT 1],
                     const void *const CPH_FAST_PARSE_RESTRICT targets[static const CPH_FAST_PARSE_RESTRICT 1]) {

    Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;

    // 1. Pre-flight Check
    if (CPH_FAST_PARSE_UNLIKELY(nargs > (Py_ssize_t)fastparser->count)) {
        return CPH_CulvFastParseReportTooMany(fastparser, nargs);
    }

    // 2. Handle Positionals
    if (nargs > 0) {
        if (CPH_FAST_PARSE_UNLIKELY(!culv_fast_parse_process_pos_legacy(fastparser, args, nargs, targets))) {
            return false;
        }
    }

    // 3. Setup Mask & Handle Keywords
    uint64_t mask = culv_fast_parse_make_mask((size_t)nargs);

    if (kwargs && PyDict_GET_SIZE(kwargs) > 0) {
        if (CPH_FAST_PARSE_UNLIKELY(!culv_fast_parse_process_kw_legacy(fastparser, kwargs, &mask, targets))) {
            return false;
        }
    }

    // 4. Final Verification
    if (CPH_FAST_PARSE_UNLIKELY((mask & fastparser->required_mask) != fastparser->required_mask)) {
        return CPH_CulvFastParseReportMissing(fastparser, mask);
    }

    return true;
}