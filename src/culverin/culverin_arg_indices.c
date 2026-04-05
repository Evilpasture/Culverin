#include "culverin_arg_indices.h"

// --- 1. THE SAFETY ENGINE ---

// Helper macro to count schema entries at compile time
#define COUNT_X(ID, NAME, TYPE, REQ) +1

/**
 * INIT_PARSER_ST
 * Builds the spec array inside the struct and initializes the parser.
 */
#define INIT_PARSER_ST(cp, ParserName, GroupName, Schema)                                          \
    do {                                                                                           \
        static_assert((0 Schema(COUNT_X)) == GroupName##_COUNT,                                    \
                      "FastParse: Schema length mismatch for " #ParserName);                       \
        FastArgSpec temp[] = {Schema(GEN_SPEC)};                                                   \
        memcpy((cp)->ParserName##Specs, temp, sizeof(temp));                                       \
        fp_init_impl(&(cp)->ParserName##Parser, (cp)->ParserName##Specs, GroupName##_COUNT);       \
    } while (0)

// --- 2. REGISTRATION & SETUP MACROS ---

#define REGISTER_PARSER_ST(cp, ParserName)                                                         \
    do {                                                                                           \
        (cp)->ParserName##Parser.parser_name = #ParserName;                                        \
        if ((cp)->registry_count < PARSER_REGISTRY_SIZE) {                                         \
            (cp)->registry[(cp)->registry_count++] = &(cp)->ParserName##Parser;                    \
        }                                                                                          \
    } while (0)

#define SETUP_PARSER_ST(cp, ParserName, GroupName, Schema)                                         \
    do {                                                                                           \
        INIT_PARSER_ST(cp, ParserName, GroupName, Schema);                                         \
        REGISTER_PARSER_ST(cp, ParserName);                                                        \
    } while (0)

#define TEARDOWN_PARSER_ST(cp, ParserName)                                                         \
    fp_deinit(&(cp)->ParserName##Parser);

// --- 3. THE GENERATOR ---

#define GEN_SPEC(ID, NAME, TYPE, REQ)                                                              \
    [ID] = {.name      = (NAME),                                                                   \
            .type_name = #TYPE,                                                                    \
            .required  = (bool)(REQ),                                                              \
            .convert   = FP_GET_CONVERTER((TYPE){0})},

// --- 4. INITIALIZATION & CLEANUP ---

void culverin_init_all_parsers(CulverinParsers *cp) {
    // Reset registry for this specific interpreter
    cp->registry_count = 0; 

    #define DO_SETUP(P, G, S) SETUP_PARSER_ST(cp, P, G, S);
    FOR_ALL_PARSERS(DO_SETUP)
    #undef DO_SETUP
}

void culverin_free_all_parsers(CulverinParsers *cp) {
    #define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(cp, P);
    FOR_ALL_PARSERS(DO_FREE)
    #undef DO_FREE
}

void fp_dump_schemas_json(CulverinParsers *cp, FILE *out) {
    fprintf(out, "{\n");
    for (size_t i = 0; i < cp->registry_count; i++) {
        FastParser *fp = cp->registry[i];
        fprintf(out, "  \"%s\": [\n", fp->parser_name);
        for (size_t j = 0; j < fp->count; j++) {
            fprintf(out, "    {\"name\": \"%s\", \"type\": \"%s\", \"required\": %s}%s\n",
                    fp->specs[j].name, fp->specs[j].type_name,
                    (int)fp->specs[j].required ? "true" : "false", (j == fp->count - 1) ? "" : ",");
        }
        fprintf(out, "  ]%s\n", (i == cp->registry_count - 1) ? "" : ",");
    }
    fprintf(out, "}\n");
}