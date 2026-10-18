#include "bindings/parsing/ArgIndices.h"

// --- 1. THE SAFETY ENGINE ---

// Helper macro to count schema entries at compile time
#define CPH_COUNT_X(ID, NAME, TYPE, REQ) +1

/**
 * INIT_PARSER_ST
 * Initializes the parser using a temporary stack-allocated definition array.
 * culv_fast_parse_init_impl will copy the data into its own optimized internal storage.
 */
#define CPH_INIT_PARSER_ST(cp, ParserName, GroupName, Schema)                                          \
    do {                                                                                           \
        static_assert((0 Schema(CPH_COUNT_X)) == GroupName##_COUNT,                                    \
                      "Culverin parser: schema length mismatch for " #ParserName);                       \
        (cp)->ParserName##Parser.parser_name = #ParserName;                                        \
        /* 1. Create the definitions on the stack */                                               \
        CPH_FastArgDef temp[] = {Schema(CPH_GEN_SPEC)};                                                    \
        /* 2. Pass the stack pointer directly. culv_fast_parse_init_impl handles the mallocing now. */          \
        CPH_FastParseInitImpl(&(cp)->ParserName##Parser, temp, GroupName##_COUNT);                          \
    } while (0)

// --- 2. REGISTRATION & SETUP MACROS ---

#define CPH_REGISTER_PARSER_ST(cp, ParserName)                                                         \
    do {                                                                                           \
        (cp)->ParserName##Parser.parser_name = #ParserName;                                        \
        if ((cp)->registry_count < kParserRegistrySize) {                                         \
            (cp)->registry[(cp)->registry_count++] = &(cp)->ParserName##Parser;                    \
        }                                                                                          \
    } while (0)

#define CPH_SETUP_PARSER_ST(cp, ParserName, GroupName, Schema)                                         \
    do {                                                                                           \
        CPH_INIT_PARSER_ST(cp, ParserName, GroupName, Schema);                                         \
        CPH_REGISTER_PARSER_ST(cp, ParserName);                                                        \
    } while (0)

#define CPH_TEARDOWN_PARSER_ST(cp, ParserName) CPH_FastParseDeinit(&(cp)->ParserName##Parser);

// --- 3. THE GENERATOR ---

#define CPH_GET_TYPE_GUARD(T) _Generic((T), bool: &PyBool_Type, default: (PyTypeObject *)nullptr)

#define CPH_GEN_SPEC(ID, NAME, TYPE, REQ)                                                              \
    [ID] = {.name       = (NAME),                                                                  \
            .type_name  = CPH_FAST_PARSE_GET_TYPE_NAME((TYPE){}),                                              \
            .required   = (bool)(REQ),                                                             \
            .type_guard = CPH_GET_TYPE_GUARD((TYPE){}),                                                \
            .convert    = CPH_FAST_PARSE_GET_CONVERTER((TYPE){})},

// --- 4. INITIALIZATION & CLEANUP ---

void CPH_InitWorldParsers(CPH_WorldParsers *wp) {
    // Reset registry for this specific instance
    wp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(wp, P, G, S);
    CPH_FOR_ALL_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeWorldParsers(CPH_WorldParsers *wp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(wp, P);
    CPH_FOR_ALL_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitCharParsers(CPH_CharacterParsers *cp) {
    // Reset registry for this specific instance
    cp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(cp, P, G, S);
    CPH_FOR_ALL_CHAR_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeCharParsers(CPH_CharacterParsers *cp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(cp, P);
    CPH_FOR_ALL_CHAR_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitVehicleParserGroup(CPH_VehicleParsers *vp) {
    // Reset registry for this specific instance
    vp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(vp, P, G, S);
    CPH_FOR_ALL_VEHICLE_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeVehicleParserGroup(CPH_VehicleParsers *vp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(vp, P);
    CPH_FOR_ALL_VEHICLE_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitEcsParsers(CPH_ECSParsers *ep) {
    // Reset registry for this specific instance
    ep->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(ep, P, G, S);
    CPH_FOR_ALL_ECS_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeEcsParsers(CPH_ECSParsers *ep) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(ep, P);
    CPH_FOR_ALL_ECS_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitSkeletonParsers(CPH_SkeletonParsers *sp) {
    sp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(sp, P, G, S);
    CPH_FOR_ALL_SKELETON_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeSkeletonParsers(CPH_SkeletonParsers *sp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(sp, P)
    CPH_FOR_ALL_SKELETON_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitRagdollParsers(CPH_RagdollParsers *rp) {
    rp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(rp, P, G, S);
    CPH_FOR_ALL_RAGDOLL_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeRagdollParsers(CPH_RagdollParsers *rp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(rp, P)
    CPH_FOR_ALL_RAGDOLL_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitRagdollSettingsParsers(CPH_RagdollSettingsParsers *rsp) {
    rsp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(rsp, P, G, S);
    CPH_FOR_ALL_RAGDOLL_SETTINGS_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeRagdollSettingsParsers(CPH_RagdollSettingsParsers *rsp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(rsp, P)
    CPH_FOR_ALL_RAGDOLL_SETTINGS_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitShipParsers(CPH_ShipParsers *sp) {
    sp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(sp, P, G, S);
    CPH_FOR_ALL_SHIP_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeShipParsers(CPH_ShipParsers *sp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(sp, P)
    CPH_FOR_ALL_SHIP_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitSbssParsers(CPH_SoftBodySharedSettingsParsers *sbssp) {
    sbssp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(sbssp, P, G, S);
    CPH_FOR_ALL_SBSS_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeSbssParsers(CPH_SoftBodySharedSettingsParsers *sbssp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(sbssp, P)
    CPH_FOR_ALL_SBSS_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_MathInitAllParsers(CPH_MathParsers *mp) {
    mp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(mp, P, G, S);
    CPH_FOR_ALL_MATH_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_MathFreeAllParsers(CPH_MathParsers *mp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(mp, P)
    CPH_FOR_ALL_MATH_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_InitModuleParsers(CPH_ModuleParsers *mp) {
    mp->registry_count = 0;
#define CPH_DO_SETUP(P, G, S) CPH_SETUP_PARSER_ST(mp, P, G, S);
    CPH_FOR_ALL_MODULE_PARSERS(CPH_DO_SETUP)
#undef CPH_DO_SETUP
}

void CPH_FreeModuleParsers(CPH_ModuleParsers *mp) {
#define CPH_DO_FREE(P, G, S) CPH_TEARDOWN_PARSER_ST(mp, P);
    CPH_FOR_ALL_MODULE_PARSERS(CPH_DO_FREE)
#undef CPH_DO_FREE
}

void CPH_FastParseDumpSchemasJson(CPH_WorldParsers *wp, FILE *out) {
    fprintf(out, "{\n");
    for (size_t i = 0; i < wp->registry_count; i++) {
        CPH_FastParser *fp = wp->registry[i];
        fprintf(out, "  \"%s\": [\n", fp->parser_name);
        for (size_t j = 0; j < fp->count; j++) {
            // FIX: Access strings from cold_specs, but requirement status from the mask
            bool is_req = (fp->required_mask & (1ULL << j)) != 0;

            fprintf(out, "    {\"name\": \"%s\", \"type\": \"%s\", \"required\": %s}%s\n",
                    fp->cold_specs[j].name, fp->cold_specs[j].type_name, is_req ? "true" : "false",
                    (j == fp->count - 1) ? "" : ",");
        }
        fprintf(out, "  ]%s\n", (i == wp->registry_count - 1) ? "" : ",");
    }
    fprintf(out, "}\n");
}