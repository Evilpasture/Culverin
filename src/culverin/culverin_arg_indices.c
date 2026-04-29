#include "culverin_arg_indices.h"

// --- 1. THE SAFETY ENGINE ---

// Helper macro to count schema entries at compile time
#define COUNT_X(ID, NAME, TYPE, REQ) +1

/**
 * INIT_PARSER_ST
 * Initializes the parser using a temporary stack-allocated definition array.
 * fp_init_impl will copy the data into its own optimized internal storage.
 */
#define INIT_PARSER_ST(cp, ParserName, GroupName, Schema)                                          \
    do {                                                                                           \
        static_assert((0 Schema(COUNT_X)) == GroupName##_COUNT,                                    \
                      "FastParse: Schema length mismatch for " #ParserName);                       \
        (cp)->ParserName##Parser.parser_name = #ParserName;                                        \
        /* 1. Create the definitions on the stack */                                               \
        FastArgDef temp[] = {Schema(GEN_SPEC)};                                                    \
        /* 2. Pass the stack pointer directly. fp_init_impl handles the mallocing now. */          \
        fp_init_impl(&(cp)->ParserName##Parser, temp, GroupName##_COUNT);                          \
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

#define TEARDOWN_PARSER_ST(cp, ParserName) fp_deinit(&(cp)->ParserName##Parser);

// --- 3. THE GENERATOR ---

#define GET_TYPE_GUARD(T) _Generic((T), bool: &PyBool_Type, default: (PyTypeObject *)nullptr)

#define GEN_SPEC(ID, NAME, TYPE, REQ)                                                              \
    [ID] = {.name       = (NAME),                                                                  \
            .type_name  = FP_GET_TYPE_NAME((TYPE){}),                                              \
            .required   = (bool)(REQ),                                                             \
            .type_guard = GET_TYPE_GUARD((TYPE){}),                                                \
            .convert    = FP_GET_CONVERTER((TYPE){})},

// --- 4. INITIALIZATION & CLEANUP ---

void culverin_init_world_parsers(WorldParsers *wp) {
    // Reset registry for this specific instance
    wp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(wp, P, G, S);
    FOR_ALL_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_world_parsers(WorldParsers *wp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(wp, P);
    FOR_ALL_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_char_parsers(CharacterParsers *cp) {
    // Reset registry for this specific instance
    cp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(cp, P, G, S);
    FOR_ALL_CHAR_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_char_parsers(CharacterParsers *cp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(cp, P);
    FOR_ALL_CHAR_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_vehicle_parsers(VehicleParsers *vp) {
    // Reset registry for this specific instance
    vp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(vp, P, G, S);
    FOR_ALL_VEHICLE_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_vehicle_parsers(VehicleParsers *vp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(vp, P);
    FOR_ALL_VEHICLE_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_ecs_parsers(ECSParsers *ep) {
    // Reset registry for this specific instance
    ep->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(ep, P, G, S);
    FOR_ALL_ECS_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_ecs_parsers(ECSParsers *ep) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(ep, P);
    FOR_ALL_ECS_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_skeleton_parsers(SkeletonParsers *sp) {
    sp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(sp, P, G, S);
    FOR_ALL_SKELETON_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_skeleton_parsers(SkeletonParsers *sp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(sp, P)
    FOR_ALL_SKELETON_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_ragdoll_parsers(RagdollParsers *rp) {
    rp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(rp, P, G, S);
    FOR_ALL_RAGDOLL_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_ragdoll_parsers(RagdollParsers *rp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(rp, P)
    FOR_ALL_RAGDOLL_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_ragdoll_settings_parsers(RagdollSettingsParsers *rsp) {
    rsp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(rsp, P, G, S);
    FOR_ALL_RAGDOLL_SETTINGS_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_ragdoll_settings_parsers(RagdollSettingsParsers *rsp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(rsp, P)
    FOR_ALL_RAGDOLL_SETTINGS_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_ship_parsers(ShipParsers *sp) {
    sp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(sp, P, G, S);
    FOR_ALL_SHIP_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_ship_parsers(ShipParsers *sp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(sp, P)
    FOR_ALL_SHIP_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_sbss_parsers(SoftBodySharedSettingsParsers *sbssp) {
    sbssp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(sbssp, P, G, S);
    FOR_ALL_SBSS_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_sbss_parsers(SoftBodySharedSettingsParsers *sbssp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(sbssp, P)
    FOR_ALL_SBSS_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_math_init_all_parsers(MathParsers *mp) {
    mp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(mp, P, G, S);
    FOR_ALL_MATH_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_math_free_all_parsers(MathParsers *mp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(mp, P)
    FOR_ALL_MATH_PARSERS(DO_FREE)
#undef DO_FREE
}

void culverin_init_module_parsers(ModuleParsers *mp) {
    mp->registry_count = 0;
#define DO_SETUP(P, G, S) SETUP_PARSER_ST(mp, P, G, S);
    FOR_ALL_MODULE_PARSERS(DO_SETUP)
#undef DO_SETUP
}

void culverin_free_module_parsers(ModuleParsers *mp) {
#define DO_FREE(P, G, S) TEARDOWN_PARSER_ST(mp, P);
    FOR_ALL_MODULE_PARSERS(DO_FREE)
#undef DO_FREE
}

void fp_dump_schemas_json(WorldParsers *wp, FILE *out) {
    fprintf(out, "{\n");
    for (size_t i = 0; i < wp->registry_count; i++) {
        FastParser *fp = wp->registry[i];
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