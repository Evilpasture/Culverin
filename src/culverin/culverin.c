// clang-format off
#if !defined(_CRT_SECURE_NO_WARNINGS)
#    define _CRT_SECURE_NO_WARNINGS
#endif
// clang-format on
#include "culverin.h"
#include "culverin_arg_indices.h"
#include "culverin_character.h"
#include "culverin_compiler_specifics.h"
#include "culverin_contact_listener.h"
#include "culverin_debug_render.h"
#include "culverin_filters.h"
#include "culverin_getters.h"
#include "culverin_handler.h"
#include "culverin_math.h"
#include "culverin_module.h"
#include "culverin_python.h"
#include "culverin_threading.h"
#include "docs_embedder.h"
#include "joltc.h"
#include <stdatomic.h>

/**
 * INTERNAL HELPER: dump_parser_registry
 * Iterates through a specific registry array and writes the JSON entries.
 */
static void dump_parser_registry(FastParser *const *const registry, const size_t count,
                                 FILE *const out, bool *const is_first) {
    for (size_t i = 0; i < count; ++i) {
        const auto fp = registry[i];
        if (!(*is_first)) {
            fprintf(out, ",\n");
        }
        *is_first = false;

        fprintf(out, "  \"%s\": [\n", fp->parser_name);
        for (size_t j = 0; j < fp->count; ++j) {
            const bool is_req = (fp->required_mask & (1ULL << j)) != 0;
            fprintf(out, "    {\"name\": \"%s\", \"type\": \"%s\", \"required\": %s}%s\n",
                    fp->cold_specs[j].name, fp->cold_specs[j].type_name, is_req ? "true" : "false",
                    (j == fp->count - 1) ? "" : ",");
        }
        fprintf(out, "  ]");
    }
}

/**
 * PyCFunction: dump_schema
 * Initializes every known parser group, dumps them to a unified JSON, and cleans up.
 */
PyCFunction_DeclareMethod culv_dump_schema_json(PyObject *const self, PyObject *const *args,
                                                size_t nargsf, PyObject *kwnames) {
    const auto st        = get_culverin_state(self);
    const char *filename = "culverin_schema.json";

    // 1. Thread-safe, subinterpreter-safe parsing via FastParser
    void *targets[DumpSchema_COUNT] = {[IDX_DS_PATH] = (void *)&filename};

    if (!FastParse_Unified(args, PyVectorcall_NARGS(nargsf), kwnames, &st->parsers.DumpSchemaParser,
                           targets)) {
        return nullptr;
    }

    FILE *const f = fopen(filename, "w");
    if (f == nullptr) {
        return PyErr_SetFromErrno(PyExc_IOError);
    }

    // 2. Initialize temporary registries to extract cold metadata
    struct {
        WorldParsers world;
        CharacterParsers character;
        VehicleParsers vehicle;
        ECSParsers ecs;
        SkeletonParsers skeleton;
        RagdollParsers ragdoll;
        RagdollSettingsParsers ragdoll_settings;
        ShipParsers ship;
        SoftBodySharedSettingsParsers sbss;
        MathParsers math;
        ModuleParsers module;
    } all = {};

    culverin_init_world_parsers(&all.world);
    culverin_init_char_parsers(&all.character);
    culverin_init_vehicle_parsers(&all.vehicle);
    culverin_init_ecs_parsers(&all.ecs);
    culverin_init_skeleton_parsers(&all.skeleton);
    culverin_init_ragdoll_parsers(&all.ragdoll);
    culverin_init_ragdoll_settings_parsers(&all.ragdoll_settings);
    culverin_init_ship_parsers(&all.ship);
    culverin_init_sbss_parsers(&all.sbss);
    culverin_math_init_all_parsers(&all.math);
    culverin_init_module_parsers(&all.module);

    // 3. Export to JSON
    fprintf(f, "{\n");
    bool is_first = true;

#define DUMP_GROUP(member)                                                                         \
    dump_parser_registry(all.member.registry, all.member.registry_count, f, &is_first)

    DUMP_GROUP(world);
    DUMP_GROUP(character);
    DUMP_GROUP(vehicle);
    DUMP_GROUP(ecs);
    DUMP_GROUP(skeleton);
    DUMP_GROUP(ragdoll);
    DUMP_GROUP(ragdoll_settings);
    DUMP_GROUP(ship);
    DUMP_GROUP(sbss);
    DUMP_GROUP(math);
    DUMP_GROUP(module);

#undef DUMP_GROUP

    fprintf(f, "\n}\n");
    fclose(f);

    // 4. Cleanup temporary allocations
    culverin_free_world_parsers(&all.world);
    culverin_free_char_parsers(&all.character);
    culverin_free_vehicle_parsers(&all.vehicle);
    culverin_free_ecs_parsers(&all.ecs);
    culverin_free_skeleton_parsers(&all.skeleton);
    culverin_free_ragdoll_parsers(&all.ragdoll);
    culverin_free_ragdoll_settings_parsers(&all.ragdoll_settings);
    culverin_free_ship_parsers(&all.ship);
    culverin_free_sbss_parsers(&all.sbss);
    culverin_math_free_all_parsers(&all.math);
    culverin_free_module_parsers(&all.module);

    Py_RETURN_NONE;
}

#include <stdatomic.h>
#include <tupleobject.h>

PyCFunction_DeclareMethod culv_mutate_tuple(CULV_MAYBE_UNUSED PyObject *self, PyObject *const *args,
                                            Py_ssize_t nargsf) {
    enum : uint8_t { ARG_0, ARG_1, ARG_2, ARG_3, ARG_4 };
    // --- 1. DECLARATIONS (C / goto safety) ---
    auto nargs              = PyVectorcall_NARGS(nargsf);
    constexpr auto MIN_ARGS = 3;
    constexpr auto MAX_ARGS = 5;

    PyObject *target   = nullptr;
    PyObject *new_val  = nullptr;
    PyObject *registry = nullptr;
    PyObject *key      = nullptr;
    PyObject *old_val  = nullptr;
    PyObject *stored   = nullptr;

    Py_ssize_t index     = 0;
    Py_hash_t final_hash = -1;
    int success          = 0;

    // --- 2. PRE-FLIGHT CHECKS ---
    if (nargs != MIN_ARGS && nargs != MAX_ARGS) {
        PyErr_Format(PyExc_TypeError, "mutate_tuple() takes 3 or 5 arguments (%zd given)", nargs);
        return nullptr;
    }

    target  = args[ARG_0];
    new_val = args[ARG_2];
    if (!PyTuple_Check(target)) {
        PyErr_SetString(PyExc_TypeError, "arg 0 must be a tuple");
        return nullptr;
    }

    index = PyLong_AsSsize_t(args[ARG_1]);
    if (index == -1 && PyErr_Occurred()) {
        return nullptr;
    }

    Py_ssize_t tuple_len = Py_SIZE(target);
    if (index < 0) {
        index += tuple_len;
    }
    if (index < 0 || index >= tuple_len) {
        PyErr_Format(PyExc_IndexError, "tuple index %zd out of range", index);
        return nullptr;
    }

    if (nargs == MAX_ARGS) {
        registry = args[ARG_3];
        key      = args[ARG_4];
        if (!PyDict_Check(registry)) {
            PyErr_SetString(PyExc_TypeError, "arg 3 must be a dict");
            return nullptr;
        }
    }

    // --- 3. CRITICAL SECTION ---
#if defined(Py_BEGIN_CRITICAL_SECTION)
    Py_BEGIN_CRITICAL_SECTION(target);
#endif

    // Step A: Registry Removal
    if (registry) {
        stored = PyDict_GetItemWithError(registry, key);
        if (!stored) {
            if (!PyErr_Occurred()) {
                PyErr_SetObject(PyExc_KeyError, key);
            }
            goto exit_critical;
        }
        if (stored != target) {
            PyErr_SetString(PyExc_ValueError, "registry[key] is not the same object as target");
            goto exit_critical;
        }
        Py_INCREF(target);
        if (PyDict_DelItem(registry, key) < 0) {
            Py_DECREF(target);
            goto exit_critical;
        }
    }

    // Step B: The Mutation (Atomic Swap)
    {
        PyTupleObject *t = (PyTupleObject *)target;
        Py_INCREF(new_val);

        // Cast the item slot to an atomic pointer.
        // This prevents TSan "Data Race" warnings and ensures memory visibility.
        _Atomic(PyObject *) *atomic_slot = (_Atomic(PyObject *) *)&(t->ob_item[index]);
        old_val = atomic_exchange_explicit(atomic_slot, new_val, memory_order_acq_rel);

        // Step C: Rehash Cache Bust (Atomic)
#if PY_VERSION_HEX >= 0x030E0000
        _Atomic(Py_hash_t) *atomic_hash = (_Atomic(Py_hash_t) *)&(t->ob_hash);
        atomic_store_explicit(atomic_hash, -1, memory_order_relaxed);
#endif
    }

    final_hash = PyObject_Hash(target);

    if (final_hash == -1) {
        // Rollback (Atomic)
        _Atomic(PyObject *) *atomic_slot =
            (_Atomic(PyObject *) *)&((PyTupleObject *)target)->ob_item[index];
        atomic_store_explicit(atomic_slot, old_val, memory_order_relaxed);
        Py_DECREF(new_val);
        if (registry) {
            Py_DECREF(target);
        }
        goto exit_critical;
    }

    // Step D: Registry Re-insertion
    if (registry) {
        if (PyDict_SetItem(registry, key, target) < 0) {
            // Rollback (Atomic)
            _Atomic(PyObject *) *atomic_slot =
                (_Atomic(PyObject *) *)&((PyTupleObject *)target)->ob_item[index];
            atomic_store_explicit(atomic_slot, old_val, memory_order_relaxed);
            Py_DECREF(new_val);
            Py_DECREF(target);
            final_hash = -1;
            goto exit_critical;
        }
        Py_DECREF(target);
    }

    Py_DECREF(old_val);
    success = 1;

exit_critical:
#if defined(Py_BEGIN_CRITICAL_SECTION)
    Py_END_CRITICAL_SECTION();
#endif

    if (!success) {
        return nullptr;
    }
    return PyLong_FromSsize_t(final_hash);
}

// --- The Documentation System ---

#ifndef CULVERIN_DOCS_PATH
#    define CULVERIN_DOCS_PATH "../../docs/DOCS.md"
#endif

static const unsigned char ALL_DOCS[] = {
#ifdef __has_embed
#    if __has_embed(CULVERIN_DOCS_PATH)
#        embed CULVERIN_DOCS_PATH suffix(, 0)
#        define CULV_DOCS_EMBEDDED
#    endif
#endif

#ifndef CULV_DOCS_EMBEDDED
#    include "ALL_DOCS.inc"
#endif
};

// Global flag to ensure we only stitch once (important for subinterpreters)
static atomic_int docs_status = 0;

// =============================================================================================

// --- Macros ---

// User-facing macros for module-level methods
#define MOD_FASTCALL(name) CULV_FEAT(culv, name, METH_FASTCALL | METH_KEYWORDS)
#define MOD_NOARGS(name) CULV_FEAT(culv, name, METH_NOARGS)
#define MOD_VARARGS(name) CULV_FEAT(culv, name, METH_VARARGS | METH_KEYWORDS)

#define MOD_NOARGS_INTERNAL(name) CULV_FEAT_INTERNAL(culv, name, METH_NOARGS)
#define MOD_FASTCALL_INTERNAL(name) CULV_FEAT_INTERNAL(culv, name, METH_FASTCALL | METH_KEYWORDS)

// --- Module Initialization ---

// Embed the entire TOML file as a static string
static const unsigned char PYPROJECT_TOML[] = {
#ifdef __has_embed
#    if __has_embed("../../pyproject.toml")
#        embed "../../pyproject.toml" suffix(, 0)
#        define CULV_TOML_EMBEDDED
#    endif
#endif

#ifndef CULV_TOML_EMBEDDED
#    include "PYPROJECT_TOML.inc"
#endif
};

// Helper function to extract the version string at runtime
static const char *extract_version_from_toml(void) {
    // Look for the specific pattern 'version = "'
    const char *key = "version = \"";
    auto start      = strstr((char *)PYPROJECT_TOML, key);

    if (!start) {
        return "0.0.0-unknown";
    }

    start += strlen(key); // Move pointer to the start of the actual version number

    // Find the closing quote
    const char *end = strchr(start, '\"');
    if (!end) {
        return "0.0.0-malformed";
    }

    // Create a static buffer to hold the version (e.g., "0.4.0")
    static constexpr size_t VERSION_BUFFER = 32;
    static char version_buffer[VERSION_BUFFER];
    size_t len = end - start;
    if (len >= sizeof(version_buffer)) {
        len = sizeof(version_buffer) - 1;
    }

    strncpy(version_buffer, start, len);
    version_buffer[len] = '\0';

    return version_buffer;
}

static int init_types(PyObject *m, CulverinState *st) {
    struct {
        PyType_Spec *spec;
        PyObject **slot;
        const char *name;
    } types[] = {
        {.spec = (&PhysicsWorld_spec), .slot = &st->PhysicsWorldType, .name = "PhysicsWorld"},
        {.spec = (&Character_spec), .slot = &st->CharacterType, .name = "Character"},
        {.spec = (&Vehicle_spec), .slot = &st->VehicleType, .name = "Vehicle"},
        {.spec = (&Ship_spec), .slot = &st->ShipType, .name = "Ship"},
        {.spec = (&RagdollSettings_spec),
         .slot = &st->RagdollSettingsType,
         .name = "RagdollSettings"},
        {.spec = (&Ragdoll_spec), .slot = &st->RagdollType, .name = "Ragdoll"},
        {.spec = (&Skeleton_spec), .slot = &st->SkeletonType, .name = "Skeleton"},
        {.spec = (&BufferProxy_spec), .slot = &st->BufferProxyType, .name = "BufferProxyObject"},
        {.spec = (&SoftBodySharedSettings_spec),
         .slot = &st->SoftBodySharedSettingsType,
         .name = "SoftBodySharedSettings"},
        {.spec = (&Registry_spec), .slot = &st->RegistryType, .name = "Registry"},
        {.spec = (&MathService_spec), .slot = &st->MathServiceType, .name = "MathService"}};

    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); i++) {
        auto type = PyType_FromModuleAndSpec(m, types[i].spec, nullptr);
        if (!type) {
            return -1;
        }
        auto mod_name = PyUnicode_FromString("culverin");
        if (!mod_name) {
            Py_DECREF(type);
            return -1;
        }
        PyObject_SetAttrString(type, "__module__", mod_name);
        if (PyModule_AddObject(m, types[i].name, type) < 0) {
            Py_DECREF(type);
            return -1;
        }
        Py_DECREF(mod_name);
        *types[i].slot = type;
    }
    return 0;
}

static int init_constants(PyObject *m) {
    static const struct {
        const char *name;
        int value;
    } consts[] = {{.name = "SHAPE_BOX", .value = CULV_SHAPE_BOX},
                  {.name = "SHAPE_SPHERE", .value = CULV_SHAPE_SPHERE},
                  {.name = "SHAPE_CAPSULE", .value = CULV_SHAPE_CAPSULE},
                  {.name = "SHAPE_CYLINDER", .value = CULV_SHAPE_CYLINDER},
                  {.name = "SHAPE_PLANE", .value = CULV_SHAPE_PLANE},
                  {.name = "SHAPE_MESH", .value = CULV_SHAPE_MESH},
                  {.name = "SHAPE_HEIGHTFIELD", .value = CULV_SHAPE_HEIGHTFIELD},
                  {.name = "SHAPE_CONVEX_HULL", .value = CULV_SHAPE_CONVEX_HULL},
                  {.name = "MOTION_STATIC", .value = MOTION_STATIC},
                  {.name = "MOTION_KINEMATIC", .value = MOTION_KINEMATIC},
                  {.name = "MOTION_DYNAMIC", .value = MOTION_DYNAMIC},
                  {.name = "CONSTRAINT_FIXED", .value = CONSTRAINT_FIXED},
                  {.name = "CONSTRAINT_POINT", .value = CONSTRAINT_POINT},
                  {.name = "CONSTRAINT_HINGE", .value = CONSTRAINT_HINGE},
                  {.name = "CONSTRAINT_SLIDER", .value = CONSTRAINT_SLIDER},
                  {.name = "CONSTRAINT_DISTANCE", .value = CONSTRAINT_DISTANCE},
                  {.name = "CONSTRAINT_CONE", .value = CONSTRAINT_CONE},
                  {.name = "EVENT_ADDED", .value = EVENT_ADDED},
                  {.name = "EVENT_PERSISTED", .value = EVENT_PERSISTED},
                  {.name = "EVENT_REMOVED", .value = EVENT_REMOVED},
// Build Metadata
#if defined(JPH_DOUBLE_PRECISION)
                  {.name = "USE_DOUBLE_PRECISION", .value = 1},
#else
                  {.name = "USE_DOUBLE_PRECISION", .value = 0},
#endif
                  {.name = "FREE_THREADED",
                   .value =
#if defined(Py_GIL_DISABLED) && Py_GIL_DISABLED
                       1
#else
                       0
#endif
                  },
                  {.name = "DEBUG_BUILD",
                   .value =
#if defined(CULVERIN_DEBUG)
                       1
#else
                       0
#endif
                  },
                  {.name = "BEND_DIHEDRAL", .value = JPH_SoftBodyBendType_Dihedral},
                  {.name = "BEND_DISTANCE", .value = JPH_SoftBodyBendType_Distance},
                  {.name = "BEND_NONE", .value = JPH_SoftBodyBendType_None}};

    for (size_t i = 0; i < sizeof(consts) / sizeof(consts[0]); i++) {
        if (PyModule_AddIntConstant(m, consts[i].name, consts[i].value) < 0) {
            return -1;
        }
    }
    return 0;
}

static constexpr auto MAGIC_BUFFER = 128;
static char shared_version[MAGIC_BUFFER];

extern PyModuleDef culverin_module;

PyType_DeclareSlot_Status culverin_exec(PyObject *m) {
    CulverinState *st = get_culverin_state(m);
    // 1. THE MASTER GATE: Protects all static global memory in the process
    int expected = 0;
    if (atomic_compare_exchange_strong(&docs_status, &expected, 1)) {
        // --- 1A. VERSION & BUILD METADATA ---
        const char *ver_temp = extract_version_from_toml();

        const char *gil_status =
#if defined(Py_GIL_DISABLED) && Py_GIL_DISABLED
            "free-threaded";
#else
            "gil-enabled";
#endif

        const char *precision =
#if JPH_DOUBLE_PRECISION
            "double-precision";
#else
            "single-precision";
#endif

        const char *build_type =
#if defined(ENABLE_SANITIZER) || defined(__SANITIZE_THREAD__)
            "sanitized";
#elif defined(CULVERIN_DEBUG)
            "debug";
#else
                    "release";
#endif
        // Determine Compiler ID
        const char *compiler_id =
#if defined(__clang__)
            "Clang " __clang_version__;
#elif defined(__GNUC__)
            "GCC " __VERSION__;
#elif defined(_MSC_VER)
            "MSVC " _CRT_STRINGIZE(_MSC_VER);
#else
                "Unknown Compiler";
#endif

        // Result: e.g. "0.6.0 (free-threaded, double-precision, release, Clang 22.1.0)"
        snprintf(shared_version, MAGIC_BUFFER, "%s (%s, %s, %s, %s)", ver_temp, gil_status,
                 precision, build_type, compiler_id);

        // --- THE WINNER: Run exactly once per process life ---
        const char *docs_str = (const char *)ALL_DOCS;
        md_stitch_methods(culverin_module.m_methods, "Module", docs_str);
        md_stitch_spec(&PhysicsWorld_spec, "PhysicsWorld", docs_str);
        md_stitch_spec(&Character_spec, "Character", docs_str);
        md_stitch_spec(&Vehicle_spec, "Vehicle", docs_str);
        md_stitch_spec(&Skeleton_spec, "Skeleton", docs_str);
        md_stitch_spec(&Ragdoll_spec, "Ragdoll", docs_str);
        md_stitch_spec(&RagdollSettings_spec, "RagdollSettings", docs_str);
        md_stitch_spec(&SoftBodySharedSettings_spec, "SoftBodySharedSettings", docs_str);
        md_stitch_spec(&Registry_spec, "Registry", docs_str);
        md_stitch_spec(&MathService_spec, "MathService", docs_str);
        md_stitch_spec(&Ship_spec, "Ship", docs_str);

        // Gated Handler Registration
        JPH_SetTraceHandler(culv_jph_trace);
        JPH_SetAssertFailureHandler(culv_jph_assert);

        if (!JPH_Init()) {
            PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
            atomic_store_explicit(&docs_status, 0, memory_order_seq_cst);
            return -1;
        }

        JPH_BroadPhaseLayerFilter_SetProcs(&global_bp_procs);
        JPH_ObjectLayerFilter_SetProcs(&global_obj_procs);
        JPH_BodyFilter_SetProcs(&global_bf_procs);
        JPH_ShapeFilter_SetProcs(&global_sf_procs);
        JPH_DebugRenderer_SetProcs(&debug_procs);
        JPH_ContactListener_SetProcs(&contact_procs);
        JPH_CharacterContactListener_SetProcs(&char_listener_procs);

        if (INIT_NATIVE_MUTEX(g_jph_init_lock) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to initialize global initialization lock");
            return -1;
        }

        atomic_store_explicit(&docs_status, 2, memory_order_seq_cst);

    } else {
        // --- THE LOSERS: Wait for the Winner to finish ---
        while (atomic_load_explicit(&docs_status, memory_order_acquire) != 2) {
            culverin_yield();
        }
    }
    // --- 2. PER-INTERPRETER SETUP (Runs for every import) ---
    if (PyModule_AddStringConstant(m, "__version__", shared_version) < 0) {
        return -1;
    }
    CULV_INIT_PROFILER();

    st->helper = PyImport_ImportModule("culverin._culverin");
    if (!st->helper) {
        return -1;
    }
    culverin_init_module_parsers(&st->parsers);

    if (init_types(m, st) < 0) {
        return -1;
    }
    if (init_constants(m) < 0) {
        return -1;
    }
    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_Status culverin_traverse(PyObject *m, visitproc visit, void *arg) {
    CulverinState *st = get_culverin_state(m);
    Py_VISIT(st->helper);
    Py_VISIT(st->PhysicsWorldType);
    Py_VISIT(st->CharacterType);
    Py_VISIT(st->VehicleType);
    Py_VISIT(st->RagdollSettingsType);
    Py_VISIT(st->RagdollType);
    Py_VISIT(st->SkeletonType);
    Py_VISIT(st->ShipType);
    Py_VISIT(st->BufferProxyType);
    Py_VISIT(st->MathServiceType);
    Py_VISIT(st->SoftBodySharedSettingsType);
    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_Status culverin_clear(PyObject *m) {
    CulverinState *st = get_culverin_state(m);
    Py_CLEAR(st->helper);
    Py_CLEAR(st->PhysicsWorldType);
    Py_CLEAR(st->CharacterType);
    Py_CLEAR(st->VehicleType);
    Py_CLEAR(st->RagdollSettingsType);
    Py_CLEAR(st->RagdollType);
    Py_CLEAR(st->SkeletonType);
    Py_CLEAR(st->ShipType);
    Py_CLEAR(st->BufferProxyType);
    Py_CLEAR(st->MathServiceType);
    Py_CLEAR(st->SoftBodySharedSettingsType);
    culverin_free_module_parsers(&st->parsers);
    return 0;
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
[[gnu::used]]
PyModuleDef culverin_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_culverin_c",
    .m_doc  = "Culverin Physics Engine Core",
    .m_size = sizeof(CulverinState),
    .m_methods =
        (PyMethodDef[]){

            MOD_FASTCALL_INTERNAL(dump_schema_json), MOD_FASTCALL(mutate_tuple), {}

        },
    .m_slots =
        (PyModuleDef_Slot[]){

            {.slot = Py_mod_exec, .value = culverin_exec},

// 1. Handle the Free-threaded (No GIL) declaration (3.13+)
#if defined(Py_MOD_GIL_NOT_USED)
            {.slot = Py_mod_gil, .value = Py_MOD_GIL_NOT_USED},
#endif

            // 2. Handle Subinterpreter support
            {.slot = Py_mod_multiple_interpreters,
#if PY_VERSION_HEX >= 0x030D0000
             .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED
#else
             .value = Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
#endif
            },

            {}

        },
    .m_traverse = culverin_traverse,
    .m_clear    = culverin_clear,
};
[[gnu::visibility("default")]]
extern PyMODINIT_FUNC PyInit__culverin_c(void) {
    return PyModuleDef_Init(&culverin_module);
}