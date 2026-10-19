// clang-format off
#if !defined(_CRT_SECURE_NO_WARNINGS)
#    define _CRT_SECURE_NO_WARNINGS
#endif
// clang-format on
#include "bindings/Getters.h"
#include "bindings/ModuleState.h"
#include "bindings/PythonSupport.h"
#include "bindings/docstrings/Docstrings.h"
#include "characters/Character.h"
#include "core/CompilerSpecifics.h"
#include "core/Profiling.h"
#include "core/Threading.h"
#include "debug/DebugRender.h"
#include "ecs/Ecs.h"
#include "jolt/JoltHandler.h"
#include "joltc.h"
#include "math/Math.h"
#include "physics/ContactListener.h"
#include "physics/Filters.h"
#include "physics/PhysicsWorldBinding.h"
#include "ragdolls/Ragdoll.h"
#include "soft_bodies/SoftBody.h"
#include "vehicles/Ship.h"
#include "vehicles/Vehicle.h"
#include <new>
#include <stdatomic.h>

/**
 * INTERNAL HELPER: dump_parser_catalog
 * Iterates through one parser group's self-populated catalog and writes its JSON entries.
 */
static void dump_parser_catalog(CPH_FastParser *const *const parsers, const size_t count,
                                FILE *const out, bool *const is_first) {
    for (size_t i = 0; i < count; ++i) {
        const auto fp = parsers[i];
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
 * Constructs each self-describing parser group, dumps their schemas to one JSON, and cleans up.
 */
PyCFunction_DeclareMethod CPH_PyModule_DumpSchemaJson(PyObject *const self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames) {
    const auto st        = CPH_GetModuleState(self);
    const char *filename = "culverin_schema.json";

    // 1. Thread-safe, subinterpreter-safe parsing via CulvFastParser

    if (!st->parsers->DumpSchema.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, filename)) {
        return nullptr;
    }

    FILE *const f = fopen(filename, "w");
    if (f == nullptr) {
        return PyErr_SetFromErrno(PyExc_IOError);
    }

    // 2. Construct temporary parser groups. Each parser initializes itself and
    // joins its owner's catalog during construction.
    struct {
        CPH::Python::WorldParsers world;
        CPH::Python::CharacterParsers character;
        CPH::Python::VehicleParsers vehicle;
        CPH::Python::EcsParsers ecs;
        CPH::Python::SkeletonParsers skeleton;
        CPH::Python::RagdollParsers ragdoll;
        CPH::Python::RagdollSettingsParsers ragdoll_settings;
        CPH::Python::ShipParsers ship;
        CPH::Python::SoftBodySharedSettingsParsers sbss;
        CPH::Python::MathParsers math;
        CPH::Python::ModuleParsers module;
    } all{};

    // 3. Export to JSON
    fprintf(f, "{\n");
    bool is_first = true;

#define CPH_DUMP_GROUP(member)                                                                     \
    dump_parser_catalog(all.member.catalog.Data(), all.member.catalog.Size(), f, &is_first)

    CPH_DUMP_GROUP(world);
    CPH_DUMP_GROUP(character);
    CPH_DUMP_GROUP(vehicle);
    CPH_DUMP_GROUP(ecs);
    CPH_DUMP_GROUP(skeleton);
    CPH_DUMP_GROUP(ragdoll);
    CPH_DUMP_GROUP(ragdoll_settings);
    CPH_DUMP_GROUP(ship);
    CPH_DUMP_GROUP(sbss);
    CPH_DUMP_GROUP(math);
    CPH_DUMP_GROUP(module);

#undef CPH_DUMP_GROUP

    fprintf(f, "\n}\n");
    fclose(f);

    // 4. Cleanup temporary allocations

    Py_RETURN_NONE;
}

#include <stdatomic.h>
#include <tupleobject.h>

PyCFunction_DeclareMethod CPH_PyModule_MutateTuple([[maybe_unused]] PyObject *self,
                                                   PyObject *const *args, Py_ssize_t nargsf) {
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
        PyTupleObject *t = reinterpret_cast<PyTupleObject *>(target);
        Py_INCREF(new_val);

        // Cast the item slot to an atomic pointer.
        // This prevents TSan "Data Race" warnings and ensures memory visibility.
        _Atomic(PyObject *) *atomic_slot =
            reinterpret_cast<_Atomic(PyObject *) *>(&(t->ob_item[index]));
        old_val = atomic_exchange_explicit(atomic_slot, new_val, memory_order_acq_rel);

        // Step C: Rehash Cache Bust (Atomic)
#if PY_VERSION_HEX >= 0x030E0000
        _Atomic(Py_hash_t) *atomic_hash = reinterpret_cast<_Atomic(Py_hash_t) *>(&t->ob_hash);
        atomic_store_explicit(atomic_hash, -1, memory_order_relaxed);
#endif
    }

    final_hash = PyObject_Hash(target);

    if (final_hash == -1) {
        // Rollback (Atomic)
        _Atomic(PyObject *) *atomic_slot = reinterpret_cast<_Atomic(PyObject *) *>(
            &(reinterpret_cast<PyTupleObject *>(target))->ob_item[index]);
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
            _Atomic(PyObject *) *atomic_slot = reinterpret_cast<_Atomic(PyObject *) *>(
                &(reinterpret_cast<PyTupleObject *>(target))->ob_item[index]);
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

#ifndef CPH_DOCS_PATH
#    define CPH_DOCS_PATH "../../docs/DOCS.md"
#endif

static const unsigned char kAllDocs[] = {
#ifdef __has_embed
#    if __has_embed(CPH_DOCS_PATH)
#        embed CPH_DOCS_PATH suffix(, 0)
#        define CPH_DOCS_EMBEDDED
#    endif
#endif

#ifndef CPH_DOCS_EMBEDDED
#    include "ALL_DOCS.inc"
#endif
};

// Global flag to ensure we only stitch once (important for subinterpreters)
static atomic_int docs_status = 0;

// =============================================================================================

// --- Module Initialization ---

// Embed the entire TOML file as a static string
static const unsigned char kPyprojectToml[] = {
#ifdef __has_embed
#    if __has_embed("../../pyproject.toml")
#        embed "../../pyproject.toml" suffix(, 0)
#        define CPH_TOML_EMBEDDED
#    endif
#endif

#ifndef CPH_TOML_EMBEDDED
#    include "PYPROJECT_TOML.inc"
#endif
};

// Helper function to extract the version string at runtime
static const char *extract_version_from_toml(void) {
    // Look for the specific pattern 'version = "'
    const char *key = "version = \"";
    auto start      = strstr(reinterpret_cast<const char *>(kPyprojectToml), key);

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

static int init_types(PyObject *m, CPH_ModuleState *st) {
    struct {
        PyType_Spec *spec;
        PyObject **slot;
        const char *name;
    } types[] = {
        {.spec = (&CPH_PyWorld_Spec), .slot = &st->PhysicsWorldType, .name = "PhysicsWorld"},
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
                  {.name = "DOF_TRANSLATION_X", .value = JPH_AllowedDOFs_TranslationX},
                  {.name = "DOF_TRANSLATION_Y", .value = JPH_AllowedDOFs_TranslationY},
                  {.name = "DOF_TRANSLATION_Z", .value = JPH_AllowedDOFs_TranslationZ},
                  {.name = "DOF_ROTATION_X", .value = JPH_AllowedDOFs_RotationX},
                  {.name = "DOF_ROTATION_Y", .value = JPH_AllowedDOFs_RotationY},
                  {.name = "DOF_ROTATION_Z", .value = JPH_AllowedDOFs_RotationZ},
                  {.name = "DOF_ALL", .value = JPH_AllowedDOFs_All},
                  {.name  = "DOF_TRANSLATE_XY",
                   .value = JPH_AllowedDOFs_TranslationX | JPH_AllowedDOFs_TranslationY},
                  {.name  = "DOF_PLANE_XY",
                   .value = JPH_AllowedDOFs_TranslationX | JPH_AllowedDOFs_TranslationY},
                  {.name = "DOF_PLANE_2D", .value = JPH_AllowedDOFs_Plane2D},
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

static constexpr auto kMagicBuffer = 128;
static char shared_version[kMagicBuffer];

extern PyModuleDef culverin_module;

PyType_DeclareSlot_Status culverin_exec(PyObject *m) {
    auto *storage = CPH_GetModuleStateStorage(m);
    if (storage == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "CPython module state is unavailable");
        return -1;
    }
    if (storage->state == nullptr) {
        storage->state = new (std::nothrow) CPH_ModuleState{};
        if (storage->state == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
    }
    CPH_ModuleState *st = storage->state;
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
        snprintf(shared_version, kMagicBuffer, "%s (%s, %s, %s, %s)", ver_temp, gil_status,
                 precision, build_type, compiler_id);

        // --- THE WINNER: Run exactly once per process life ---
        const char *docs_str = reinterpret_cast<const char *>(kAllDocs);
        CPH_DocsStitchMethods(culverin_module.m_methods, "Module", docs_str);
        CPH_DocsStitchSpec(&CPH_PyWorld_Spec, "PhysicsWorld", docs_str);
        CPH_DocsStitchSpec(&Character_spec, "Character", docs_str);
        CPH_DocsStitchSpec(&Vehicle_spec, "Vehicle", docs_str);
        CPH_DocsStitchSpec(&Skeleton_spec, "Skeleton", docs_str);
        CPH_DocsStitchSpec(&Ragdoll_spec, "Ragdoll", docs_str);
        CPH_DocsStitchSpec(&RagdollSettings_spec, "RagdollSettings", docs_str);
        CPH_DocsStitchSpec(&SoftBodySharedSettings_spec, "SoftBodySharedSettings", docs_str);
        CPH_DocsStitchSpec(&Registry_spec, "Registry", docs_str);
        CPH_DocsStitchSpec(&MathService_spec, "MathService", docs_str);
        CPH_DocsStitchSpec(&Ship_spec, "Ship", docs_str);

        // Gated Handler Registration
        JPH_SetTraceHandler(CPH_JphTrace);
        JPH_SetAssertFailureHandler(CPH_JphAssert);

        if (!JPH_Init()) {
            PyErr_SetString(PyExc_RuntimeError, "Jolt initialization failed");
            atomic_store_explicit(&docs_status, 0, memory_order_seq_cst);
            return -1;
        }

        JPH_BroadPhaseLayerFilter_SetProcs(&kGlobalBpProcs);
        JPH_ObjectLayerFilter_SetProcs(&kGlobalObjProcs);
        JPH_BodyFilter_SetProcs(&kGlobalBfProcs);
        JPH_ShapeFilter_SetProcs(&kGlobalSfProcs);
        JPH_DebugRenderer_SetProcs(&kDebugProcs);
        JPH_ContactListener_SetProcs(&contact_procs);
        JPH_CharacterContactListener_SetProcs(&kCharListenerProcs);

        CPH_INIT_NATIVE_MUTEX(g_jph_init_lock);

        atomic_store_explicit(&docs_status, 2, memory_order_seq_cst);

    } else {
        // --- THE LOSERS: Wait for the Winner to finish ---
        while (atomic_load_explicit(&docs_status, memory_order_acquire) != 2) {
            CPH_CpuYield();
        }
    }
    // --- 2. PER-INTERPRETER SETUP (Runs for every import) ---
    if (PyModule_AddStringConstant(m, "__version__", shared_version) < 0) {
        return -1;
    }
    CPH_INIT_PROFILER();

    st->helper = PyImport_ImportModule("culverin._culverin");
    if (!st->helper) {
        return -1;
    }
    if (st->parsers == nullptr) {
        st->parsers.reset(CPH::Python::NewParserGroup<CPH::Python::ModuleParsers>());
        if (st->parsers == nullptr) {
            return -1;
        }
    }

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
    CPH_ModuleState *st = CPH_GetModuleState(m);
    if (st == nullptr) {
        return 0;
    }
    Py_VISIT(st->helper);
    Py_VISIT(st->PhysicsWorldType);
    Py_VISIT(st->CharacterType);
    Py_VISIT(st->VehicleType);
    Py_VISIT(st->RagdollSettingsType);
    Py_VISIT(st->RagdollType);
    Py_VISIT(st->SkeletonType);
    Py_VISIT(st->ShipType);
    Py_VISIT(st->BufferProxyType);
    Py_VISIT(st->RegistryType);
    Py_VISIT(st->MathServiceType);
    Py_VISIT(st->SoftBodySharedSettingsType);
    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
PyType_DeclareSlot_Status culverin_clear(PyObject *m) {
    CPH_ModuleState *st = CPH_GetModuleState(m);
    if (st == nullptr) {
        return 0;
    }
    Py_CLEAR(st->helper);
    Py_CLEAR(st->PhysicsWorldType);
    Py_CLEAR(st->CharacterType);
    Py_CLEAR(st->VehicleType);
    Py_CLEAR(st->RagdollSettingsType);
    Py_CLEAR(st->RagdollType);
    Py_CLEAR(st->SkeletonType);
    Py_CLEAR(st->ShipType);
    Py_CLEAR(st->BufferProxyType);
    Py_CLEAR(st->RegistryType);
    Py_CLEAR(st->MathServiceType);
    Py_CLEAR(st->SoftBodySharedSettingsType);
    st->parsers.reset();
    return 0;
}

static void culverin_free(void *module) {
    auto *py_module = static_cast<PyObject *>(module);
    auto *storage   = CPH_GetModuleStateStorage(py_module);
    if (storage == nullptr || storage->state == nullptr) {
        return;
    }

    // m_clear can run before m_free, so keep this idempotent and clear any
    // remaining Python references before destroying the C++ state object.
    [[maybe_unused]] const auto clear_status = culverin_clear(py_module);
    delete storage->state;
    storage->state = nullptr;
}

static PyMethodDef kModuleMethods[] = {

    CPH::Python::FastCallMethod("_dump_schema_json", CPH_PyModule_DumpSchemaJson),
    CPH::Python::FastCallMethod("mutate_tuple", CPH_PyModule_MutateTuple),
    {nullptr, nullptr, 0, nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
[[gnu::used]]
static PyModuleDef_Slot CPH_PyModule_Slots[] = {
    {.slot = Py_mod_exec, .value = reinterpret_cast<void *>(culverin_exec)},

#if defined(Py_MOD_GIL_NOT_USED)
    {.slot = Py_mod_gil, .value = Py_MOD_GIL_NOT_USED},
#endif

    {.slot = Py_mod_multiple_interpreters,
#if PY_VERSION_HEX >= 0x030D0000
     .value = Py_MOD_MULTIPLE_INTERPRETERS_SUPPORTED
#else
     .value = Py_MOD_PER_INTERPRETER_GIL_SUPPORTED
#endif
    },
    {},
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
[[gnu::used]]
PyModuleDef culverin_module = {
    .m_base    = PyModuleDef_HEAD_INIT,
    .m_name    = "_culverin_c",
    .m_doc     = "Culverin Physics Engine Core",
    .m_size    = sizeof(CPH_ModuleStateStorage),
    .m_methods = kModuleMethods,
    .m_slots   = CPH_PyModule_Slots,
    .m_traverse = culverin_traverse,
    .m_clear    = culverin_clear,
    .m_free     = culverin_free,
};
PyMODINIT_FUNC PyInit__culverin_c(void) { return PyModuleDef_Init(&culverin_module); }
