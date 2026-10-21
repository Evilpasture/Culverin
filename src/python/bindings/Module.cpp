// clang-format off
#if !defined(_CRT_SECURE_NO_WARNINGS)
#    define _CRT_SECURE_NO_WARNINGS
#endif
// clang-format on
#include "bindings/Getters.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/docstrings/Docstrings.hpp"
#include "characters/Character.hpp"
#include "core/Compiler.hpp"
#include "core/Profiling.hpp"
#include "core/Threading.hpp"
#include "debug/DebugRender.hpp"
#include "ecs/Ecs.hpp"
#include "jolt/JoltLog.hpp"
#include "math/Math.hpp"
#include "physics/ContactListener.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "ragdolls/Ragdoll.hpp"
#include "soft_bodies/SoftBody.hpp"
#include "vehicles/ShipBinding.hpp"
#include "vehicles/Vehicle.hpp"
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/Memory.h>
#include <Jolt/Physics/Body/AllowedDOFs.h>
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include <Jolt/RegisterTypes.h>
#include <atomic>
#include <new>
#include <tupleobject.h>

/**
 * INTERNAL HELPER: DumpParserCatalog
 * Iterates through one parser group's self-populated catalog and writes its JSON entries.
 */

namespace CPH::Python {
static void DumpParserCatalog(FastParserCore *const *const parsers, const size_t count,
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
[[nodiscard]] static PyObject * ModuleDumpSchemaJson(PyObject *const self, PyObject *const *args,
                                                      size_t nargsf, PyObject *kwnames) {
    const auto st        = GetModuleState(self);
    const char *filename = "culverin_schema.json";

    // 1. Thread-safe, subinterpreter-safe parsing via FastParser

    const ParseResult parsed =
        st->parsers->DumpSchema.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, filename);
    if (!parsed) {
        RaiseParseError(parsed.error());
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

    const auto dump_group = [&](const auto &group) {
        DumpParserCatalog(group.catalog.Data(), group.catalog.Size(), f, &is_first);
    };
    dump_group(all.world);
    dump_group(all.character);
    dump_group(all.vehicle);
    dump_group(all.ecs);
    dump_group(all.skeleton);
    dump_group(all.ragdoll);
    dump_group(all.ragdoll_settings);
    dump_group(all.ship);
    dump_group(all.sbss);
    dump_group(all.math);
    dump_group(all.module);

    fprintf(f, "\n}\n");
    fclose(f);

    // 4. Cleanup temporary allocations

    Py_RETURN_NONE;
}


[[nodiscard]] static PyObject * ModuleMutateTuple([[maybe_unused]] PyObject *self,
                                                   PyObject *const *args, Py_ssize_t nargsf) {
    constexpr Py_ssize_t kArg0 = 0;
    constexpr Py_ssize_t kArg1 = 1;
    constexpr Py_ssize_t kArg2 = 2;
    constexpr Py_ssize_t kArg3 = 3;
    constexpr Py_ssize_t kArg4 = 4;
    // --- 1. DECLARATIONS (C / goto safety) ---
    auto nargs              = PyVectorcall_NARGS(nargsf);
    constexpr auto kMinArgs = 3;
    constexpr auto kMaxArgs = 5;

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
    if (nargs != kMinArgs && nargs != kMaxArgs) {
        PyErr_Format(PyExc_TypeError, "mutate_tuple() takes 3 or 5 arguments (%zd given)", nargs);
        return nullptr;
    }

    target  = args[kArg0];
    new_val = args[kArg2];
    if (!PyTuple_Check(target)) {
        PyErr_SetString(PyExc_TypeError, "arg 0 must be a tuple");
        return nullptr;
    }

    index = PyLong_AsSsize_t(args[kArg1]);
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

    if (nargs == kMaxArgs) {
        registry = args[kArg3];
        key      = args[kArg4];
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
        auto *atomic_slot = reinterpret_cast<std::atomic<PyObject *> *>(&(t->ob_item[index]));
        old_val           = atomic_slot->exchange(new_val, std::memory_order::acq_rel);

        // Step C: Rehash Cache Bust (Atomic)
#if PY_VERSION_HEX >= 0x030E0000
        auto *atomic_hash = reinterpret_cast<std::atomic<Py_hash_t> *>(&t->ob_hash);
        atomic_hash->store(-1, std::memory_order::relaxed);
#endif
    }

    final_hash = PyObject_Hash(target);

    if (final_hash == -1) {
        // Rollback (Atomic)
        auto *atomic_slot = reinterpret_cast<std::atomic<PyObject *> *>(
            &(reinterpret_cast<PyTupleObject *>(target))->ob_item[index]);
        atomic_slot->store(old_val, std::memory_order::relaxed);
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
            auto *atomic_slot = reinterpret_cast<std::atomic<PyObject *> *>(
                &(reinterpret_cast<PyTupleObject *>(target))->ob_item[index]);
            atomic_slot->store(old_val, std::memory_order::relaxed);
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
static std::atomic<int> docs_status = 0;

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
static const char *ExtractVersionFromToml(void) {
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
    static constexpr size_t kVersionBuffer = 32;
    static char version_buffer[kVersionBuffer];
    size_t len = end - start;
    if (len >= sizeof(version_buffer)) {
        len = sizeof(version_buffer) - 1;
    }

    strncpy(version_buffer, start, len);
    version_buffer[len] = '\0';

    return version_buffer;
}

static int InitTypes(PyObject *m, ModuleState *st) {
    struct {
        PyType_Spec *spec;
        PyObject **slot;
        const char *name;
    } types[] = {
        {.spec = (&WorldSpec), .slot = &st->PhysicsWorldType, .name = "PhysicsWorld"},
        {.spec = (&CharacterSpec), .slot = &st->CharacterType, .name = "Character"},
        {.spec = (&VehicleSpec), .slot = &st->VehicleType, .name = "Vehicle"},
        {.spec = (&ShipSpec), .slot = &st->ShipType, .name = "Ship"},
        {.spec = (&RagdollSettingsSpec),
         .slot = &st->RagdollSettingsType,
         .name = "RagdollSettings"},
        {.spec = (&RagdollSpec), .slot = &st->RagdollType, .name = "Ragdoll"},
        {.spec = (&SkeletonSpec), .slot = &st->SkeletonType, .name = "Skeleton"},
        {.spec = (&BufferProxySpec), .slot = &st->BufferProxyType, .name = "BufferProxyObject"},
        {.spec = (&SoftBodySharedSettingsSpec),
         .slot = &st->SoftBodySharedSettingsType,
         .name = "SoftBodySharedSettings"},
        {.spec = (&RegistrySpec), .slot = &st->RegistryType, .name = "Registry"},
        {.spec = (&MathServiceSpec), .slot = &st->MathServiceType, .name = "MathService"}};

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

static int InitConstants(PyObject *m) {
    static const struct {
        const char *name;
        int value;
    } consts[] = {{.name = "SHAPE_BOX", .value = std::to_underlying(ShapeType::Box)},
                  {.name = "SHAPE_SPHERE", .value = std::to_underlying(ShapeType::Sphere)},
                  {.name = "SHAPE_CAPSULE", .value = std::to_underlying(ShapeType::Capsule)},
                  {.name = "SHAPE_CYLINDER", .value = std::to_underlying(ShapeType::Cylinder)},
                  {.name = "SHAPE_PLANE", .value = std::to_underlying(ShapeType::Plane)},
                  {.name = "SHAPE_MESH", .value = std::to_underlying(ShapeType::Mesh)},
                  {.name = "SHAPE_HEIGHTFIELD", .value = std::to_underlying(ShapeType::Heightfield)},
                  {.name = "SHAPE_CONVEX_HULL", .value = std::to_underlying(ShapeType::ConvexHull)},
                  {.name = "MOTION_STATIC", .value = std::to_underlying(MotionType::Static)},
                  {.name = "MOTION_KINEMATIC", .value = std::to_underlying(MotionType::Kinematic)},
                  {.name = "MOTION_DYNAMIC", .value = std::to_underlying(MotionType::Dynamic)},
                  {.name = "DOF_TRANSLATION_X", .value = static_cast<int>(JPH::EAllowedDOFs::TranslationX)},
                  {.name = "DOF_TRANSLATION_Y", .value = static_cast<int>(JPH::EAllowedDOFs::TranslationY)},
                  {.name = "DOF_TRANSLATION_Z", .value = static_cast<int>(JPH::EAllowedDOFs::TranslationZ)},
                  {.name = "DOF_ROTATION_X", .value = static_cast<int>(JPH::EAllowedDOFs::RotationX)},
                  {.name = "DOF_ROTATION_Y", .value = static_cast<int>(JPH::EAllowedDOFs::RotationY)},
                  {.name = "DOF_ROTATION_Z", .value = static_cast<int>(JPH::EAllowedDOFs::RotationZ)},
                  {.name = "DOF_ALL", .value = static_cast<int>(JPH::EAllowedDOFs::All)},
                  {.name  = "DOF_TRANSLATE_XY",
                   .value = static_cast<int>(JPH::EAllowedDOFs::TranslationX |
                                                JPH::EAllowedDOFs::TranslationY)},
                  {.name  = "DOF_PLANE_XY",
                   .value = static_cast<int>(JPH::EAllowedDOFs::TranslationX |
                                                JPH::EAllowedDOFs::TranslationY)},
                  {.name = "DOF_PLANE_2D", .value = static_cast<int>(JPH::EAllowedDOFs::Plane2D)},
                  {.name = "CONSTRAINT_FIXED", .value = std::to_underlying(ConstraintType::Fixed)},
                  {.name = "CONSTRAINT_POINT", .value = std::to_underlying(ConstraintType::Point)},
                  {.name = "CONSTRAINT_HINGE", .value = std::to_underlying(ConstraintType::Hinge)},
                  {.name = "CONSTRAINT_SLIDER", .value = std::to_underlying(ConstraintType::Slider)},
                  {.name = "CONSTRAINT_DISTANCE", .value = std::to_underlying(ConstraintType::Distance)},
                  {.name = "CONSTRAINT_CONE", .value = std::to_underlying(ConstraintType::Cone)},
                  {.name = "EVENT_ADDED", .value = std::to_underlying(ContactEventType::Added)},
                  {.name = "EVENT_PERSISTED", .value = std::to_underlying(ContactEventType::Persisted)},
                  {.name = "EVENT_REMOVED", .value = std::to_underlying(ContactEventType::Removed)},
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
                  {.name = "BEND_DIHEDRAL", .value = static_cast<int>(JPH::SoftBodySharedSettings::EBendType::Dihedral)},
                  {.name = "BEND_DISTANCE", .value = static_cast<int>(JPH::SoftBodySharedSettings::EBendType::Distance)},
                  {.name = "BEND_NONE", .value = static_cast<int>(JPH::SoftBodySharedSettings::EBendType::None)}};

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

[[nodiscard]] static int culverin_exec(PyObject *m) {
    auto *storage = GetModuleStateStorage(m);
    if (storage == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "CPython module state is unavailable");
        return -1;
    }
    if (storage->state == nullptr) {
        storage->state = new (std::nothrow) ModuleState{};
        if (storage->state == nullptr) {
            PyErr_NoMemory();
            return -1;
        }
    }
    ModuleState *st = storage->state;
    // 1. THE MASTER GATE: Protects all static global memory in the process
    int expected = 0;
    if (atomic_compare_exchange_strong(&docs_status, &expected, 1)) {
        // --- 1A. VERSION & BUILD METADATA ---
        const char *ver_temp = ExtractVersionFromToml();

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
        DocsStitchMethods(culverin_module.m_methods, "Module", docs_str);
        DocsStitchSpec(&WorldSpec, "PhysicsWorld", docs_str);
        DocsStitchSpec(&CharacterSpec, "Character", docs_str);
        DocsStitchSpec(&VehicleSpec, "Vehicle", docs_str);
        DocsStitchSpec(&SkeletonSpec, "Skeleton", docs_str);
        DocsStitchSpec(&RagdollSpec, "Ragdoll", docs_str);
        DocsStitchSpec(&RagdollSettingsSpec, "RagdollSettings", docs_str);
        DocsStitchSpec(&SoftBodySharedSettingsSpec, "SoftBodySharedSettings", docs_str);
        DocsStitchSpec(&RegistrySpec, "Registry", docs_str);
        DocsStitchSpec(&MathServiceSpec, "MathService", docs_str);
        DocsStitchSpec(&ShipSpec, "Ship", docs_str);

        // Was JPH_Init(). joltc implemented that as exactly these three calls
        // plus a process-global TempAllocatorImplWithMallocFallback that
        // Culverin never used: every PhysicsWorld owns its own temp allocator
        // (PhysicsWorld::InitJolt). Jolt's own integration guide prescribes this
        // order -- allocator, factory, type registration -- so the sequence is
        // inlined here rather than shimmed. The guard reproduces JPH_Init's
        // idempotence: module init re-runs under sub-interpreters, and calling
        // RegisterTypes twice would double-register into the factory.
        static bool s_jolt_types_registered = false;
        if (!s_jolt_types_registered) {
            JPH::RegisterDefaultAllocator();
            JPH::Factory::sInstance = new JPH::Factory();
            JPH::RegisterTypes();
            s_jolt_types_registered = true;
        }

        // Trace/assert hooks are Jolt function pointers now (A3). A5 deleted
        // JPH_Init(), whose only remaining relevance was that it used to assign
        // both hooks itself and so silently undid an earlier install; the
        // inlined sequence above does not touch them. See jolt/JoltLog.hpp. The contact listener and debug renderer are native
        // subclasses owned by each world, and the query filters are native
        // Jolt::BodyFilter &c. constructed per query, and each character owns a
        // native CharacterContactListener subclass, so no proc tables are
        // registered at all.
        CPH::InstallJoltLogging();

        (docs_status).store(2, std::memory_order::seq_cst);

    } else {
        // --- THE LOSERS: Wait for the Winner to finish ---
        while ((docs_status).load(std::memory_order::acquire) != 2) {
            CpuYield();
        }
    }
    // --- 2. PER-INTERPRETER SETUP (Runs for every import) ---
    if (PyModule_AddStringConstant(m, "__version__", shared_version) < 0) {
        return -1;
    }

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

    if (InitTypes(m, st) < 0) {
        return -1;
    }
    if (InitConstants(m) < 0) {
        return -1;
    }
    return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] static int culverin_traverse(PyObject *m, visitproc visit, void *arg) {
    ModuleState *st = GetModuleState(m);
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
[[nodiscard]] static int culverin_clear(PyObject *m) {
    ModuleState *st = GetModuleState(m);
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
    auto *storage   = GetModuleStateStorage(py_module);
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

    CPH::Python::FastCallMethod("_dump_schema_json", ModuleDumpSchemaJson),
    CPH::Python::FastCallMethod("mutate_tuple", ModuleMutateTuple),
    {nullptr, nullptr, 0, nullptr}};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
[[gnu::used]]
static PyModuleDef_Slot ModuleSlots[] = {
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
    .m_size    = sizeof(ModuleStateStorage),
    .m_methods = kModuleMethods,
    .m_slots   = ModuleSlots,
    .m_traverse = culverin_traverse,
    .m_clear    = culverin_clear,
    .m_free     = culverin_free,
};

[[nodiscard]] PyObject *InitializeModule() noexcept { return PyModuleDef_Init(&culverin_module); }

} // namespace CPH::Python

PyMODINIT_FUNC PyInit__culverin_c(void) { return CPH::Python::InitializeModule(); }
