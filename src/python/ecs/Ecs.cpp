// Python adapter for CPH::Registry.
//
// Everything here is one of three things: parse arguments into natives, call a
// method on the registry, or wrap a result into a Python object. The ECS data
// structures themselves live in src/ecs/ and know nothing about Python; this
// file's whole job is the language boundary.

#include "ecs/Ecs.hpp"
#include "bindings/Getters.hpp"
#include "bindings/BufferView.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/PythonSupport.hpp"
#include "physics/EcsSync.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include <span>
#include <string>
#include <string_view>

namespace CPH::Python {

namespace {

// Maps a native ECS failure onto the Python exception the API has always
// raised for it, message included.
PyObject *RaiseEcsError(const CPH::Error &error) {
    const CPH::ErrorCode code = error.Code();

    if (code.Is(CPH::EcsError::OutOfMemory)) {
        return PyErr_NoMemory();
    }

    PyObject *type = PyExc_RuntimeError;
    if (code.Is(CPH::EcsError::ComponentResizeWhileViewed) ||
        code.Is(CPH::EcsError::RegisterWhileViewed) ||
        code.Is(CPH::EcsError::ClearWhileViewed)) {
        type = PyExc_BufferError;
    } else if (code.Is(CPH::EcsError::InvalidEntity) ||
               code.Is(CPH::EcsError::InvalidComponent) ||
               code.Is(CPH::EcsError::InvalidHandleComponent) ||
               code.Is(CPH::EcsError::ComponentSizeMismatch)) {
        type = PyExc_ValueError;
    } else if (code.Is(CPH::EcsError::HandleComponentSize) ||
               code.Is(CPH::EcsError::PositionComponentSize) ||
               code.Is(CPH::EcsError::RotationComponentSize)) {
        type = PyExc_TypeError;
    }

    const std::string_view message = error.Message();
    PyErr_SetString(type, std::string(message).c_str());
    return nullptr;
}

PyObject *MakeEcsProxy(RegistryObject *self, std::uint32_t comp_id, bool entities) {
    const std::expected<CPH::Registry::ComponentView, CPH::Error> view =
        self->registry->View(comp_id);
    if (!view) {
        return RaiseEcsError(view.error());
    }

    // The registry pins the dense arrays while an export is outstanding, so the
    // addresses below cannot move before the proxy is released.
    return NewBufferProxy(
        reinterpret_cast<PyObject *>(self),
        RegistryBuffer(*self->registry,
                       entities ? RegistryChannel::EntityHandles : RegistryChannel::ComponentData,
                       entities ? static_cast<void *>(view->entities)
                                : static_cast<void *>(view->data),
                       view->count, view->elementSize));
}

} // namespace

// --- LIFECYCLE ---

[[nodiscard]] int RegistryInit(RegistryObject *self, [[maybe_unused]] PyObject *args,
                               [[maybe_unused]] PyObject *kwds) {
    self->registry = std::make_unique<CPH::Registry>();
    self->parsers  = CPH::Python::NewParserGroup<CPH::Python::EcsParsers>();
    if (!self->parsers) {
        return -1;
    }
    return 0;
}

void RegistryDealloc(RegistryObject *self) {
    delete self->parsers;
    self->parsers = nullptr;
    self->registry.reset();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// --- ENTITY MANAGEMENT ---

[[nodiscard]] PyObject *RegistryCreate(RegistryObject *self, [[maybe_unused]] PyObject *args) {
    const CPH::Entity entity = self->registry->Create();
    if (!entity.IsValid()) {
        return PyErr_NoMemory();
    }
    return PyLong_FromUnsignedLongLong(entity.Raw());
}

[[nodiscard]] PyObject *RegistryDestroy(RegistryObject *self, PyObject *const *args,
                                        size_t nargsf, PyObject *kwnames) {
    uint64_t handle_raw;

    const ParseResult parsed =
        self->parsers->regEntityOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    self->registry->Destroy(CPH::Entity(handle_raw));
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *RegistryIsAlive(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                        PyObject *kwnames) {
    uint64_t handle_raw;

    const ParseResult parsed =
        self->parsers->regEntityOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->registry->IsAlive(CPH::Entity(handle_raw))) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

[[nodiscard]] PyObject *RegistryClear(RegistryObject *self, [[maybe_unused]] PyObject *args) {
    const std::expected<void, CPH::Error> cleared = self->registry->Clear();
    if (!cleared) {
        return RaiseEcsError(cleared.error());
    }
    Py_RETURN_NONE;
}

// --- COMPONENT MANAGEMENT ---

[[nodiscard]] PyObject *RegistryRegisterComponent(RegistryObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames) {
    uint32_t size;

    const ParseResult parsed =
        self->parsers->regRegComp.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, size);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const std::expected<CPH::Registry::ComponentId, CPH::Error> component =
        self->registry->RegisterComponent(size);
    if (!component) {
        return RaiseEcsError(component.error());
    }
    return PyLong_FromUnsignedLong(*component);
}

[[nodiscard]] PyObject *RegistryAdd(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames) {
    uint64_t handle_raw;
    uint32_t comp_id;
    PyObject *data_obj = nullptr;

    const ParseResult parsed = self->parsers->regAdd.Parse(
        args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw, comp_id, data_obj);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // The payload arrives as a Python buffer; this is the one place the
    // adapter looks at its bytes, and only to hand the registry a span.
    std::optional<BufferView> payload_view;
    if (data_obj != nullptr && data_obj != Py_None) {
        payload_view = BufferView::Acquire(data_obj);
        if (!payload_view) {
            return nullptr;
        }
    }

    // Captured before the view dies: the size-mismatch diagnostic below is
    // raised after the registry has already decided, by which point the
    // buffer is gone.
    const Py_ssize_t view_len =
        payload_view ? static_cast<Py_ssize_t>(payload_view->SizeBytes()) : 0;

    std::optional<std::span<const std::byte>> payload;
    if (payload_view) {
        payload = payload_view->Bytes();
    }

    const std::expected<void, CPH::Error> added =
        self->registry->Add(CPH::Entity(handle_raw), comp_id, payload);

    if (!added) {
        if (added.error().Code().Is(CPH::EcsError::ComponentSizeMismatch)) {
            // The sizes are known only here (buffer length) and in the
            // registry (registered element size), so this message is
            // formatted at the boundary.
            return PyErr_Format(PyExc_ValueError, "Data size mismatch: expected %u, got %zd",
                                self->registry->ComponentElementSize(comp_id), view_len);
        }
        return RaiseEcsError(added.error());
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *RegistryRemove(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                       PyObject *kwnames) {
    uint64_t handle_raw;
    uint32_t comp_id;

    const ParseResult parsed = self->parsers->regEntComp.Parse(
        args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw, comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const std::expected<void, CPH::Error> removed =
        self->registry->Remove(CPH::Entity(handle_raw), comp_id);
    if (!removed) {
        return RaiseEcsError(removed.error());
    }
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *RegistryHas(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames) {
    uint64_t handle_raw;
    uint32_t comp_id;

    const ParseResult parsed = self->parsers->regEntComp.Parse(
        args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw, comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const std::expected<bool, CPH::Error> has =
        self->registry->Has(CPH::Entity(handle_raw), comp_id);
    if (!has) {
        return RaiseEcsError(has.error());
    }
    if (*has) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

// --- DATA ACCESS ---

[[nodiscard]] PyObject *RegistryGet(RegistryObject *self, PyObject *const *args, size_t nargsf,
                                    PyObject *kwnames) {
    uint64_t handle_raw;
    uint32_t comp_id;

    const ParseResult parsed = self->parsers->regEntComp.Parse(
        args, PyVectorcall_NARGS(nargsf), kwnames, handle_raw, comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const std::expected<std::optional<std::span<const std::byte>>, CPH::Error> found =
        self->registry->Get(CPH::Entity(handle_raw), comp_id);
    if (!found) {
        return RaiseEcsError(found.error());
    }
    if (!*found) {
        Py_RETURN_NONE;
    }
    return PyBytes_FromStringAndSize(reinterpret_cast<const char *>(found->value().data()),
                                     static_cast<Py_ssize_t>(found->value().size()));
}

[[nodiscard]] PyObject *RegistryGetView(RegistryObject *self, PyObject *const *args,
                                        size_t nargsf, PyObject *kwnames) {
    uint32_t comp_id;

    const ParseResult parsed =
        self->parsers->regCompOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    return MakeEcsProxy(self, comp_id, false);
}

[[nodiscard]] PyObject *RegistryGetEntities(RegistryObject *self, PyObject *const *args,
                                            size_t nargsf, PyObject *kwnames) {
    uint32_t comp_id;

    const ParseResult parsed =
        self->parsers->regCompOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    return MakeEcsProxy(self, comp_id, true);
}

// --- SYNCHRONIZATION ---

[[nodiscard]] PyObject *RegistrySyncFromWorld(RegistryObject *self, PyObject *const *args,
                                              size_t nargsf, PyObject *kwnames) {
    PyObject *world_obj = nullptr;
    uint32_t h_comp_id;
    int p_comp_id = -1;
    int r_comp_id = -1; // -1 means ignore

    const ParseResult parsed = self->parsers->regSyncPhys.Parse(
        args, PyVectorcall_NARGS(nargsf), kwnames, world_obj, h_comp_id, p_comp_id, r_comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    PhysicsWorldObject *world = reinterpret_cast<PhysicsWorldObject *>(world_obj);

    const CPH::EcsSyncConfig config{.handleComponent   = h_comp_id,
                                    .positionComponent = p_comp_id,
                                    .rotationComponent = r_comp_id};

    const std::expected<void, CPH::Error> synced =
        CPH::SyncEcsTransforms(world->physics, *self->registry, config);
    if (!synced) {
        return RaiseEcsError(synced.error());
    }
    Py_RETURN_NONE;
}

// --- STATS ---

[[nodiscard]] PyObject *RegistryGetActiveCount(RegistryObject *self,
                                               [[maybe_unused]] PyObject *args) {
    return PyLong_FromUnsignedLong(self->registry->active_entity_count());
}

[[nodiscard]] PyObject *RegistryGetComponentCount(RegistryObject *self, PyObject *const *args,
                                                  size_t nargsf, PyObject *kwnames) {
    uint32_t comp_id;

    const ParseResult parsed =
        self->parsers->regCompOnly.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, comp_id);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    return PyLong_FromSsize_t(self->registry->ComponentCount(comp_id));
}

static PyMethodDef kRegistryMethods[] = {

    CPH::Python::NoArgsMethod("create", RegistryCreate),
    CPH::Python::FastCallMethod("destroy", RegistryDestroy),
    CPH::Python::FastCallMethod("is_alive", RegistryIsAlive),
    CPH::Python::NoArgsMethod("clear", RegistryClear), // Wipes the registry
    CPH::Python::FastCallMethod("register_component", RegistryRegisterComponent),
    CPH::Python::FastCallMethod("add", RegistryAdd),
    CPH::Python::FastCallMethod("remove", RegistryRemove),
    CPH::Python::FastCallMethod("has", RegistryHas),
    CPH::Python::FastCallMethod("get", RegistryGet), // Single entity data access
    CPH::Python::FastCallMethod("get_view", RegistryGetView),
    CPH::Python::FastCallMethod("get_entities", RegistryGetEntities),
    CPH::Python::FastCallMethod("sync_from_world", RegistrySyncFromWorld),
    CPH::Python::NoArgsMethod("get_active_count", RegistryGetActiveCount), // ECS Statistics
    CPH::Python::FastCallMethod("get_component_count",
                                RegistryGetComponentCount), // ECS Statistics
    {nullptr, nullptr, 0, nullptr}};

static PyType_Slot RegistrySlots[] = {
    {.slot = Py_tp_new, .pfunc = reinterpret_cast<void *>(PyType_GenericNew)},
    {.slot = Py_tp_init, .pfunc = reinterpret_cast<void *>(RegistryInit)},
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(RegistryDealloc)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kRegistryMethods)},
    {},
};

PyType_Spec RegistrySpec = {
    .name      = "culverin._culverin_c.Registry",
    .basicsize = sizeof(RegistryObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = RegistrySlots,
};

} // namespace CPH::Python
