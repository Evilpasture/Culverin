// Python adapter for the soft-body topology builder.
//
// Acquiring the caller's buffers, deciding whether a buffer is a float32
// triplet, and mapping native error codes onto the exceptions callers expect
// live here. Packing vertices and faces into Jolt's layout, index validation
// and the constraint generation live in src/soft_bodies/SoftBodySettings.cpp.

#include "soft_bodies/SoftBodyBinding.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/PythonSupport.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Compiler.hpp"

namespace CPH::Python {

namespace {

// The two "modified after optimize" diagnostics have always differed between
// add_vertex/add_face/add_faces and add_vertices. They are raised here so the
// strings stay put.
constexpr const char *kOptimizedMessageSingle = "Cannot modify settings after optimize()";
constexpr const char *kOptimizedMessageBatch  = "Cannot modify after optimize()";

} // namespace

// --- SharedSettings Lifecycle ---

[[nodiscard]] int SoftBodySharedSettingsInit(SoftBodySharedSettingsObject *self,
                                             [[maybe_unused]] PyObject *args,
                                             [[maybe_unused]] PyObject *kwds) {
    self->settings = nullptr;
    self->parsers  = nullptr;

    // The native factory reproduces what joltc's
    // JPH_SoftBodySharedSettings_Create did: allocate, push a default material
    // (Jolt requires one to resolve face collisions), then AddRef.
    std::unique_ptr<CPH::SoftBodySettings> settings = CPH::SoftBodySettings::Create();
    if (!settings) {
        PyErr_NoMemory();
        return -1;
    }

    self->parsers = CPH::Python::NewParserGroup<CPH::Python::SoftBodySharedSettingsParsers>();
    if (!self->parsers) {
        PyErr_NoMemory();
        return -1;
    }

    self->settings = settings.release();
    return 0;
}

void SoftBodySharedSettingsDealloc(SoftBodySharedSettingsObject *self) {
    delete self->settings;
    self->settings = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    PyTypeObject *tp = Py_TYPE(self);
    tp->tp_free(reinterpret_cast<PyObject *>(self));
    Py_DECREF(tp);
}

// --- SharedSettings Topology Methods ---

[[nodiscard]] PyObject * SoftBodySharedSettingsAddVertex(SoftBodySharedSettingsObject *self,
                                                         PyObject *const *args, size_t nargsf,
                                                         PyObject *kwnames) {
    const Py_ssize_t nargs           = PyVectorcall_NARGS(nargsf);
    constexpr float default_inv_mass = 1.0F;

    PyObject *o_pos = nullptr;
    PyObject *o_vel = nullptr;
    float inv_mass  = default_inv_mass;

    const ParseResult parsed =
        self->parsers->SbssAddVertex.Parse(args, nargs, kwnames, o_pos, inv_mass, o_vel);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->settings->optimized()) {
        PyErr_SetString(PyExc_RuntimeError, kOptimizedMessageSingle);
        return nullptr;
    }

    // 1. Parse Position (Required)
    JPH::Float3 pos = {};
    if (ParseResult parsed_pos = ParseVec3Into(o_pos, pos.x, pos.y, pos.z); !parsed_pos) {
        RaiseParseError(parsed_pos.error());
        return nullptr;
    }

    // 2. Parse Velocity (Optional)
    JPH::Float3 vel = {};
    if (o_vel && o_vel != Py_None) {
        if (ParseResult parsed_vel = ParseVec3Into(o_vel, vel.x, vel.y, vel.z); !parsed_vel) {
            RaiseParseError(parsed_vel.error());
            return nullptr;
        }
    }

    // 3. Native append
    self->settings->AddVertex(pos, vel, inv_mass);
    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * SoftBodySharedSettingsAddVertices(SoftBodySharedSettingsObject *self,
                                                           PyObject *const *args, size_t nargsf,
                                                           PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject *o_pos        = nullptr;
    PyObject *o_mass       = nullptr;
    PyObject *o_vel        = nullptr;

    const ParseResult parsed =
        self->parsers->SbssAddVertices.Parse(args, nargs, kwnames, o_pos, o_mass, o_vel);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    if (self->settings->optimized()) {
        return PyErr_Format(PyExc_RuntimeError, kOptimizedMessageBatch);
    }

    // 1. Buffer Acquisitions
    Py_buffer pos_view;
    if (PyObject_GetBuffer(o_pos, &pos_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    constexpr auto vec3_size = 3 * sizeof(float);
    if (pos_view.len % vec3_size != 0) {
        PyBuffer_Release(&pos_view);
        return PyErr_Format(PyExc_ValueError, "positions must be float32 triplets");
    }

    const auto count = static_cast<uint32_t>(pos_view.len / vec3_size);

    Py_buffer mass_view = {};
    float *masses       = nullptr;
    if (o_mass && o_mass != Py_None) {
        if (PyObject_GetBuffer(o_mass, &mass_view, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&pos_view);
            return nullptr;
        }
        if (mass_view.len / sizeof(float) != count) {
            PyBuffer_Release(&pos_view);
            PyBuffer_Release(&mass_view);
            return PyErr_Format(PyExc_ValueError, "inv_masses length must match vertex count");
        }
        masses = static_cast<float *>(mass_view.buf);
    }

    Py_buffer vel_view = {};
    float *velocities  = nullptr;
    if (o_vel && o_vel != Py_None) {
        if (PyObject_GetBuffer(o_vel, &vel_view, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&pos_view);
            if (masses) {
                PyBuffer_Release(&mass_view);
            }
            return nullptr;
        }
        if (vel_view.len / vec3_size != count) {
            PyBuffer_Release(&pos_view);
            if (masses) {
                PyBuffer_Release(&mass_view);
            }
            PyBuffer_Release(&vel_view);
            return PyErr_Format(PyExc_ValueError, "velocities length must match vertex count");
        }
        velocities = static_cast<float *>(vel_view.buf);
    }

    // 2. Native packing (reserve-once and the zero-count early return are what
    // joltc's AddVertices did and are preserved inside the native method).
    const std::size_t float_count = static_cast<std::size_t>(pos_view.len) / sizeof(float);
    const std::size_t vert_count  = static_cast<std::size_t>(count);
    self->settings->AddVertices({static_cast<const float *>(pos_view.buf), float_count},
                                {masses, masses != nullptr ? vert_count : 0},
                                {velocities, velocities != nullptr ? vert_count * 3 : 0});

    PyBuffer_Release(&pos_view);
    if (masses) {
        PyBuffer_Release(&mass_view);
    }
    if (velocities) {
        PyBuffer_Release(&vel_view);
    }

    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * SoftBodySharedSettingsAddFace(SoftBodySharedSettingsObject *self,
                                                       PyObject *const *args, size_t nargsf,
                                                       PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    uint32_t v1            = 0;
    uint32_t v2            = 0;
    uint32_t v3            = 0;
    uint32_t mat_idx       = 0;

    const ParseResult parsed =
        self->parsers->SbssAddFace.Parse(args, nargs, kwnames, v1, v2, v3, mat_idx);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->settings->optimized()) {
        PyErr_SetString(PyExc_RuntimeError, kOptimizedMessageSingle);
        return nullptr;
    }

    const std::expected<void, Error> added = self->settings->AddFace(v1, v2, v3, mat_idx);
    if (!added) {
        if (added.error().Code().Is(SoftBodySettingsError::DegenerateFace)) {
            PyErr_SetString(PyExc_ValueError, "Face must have 3 distinct vertex indices");
            return nullptr;
        }
        // Out of range: the diagnostic spells out the indices and the total.
        return PyErr_Format(PyExc_IndexError,
                            "Face vertex index out of range (Indices: %u,%u,%u | Total: %u)", v1,
                            v2, v3, self->settings->vertex_count());
    }

    Py_RETURN_NONE;
}

[[nodiscard]] PyObject * SoftBodySharedSettingsAddFaces(SoftBodySharedSettingsObject *self,
                                                        PyObject *const *args, size_t nargsf,
                                                        PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    PyObject *o_ind        = nullptr;
    PyObject *o_mat        = nullptr;

    const ParseResult parsed =
        self->parsers->SbssAddFaces.Parse(args, nargs, kwnames, o_ind, o_mat);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (self->settings->optimized()) {
        PyErr_SetString(PyExc_RuntimeError, kOptimizedMessageSingle);
        return nullptr;
    }

    // 1. Index Buffer Acquisition
    constexpr auto triplet_size = 3 * sizeof(uint32_t);

    Py_buffer ind_view;
    if (PyObject_GetBuffer(o_ind, &ind_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }
    if (ind_view.len % triplet_size != 0) {
        PyBuffer_Release(&ind_view);
        PyErr_SetString(PyExc_ValueError, "indices buffer must be uint32 triplets");
        return nullptr;
    }

    const auto face_count = static_cast<uint32_t>(ind_view.len / triplet_size);
    const auto inds       = static_cast<const uint32_t *>(ind_view.buf);

    // 2. Optional Material Buffer Acquisition
    Py_buffer mat_view   = {};
    const uint32_t *mats = nullptr;
    if (o_mat && o_mat != Py_None) {
        if (PyObject_GetBuffer(o_mat, &mat_view, PyBUF_SIMPLE) != 0) {
            PyBuffer_Release(&ind_view);
            return nullptr;
        }
        if (mat_view.len / sizeof(uint32_t) != face_count) {
            PyBuffer_Release(&ind_view);
            PyBuffer_Release(&mat_view);
            PyErr_SetString(PyExc_ValueError, "materials length must match face count");
            return nullptr;
        }
        mats = static_cast<const uint32_t *>(mat_view.buf);
    }

    // 3. Native packing and validation
    const std::size_t index_count = static_cast<std::size_t>(face_count) * 3;
    const std::size_t mat_count   = mats != nullptr ? static_cast<std::size_t>(face_count) : 0;
    uint32_t bad_face             = 0;
    const std::expected<void, Error> added =
        self->settings->AddFaces({inds, index_count}, {mats, mat_count}, &bad_face);

    PyBuffer_Release(&ind_view);
    if (mats != nullptr) {
        PyBuffer_Release(&mat_view);
    }

    if (!added) {
        return PyErr_Format(PyExc_IndexError, "Face %u uses out-of-range vertex index", bad_face);
    }

    Py_RETURN_NONE;
}

// Method: settings.add_pinned_vertex(index) (METH_O)
/**
 * @deprecated Use add_vertices with an inverse mass of 0.0 instead.
 * This method is now a no-op because the underlying Jolt wrapper does not
 * support post-insertion vertex modification.
 */
[[gnu::deprecated("Use add_vertices with inv_mass=0.0 instead")]]
[[nodiscard]] PyObject *
SoftBodySharedSettingsAddPinnedVertex(SoftBodySharedSettingsObject *self, PyObject *arg) {
    // 1. Validate the argument type so we don't break existing logic flows
    const auto index_raw = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    // 2. Issue a Python-level DeprecationWarning
    // This allows Python users to see the fix required in their logs.
    const int warn_status = PyErr_WarnEx(PyExc_DeprecationWarning,
                                         "add_pinned_vertex() is deprecated and has no effect. "
                                         "Pin vertices by passing 0.0 mass in add_vertices().",
                                         1);

    if (warn_status < 0) {
        return nullptr; // User has warnings-as-errors enabled
    }

    // 3. Safety Check: Still perform bounds checking to help users debug their indices
    if (index_raw < 0 || static_cast<uint32_t>(index_raw) >= self->settings->vertex_count()) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    // 4. Return None without modifying Jolt state
    Py_RETURN_NONE;
}

[[gnu::const]]
static inline bool CheckBendType(int val, int max) {
    return ((val) >= 0 && (val) < (max));
}

// Method: settings.create_constraints(compliance, bend_type=1)
[[nodiscard]] PyObject *
SoftBodySharedSettingsCreateConstraints(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                        size_t nargsf, PyObject *kwnames) {
    const Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);
    float compliance       = 0.0001f;
    int bend_type          = 1; // EBendType::Distance

    const ParseResult parsed =
        self->parsers->SbssCreateConstraints.Parse(args, nargs, kwnames, compliance, bend_type);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }
    if (!CheckBendType(bend_type, 3)) {
        PyErr_Format(PyExc_ValueError,
                     "bend_type must be 0 (None), 1 (Distance), or 2 (Dihedral), got %d",
                     bend_type);
        return nullptr;
    }

    const std::expected<void, Error> created =
        self->settings->CreateConstraints(compliance, bend_type);
    if (!created) {
        PyErr_SetString(PyExc_RuntimeError, "create_constraints already called");
        return nullptr;
    }
    Py_RETURN_NONE;
}

// Method: settings.optimize() (METH_NOARGS)
[[nodiscard]] PyObject * SoftBodySharedSettingsOptimize(SoftBodySharedSettingsObject *self,
                                                        [[maybe_unused]] PyObject *args) {
    const std::expected<void, Error> optimized = self->settings->Optimize();
    if (!optimized) {
        PyErr_SetString(PyExc_RuntimeError,
                        "create_constraints must be called before optimize");
        return nullptr;
    }
    Py_RETURN_NONE;
}

// Method: settings.get_vertex_position(index) (METH_O) - Returns REST pose
[[nodiscard]] PyObject *
SoftBodySharedSettingsGetVertexPosition(SoftBodySharedSettingsObject *self, PyObject *arg) {
    const auto index_raw = PyLong_AsLong(arg);
    if (PyErr_Occurred()) {
        return nullptr;
    }

    // Safety: check vertex count using the updated API
    if (index_raw < 0 || static_cast<uint32_t>(index_raw) >= self->settings->vertex_count()) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    const std::expected<JPH::Float3, Error> vertex =
        self->settings->GetVertexPosition(static_cast<uint32_t>(index_raw));
    if (!vertex) {
        // Bounds were checked above, so this cannot happen; the branch is kept
        // because joltc's GetVertex could return false and the diagnostic has
        // to stay reachable for tests/test_diagnostics.py.
        PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve vertex data from Jolt");
        return nullptr;
    }

    // Return only the position component as requested
    return BuildTuple(vertex->x, vertex->y, vertex->z);
}

// The legacy entry remains available for compatibility; its implementation
// emits the user-facing Python DeprecationWarning when called.
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
static PyMethodDef kSoftBodySharedSettingsMethods[] = {

    CPH::Python::FastCallMethod("add_vertex", SoftBodySharedSettingsAddVertex),
    CPH::Python::FastCallMethod("add_vertices", SoftBodySharedSettingsAddVertices),
    CPH::Python::OneArgMethod("add_pinned_vertex", SoftBodySharedSettingsAddPinnedVertex),
    CPH::Python::OneArgMethod("get_vertex_position", SoftBodySharedSettingsGetVertexPosition),
    CPH::Python::FastCallMethod("add_face", SoftBodySharedSettingsAddFace),
    CPH::Python::FastCallMethod("add_faces", SoftBodySharedSettingsAddFaces),
    CPH::Python::FastCallMethod("create_constraints", SoftBodySharedSettingsCreateConstraints),
    CPH::Python::NoArgsMethod("optimize", SoftBodySharedSettingsOptimize),
    {nullptr, nullptr, 0, nullptr}};
#if defined(__clang__)
#    pragma clang diagnostic pop
#elif defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

static PyType_Slot SoftBodySharedSettingsSlots[] = {
    {.slot = Py_tp_new, .pfunc = reinterpret_cast<void *>(PyType_GenericNew)},
    {.slot = Py_tp_init, .pfunc = reinterpret_cast<void *>(SoftBodySharedSettingsInit)},
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(SoftBodySharedSettingsDealloc)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kSoftBodySharedSettingsMethods)},
    {},
};

PyType_Spec SoftBodySharedSettingsSpec = {
    .name      = "culverin._culverin_c.SoftBodySharedSettingsObject",
    .basicsize = sizeof(SoftBodySharedSettingsObject),
    .flags     = Py_TPFLAGS_DEFAULT,
    .slots     = SoftBodySharedSettingsSlots,
};

} // namespace CPH::Python
