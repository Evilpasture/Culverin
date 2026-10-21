#include "soft_bodies/SoftBody.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/PythonSupport.hpp"
#include "core/Compiler.hpp"

namespace CPH::Python {

namespace {

/// What JPH_SoftBodySharedSettings_GetVertexCount returned, including the
/// narrowing to uint32_t that the index comparisons below rely on.
[[nodiscard]] inline uint32_t VertexCountOf(const JPH::SoftBodySharedSettings *s) noexcept {
    return (uint32_t)s->mVertices.size();
}

} // namespace

// --- SharedSettings Lifecycle ---

[[nodiscard]] int SoftBodySharedSettingsInit(SoftBodySharedSettingsObject *self,
                                                    [[maybe_unused]] PyObject *args,
                                                    [[maybe_unused]] PyObject *kwds) {
    // joltc's JPH_SoftBodySharedSettings_Create did three things, all of which
    // matter: allocate, push a default material ("Jolt requires at least one
    // material to resolve face collisions"), then AddRef. Reproduced verbatim so
    // the refcount and the material list are unchanged.
    self->settings = new JPH::SoftBodySharedSettings();
    self->settings->mMaterials.push_back(JPH::PhysicsMaterial::sDefault);
    self->settings->AddRef();
    self->num_vertices        = 0;
    self->constraints_created = false;
    self->optimized           = false;

    if (!self->settings) {
        PyErr_NoMemory();
        return -1;
    }
    self->parsers = CPH::Python::NewParserGroup<CPH::Python::SoftBodySharedSettingsParsers>();
    if (!self->parsers) {
        self->settings->Release();
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

void SoftBodySharedSettingsDealloc(SoftBodySharedSettingsObject *self) {
    if (self->settings) {
        self->settings->Release();
        self->settings = nullptr;
    }
    if (self->parsers) {

        delete self->parsers;
    }
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

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    // 1. Parse Position (Required)
    JPH::Float3 pos = {};
    if (ParseResult parsed = parse_vec3_into(o_pos, pos.x, pos.y, pos.z); !parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    // 2. Parse Velocity (Optional)
    JPH::Float3 vel = {};
    if (o_vel && o_vel != Py_None) {
        if (ParseResult parsed = parse_vec3_into(o_vel, vel.x, vel.y, vel.z); !parsed) {
            RaiseParseError(parsed.error());
            return nullptr;
        }
    }

    // 3. Construct the Jolt vertex and append it (what AddVertex did internally)
    const JPH::SoftBodySharedSettings::Vertex vertex(pos, vel, inv_mass);

    // 4. Dispatch and track
    self->settings->mVertices.push_back(vertex);
    self->num_vertices++;

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
    if (self->optimized) {
        return PyErr_Format(PyExc_RuntimeError, "Cannot modify after optimize()");
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

    // 2. The Interleave Allocation
    auto staging = static_cast<JPH::SoftBodySharedSettings::Vertex *>(Allocate(sizeof(JPH::SoftBodySharedSettings::Vertex) * count));
    if (!staging) {
        PyBuffer_Release(&pos_view);
        if (masses) {
            PyBuffer_Release(&mass_view);
        }
        if (velocities) {
            PyBuffer_Release(&vel_view);
        }
        return PyErr_NoMemory();
    }

    // 3. High-Speed Packing Loop
    const auto src_pos = static_cast<const float *>(pos_view.buf);
    for (size_t i = 0; i < count; ++i) {
        const size_t v_idx = i * 3;

        staging[i] = JPH::SoftBodySharedSettings::Vertex(
            JPH::Float3(src_pos[v_idx], src_pos[v_idx + 1], src_pos[v_idx + 2]),
            velocities ? JPH::Float3(velocities[v_idx], velocities[v_idx + 1],
                                     velocities[v_idx + 2])
                       : JPH::Float3(0, 0, 0),
            masses ? masses[i] : 1.0F);
    }

    // 4. Jolt Dispatch & Cleanup. joltc's AddVertices early-returned on a zero
    // count and reserved once for the batch; both preserved.
    if (count != 0) {
        self->settings->mVertices.reserve(self->settings->mVertices.size() + count);
        for (uint32_t i = 0; i < count; ++i) {
            self->settings->mVertices.push_back(staging[i]);
        }
    }
    self->num_vertices += count;

    Deallocate(staging);
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

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    // 1. Degenerate Triangle Validation
    if (v1 == v2 || v2 == v3 || v1 == v3) {
        PyErr_SetString(PyExc_ValueError, "Face must have 3 distinct vertex indices");
        return nullptr;
    }

    // 2. Source of Truth Validation
    const auto vertex_count = VertexCountOf(self->settings);
    if (v1 >= vertex_count || v2 >= vertex_count || v3 >= vertex_count) {
        PyErr_Format(PyExc_IndexError,
                     "Face vertex index out of range (Indices: %u,%u,%u | Total: %u)", v1, v2, v3,
                     vertex_count);
        return nullptr;
    }

    // 3. Pack into Jolt's new AoS structure using designated initializers
    const JPH::SoftBodySharedSettings::Face face(v1, v2, v3, mat_idx);

    // 4. Dispatch to Jolt
    self->settings->mFaces.push_back(face);

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

    if (self->optimized) {
        PyErr_SetString(PyExc_RuntimeError, "Cannot modify settings after optimize()");
        return nullptr;
    }

    // 1. Index Buffer Acquisition
    Py_buffer ind_view;
    if (PyObject_GetBuffer(o_ind, &ind_view, PyBUF_SIMPLE) != 0) {
        return nullptr;
    }

    constexpr auto triplet_size = 3 * sizeof(uint32_t);
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

    // 3. Resource & Source-of-Truth Validation
    const auto vertex_count = VertexCountOf(self->settings);

    // 4. Interleave Packing Loop
    auto staging = static_cast<JPH::SoftBodySharedSettings::Face *>(Allocate(sizeof(JPH::SoftBodySharedSettings::Face) * face_count));
    if (staging == nullptr) {
        PyBuffer_Release(&ind_view);
        if (mats) {
            PyBuffer_Release(&mat_view);
        }
        return PyErr_NoMemory();
    }

    constexpr uint32_t STRIDE = 3;
    for (uint32_t i = 0; i < face_count; ++i) {
        const auto v1 = inds[(i * STRIDE) + 0];
        const auto v2 = inds[(i * STRIDE) + 1];
        const auto v3 = inds[(i * STRIDE) + 2];

        // Hardware-friendly bounds check
        if (v1 >= vertex_count || v2 >= vertex_count || v3 >= vertex_count) {
            Deallocate(staging);
            PyBuffer_Release(&ind_view);
            if (mats) {
                PyBuffer_Release(&mat_view);
            }
            PyErr_Format(PyExc_IndexError, "Face %u uses out-of-range vertex index", i);
            return nullptr;
        }

        // Interleave into Jolt's AoS layout
        staging[i] = JPH::SoftBodySharedSettings::Face(v1, v2, v3, (mats != nullptr) ? mats[i] : 0);
    }

    // 5. Atomic Push to Jolt. Reserve-once and the zero-count early return are
    // what joltc's AddFaces did.
    if (face_count != 0) {
        self->settings->mFaces.reserve(self->settings->mFaces.size() + face_count);
        for (uint32_t i = 0; i < face_count; ++i) {
            self->settings->mFaces.push_back(staging[i]);
        }
    }

    // 6. Finalization
    Deallocate(staging);
    PyBuffer_Release(&ind_view);
    if (mats != nullptr) {
        PyBuffer_Release(&mat_view);
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
    const auto vertex_count = VertexCountOf(self->settings);
    if (index_raw < 0 || static_cast<uint32_t>(index_raw) >= vertex_count) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    // 4. Return None without modifying Jolt state
    Py_RETURN_NONE;
}
[[gnu::const]]
static inline bool check_bend_type(int val, int max) {
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
    if (!check_bend_type(bend_type, 3)) {
        PyErr_Format(PyExc_ValueError,
                     "bend_type must be 0 (None), 1 (Distance), or 2 (Dihedral), got %d",
                     bend_type);
        return nullptr;
    }
    if (self->constraints_created) {
        PyErr_SetString(PyExc_RuntimeError, "create_constraints already called");
        return nullptr;
    }
    self->constraints_created = true;

    // Binder handles both Edge compliance and Shear compliance. joltc built a
    // VertexAttributes array sized to the vertex count with only those two
    // fields set, leaving mBendCompliance at its FLT_MAX default and the LRA
    // fields at theirs; same thing here.
    JPH::Array<JPH::SoftBodySharedSettings::VertexAttributes> attributes;
    attributes.resize(self->settings->mVertices.size());
    for (auto &attr : attributes) {
        attr.mCompliance      = compliance;
        attr.mShearCompliance = compliance;
    }
    self->settings->CreateConstraints(
        attributes.data(), (JPH::uint)attributes.size(),
        static_cast<JPH::SoftBodySharedSettings::EBendType>(bend_type));
    Py_RETURN_NONE;
}

// Method: settings.optimize() (METH_NOARGS)
[[nodiscard]] PyObject * SoftBodySharedSettingsOptimize(SoftBodySharedSettingsObject *self,
                                                        [[maybe_unused]] PyObject *args) {
    if (!self->constraints_created) {
        PyErr_SetString(PyExc_RuntimeError, "create_constraints must be called before optimize");
        return nullptr;
    }
    self->settings->Optimize();
    self->optimized = true;
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
    const auto vertex_count = VertexCountOf(self->settings);
    if (index_raw < 0 || static_cast<uint32_t>(index_raw) >= vertex_count) {
        PyErr_SetString(PyExc_IndexError, "Vertex index out of range");
        return nullptr;
    }

    const auto index = static_cast<uint32_t>(index_raw);

    // Designated initialization for the return buffer
    // Bounds were checked against VertexCountOf above, so this cannot fail; the
    // error path is kept because joltc's GetVertex could return false and the
    // diagnostic has to stay reachable for tests/test_diagnostics.py.
    if (index >= self->settings->mVertices.size()) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to retrieve vertex data from Jolt");
        return nullptr;
    }
    const JPH::SoftBodySharedSettings::Vertex &vertex = self->settings->mVertices[index];

    // Return only the position component as requested
    return BuildTuple(vertex.mPosition.x, vertex.mPosition.y, vertex.mPosition.z);
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
