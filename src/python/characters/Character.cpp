// Python adapter for CPH::Character.
//
// The contact listener, the platform-interaction maths and the Jolt update all
// live in src/characters/Character.cpp. Everything here parses arguments,
// calls the native character and builds Python return values.

#include "characters/CharacterBinding.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/PythonSupport.hpp"
#include "core/Compiler.hpp"
#include "core/Types.hpp"
#include "math/MathCore.hpp"
#include "physics/PhysicsSync.hpp"
#include <Jolt/Math/Math.h>
#include <cmath>

namespace CPH::Python {

// --- Move --------------------------------------------------------------------

[[nodiscard]] PyObject *CharacterMove(CharacterObject *self, PyObject *const *args, size_t nargsf,
                                      PyObject *kwnames) {
    JPH::Float3 v_in{0.0f, 0.0f, 0.0f};
    float dt = 0.0f;

    const ParseResult parsed =
        self->parsers->CharMove.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_in, dt);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    if (!ValidateFinite("Character velocity", v_in.x, v_in.y, v_in.z)) {
        return nullptr;
    }

    // The GIL is dropped for the call: the native move parks on the step-sync
    // condition while a step is in flight, and holding the GIL there would
    // deadlock against the stepping thread.
    Py_BEGIN_ALLOW_THREADS
    self->character->Move(v_in.x, v_in.y, v_in.z, dt);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// --- Transforms --------------------------------------------------------------

[[nodiscard]] PyObject *CharacterGetPosition(CharacterObject *self,
                                             PyObject *ignored [[maybe_unused]]) {
    const JPH::RVec3 position = self->character->GetPosition();

    PyObject *result = PyTuple_New(3);
    if (!result) {
        return nullptr;
    }
    // RVec3 carries the full precision the world was built with.
    PyTuple_SET_ITEM(result, 0, PyFloat_FromDouble(position.GetX()));
    PyTuple_SET_ITEM(result, 1, PyFloat_FromDouble(position.GetY()));
    PyTuple_SET_ITEM(result, 2, PyFloat_FromDouble(position.GetZ()));
    return result;
}

[[nodiscard]] PyObject *CharacterGetLinearVelocity(CharacterObject *self,
                                                   PyObject *ignored [[maybe_unused]]) {
    const JPH::Vec3 velocity = self->character->GetLinearVelocity();
    return BuildTuple(velocity.GetX(), velocity.GetY(), velocity.GetZ());
}

[[nodiscard]] PyObject *CharacterSetPosition(CharacterObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    PosStride pos = {};

    const ParseResult parsed =
        self->parsers->SetPosChar.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, pos);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS
    self->character->SetPosition(JPH::RVec3(pos.x, pos.y, pos.z));
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *CharacterSetRotation(CharacterObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    AuxStride rot = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    const ParseResult parsed =
        self->parsers->SetRotChar.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, rot);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS
    self->character->SetRotation(JPH::Quat(rot.x, rot.y, rot.z, rot.w));
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

[[nodiscard]] PyObject *CharacterSetStrength(CharacterObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    float strength = 0.0f;

    const ParseResult parsed =
        self->parsers->SetStrengthChar.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, strength);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    Py_BEGIN_ALLOW_THREADS
    self->character->SetStrength(strength);
    Py_END_ALLOW_THREADS

    Py_RETURN_NONE;
}

// Interpolates between the previous and current shadowed transform. Pure
// wrapping: the interpolation itself is MathInterpolateCharacterTransform.
[[nodiscard]] PyObject *CharacterGetRenderTransform(CharacterObject *self, PyObject *arg) {
    const auto alpha_raw = PyFloat_AsDouble(arg);
    if (alpha_raw == -1.0 && PyErr_Occurred()) {
        return nullptr;
    }

    constexpr float kMinAlpha = 0.0f;
    constexpr float kMaxAlpha = 1.0f;
    const auto alpha = static_cast<float>(std::fmax(kMinAlpha, std::fmin(kMaxAlpha, alpha_raw)));

    alignas(16) float result_p[3];
    alignas(16) float result_r[4];

    {
        const std::scoped_lock shadow(self->world->physics.shadow_lock);
        BlockUntilNotStepping(self->world);

        const std::uint32_t slot  = self->character->Handle().Index();
        const std::uint32_t dense = self->world->physics.bodies.slot_to_dense[slot];

        const auto *prev_p = &self->world->physics.bodies.prev_positions[dense];
        const auto *prev_r = &self->world->physics.bodies.prev_rotations[dense];

        const JPH::RVec3 end_p = self->character->Virtual()->GetPosition();
        const JPH::Quat end_r  = self->character->Virtual()->GetRotation();

        MathInterpolateCharacterTransform(prev_p, prev_r, end_p, end_r, alpha, result_p, result_r);
    }

    return BuildTuple(BuildTuple(result_p[0], result_p[1], result_p[2]),
                      BuildTuple(result_r[0], result_r[1], result_r[2], result_r[3]));
}

[[nodiscard]] PyObject *CharacterIsGrounded(CharacterObject *self,
                                            PyObject *ignored [[maybe_unused]]) {
    if (self->character->IsGrounded()) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

[[nodiscard]] PyObject *CharacterGetHandle(CharacterObject *self,
                                           [[maybe_unused]] void *closure) {
    return BuildValue(self->character->Handle());
}

// --- Object lifecycle --------------------------------------------------------

[[nodiscard]] int CharacterTraverse(CharacterObject *self, visitproc visit, void *arg) {
    Py_VISIT(self->world);
    return 0;
}

[[nodiscard]] int CharacterClear(CharacterObject *self) {
    Py_CLEAR(self->world);
    return 0;
}

void CharacterDealloc(CharacterObject *self) {
    PyObject_GC_UnTrack(self);

    // Destroying the character removes its step listener, releases the Jolt
    // character and gives its body slot back; see ~CPH::Character.
    delete self->character;
    self->character = nullptr;

    delete self->parsers;
    self->parsers = nullptr;

    Py_XDECREF(self->world);
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject *>(self));
}

// --- Creation ----------------------------------------------------------------

[[nodiscard]] PyObject *WorldCreateCharacter(PhysicsWorldObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    PosStride pos = {};
    float height  = 1.8f;
    float radius  = 0.4f;
    float step_h  = 0.4f;
    float slope   = 45.0f;

    const ParseResult parsed = self->parsers->CreateChar.Parse(args, PyVectorcall_NARGS(nargsf),
                                                               kwnames, pos, height, radius,
                                                               step_h, slope);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    const CPH::CharacterParams params{.height = height, .radius = radius, .max_slope = slope};
    const JPH::RVec3 position(pos.x, pos.y, pos.z);

    std::unique_ptr<CPH::Character> character;
    Py_BEGIN_ALLOW_THREADS
    character = CPH::Character::Create(self->physics, params, position);
    Py_END_ALLOW_THREADS

    if (!character) {
        // Either the capsule shape failed or the world could not grow. Both
        // have always surfaced as a bare MemoryError here.
        return PyErr_NoMemory();
    }

    auto *obj = PyObject_GC_New(
        CharacterObject,
        reinterpret_cast<PyTypeObject *>(GetModuleState(PyType_GetModule(Py_TYPE(self)))
                                             ->CharacterType));
    if (!obj) {
        return nullptr;
    }

    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::CharacterParsers>();
    if (!obj->parsers) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return nullptr;
    }

    obj->world     = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->character = character.release();

    // A created character may have grown the world, so an exported buffer's
    // shape has to follow the new body count.
    self->view_shape[0] =
        static_cast<Py_ssize_t>(self->physics.count.load(std::memory_order::relaxed));

    PyObject_GC_Track(reinterpret_cast<PyObject *>(obj));
    return reinterpret_cast<PyObject *>(obj);
}

static PyMethodDef kCharacterMethods[] = {

    CPH::Python::FastCallMethod("move", CharacterMove),
    CPH::Python::NoArgsMethod("get_position", CharacterGetPosition),
    CPH::Python::NoArgsMethod("get_linear_velocity", CharacterGetLinearVelocity),
    CPH::Python::FastCallMethod("set_position", CharacterSetPosition),
    CPH::Python::FastCallMethod("set_rotation", CharacterSetRotation),
    CPH::Python::NoArgsMethod("is_grounded", CharacterIsGrounded),
    CPH::Python::FastCallMethod("set_strength", CharacterSetStrength),
    CPH::Python::OneArgMethod("get_render_transform", CharacterGetRenderTransform),
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef CharacterGetSets[] = {
    CPH::Python::GetSet("handle", CharacterGetHandle),
    {},
};

static PyType_Slot CharacterSlots[] = {
    {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(CharacterDealloc)},
    {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(CharacterTraverse)},
    {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(CharacterClear)},
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kCharacterMethods)},
    {.slot = Py_tp_getset, .pfunc = reinterpret_cast<void *>(CharacterGetSets)},
    {},
};

PyType_Spec CharacterSpec = {
    .name      = "culverin._culverin_c.Character",
    .basicsize = sizeof(CharacterObject),
    .flags     = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC,
    .slots     = CharacterSlots,
};

} // namespace CPH::Python
