// Python adapter for CPH::Character.
//
// The contact listener, the platform-interaction maths and the Jolt update all
// live in src/characters/Character.cpp. Everything here parses arguments,
// calls the native character and builds Python return values.

#include "characters/CharacterBinding.hpp"
#include "bindings/FastBuild.hpp"
#include "bindings/ThreadState.hpp"
#include "bindings/ModuleState.hpp"
#include "bindings/PythonCommon.hpp"
#include "bindings/PythonSupport.hpp"
#include "core/Compiler.hpp"
#include "core/Types.hpp"
#include "math/MathCore.hpp"
#include "physics/PhysicsSync.hpp"
#include <Jolt/Math/Math.h>
#include <cmath>
#include "bindings/Result.hpp"

namespace CPH::Python {

// --- Move --------------------------------------------------------------------

[[nodiscard]] BoundResult CharacterMove(CharacterObject *self, PyObject *const *args, size_t nargsf,
                                      PyObject *kwnames) {
    JPH::Float3 v_in{0.0f, 0.0f, 0.0f};
    float dt = 0.0f;

    const ParseResult parsed =
        self->parsers->charMove.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, v_in, dt);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    if (!ValidateFinite("Character velocity", v_in.x, v_in.y, v_in.z)) {
        return FailCurrent();
    }

    // The GIL is dropped for the call: the native move parks on the step-sync
    // condition while a step is in flight, and holding the GIL there would
    // deadlock against the stepping thread.
    {
        const AllowThreads allow;
    self->character->Move(v_in.x, v_in.y, v_in.z, dt);
    }

    return BorrowedValue(Py_None);
}

// --- Transforms --------------------------------------------------------------

[[nodiscard]] BoundResult CharacterGetPosition(CharacterObject *self,
                                             PyObject *ignored [[maybe_unused]]) {
    const JPH::RVec3 position = self->character->GetPosition();

    // BuildTuple, not PyTuple_New + PyTuple_SET_ITEM: SET_ITEM *steals*, so
    // `SET_ITEM(t, i, PyFloat_FromDouble(v))` stored NULL into the slot if the
    // float conversion failed (e.g. MemoryError). Py_DECREF tolerates NULL
    // slots, so the tuple survived and reached the caller -- CPython then
    // crashed when it touched the hole, far from the cause. BuildTuple checks
    // every element and fails the whole call instead.
    //
    // RVec3 carries the full precision the world was built with, and BuildTuple
    // dispatches double to PyFloat_FromDouble without narrowing.
    return Adopt(BuildTuple(position.GetX(), position.GetY(), position.GetZ()));
}

[[nodiscard]] BoundResult CharacterGetLinearVelocity(CharacterObject *self,
                                                   PyObject *ignored [[maybe_unused]]) {
    const JPH::Vec3 velocity = self->character->GetLinearVelocity();
    return Adopt(BuildTuple(velocity.GetX(), velocity.GetY(), velocity.GetZ()));
}

[[nodiscard]] BoundResult CharacterSetPosition(CharacterObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    PosStride pos = {};

    const ParseResult parsed =
        self->parsers->setPosChar.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, pos);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    {
        const AllowThreads allow;
    self->character->SetPosition(JPH::RVec3(pos.x, pos.y, pos.z));
    }

    return BorrowedValue(Py_None);
}

[[nodiscard]] BoundResult CharacterSetRotation(CharacterObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    AuxStride rot = {.x = 0.0f, .y = 0.0f, .z = 0.0f, .w = 1.0f};

    const ParseResult parsed =
        self->parsers->setRotChar.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, rot);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    {
        const AllowThreads allow;
    self->character->SetRotation(JPH::Quat(rot.x, rot.y, rot.z, rot.w));
    }

    return BorrowedValue(Py_None);
}

[[nodiscard]] BoundResult CharacterSetStrength(CharacterObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    float strength = 0.0f;

    const ParseResult parsed =
        self->parsers->setStrengthChar.Parse(args, PyVectorcall_NARGS(nargsf), kwnames, strength);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    {
        const AllowThreads allow;
    self->character->SetStrength(strength);
    }

    return BorrowedValue(Py_None);
}

// Interpolates between the previous and current shadowed transform. Pure
// wrapping: the interpolation itself is MathInterpolateCharacterTransform.
[[nodiscard]] BoundResult CharacterGetRenderTransform(CharacterObject *self, PyObject *arg) {
    const auto alpha_raw = PyFloat_AsDouble(arg);
    if (alpha_raw == -1.0 && PyErr_Occurred()) {
        return FailCurrent();
    }

    constexpr float kMinAlpha = 0.0f;
    constexpr float kMaxAlpha = 1.0f;
    const auto alpha = static_cast<float>(std::fmax(kMinAlpha, std::fmin(kMaxAlpha, alpha_raw)));

    alignas(16) float result_p[3];
    alignas(16) float result_r[4];

    {
        const std::scoped_lock shadow(self->world->physics.shadowLock);
        BlockUntilNotStepping(self->world);

        const std::uint32_t slot  = self->character->Handle().Index();
        const std::uint32_t dense = self->world->physics.bodies.slotToDense[slot];

        const auto *prev_p = &self->world->physics.bodies.prevPositions[dense];
        const auto *prev_r = &self->world->physics.bodies.prevRotations[dense];

        const JPH::RVec3 end_p = self->character->Virtual()->GetPosition();
        const JPH::Quat end_r  = self->character->Virtual()->GetRotation();

        MathInterpolateCharacterTransform(prev_p, prev_r, end_p, end_r, alpha, result_p, result_r);
    }

    return Adopt(BuildTuple(BuildTuple(result_p[0], result_p[1], result_p[2]),
                            BuildTuple(result_r[0], result_r[1], result_r[2], result_r[3])));
}

[[nodiscard]] BoundResult CharacterIsGrounded(CharacterObject *self,
                                            PyObject *ignored [[maybe_unused]]) {
    if (self->character->IsGrounded()) {
        return BorrowedValue(Py_True);
    }
    return BorrowedValue(Py_False);
}

[[nodiscard]] BoundResult CharacterGetHandle(CharacterObject *self,
                                           [[maybe_unused]] void *closure) {
    return Adopt(BuildValue(self->character->Handle()));
}

// --- Object lifecycle --------------------------------------------------------

// There is no hand-written dealloc, traverse or clear here: PyGcObject derives
// all three from the hooks in CharacterBinding.hpp.

// --- Creation ----------------------------------------------------------------

[[nodiscard]] BoundResult WorldCreateCharacter(PhysicsWorldObject *self, PyObject *const *args,
                                             size_t nargsf, PyObject *kwnames) {
    PosStride pos = {};
    float height  = 1.8f;
    float radius  = 0.4f;
    float step_h  = 0.4f;
    float slope   = 45.0f;

    const ParseResult parsed = self->parsers->createChar.Parse(args, PyVectorcall_NARGS(nargsf),
                                                               kwnames, pos, height, radius,
                                                               step_h, slope);
    if (!parsed) {
        RaiseParseError(parsed.error());
        return FailCurrent();
    }

    const CPH::CharacterParams params{.height = height, .radius = radius, .maxSlope = slope};
    const JPH::RVec3 position(pos.x, pos.y, pos.z);

    std::unique_ptr<CPH::Character> character;
    {
        const AllowThreads allow;
    character = CPH::Character::Create(self->physics, params, position);
    }

    if (!character) {
        // Either the capsule shape failed or the world could not grow. Both
        // have always surfaced as a bare MemoryError here.
        return Adopt(PyErr_NoMemory());
    }

    auto *obj = CharacterObject::New(
        reinterpret_cast<PyTypeObject *>(GetModuleState(PyType_GetModule(Py_TYPE(self)))
                                             ->characterType));
    if (!obj) {
        return FailCurrent();
    }

    obj->parsers = CPH::Python::NewParserGroup<CPH::Python::CharacterParsers>();
    if (!obj->parsers) {
        Py_DECREF(obj);
        PyErr_NoMemory();
        return FailCurrent();
    }

    obj->world     = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(self));
    obj->character = character.release();

    // A created character may have grown the world, so an exported buffer's
    // shape has to follow the new body count.
    self->viewShape[0] =
        static_cast<Py_ssize_t>(self->physics.count.load(std::memory_order::relaxed));

    return Adopt(obj->Publish());
}

static PyMethodDef kCharacterMethods[] = {

    CPH::Python::Method<&CharacterMove>("move"),
    CPH::Python::NoArgsMethod<&CharacterGetPosition>("get_position"),
    CPH::Python::NoArgsMethod<&CharacterGetLinearVelocity>("get_linear_velocity"),
    CPH::Python::Method<&CharacterSetPosition>("set_position"),
    CPH::Python::Method<&CharacterSetRotation>("set_rotation"),
    CPH::Python::NoArgsMethod<&CharacterIsGrounded>("is_grounded"),
    CPH::Python::Method<&CharacterSetStrength>("set_strength"),
    CPH::Python::OneArgMethod<&CharacterGetRenderTransform>("get_render_transform"),
    {nullptr, nullptr, 0, nullptr}};

static PyGetSetDef CharacterGetSets[] = {
    CPH::Python::GetSet<&CharacterGetHandle>("handle"),
    {},
};

static PyType_Slot CharacterSlots[] = {
    CPH::Python::GcDeallocSlot<CharacterObject>(),
    CPH::Python::GcTraverseSlot<CharacterObject>(),
    CPH::Python::GcClearSlot<CharacterObject>(),
    {.slot = Py_tp_methods, .pfunc = reinterpret_cast<void *>(kCharacterMethods)},
    {.slot = Py_tp_getset, .pfunc = reinterpret_cast<void *>(CharacterGetSets)},
    {},
};

PyType_Spec CharacterSpec =
    CPH::Python::GcTypeSpec("culverin._culverin_c.Character", sizeof(CharacterObject),
                            CharacterSlots);

} // namespace CPH::Python
