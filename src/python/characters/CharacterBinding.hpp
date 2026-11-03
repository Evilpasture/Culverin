#pragma once

// Thin adapter: the character is CPH::Character (src/characters/); this object
// holds it plus the CPython header and the argument parsers. No listener and
// no platform-interaction maths live here any more.

#include "bindings/PyGcObject.hpp"
#include "bindings/PythonCommon.hpp"
#include "characters/Character.hpp"
#include "characters/CharacterParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct PhysicsWorldObject;

// Lifetime, traverse/clear and dealloc all come from PyGcObject; this struct
// only says what it owns. The PyObject_HEAD is supplied by the base.
struct CharacterObject : PyGcObject<CharacterObject> {
    CPH::Character                *character = nullptr; // owned
    struct PhysicsWorldObject     *world     = nullptr;
    CPH::Python::CharacterParsers *parsers   = nullptr;

    // The character is registered with the physics system and borrows the
    // world's raw pointer in ~Character, so it must go before the world
    // reference is dropped. Idempotent: clear and dealloc both call this.
    void BeforeClear() noexcept {
        delete character;
        character = nullptr;
    }

    template <typename Visitor>
    void VisitRefs(Visitor &&visit) noexcept {
        visit(world);
    }

    // Dealloc-only teardown for plain C++ members the base cannot know about.
    // Parsers hold no references and have no ordering constraint against the
    // world, so they go after the references, not before them.
    void BeforeFree() noexcept {
        delete parsers;
        parsers = nullptr;
    }
};

extern PyType_Spec CharacterSpec;

[[nodiscard]] PyObject *WorldCreateCharacter(struct PhysicsWorldObject *self,
                                             PyObject *const *args, size_t nargsf,
                                             PyObject *kwnames);

} // namespace CPH::Python
