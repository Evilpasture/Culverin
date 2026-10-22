#pragma once

// Thin adapter: the character is CPH::Character (src/characters/); this object
// holds it plus the CPython header and the argument parsers. No listener and
// no platform-interaction maths live here any more.

#include "bindings/PythonCommon.hpp"
#include "characters/Character.hpp"
#include "characters/CharacterParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct PhysicsWorldObject;

struct CharacterObject {
    PyObject_HEAD
    CPH::Character              *character; // owned
    struct PhysicsWorldObject   *world;
    CPH::Python::CharacterParsers *parsers;
};

extern PyType_Spec CharacterSpec;

[[nodiscard]] PyObject *WorldCreateCharacter(struct PhysicsWorldObject *self,
                                             PyObject *const *args, size_t nargsf,
                                             PyObject *kwnames);

} // namespace CPH::Python
