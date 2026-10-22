#pragma once

// Thin adapter: the skeleton, the settings and the ragdoll are the native
// classes in src/ragdolls/. These objects hold one of those plus the CPython
// header and the argument parsers.

#include "core/Compiler.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "ragdolls/Ragdoll.hpp"
#include "ragdolls/RagdollParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct SkeletonObject {
    PyObject_HEAD
    CPH::Skeleton *skeleton;
    CPH::Python::SkeletonParsers *parsers;
};

struct RagdollSettingsObject {
    PyObject_HEAD
    CPH::RagdollSettings *settings;
    // Kept so the part builder can reach the shape cache.
    struct PhysicsWorldObject *world;
    CPH::Python::RagdollSettingsParsers *parsers;
};

struct RagdollObject {
    PyObject_HEAD
    CPH::Ragdoll *ragdoll;
    struct PhysicsWorldObject *world;
    CPH::Python::RagdollParsers *parsers;
};

extern PyType_Spec SkeletonSpec;
extern PyType_Spec RagdollSettingsSpec;
extern PyType_Spec RagdollSpec;

} // namespace CPH::Python
