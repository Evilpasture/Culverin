#pragma once

// Thin adapter: the skeleton, the settings and the ragdoll are the native
// classes in src/ragdolls/. These objects hold one of those plus the CPython
// header and the argument parsers.

#include "core/Compiler.hpp"
#include "physics/PhysicsWorldBinding.hpp"
#include "ragdolls/Ragdoll.hpp"
#include "ragdolls/RagdollParsers.hpp"
#include "bindings/PyGcObject.hpp"
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

// Lifetime, traverse/clear and dealloc come from PyGcObject. The PyObject_HEAD
// is supplied by the base.
struct RagdollObject : PyGcObject<RagdollObject> {
    CPH::Ragdoll                *ragdoll = nullptr;
    struct PhysicsWorldObject   *world   = nullptr;
    CPH::Python::RagdollParsers *parsers = nullptr;

    // ~Ragdoll detaches through the world's shadow lock and releases body
    // slots, so the ragdoll goes before the world reference is dropped.
    // destroy() nulls `ragdoll`, so a second call from clear or dealloc is a
    // no-op.
    void BeforeClear() noexcept {
        delete ragdoll;
        ragdoll = nullptr;
    }

    template <typename Visitor>
    void VisitRefs(Visitor &&visit) noexcept {
        visit(world);
    }

    void BeforeFree() noexcept {
        delete parsers;
        parsers = nullptr;
    }
};

extern PyType_Spec SkeletonSpec;
extern PyType_Spec RagdollSettingsSpec;
extern PyType_Spec RagdollSpec;

} // namespace CPH::Python
