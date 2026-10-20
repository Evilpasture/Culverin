#pragma once
#include "bindings/PythonCommon.hpp"
#include "characters/CharacterParsers.hpp"
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Python.h>
#include <atomic>

namespace CPH::Python {

// --- Character Object ---
struct CharacterObject {
    PyObject_HEAD JPH::CharacterVirtual *character;
    struct PhysicsWorldObject *world;
    std::atomic<BodyHandle> handle;

    // Owned here; CharacterVirtual::SetListener stores a non-owning pointer,
    // so this must outlive `character` and be deleted after it.
    JPH::CharacterContactListener *listener;

    // Parsers
    CPH::Python::CharacterParsers *parsers;

    // ATOMIC INPUTS: Read by Jolt worker threads in callbacks
    std::atomic<float> push_strength;
    std::atomic<float> last_vx;
    std::atomic<float> last_vy;
    std::atomic<float> last_vz;

    // Non-atomic: Used by main thread only for rendering
    // AVOID FALSE SHARING.
    JPH_Real prev_px, prev_py, prev_pz;
    float prev_rx, prev_ry, prev_rz, prev_rw;
};

struct CharacterParams {
    float height;
    float radius;
    float max_slope;
};

extern PyType_Spec CharacterSpec;

struct PhysicsWorldObject;

[[nodiscard]] PyObject * WorldCreateCharacter(struct PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
