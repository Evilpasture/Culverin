#pragma once
#include "bindings/PythonCommon.hpp"
#include "characters/CharacterParsers.hpp"
#include <Python.h>
#include <atomic>

namespace CPH::Python {

// --- Character Object ---
struct CharacterObject {
    PyObject_HEAD JPH_CharacterVirtual *character;
    struct PhysicsWorldObject *world;
    std::atomic<BodyHandle> handle;

    // Filters and listeners
    JPH_BodyFilter *body_filter;
    JPH_ShapeFilter *shape_filter;
    JPH_BroadPhaseLayerFilter *bp_filter;
    JPH_ObjectLayerFilter *obj_filter;
    JPH_CharacterContactListener *listener;

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

/* We expose the Procs table so the main module can assign it
   when creating the Character Virtual instance.
*/

extern const JPH_CharacterContactListener_Procs kCharListenerProcs;

extern PyType_Spec CharacterSpec;

struct PhysicsWorldObject;

[[nodiscard]] PyObject * WorldCreateCharacter(struct PhysicsWorldObject *self,
                                                                PyObject *const *args,
                                                                size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
