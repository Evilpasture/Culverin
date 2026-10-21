#pragma once
#include "jolt/Jolt.hpp"
#include <Jolt/Physics/SoftBody/SoftBodySharedSettings.h>
#include "soft_bodies/SoftBodyCore.hpp"
#include "soft_bodies/SoftBodyParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct SoftBodySharedSettingsObject {
    PyObject_HEAD JPH::SoftBodySharedSettings *settings;
    CPH::Python::SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
};

} // namespace CPH::Python
