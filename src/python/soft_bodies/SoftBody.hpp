#pragma once
#include "joltc.h"
#include "soft_bodies/SoftBodyCore.hpp"
#include "soft_bodies/SoftBodyParsers.hpp"
#include <Python.h>

namespace CPH::Python {

struct SoftBodySharedSettingsObject {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    CPH::Python::SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
};

} // namespace CPH::Python
