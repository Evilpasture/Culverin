#pragma once

// Thin adapter: the topology builder is CPH::SoftBodySettings
// (src/soft_bodies/); this object holds it plus the CPython header and the
// argument parsers.

#include "core/Compiler.hpp"
#include "soft_bodies/SoftBodyParsers.hpp"
#include "soft_bodies/SoftBodySettings.hpp"
#include <Python.h>

namespace CPH::Python {

struct SoftBodySharedSettingsObject {
    PyObject_HEAD CPH::SoftBodySettings *settings;
    CPH::Python::SoftBodySharedSettingsParsers *parsers;
};

extern PyType_Spec SoftBodySharedSettingsSpec;

} // namespace CPH::Python
