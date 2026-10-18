#pragma once
#include "bindings/PythonDeclarations.h"
#include "joltc.h"
#include "soft_bodies/SoftBodyCore.h"
#include "soft_bodies/SoftBodyParsers.hpp"
#include <Python.h>

typedef struct {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    CPH::Python::SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
} CPH_SoftBodySharedSettingsObject;
