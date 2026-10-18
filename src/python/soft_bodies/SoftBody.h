#pragma once
#include "bindings/PythonDeclarations.h"
#include "bindings/parsing/ParserRegistry.hpp"
#include "joltc.h"
#include "soft_bodies/SoftBodyCore.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    CPH_SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
} CPH_SoftBodySharedSettingsObject;
