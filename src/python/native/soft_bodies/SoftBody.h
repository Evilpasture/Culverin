#pragma once
#include "bindings/parsing/ArgIndices.h"
#include "soft_bodies/SoftBodyCore.h"
#include "joltc.h"
#include <Python.h>
#include "bindings/PythonDeclarations.h"

typedef struct {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    CPH_SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
} CPH_SoftBodySharedSettingsObject;
