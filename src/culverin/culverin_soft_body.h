#pragma once
#include "culverin_arg_indices.h"
#include "culverin_soft_body_core.h"
#include "joltc.h"
#include <Python.h>
#include "culverin_python_declare.h"

typedef struct {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
} SoftBodySharedSettingsObject;
