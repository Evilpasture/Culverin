#pragma once
#include "culverin_arg_indices.h"
#include "culverin_types.h"
#include "joltc.h"
#include <Python.h>

typedef struct {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    SoftBodySharedSettingsParsers *parsers;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
} SoftBodySharedSettingsObject;

typedef struct {
    JPH_Real *vertices; // Shadow buffer for positions (Vec3/Vec4)
    float *normals;     // Optional shadow buffer for normals
    float *velocities;  // Optional
    uint32_t num_vertices;
} SoftBodyShadow;