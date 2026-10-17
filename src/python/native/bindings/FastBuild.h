#pragma once

#include "core/CompilerSpecifics.h"

// Override FastBuild annotations with the project defaults.
#define CULV_PY_BUILD_FORCE_INLINE CULV_FORCE_INLINE
#define CULV_PY_BUILD_NODISCARD CULV_NODISCARD

// (Optional): If you later want to build Vec3 structs directly into PyTuple objects,
// you would define the function here and inject it like this:
/*
CULV_NODISCARD CULV_FORCE_INLINE static PyObject* culv_py_build_from_vec3f(Vec3f v) {
    return Py_BuildValue("(fff)", v.x, v.y, v.z); // paradoxical, isn't it? just treat this as an example.
}
#define CULV_PY_BUILD_CUSTOM_CONVERTERS , Vec3f: culv_py_build_from_vec3f
*/

#include "FastBuildCore.h" // IWYU pragma: export
