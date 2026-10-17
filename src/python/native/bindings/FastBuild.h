#pragma once

#include "core/CompilerSpecifics.h"

// Override FastBuild annotations with the project defaults.
#define FB_FORCE_INLINE CULV_FORCE_INLINE
#define FB_NODISCARD CULV_NODISCARD

// (Optional): If you later want to build Vec3 structs directly into PyTuple objects,
// you would define the function here and inject it like this:
/*
CULV_NODISCARD CULV_FORCE_INLINE static PyObject* fb_from_vec3f(Vec3f v) {
    return Py_BuildValue("(fff)", v.x, v.y, v.z); // paradoxical, isn't it? just treat this as an example.
}
#define FB_CUSTOM_CONVERTERS , Vec3f: fb_from_vec3f
*/

#include <fast_build.h> // IWYU pragma: export
