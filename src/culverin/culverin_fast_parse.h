#pragma once

#include "culverin_compiler_specifics.h"
#include "culverin_types.h"
#include "culverin_parsers.h"

// Define Culverin-Specific Converters
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_vec3f(PyObject *o, void *t) {
    Vec3f *v = (Vec3f *)t;
    return parse_vec3_f32(o, &v->x, &v->y, &v->z) != 0;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_pos_stride(PyObject *o, void *t) {
    PosStride *v = (PosStride *)t;
    return parse_vec3_r64(o, &v->x, &v->y, &v->z) != 0;
}
CULV_MAYBE_UNUSED CULV_NODISCARD static inline bool fp_conv_aux_stride(PyObject *o, void *t) {
    AuxStride *v = (AuxStride *)t;
    return parse_quat_f32(o, &v->x, &v->y, &v->z, &v->w) != 0;
}

// Inject them into the submodule's _Generic macro
// Note the leading comma!
#define FP_CUSTOM_CONVERTERS       \
    , Vec3f: fp_conv_vec3f         \
    , PosStride: fp_conv_pos_stride \
    , AuxStride: fp_conv_aux_stride

// 4. Include the actual submodule
#include <fast_parse.h>