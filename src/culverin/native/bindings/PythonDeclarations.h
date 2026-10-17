#pragma once

#include "core/CompilerSpecifics.h"

// CPython declaration helpers. Kept separate from compiler/native headers so
// the physics core can include compiler specifics without Python vocabulary.
#define PyCFunction_DeclareMethod CULV_NODISCARD static PyObject *
#define PyCFunction_DeclareMethodFromModule CULV_NODISCARD extern PyObject *
#define PyGetSet_DeclareGetter CULV_NODISCARD extern PyObject *
#define PyGetSet_DeclareSetter CULV_NODISCARD extern PyObject *

#define PyType_DeclareSlot_Object CULV_NODISCARD static PyObject *
#define PyType_DeclareSlot_ObjectFromModule CULV_NODISCARD extern PyObject *
#define PyType_DeclareSlot_Status CULV_NODISCARD static int
#define PyType_DeclareSlot_StatusFromModule CULV_NODISCARD extern int
#define PyType_DeclareSlot_Void static void
#define PyType_DeclareSlot_VoidFromModule extern void
