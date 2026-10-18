#pragma once

#include "core/CompilerSpecifics.h"

// CPython declaration helpers. Kept separate from compiler/native headers so
// the physics core can include compiler specifics without Python vocabulary.
#define PyCFunction_DeclareMethod [[nodiscard]] static PyObject *
#define PyCFunction_DeclareMethodFromModule [[nodiscard]] extern PyObject *
#define PyGetSet_DeclareGetter [[nodiscard]] extern PyObject *
#define PyGetSet_DeclareSetter [[nodiscard]] extern PyObject *

#define PyType_DeclareSlot_Object [[nodiscard]] static PyObject *
#define PyType_DeclareSlot_ObjectFromModule [[nodiscard]] extern PyObject *
#define PyType_DeclareSlot_Status [[nodiscard]] static int
#define PyType_DeclareSlot_StatusFromModule [[nodiscard]] extern int
#define PyType_DeclareSlot_Void static void
#define PyType_DeclareSlot_VoidFromModule extern void
