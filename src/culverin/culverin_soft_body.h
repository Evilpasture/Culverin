#pragma once
#include "culverin.h"

typedef struct {
    PyObject_HEAD JPH_SoftBodySharedSettings *settings;
    uint32_t num_vertices;
    bool constraints_created;
    bool optimized;
} SoftBodySharedSettingsObject;

PyType_DeclareSlot_StatusFromModule SoftBodySharedSettings_init(SoftBodySharedSettingsObject *self,
                                                                PyObject *args, PyObject *kwds);

PyType_DeclareSlot_VoidFromModule
SoftBodySharedSettings_dealloc(SoftBodySharedSettingsObject *self);
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertex(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                  Py_ssize_t nargs, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_vertices(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                    Py_ssize_t nargs, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_face(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                Py_ssize_t nargs, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_faces(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                 Py_ssize_t nargs, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_create_constraints(SoftBodySharedSettingsObject *self, PyObject *const *args,
                                          Py_ssize_t nargs, PyObject *kwnames);
PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_optimize(SoftBodySharedSettingsObject *self, PyObject *arg);

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_add_pinned_vertex(SoftBodySharedSettingsObject *self, PyObject *arg);

PyCFunction_DeclareMethodFromModule
SoftBodySharedSettings_get_vertex_position(SoftBodySharedSettingsObject *self, PyObject *arg);

PyCFunction_DeclareMethodFromModule PhysicsWorld_create_soft_body(PhysicsWorldObject *self,
                                                                  PyObject *const *args,
                                                                  size_t nargsf, PyObject *kwnames);