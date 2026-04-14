#pragma once
#include "culverin_compiler_specifics.h"
#include <Python.h>

// Forward declarations: "Trust me, these exist"
struct PhysicsWorldObject;
struct CharacterObject;
struct VehicleObject;

typedef enum : uint8_t {
    PROXY_POSITIONS,
    PROXY_ROTATIONS,
    PROXY_LINEAR_VELOCITIES,
    PROXY_ANGULAR_VELOCITIES,
    PROXY_USER_DATA,
    PROXY_DYNAMIC,
    PROXY_ECS_DATA,     // Contiguous component data
    PROXY_ECS_ENTITIES   // Dense entity handle array
} ProxyBufferType;

typedef struct {
    PyObject_HEAD 
    PyObject *owner;
    ProxyBufferType buf_type;
    const char *format;
    void *dynamic_ptr;
    size_t itemsize;
    int stride;
    // Buffer protocol metadata storage
    Py_ssize_t shape[1];
    Py_ssize_t strides[1];
} BufferProxyObject;

extern PyType_Slot BufferProxy_slots[];
extern PyType_Spec BufferProxy_spec;

PyGetSet_DeclareGetter get_is_step_pending(struct PhysicsWorldObject *self,
                                           CULV_MAYBE_UNUSED void *closure);
PyGetSet_DeclareGetter get_positions(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_rotations(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_velocities(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_angular_velocities(struct PhysicsWorldObject *self,
                                              CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_count(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_time(struct PhysicsWorldObject *self, CULV_MAYBE_UNUSED void *c);
PyGetSet_DeclareGetter get_user_data_buffer(struct PhysicsWorldObject *self,
                                            CULV_MAYBE_UNUSED void *c);

PyGetSet_DeclareGetter Character_get_handle(struct CharacterObject *self,
                                            CULV_MAYBE_UNUSED void *closure);

PyGetSet_DeclareGetter Vehicle_get_wheel_count(struct VehicleObject *self,
                                               CULV_MAYBE_UNUSED void *closure);

PyGetSet_DeclareGetter get_shape_count(struct PhysicsWorldObject *self,
                                       CULV_MAYBE_UNUSED void *closure);

PyGetSet_DeclareGetter PhysicsWorld_get_max_bodies(struct PhysicsWorldObject *self,
                                                   CULV_MAYBE_UNUSED void *closure);
PyGetSet_DeclareGetter PhysicsWorld_get_remaining_capacity(struct PhysicsWorldObject *self,
                                                           CULV_MAYBE_UNUSED void *closure);
PyCFunction_DeclareMethodFromModule PhysicsWorld_get_soft_body_vertices(struct PhysicsWorldObject *self, PyObject *const *args, size_t nargsf, PyObject *kwnames);