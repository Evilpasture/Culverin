#pragma once
#include "core/CompilerSpecifics.h"
#include <Python.h>
#include "bindings/PythonDeclarations.h"

// Forward declarations: "Trust me, these exist"
struct CPH_PhysicsWorldObject;
struct CPH_CharacterObject;
struct CPH_VehicleObject;

typedef enum : uint8_t {
    PROXY_POSITIONS,
    PROXY_ROTATIONS,
    PROXY_LINEAR_VELOCITIES,
    PROXY_ANGULAR_VELOCITIES,
    PROXY_USER_DATA,
    PROXY_DYNAMIC,
    PROXY_ECS_DATA,    // Contiguous component data
    PROXY_ECS_ENTITIES // Dense entity handle array
} ProxyBufferType;

typedef struct {
    PyObject_HEAD PyObject *owner;
    ProxyBufferType buf_type;
    const char *format;
    void *dynamic_ptr;
    size_t itemsize;
    int stride;
    // Buffer protocol metadata storage
    Py_ssize_t shape[1];
    Py_ssize_t strides[1];
} CPH_BufferProxyObject;

extern PyType_Slot BufferProxy_slots[];
extern PyType_Spec BufferProxy_spec;

PyGetSet_DeclareGetter CPH_GetIsStepPending(struct CPH_PhysicsWorldObject *self,
                                           [[maybe_unused]] void *closure);
PyGetSet_DeclareGetter CPH_GetPositions(struct CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c);
PyGetSet_DeclareGetter CPH_GetRotations(struct CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c);
PyGetSet_DeclareGetter CPH_GetVelocities(struct CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c);
PyGetSet_DeclareGetter CPH_GetAngularVelocities(struct CPH_PhysicsWorldObject *self,
                                              [[maybe_unused]] void *c);
PyGetSet_DeclareGetter CPH_GetCount(struct CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c);
PyGetSet_DeclareGetter CPH_GetTime(struct CPH_PhysicsWorldObject *self, [[maybe_unused]] void *c);
PyGetSet_DeclareGetter CPH_GetUserDataBuffer(struct CPH_PhysicsWorldObject *self,
                                            [[maybe_unused]] void *c);

PyGetSet_DeclareGetter CPH_PyVehicle_GetWheelCount(struct CPH_VehicleObject *self,
                                               [[maybe_unused]] void *closure);

PyGetSet_DeclareGetter CPH_GetShapeCount(struct CPH_PhysicsWorldObject *self,
                                       [[maybe_unused]] void *closure);

PyGetSet_DeclareGetter CPH_PyWorld_GetMaxBodies(struct CPH_PhysicsWorldObject *self,
                                                   [[maybe_unused]] void *closure);
PyGetSet_DeclareGetter CPH_PyWorld_GetRemainingCapacity(struct CPH_PhysicsWorldObject *self,
                                                           [[maybe_unused]] void *closure);
PyCFunction_DeclareMethodFromModule
CPH_PyWorld_GetSoftBodyVertices(struct CPH_PhysicsWorldObject *self, PyObject *const *args,
                                    size_t nargsf, PyObject *kwnames);