#pragma once
#include "core/CompilerSpecifics.h"
#include <Python.h>

namespace CPH::Python {

// Forward declarations: "Trust me, these exist"
struct PhysicsWorldObject;
struct CharacterObject;
struct VehicleObject;

enum ProxyBufferType : uint8_t {
    PROXY_POSITIONS,
    PROXY_ROTATIONS,
    PROXY_LINEAR_VELOCITIES,
    PROXY_ANGULAR_VELOCITIES,
    PROXY_USER_DATA,
    PROXY_DYNAMIC,
    PROXY_ECS_DATA,    // Contiguous component data
    PROXY_ECS_ENTITIES // Dense entity handle array
};

struct BufferProxyObject {
    PyObject_HEAD PyObject *owner;
    ProxyBufferType buf_type;
    const char *format;
    void *dynamic_ptr;
    size_t itemsize;
    int stride;
    // Buffer protocol metadata storage
    Py_ssize_t shape[1];
    Py_ssize_t strides[1];
};

extern PyType_Slot BufferProxySlots[];
extern PyType_Spec BufferProxySpec;

[[nodiscard]] PyObject * WorldGetIsStepPending(struct PhysicsWorldObject *self,
                                                    [[maybe_unused]] void *closure);
[[nodiscard]] PyObject * WorldGetPositions(struct PhysicsWorldObject *self,
                                                [[maybe_unused]] void *c);
[[nodiscard]] PyObject * WorldGetRotations(struct PhysicsWorldObject *self,
                                                [[maybe_unused]] void *c);
[[nodiscard]] PyObject * WorldGetVelocities(struct PhysicsWorldObject *self,
                                                 [[maybe_unused]] void *c);
[[nodiscard]] PyObject * WorldGetAngularVelocities(struct PhysicsWorldObject *self,
                                                        [[maybe_unused]] void *c);
[[nodiscard]] PyObject * WorldGetCount(struct PhysicsWorldObject *self,
                                            [[maybe_unused]] void *c);
[[nodiscard]] PyObject * WorldGetTime(struct PhysicsWorldObject *self,
                                           [[maybe_unused]] void *c);
[[nodiscard]] PyObject * WorldGetUserDataBuffer(struct PhysicsWorldObject *self,
                                                     [[maybe_unused]] void *c);

[[nodiscard]] PyObject * VehicleGetWheelCount(struct VehicleObject *self,
                                                   [[maybe_unused]] void *closure);

[[nodiscard]] PyObject * WorldGetShapeCount(struct PhysicsWorldObject *self,
                                                 [[maybe_unused]] void *closure);

[[nodiscard]] PyObject * WorldGetMaxBodies(struct PhysicsWorldObject *self,
                                                [[maybe_unused]] void *closure);
[[nodiscard]] PyObject * WorldGetRemainingCapacity(struct PhysicsWorldObject *self,
                                                        [[maybe_unused]] void *closure);
[[nodiscard]] PyObject *
WorldGetSoftBodyVertices(struct PhysicsWorldObject *self, PyObject *const *args,
                                size_t nargsf, PyObject *kwnames);

} // namespace CPH::Python
