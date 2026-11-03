#pragma once
#include "bindings/BufferProxy.hpp"
#include "core/Compiler.hpp"
#include <cstddef>
#include <new>
#include "bindings/PyGcObject.hpp"
#include <Python.h>

namespace CPH::Python {

// Forward declarations: "Trust me, these exist"
struct PhysicsWorldObject;
struct CharacterObject;
struct VehicleObject;

// The Python adapter for CPH::Python::BufferProxy.
//
// Everything domain-shaped -- which array, how it is locked, when its address
// may move -- lives in BufferProxy.hpp and is invisible here. What is left is
// the part CPython insists on: the object header, one owned reference to the
// exporter, and the shape/strides arrays a Py_buffer is allowed to point at.
//
// The C++ object is placement-new'd into `proxyStorage` because PyObject_GC_New
// only hands back zeroed, unconstructed memory; it is destroyed by hand in
// dealloc. Keeping it inline rather than behind a pointer preserves the single
// allocation the old design had -- these objects are created per attribute
// access, so an extra malloc per `world.positions` would be measurable.
using BufferProxyPayload = CPH::Python::BufferProxy::Payload;

struct BufferProxyObject : PyGcObject<BufferProxyObject> {
    PyObject *owner = nullptr;
    alignas(CPH::Python::BufferProxy) std::byte proxyStorage[sizeof(CPH::Python::BufferProxy)] = {};
    // Buffer protocol metadata storage; the exported Py_buffer points here.
    Py_ssize_t shape[1]   = {0};
    Py_ssize_t strides[1] = {0};

    // The export is returned to whichever domain issued it while the owner is
    // still alive. This is why the hook runs before references are cleared.
    void BeforeClear() noexcept {
        if (owner != nullptr) {
            Proxy().ReleaseExport();
        }
    }

    template <typename Visitor>
    void VisitRefs(Visitor &&visit) noexcept {
        visit(owner);
    }

    // The storage comes from PyObject_GC_New and never held a C++ object until
    // ProxyStorageInit() placement-news one, so the destructor has to be run by
    // hand at dealloc and nowhere else.
    void BeforeFree() noexcept {
        Proxy().~BufferProxy();
    }

    // BufferProxy is constructed from its Payload, not from another proxy.
    void ProxyStorageInit(CPH::Python::BufferProxy::Payload payload) noexcept {
        new (proxyStorage) CPH::Python::BufferProxy(std::move(payload));
    }

    [[nodiscard]] CPH::Python::BufferProxy &Proxy() noexcept {
        return *std::launder(reinterpret_cast<CPH::Python::BufferProxy *>(proxyStorage));
    }
    [[nodiscard]] const CPH::Python::BufferProxy &Proxy() const noexcept {
        return *std::launder(reinterpret_cast<const CPH::Python::BufferProxy *>(proxyStorage));
    }
};

/// Allocates the Python object around an already-built payload, takes the
/// owner reference, and records the export exactly once. Both domains create
/// their proxies through here.
[[nodiscard]] PyObject *NewBufferProxy(PyObject *owner, BufferProxyPayload payload);

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
