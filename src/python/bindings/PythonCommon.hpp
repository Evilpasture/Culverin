#pragma once

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#endif

#include "core/FloatBits.hpp"
#include "core/Types.hpp"
#include "physics/CommandBuffer.hpp"
#include "physics/SlotState.hpp"
#include <Python.h>

#include <bit>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <type_traits>
#include <utility>

namespace CPH::Python {

[[nodiscard]] inline void *Allocate(std::size_t size) noexcept {
#ifdef ENABLE_SANITIZER
    return std::malloc(size);
#else
    return PyMem_RawMalloc(size);
#endif
}

[[nodiscard]] inline void *AllocateZeroed(std::size_t count, std::size_t size) noexcept {
#ifdef ENABLE_SANITIZER
    return std::calloc(count, size);
#else
    return PyMem_RawCalloc(count, size);
#endif
}

[[nodiscard]] inline void *Reallocate(void *memory, std::size_t size) noexcept {
#ifdef ENABLE_SANITIZER
    return std::realloc(memory, size);
#else
    return PyMem_RawRealloc(memory, size);
#endif
}

inline void Deallocate(void *memory) noexcept {
#ifdef ENABLE_SANITIZER
    std::free(memory);
#else
    PyMem_RawFree(memory);
#endif
}

class ScopedGilRelease {
  public:
    ScopedGilRelease() noexcept : mThreadState(PyEval_SaveThread()) {}

    ~ScopedGilRelease() {
        if (mThreadState != nullptr) {
            PyEval_RestoreThread(mThreadState);
        }
    }

    ScopedGilRelease(const ScopedGilRelease &)            = delete;
    ScopedGilRelease &operator=(const ScopedGilRelease &) = delete;
    ScopedGilRelease(ScopedGilRelease &&)                 = delete;
    ScopedGilRelease &operator=(ScopedGilRelease &&)      = delete;

  private:
    PyThreadState *mThreadState;
};

// --- Reference vocabulary (Phase 2, step B3) ---
//
// Two names, two meanings, no ambiguity at a call site:
//
//   Borrowed  a pointer this code does NOT own. Never decrefs. Debug builds
//             trap on dereferencing a null borrow.
//   Ref       this code owns exactly one reference and will decref it. The
//             constructors differ on purpose:
//               Ref(PyObject*)      adopts a reference the caller already
//                                   owns (a "new reference" from the C API)
//               Ref::New(Borrowed)  takes an additional reference to
//                                   something we only borrow
//
// Both are trivially cheap; the previous `OwnedPyObject` is this `Ref`.

class Borrowed {
  public:
    constexpr Borrowed() noexcept = default;
    // Implicit on purpose: every non-owning use is a borrow, and spelling
    // `Borrowed{obj}` at hundreds of call sites adds noise without adding
    // safety. What matters is that nothing here can *steal* a reference.
    constexpr Borrowed(PyObject *object) noexcept : mObject(object) {}

    [[nodiscard]] constexpr PyObject *get() const noexcept { return mObject; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return mObject != nullptr; }

    // Shortcut for `obj == Py_None` / `obj != Py_None`.
    [[nodiscard]] constexpr bool is_none() const noexcept { return mObject == Py_None; }

    [[nodiscard]] PyObject *operator->() const noexcept {
        debug_check();
        return mObject;
    }

    // Implicit on purpose: every CPython entry point takes `PyObject *`, and
    // making this explicit would bury the call sites in get() noise.
    [[nodiscard]] constexpr operator PyObject *() const noexcept { return mObject; }

  private:
    void debug_check() const noexcept {
#if defined(CULVERIN_DEBUG)
        if (mObject == nullptr) {
            Py_FatalError("CPH::Python::Borrowed: dereferenced a null borrow");
        }
#endif
    }

    PyObject *mObject = nullptr;
};

class Ref {
  public:
    Ref() noexcept = default;
    // Adopts an already-owned reference (a C API "new reference").
    explicit Ref(PyObject *object) noexcept : mObject(object) {}
    ~Ref() { Py_XDECREF(mObject); }

    Ref(const Ref &)            = delete;
    Ref &operator=(const Ref &) = delete;

    Ref(Ref &&other) noexcept : mObject(std::exchange(other.mObject, nullptr)) {}

    Ref &operator=(Ref &&other) noexcept {
        if (this != &other) {
            Py_XDECREF(mObject);
            mObject = std::exchange(other.mObject, nullptr);
        }
        return *this;
    }

    Ref &operator=(PyObject *object) noexcept {
        if (mObject != object) {
            Py_XDECREF(mObject);
            mObject = object;
        }
        return *this;
    }

    // Takes an additional reference to a pointer we only borrow.
    [[nodiscard]] static Ref New(Borrowed object) noexcept {
        Py_XINCREF(object.get());
        return Ref{object.get()};
    }

    [[nodiscard]] PyObject *get() const noexcept { return mObject; }
    [[nodiscard]] PyObject *release() noexcept { return std::exchange(mObject, nullptr); }
    [[nodiscard]] Borrowed borrow() const noexcept { return Borrowed{mObject}; }
    [[nodiscard]] explicit operator bool() const noexcept { return mObject != nullptr; }

    void reset() noexcept {
        Py_XDECREF(mObject);
        mObject = nullptr;
    }

  private:
    PyObject *mObject = nullptr;
};

#if defined(CULVERIN_DEBUG)
inline constexpr bool kDebugLogging = true;
#else
inline constexpr bool kDebugLogging = false;
#endif

inline void DebugLog(const char *format, ...) noexcept {
    if constexpr (kDebugLogging) {
        std::fputs("[Culverin] ", stderr);
        va_list args;
        va_start(args, format);
        std::vfprintf(stderr, format, args);
        va_end(args);
        std::fputc('\n', stderr);
    } else {
        (void)format;
    }
}

struct PositionVector {
    JPH::Real x;
    JPH::Real y;
    JPH::Real z;
};

struct MaterialSettings {
    float friction;
    float restitution;
};

struct CompoundPart {
    JPH::Vec3 localPosition;
    JPH::Quat localRotation;
    float params[4];
    int type;
};

struct MeshBounds {
    std::uint32_t triangleCount;
    std::uint32_t vertexCount;
};

// IsFinite lives in core/FloatBits.hpp: the bit-level test is shared with the
// native sanitizers, and CPH::Python resolves it through the enclosing CPH
// namespace.
template <typename... Values>
[[nodiscard]] inline bool ValidateFinite(const char *name, Values... values) noexcept {
    if ((IsFinite(values) && ...)) {
        return true;
    }
    PyErr_Format(PyExc_ValueError, "Numerical Error: '%s' must be finite", name);
    return false;
}

} // namespace CPH::Python
