#pragma once

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#endif

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
    ScopedGilRelease() noexcept : m_thread_state(PyEval_SaveThread()) {}

    ~ScopedGilRelease() {
        if (m_thread_state != nullptr) {
            PyEval_RestoreThread(m_thread_state);
        }
    }

    ScopedGilRelease(const ScopedGilRelease &)            = delete;
    ScopedGilRelease &operator=(const ScopedGilRelease &) = delete;
    ScopedGilRelease(ScopedGilRelease &&)                 = delete;
    ScopedGilRelease &operator=(ScopedGilRelease &&)      = delete;

  private:
    PyThreadState *m_thread_state;
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
    constexpr Borrowed(PyObject *object) noexcept : m_object(object) {}

    [[nodiscard]] constexpr PyObject *get() const noexcept { return m_object; }
    [[nodiscard]] constexpr explicit operator bool() const noexcept { return m_object != nullptr; }

    // Shortcut for `obj == Py_None` / `obj != Py_None`.
    [[nodiscard]] constexpr bool is_none() const noexcept { return m_object == Py_None; }

    [[nodiscard]] PyObject *operator->() const noexcept {
        debug_check();
        return m_object;
    }

    // Implicit on purpose: every CPython entry point takes `PyObject *`, and
    // making this explicit would bury the call sites in get() noise.
    [[nodiscard]] constexpr operator PyObject *() const noexcept { return m_object; }

  private:
    void debug_check() const noexcept {
#if defined(CULVERIN_DEBUG)
        if (m_object == nullptr) {
            Py_FatalError("CPH::Python::Borrowed: dereferenced a null borrow");
        }
#endif
    }

    PyObject *m_object = nullptr;
};

class Ref {
  public:
    Ref() noexcept = default;
    // Adopts an already-owned reference (a C API "new reference").
    explicit Ref(PyObject *object) noexcept : m_object(object) {}
    ~Ref() { Py_XDECREF(m_object); }

    Ref(const Ref &)            = delete;
    Ref &operator=(const Ref &) = delete;

    Ref(Ref &&other) noexcept : m_object(std::exchange(other.m_object, nullptr)) {}

    Ref &operator=(Ref &&other) noexcept {
        if (this != &other) {
            Py_XDECREF(m_object);
            m_object = std::exchange(other.m_object, nullptr);
        }
        return *this;
    }

    Ref &operator=(PyObject *object) noexcept {
        if (m_object != object) {
            Py_XDECREF(m_object);
            m_object = object;
        }
        return *this;
    }

    // Takes an additional reference to a pointer we only borrow.
    [[nodiscard]] static Ref New(Borrowed object) noexcept {
        Py_XINCREF(object.get());
        return Ref{object.get()};
    }

    [[nodiscard]] PyObject *get() const noexcept { return m_object; }
    [[nodiscard]] PyObject *release() noexcept { return std::exchange(m_object, nullptr); }
    [[nodiscard]] Borrowed borrow() const noexcept { return Borrowed{m_object}; }
    [[nodiscard]] explicit operator bool() const noexcept { return m_object != nullptr; }

    void reset() noexcept {
        Py_XDECREF(m_object);
        m_object = nullptr;
    }

  private:
    PyObject *m_object = nullptr;
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
    JPH::Vec3 local_position;
    JPH::Quat local_rotation;
    float params[4];
    int type;
};

struct MeshBounds {
    std::uint32_t triangle_count;
    std::uint32_t vertex_count;
};

[[nodiscard]] inline bool IsFinite(float value) noexcept {
    constexpr std::uint32_t exponent_mask = 0x7F800000U;
    // The build enables -ffast-math; volatile integer bits keep LTO from
    // assuming the input is finite and folding this check away.
    volatile std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & exponent_mask) != exponent_mask;
}

[[nodiscard]] inline bool IsFinite(double value) noexcept {
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    volatile std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return (bits & exponent_mask) != exponent_mask;
}

template <typename T>
[[nodiscard]] inline bool IsFinite(T value) noexcept {
    if constexpr (std::is_same_v<std::remove_cv_t<T>, float>) {
        return IsFinite(static_cast<float>(value));
    } else if constexpr (std::is_same_v<std::remove_cv_t<T>, double>) {
        return IsFinite(static_cast<double>(value));
    } else {
        static_assert(std::is_arithmetic_v<T>, "finite checks require an arithmetic type");
        return true;
    }
}

template <typename... Values>
[[nodiscard]] inline bool ValidateFinite(const char *name, Values... values) noexcept {
    if ((IsFinite(values) && ...)) {
        return true;
    }
    PyErr_Format(PyExc_ValueError, "Numerical Error: '%s' must be finite", name);
    return false;
}

} // namespace CPH::Python
