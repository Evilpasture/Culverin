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

[[nodiscard]] inline bool Likely(bool condition) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_expect(condition, true);
#else
    return condition;
#endif
}

[[nodiscard]] inline bool Unlikely(bool condition) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_expect(condition, false);
#else
    return condition;
#endif
}

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

class OwnedPyObject {
  public:
    OwnedPyObject() noexcept = default;
    explicit OwnedPyObject(PyObject *object) noexcept : m_object(object) {}

    ~OwnedPyObject() { Py_XDECREF(m_object); }

    OwnedPyObject(const OwnedPyObject &)            = delete;
    OwnedPyObject &operator=(const OwnedPyObject &) = delete;

    OwnedPyObject(OwnedPyObject &&other) noexcept : m_object(std::exchange(other.m_object, nullptr)) {}

    OwnedPyObject &operator=(OwnedPyObject &&other) noexcept {
        if (this != &other) {
            Py_XDECREF(m_object);
            m_object = std::exchange(other.m_object, nullptr);
        }
        return *this;
    }

    OwnedPyObject &operator=(PyObject *object) noexcept {
        if (m_object != object) {
            Py_XDECREF(m_object);
            m_object = object;
        }
        return *this;
    }

    [[nodiscard]] PyObject *get() const noexcept { return m_object; }
    [[nodiscard]] PyObject *release() noexcept { return std::exchange(m_object, nullptr); }
    [[nodiscard]] explicit operator bool() const noexcept { return m_object != nullptr; }

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
    JPH_Real x;
    JPH_Real y;
    JPH_Real z;
};

struct MaterialSettings {
    float friction;
    float restitution;
};

struct CompoundPart {
    JPH_Vec3 local_position;
    JPH_Quat local_rotation;
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
