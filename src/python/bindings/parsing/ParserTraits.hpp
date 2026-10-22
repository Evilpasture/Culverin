#pragma once

// What one Python argument type *is*, as far as the argument parser cares.
//
// There is one `ParserTrait` specialisation per accepted C++ type and it
// carries exactly three things:
//
//   name       the type name diagnostics print ("float", "Quaternion", ...)
//   kTypeGuard optional exact-type-or-subclass check that runs before conversion
//   convert    the conversion itself, as a `constexpr` lambda
//
// `convert` writes through `T &`, not through `const void *`. That is the
// point of the redesign: the previous table stored `bool (*)(PyObject *,
// const void *)` and the engine reached every argument through that pointer,
// so each one cost an indirect call plus a round trip through a stack array of
// erased target pointers. Here the fold in Parser.hpp knows the concrete type
// at compile time and calls the lambda directly, which lets the conversion
// inline into its caller.
//
// The converter contract itself is unchanged: return `false` and leave the
// reason in the thread state. The engine either decorates that reason with the
// argument signature or hands it back verbatim -- see Parser.hpp.

#include "bindings/PythonCommon.hpp"
#include "bindings/parsing/ParseError.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Types.hpp"

#include <cstddef>
#include <cstdint>

namespace CPH::Python {

// Deliberately left undefined: an argument type without a trait must fail to
// compile rather than silently convert through some fallback.
template <typename T> struct ParserTrait;

template <> struct ParserTrait<float> {
    static constexpr const char *name             = "float";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, float &target) noexcept -> bool {
        // HOT PATH: exact float, no subclass or __float__ dance
        if (PyFloat_CheckExact(obj)) {
            target = static_cast<float>(reinterpret_cast<PyFloatObject *>(obj)->ob_fval);
            return true;
        }

        const double value = PyFloat_AsDouble(obj);
        if (value == -1.0 && PyErr_Occurred()) [[unlikely]] {
            return false;
        }
        target = static_cast<float>(value);
        return true;
    };
};

template <> struct ParserTrait<double> {
    static constexpr const char *name             = "float";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, double &target) noexcept -> bool {
        if (PyFloat_CheckExact(obj)) {
            target = reinterpret_cast<PyFloatObject *>(obj)->ob_fval;
            return true;
        }

        if (obj == Py_None) [[unlikely]] {
            PyErr_SetString(PyExc_TypeError, "double argument cannot be None");
            return false;
        }

        const double value = PyFloat_AsDouble(obj);
        if (value == -1.0 && PyErr_Occurred()) [[unlikely]] {
            return false;
        }
        target = value;
        return true;
    };
};

template <> struct ParserTrait<int> {
    static constexpr const char *name             = "int";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, int &target) noexcept -> bool {
        if (obj == Py_True) {
            target = 1;
            return true;
        }
        if (obj == Py_False) {
            target = 0;
            return true;
        }
        if (obj == Py_None) [[unlikely]] {
            PyErr_SetString(PyExc_TypeError, "int argument cannot be None");
            return false;
        }

        const long value = PyLong_AsLong(obj);
        if (value != -1) [[likely]] {
            target = static_cast<int>(value);
            return true;
        }
        if (PyErr_Occurred()) {
            return false;
        }
        target = -1;
        return true;
    };
};

template <> struct ParserTrait<uint32_t> {
    static constexpr const char *name             = "int";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, uint32_t &target) noexcept -> bool {
        if (obj == Py_True) {
            target = 1;
            return true;
        }
        if (obj == Py_False) {
            target = 0;
            return true;
        }
        if (obj == Py_None) [[unlikely]] {
            PyErr_SetString(PyExc_TypeError, "uint32 argument cannot be None");
            return false;
        }

        const unsigned long value = PyLong_AsUnsignedLong(obj);
        if (value != static_cast<unsigned long>(-1)) [[likely]] {
            target = static_cast<uint32_t>(value);
            return true;
        }
        if (PyErr_Occurred()) {
            return false;
        }
        target = static_cast<uint32_t>(-1);
        return true;
    };
};

template <> struct ParserTrait<uint64_t> {
    static constexpr const char *name             = "int";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, uint64_t &target) noexcept -> bool {
        if (obj == Py_True) {
            target = 1;
            return true;
        }
        if (obj == Py_False) {
            target = 0;
            return true;
        }
        // Handles are the common u64 argument and are routinely None.
        if (obj == Py_None) {
            target = 0;
            return true;
        }

        const unsigned long long value = PyLong_AsUnsignedLongLong(obj);
        if (value != static_cast<unsigned long long>(-1)) [[likely]] {
            target = static_cast<uint64_t>(value);
            return true;
        }
        if (PyErr_Occurred()) {
            return false;
        }
        target = static_cast<uint64_t>(-1);
        return true;
    };
};

template <> struct ParserTrait<Py_ssize_t> {
    static constexpr const char *name             = "int";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, Py_ssize_t &target) noexcept -> bool {
        if (PyLong_CheckExact(obj)) [[likely]] {
            const Py_ssize_t value = PyLong_AsSsize_t(obj);
            if (value != -1) [[likely]] {
                target = value;
                return true;
            }
        }

        if (obj == Py_None) {
            PyErr_SetString(PyExc_TypeError, "size/index argument cannot be None");
            return false;
        }

        const Py_ssize_t value = PyLong_AsSsize_t(obj);
        if (value == -1 && PyErr_Occurred()) {
            return false;
        }
        target = value;
        return true;
    };
};

template <> struct ParserTrait<bool> {
    static constexpr const char *name  = "bool";
    static constexpr auto convert      = [](PyObject *obj, bool &target) noexcept -> bool {
        if (obj == Py_True) {
            target = true;
            return true;
        }
        if (obj == Py_False) {
            target = false;
            return true;
        }

        const int value = PyObject_IsTrue(obj);
        if (value == -1) [[unlikely]] {
            return false;
        }
        target = value != 0;
        return true;
    };

    // Not constexpr, and it cannot be: on the MSVC ABI the CPython headers
    // declare PyBool_Type __declspec(dllimport), so its address resolves
    // through the import table and is not a constant expression to clang
    // ("constexpr variable 'type_guard' must be initialized by a constant
    // expression"). Every other trait's guard is a null pointer, which is.
    // The value is only ever read at call time, so an inline variable
    // initialised at load is the same thing at runtime.
    static inline PyTypeObject *const kTypeGuard = &PyBool_Type;
};

template <> struct ParserTrait<const char *> {
    static constexpr const char *name             = "str";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, const char *&target) noexcept -> bool {
        if (!PyUnicode_Check(obj)) [[unlikely]] {
            PyErr_Format(PyExc_TypeError, "expected str, got %s", Py_TYPE(obj)->tp_name);
            return false;
        }
        const char *text = PyUnicode_AsUTF8(obj);
        if (text == nullptr) [[unlikely]] {
            return false;
        }
        target = text;
        return true;
    };
};

template <> struct ParserTrait<PyObject *> {
    static constexpr const char *name             = "object";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    // Borrowed, by design: the caller's local holds the reference for the
    // duration of the call and nothing outlives the vectorcall frame.
    static constexpr auto convert = [](PyObject *obj, PyObject *&target) noexcept -> bool {
        target = obj;
        return true;
    };
};

namespace detail {

// The sequence parsers hand back a captured ParseError instead of a plain
// false + pending exception. Both are the same thing to the engine -- it only
// ever looks at the thread state -- so the adapter re-raises the captured
// instance and reports failure in the classic form.
template <typename T, typename ParseInto>
[[nodiscard]] static inline bool ConvertSequence(PyObject *obj, T &target, ParseInto parse_into) {
    if (ParseResult parsed = parse_into(Borrowed{obj}, target); !parsed) {
        RaiseParseError(parsed.error());
        return false;
    }
    return true;
}

} // namespace detail

template <> struct ParserTrait<JPH::Float3> {
    static constexpr const char *name             = "Vector3";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, JPH::Float3 &target) noexcept -> bool {
        return detail::ConvertSequence(
            obj, target, [](Borrowed o, JPH::Float3 &v) { return ParseVec3Into(o, v); });
    };
};

template <> struct ParserTrait<PosStride> {
    static constexpr const char *name             = "Vector3";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, PosStride &target) noexcept -> bool {
        return detail::ConvertSequence(
            obj, target, [](Borrowed o, PosStride &v) { return ParseVec3Into(o, v); });
    };
};

template <> struct ParserTrait<AuxStride> {
    static constexpr const char *name             = "Quaternion";
    static constexpr PyTypeObject *const kTypeGuard = nullptr;

    static constexpr auto convert = [](PyObject *obj, AuxStride &target) noexcept -> bool {
        return detail::ConvertSequence(
            obj, target, [](Borrowed o, AuxStride &v) { return ParseQuatInto(o, v); });
    };
};

// Handles cross the boundary as JPH::uint64 through `.Raw()`; the parser spells
// that as an ordinary u64 argument.
using ParseBodyHandle = uint64_t;

} // namespace CPH::Python
