#pragma once

#include <Python.h>

#include "bindings/Result.hpp"

#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>

namespace CPH::Python {
namespace detail {

template <typename> inline constexpr bool kUnsupportedValue = false;

[[nodiscard]] inline PyObject *MakeName(std::string_view name) noexcept {
    if (name.size() > static_cast<std::size_t>(std::numeric_limits<Py_ssize_t>::max())) {
        return nullptr;
    }
    const char *data = name.empty() ? "" : name.data();
    return PyUnicode_FromStringAndSize(data, static_cast<Py_ssize_t>(name.size()));
}

template <typename T> [[nodiscard]] inline T ConvertValue(PyObject *obj) noexcept {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<float>(PyFloat_AsDouble(obj));
    } else if constexpr (std::is_same_v<T, double>) {
        return PyFloat_AsDouble(obj);
    } else if constexpr (std::is_same_v<T, int>) {
        return static_cast<int>(PyLong_AsLong(obj));
    } else if constexpr (std::is_same_v<T, long>) {
        return PyLong_AsLong(obj);
    } else if constexpr (std::is_same_v<T, const char *>) {
        return PyUnicode_AsUTF8(obj);
    } else {
        static_assert(kUnsupportedValue<T>, "Unsupported Python value conversion");
    }
}

// --- Registration: the compile-time gate -------------------------------------
//
// Every Python-facing implementation returns CPH::Python::BoundResult. These
// thunks adapt one implementation shape to the C function pointer CPython
// expects and wrap the call in Bridge(), so the two error channels are written
// together in exactly one place.
//
// A function that still returns `PyObject *` -- one that manages the error
// indicator by hand -- has no thunk, so registering it does not compile. That is
// the point: a method whose failure convention is implicit is not something you
// can forget to handle, it is something the compiler will not let you express.
template <typename Signature> struct BoundThunkKind {
    static constexpr bool is_valid = false;
};

// METH_FASTCALL, with or without keywords.
template <typename Self, typename ArgCount>
struct BoundThunkKind<BoundResult (*)(Self *, PyObject *const *, ArgCount)> {
    static constexpr bool is_valid = true;
    static constexpr int flags     = METH_FASTCALL;

    template <auto Impl>
    static PyObject *Call(Self *self, PyObject *const *args, ArgCount nargsf) noexcept {
        return Bridge([&]() -> BoundResult { return Impl(self, args, nargsf); });
    }
};
template <typename Self, typename ArgCount>
struct BoundThunkKind<BoundResult (*)(Self *, PyObject *const *, ArgCount, PyObject *)> {
    static constexpr bool is_valid = true;
    static constexpr int flags     = METH_FASTCALL | METH_KEYWORDS;

    template <auto Impl>
    static PyObject *Call(Self *self, PyObject *const *args, ArgCount nargsf,
                          PyObject *kwnames) noexcept {
        return Bridge([&]() -> BoundResult { return Impl(self, args, nargsf, kwnames); });
    }
};

// (self, argument) -- the shape shared by METH_O and METH_NOARGS, which differ
// only in the flag the table entry carries.
template <typename Self> struct SimpleThunkKind {
    static constexpr bool is_valid = false;
};
template <typename Self>
struct SimpleThunkKind<BoundResult (*)(Self *, PyObject *)> {
    static constexpr bool is_valid = true;

    template <auto Impl>
    static PyObject *Call(Self *self, PyObject *arg) noexcept {
        return Bridge([&]() -> BoundResult { return Impl(self, arg); });
    }
};

// tp_new: (type, args, kwds) -> a new reference to a fresh instance.
template <typename Signature> struct NewThunkKind {
    static constexpr bool is_valid = false;
};
template <>
struct NewThunkKind<BoundResult (*)(PyTypeObject *, PyObject *, PyObject *)> {
    static constexpr bool is_valid = true;

    template <auto Impl>
    static PyObject *Call(PyTypeObject *type, PyObject *args, PyObject *kwds) noexcept {
        return Bridge([&]() -> BoundResult { return Impl(type, args, kwds); });
    }
};

// Getters share the (self, closure) shape.
template <typename Self> struct GetterThunkKind {
    static constexpr bool is_valid = false;
};
template <typename Self>
struct GetterThunkKind<BoundResult (*)(Self *, void *)> {
    static constexpr bool is_valid = true;

    template <auto Impl>
    static PyObject *Call(Self *self, void *closure) noexcept {
        return Bridge([&]() -> BoundResult { return Impl(self, closure); });
    }
};

} // namespace detail

// A conversion proxy lets the destination type select the Python conversion:
// `float x = CPH::Python::ToValue(obj);`
class Value {
  public:
    explicit Value(PyObject *obj) noexcept : mObject(obj) {}

    template <typename T> [[nodiscard]] operator T() const noexcept {
        return detail::ConvertValue<T>(mObject);
    }

  private:
    PyObject *mObject;
};

[[nodiscard]] inline Value ToValue(PyObject *obj) noexcept { return Value{obj}; }

template <typename T>
[[nodiscard]] inline T GetAttribute(PyObject *obj, std::string_view name,
                                    T default_value) noexcept {
    if (obj == nullptr || obj == Py_None) {
        return default_value;
    }

    PyObject *py_name = detail::MakeName(name);
    if (py_name == nullptr) {
        PyErr_Clear();
        return default_value;
    }

    PyObject *attr = PyObject_GetAttr(obj, py_name);
    Py_DECREF(py_name);
    if (attr == nullptr) {
        PyErr_Clear();
        return default_value;
    }

    const T value = ToValue(attr);
    Py_DECREF(attr);
    if (PyErr_Occurred()) {
        PyErr_Clear();
        return default_value;
    }
    return value;
}

template <typename T>
[[nodiscard]] inline T GetDictionaryItem(PyObject *dict, std::string_view key,
                                         T default_value) noexcept {
    if (dict == nullptr || !PyDict_Check(dict)) {
        return default_value;
    }

    PyObject *py_key = detail::MakeName(key);
    if (py_key == nullptr) {
        PyErr_Clear();
        return default_value;
    }

    PyObject *item = PyDict_GetItemWithError(dict, py_key);
    Py_DECREF(py_key);
    if (item != nullptr) {
        const T value = ToValue(item);
        if (!PyErr_Occurred()) {
            return value;
        }
    }
    PyErr_Clear();
    return default_value;
}

// --- Method table entries -----------------------------------------------------
//
// The implementation is the template argument, not a function argument, so its
// exact type is visible here and can be checked. `Method` covers both fastcall
// shapes and picks METH_KEYWORDS from the signature; `NoArgsMethod` and
// `OneArgMethod` cover the two-argument shape, where only the table knows
// whether CPython will pass an argument at all.

template <auto Impl>
[[nodiscard]] constexpr PyMethodDef Method(const char *name) noexcept {
    using Thunk = detail::BoundThunkKind<decltype(Impl)>;
    if constexpr (!Thunk::is_valid) {
        static_assert(Thunk::is_valid,
                      "CPH::Python::Method<&Impl> requires an implementation returning "
                      "CPH::Python::BoundResult; see bindings/Result.hpp");
        return {};
    } else {
        return {.ml_name  = name,
                .ml_meth  = _PyCFunction_CAST(&Thunk::template Call<Impl>),
                .ml_flags = Thunk::flags,
                .ml_doc   = nullptr};
    }
}

template <auto Impl>
[[nodiscard]] constexpr PyMethodDef NoArgsMethod(const char *name) noexcept {
    using Thunk = detail::SimpleThunkKind<decltype(Impl)>;
    if constexpr (!Thunk::is_valid) {
        static_assert(Thunk::is_valid,
                      "CPH::Python::NoArgsMethod<&Impl> requires a (self, ignored) implementation "
                      "returning CPH::Python::BoundResult; see bindings/Result.hpp");
        return {};
    } else {
        return {.ml_name  = name,
                .ml_meth  = _PyCFunction_CAST(&Thunk::template Call<Impl>),
                .ml_flags = METH_NOARGS,
                .ml_doc   = nullptr};
    }
}

template <auto Impl>
[[nodiscard]] constexpr PyMethodDef OneArgMethod(const char *name) noexcept {
    using Thunk = detail::SimpleThunkKind<decltype(Impl)>;
    if constexpr (!Thunk::is_valid) {
        static_assert(Thunk::is_valid,
                      "CPH::Python::OneArgMethod<&Impl> requires a (self, arg) implementation "
                      "returning CPH::Python::BoundResult; see bindings/Result.hpp");
        return {};
    } else {
        return {.ml_name  = name,
                .ml_meth  = _PyCFunction_CAST(&Thunk::template Call<Impl>),
                .ml_flags = METH_O,
                .ml_doc   = nullptr};
    }
}

template <auto Impl>
[[nodiscard]] constexpr PyGetSetDef GetSet(const char *name) noexcept {
    using Thunk = detail::GetterThunkKind<decltype(Impl)>;
    if constexpr (!Thunk::is_valid) {
        static_assert(Thunk::is_valid,
                      "CPH::Python::GetSet<&Impl> requires a (self, closure) getter returning "
                      "CPH::Python::BoundResult; see bindings/Result.hpp");
        return {};
    } else {
        return {.name    = name,
                .get     = _Py_CAST(getter, _Py_CAST(void (*)(void), &Thunk::template Call<Impl>)),
                .set     = nullptr,
                .doc     = nullptr,
                .closure = nullptr};
    }
}

// tp_new is a slot rather than a table entry, so it gets its own builder: the
// allocation path is where "returned NULL without setting an exception" used to
// live, and it is gated the same way.
template <auto Impl>
[[nodiscard]] constexpr PyType_Slot NewSlot() noexcept {
    using Thunk = detail::NewThunkKind<decltype(Impl)>;
    if constexpr (!Thunk::is_valid) {
        static_assert(Thunk::is_valid,
                      "CPH::Python::NewSlot<&Impl> requires a (type, args, kwds) implementation "
                      "returning CPH::Python::BoundResult; see bindings/Result.hpp");
        return {};
    } else {
        return {.slot  = Py_tp_new,
                .pfunc = _Py_CAST(void *, &Thunk::template Call<Impl>)};
    }
}

extern PyType_Spec WorldSpec;
extern PyType_Spec VehicleSpec;
extern PyType_Spec RagdollSettingsSpec;
extern PyType_Spec RagdollSpec;
extern PyType_Spec SkeletonSpec;
extern PyType_Spec SoftBodySharedSettingsSpec;
extern PyType_Spec RegistrySpec;
extern PyType_Spec ShipSpec;

} // namespace CPH::Python
