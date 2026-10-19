#pragma once

#include <Python.h>

#include <cstddef>
#include <limits>
#include <string_view>
#include <type_traits>

namespace CPH::Python {
namespace detail {

template <typename> inline constexpr bool kUnsupportedValue = false;

template <typename Function> struct FastCallMethodTraits {
    static constexpr bool is_valid = false;
    static constexpr int flags     = 0;
};
template <typename Self, typename ArgCount>
struct FastCallMethodTraits<PyObject *(*)(Self *, PyObject *const *, ArgCount)> {
    static constexpr bool is_valid = std::is_integral_v<ArgCount>;
    static constexpr int flags     = METH_FASTCALL;
};
template <typename Self, typename ArgCount>
struct FastCallMethodTraits<PyObject *(*)(Self *, PyObject *const *, ArgCount, PyObject *)> {
    static constexpr bool is_valid = std::is_integral_v<ArgCount>;
    static constexpr int flags     = METH_FASTCALL | METH_KEYWORDS;
};

template <typename Function> struct IsSimpleMethodFunction : std::false_type {};
template <typename Self>
struct IsSimpleMethodFunction<PyObject *(*)(Self *, PyObject *)> : std::true_type {};

template <typename Function> struct IsGetterFunction : std::false_type {};
template <typename Self> struct IsGetterFunction<PyObject *(*)(Self *, void *)> : std::true_type {};

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

} // namespace detail

// A conversion proxy lets the destination type select the Python conversion:
// `float x = CPH::Python::ToValue(obj);`
class Value {
  public:
    explicit Value(PyObject *obj) noexcept : m_object(obj) {}

    template <typename T> [[nodiscard]] operator T() const noexcept {
        return detail::ConvertValue<T>(m_object);
    }

  private:
    PyObject *m_object;
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

template <typename Function>
[[nodiscard]] constexpr PyMethodDef FastCallMethod(const char *name, Function function) noexcept {
    using Traits = detail::FastCallMethodTraits<Function>;
    static_assert(Traits::is_valid,
                  "FastCallMethod requires (self, args, nargsf[, kwnames]) handler");
    return {.ml_name  = name,
            .ml_meth  = _PyCFunction_CAST(function),
            .ml_flags = Traits::flags,
            .ml_doc   = nullptr};
}

template <typename Function>
[[nodiscard]] constexpr PyMethodDef NoArgsMethod(const char *name, Function function) noexcept {
    static_assert(detail::IsSimpleMethodFunction<Function>::value,
                  "NoArgsMethod requires a (self, ignored) handler");
    return {.ml_name  = name,
            .ml_meth  = _PyCFunction_CAST(function),
            .ml_flags = METH_NOARGS,
            .ml_doc   = nullptr};
}

template <typename Function>
[[nodiscard]] constexpr PyMethodDef OneArgMethod(const char *name, Function function) noexcept {
    static_assert(detail::IsSimpleMethodFunction<Function>::value,
                  "OneArgMethod requires a (self, arg) handler");
    return {.ml_name  = name,
            .ml_meth  = _PyCFunction_CAST(function),
            .ml_flags = METH_O,
            .ml_doc   = nullptr};
}

template <typename Function>
[[nodiscard]] constexpr PyGetSetDef GetSet(const char *name, Function function) noexcept {
    static_assert(detail::IsGetterFunction<Function>::value,
                  "GetSet requires a (self, closure) getter");
    return {.name    = name,
            .get     = _Py_CAST(getter, _Py_CAST(void (*)(void), function)),
            .set     = nullptr,
            .doc     = nullptr,
            .closure = nullptr};
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
