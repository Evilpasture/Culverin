#pragma once

#include "bindings/parsing/FastParserCore.h"
#include "bindings/parsing/Parsers.h"
#include "core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace CPH {

template <typename T>
struct FastParseTrait;

template <>
struct FastParseTrait<float> {
    static constexpr const char *type_name = "float";
    static constexpr auto converter = ::culv_fast_parse_conv_float;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<double> {
    static constexpr const char *type_name = "float";
    static constexpr auto converter = ::culv_fast_parse_conv_double;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<int> {
    static constexpr const char *type_name = "int";
    static constexpr auto converter = ::culv_fast_parse_conv_int;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<uint32_t> {
    static constexpr const char *type_name = "int";
    static constexpr auto converter = ::culv_fast_parse_conv_u32;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<uint64_t> {
    static constexpr const char *type_name = "int";
    static constexpr auto converter = ::culv_fast_parse_conv_u64;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<Py_ssize_t> {
    static constexpr const char *type_name = "int";
    static constexpr auto converter = ::culv_fast_parse_conv_ssize;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<bool> {
    static constexpr const char *type_name = "bool";
    static constexpr auto converter = ::culv_fast_parse_conv_bool;
    static constexpr PyTypeObject *type_guard = &PyBool_Type;
};

template <>
struct FastParseTrait<const char *> {
    static constexpr const char *type_name = "str";
    static constexpr auto converter = ::culv_fast_parse_conv_str;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<PyObject *> {
    static constexpr const char *type_name = "object";
    static constexpr auto converter = ::culv_fast_parse_conv_pyobj;
    static constexpr PyTypeObject *type_guard = nullptr;
};

[[nodiscard]] static inline bool fast_parse_conv_vec3f(PyObject *obj, const void *target) {
    auto *value = const_cast<CPH_Vec3f *>(static_cast<const CPH_Vec3f *>(target));
    return CPH_ParseVec3F32(obj, &value->x, &value->y, &value->z) != 0;
}

[[nodiscard]] static inline bool fast_parse_conv_pos_stride(PyObject *obj, const void *target) {
    auto *value = const_cast<CPH_PosStride *>(static_cast<const CPH_PosStride *>(target));
    return CPH_ParseVec3R64(obj, &value->x, &value->y, &value->z) != 0;
}

[[nodiscard]] static inline bool fast_parse_conv_aux_stride(PyObject *obj, const void *target) {
    auto *value = const_cast<CPH_AuxStride *>(static_cast<const CPH_AuxStride *>(target));
    return CPH_ParseQuatF32(obj, &value->x, &value->y, &value->z, &value->w) != 0;
}

template <>
struct FastParseTrait<CPH_Vec3f> {
    static constexpr const char *type_name = "Vector3";
    static constexpr auto converter = fast_parse_conv_vec3f;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<CPH_PosStride> {
    static constexpr const char *type_name = "Vector3";
    static constexpr auto converter = fast_parse_conv_pos_stride;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <>
struct FastParseTrait<CPH_AuxStride> {
    static constexpr const char *type_name = "Quaternion";
    static constexpr auto converter = fast_parse_conv_aux_stride;
    static constexpr PyTypeObject *type_guard = nullptr;
};

// Helper to define argument names and requirement status.
template <typename T>
struct FastArg {
    const char *name;
    bool required;

    constexpr FastArg(const char *arg_name, bool is_required = false) noexcept
        : name(arg_name), required(is_required) {}
};

template <typename... Ts>
class FastParser {
    CPH_FastParser core_{};
    bool initialized_{};

public:
    FastParser() = default;
    FastParser(const FastParser &) = delete;
    FastParser &operator=(const FastParser &) = delete;
    FastParser(FastParser &&) = delete;
    FastParser &operator=(FastParser &&) = delete;

    ~FastParser() noexcept {
        if (initialized_) {
            CPH_FastParseDeinit(&core_);
        }
    }

    void Init(const char *parser_name, FastArg<Ts>... args) {
        if (initialized_) {
            CPH_FastParseDeinit(&core_);
            core_ = CPH_FastParser{};
            initialized_ = false;
        }

        if constexpr (sizeof...(Ts) > 0) {
            const std::array<CPH_FastArgDef, sizeof...(Ts)> c_defs{{
                CPH_FastArgDef{
                    .name = args.name,
                    .type_name = FastParseTrait<Ts>::type_name,
                    .type_guard = FastParseTrait<Ts>::type_guard,
                    .convert = FastParseTrait<Ts>::converter,
                    .required = args.required,
                }...,
            }};
            CPH_FastParseInitImpl(&core_, c_defs.data(), c_defs.size());
        } else {
            CPH_FastParseInitImpl(&core_, nullptr, 0);
        }
        core_.parser_name = parser_name;
        initialized_ = true;
    }

    [[nodiscard, gnu::always_inline]] bool Parse(PyObject *const *args, Py_ssize_t nargs,
                                                PyObject *kwnames, Ts &...targets) const {
        if constexpr (sizeof...(Ts) == 0) {
            return core_.hot_path(args, nargs, kwnames, &core_, nullptr);
        } else {
            const void *const target_ptrs[] = {
                reinterpret_cast<const void *>(&targets)...,
            };
            return core_.hot_path(args, nargs, kwnames, &core_, target_ptrs);
        }
    }

    [[nodiscard, gnu::always_inline]] bool Parse(PyObject *args, PyObject *kwargs,
                                                Ts &...targets) const {
        if constexpr (sizeof...(Ts) == 0) {
            return CPH_FastParseLegacy(args, kwargs, nullptr, &core_, nullptr);
        } else {
            const void *const target_ptrs[] = {
                reinterpret_cast<const void *>(&targets)...,
            };
            return CPH_FastParseLegacy(args, kwargs, nullptr, &core_, target_ptrs);
        }
    }

    [[nodiscard]] CPH_FastParser *Core() noexcept { return &core_; }
    [[nodiscard]] const CPH_FastParser *Core() const noexcept { return &core_; }
};

} // namespace CPH
