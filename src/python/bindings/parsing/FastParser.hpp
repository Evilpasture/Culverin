#pragma once

#include "bindings/parsing/FastParserCore.h"
#include "bindings/parsing/Parsers.h"
#include "core/Types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

namespace CPH::Python {

template <typename T> struct FastParseTrait;

template <> struct FastParseTrait<float> {
    static constexpr const char *type_name    = "float";
    static constexpr auto converter           = ::culv_fast_parse_conv_float;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<double> {
    static constexpr const char *type_name    = "float";
    static constexpr auto converter           = ::culv_fast_parse_conv_double;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<int> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = ::culv_fast_parse_conv_int;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<uint32_t> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = ::culv_fast_parse_conv_u32;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<uint64_t> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = ::culv_fast_parse_conv_u64;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<Py_ssize_t> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = ::culv_fast_parse_conv_ssize;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<bool> {
    static constexpr const char *type_name    = "bool";
    static constexpr auto converter           = ::culv_fast_parse_conv_bool;
    static constexpr PyTypeObject *type_guard = &PyBool_Type;
};

template <> struct FastParseTrait<const char *> {
    static constexpr const char *type_name    = "str";
    static constexpr auto converter           = ::culv_fast_parse_conv_str;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<PyObject *> {
    static constexpr const char *type_name    = "object";
    static constexpr auto converter           = ::culv_fast_parse_conv_pyobj;
    static constexpr PyTypeObject *type_guard = nullptr;
};

namespace detail {
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
} // namespace detail

template <> struct FastParseTrait<CPH_Vec3f> {
    static constexpr const char *type_name    = "Vector3";
    static constexpr auto converter           = detail::fast_parse_conv_vec3f;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<CPH_PosStride> {
    static constexpr const char *type_name    = "Vector3";
    static constexpr auto converter           = detail::fast_parse_conv_pos_stride;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<CPH_AuxStride> {
    static constexpr const char *type_name    = "Quaternion";
    static constexpr auto converter           = detail::fast_parse_conv_aux_stride;
    static constexpr PyTypeObject *type_guard = nullptr;
};

using ParseBodyHandle = uint64_t;

// Runtime argument metadata used by the typed parser implementation.
template <typename T> struct FastArg {
    const char *name;
    bool required;

    constexpr FastArg(const char *arg_name, bool is_required = false) noexcept
        : name(arg_name), required(is_required) {}
};

// A structural string type lets parser names and argument names live in their
// declarations instead of a separate runtime initialization list.
template <std::size_t N> struct FixedString {
    char value[N]{};

    consteval FixedString(const char (&text)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            value[i] = text[i];
        }
    }

    [[nodiscard]] constexpr const char *c_str() const noexcept { return value; }
};

template <FixedString Name, typename T, bool Required = false> struct Arg {
    using ValueType = T;

    [[nodiscard]] static constexpr FastArg<T> Make() noexcept { return {Name.c_str(), Required}; }
};

inline constexpr std::size_t kParserCatalogCapacity = 128;

// Each parser adds itself to the catalog owned by its group as it is
// constructed. This keeps schema dumping available without a second registry
// declaration or a manual registration pass.
class ParserCatalog {
    std::array<CPH_FastParser *, kParserCatalogCapacity> parsers_{};
    std::size_t size_{};

  public:
    void Add(CPH_FastParser *parser) noexcept {
        if (size_ >= parsers_.size()) {
            Py_FatalError("parser group exceeded the parser catalog capacity");
        }
        parsers_[size_++] = parser;
    }

    [[nodiscard]] CPH_FastParser *const *Data() const noexcept { return parsers_.data(); }
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }
};

template <typename... Ts> class FastParser {
    CPH_FastParser core_{};
    bool initialized_{};

  public:
    FastParser()                              = default;
    FastParser(const FastParser &)            = delete;
    FastParser &operator=(const FastParser &) = delete;
    FastParser(FastParser &&)                 = delete;
    FastParser &operator=(FastParser &&)      = delete;

    ~FastParser() noexcept {
        if (initialized_) {
            CPH_FastParseDeinit(&core_);
        }
    }

    void Init(const char *parser_name, FastArg<Ts>... args) noexcept {
        if (initialized_) {
            CPH_FastParseDeinit(&core_);
            core_        = CPH_FastParser{};
            initialized_ = false;
        }

        if constexpr (sizeof...(Ts) > 0) {
            const std::array<CPH_FastArgDef, sizeof...(Ts)> c_defs{{
                CPH_FastArgDef{
                    .name       = args.name,
                    .type_name  = FastParseTrait<Ts>::type_name,
                    .type_guard = FastParseTrait<Ts>::type_guard,
                    .convert    = FastParseTrait<Ts>::converter,
                    .required   = args.required,
                }...,
            }};
            CPH_FastParseInitImpl(&core_, c_defs.data(), c_defs.size());
        } else {
            CPH_FastParseInitImpl(&core_, nullptr, 0);
        }
        core_.parser_name = parser_name;
        initialized_      = true;
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

// A parser object is fully described by its compile-time name and argument
// metadata. Its constructor initializes the fast core and registers that core
// with the owning group's catalog.
template <FixedString Name, typename... Args>
class Parser final : public FastParser<typename Args::ValueType...> {
    using Base = FastParser<typename Args::ValueType...>;

  public:
    explicit Parser(ParserCatalog &catalog) noexcept {
        Base::Init(Name.c_str(), Args::Make()...);
        catalog.Add(Base::Core());
    }

    using Base::Core;
    using Base::Parse;
};

template <typename ParserGroup> [[nodiscard]] ParserGroup *NewParserGroup() noexcept {
    auto *group = new (std::nothrow) ParserGroup{};
    if (group == nullptr) {
        PyErr_NoMemory();
        return nullptr;
    }
    return group;
}

} // namespace CPH::Python
