#pragma once

#include "bindings/parsing/FastParserCore.hpp"
#include "bindings/parsing/ParseError.hpp"
#include "bindings/parsing/Parsers.hpp"
#include "core/Types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>

namespace CPH::Python {

template <typename T> struct FastParseTrait;

template <> struct FastParseTrait<float> {
    static constexpr const char *type_name    = "float";
    static constexpr auto converter           = FastParseConvFloat;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<double> {
    static constexpr const char *type_name    = "float";
    static constexpr auto converter           = FastParseConvDouble;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<int> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = FastParseConvInt;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<uint32_t> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = FastParseConvU32;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<uint64_t> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = FastParseConvU64;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<Py_ssize_t> {
    static constexpr const char *type_name    = "int";
    static constexpr auto converter           = FastParseConvSsize;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<bool> {
    static constexpr const char *type_name    = "bool";
    static constexpr auto converter           = FastParseConvBool;
    static constexpr PyTypeObject *type_guard = &PyBool_Type;
};

template <> struct FastParseTrait<const char *> {
    static constexpr const char *type_name    = "str";
    static constexpr auto converter           = FastParseConvStr;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<PyObject *> {
    static constexpr const char *type_name    = "object";
    static constexpr auto converter           = FastParseConvPyobj;
    static constexpr PyTypeObject *type_guard = nullptr;
};

namespace detail {
// The converter contract is still `bool` + ambient PyErr: the typed parsers
// hand back a captured ParseError, so each adapter raises it again before
// returning false. FastParseReportConversionError then decorates that
// exception exactly as it always has.
template <typename T, typename ParseInto>
[[nodiscard]] static inline bool ConvertStride(PyObject *obj, const void *target,
                                                ParseInto parse_into) {
    auto *value = const_cast<T *>(static_cast<const T *>(target));
    if (ParseResult parsed = parse_into(Borrowed{obj}, *value); !parsed) {
        RaiseParseError(parsed.error());
        return false;
    }
    return true;
}

[[nodiscard]] static inline bool FastParseConvFloat3(PyObject *obj, const void *target) {
    return ConvertStride<JPH::Float3>(obj, target, [](Borrowed o, JPH::Float3 &v) {
        return ParseVec3Into(o, v);
    });
}

[[nodiscard]] static inline bool FastParseConvPosStride(PyObject *obj, const void *target) {
    return ConvertStride<PosStride>(obj, target, [](Borrowed o, PosStride &v) {
        return ParseVec3Into(o, v);
    });
}

[[nodiscard]] static inline bool FastParseConvAuxStride(PyObject *obj, const void *target) {
    return ConvertStride<AuxStride>(obj, target, [](Borrowed o, AuxStride &v) {
        return ParseQuatInto(o, v);
    });
}
} // namespace detail

template <> struct FastParseTrait<JPH::Float3> {
    static constexpr const char *type_name    = "Vector3";
    static constexpr auto converter           = detail::FastParseConvFloat3;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<PosStride> {
    static constexpr const char *type_name    = "Vector3";
    static constexpr auto converter           = detail::FastParseConvPosStride;
    static constexpr PyTypeObject *type_guard = nullptr;
};

template <> struct FastParseTrait<AuxStride> {
    static constexpr const char *type_name    = "Quaternion";
    static constexpr auto converter           = detail::FastParseConvAuxStride;
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

// TODO(smell): `Arg` carries the name and the required flag but not the
// default value. Every call site sets its default on its own local before
// `Parse`, and a missing optional argument leaves the target untouched -- so
// no generic wrapper can reproduce a method with defaults, which is why
// Thunk.hpp is limited to all-required parsers and why the wide parsers keep
// their hand-written locals. Giving `Arg` a default value is the fix.
template <FixedString Name, typename T, bool Required = false> struct Arg {
    using ValueType = T;

    [[nodiscard]] static constexpr FastArg<T> Make() noexcept { return {Name.c_str(), Required}; }
};

inline constexpr std::size_t kParserCatalogCapacity = 128;

// Each parser adds itself to the catalog owned by its group as it is
// constructed. This keeps schema dumping available without a second registry
// declaration or a manual registration pass.
class ParserCatalog {
    std::array<FastParserCore *, kParserCatalogCapacity> parsers_{};
    std::size_t size_{};

  public:
    void Add(FastParserCore *parser) noexcept {
        if (size_ >= parsers_.size()) {
            Py_FatalError("parser group exceeded the parser catalog capacity");
        }
        parsers_[size_++] = parser;
    }

    [[nodiscard]] FastParserCore *const *Data() const noexcept { return parsers_.data(); }
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }
};

template <typename... Ts> class FastParser {
    FastParserCore core_{};
    bool initialized_{};

  public:
    FastParser()                              = default;
    FastParser(const FastParser &)            = delete;
    FastParser &operator=(const FastParser &) = delete;
    FastParser(FastParser &&)                 = delete;
    FastParser &operator=(FastParser &&)      = delete;

    ~FastParser() noexcept {
        if (initialized_) {
            FastParseDeinit(&core_);
        }
    }

    void Init(const char *parser_name, FastArg<Ts>... args) noexcept {
        if (initialized_) {
            FastParseDeinit(&core_);
            core_        = FastParserCore{};
            initialized_ = false;
        }

        if constexpr (sizeof...(Ts) > 0) {
            const std::array<FastArgDef, sizeof...(Ts)> c_defs{{
                FastArgDef{
                    .name       = args.name,
                    .type_name  = FastParseTrait<Ts>::type_name,
                    .type_guard = FastParseTrait<Ts>::type_guard,
                    .convert    = FastParseTrait<Ts>::converter,
                    .required   = args.required,
                }...,
            }};
            FastParseInitImpl(&core_, c_defs.data(), c_defs.size());
        } else {
            FastParseInitImpl(&core_, nullptr, 0);
        }
        core_.parser_name = parser_name;
        initialized_      = true;
    }

    [[nodiscard, gnu::always_inline]] ParseResult Parse(PyObject *const *args, Py_ssize_t nargs,
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

    [[nodiscard, gnu::always_inline]] ParseResult Parse(PyObject *args, PyObject *kwargs,
                                                        Ts &...targets) const {
        if constexpr (sizeof...(Ts) == 0) {
            return FastParseLegacy(args, kwargs, nullptr, &core_, nullptr);
        } else {
            const void *const target_ptrs[] = {
                reinterpret_cast<const void *>(&targets)...,
            };
            return FastParseLegacy(args, kwargs, nullptr, &core_, target_ptrs);
        }
    }

    [[nodiscard]] FastParserCore *Core() noexcept { return &core_; }
    [[nodiscard]] const FastParserCore *Core() const noexcept { return &core_; }
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
