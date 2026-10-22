#pragma once

// The compile-time half of the argument parser.
//
// `Parser<Name, Args...>` knows its argument list as a type, so the whole
// positional path is a single fold over `std::index_sequence`:
//
//     (ConvertArg<Is>() && ...)
//
// which expands to one straight-line check-and-convert per argument, fully
// inlined, storing directly into the caller's locals through the references
// the parser was handed. No target-pointer array on the stack, no indirection
// through a spec table, no speculative per-arity entry points -- the argument
// count is a compile-time constant, so there is nothing left to speculate
// about. What used to be nine hand-written stubs plus a generic vectorcall
// loop is one function with a different expansion per parser.
//
// Keywords are the other half: `kwnames != nullptr` leaves the fold and calls
// `ParseSlow`, which builds the runtime mask and dispatches each keyword by
// index through a compile-time binary split. Diagnostics live behind
// `ParserCore::Report*`, all `noinline` and `cold`.

#include "bindings/PythonCommon.hpp"
#include "bindings/parsing/ParseError.hpp"
#include "bindings/parsing/ParserCore.hpp"
#include "bindings/parsing/ParserTraits.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <new>
#include <tuple>
#include <utility>

namespace CPH::Python {

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

    static constexpr bool kRequired    = Required;
    static constexpr const char *kName = Name.c_str();
};

inline constexpr std::size_t kParserCatalogCapacity = 128;

// Each parser adds itself to the catalog owned by its group as it is
// constructed. This keeps schema dumping available without a second registry
// declaration or a manual registration pass.
class ParserCatalog {
    std::array<ParserCore *, kParserCatalogCapacity> parsers_{};
    std::size_t size_{};

  public:
    void Add(ParserCore *parser) noexcept {
        if (size_ >= parsers_.size()) {
            Py_FatalError("parser group exceeded the parser catalog capacity");
        }
        parsers_[size_++] = parser;
    }

    [[nodiscard]] ParserCore *const *Data() const noexcept { return parsers_.data(); }
    [[nodiscard]] std::size_t Size() const noexcept { return size_; }
};

namespace detail {

// Where a positional argument comes from. Vectorcall hands over a bare array;
// the legacy `__init__`-style call hands over a tuple. Everything downstream
// indexes through `operator[]`, so there is one fold, not two.
struct VectorArguments {
    PyObject *const *values;

    [[gnu::always_inline]] PyObject *operator[](Py_ssize_t index) const noexcept {
        return values[index];
    }
};

struct TupleArguments {
    PyObject *tuple;

    [[gnu::always_inline]] PyObject *operator[](Py_ssize_t index) const noexcept {
        return PyTuple_GET_ITEM(tuple, index);
    }
};

// Compile-time half of the type check: the trait's guard is a constant for
// every type except `bool`, so for the rest this folds away entirely.
template <typename T>
[[gnu::always_inline]] inline bool PassesTypeGuard(PyObject *value) noexcept {
    PyTypeObject *guard = ParserTrait<T>::kTypeGuard;
    if (guard == nullptr) [[likely]] {
        return true;
    }
    if (Py_IS_TYPE(value, guard)) [[likely]] {
        return true;
    }
    return PyObject_TypeCheck(value, guard);
}

// Which conversion-error diagnostic a call site gets. See kPositionalDiagnostics.
enum class ConversionDiagnostics {
    Decorated,      // signature context, always
    Raw,            // the converter's own error, verbatim
    RawWhenStubbed, // raw only for a full-arity positional call to a stubbed parser
};

} // namespace detail

template <FixedString Name, typename... Args> class Parser final {
  public:
    static constexpr std::size_t kArity = sizeof...(Args);

  private:
    using ArgList                          = std::tuple<Args...>;
    template <std::size_t I> using ArgAt   = std::tuple_element_t<I, ArgList>;
    template <std::size_t I> using ValueAt = typename ArgAt<I>::ValueType;

    // Index of the last required argument, + 1. The positional path can only
    // fill a prefix of the argument list, so "every required argument was
    // supplied" is exactly `nargs >= kRequiredSpan` -- the same test the old
    // engine expressed as `(MakeMask(nargs) & requiredMask) == requiredMask`,
    // resolved at compile time instead.
    [[nodiscard]] static constexpr std::size_t ComputeRequiredSpan() noexcept {
        if constexpr (sizeof...(Args) == 0) {
            return 0;
        } else {
            constexpr bool required[] = {Args::kRequired...};
            std::size_t span          = 0;
            for (std::size_t i = 0; i < sizeof...(Args); ++i) {
                if (required[i]) {
                    span = i + 1;
                }
            }
            return span;
        }
    }

    static constexpr bool kAllRequired         = (Args::kRequired && ...);
    static constexpr std::size_t kRequiredSpan = ComputeRequiredSpan();

    // Bug-compatibility, not design. Before this redesign the engine picked one
    // of nine monomorphic stubs when a parser had 1..8 arguments, all of them
    // required, *and* the call supplied exactly that many positionals and no
    // keywords; those stubs returned the converter's raw error, while every
    // other path decorated it with the argument signature. The distinction is
    // an artefact of which entry point happened to be selected and is
    // otherwise unobservable -- but it is observable in the text of a
    // TypeError, so it is preserved here rather than changed under the banner
    // of a refactor.
    // TODO(smell): unify on the decorated message; tests/test_diagnostics.py
    // pins neither form, so this is a deliberate, disclosed choice.
    static constexpr bool kStubbed = kArity >= 1 && kArity <= 8 && kAllRequired;
    static constexpr detail::ConversionDiagnostics kPositionalDiagnostics =
        kStubbed ? detail::ConversionDiagnostics::RawWhenStubbed
                 : detail::ConversionDiagnostics::Decorated;

    ParserCore core_{};

    template <std::size_t I> [[nodiscard]] static ArgDef Describe() noexcept {
        return ArgDef{.name      = ArgAt<I>::kName,
                      .typeName  = ParserTrait<ValueAt<I>>::name,
                      .typeGuard = ParserTrait<ValueAt<I>>::kTypeGuard,
                      .required  = ArgAt<I>::kRequired};
    }

    void Initialize() noexcept {
        if constexpr (kArity == 0) {
            core_.Init(nullptr, 0);
        } else {
            [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                const std::array<ArgDef, kArity> defs{{Describe<Is>()...}};
                core_.Init(defs.data(), defs.size());
            }(std::index_sequence_for<Args...>{});
        }
        core_.parserName = Name.c_str();
    }

    // --- the argument fold ------------------------------------------------

    // One argument: guard, then conversion, straight into the caller's slot.
    template <std::size_t I, detail::ConversionDiagnostics kDiagnostics, typename TargetTuple>
    [[gnu::always_inline]] ParseResult ConvertValue(PyObject *value,
                                                    [[maybe_unused]] Py_ssize_t nargs,
                                                    TargetTuple &targets) const {
        using Value = ValueAt<I>;

        if (!detail::PassesTypeGuard<Value>(value)) [[unlikely]] {
            return std::unexpected(core_.ReportTypeError(I, value));
        }
        if (!ParserTrait<Value>::convert(value, std::get<I>(targets))) [[unlikely]] {
            if constexpr (kDiagnostics == detail::ConversionDiagnostics::Raw) {
                return CaptureFailure();
            } else if constexpr (kDiagnostics == detail::ConversionDiagnostics::Decorated) {
                return std::unexpected(core_.ReportConversionError(I, value));
            } else {
                // Raw only on a full-arity call, exactly like the old stub.
                // The compare lives in this branch, so it costs the hot path
                // nothing; see kStubbed.
                if (nargs != static_cast<Py_ssize_t>(kArity)) {
                    return std::unexpected(core_.ReportConversionError(I, value));
                }
                return CaptureFailure();
            }
        }
        return {};
    }

    template <std::size_t I, detail::ConversionDiagnostics kDiagnostics, typename Source,
              typename TargetTuple>
    [[gnu::always_inline]] bool ConvertArg(const Source &source, Py_ssize_t nargs,
                                           TargetTuple &targets, ParseError &failure) const {
        if (nargs <= static_cast<Py_ssize_t>(I)) {
            return true; // optional tail not supplied
        }
        if (ParseResult step = ConvertValue<I, kDiagnostics>(source[I], nargs, targets); !step) {
            failure = std::move(step.error());
            return false;
        }
        return true;
    }

    template <detail::ConversionDiagnostics kDiagnostics, typename Source, typename TargetTuple>
    [[gnu::always_inline]] ParseResult ConvertPositional(const Source &source, Py_ssize_t nargs,
                                                         TargetTuple &targets) const {
        ParseError failure{};
        const bool converted = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return (ConvertArg<Is, kDiagnostics>(source, nargs, targets, failure) && ...);
        }(std::index_sequence_for<Args...>{});
        if (!converted) [[unlikely]] {
            return std::unexpected(std::move(failure));
        }
        return {};
    }

    // Keyword dispatch: the index is a runtime value, the converters are
    // compile-time. A binary split costs ceil(log2 N) well-predicted compares
    // and still ends in an inlined conversion -- no erased function pointer.
    template <detail::ConversionDiagnostics kDiagnostics, std::size_t Begin, std::size_t End,
              typename TargetTuple>
    [[gnu::always_inline]] ParseResult ConvertRange(std::size_t index, PyObject *value,
                                                    TargetTuple &targets) const {
        if constexpr (End - Begin == 1) {
            // Keywords always decorate, so the arity argument is never read here.
            return ConvertValue<Begin, kDiagnostics>(value, static_cast<Py_ssize_t>(kArity),
                                                     targets);
        } else {
            constexpr std::size_t kMid = Begin + (End - Begin) / 2;
            if (index < kMid) {
                return ConvertRange<kDiagnostics, Begin, kMid>(index, value, targets);
            }
            return ConvertRange<kDiagnostics, kMid, End>(index, value, targets);
        }
    }

    template <detail::ConversionDiagnostics kDiagnostics, typename TargetTuple>
    [[gnu::always_inline]] ParseResult ConvertKeyword(PyObject *key, PyObject *value,
                                                      TargetTuple &targets, uint64_t &mask) const {
        const std::size_t index = core_.FindKeywordIndex(key);
        if (index == kParserEmptySlot) [[unlikely]] {
            return std::unexpected(core_.ReportUnknownKeyword(key));
        }
        if ((mask & (1ULL << index)) != 0) [[unlikely]] {
            return std::unexpected(core_.ReportMultiple(index));
        }
        mask |= 1ULL << index;

        if constexpr (kArity == 0) {
            // Unreachable: a parser with no arguments accepts no keywords, so
            // FindKeywordIndex cannot produce an index.
            return std::unexpected(core_.ReportUnknownKeyword(key));
        } else {
            return ConvertRange<kDiagnostics, 0, kArity>(index, value, targets);
        }
    }

    // Keyword and mixed calls. Out of line: it carries the mask bookkeeping and
    // the per-index dispatch, none of which belongs in a caller's frame.
    [[gnu::noinline]] ParseResult ParseSlow(PyObject *const *args, Py_ssize_t nargs,
                                            PyObject *kwnames,
                                            typename Args::ValueType &...values) const {
        if (nargs > static_cast<Py_ssize_t>(kArity)) [[unlikely]] {
            return std::unexpected(core_.ReportTooMany(nargs));
        }

        auto targets = std::forward_as_tuple(values...);
        if (ParseResult converted = ConvertPositional<detail::ConversionDiagnostics::Decorated>(
                detail::VectorArguments{args}, nargs, targets);
            !converted) [[unlikely]] {
            return converted;
        }

        uint64_t mask = MakeMask(static_cast<std::size_t>(nargs));

        const Py_ssize_t keyword_count  = PyTuple_GET_SIZE(kwnames);
        PyObject *const *keyword_values = args + nargs;

        for (Py_ssize_t i = 0; i < keyword_count; ++i) {
            if (ParseResult step =
                    ConvertKeyword<detail::ConversionDiagnostics::Decorated>(
                        PyTuple_GET_ITEM(kwnames, i), keyword_values[i], targets, mask);
                !step) [[unlikely]] {
                return step;
            }
        }

        if ((mask & core_.requiredMask) != core_.requiredMask) [[unlikely]] {
            return std::unexpected(core_.ReportMissing(mask));
        }
        return {};
    }

  public:
    explicit Parser(ParserCatalog &catalog) noexcept {
        Initialize();
        catalog.Add(&core_);
    }

    Parser(const Parser &)            = delete;
    Parser &operator=(const Parser &) = delete;
    Parser(Parser &&)                 = delete;
    Parser &operator=(Parser &&)      = delete;

    ~Parser() noexcept { core_.Deinit(); }

    // Vectorcall: positionals go through the fold, keywords through ParseSlow.
    [[nodiscard, gnu::always_inline]] ParseResult
    Parse(PyObject *const *args, Py_ssize_t nargs, PyObject *kwnames,
          typename Args::ValueType &...values) const {
        if (kwnames != nullptr) [[unlikely]] {
            return ParseSlow(args, nargs, kwnames, values...);
        }

        if (nargs > static_cast<Py_ssize_t>(kArity)) [[unlikely]] {
            return std::unexpected(core_.ReportTooMany(nargs));
        }

        auto targets = std::forward_as_tuple(values...);
        if (ParseResult converted = ConvertPositional<kPositionalDiagnostics>(
                detail::VectorArguments{args}, nargs, targets);
            !converted) [[unlikely]] {
            return converted;
        }

        if constexpr (kRequiredSpan != 0) {
            if (nargs < static_cast<Py_ssize_t>(kRequiredSpan)) [[unlikely]] {
                return std::unexpected(
                    core_.ReportMissing(MakeMask(static_cast<std::size_t>(nargs))));
            }
        }
        return {};
    }

    // Legacy: `__init__`-shaped calls hand over a tuple and a dict.
    [[nodiscard, gnu::hot, gnu::noinline]] ParseResult
    Parse(PyObject *args, PyObject *kwargs, typename Args::ValueType &...values) const {
        const Py_ssize_t nargs = args ? PyTuple_GET_SIZE(args) : 0;

        if (nargs > static_cast<Py_ssize_t>(kArity)) [[unlikely]] {
            return std::unexpected(core_.ReportTooMany(nargs));
        }

        auto targets = std::forward_as_tuple(values...);
        if (ParseResult converted =
                ConvertPositional<detail::ConversionDiagnostics::Raw>(detail::TupleArguments{args},
                                                                    nargs, targets);
            !converted) [[unlikely]] {
            return converted;
        }

        uint64_t mask = MakeMask(static_cast<std::size_t>(nargs));
        if (kwargs != nullptr && PyDict_GET_SIZE(kwargs) > 0) {
            PyObject *key   = nullptr;
            PyObject *value = nullptr;
            Py_ssize_t pos  = 0;
            while (PyDict_Next(kwargs, &pos, &key, &value)) {
                // Undecorated conversion errors, as before: this path has
                // always reported the converter's raw error.
                if (ParseResult step =
                        ConvertKeyword<detail::ConversionDiagnostics::Raw>(key, value, targets,
                                                                          mask);
                    !step) [[unlikely]] {
                    return step;
                }
            }
        }

        if ((mask & core_.requiredMask) != core_.requiredMask) [[unlikely]] {
            return std::unexpected(core_.ReportMissing(mask));
        }
        return {};
    }

    [[nodiscard]] ParserCore *Core() noexcept { return &core_; }
    [[nodiscard]] const ParserCore *Core() const noexcept { return &core_; }
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
