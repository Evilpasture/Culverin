#pragma once

// A vectorcall thunk: let the parser's own type carry the argument list.
//
// A parser is `Parser<Name, Args...>` with `Parse(args, nargs, kwnames, ...)`,
// so the set of locals a binding method needs is already known at compile
// time. `InvokeThunk` allocates those locals as one tuple, runs the parse,
// raises the captured error through the same `RaiseParseError` channel the
// hand-written adapters use, and calls an action with the unpacked values.
// The CPython plumbing disappears from the method body without moving any
// behaviour out of it: the parser still owns the diagnostics, and the action
// still owns the order of validation and the native call.
//
// Deliberately not supported: parsers with optional arguments whose default is
// not the value-initialised one. `Arg<Name, T, Required>` carries no default --
// each call site sets it on its own local before `Parse` -- and a missing
// optional argument leaves the target untouched, so a tuple of zeroes would
// silently change such a method's behaviour. Those keep their hand-written
// wrapper until `Arg` grows a default.

#include "bindings/parsing/Parser.hpp"
#include "bindings/parsing/ParseError.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace CPH::Python {

// The value types behind a parser: one local per declared argument.
template <typename T> struct ParserSignature {
    static constexpr bool is_parser = false;
};

template <FixedString Name, typename... Args>
struct ParserSignature<Parser<Name, Args...>> {
    static constexpr bool is_parser    = true;
    using Tuple                        = std::tuple<typename Args::ValueType...>;
    using ParserType                   = Parser<Name, Args...>;
    static constexpr std::size_t kArity = sizeof...(Args);

    template <std::size_t... Is>
    [[nodiscard]] static ParseResult ParseTuple(const ParserType &parser, PyObject *const *args,
                                                Py_ssize_t nargs, PyObject *kwnames, Tuple &values,
                                                std::index_sequence<Is...>) {
        return parser.Parse(args, nargs, kwnames, std::get<Is>(values)...);
    }

    // An empty pack is fine here: `Parser` handles arity zero in its own
    // vectorcall overload, so the zero-argument case needs no special spelling.
    [[nodiscard]] static ParseResult ParseTuple(const ParserType &parser, PyObject *const *args,
                                                Py_ssize_t nargs, PyObject *kwnames, Tuple &values) {
        return ParseTuple(parser, args, nargs, kwnames, values, std::make_index_sequence<kArity>{});
    }
};

// The two return shapes a thunk action may use. Anything else is rejected at
// the call site rather than being converted through one of these.
[[nodiscard]] inline PyObject *MarshalThunkResult(PyObject *result) noexcept { return result; }

[[nodiscard]] inline PyObject *MarshalThunkResult(bool result) noexcept {
    return PyBool_FromLong(result ? 1 : 0);
}

template <typename Result> PyObject *MarshalThunkResult(Result) = delete;

template <auto ParserMember, typename Self, typename Action>
[[nodiscard]] inline PyObject *InvokeThunk(Self *self, PyObject *const *args, size_t nargsf,
                                           PyObject *kwnames, Action &&action) {
    using ParserType = std::remove_cvref_t<decltype(self->parsers->*ParserMember)>;
    using Sig        = ParserSignature<ParserType>;
    static_assert(Sig::is_parser, "the member must be a Parser declared in the group");

    // Every argument starts value-initialised; see the header comment for why
    // that restricts this to all-required parsers.
    typename Sig::Tuple values{};

    const ParseResult parsed = Sig::ParseTuple(self->parsers->*ParserMember, args,
                                               PyVectorcall_NARGS(nargsf), kwnames, values);
    if (!parsed) [[unlikely]] {
        RaiseParseError(parsed.error());
        return nullptr;
    }

    auto invoke = [self, &action](auto &...unpacked) { return action(self, unpacked...); };

    if constexpr (std::is_void_v<decltype(std::apply(invoke, values))>) {
        std::apply(invoke, values);
        Py_RETURN_NONE;
    } else {
        return MarshalThunkResult(std::apply(invoke, values));
    }
}

} // namespace CPH::Python
