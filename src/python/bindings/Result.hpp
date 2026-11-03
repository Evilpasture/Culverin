#pragma once

// The return channel (Phase 3, step C1).
//
// CPython reports failure through two unsynchronised channels at once: the
// return value (`PyObject *`, NULL means "failed") and the thread-state error
// indicator (`PyErr_SetString`, "an exception is pending"). Nothing in the C
// API ties them together, so both mismatches are reachable from ordinary code:
//
//   * NULL with no exception pending  -> CPython substitutes
//                                        "SystemError: <built-in> returned NULL
//                                        without setting an exception"
//   * a value with an exception pending -> the caller receives the value and the
//                                        exception latches, poisoning the next
//                                        unrelated call
//
// Neither is detectable by the compiler while a function returns `PyObject *`,
// because a pointer carries no information about the error indicator.
//
// `BoundResult` carries both facts in one value, so the mismatch does not
// type-check:
//
//   std::expected<BoundValue, Exception>
//
// A success owns exactly one reference and implies no exception; a failure owns
// an exception description and implies no value. There is no third
// representation and no way to build one by accident.
//
// Code never returns to CPython directly. `CPH::Python::Bridge` is the single
// place that reads a `BoundResult` and writes the two channels together, and
// the registration helpers in PythonSupport.hpp refuse (at compile time) to
// register any function that does not return `BoundResult`.

#include "bindings/PythonCommon.hpp"
#include "bindings/parsing/ParseError.hpp"

#include <expected>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include "bindings/Result.hpp"

namespace CPH::Python {

// What failure looks like before it reaches the boundary.
//
// Three states, each named, none of them a default:
//
//   Instance  the exact exception object to reinstate (a captured ParseError,
//             or one adopted from the indicator). Reinstating it verbatim keeps
//             the diagnostic byte-identical, which tests/test_diagnostics.py
//             pins.
//   Message   a type and a message to build at the boundary.
//   Silent    deliberately no exception -- the pre-existing behaviour of a few
//             converters that reported a bare failure. Naming it keeps the
//             "returned NULL without setting an exception" case from being
//             something you arrive at by forgetting; you have to ask for it.
class Exception {
  public:
    Exception() noexcept = default;

    // Adopts whatever exception is pending in the thread state, leaving the
    // indicator clean. A pending indicator is optional in CPython's own API
    // (see ParseError's "no reason" note), so this can yield Silent.
    [[nodiscard]] static Exception Current() noexcept {
        Exception result;
        result.mInstance = Ref{PyErr_GetRaisedException()};
        return result;
    }

    // Adopts a captured parse error's exception verbatim.
    [[nodiscard]] static Exception From(ParseError error) noexcept {
        Exception result;
        result.mInstance = error.release();
        return result;
    }

    // A fresh type + message, built at the boundary.
    [[nodiscard]] static Exception Message(PyObject *type, std::string_view text) noexcept {
        Exception result;
        result.mType    = type;
        result.mMessage = std::string{text};
        return result;
    }

    // A failure that carries no exception on purpose.
    [[nodiscard]] static Exception Silent() noexcept { return Exception{}; }

    [[nodiscard]] bool has_instance() const noexcept { return static_cast<bool>(mInstance); }
    [[nodiscard]] bool is_silent() const noexcept {
        return !mInstance && mType == nullptr && mMessage.empty();
    }
    [[nodiscard]] PyObject *instance() const noexcept { return mInstance.get(); }
    [[nodiscard]] PyObject *type() const noexcept { return mType; }
    [[nodiscard]] std::string_view message() const noexcept { return mMessage; }

    // Writes this failure into the thread state. Called by Bridge and nowhere
    // else; returns the pointer the caller must hand back to CPython.
    [[nodiscard]] PyObject *Raise() noexcept {
        if (mInstance) {
            // Steals the reference, preserving type, args and any traceback.
            PyErr_SetRaisedException(mInstance.release());
            return nullptr;
        }
        if (mType != nullptr) {
            PyErr_SetString(mType, mMessage.c_str());
            return nullptr;
        }
        // Silent: leave the indicator untouched. CPython will produce its own
        // diagnostic, exactly as it did before this channel existed.
        return nullptr;
    }

  private:
    Ref mInstance;
    PyObject *mType = nullptr;
    std::string mMessage;
};

// A value on its way back to CPython: exactly one owned reference, and
// provably not null once constructed through Adopt.
class BoundValue {
  public:
    explicit BoundValue(Ref value) noexcept : mValue(std::move(value)) {
        debug_check();
    }

    [[nodiscard]] PyObject *release() noexcept { return mValue.release(); }
    [[nodiscard]] PyObject *get() const noexcept { return mValue.get(); }

  private:
    void debug_check() const noexcept {
#if defined(CULVERIN_DEBUG)
        if (!mValue) {
            Py_FatalError("CPH::Python::BoundValue: constructed from a null reference");
        }
#endif
    }

    Ref mValue;
};

using BoundResult = std::expected<BoundValue, Exception>;

// --- Building results ---------------------------------------------------------
//
// Every function that returns to CPython ends in one of these. They are the
// whole vocabulary: adopt a new reference, adopt the pending exception, or
// declare a failure.

// Adopts a new reference from the C API. A NULL means the call failed and left
// its exception pending, so the two are folded into one value right here --
// this is the only place the legacy "NULL means look at the indicator"
// convention is interpreted.
[[nodiscard]] inline BoundResult Adopt(PyObject *object) noexcept {
    if (object == nullptr) {
        return std::unexpected(Exception::Current());
    }
    return BoundValue{Ref{object}};
}

// Success with a borrowed object (None, True, an argument being passed through).
[[nodiscard]] inline BoundResult BorrowedValue(PyObject *object) noexcept {
    return BoundValue{Ref::New(object)};
}

// Failure described by type and message.
[[nodiscard]] inline std::unexpected<Exception> Fail(PyObject *type, std::string_view text) noexcept {
    return std::unexpected(Exception::Message(type, text));
}

// Failure adopting the exception the failure path just set or captured.
[[nodiscard]] inline std::unexpected<Exception> Fail(ParseError error) noexcept {
    return std::unexpected(Exception::From(std::move(error)));
}

[[nodiscard]] inline std::unexpected<Exception> FailCurrent() noexcept {
    return std::unexpected(Exception::Current());
}

[[nodiscard]] inline std::unexpected<Exception> FailSilently() noexcept {
    return std::unexpected(Exception::Silent());
}

// --- The boundary -------------------------------------------------------------
//
// The single point where a C++ result becomes CPython's two channels. Method
// thunks (see PythonSupport.hpp) call this; nothing else should.

template <typename Function>
[[nodiscard]] inline PyObject *Bridge(Function &&function) noexcept {
    static_assert(std::is_same_v<std::invoke_result_t<Function>, BoundResult>,
                  "Bridge requires a callable returning CPH::Python::BoundResult");

    BoundResult result = std::forward<Function>(function)();
    if (!result.has_value()) {
        return result.error().Raise();
    }

    PyObject *value = result->release();
#if defined(CULVERIN_DEBUG)
    // Success means no exception may be pending. Reaching here with one set
    // means something wrote to the indicator mid-call and then returned a
    // value; surface it instead of letting it latch onto an unrelated call.
    if (PyErr_Occurred() != nullptr) {
        Py_DECREF(value);
        PyErr_Format(PyExc_SystemError,
                     "[Culverin Bug] method returned a value while an exception was pending");
        return nullptr;
    }
#endif
    return value;
}

} // namespace CPH::Python
