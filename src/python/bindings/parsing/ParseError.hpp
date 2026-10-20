#pragma once

// The parser layer's error channel (Phase 2, step B3).
//
// Before B3 every parse failure reported itself by leaving a Python exception
// in the thread state and returning `false`; every call site then had to
// remember to `return nullptr` (and hope nothing in between cleared it). The
// channel is now explicit:
//
//   ParseResult = std::expected<void, ParseError>
//
// and `ParseError` owns *the exact exception instance* the diagnostic builder
// produced. Raising it at the boundary (`RaiseParseError`) reinstates that
// object verbatim, so exception type, message, traceback-relevant state and
// even the repr embedded in the message are byte-identical to the pre-B3
// behaviour — `tests/test_diagnostics.py` pins exactly that.
//
// Ownership note: a ParseError never touches the thread-state error indicator
// while it is alive. Building one *does* use PyErr as a scratch pad (the
// diagnostic formatters set and re-read it), and `CaptureError()` empties it
// into the returned ParseError. That is why call sites must raise the error
// rather than assume an exception is already pending.

#include "bindings/PythonCommon.hpp"

#include <expected>
#include <utility>


namespace CPH::Python {

class ParseError {
  public:
    ParseError() noexcept = default;
    explicit ParseError(Ref exception) noexcept : m_exception(std::move(exception)) {}

    [[nodiscard]] bool valid() const noexcept { return static_cast<bool>(m_exception); }
    [[nodiscard]] PyObject *exception() const noexcept { return m_exception.get(); }

    // Hands the exception back out (used when a decorator re-wraps it).
    [[nodiscard]] Ref release() noexcept { return std::move(m_exception); }

  private:
    Ref m_exception;
};

using ParseResult = std::expected<void, ParseError>;

// Moves whatever exception is currently set into a ParseError, leaving the
// thread state clean.
//
// A null instance is allowed and means "the failure carried no exception".
// That is not an oversight: several converters (``parse_sequence_internal``
// on ``None``, ``ParseTracksToC`` on a non-list) returned a bare false
// without reporting anything, and the pre-B3 code let CPython turn the NULL
// return into ``SystemError: ... returned NULL without setting an
// exception``. Fabricating a message here would change those diagnostics, so
// the channel carries "no reason" verbatim and ``RaiseParseError`` stays
// silent for it.
[[nodiscard]] inline ParseError CaptureError() noexcept {
    return ParseError{Ref{PyErr_GetRaisedException()}};
}

[[nodiscard]] inline ParseResult CaptureFailure() noexcept {
    return std::unexpected(CaptureError());
}

// A failure that must not disturb the error indicator — for the legacy silent
// paths described above. The caller is expected to return nullptr and let
// CPython (or the enclosing boundary's own message) produce the diagnostic.
// Templated on the expected payload so value-returning parsers can use it too.
template <typename T = void>
[[nodiscard]] inline std::expected<T, ParseError> SilentFailure() noexcept {
    return std::unexpected(ParseError{});
}

// Raises a captured parse error at the Python boundary.
//
// PyErr_SetRaisedException steals its argument, so the ParseError keeps
// ownership of its instance and the indicator gets its own reference.
inline void RaiseParseError(const ParseError &error) noexcept {
    PyObject *exception = error.exception();
    if (exception != nullptr) {
        PyErr_SetRaisedException(Py_NewRef(exception));
    }
    // else: intentional no-op, see SilentFailure above.
}

} // namespace CPH::Python
