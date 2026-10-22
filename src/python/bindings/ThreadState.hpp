#pragma once

#include <Python.h>

namespace CPH::Python {

/// Releases the GIL for the lifetime of the object and reacquires it when the
/// object dies.
///
/// This replaces Py_BEGIN_ALLOW_THREADS / Py_END_ALLOW_THREADS, which are two
/// halves of one block: an early `return`, a `break` or a hand-added statement
/// between them leaves the interpreter state released, and the failure surfaces
/// much later and far away (usually as a crash inside an unrelated Python call
/// that ran without the GIL). With an object the release is tied to a scope, so
/// every exit path restores the state exactly once.
///
/// A `PyThreadState` is captured rather than restored blind, so nesting and
/// conditional use are both safe.
///
/// Everything between construction and destruction must stay out of the Python
/// API: no PyErr_*, no Py*_DECREF, no tuple building. Build the arguments
/// before the object exists, and read the results after it is gone.
class AllowThreads final {
  public:
    AllowThreads() noexcept : state_(PyEval_SaveThread()) {}

    ~AllowThreads() { PyEval_RestoreThread(state_); }

    AllowThreads(const AllowThreads &)            = delete;
    AllowThreads &operator=(const AllowThreads &) = delete;
    AllowThreads(AllowThreads &&)                 = delete;
    AllowThreads &operator=(AllowThreads &&)      = delete;

  private:
    PyThreadState *state_;
};

} // namespace CPH::Python
