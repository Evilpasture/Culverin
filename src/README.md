# Source boundaries

- `native/` is the Python-independent C/C++ engine. The `culverin_core` target has no Python include directories, Python sources, or CPython link dependency.
- `python/culverin/` contains the importable Python package and its typing stubs.
- `python/native/` contains the CPython extension and its adapters. This is the only project source tree that includes `<Python.h>`; CMake supplies that upstream header from the selected Python development installation. CPython headers are intentionally not vendored.

The extension depends on `culverin_core`; the native core must never depend on the Python layer. Shared native data contracts belong under `native/`, not in the binding tree.

CMake follows the same boundaries. `native/core` defines the shared header-only `culverin_native_common` target, while `native/threading` supplies the Python-independent mutex and fiber primitives compiled into `culverin_core`. `native/jolt` owns Jolt-specific configuration and utilities plus separate runtime and RTTI object targets. `native/math`, `native/debug`, and `native/physics` each define domain object targets. Jolt runtime, math, debug, and physics objects are archived into `culverin_core`; the RTTI anchor is linked directly into the extension. Python adapter directories define `culverin_python_*` object targets, including the local parser and docstring helpers under `python/native/bindings/`, which are composed with the module initializer into `_culverin_c`. Header-only `native/cpp` and `native/soft_bodies` are interface targets rather than artificial compiled libraries.
