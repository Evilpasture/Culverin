# Source boundaries

- `native/` is the Python-independent C/C++ engine. The `culverin_core` target has no Python include directories, Python sources, or CPython link dependency.
- `python/culverin/` contains the importable Python package and its typing stubs.
- `python/native/` contains the CPython extension and its adapters. This is the only project source tree that includes `<Python.h>`; CMake supplies that upstream header from the selected Python development installation. CPython headers are intentionally not vendored.

The extension depends on `culverin_core`; the native core must never depend on the Python layer. Shared native data contracts belong under `native/`, not in the binding tree.
