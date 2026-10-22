# Source layout

- The Python-independent C/C++ engine lives directly under `src/` (`core/`, `cpp/`, `debug/`, `jolt/`, `math/`, `physics/`, `soft_bodies/`, and `threading/`). It builds `culverin_core` and has no Python include directories or CPython link dependency.
- `src/python/culverin/` is the importable Python package and contains its typing stubs.
- The CPython extension and its adapters also live directly under `src/python/`, in `bindings/`, `characters/`, `ecs/`, `math/`, `ragdolls/`, `soft_bodies/`, and `vehicles/`. These adapter sources are the only project sources that include `<Python.h>`; CMake supplies CPython headers from the selected Python development installation, and they are not vendored.

The extension depends on `culverin_core`; the engine must never depend on the Python layer. Shared C/C++ data contracts belong with the engine sources, not in the binding code.

CMake follows these boundaries: `core/` defines the shared header-only `culverin_native_common` target; `threading/` supplies Python-independent mutex and fiber primitives; `jolt/` owns Jolt-specific configuration and runtime/RTTI object targets; and `math/`, `debug/`, and `physics/` define engine-domain targets. Jolt runtime, math, debug, and physics objects are archived into `culverin_core`; the RTTI anchor is linked directly into the extension. The Python adapter directories define `culverin_python_*` object targets, including parser and docstring helpers under `python/bindings/`, and compose them with the module initializer into `_culverin_c`. Header-only `cpp/` and `soft_bodies/` are interface targets rather than artificial compiled libraries.
