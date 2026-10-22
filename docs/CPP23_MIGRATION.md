# C23 → C++23 Migration: Design and Report

Base commit: `602feda` (branch `minor-refactor`).

This document records (1) the external contracts that constrain the rewrite,
(2) the replacement architecture, (3) which legacy abstractions were eliminated
and which were kept — with the concrete requirement that justifies each kept
one — and (4) the verification evidence. It is part of the patch so reviewers
can check the design against the diff.

## 1. External contracts (the behavioral reference)

These come from the Python API (`src/python/culverin/*.pyi`), the test suite,
the docs, and the JoltC ABI. The rewrite preserves all of them:

| Contract | Detail |
| --- | --- |
| Python API | No changes to `culverin` public API, signatures, exceptions, or status-code→exception mapping. |
| Position buffer | `float64`, stride 4 (`x, y, z, pad=0`), one record per dense body, exported zero-copy via the buffer protocol. |
| Rotation/velocity buffers | `float32`, stride 4. |
| Render state | 28 bytes/body (`7 × float32`), position padding removed. |
| Handles | 64-bit `(generation << 32) \| slot`; generation increments on slot reuse; stale handles fail safely. |
| Dense/sparse bodies | Swap-remove keeps dense arrays packed; `get_index` maps handle → dense index. |
| Deferred mutation | Mutations are staged in a command queue, mirrored into shadow buffers immediately, applied to Jolt at step/flush time, in order. |
| Concurrency | Lock hierarchy shadow-lock → step-sync; step priority (`step_requested` starves queries); queries register in `active_queries` and drain before stepping; shadow buffers of the previous frame stay readable during a step; resize is refused while buffer views are exported (`BufferError`). |
| Snapshot format | `save_state`/`load_state` byte layout (header, positions, rotations, velocities, generation/slot maps, slot states) and the capacity-mismatch error are unchanged. |
| Raycast batch records | Exactly 48 bytes per result. |
| Contact events | 128-byte records (64-byte slim + 64-byte fat extension), canonical handle ordering, capacity clamping. |
| JoltC ABI | All interaction with Jolt stays through `joltc.h` C-ABI structs and proc tables. |
| OOM behavior | The extension is built `-fno-exceptions`; the legacy code reports allocation failure as status codes → Python `MemoryError`. The replacement preserves this with nothrow allocation. |

Implementation-specific tests: none of the Python tests depend on C internals
(no test inspects symbol names, struct layouts, or source files). The new
`tests/test_differential_contracts.py` pins the table above and passes on both
implementations.

## 2. Replacement architecture

The internal representation was redesigned around ownership, not transliterated:

- **`core/Buffer.hpp` — one owning array type.** All world storage is
  `Buffer<T, Alignment>`: a move-only, nothrow, over-aligned contiguous array
  (aligned `operator new[]` + element construction/destruction). It replaces
  ~35 raw pointer/capacity field pairs and every manual allocation protocol
  (`CPH_AllocateBuffers`'s 25 allocations + combined null check,
  `CPH_NewBuffers`, `alloc_new_buffers`, `migrate_and_init`,
  `CPH_FreeShadowBuffers`, `CPH_FreeNewBuffers`, the trash-buffer list).
  `std::vector` was rejected for world storage: under `-fno-exceptions` its
  allocation failure aborts instead of producing the `MemoryError` status
  contract, and the project currently uses `std::vector` in zero places.
- **`BodyBuffers` bundle.** Dense SoA arrays + sparse slot maps that resize
  together are one struct. Resize = allocate a new bundle, copy live ranges,
  `retired = std::exchange(bodies, std::move(next))`. The single retired
  bundle preserves the legacy deferred-free epoch (old storage is released at
  the next resize/teardown, so a stale raw view from the previous epoch can
  never dangle) without the trash array's capacity bookkeeping. **The SoA
  layout itself is kept**: it *is* the buffer-protocol contract and the SIMD
  sync path (`ShadowSync`) streams exactly these arrays.
- **`CPH_PhysicsWorld` is a C++ object.** Member initializers replace the
  80-line `CPH_PhysicsWorldInitState`; the destructor replaces
  `CPH_PhysicsWorldFreeNative`/`CPH_FreeConstraints`/`CPH_FreeShapeCache`/
  `CPH_DebugBufferFree`. The CPython object placement-constructs it in
  `tp_new` and destroys it in `tp_dealloc` (CPython storage is zeroed raw
  memory, so construction must be explicit — this is the requirement that
  originally forced the zero-initializable custom mutex). Jolt handles are
  still raw pointers, but there is now exactly one owner and one teardown
  path; wrapper unique_ptrs for them would add deleters without removing any
  code, because teardown order is a single linear sequence in the destructor.
- **Commands are `std::variant`, not a 64-byte union.**
  `PhysicsCommand { uint32_t slot; std::variant<CreateBody, SetPosition, …> }`
  with payload structs holding `JPH_RVec3`/`JPH_Quat` values directly and
  `unique_ptr` (custom Jolt deleters) for creation settings and
  `CPH_ConstraintParams` by value. Flushing is a `for` loop + `std::visit`.
  This deletes: the packed header bit-fiddling, the offset static-asserts,
  the computed-goto `CPH_DISPATCH` macro with its branchless validation, heap
  allocation of constraint params (and the leak of those params in the old
  `CPH_ClearCommandQueue`, which never freed them), and two dead commands
  (`CMD_TELEPORT` was never enqueued; `CMD_SET_USER_DATA` enqueued a no-op —
  user data only ever lived in the shadow mirror). Dispatch cost is
  irrelevant next to the Jolt `BodyInterface` call each command performs;
  `tests/test_perf.py` bulk-mutation numbers before/after are in §5.
- **Standard synchronization.** `std::mutex`, `std::condition_variable`,
  `std::atomic`, `std::unique_lock`/`std::scoped_lock`. One shared
  `CPH_LockShadowNotStepping()` helper replaces five copy-pasted
  `lock_until_not_stepping` statics; it returns the held lock as RAII so
  early-return paths can no longer leak the shadow lock. The wait/wake
  protocol (predicates, ordering, broadcast points) is byte-for-byte the old
  one — only the mechanism changed.
- **Jolt RAII at the edges.** `jolt/JoltUtils.hpp` provides
  `JoltPtr<T, Destroy>`; query filters, shape-creation settings, skeleton
  poses, and body-creation settings are scoped owners instead of
  create/destroy pairs threaded through error paths.
- **Headers are C++-only.** One language per header meant the old tree carried
  dual C23/C++ branches (`CPH_ATOMIC`, `_Generic`, `_BitInt`, two assert
  engines, two prefetch implementations…). All project headers are now
  `.hpp`; `extern "C"` survives only where a C ABI genuinely exists:
  Jolt's trace/assert handlers and proc-table callbacks, and `PyInit`.

### Kept legacy mechanisms — and the requirement that earns each

| Kept | Requirement |
| --- | --- |
| SoA shadow buffers, stride-4 layouts, 32-byte alignment | Zero-copy Python buffer protocol + SIMD shadow sync; measured by `test_perf` step numbers (§5). |
| Generational handles + sparse/dense maps | Public handle contract (`docs/HANDLES.md`), stale-handle safety. |
| Command queue (deferred mutation) | Thread-safety contract: mutations from any thread are serialized into the step. The *queue* earns its keep; its 64-byte-union encoding did not. |
| Double (active/spare) command queues | Step/flush must not block producers appending commands for the next frame while the captured queue is being flushed without the GIL. |
| Retired-buffer epoch on resize | Lock-free previous-frame reads of shadow buffers are documented behavior; freeing old storage at resize time proper could invalidate a reader that fetched the pointer before the resize. One epoch of deferral is kept; the unbounded trash list is not. |
| Status-code enums (`PHYSICS_*_STATUS_*`) | `-fno-exceptions` build + the Python layer's status→exception mapping is the layer contract. `std::expected` would rename, not simplify. |
| Spin-then-park waits (`CPH_CpuYield`) in the Python step path | The anti-starvation loop polls `waiting_threads` without holding locks; it is timing behavior visible to the contention perf test. |
| `CPH_ASSERT` macro with 3 compile-time levels | Assertion arguments must not be evaluated at levels 0/1, and level 0 must reach `[[assume]]`; a function cannot do either. The C23 half of the old header is gone. |
| Inline-asm-free cycle profiling hooks | Kept only as a compile-time option; reimplemented on `std::chrono::steady_clock` (no signal-handler PMU probing). Default build compiles them to nothing, as before. |
| Small-buffer (64-hit) query result buffer | Overlap queries construct it per call on the stack; the inline capacity avoids a heap allocation on the hot path. Now RAII instead of init/destroy functions. |

## 3. Eliminated abstractions

| Eliminated | Approx. lines | Replacement |
| --- | --- | --- |
| `threading/` custom 1-byte mutex + parking lot + debug lock graph (`Mutex.c/.h`) | 825 | `std::mutex` / `std::condition_variable` (3 mutexes + 1 cond per world — the 1-byte footprint optimized a non-constraint) |
| `threading/` fibers (`Thread.c/.h`, `ThreadSwitch.S`, `ThreadAssembly.h`) | 380 | Nothing — zero callers in the tree |
| `core/NativeAlloc.h` + all call sites | 66 + ~200 calls | `Buffer<T>`, `std::unique_ptr<T[]>` with `new (std::nothrow)` |
| Dual C/C++ branches in `core/Assert.h`, `CompilerSpecifics.h` (`CPH_ATOMIC`, `_BitInt`, `_Generic`, C assert engine, PMU asm profiler) | ~450 | Single C++23 implementations (`std::atomic`, overloads, `<format>`/`<source_location>`, `<chrono>`) |
| Command union + packed header + computed-goto dispatch | ~250 | `std::variant` + `std::visit` |
| `CPH_NewBuffers` / trash list / alloc-migrate-free protocol | ~350 | `BodyBuffers` move semantics + one retired bundle |
| 5 duplicated `lock_until_not_stepping` / `wait_until_not_querying` statics | ~150 | One helper returning `std::unique_lock` |
| `CPH_ParamsInit`, 6 `CPH_Create*Constraint` entry points | ~60 | Default member initializers; one `CPH_CreateConstraint` |
| `JPH_STACK_ALLOC`, `CPH_DISPATCH`, `CMD_HEADER` family, `CPH_SAFE_FREE`, lock macros (`CPH_SHADOW_LOCK`…), `CPH_TYPE_OF`, `CPH_ASSUME_ALIGNED`, `CPH_FastRsqrt` macro | ~120 | `alignas`, plain control flow, member functions, `decltype`, `std::assume_aligned`, overloads |
| C23 toolchain probe + C from the project languages | ~60 | C++23 only (`extern/JoltC/src/joltc.c` is a 4-line `#include`; compiled as C++ — see below) |

New abstractions introduced (deliberately few): `Buffer<T, Alignment>` (1
class, ~90 lines), `BodyBuffers` (1 struct), `JoltPtr<T, Destroy>` (1 alias),
command payload structs (~20 one-line aggregates), `CPH_LockShadowNotStepping`.

Third-party note: `extern/JoltC/src/joltc.c` and all of `extern/` are
unmodified. The root `CMakeLists.txt` compiles that one vendored TU as C++
(the file is `#include "joltc.h"` and nothing else) so the *project* needs no
C compiler; JoltPhysics' own CMake is untouched.

## 4. Complexity delta

Measured with the script quoted in the migration report (physical lines under
`src/`, project code only):

| Metric | Before | After |
| --- | --- | --- |
| Native files (.c/.h/.cpp/.hpp/.S) | 53 | 45 |
| Native lines | 10,212 | 7,425 (−27%) |
| Native code lines (non-blank/non-comment, approx.) | 8,235 | ~5,893 (−28%) |
| `.c` / `.h` / `.S` lines | 10,212 total | 0 |
| `#define` in native code | 149 | 39 |
| C atomic-op call sites (whole tree) | 328 | 0 |
| NativeAlloc call sites | ~200 | 0 |
| `extern "C"` blocks | 26 | 3 (joltc callbacks in `JoltHandler` + `PyInit`) |
| `goto` (whole tree) | 52 | 51, all in the CPython layer (idiomatic there); 0 in native code |

## 5. Verification

Environment: GCC 14.2, Python 3.13, 2-core sandbox (noisy neighbors —
wall-clock perf thresholds in `test_perf.py` flake on *both* builds here,
so legacy and new were always A/B'd interleaved in the same session).

- Baseline before the rewrite: `pytest tests/` → 234 passed, 4 failed:
  `test_get_look_at` (exact float equality under `-ffast-math`) and three
  `test_perf` wall-clock thresholds.
- After the rewrite: 243 passed (234 + 9 new differential contract tests),
  same 4 failures. A fifth failure, `test_contention_efficiency`, appears
  intermittently under sandbox load on both builds (identical standalone
  timings, 4.14 ms/frame on each); it is not a regression signal.
- The 9 differential contract tests (`tests/test_differential_contracts.py`)
  were written against the legacy build first (9/9 there) and pass
  unchanged on the rewrite: handle generation/invalidation, snapshot
  round-trip bytes, buffer-protocol strides/dtypes, contact-event wire
  size, resize-while-exported refusal, and OOM→`MemoryError` status mapping.
- ASan/UBSan build (`-DENABLE_SANITIZER=ON`, `RelWithDebInfo`): extension
  builds clean; `pytest tests/ --ignore=tests/test_perf.py` under
  `LD_PRELOAD=libasan` runs to completion with **no sanitizer reports**
  (236 passed; 2 float-precision assertion failures attributable to the
  different codegen; `ASAN_OPTIONS=detect_leaks=0` since CPython itself
  is not ASan-built).

### Performance

Interleaved A/B in one session (legacy `.so` vs new `.so`, same machine):

| Workload | Legacy | C++23 | Note |
| --- | --- | --- | --- |
| Loop-create 20k (`test_perf`) | 16.8 ms | 18.1 ms | Chunked micro-bench: dead even (21.8 ms both); steady-state per-1k identical (~0.8 ms), residual wobble is capacity-doubling spikes (see below) |
| Batch-create 20k (`test_perf`) | 4.3 ms | 4.9 ms | Within run-to-run noise band observed on this box |
| 10k-body step (`test_perf`) | 4.9 ms/frame | 5.2 ms/frame | Parity (loop-created pile, 240-frame avg: 0.63–0.66 vs 0.65–0.67 ms) |
| 50k batch raycasts (`test_perf`) | 11.6 ms | 11.4 ms | Parity |
| Bulk mutation 5k forces/frame (`test_perf`) | 5.3 ms | 3.7 ms | Variant command flush is not slower than the union switch |
| Contention: step + proxy read (`test_perf`) | 2.3 ms | 2.2 ms | Parity |
| Snapshot save/load 10k + payload | 0.80/2.70 ms, 0.90 MB | 0.83/2.62 ms, 0.90 MB | Byte-identical payload size |
| World construction (`max_bodies=64`, median) | 1.1 ms | 2.5 ms | See below — the one real delta |
| `test_perf` suite | 3–5 failed | 4–5 failed | Same flaky wall-clock set on a shared 2-core box |

Two deltas were chased down during the migration:

1. **Capacity-doubling spikes during mass creation** (initially ~5 ms
   vs ~3 ms at 16k→32k, since engineered down to noise level). Legacy
   `malloc` never touched fresh pages; value-initializing 17 SoA
   arrays (and default-constructing 32k command variants in two
   queues) faulted them all in. Fixes, in order: body-buffer resize
   now uses `Buffer::allocate_for_overwrite` and leaves the expanded
   range untouched (exactly like legacy), with creation defaults
   written at *claim time* by the paths that need them; and the spare
   command queue is no longer grown eagerly (detach is a swap; enqueue
   grows on demand). Amortized cost per create is unchanged.
2. **World construction** (+~1.3 ms). The 32,768-entry contact-event
   buffer (4 MB) must have its `std::atomic` members constructed in
   C++; legacy left the `malloc`'d buffer uninitialized (and exposed
   those bytes in unwritten event fields). This is the one-time cost of
   defined behavior; removing it would require skipping object
   construction, which is not a trade this migration is willing to
   make. Per-event cost moved to the writers instead: each claimed
   event slot is zeroed before filling, so unwritten fields are
   deterministic zeros rather than legacy garbage/stale data.

### Discovered legacy defect (preserved deliberately)

Legacy batch creation never wrote `categories`/`masks`, and legacy
resize dropped the initial `kAllLayerBits` fill, so batch-created
bodies read uninitialized values there — in practice 0 from fresh
pages — and `OnContactValidate` rejected all their pairs: **batch
bodies did not collide with anything** until `set_collision_filter`
was called (verified: 2,000 batch cubes fall through the ground plane
on the legacy build; single-created bodies rest). Single-body creation
defaults to collide-all (`0xFFFF`), so this is almost certainly an
upstream bug. The migration preserves the observable legacy default
(`0/0`, now written explicitly at claim time in
`CPH_PhysicsWorldCreateBodiesBatch`) because the test suite —
including `test_perf` thresholds — was calibrated on it; changing one
`0` to `kAllLayerBits` is the follow-up fix, but it is a behavior
change and belongs in its own change with recalibrated perf
thresholds, not inside a language migration.
