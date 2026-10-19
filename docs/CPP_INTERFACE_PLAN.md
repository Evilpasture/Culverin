# Phase 2 Plan: Direct Jolt C++ API + C++ Interfaces Between Subsystems

Status: **A0 + naming dissolution landed** (A0 on top of `601b1a2`
"convert all to C++", including the `CPH::Error` foundation header;
the `CPH_` → `CPH::` dissolution on top of `07d6f5d`); B1 next.
Prerequisite: the C++23 migration
(`docs/CPP23_MIGRATION.md`) has landed. All acceptance oracles from that
migration apply unchanged to every phase below: full test suite at
243 passed / 4 pre-existing failures, 9/9 differential contract tests,
ASan clean, and interleaved perf A/B (step, loop/batch create, bulk
mutation, raycast, snapshot) within noise of the previous phase.

Two C surfaces remain after the C++23 migration:

1. **JoltC** — all Jolt access goes through the C wrapper
   (`extern/JoltC`, `joltc.h`), even though the code is now C++ and
   already links `libJolt` directly.
2. **C-style internal ABIs** — subsystem boundaries still speak
   `int`/`enum` statuses, out-parameters, and raw pointers
   (physics ↔ Python bindings ↔ parsers ↔ adapters).

This plan removes both, in independently landable phases, with no
permanent dual paths: each phase deletes what it replaces.

## Inventory (measured on the migrated tree)

| Surface | Size |
| --- | --- |
| Distinct `JPH_` symbols used | 509 |
| JoltC call sites | ~580 (physics 317, vehicles 171, characters 58, physics bindings 23, bindings 10) |
| Files including `joltc.h` | 18 |
| Proc tables registered in `Module.cpp` (+`Ship.cpp`) | 8 (contact, character-contact, debug renderer, 4 query filters, step listener) |
| JoltC-only helper *implementations* we rely on | 4 (BroadPhaseLayerInterfaceTable, ObjectLayerPairFilterTable, ObjectVsBroadPhaseLayerFilterTable, CharacterVsCharacterCollisionSimple) |
| Translation units already using `JPH::` directly | 4 (`ShadowSync` — the hottest path, `ContactListener`, `Math`, `JoltRttiShim`) |
| Status enums in native API | 7 (`PhysicsWorldStatus`, `…CreationStatus`, `…RuntimeStatus`, `…SnapshotStatus`, `PhysicsQueryStatus`, `PhysicsRagdollStatus`, `PhysicsBodyOperationStatus`); creation functions return one enum *and* write a second through `out_native_status` |
| `*out_` parameters in native headers | ~36 |
| Parser layer | 2,062 lines; `int Parse*(PyObject*, float*, float*, …)` signatures; 20 `goto`s in `FastParserCore.cpp` |
| Manual `Py_XDECREF`/`Py_DECREF` | scattered (16 in `physics/PhysicsWorld.cpp` alone); no RAII ref type exists yet |
| Remaining `extern "C"` | 3 (the joltc callback registrations + `PyInit`) |

## Constraints that shape the design

- **`-fno-exceptions` everywhere.** Jolt's C++ API is exception-free by
  design and already compiles in our TUs (`ShadowSync` proves it).
  `std::expected<T, CPH::Error>` works without exceptions and is the
  error channel for Workstream B.
- **ABI defines must stay uniform**: `JPH_DOUBLE_PRECISION`,
  `JPH_OBJECT_LAYER_BITS=32` (already global `add_definitions`).
  Direct API use makes getting this wrong an ODR/layout bug instead
  of a link error, so Phase A0 moves the defines onto the Jolt CMake
  target as `PUBLIC` compile definitions and static-asserts
  `sizeof(JPH::RVec3)`, `JPH::ObjectLayer` width, etc. in one place.
- **Jolt handles already *are* the C++ objects.** JoltC's opaque
  pointers are the native pointers (our `ContactListener` relies on
  this today). Migration can therefore be file-by-file with both APIs
  linking against the same `libJolt` — but JoltC is deleted at the
  end (Phase A5), not kept as a fallback.
- **RTTI** (resolved in A0): Jolt was already built with its own
  `CPP_RTTI_ENABLED=ON` option, and nothing in Culverin uses
  `typeid`/`dynamic_cast` on Jolt types. `JoltRttiShim.cpp` was
  vestigial and is deleted (link + full suite verified without it).
- **Wire/behavior contracts are untouched**: Python API, buffer
  strides, handle format, snapshot bytes, 128 B contact events,
  FastParse diagnostic strings (`tests/test_diagnostics.py` pins
  them), and the batch-create `0/0` category default discovered in
  Phase 1.

## Workstream A — Jolt C++ API directly

Handle mapping (representative):

| Today (JoltC) | Direct C++ |
| --- | --- |
| `JPH_PhysicsSystem*` + `JPH_PhysicsSystem_Create/Destroy` | `JPH::PhysicsSystem` member (value) |
| `JPH_BodyInterface*` | `JPH::BodyInterface&` from the system — never null, never owned |
| `JPH_JobSystem*`, `JPH_TempAllocator*` | `JPH::JobSystemThreadPool`, `JPH::TempAllocatorImpl` members |
| `JPH_BodyCreationSettings*` + 10 setter calls | `JPH::BodyCreationSettings` value, public fields |
| `JPH_Shape*`/`JPH_ShapeSettings*` + manual `_Destroy` | `JPH::RefConst<JPH::Shape>`; settings values, `->Create()` |
| `JPH_Constraint*` | `JPH::Ref<JPH::Constraint>`; per-type settings values |
| `…Table_Create` filters | ~80 lines of own `final` classes (fixed 3-layer matrix as `constexpr` tables; stateless, can be function-local statics — the ownership-transfer dance in `~CPH_PhysicsWorld` disappears) |
| 8 proc tables | `final` subclasses: `JPH::ContactListener`, `JPH::DebugRenderer`, `JPH::BroadPhaseLayerFilter`/`ObjectLayerFilter`/`BodyFilter`/`ShapeFilter`, character contact listener, `JPH::PhysicsStepListener` |
| Query free functions | `JPH::NarrowPhaseQuery` with stack collectors |

**A0 — Groundwork. (Done.)** Umbrella header `src/jolt/Jolt.hpp`
(`<Jolt/Jolt.h>` first, ABI `#error`/`static_assert` pins) adopted by
the four TUs already using `JPH::`. The three global `-D`s
(`JPH_DOUBLE_PRECISION`, `JPH_OBJECT_LAYER_BITS=32`,
`JPH_DEBUG_RENDERER`) are gone: `OBJECT_LAYER_BITS=32` is now forced
as a Jolt build option alongside `DOUBLE_PRECISION`, and Jolt exports
all three as PUBLIC definitions on `Jolt::Jolt`, which every Culverin
target reaches via `culverin_native_support` (verified: all 180
compile lines carry exactly one value per macro — previously the
global `=32` silently fought Jolt's own `=16` on the same command
lines). The prebuilt-Jolt branch states the consumer-side ABI
explicitly. `JPH_Init()` is **kept for now**: the vendored JoltC is a
Culverin-patched fork whose init also installs trace forwarding and
a global temp-allocator policy, so replacing it belongs to A5 with a
fork-delta audit, not to A0. *Deleted:* `JoltRttiShim.cpp` + target
(vestigial; see RTTI note above). A0 also lands `src/core/Error.hpp` —
the `CPH::Error`/`ErrorCode`/`ErrorCategory` foundation for B1
(enum-typed categories, 0 = no error, `std::expected<T, CPH::Error>`
as the consumer idiom) — adopted by nothing yet, and regularizes the
creation-request structs to all-fields-initialized. Identifiers A0
introduces live in `CPH::` (e.g. `CPH::Stat`, `CPH::Prof*` in
`core/Profiling.hpp`); the tree-wide `CPH_` dissolution is the next,
separate mechanical patch.

**A1 — Core system + stepping** (`PhysicsWorldState.hpp`,
`PhysicsWorld.cpp`, `ShadowSync` untouched in behavior).
`CPH_PhysicsWorld` owns `PhysicsSystem`/`JobSystemThreadPool`/
`TempAllocatorImpl` by value; own layer/filter classes replace the
three Table helpers; `CharacterVsCharacterCollisionSimple` member.
*Deletes:* `JPH_PhysicsWorldInitJolt`'s create/destroy plumbing, the
dtor's ownership-transfer branches, `g_jph_init_lock` rationale
re-checked against direct `PhysicsSystem::Init` (keep the lock if
Jolt lazy statics still need it under TSan — verify, document).

**A2 — Body creation & shapes** (`PhysicsWorldCreation`,
`CommandBuffer`, shape cache). Settings become values;
`CPH_ConfigureBodySettings` becomes field assignment on a
`JPH::BodyCreationSettings&`. Shape cache stores `RefConst<Shape>`.
Command payloads: hold `BodyCreationSettings` by value if the
variant size/perf A/B allows (measure flush-heavy bulk test), else
`std::unique_ptr`. *Deletes:* `JoltPtr` settings deleters,
`CPH_ParamsInit`-era remains in creation, shape `_Destroy` loops.

**A3 — Listeners, filters, debug rendering** (`ContactListener`,
`Filters.hpp`, `DebugRender`, `JoltHandler`). All 8 proc tables
become subclasses; `JoltHandler.{hpp,cpp}` is deleted wholesale.
Contact listener loses its `reinterpret_cast`s (callbacks receive
`JPH::Body&` natively). *Deletes:* every remaining `extern "C"`
except `PyInit`, `contact_procs`, `kGlobal*Procs`.

**A4 — Queries, constraints, characters, vehicles, ragdolls, soft
bodies** (largest call-site count; vehicles alone are 171).
Collectors replace query out-arrays (wire formats preserved at the
binding edge); constraint/character/vehicle/ragdoll settings become
typed values; `ConstraintSlot` holds `Ref<Constraint>`. *Deletes:*
the per-type `CPH_CreateConstraint` setter switch in favor of
settings structs, character/vehicle proc structs.

**A5 — Remove JoltC.** First, audit the fork: `extern/JoltC` carries
Culverin-local patches (at minimum `JPH_Init`'s trace forwarding and
global `s_TempAllocator` with its TSan fallback — diff against
upstream joltc at the pinned base to enumerate all of them), and port
whatever is still needed. Replace `JPH_Init()` with explicit
`RegisterDefaultAllocator` + trace/assert hooks + `Factory` +
`RegisterTypes`. Then delete the `extern/JoltC` submodule,
`joltc_double` target, the `joltc.c`-compiled-as-C++ CMake hack,
`src/jolt/` leftovers. Extension links `Jolt` only. Re-run metrics;
expect ~500 wrapper call sites gone and one less vendored dependency
whose release cadence we no longer track (JoltC lags Jolt releases;
the submodule pin becomes the single version source).

## Workstream B — C++ interfaces between subsystems

Right after A0 comes the mechanical naming patch (below), then B1
(it is Jolt-independent), then B2–B4 subsystem by subsystem;
physics-internal signatures should land *after* the matching A phase
so they can mention `JPH::` types directly
(e.g. `expected<JPH::RefConst<JPH::Shape>, CPH::Error>`).

**B1 — One error model.** Adopt the `CPH::Error` foundation
(`src/core/Error.hpp`, landed in A0). The 7 status enums become
7 error categories: tidy `CPH::` names (`CPH::WorldStatus`, …),
the `OK = 0` enumerator deleted and the negative values renumbered
positive (0 means "no error" in every category). Native functions
return `std::expected<T, CPH::Error>` / `std::expected<void,
CPH::Error>`; the creation API's dual "return enum +
`out_native_status`" pattern disappears (one `CPH::Error` carries its
category with it). The Python layer translates `CPH::Error` →
exception in exactly one function per module, keying on
`error.Code().Is<…>()` / `.As<…>()` and using `error.Message()` where
today's messages are hand-written strings (Python-visible mapping
stays bit-exact; per-enum `ErrorMessage(E)` ADL hooks provide the
human text where the enumerator name is not the message). Foreign
C result domains (JoltC results, errno-style values) register via
`CPH::RegisterErrorCategory` instead of growing a fake enum.

**B2 — Native API signatures.** `std::span` for array inputs/outputs,
references where null is not a state, no `*out_` parameters inside
the native tree (~36 today), `[[nodiscard]]` on all fallible ops.
`CPH_PhysicsWorld` free functions become member functions where the
world is the obvious subject (keep free functions that are genuinely
algorithms over spans, e.g. `ShadowSync`). By the time B2 runs, the
naming dissolution (below) has made `CPH::` the single convention,
so new signatures are written directly in it.

**B3 — Python boundary hygiene.** Introduce `Ref` (owning RAII,
`Py_DECREF` in dtor) and `Borrowed` (non-owning, debug-checked)
in `CPH::Python` (`PythonCommon.hpp`); migrate bindings per file — the 51 remaining
`goto`s (20 in `FastParserCore.cpp`) are the cleanup idiom this
replaces. Parsers: `int Parse(obj, &x, &y, &z)` →
`expected<Vec3, ParseError> parse_vec3(Borrowed obj)`, FastParser
traits return `expected<T, ParseError>`; `ParseError` materializes
into the *identical* Python exception (type + message) at the call
boundary — `test_diagnostics.py` is the oracle that pins this.
Vectorcall fast paths stay; measure `test_fastparse_*` A/B.

**B4 — Adapters/ECS/math/debug.** `WorldAdapters`, `PhysicsSync`,
ECS registry, and the math service adopt the same patterns
(expected/span/Ref); ECS's placement-new mutex (a Phase-1 wart forced
by CPython allocation) gets wrapped in a small `PyObject`-storage
helper so the pattern is written once, correctly, instead of being
re-derived per type.

## Naming: one root, `CPH::`

`CPH` is the project's only namespace root (`CPH::Python` for the
whole Python layer; `CPH::cmd` stays as is). The `CPH_` prefix
dissolves into `CPH::` tree-wide in one mechanical patch immediately
after A0 — tidy names (`CPH_PhysicsWorldStatus` → `CPH::WorldStatus`,
`CPH_BodyConfig` → `CPH::BodyConfig`, functions likewise). What
keeps `CPH_`: ALL_CAPS macros (preprocessor has no namespaces) and
genuine C-linkage names (the JoltHandler callback surface until A3
deletes it; `PyInit`). Measured size: 181 distinct non-macro
identifiers, ~1,430 reference sites, 62 files; nothing outside the
C++ tree references these names (one stale comment in
`tools/gen_stubs.py`). Doing it before B1/A1 means every later phase
writes new signatures in final form instead of churning twice.

## Sequencing, gates, risks

Order: **A0 → naming dissolution → B1 → A1 → A2 → B2(physics) → A3
→ B3 → A4 → B2/B4(rest) → A5.** Each phase is one patch, gated on: full suite 243/4,
differential 9/9, ASan clean, and the interleaved perf table (loop &
batch create, 10k step, bulk mutation, raycast, snapshot, ctor)
within noise of the previous phase. TSan build re-run at A1 and A3
(the locking-sensitive phases).

Main risks, with mitigations already built into the phases:
silent ABI drift via defines (A0 asserts), `Ref<>` lifetimes inside
the command queue (A2 measurement gate), filter-matrix transcription
errors when hand-writing the layer tables (A1: port the exact matrix,
covered by contact/differential tests), and FastParse message drift
(B3: diagnostics tests as oracle). Version coupling does not
increase: JoltC was already version-locked to the same Jolt pin.
