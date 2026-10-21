# Phase 2 Plan: Direct Jolt C++ API + C++ Interfaces Between Subsystems

Status: **A0, naming dissolution, enum-class sweep, B1, A1, A2,
B2(physics), A3, and B3 landed** (A3 and B3 on top of the lock-order fix
`ee5076d`), plus the `[[likely]]`/`[[unlikely]]` cleanup, which deleted
`CPH::Python::Likely`/`Unlikely` in favour of the standard statement
attributes. **A4 is in progress: the query layer and its filters,
constraints, characters, vehicles and ships are on the direct Jolt API;
ragdolls and soft bodies remain, along with the
`PhysicsWorldRuntime`/`PhysicsWorldCreation` and `JPH_BodyID`/
`JPH_BodyInterface` sites those still touch.** With ships done, all
eight global proc tables are gone.
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
- **Most Jolt handles already *are* the C++ objects.** JoltC's opaque
  pointers are the native pointers (our `ContactListener` relies on
  this today), so migration can be file-by-file with both APIs
  linking against the same `libJolt`. The one exception, discovered
  in A1: `JPH_PhysicsSystem` is a JoltC *wrapper* (three filter
  pointers + a `JPH::PhysicsSystem*`, registered in a private
  native→wrapper reverse map), so a directly owned system can never
  be passed to a `JPH_PhysicsSystem_*` function; every system
  consumer had to convert in A1 rather than alias. JoltC is still
  deleted at the end (Phase A5), not kept as a fallback.
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
| `JPH_JobSystem*`, `JPH_TempAllocator*` | `std::unique_ptr<JPH::JobSystemThreadPool>` (64-byte alignment exceeds what CPython world-object allocation guarantees), `std::unique_ptr<JPH::TempAllocator>` over `TempAllocatorImplWithMallocFallback` |
| `JPH_BodyCreationSettings*` + 10 setter calls | `JPH::BodyCreationSettings` value, public fields; boxed in command payloads (`unique_ptr`) so the 256-byte settings don't set the size of every command |
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
as the consumer idiom), and regularizes the
creation-request structs to all-fields-initialized. Identifiers A0
introduces live in `CPH::` (e.g. `CPH::Stat`, `CPH::Prof*` in
`core/Profiling.hpp`); the tree-wide `CPH_` dissolution is the next,
separate mechanical patch.

**A1 — Core system + stepping. (Done.)** `PhysicsWorld` owns
`JPH::PhysicsSystem` and `JPH::CharacterVsCharacterCollisionSimple`
by value, declared last so member destruction tears the system down
before all body/shape/constraint storage; the job system and temp
allocator are `unique_ptr`-owned natives (the thread pool needs
64-byte alignment CPython allocation cannot promise, and the
allocator keeps JoltC's `ImplWithMallocFallback` overflow behavior).
Stateless `final` layer/filter classes in `PhysicsWorld.cpp` replace
the three Table helpers with the exact fixed 3-layer matrix. Because
`JPH_PhysicsSystem` is a wrapper (see Constraints), A1 converted
*every* system consumer, not just init/step: stepping, gravity,
contact-listener attach, constraint add/remove, body-lock/narrow/
broad query getters, ShadowSync active-body access, debug drawing,
Ship/Vehicle step listeners (Ship's callback now takes the no-lock
body interface from its own world — JoltC's reverse map cannot see a
direct system), `CharacterVirtual` construction + `ExtendedUpdate`
(system default layer filters, world temp allocator), and ragdoll
creation. `body_interface` remains as a JoltC view of
`physics_system.GetBodyInterface()` until A2/B2; no
`JPH_PhysicsSystem_*` call remains anywhere. `g_jph_init_lock` is
**kept**: direct `PhysicsSystem::Init` still reaches process-wide
lazy state that TSan cannot prove race-free across worlds, so
bring-up stays serialized and the comment says why. One deliberate
fix falls out: sanitizer builds now pass 0 worker threads to the
pool directly — JoltC used to translate that 0 into auto-detect, so
the intended synchronous TSan stepping never actually happened.

**A2 — Body creation & shapes. (Done.)** All body/soft-body
creation is direct: `JPH::BodyCreationSettings` /
`JPH::SoftBodyCreationSettings` values configured by field
assignment (`ConfigureBodySettings`/`ApplyBodyCreationProps` keep
their exact semantics, including the scaled mass-properties override
with per-column XYZ inertia scaling), created and added through
`physics_system.GetBodyInterface()`. Shapes are built from native
settings values (`Box/Sphere/Capsule/Cylinder/Plane` primitives,
convex hull, mesh, heightfield, static compound) and held as
`JPH::ShapeRefC`; the shape cache stores `ShapeRefC`, which reduces
`ShapeEntry` to a plain aggregate, and the `owns_shape` destroy dance
in creation is just refcounting now. Command payloads box the
settings in `std::unique_ptr` — at 256 bytes, `BodyCreationSettings`
would otherwise dictate the size of every command slot, and the
allocation count is unchanged from the JoltC heap settings. Boundary
consumers (`Query` shapecast shape, ragdoll part shapes) borrow the
cache-owned shape pointer exactly as before. *Deleted:* the `JoltPtr`
settings/shape aliases, every `JPH_*ShapeSettings_Create/CreateShape`
and `JPH_Shape_Destroy` call in the creation files, and the
settings-destroy error paths that JoltC's never-null `new` made dead.

**A3 — Listeners, debug rendering, logging. (Done.)** The contact
listener is `CPH::WorldContactListener`, a value member of
`PhysicsWorld` constructed with `this` and registered directly with
`PhysicsSystem::SetContactListener`; callbacks now receive
`JPH::Body`/`ContactManifold`/`SubShapeIDPair` natively, so the five
`reinterpret_cast`s in the hot contact path are gone and the 128-byte
event records are written exactly as before. The debug renderer is
`CPH::WorldDebugRenderer : JPH::DebugRendererSimple` (only
`DrawLine`/`DrawTriangle`/`DrawText3D` are overridden; the batch and
`DrawGeometry` machinery is inherited and unused), also a value
member — `BuildDebugData` passes it directly and drops its
`reinterpret_cast`. `jolt/JoltHandler.{hpp,cpp}` is deleted and
replaced by `jolt/JoltLog.{hpp,cpp}`: `InstallJoltLogging()` assigns
`JPH::Trace` (a *variadic* `void(*)(const char*, ...)`, so the bridge
does its own bounded `vsnprintf` into the existing pretty-printer) and
`JPH::AssertFailed` under `JPH_ENABLE_ASSERTS`. Both assignments must
happen **after** `JPH_Init()`, which sets the same two pointers
itself; A5 deletes `JPH_Init()` and turns this into the only
initialization path. `Jolt.hpp` now also `#error`s when
`JPH_DEBUG_RENDERER` is missing, since debug draw is a shipped API.

*Scope note (deviation from the phase text):* the four query-filter
proc tables stay in `Filters.hpp` until A4. JoltC's filter wrappers
are not standalone — `JPH_*Filter_Create` objects can only be consumed
by `JPH_*Query_*` functions — so converting them forces the whole
query layer (cast/collide collectors, `NarrowPhaseQuery`,
`BroadPhaseQuery`, shapecast) to the direct API in the same patch,
which is A4's work item, not this one's. A3 therefore deletes the
contact, debug-renderer, and both `extern "C"` blocks, leaving
`PyInit` as the only C-linkage symbol besides the character-contact
and step-listener proc tables, which belong to the files A4 rewrites.

**A4 — Queries, constraints, characters, vehicles, ragdolls, soft
bodies** (largest call-site count; vehicles alone are 171).
Collectors replace query out-arrays (wire formats preserved at the
binding edge); constraint/character/vehicle/ragdoll settings become
typed values; `ConstraintSlot` holds `Ref<Constraint>`. *Deletes:*
the per-type `CPH_CreateConstraint` setter switch in favor of
settings structs, character/vehicle proc structs.

*Landed so far — queries and filters.* The query layer no longer
touches JoltC. `Filters.hpp` is deleted outright: its four proc tables
existed only to reproduce Jolt's own defaults, and every
`JPH::*Filter::ShouldCollide` already returns `true`, so "allow all"
is now a default-constructed base class and the single-body ignore is
`JPH::IgnoreSingleBodyFilter`. `Query.cpp` and `InternalQuery.cpp`
call `NarrowPhaseQuery::{CastRay, CollideShape, CastShape}`,
`BroadPhaseQuery::CollideAABox`, `BodyLockRead` and
`BodyInterface::GetUserData` directly, and the two C overlap callbacks
plus the shapecast collector callback are now `CollisionCollector`
subclasses that stream into the caller's `PhysicsQueryHitBuffer`.
That removes two or three heap-allocated JoltC filter objects per
query. `UnifiedBodyFilter`, `CastShapeContext`, `CastShapeFilter` and
`CastShape_ClosestCollector` are gone, and the query entry points
(`PhysicsWorldState.hpp`) and `QueryMethods.cpp` speak `JPH::RVec3`,
`JPH::Vec3` and `JPH::Quat` instead of the JoltC value types.

*Landed so far — constraints.* The constraint subsystem no longer
touches JoltC either. `ConstraintFactory.cpp` builds native
`{Fixed,Point,Hinge,Slider,Cone,Distance}ConstraintSettings` and calls
`Settings::Create(Body &, Body &)`, which replaces the six
`JPH_*ConstraintSettings_Init` + `JPH_*Constraint_Create` pairs and the
`(JPH_Constraint *)` downcast on every return path. Two things got
strictly simpler: `mEnabled` and `mSpace` already default to `true` and
`WorldSpace` in Jolt's settings structs, so eight assignments that only
restated a default are gone; and `mFrequency`/`mDamping` are real
members instead of JoltC's `frequencyOrStiffness` union tag. The
axis sanitisation and the hinge's pivot-to-body-2 normal projection
are unchanged arithmetic — they are what make `angle = 0` mean "body 2
at its initial position", and they stay bit-identical so recorded-scene
replays do not drift. `ConstraintSlot::constraint` is now
`JPH::Constraint *`; `JPH_Constraint_Destroy` becomes
`Constraint::Release()`, `JPH_Constraint_GetType/GetSubType` become
virtual `GetType()/GetSubType()`, and the motor retargeting path calls
`HingeConstraint::SetTargetAngle` / `SliderConstraint::SetTargetPosition`
directly instead of through `reinterpret_cast`. The `CreateConstraint`
command handler locks bodies with RAII `BodyLockWrite`, still one body
per scope for the same EDEADLK reason, and passes the constraint to
`AddConstraint` without a cast.

*Landed so far — characters.* The character bindings no longer touch
JoltC. The largest win is structural: `kCharListenerProcs` is gone, and
with it the last character-side proc table. JoltC's
`JPH_CharacterContactListener_SetProcs` installed *one process-wide*
procs pointer that every character shared, with each callback recovering
its owner from a `void *userData` — which is why creating a listener had
to be serialized behind the trampoline lock. `CharacterListener` is now
a `JPH::CharacterContactListener` subclass owned per character, so there
is no shared state left to serialize and `setup_char_filters` no longer
takes the lock at all. The four heap-allocated JoltC filters go with it:
`body_filter` and `shape_filter` become default-constructed stack locals
in `CharacterMove` (`ManagedBodyFilter::ShouldCollide` falls through to
`return true` when no procs are installed, so a default `JPH::BodyFilter`
is the exact equivalent), and `bp_filter`/`obj_filter` are deleted
outright — they were constructed and destroyed but never passed to
anything, since `ExtendedUpdate` already takes the system's default
broad-phase and layer filters. `CharacterObject::character` is
`JPH::CharacterVirtual *`, which retires the
`reinterpret_cast<JPH_CharacterBase *>` on every ground-state read; the
`alignas(alignof(JPH_RVec3))` out-parameter staging disappears with the
accessors, and the `const_cast<JPH_Body *>` calls in the velocity
callback are gone because `Body::GetUserData`, `GetAngularVelocity` and
`GetPosition` are all `const`. `alloc_j_char` builds a
`CapsuleShapeSettings` on the stack and checks `ShapeResult::IsValid()`
instead of juggling `JPH_ShapeSettings_Destroy`/`JPH_Shape_Destroy`
around a raw pointer, and `DegreesToRadians` replaces the hand-rolled
`JPH_M_PI / 180.0f` (identical constant, identical expression).
`MathInterpolateCharacterTransform` takes `const JPH::RVec3 &` and
`const JPH::Quat &`.

*Landed so far — wheeled and tracked vehicles.* `Vehicle.cpp`,
`TrackedVehicle.cpp` and `Vehicle.hpp` no longer touch JoltC: 238 JoltC
lines out, net −150. The largest win is lifetime, not syntax.
`VehicleResources` and its `CleanupVehicleResources` teardown walk are
gone entirely, along with five ownership pointers on `VehicleObject`.
They existed only to free the settings objects later, but nothing needs
to: `Wheel` holds a `RefConst<WheelSettings>`, so wheel settings live as
long as the constraint does, and
`WheeledVehicleControllerSettings`/`TrackedVehicleControllerSettings`
hold `VehicleEngineSettings` and `VehicleTransmissionSettings` *by
value*, copied at construction. `LinearCurve` is likewise a value type,
so the shared friction/torque curves become locals. Everything is
`Ref<>`-counted and released on scope exit, which also collapses the
`goto jolt_fail` unwind into one `jolt_created_fail` path that only has
to undo `AddConstraint`/`AddStepListener`. `AsPhysicsStepListener`
disappears because `VehicleConstraint` already derives from
`PhysicsStepListener`. Both `const_cast<JPH_VehicleTransmission *>`
calls go away: `GetController()` and `GetTransmission()` have non-const
overloads, and `VehicleTransmission` derives from
`VehicleTransmissionSettings`, which is how `SetMode` ever worked — it
writes `mMode` on the runtime object. The two wheel-transform getters
shrink considerably: `RMat44::GetQuaternion()` already rotates only the
3×3 part, which is what the code was doing by hand (copy three columns
into a `Mat44`, zero the translation, convert), and `alignas(alignof(...))`
out-parameter staging goes with it.

Three pre-existing defects were found and **deliberately preserved**,
each marked `TODO` at the site:

- `create_vehicle(transmission=...)` has never had any effect on a
  wheeled vehicle. The code builds a standalone transmission settings
  object, configures mode/clutch/gear ratios/differential ratio on it,
  stores it on the wrapper, and only ever frees it — it is never copied
  into the controller settings (joltc's
  `JPH_WheeledVehicleControllerSettings_SetTransmission` exists and is
  never called). The controller therefore runs on Jolt's defaults: Auto
  mode and ratios `{2.66, 1.78, 1.3, 1.0, 0.74}`, not the six built here.
  Wiring it up is one assignment and a behaviour change. The tracked
  path does *not* have this bug — it writes straight through to
  `mTransmission`.
- `WorldCreateVehicle` takes a world reference with `Py_NewRef` and then
  calls `Py_INCREF(self)` again, while `VehicleDealloc` releases one.
  Each wheeled vehicle leaks a world reference; the tracked constructor
  does not do this.
- `TrackedVehicleController::SetDriverInput` asserts
  `inLeftRatio != 0 && inRightRatio != 0`, but `set_tank_input(0, 0, 0)`
  reaches it. Release builds compile the assert out, so it has never
  fired.

Also preserved and marked: `kVehicleCollisionTesterScale = 2.0f` is
named "scale" but is passed as the tester's max slope angle in radians
(~114.6°, i.e. effectively unconstrained); steering is inferred from the
sign of a wheel's local Z, so a −Z layout gets no steering wheels;
`mMaxSteerAngle` is written twice in `create_single_wheel`; and
`pack_transform` leaks its seven `PyFloat_FromDouble` references because
`PyTuple_Pack` already increfs.

*Landed so far — ships, and with them the last global proc table.*
`Ship.cpp` no longer touches JoltC's listener machinery. JoltC's
`JPH_PhysicsStepListener_SetProcs` installed **one process-wide
`s_Procs` pointer** shared by every ship in every world, with `OnStep`
recovering its owner from a `void *userData`. That is the last instance
of the pattern in the tree (`kCharListenerProcs` went with characters),
and it is replaced by `ShipStepListener : public JPH::PhysicsStepListener`
in an anonymous namespace holding a `ShipObject *`. Each ship now owns
its listener outright, so the shared trampoline table, the C callback
and the `reinterpret_cast` on both ends are gone, and
`JPH_PhysicsStepListener_Destroy` becomes a plain `delete`.

`Ship.hpp` keeps `#include "joltc.h"` deliberately: `sled_bid` is still a
`JPH_BodyID`, which is `typedef uint32_t` — the raw packed
index-and-generation word, not a Jolt type. That field belongs to the
body-ID slice, not this one. Constructing `JPH::BodyID(self->sled_bid)`
at the use site is exactly what joltc did internally
(`joltc.cpp:6285-6288`), so the conversion is unchanged.

Two things were preserved on purpose rather than "improved":

- `jph_trampoline_lock` is still taken around
  `AddStepListener`/`RemoveStepListener` even though the proc table it
  existed for is gone. It is part of the documented acquisition order
  `jph_trampoline_lock -> shadow_lock -> step_sync`
  (`PhysicsWorldState.hpp:370-373`); dropping it is a concurrency change
  nobody asked for. `PhysicsSystem::AddStepListener`/`RemoveStepListener`
  take `mStepListenersMutex` themselves, so this is a lock we hold on top,
  not a missing one.
- Both dot products and the drag division stay as explicit scalar
  arithmetic instead of `Vec3::Dot` and `operator/`. `Dot` is SIMD and
  reorders the adds, and Jolt's scalar `operator/` multiplies by a
  reciprocal instead of dividing. This runs every step, so either would
  drift recorded-scene replays. Marked at the site.

`OnStep` runs with every body mutex held, so it uses
`GetBodyInterfaceNoLock()` — the locking interface would deadlock. That
was true before and is now commented rather than implied.

Remaining A4 work: `PhysicsWorldRuntime`/`PhysicsWorldCreation`, the
ragdoll and soft-body bindings, and the `JPH_BodyInterface` field those
still read.

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

**B1 — One error model. (Done.)** Adopt the `CPH::Error` foundation
(`src/core/Error.hpp`, landed in A0). The 7 status enums are now
7 error-only categories in `CPH::` (`WorldError`,
`WorldCreationError`, `WorldRuntimeError`, `WorldSnapshotError`,
`QueryError`, `RagdollError`, and `BodyOperationError`): the
`OK = 0` enumerator is deleted and the negative values are renumbered
positive (0 means "no error" in every category). Native fallible
functions return `std::expected<T, CPH::Error>` /
`std::expected<void, CPH::Error>`; the creation API's dual "return
enum + `out_native_status`" pattern is gone because lower-level
errors propagate with their original category intact. The Python
layer's domain translators take `CPH::Error` and key on
`error.Code().Is<…>()` / `.As<…>()`; their existing exception types
and messages, including caller- and world-specific context, remain
the Python-visible contract. Foreign C result domains (JoltC
results, errno-style values) register via
`CPH::RegisterErrorCategory` instead of growing a fake enum.

**B2 — Native API signatures. (Physics done.)** `std::span` for
array inputs/outputs, references where null is not a state, no
`*out_` parameters inside the native tree (~36 today),
`[[nodiscard]]` on all fallible ops. `CPH_PhysicsWorld` free
functions become member functions where the world is the obvious
subject (keep free functions that are genuinely algorithms over
spans, e.g. `ShadowSync`). By the time B2 runs, the naming
dissolution (below) has made `CPH::` the single convention, so new
signatures are written directly in it. The physics pass moved every
world operation onto `PhysicsWorld` (declared in
`PhysicsWorldState.hpp`): creations return
`expected<BodyHandle, Error>`, always-full buffers
(`TakeContactEvents`, `SaveSnapshot`, `GetRenderState`, soft-body
vertices, `BuildDebugData`) are returned by value, only the
prefix-filled `GetActiveIndices` keeps a workspace buffer and
returns its valid count, `UnpackHandle` is a const
`std::optional<std::uint32_t>`, and snapshot load returns the
restored body count with `ReadSnapshotHeader` exposed for the
binding's capacity-mismatch message. What stayed free, on purpose:
`SyncShadowBuffers` (restrict-qualified hot algorithm), the
settings-value helpers (`ConfigureBodySettings` /
`ApplyBodyCreationProps`), `PhysicsWorldCreateRagdollSettings`
and the ragdoll skeleton/settings/drive functions (their subject is
not a world; they move onto direct Jolt types in A4),
`PhysicsRagdollDeallocate` (null world is a real state when the
owning Python object dies first), `FindInvalidMeshTriangle` (pure
span scan the binding pre-checks so its error can name the bad
triangle), and `ShapecastExecuteInternal`. Query vector parameters
still used the JoltC `JPH_RVec3`/`JPH_Vec3` value types when this
landed; A4's query slice moved them, and `ShapecastExecuteInternal`
with them, onto `JPH::RVec3`/`JPH::Vec3`/`JPH::Quat`. B2 for the
remaining subsystems rides with B4.

**B3 — Python boundary hygiene (landed).** `Ref` (owning RAII,
`Py_XDECREF` in dtor) and `Borrowed` (non-owning, debug-checked
dereference) now live in `CPH::Python` (`PythonCommon.hpp`); `Ref`
subsumes the old `OwnedPyObject`, and its `Ref(PyObject*)` /
`Ref::New(Borrowed)` split states the adopt-vs-copy distinction that
the raw C API leaves to comments. Parsers: `int ParseVec3F32(obj, &x,
&y, &z)` → `ParseResult parse_vec3_into(Borrowed, float&, float&,
float&)` plus value forms `parse_vec3` / `parse_position` /
`parse_rotation` returning `expected<Vec3f|PosStride|AuxStride,
ParseError>`; `ParseError` owns the exact exception instance the
diagnostic builder produced, and `RaiseParseError` reinstates it
verbatim at the boundary — `test_diagnostics.py` is the oracle that
pins this.

Two deliberate deviations from the shape sketched above:

1. *The converter contract stays in-place.* `FastArgSpecHot::convert`
   is still `bool (PyObject *, const void *)` and
   `FastParser<Ts...>::Parse` still returns `ParseResult`
   (`expected<void, ParseError>`) rather than
   `expected<tuple<Ts...>, ParseError>`. The engine is type-erased
   behind a function pointer and writes into caller-owned argument
   slots; returning values would add a move per argument to the
   hot path (the 64-argument stress parser is the extreme case). The
   `_into` spelling is therefore the canonical one and the
   value-returning forms exist for non-hot call sites.
2. *`goto` count 51 → 31, not 0.* All 20 `goto`s in
   `FastParserCore.cpp` (the diagnostic builders' `basic_error` /
   `oom_fallback` / `cleanup` ladders) are gone — `Ref` replaced them
   wholesale. The remaining 31 are resource-cleanup labels
   (PhysicsWorld 9, Module 6, TrackedVehicle 5, Vehicle 4, Character 4,
   Ecs 2, CommandBuffer.hpp 1): `python_fail` / `jolt_fail` / `fail_py`
   ladders that unwind Jolt allocations, JoltC handles and locks. Their
   unwind blocks stop being needed when A4 removes the JoltC handles and
   B4 wraps the ECS placement-new storage.

Behaviour preserved verbatim, including three pre-existing quirks that
a diagnostics change would have "fixed" by accident: `fast_parse_process_pos`
and `fast_parse_process_kw` decorate a conversion error with the
signature context, while the monomorphic stubs and the two legacy
helpers return the converter's raw error (so they still do);
`parse_sequence_internal` treats `None` as a failure that
reports no exception; and `Ragdoll.Create`'s malformed-`pos` path
carries on with the error pending. Each is marked in the source.
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
within noise of the previous phase. TSan build re-run at A1, A3, and
A4 (the locking-sensitive phases; the contact listener and the debug
renderer both run inside the step).

Main risks, with mitigations already built into the phases:
silent ABI drift via defines (A0 asserts), `Ref<>` lifetimes inside
the command queue (A2 measurement gate), filter-matrix transcription
errors when hand-writing the layer tables (A1: port the exact matrix,
covered by contact/differential tests), and FastParse message drift
(B3: diagnostics tests as oracle — landed, see the B3 section for the
three pre-existing diagnostic quirks it deliberately reproduces).
Version coupling does not
increase: JoltC was already version-locked to the same Jolt pin.
