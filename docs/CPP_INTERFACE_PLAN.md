# Phase 2 Plan: Direct Jolt C++ API + C++ Interfaces Between Subsystems

Status: **A0, naming dissolution, enum-class sweep, B1, A1, A2,
B2(physics), A3, and B3 landed** (A3 and B3 on top of the lock-order fix
`ee5076d`), plus the `[[likely]]`/`[[unlikely]]` cleanup, which deleted
`CPH::Python::Likely`/`Unlikely` in favour of the standard statement
attributes, plus a compiler-support sweep that removed the remaining
obsolete `CPH_*` macros: `kJphInvalidBodyId` and `JPH_ID_TO_INDEX`
duplicated `JPH::BodyID::cInvalidBodyID`/`cMaxBodyIndex` and are gone in
favour of Jolt's own constants, `CPH_LIKELY`/`CPH_UNLIKELY` are now
`[[likely]]`/`[[unlikely]]` statement attributes, `Unreachable()` is
`std::unreachable()`, `CPH_NO_TSAN` is spelled as a C++ attribute, and
`CPH_UNROLL_LOOP` is replaced by `Unroll`/`UnrollLoop`/`Repeat`
index-sequence templates. **A4 is in progress: the query layer and its
filters,
constraints, characters, vehicles, ships, ragdolls and soft bodies are on
the direct Jolt API, and the body-operation paths (immediate and deferred)
are off `JPH_BodyInterface` too, and so are the ragdoll, buoyancy,
activation, motion-type and constraint-wake paths. The only
`JPH_BodyInterface_*` calls left anywhere are the four inside
`VerifyAbiAlignment`, which stay on the C boundary on purpose -- that
function is the startup canary that detects a JoltC/extension precision
mismatch, and rewriting it natively would make it unable to fail. It goes
when JoltC goes.** The shadow buffer's element type is now `JPH::BodyID`
rather than a raw uint32, so the pack/unpack round-trips, the
`ActivateBodies` reinterpret_cast and `FlushContext::BodyId()` are all
gone. The `JPH_BodyLockInterface_*` / `JPH_Body_*` surface that the
soft-body read path sat on is gone too: `lock_soft_body_for_read` became
`WithSoftBodyRead`, a scoped callback, because `JPH::BodyLockRead` is
neither movable nor default-constructible and so cannot be returned the
way JoltC's placement-new'd byte-blob could be. That also removed the six
hand-written `UnlockRead` calls. The C value types are gone too: every
command payload, body-creation request and body operation now holds a
native `JPH::Vec3`/`JPH::RVec3`/`JPH::Quat`, the `ToJolt` shim that
bridged the two had no callers left and is deleted, and the soft-body
local-vertex buffer moved off `JPH_Vec3` because that buffer is copied
straight into a Python `bytes` object and `JPH::Vec3` is 16 bytes with
padding. It is `JPH::Float3` now: Jolt already ships the tight, unpadded
`Float2`/`Float3`/`Float4` storage types, so there is no need for a
project-local `CPH::Vec3f`, which is deleted. `sizeof(JPH::Float3) == 12`
is pinned by a `static_assert` next to `GetSoftBodyLocalVertices` because
that size is the wire format. With ships done, all eight global proc
tables are gone.

**A5 is done: JoltC is gone.** A follow-up sweep finished the
standardisation it started: the project-local `JPH_Real` typedef and its
`JPH_REAL_CHAR`/`JPH_REAL_STRING` macros are replaced by `JPH::Real` plus the
typed `CPH::kRealFormatString` (`JPH_REAL_CHAR` had no users at all), so
Culverin no longer defines a single `JPH_*` macro of its own. `BodyHandle`
and `ConstraintHandle` are no longer `using = std::uint64_t`: they are
`CPH::GenerationalHandle<Tag>` instantiations (core/Handle.hpp) with
`Index()`, `Generation()`, `Raw()`, `IsValid()` and `FromParts()`, so a body
handle can no longer be passed where a constraint handle is expected. The
whole legacy helper set -- `make_handle`, `kHandleIndexBits`,
`kHandleIndexMask` and every hand-written `& mask` / `>> bits` -- is gone,
including in the ECS, whose `Entity` is the same instantiation.
`readability-identifier-naming` is now enforced: 273 findings across 32 TUs
when the sweep started, 0 now, with the exceptions written into `.clang-tidy`
rather than left implicit (third-party headers, CPython C slots, Jolt's
`in*` override parameters, `Buffer<T>`'s STL-compatible member names, and
the two deliberate constant conventions). `extern/JoltC`, the `joltc_double` static
archive, the `joltc.c`-compiled-as-C++ hack, the last `joltc.h` include
and `src/jolt/JoltUtils.hpp` (whose `JoltPtr`/`JoltDeleter`/`ToJoltBodyId`
shims had no callers) are all deleted; nothing under `src/` names a
`JPH_*` symbol and the built extension contains none. The 14 enum values
Python sees (`DOF_*`, `BEND_*`) were re-pointed at
`JPH::EAllowedDOFs`, `JPH::SoftBodySharedSettings::EBendType` and
`JPH::EMotionType` only after a program proved every one is numerically
identical, so no constant changed. `JPH_M_PI` became `JPH::JPH_PI`: the
old macro was a float copy of the same literal (its own comment said "to
avoid collision with JPH_PI"), so the hinge defaults are bit-identical.
`JPH_Init()` is inlined as `RegisterDefaultAllocator` + `Factory` +
`RegisterTypes` behind an idempotence guard; its process-global
`TempAllocator` was never used because every world owns one.
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

*Landed so far — ragdolls.* The ragdoll subsystem is on the direct Jolt
API: `Ragdoll.{hpp,cpp}`, the ragdoll half of `PhysicsWorldCreation.cpp`,
the three declarations in `PhysicsWorldState.hpp`, the ragdoll structs in
`PhysicsParams.hpp`, and the Python object types in
`src/python/ragdolls/`. `JPH_Skeleton`/`JPH_RagdollSettings`/`JPH_Ragdoll`
become `JPH::Skeleton`/`JPH::RagdollSettings`/`JPH::Ragdoll`, and the two
`reinterpret_cast` bridges that the code was already using to reach
`RagdollSettings::CreateRagdoll` are gone with them.

Lifetime is unchanged because it was already correct: all three are
`RefTarget`s, and joltc's `JPH_*_Create` did `new T()` + `AddRef()` while
`JPH_*_Destroy` did `Release()`. The conversion reproduces that exactly
rather than switching to `Ref<>`, so the Python side's ownership
accounting does not move.

Two equivalences were checked against joltc's source rather than assumed:

- `JPH_SwingTwistConstraintSettings_Init` memsets the C struct and then
  copies the defaults back out of a default-constructed native
  `SwingTwistConstraintSettings`; `FromJolt` mirrors every field that
  `SetPartToParent` later reads. So a native
  `new SwingTwistConstraintSettings()` plus only the overrides the code
  actually sets is equivalent field for field, and the eleven
  `JPH_RagdollSettings_SetPart*` calls collapse into direct writes to
  `RagdollSettings::Part` (which derives from `BodyCreationSettings`).
- `JPH_Mat4` and `JPH::Mat44` are both four column-major `Vec4` — 64 bytes
  each, verified with `sizeof` — so the caller's float32 matrices buffer
  is still read through a `static_cast`, and `sizeof(JPH_Mat4)` in the
  "buffer too small" message keeps the same value.

`JPH_Vec3`/`JPH_RVec3` are **not** layout-compatible with
`JPH::Vec3`/`JPH::RVec3` (12 vs 16 and 24 vs 32 bytes), so
`PhysicsRagdollPartRequest` converts at construction instead of casting.
`CreateRagdollInstance` kept its `JPH_RVec3`/`JPH_Quat` parameters at
this point — those are shared value types on the body-creation path and
belonged to a different slice — and converted at the boundary. That
later slice has landed: the parameters are native now and the boundary
conversion is gone.

`JPH_BodyID` survived in `CopyRagdollDebugInfo` while `body_interface`
was still a `JPH_BodyInterface *`, handing it the same packed word joltc
did. The layering phase below removes that too: `body_interface` is a
`JPH::BodyInterface *` now, so the reads go through
`GetPosition`/`GetRotation`/`GetLinearVelocity` on it.

One pre-existing defect found and **preserved**, marked `TODO(bug)`:
`JPH_RagdollSettings_SetPartToParent` copies fifteen fields but never
`base` — so `constraint.base.enabled = true` was a dead write, as were
`constraintPriority`, both step-count overrides, `drawConstraintSize` and
`userData`. It was harmless only because `ConstraintSettings::mEnabled`
defaults to `true` (`Constraint.h:15`). Left unset; wiring it up would be
a no-op today and a behaviour change if that default ever moved.

*Landed so far — soft bodies.* `SoftBody.{hpp,cpp}` and the soft-body half
of `PhysicsWorldCreation.cpp` are on the direct Jolt API; that completes
the per-subsystem work in A4. `JPH_SoftVertex`/`JPH_SoftFace` were
JoltC's C mirrors of `SoftBodySharedSettings::Vertex`/`::Face`, and every
`JPH_SoftBodySharedSettings_*` call was a thin wrapper around a public
member, so all eleven of them collapse into direct access.

Three things about the JoltC wrappers had to be reproduced rather than
approximated, because they are not obvious from the signatures:

- `JPH_SoftBodySharedSettings_Create` did not just allocate. It pushed
  `JPH::PhysicsMaterial::sDefault` into `mMaterials` — joltc's own comment
  says Jolt requires at least one material to resolve face collisions —
  and then `AddRef`'d. Dropping either would change behaviour.
- `AddVertices`/`AddFaces` early-return on a zero count and `reserve` once
  for the batch rather than per element. Both kept, including the
  staging buffer: `add_faces` validates vertex indices *inside* its
  packing loop, and writing straight into `mFaces` would leave partially
  populated settings on the error path.
- `CreateConstraints` builds a `VertexAttributes` array sized to the
  vertex count with only `mCompliance` and `mShearCompliance` set, so
  `mBendCompliance` keeps its `FLT_MAX` default and the LRA fields keep
  theirs. Setting all of them "properly" would have changed the
  constraints.

`Vertex::mPosition`/`mVelocity` are `Float3`, not `Vec3`, so the packing
loops construct `JPH::Float3` directly. `JPH_SoftBodyBendType` was a
C mirror of `EBendType` with matching enumerators, so the integer that
crosses the parser boundary is unchanged and the cast is the only edit.

`PhysicsWorldSoftBodyCreateRequest::shared_settings` is now
`const JPH::SoftBodySharedSettings *`, which removes the
`reinterpret_cast` in `CreateSoftBody` — `SoftBodyCreationSettings::mSettings`
already held it read-only.

Observed but not investigated: `world.get_soft_body_vertices` returns a
stride-8 float buffer whose slot 2 holds a structured non-position value
(exact powers of two, stable across runs). It is unchanged by this patch
and the probe compares the whole buffer either way.

*Landed so far — body operations on the direct `BodyInterface`.* The two
paths that turn a `PhysicsBodyOperation` into a Jolt call —
`execute_now` in `PhysicsWorldOps.cpp` and the `FlushContext` visitor in
`CommandBuffer.cpp` — no longer go through `JPH_BodyInterface`. That is
29 of the 52 call sites.

`world->body_interface` is a `reinterpret_cast` of
`physics_system.GetBodyInterface()` (`PhysicsWorld.cpp:637`), i.e. the
**locking** interface, so both conversions go to that same one rather
than `GetBodyInterfaceNoLock()`. Worth recording because the two are not
interchangeable and nothing in the JoltC signature said which was in use.

`FlushContext::bi` becomes a `JPH::BodyInterface &`, which converts every
handler in the visitor at once; `CreateBody` in that same struct was
already calling `physics_system.GetBodyInterface()` natively, so the file
was mixed before and is consistent now.

`JoltUtils.hpp` gains `ToJolt` overloads for `JPH_Vec3`/`JPH_RVec3`/
`JPH_Quat` and a `ToJoltBodyId`. These are conversions, not casts:
`JPH_Vec3` is 12 bytes against `JPH::Vec3`'s 16 and `JPH_RVec3` is 24
against `JPH::RVec3`'s 32, so a `reinterpret_cast` would read past the
end of the source object. The remaining call sites in the next slice
should use these rather than open-coding the field copies.

_Superseded_: the value-type slice converted the payloads themselves, so
all three `ToJolt` value overloads lost their callers and are deleted.
`ToJoltBodyId` remains but is likewise caller-free now that the shadow
buffer stores `JPH::BodyID` directly.

One pre-existing smell found and **preserved**, marked `TODO(smell)`:
`CommandBuffer`'s `SetMotionType` handler derives the object layer from
the motion type — `cmd.motion_type == 0 ? 0 : 1` — because the command
only carries a motion type. `Static` maps to `ObjectLayer::Static` and
both `Kinematic` and `Dynamic` map to `ObjectLayer::Dynamic`, which
matches `Types.hpp:95-100` today, but it conflates motion type with
collision layer and can never select `ObjectLayer::Vehicle`. Fixing it
means widening the command, which is a behaviour change.

Remaining A4 work: the rest of the `JPH_BodyInterface` sites (23) in
`PhysicsWorldRuntime`, `PhysicsWorldCreation`, `PhysicsWorld.cpp`,
`Ragdoll.cpp` and `Constraint.cpp`, then `JPH_BodyID` as the shadow
buffer's storage type.

**A5 — Remove JoltC. (Done.)** First, audit the fork: `extern/JoltC` carries
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

`VerifyAbiAlignment` was the one place that could not simply be deleted.
It existed to catch a C/native double-precision mismatch, which is
structurally impossible once both sides are the same headers, so it was
rewritten natively and kept -- not for the check, which is now only a
runtime round-trip assertion, but because it is not side-effect free:
creating and destroying a body advances Jolt's `BodyID` free-list and
touches the broadphase, which changes the `BodyID` every later body
receives, and body order feeds the solver. Deleting the call left every
test green but moved 888 lines of `ragdoll_probe.py` output; restoring
the create/destroy sequence natively restored the baseline byte for byte.
The function and its `RuntimeError` wording are therefore kept verbatim,
with a comment in `PhysicsWorld.cpp` explaining why it must stay ahead of
the first user body. It is the last "JoltC" string in `src/`, and it is
historical only.

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
`parse_rotation` returning `expected<JPH::Float3|PosStride|AuxStride,
ParseError>` (`Vec3f` was the project's own tight triple and is gone in
favour of Jolt's `Float3`); `ParseError` owns the exact exception instance the
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

**Vectorcall thunks (landed).** `bindings/parsing/Thunk.hpp` turns the
parser's own type into the argument list. `ParserSignature<FastParser<Ts...>>`
exposes `std::tuple<Ts...>`, so `InvokeThunk<&WorldParsers::Impulse>(...)`
allocates the locals as one tuple, runs the parse and — on failure — hands the
captured error to the same `RaiseParseError` the hand-written adapters called,
which is why the diagnostics move not one byte. The action receives the values
already converted; `PyObject*`, `bool` and `void` returns are marshalled, and
anything else is rejected at the call site rather than being converted through
one of them.

The first family converted is the six vector body operations
(`apply_impulse`, `apply_angular_impulse`, `apply_force`, `apply_torque`,
`set_linear_velocity`, `set_angular_velocity`). Each was 26 lines of identical
code differing only in the parser, the operation type and the `ValidateFinite`
label; the label is the third template argument now, and
`MakeThunkMethod<...>(name)` writes the table entry straight from the
instantiation, so the six named functions are gone from the file. The order of
operations is unchanged — parse, then the finite check, then build the
operation, then `RunBodyOperation` — including the `Py_BEGIN_ALLOW_THREADS`
around the native call, which happens inside `RunBodyOperation` where it always
did.

One restriction, stated at the top of the header: the tuple is
value-initialised and `Arg<Name, T, Required>` carries no default — every
hand-written site set its default on its own local before `Parse`. A parser
with an optional argument would therefore silently pick up zeroes, so those
methods keep their wrapper until `Arg` grows a default value. All four
arguments of the six converted operations are required.

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

## B5 — the boundary invariant

`python/` is an adapter, not an engine. Files under `python/` may only:

1. parse Python arguments and dictionaries into native structs (`FastParser`);
2. call a method on a native C++ object in `src/`;
3. wrap native return values, spans and buffers into Python objects
   (`FastBuild`, `BufferProxy`).

Zero physics simulation, zero data-structure algorithms and zero Jolt
listener implementations belong in `python/`. The point is that
`culverin_core` is usable headless: every behaviour the Python module
exposes must be reachable from C++ without `PyObject` anywhere in the call
path.

Status of the migration:

* **`-Winvalid-offsetof`.** `PhysicsWorld` published `__weaklistoffset__`
  through `offsetof(PhysicsWorldObject, weakreflist)`, which is diagnosed by
  Clang on a non-standard-layout type. The type now sets
  `Py_TPFLAGS_MANAGED_WEAKREF` (Python >= 3.12, already the project floor)
  alongside the `MANAGED_DICT` it already used, so CPython owns the weakref
  list: the `weakreflist` member and the members table are gone, and the
  dealloc clears weakrefs unconditionally.

* **Profiling.** `CPH_PROFILE_BEGIN`/`CPH_PROFILE_END`/`CPH_PROFILE_ACCUMULATE`
  and the no-op `CPH_INIT_PROFILER` are gone, replaced by RAII
  `CPH::ProfileScope` and `CPH::ProfileAccumulator` (core/Profiling.hpp). A
  scope cannot be left open on an early return the way a paired macro could.

* **Startup ABI probe renamed.** `VerifyAbiAlignment` is
  `CPH::PrimeBodyIdFreeList`. The old name claimed a pure check; the function
  is a primitive that also advances Jolt's BodyID free list, and the recorded
  ragdoll/soft-body numerics depend on that side effect. It now returns bool
  and its comment says which half is the check and which half is the prime.

* **ECS — migrated.** `src/ecs/EcsTypes.hpp`, `SparseSet.hpp`, `Registry.hpp/.cpp`
  hold the generational entity bookkeeping and the sparse-set component
  stores. All manual allocate/reallocate/free is `CPH::Buffer`; every fallible
  operation returns `std::expected<T, CPH::Error>` with an `EcsError` code, and
  no `PyErr_SetString` remains in the data structure. `python/ecs/Ecs.cpp` is
  a parse/call/wrap adapter. The one operation that needs both halves --
  copying body transforms into components -- is `CPH::SyncEcsTransforms` in
  `src/physics/EcsSync.cpp`, so `src/ecs/` stays free of Jolt.

* **Constraints — migrated.** The sub-type dispatch (`Hinge` / `Slider`), the
  motor-state machine and the two-body wake-up are `CPH::SetConstraintTarget`
  in `src/physics/ConstraintOps.cpp`. `WorldSetConstraintTarget` parses two
  arguments, calls it with the GIL released, and maps the failure.

* **Ships — migrated.** `CPH::Ship` (src/vehicles/Ship.cpp) owns its
  `JPH::PhysicsStepListener`, its input atomics and its sled resolution; the
  flight model is `ShipConfig` plus the step body, unchanged bit for bit.
  `python/vehicles/Ship.cpp` parses and forwards. `src/python/vehicles/Ship.hpp`
  became `ShipBinding.hpp` so it cannot shadow `src/vehicles/Ship.hpp` on the
  include path.

* **Characters — migrated.** `CPH::Character` (src/characters/Character.cpp)
  owns the `JPH::CharacterVirtual`, its contact listener, and the atomics the
  listener reads (own handle, push strength, last velocity). The listener moved
  wholesale: `OnAdjustBodyVelocity` (friction-scaled platform-velocity
  inheritance), `OnContactValidate`, `OnCharacterContactValidate`, the
  Added/Persisted/Removed callbacks for bodies and for characters, and
  `ApplyImpulse`. `CPH::Character::Move` runs the extended update, including
  ground-velocity inheritance. `python/characters/Character.cpp` parses
  arguments and builds return values; `python/characters/Character.hpp` became
  `CharacterBinding.hpp` so it cannot shadow the native header.

  There are now **no `JPH::*Listener` implementations under `python/`**.

  Two defects were fixed in the port rather than carried over, both disclosed:
  `CharacterSetRotation` indexed `bodies.rotations` as `dense * 4`, a leftover
  from when that storage was a flat float array -- with `Buffer<AuxStride>`
  (already four floats per element) the write landed at four times the intended
  offset, so rotating any character but the first left its shadow rotation stale
  and clobbered a neighbouring slot. And the four write-only `prev_*` fields on
  `CharacterObject` were dead, so they are gone.

* **Vehicles — migrated.** `CPH::Vehicle` (src/vehicles/Vehicle.hpp/.cpp) owns
  the `JPH::VehicleConstraint`, so the constraint is both the vehicle and its
  own step listener. Configuration is plain structs -- `WheelConfig`,
  `EngineConfig`, `TransmissionConfig`, `WheeledVehicleConfig`, and the tracked
  `TrackWheelConfig`/`TrackConfig`/`TrackedVehicleConfig` pair -- and the two
  factories (`CreateWheeled`, `CreateTracked`) build the Jolt settings, take the
  chassis body lock under the trampoline lock, register the constraint and
  attach the ray collision tester. The arcade transmission state machine moved
  with it: gear engagement, the rolling-resistance brake while coasting, the
  "still rolling the wrong way, brake first" override, and the tank's
  kickstart-out-of-neutral are `Vehicle::SetInput`/`SetTankInput`, as are the
  wheel-world/local transforms and the drivetrain debug dump. The layer keeps
  the split the two paths always had -- a wheeled vehicle drops input that
  arrives mid-step, a tank waits for the step to finish -- because that
  difference is observable in the replays.
  `python/vehicles/Vehicle.cpp` and `TrackedVehicle.cpp` now only parse wheel,
  engine and transmission dictionaries into those structs, call the factory and
  pack the result; `python/vehicles/Vehicle.hpp` became `VehicleBinding.hpp`
  and `TrackedVehicle.hpp` (four constants and an engine struct with no
  readers) is gone.

  Four defects were fixed in the port, all disclosed: the wrapper's
  `tester` ownership was dead -- `VehicleConstraint` keeps its own
  `RefConst<VehicleCollisionTester>`, so the creation reference is simply
  released when the factory returns -- and the field is gone; the
  `PyObject_GC_New` storage is now initialised before anything can fail, so the
  parser-group failure path no longer decrefs an object full of indeterminate
  pointers; `destroy()` no longer deletes the parser group that `tp_dealloc`
  deletes again, which was a double free for any caller that destroyed a
  vehicle and then let it be collected; and teardown removes the constraint and
  its listener under `jph_trampoline_lock` (after parking on the shadow lock,
  never holding both in the inverted order) instead of mutating the physics
  system after the lock was already released.

  Preserved and marked with `TODO`s: `create_vehicle(transmission=...)` still
  never reaches the controller (the settings are built and then discarded, so
  the wheeled path runs Jolt's defaults); `WorldCreateVehicle` still takes the
  extra world reference; `set_tank_input(0, 0, 0)` still reaches Jolt's
  `inLeftRatio != 0` assert; the tester's 2.0 is still a slope angle rather
  than a scale; steering is still inferred from the sign of a wheel's local Z.
  One ordering nuance: a malformed wheel dictionary is now rejected before the
  chassis handle is resolved rather than after, so a call that gets both wrong
  raises the wheel message. Every single-fault diagnostic is unchanged, byte
  for byte.

* **Ragdolls — migrated.** `CPH::Skeleton`, `CPH::RagdollSettings` and
  `CPH::Ragdoll` (src/ragdolls/Ragdoll.hpp/.cpp) own the skeleton bookkeeping,
  the part layout and the instance, including the world-side slot table.
  `CPH::Ragdoll::Create` runs instantiate -> register -> teardown-on-failure in
  one call, so a half-built ragdoll can no longer reach a Python object.
  `PhysicsWorld::CopyRagdollBodyHandles` and `CopyRagdollDebugInfo` are gone;
  the equivalents read the slot table and the bodies through their own ragdoll.
  `python/ragdolls/Ragdoll.cpp` only parses, calls and packs, and
  `python/ragdolls/Ragdoll.hpp` became `RagdollBinding.hpp`.

* **Soft bodies — migrated.** `CPH::SoftBodySettings`
  (src/soft_bodies/SoftBodySettings.hpp/.cpp) owns the vertices, the faces and
  the constraint generation; `python/soft_bodies/SoftBody.cpp` keeps the buffer
  acquisition, the float32-triplet checks and every diagnostic string, because
  two of those strings deliberately differ between `add_vertex` and
  `add_vertices`. `python/soft_bodies/SoftBody.hpp` became `SoftBodyBinding.hpp`;
  the cached `num_vertices`/`constraints_created`/`optimized` fields it carried
  are derived from the native object now instead of being mirrored by hand.

*Landed so far — ragdolls and soft bodies leave `python/`.* This is the
one-way dependency rule turned on the last two domains that still broke
it. `src/python/ragdolls/` and `src/python/soft_bodies/` no longer own
any Jolt state or any algorithm; they are `RagdollBinding.hpp` +
`Ragdoll.cpp` and `SoftBodyBinding.hpp` + `SoftBody.cpp`, and they do
three things only: parse Python arguments into native structs, call one
method on a native object, and wrap the result.

What moved:

- `src/ragdolls/{Ragdoll.hpp,Ragdoll.cpp}` — new
  `CPH::Skeleton`, `CPH::RagdollSettings` and `CPH::Ragdoll`. The
  skeleton's AddRef/release accounting, the part layout (including the
  `CalculateInertia`-not-`MassAndInertia` mass override, the
  `mParts.resize(joint_count)` widening and the default-constructed
  `SwingTwistConstraintSettings`), the pose clamp in `DriveToPose`, the
  slot teardown and both debug reads live here. `RagdollError` moved
  with them.
- `src/soft_bodies/SoftBodySettings.{hpp,cpp}` — `CPH::SoftBodySettings`.
  `AddVertices`/`AddFaces` keep joltc's reserve-once and zero-count early
  return, `AddFaces` keeps validating inside the packing loop so a bad
  face leaves the earlier ones pushed, and `CreateConstraints` keeps the
  `VertexAttributes` array with only `mCompliance`/`mShearCompliance`
  set. `SoftBodySettingsError` carries the five failure kinds; the
  Python-visible strings stay in the binding, because two of them
  deliberately differ (`add_vertex` says "Cannot modify settings after
  optimize()", `add_vertices` says "Cannot modify after optimize()").
- `CPH::Ragdoll::Create` replaces the three-step
  create/register/fail-and-teardown dance that `WorldCreateRagdoll` used
  to spell out, so a half-built ragdoll can no longer escape into a
  Python object.
- `PhysicsWorld::CopyRagdollBodyHandles` and `CopyRagdollDebugInfo` are
  gone; the equivalents are `CPH::Ragdoll::CopyBodyHandles` and
  `CopyDebugInfo`, which own the slot table they read. The debug read
  now goes through `JPH::BodyInterface` rather than `JPH_BodyInterface_*`
  — same three getters, same lock (`shadow_lock`, not
  `LockShadowNotStepping`, because the original only ever took the
  plain lock).
- The dead free function `PhysicsWorldCreateRagdollSettings` is deleted;
  `CPH::RagdollSettings::Create` is the factory and still takes its own
  skeleton reference.

Preserved quirk-for-quirk, all now commented at the site:

- a malformed `pos` in `RagdollSettingsAddPart` is *not* fatal: the
  position is left unset, the call continues, and the exception stays
  pending so CPython reports the NULL/exception-set mismatch. Re-raising
  it through `ParseError` instead would have been a behaviour change;
- `get_debug_info` still fetches `rotation` and still never surfaces it;
- the debug-info block is still over-allocated by `alignof(…)` and
  hand-aligned, because `Allocate()` is `PyMem_RawMalloc` and
  `PhysicsRagdollBodyDebugInfo` holds SIMD types (`alignof` 32);
- `drive_to_pose` still returns `None` when the ragdoll or its world is
  gone, and still reports a failed apply as `MemoryError`.

The `culverin_native_ecs`, `culverin_native_vehicles`,
`culverin_native_characters`, `culverin_native_ragdolls` and
`culverin_native_soft_bodies` targets carry the migrated code behind
`culverin_core`, and `culverin_native_physics` no longer links
`culverin_native_soft_bodies` at all — the dependency runs one way.

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
