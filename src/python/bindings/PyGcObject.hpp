#pragma once

// A CRTP base for the extension's GC-managed Python object types.
//
// A correct GC type needs seven things to agree, and nothing in CPython warns
// you when they do not:
//
//   1. Py_TPFLAGS_HAVE_GC on the type spec
//   2. a Py_tp_traverse slot
//   3. a Py_tp_clear slot
//   4. a Py_tp_dealloc slot that untracks
//   5. allocation through PyObject_GC_New (NOT PyObject_New)
//   6. PyObject_GC_Track after the object is fully initialised
//   7. Py_DECREF on the type in dealloc, because PyObject_GC_New increfs it
//
// Every one of those is derived here from the derivee, so a type built on this
// base cannot get them wrong or leave one out. The two things that genuinely
// cannot be derived -- which Python references exist, and how to tear down the
// native object -- are declared by the derivee, and `static_assert` refuses to
// compile a derivee that omits either. Forgetting is a compile error, not a
// leak discovered months later.
//
// The other half is ordering, and it is the part that bites at runtime.
// `tp_clear` may run before `tp_dealloc`, so a native object that borrows a raw
// PhysicsWorld* out of a refcounted `world` member is a use-after-free waiting
// to happen: clear drops the world, and the native destructor then dereferences
// it. `BeforeClear()` is the one hook that runs before any reference is
// dropped, from clear and from dealloc alike, so the ordering cannot drift
// between the two paths.
//
// Usage:
//
//     struct ThingObject : PyGcObject<ThingObject> {
//         PhysicsWorldObject *world = nullptr;   // no PyObject_HEAD: the base
//         Thing               *thing = nullptr;   // supplies it
//
//         // Native teardown. Runs before ANY reference is dropped, from clear
//         // and dealloc alike; must be idempotent because both call it.
//         void BeforeClear() noexcept { delete thing; thing = nullptr; }
//
//         // Python references. The only list that is not derivable.
//         template <typename Visitor>
//         void VisitRefs(Visitor &&visit) noexcept { visit(world); }
//     };
//
// The slots and the spec come from the same place as the implementations:
//
//     static constexpr auto kThingGc = CPH::Python::GcSlots<ThingObject>();
//     static PyType_Slot ThingSlots[] = {
//         kThingGc.dealloc, kThingGc.traverse, kThingGc.clear,
//         ...other slots...
//     };
//     constexpr PyType_Spec ThingSpec =
//         CPH::Python::GcTypeSpec<ThingObject>("culverin._culverin_c.Thing", ThingSlots);
//
// Creation goes through the base too, so allocation and tracking cannot be
// separated from each other:
//
//     auto *obj = ThingObject::New(type);        // GC_New, zeroed, NOT tracked
//     if (!obj) { return nullptr; }
//     obj->world = reinterpret_cast<PhysicsWorldObject *>(Py_NewRef(world));
//     obj->thing = native.release();
//     return obj->Publish();                     // GC_Track, returns PyObject*
//
// One caveat about member declarations: New() zeroes the whole struct, but no
// constructor ever runs, so a member's in-class initialiser does NOT execute.
// `= nullptr` is therefore redundant (zero is what you get anyway) and
// anything non-zero would silently stay zero. Initialise non-zero members
// explicitly in the creation path, never in the declaration.

#include <Python.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

namespace CPH::Python {

namespace detail {

// Probe type used only to check that a derivee exposes VisitRefs at all.
// Generic because a derivee's members are usually typed as the concrete
// wrapper (PhysicsWorldObject*, ShipObject*, ...) rather than PyObject*.
struct RefVisitorProbe {
    template <typename T>
    void operator()(T *&) const noexcept {}
};

template <typename Self>
[[nodiscard]] int GcTraverseImpl(PyObject *self, visitproc visit, void *arg) noexcept {
    // The type itself is a strong reference for heap types and the collector
    // has to be able to see it, or a class held only by its own instances is
    // never collected. CPython's own *_traverse functions do this first.
    Py_VISIT(Py_TYPE(self));

    // Py_VISIT cannot expand inside a lambda (it returns from the enclosing
    // function), so the visitproc result is accumulated instead. Generic
    // because members are typed as their concrete wrapper, not PyObject*.
    int rc = 0;
    static_cast<Self *>(reinterpret_cast<void *>(self))->VisitRefs(
        [&rc, visit, arg](auto *&ref) noexcept {
            if (rc == 0 && ref != nullptr) {
                rc = visit(reinterpret_cast<PyObject *>(ref), arg);
            }
        });
    return rc;
}

template <typename Self>
[[nodiscard]] int GcClearImpl(PyObject *self) noexcept {
    auto *obj = static_cast<Self *>(reinterpret_cast<void *>(self));

    // Native teardown first: it may borrow anything the references are keeping
    // alive, so it has to run while they are still held. Required to be
    // idempotent, because dealloc calls clear and clear may already have run.
    obj->BeforeClear();

    obj->VisitRefs([](auto *&ref) noexcept {
        if (ref != nullptr) {
            PyObject *as_py = reinterpret_cast<PyObject *>(ref);
            Py_DECREF(as_py);
            ref = nullptr;
        }
    });
    return 0;
}

template <typename Self>
void GcDeallocImpl(PyObject *self) noexcept {
    PyTypeObject *tp = Py_TYPE(self);

    // Untrack before anything is torn down. The IsTracked guard keeps this
    // safe for an object that was never tracked -- for instance a construction
    // that failed before Publish().
    if (PyObject_GC_IsTracked(self)) {
        PyObject_GC_UnTrack(self);
    }

    // Weakrefs are cleared here rather than in each type, so a type that gains
    // weakref support cannot forget to. This has to be conditional: for a type
    // with no weakref support PyObject_ClearWeakRefs reports a SystemError
    // rather than doing nothing, so it may only be called when there is
    // somewhere for the list to live.
    if (PyType_HasFeature(tp, Py_TPFLAGS_MANAGED_WEAKREF) || tp->tp_weaklistoffset > 0) {
        PyObject_ClearWeakRefs(self);
    }

    // Same order as clear: native first, references second.
    [[maybe_unused]] const int cleared = GcClearImpl<Self>(self);

    // Rarely needed: a non-trivial member that needs its C++ destructor run.
    // Most types hold raw pointers and skip this entirely.
    if constexpr (requires(Self &o) { o.BeforeFree(); }) {
        static_cast<Self *>(reinterpret_cast<void *>(self))->BeforeFree();
    }

    // The storage came from PyObject_GC_New, so it goes back through
    // PyObject_GC_Del; that is the matching release for an untracked object.
    PyObject_GC_Del(self);

    // PyObject_GC_New took a reference to the type on our behalf. Holding it
    // until after the free is deliberate: tp must outlive the storage.
    Py_DECREF(tp);
}

} // namespace detail

template <typename Self>
[[nodiscard]] constexpr bool ValidateGcDerivee() noexcept {
    // Called only once `Self` is complete. During instantiation of
    // PyGcObject<Self> the derivee is still being defined, so member
    // detection cannot work there; every entry point that needs a valid
    // derivee runs this first.
    //
    // A derivee that forgets VisitRefs() would build a *tracked* type with
    // traverse and clear slots that visit nothing: it looks correct, it cannot
    // break cycles, and nothing reports the difference. A compile error is the
    // only acceptable outcome.
    static_assert(requires(Self &o) { o.VisitRefs(detail::RefVisitorProbe{}); },
                  "A PyGcObject derivee must declare 'template <typename Visitor> "
                  "void VisitRefs(Visitor&&)'. List every PyObject* member it holds.");

    // BeforeClear is required rather than optional on purpose: an empty body is
    // a deliberate statement that the type owns no native state, while a
    // missing hook is an omission a reviewer cannot distinguish from one.
    // Forgetting it is exactly how tp_clear becomes a use-after-free.
    static_assert(requires(Self &o) { o.BeforeClear(); },
                  "A PyGcObject derivee must declare 'void BeforeClear() noexcept'. "
                  "Tear down native state there, before any Python reference is "
                  "dropped; use an empty body if the type owns no native state.");
    return true;
}

// --- Slot trio -------------------------------------------------------------

// Dealloc, traverse and clear for a PyGcObject derivee. Expanding these three
// is what makes "forgot to register tp_traverse" impossible: the slots and the
// implementations are produced from one place.
template <typename Self>
[[nodiscard]] int GcTraverseEntry(PyObject *self, visitproc visit, void *arg) noexcept {
    return detail::GcTraverseImpl<Self>(self, visit, arg);
}

template <typename Self>
[[nodiscard]] int GcClearEntry(PyObject *self) noexcept {
    return detail::GcClearImpl<Self>(self);
}

template <typename Self>
void GcDeallocEntry(PyObject *self) noexcept {
    detail::GcDeallocImpl<Self>(self);
}

// Three ready-made slot initialisers. A type writes them as a block at the top
// of its slot table, so the trio is visible and cannot be partially copied.
template <typename Self>
[[nodiscard]] constexpr PyType_Slot GcDeallocSlot() noexcept {
    static_assert(ValidateGcDerivee<Self>(), "invalid PyGcObject derivee");
    return {.slot = Py_tp_dealloc, .pfunc = reinterpret_cast<void *>(&GcDeallocEntry<Self>)};
}

template <typename Self>
[[nodiscard]] constexpr PyType_Slot GcTraverseSlot() noexcept {
    static_assert(ValidateGcDerivee<Self>(), "invalid PyGcObject derivee");
    return {.slot = Py_tp_traverse, .pfunc = reinterpret_cast<void *>(&GcTraverseEntry<Self>)};
}

template <typename Self>
[[nodiscard]] constexpr PyType_Slot GcClearSlot() noexcept {
    static_assert(ValidateGcDerivee<Self>(), "invalid PyGcObject derivee");
    return {.slot = Py_tp_clear, .pfunc = reinterpret_cast<void *>(&GcClearEntry<Self>)};
}

// --- The base --------------------------------------------------------------

template <typename Self>
class PyGcObject {
  public:
    // The header lives here so every derivee has the identical layout at
    // offset 0 and PyObject_GC_New/GC_Del/visitors all agree about it. A
    // derivee therefore does NOT repeat PyObject_HEAD.
    PyObject_HEAD

    // --- Allocation ---------------------------------------------------------

    // GC-allocated, explicitly zeroed, and deliberately NOT tracked. Tracking
    // an object before its members are initialised is how half-built objects
    // reach the collector. Publish() does the tracking.
    //
    // The zeroing is not redundant. PyObject_GC_New hands back *uninitialised*
    // memory -- it is PyObject_New plus a GC header, not a calloc -- and a
    // derivee's in-class member initialisers never run because no constructor
    // is ever called. Without this, a failure path that decrefs a half-built
    // object frees whatever garbage the allocator last held there.
    [[nodiscard]] static Self *New(PyTypeObject *type) noexcept {
        static_assert(ValidateGcDerivee<Self>(), "invalid PyGcObject derivee");
        auto *obj = reinterpret_cast<Self *>(PyObject_GC_New(Self, type));
        if (obj != nullptr) {
            // uintptr_t arithmetic, not pointer arithmetic on void*, which is a
            // GNU extension and warns under -Wpointer-arith.
            std::memset(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(obj) + sizeof(PyObject)),
                        0, sizeof(Self) - sizeof(PyObject));
        }
        return obj;
    }

    // Track and hand back the PyObject. Call as the last step of construction,
    // once every member is valid.
    [[nodiscard]] PyObject *Publish() noexcept {
        PyObject_GC_Track(reinterpret_cast<PyObject *>(this));
        return reinterpret_cast<PyObject *>(this);
    }

  protected:
    // The storage comes from PyObject_GC_New and goes back through
    // PyObject_GC_Del, never through a C++ delete, so this type is not
    // destroyed polymorphically.
    PyGcObject()                            = default;
    ~PyGcObject()                           = default;
    PyGcObject(const PyGcObject &)          = delete;
    PyGcObject &operator=(const PyGcObject &) = delete;
};

// --- Type spec -------------------------------------------------------------

// Builds the spec with Py_TPFLAGS_HAVE_GC already set, so the flag and the
// slots cannot disagree. Any extra flags (BASETYPE, MANAGED_WEAKREF, ...) are
// passed through.
[[nodiscard]] constexpr PyType_Spec GcTypeSpec(const char *name, uint32_t basicsize,
                                               PyType_Slot *slots,
                                               unsigned long extra_flags = 0) noexcept {
    return PyType_Spec{.name      = name,
                       .basicsize = static_cast<int>(basicsize),
                       .itemsize  = 0,
                       .flags     = static_cast<uint32_t>(Py_TPFLAGS_DEFAULT |
                                                          Py_TPFLAGS_HAVE_GC | extra_flags),
                       .slots     = slots};
}

} // namespace CPH::Python
