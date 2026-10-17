#include "Mutex.h"
#include "Thread.h"
// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <stdlib.h>
#include <time.h>
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

static constexpr int CULV_MUTEX_MAX_SPIN_COUNT = 40;

// Address sharding for the parking lot to eliminate false-sharing lock
// contention. 256 buckets padded to Apple Silicon / Modern x86 strict
// cache-line widths.
static constexpr size_t CULV_MUTEX_BUCKET_COUNT = 256;

#ifndef CULV_MUTEX_CACHELINE_ALIGNED
#    if defined(_MSC_VER)
#        define CULV_MUTEX_CACHELINE_ALIGNED __declspec(align(128))
#    elif defined(__GNUC__) || defined(__clang__)
#        define CULV_MUTEX_CACHELINE_ALIGNED __attribute__((aligned(128)))
#    else
#        define CULV_MUTEX_CACHELINE_ALIGNED
#    endif
#endif

static_assert(ATOMIC_INT_LOCK_FREE == 2, "CulvMutex requires always-lock-free atomics!");
static_assert(ATOMIC_POINTER_LOCK_FREE == 2, "CulvMutex parking lot requires lock-free pointers!");

#ifdef CULV_MUTEX_DEBUG
static constexpr size_t CULV_MUTEX_MAX_DEBUG_EDGES = 4096;
static constexpr size_t CULV_MUTEX_MAX_HELD_LOCKS  = 64;

static culv_platform_mutex_t culv_mutex_debug_graph_lock;
static struct {
    const CulvMutex *from;
    const CulvMutex *to;
} culv_mutex_debug_lock_edges[CULV_MUTEX_MAX_DEBUG_EDGES];
static size_t culv_mutex_debug_lock_edge_count = 0;

static thread_local const CulvMutex *culv_mutex_debug_held_locks[CULV_MUTEX_MAX_HELD_LOCKS];
static thread_local size_t culv_mutex_debug_held_lock_count = 0;

static inline uintptr_t culv_mutex_get_current_id(void) {
    CulvThread *fiber = culv_thread_get_current();
    // If we are in a fiber context, the fiber pointer is our unique ID.
    if (fiber) {
        return (uintptr_t)fiber;
    }
    // Fallback for raw OS threads not managed by CulvThread
    return (uintptr_t)CULV_THREAD_PLATFORM_CURRENT_ID();
}

static bool culv_mutex_debug_dfs_check_cycle(const CulvMutex *current, const CulvMutex *target) {
    if (current == target)
        return true;
    for (size_t i = 0; i < culv_mutex_debug_lock_edge_count; i++) {
        if (culv_mutex_debug_lock_edges[i].from == current) {
            if (culv_mutex_debug_dfs_check_cycle(culv_mutex_debug_lock_edges[i].to, target))
                return true;
        }
    }
    return false;
}

static void culv_mutex_debug_add_lock_edge(const CulvMutex *from, const CulvMutex *to) {
    CULV_THREAD_PLATFORM_MUTEX_LOCK(&culv_mutex_debug_graph_lock);

    for (size_t i = 0; i < culv_mutex_debug_lock_edge_count; i++) {
        if (culv_mutex_debug_lock_edges[i].from == from && culv_mutex_debug_lock_edges[i].to == to) {
            CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&culv_mutex_debug_graph_lock);
            return;
        }
    }

    if (culv_mutex_debug_dfs_check_cycle(to, from)) {
        CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&culv_mutex_debug_graph_lock);
        assert(false && "DEADLOCK DETECTED: Lock order cycle/inversion!");
        abort();
    }

    if (culv_mutex_debug_lock_edge_count < CULV_MUTEX_MAX_DEBUG_EDGES) {
        culv_mutex_debug_lock_edges[culv_mutex_debug_lock_edge_count].from = from;
        culv_mutex_debug_lock_edges[culv_mutex_debug_lock_edge_count].to   = to;
        culv_mutex_debug_lock_edge_count++;
    }
    CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&culv_mutex_debug_graph_lock);
}

void culv_mutex_debug_clear_owner(CulvMutex *mod) {
    atomic_store_explicit(&mod->has_owner, false, memory_order_release);
}

void culv_mutex_debug_check_pre_lock(CulvMutex *mod) {
    uint8_t bits = atomic_load_explicit(&mod->bits, memory_order_relaxed);
    if (bits & CULV_MUTEX_POISONED) {
        assert(false && "FATAL: Attempting to lock a POISONED mutex!");
        abort();
    }

    if (bits & CULV_MUTEX_LOCKED) {
        if (atomic_load_explicit(&mod->has_owner, memory_order_acquire)) {
            uintptr_t current = culv_mutex_get_current_id(); // <--- Fiber-aware ID
            uintptr_t owner   = atomic_load_explicit(&mod->owner, memory_order_relaxed);

            if (current == owner) {
                assert(false && "DEADLOCK DETECTED: Recursive locking!");
                abort();
            }
        }
    }
}

void culv_mutex_debug_post_lock(CulvMutex *mod) {
    atomic_store_explicit(&mod->owner, culv_mutex_get_current_id(), memory_order_relaxed);
    atomic_store_explicit(&mod->has_owner, true, memory_order_release);

    for (size_t i = 0; i < culv_mutex_debug_held_lock_count; i++) {
        culv_mutex_debug_add_lock_edge(culv_mutex_debug_held_locks[i], mod);
    }
    if (culv_mutex_debug_held_lock_count < CULV_MUTEX_MAX_HELD_LOCKS) {
        culv_mutex_debug_held_locks[culv_mutex_debug_held_lock_count++] = mod;
    }
}

void culv_mutex_debug_pre_unlock(CulvMutex *mod) {
    if (!atomic_load_explicit(&mod->has_owner, memory_order_acquire)) {
        assert(false && "FATAL: Unlocking a mutex that has no owner!");
        abort();
    }

    uintptr_t current = culv_mutex_get_current_id(); // <--- Fiber-aware ID
    uintptr_t owner   = atomic_load_explicit(&mod->owner, memory_order_relaxed);

    if (current != owner) {
        assert(false && "FATAL: Unlocking a mutex owned by another thread!");
        abort();
    }

    for (size_t i = culv_mutex_debug_held_lock_count - 1; i < culv_mutex_debug_held_lock_count; i--) {
        if (culv_mutex_debug_held_locks[i] == mod) {
            for (size_t j = i; j < culv_mutex_debug_held_lock_count - 1; j++) {
                culv_mutex_debug_held_locks[j] = culv_mutex_debug_held_locks[j + 1];
            }
            culv_mutex_debug_held_lock_count--;
            break;
        }
    }
}
#endif

typedef struct CulvMutexWaiter CulvMutexWaiter;
static constexpr size_t CULV_MUTEX_WAITER_ALIGNMENT = 64;
struct CulvMutexWaiter {
    alignas(CULV_MUTEX_WAITER_ALIGNMENT)
        // 8-byte members grouped together
        const void *address;
    CulvThread *fiber;
    CulvMutexWaiter *next;

    // culv_platform_cond_t size varies by OS, but we keep it here
    culv_platform_cond_t cond;

    // 1-byte atomic at the end
    CULV_MUTEX_ATOMIC(bool) signaled;
};

// Padded 128-byte cacheline bucket to completely eliminate false sharing.
static constexpr size_t CULV_MUTEX_BUCKET_ALIGNMENT = 128;
typedef struct {
    alignas(CULV_MUTEX_BUCKET_ALIGNMENT) culv_platform_mutex_t mutex;
    CulvMutexWaiter *head;
    uint8_t _pad[CULV_MUTEX_BUCKET_ALIGNMENT - (sizeof(culv_platform_mutex_t) + sizeof(void *))];
} CulvMutexBucket;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
alignas(CULV_MUTEX_BUCKET_ALIGNMENT) static CulvMutexBucket culv_mutex_parking_lot[CULV_MUTEX_BUCKET_COUNT];

static_assert(sizeof(CulvMutexBucket) == CULV_MUTEX_BUCKET_ALIGNMENT, "CulvMutexBucket must be exactly 128 bytes!");

[[gnu::constructor]]
static void culv_mutex_init_parking_lot() {
#pragma unroll 4
    for (size_t i = 0; i < CULV_MUTEX_BUCKET_COUNT; i++) {
        CULV_THREAD_PLATFORM_MUTEX_INIT(&culv_mutex_parking_lot[i].mutex);
        culv_mutex_parking_lot[i].head = nullptr;
    }
}

static inline size_t culv_mutex_hash_address(const void *addr) {
    // MurmurHash3 constants for 64-bit finalizer
    constexpr uint64_t MURMUR_MIXER         = 0xff51afd7ed558ccdULL;
    constexpr unsigned int MURMUR_BIT_SHIFT = 33U;

    uint64_t hash_val = (uintptr_t)addr;

    // Standard MurmurHash3 64-bit mixer logic
    hash_val ^= hash_val >> MURMUR_BIT_SHIFT;
    hash_val *= MURMUR_MIXER;
    hash_val ^= hash_val >> MURMUR_BIT_SHIFT;

    return (size_t)(hash_val % CULV_MUTEX_BUCKET_COUNT);
}

[[gnu::cold, gnu::noinline, gnu::nonnull(1)]] CULV_MUTEX_INTERNAL void culv_mutex_lock_slow(CulvMutex *mod) {
    size_t hash    = culv_mutex_hash_address(mod);
    CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];

    // --- PHASE 1: Adaptive Exponential Backoff Spin ---
    // We increase the number of culv_cpu_relax() calls per iteration.
    // On ARM64/Apple Silicon, 'yield' is very cheap, so we need a significant count.
    size_t backoff_limit = 1;

    for (int i = 0; i < CULV_MUTEX_MAX_SPIN_COUNT; i++) {
        uint8_t val = atomic_load_explicit(&mod->bits, memory_order_relaxed);

        // If the lock appears free, try to grab it.
        if (!(val & CULV_MUTEX_LOCKED)) {
            if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, val | CULV_MUTEX_LOCKED,
                                                      memory_order_acquire, memory_order_relaxed)) {
                culv_mutex_debug_post_lock(mod);
#ifdef CULV_MUTEX_DEBUG
                atomic_fetch_add_explicit(&mod->spin_success_count, 1, memory_order_relaxed);
#endif
                return;
            }
        }

        if (CULV_MUTEX_UNLIKELY(val & CULV_MUTEX_POISONED)) {
            return;
        }

        // Perform the exponential backoff
        for (size_t j = 0; j < backoff_limit; j++) {
            culv_cpu_relax();
        }

        // Cap the backoff to prevent excessive spinning if the lock is held long-term
        if (backoff_limit < 1024) {
            backoff_limit <<= 1U;
        }
    }

    // --- PHASE 2: Parking ---
    for (;;) {
        uint8_t val = atomic_load_explicit(&mod->bits, memory_order_relaxed);

        // Final attempt to snatch the lock (Lock Stealing)
        if (!(val & CULV_MUTEX_LOCKED)) {
            if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, val | CULV_MUTEX_LOCKED,
                                                      memory_order_acquire, memory_order_relaxed)) {
                culv_mutex_debug_post_lock(mod);
                return;
            }
            continue;
        }

        // Ensure the CULV_MUTEX_HAS_WAITERS bit is set before we actually sleep.
        if (!(val & CULV_MUTEX_HAS_WAITERS)) {
            if (!atomic_compare_exchange_weak_explicit(&mod->bits, &val, val | CULV_MUTEX_HAS_WAITERS,
                                                       memory_order_relaxed,
                                                       memory_order_relaxed)) {
                continue;
            }
        }

#ifdef CULV_MUTEX_DEBUG
        atomic_fetch_add_explicit(&mod->park_count, 1, memory_order_relaxed);
#endif

        CulvThread *current_thread = culv_thread_get_current();
        bool is_fiber             = (bool)(current_thread != nullptr && !current_thread->is_main);

        CulvMutexWaiter node = {
            .address = mod, .next = nullptr, .fiber = (int)is_fiber ? current_thread : nullptr};
        atomic_init(&node.signaled, false);

        if (!is_fiber) {
            CULV_THREAD_PLATFORM_COND_INIT(&node.cond); // Only init OS condvar for OS threads
        }

        CULV_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

        val = atomic_load_explicit(&mod->bits, memory_order_relaxed);
        if (CULV_MUTEX_UNLIKELY(!(val & CULV_MUTEX_LOCKED) || !(val & CULV_MUTEX_HAS_WAITERS))) {
            CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
            if (!is_fiber) {
                CULV_THREAD_PLATFORM_COND_DESTROY(&node.cond);
            }
            continue;
        }

        node.next    = bucket->head;
        bucket->head = &node;

        if (!is_fiber) {
            // === OS THREAD SLEEP ===
            while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
                CULV_THREAD_PLATFORM_COND_WAIT(&node.cond, &bucket->mutex);
            }
            CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
            CULV_THREAD_PLATFORM_COND_DESTROY(&node.cond);
        } else {
            // === FIBER YIELD ===
            // Fibers don't need the bucket lock to wait!
            // We drop it immediately so other threads/fibers can access the bucket.
            CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);

            while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
                culv_thread_yield(); // Give CPU back to the scheduler
            }
            // No destruction needed, fiber resumes natively.
        }
    }
}

[[gnu::cold, gnu::noinline, gnu::nonnull(1)]] CULV_MUTEX_INTERNAL void culv_mutex_unlock_slow(CulvMutex *mod) {
    uint8_t val = atomic_load_explicit(&mod->bits, memory_order_relaxed);
    for (;;) {
        uint8_t desired = val & ~CULV_MUTEX_LOCKED;
        if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, desired, memory_order_release,
                                                  memory_order_relaxed)) {
            if (!(val & CULV_MUTEX_HAS_WAITERS)) {
                return;
            }
            break;
        }
    }

    size_t hash    = culv_mutex_hash_address(mod);
    CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];
    CULV_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

    CulvMutexWaiter **curr   = &bucket->head;
    CulvMutexWaiter *to_wake = nullptr;
    bool more       = false;

    while (*curr != nullptr) {
        if ((*curr)->address == mod && to_wake == nullptr) {
            to_wake = *curr;
            *curr   = to_wake->next;
            continue;
        }
        if ((*curr)->address == mod) {
            more = true;
        }
        curr = &((*curr)->next);
    }

    if (!more) {
        val = atomic_load_explicit(&mod->bits, memory_order_relaxed);
        for (;;) {
            if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, val & ~CULV_MUTEX_HAS_WAITERS,
                                                      memory_order_relaxed, memory_order_relaxed)) {
                break;
            }
        }
    }

    if (to_wake != nullptr) {
        atomic_store_explicit(&to_wake->signaled, true, memory_order_release);
        if (!to_wake->fiber) {
            // Only hit the kernel if it's an OS thread waiting
            CULV_THREAD_PLATFORM_COND_SIGNAL(&to_wake->cond);
        }
        // Note: If it IS a fiber, setting 'signaled = true' is enough.
        // The fiber's yield loop will see it and break out.
    }
    CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
}

// ============================================================================
// CulvCond Implementation
// ============================================================================

void culv_cond_wait(CulvCond *condvar, CulvMutex *mod) {
    size_t hash    = culv_mutex_hash_address(condvar);
    CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];

    CulvThread *current_thread = culv_thread_get_current();
    bool is_fiber             = (bool)(current_thread != nullptr && !current_thread->is_main);

    CulvMutexWaiter node = {
        .address = condvar, .next = nullptr, .fiber = (int)is_fiber ? current_thread : nullptr};
    atomic_init(&node.signaled, false);

    if (!is_fiber) {
        CULV_THREAD_PLATFORM_COND_INIT(&node.cond);
    }

    // Fast-path bit: let signalers know someone is waiting
    atomic_store_explicit(&condvar->bits, 1, memory_order_relaxed);

    // Step 1: Enqueue our waiter node FIRST
    CULV_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);
    node.next    = bucket->head;
    bucket->head = &node;

    if (!is_fiber) {
        // === OS THREAD PATH ===
        CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
        culv_mutex_unlock(mod); // Unlock user's mutex

        CULV_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);
        while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
            CULV_THREAD_PLATFORM_COND_WAIT(&node.cond, &bucket->mutex);
        }
        CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
        CULV_THREAD_PLATFORM_COND_DESTROY(&node.cond);
    } else {
        // === FIBER PATH ===
        CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex); // Drop bucket lock immediately
        culv_mutex_unlock(mod);            // Unlock user's mutex

        while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
            culv_thread_yield(); // Yield CPU to other fibers
        }
    }

    // Step 4: Re-acquire the user's mutex before returning
    culv_mutex_lock(mod);
}

void culv_cond_signal(CulvCond *condvar) {
    // Fast path: if no waiters exist, bail out immediately without taking the bucket lock.
    // (This is completely safe due to the Acquire/Release relationship of the user's Mutex).
    if (atomic_load_explicit(&condvar->bits, memory_order_relaxed) == 0) {
        return;
    }

    size_t hash    = culv_mutex_hash_address(condvar);
    CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];
    CULV_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

    CulvMutexWaiter **curr   = &bucket->head;
    CulvMutexWaiter *to_wake = nullptr;
    bool more       = false;

    // Extract the first waiter waiting on this specific CV address
    while (*curr != nullptr) {
        if ((*curr)->address == condvar && to_wake == nullptr) {
            to_wake = *curr;
            *curr   = to_wake->next;
            continue;
        }
        if ((*curr)->address == condvar) {
            more = true;
        }
        curr = &((*curr)->next);
    }

    if (!more) {
        atomic_store_explicit(&condvar->bits, 0, memory_order_relaxed);
    }

    if (to_wake != nullptr) {
        atomic_store_explicit(&to_wake->signaled, true, memory_order_release);
        if (!to_wake->fiber) {
            CULV_THREAD_PLATFORM_COND_SIGNAL(&to_wake->cond);
        }
    }
    CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
}

void culv_cond_broadcast(CulvCond *condvar) {
    if (atomic_load_explicit(&condvar->bits, memory_order_relaxed) == 0) {
        return;
    }

    size_t hash    = culv_mutex_hash_address(condvar);
    CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];
    CULV_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

    CulvMutexWaiter **curr     = &bucket->head;
    CulvMutexWaiter *wake_list = nullptr;

    while (*curr != nullptr) {
        if ((*curr)->address == condvar) {
            CulvMutexWaiter *waiter = *curr;
            *curr          = waiter->next; // Unlink from bucket

            waiter->next = wake_list; // Link to local stack wake_list
            wake_list    = waiter;
        } else {
            curr = &((*curr)->next);
        }
    }

    atomic_store_explicit(&condvar->bits, 0, memory_order_relaxed);

    // Wake all extracted nodes
    while (wake_list != nullptr) {
        CulvMutexWaiter *waiter = wake_list;
        wake_list      = waiter->next;

        atomic_store_explicit(&waiter->signaled, true, memory_order_release);
        if (!waiter->fiber) {
            // Only hit the kernel if it's an OS thread waiting
            CULV_THREAD_PLATFORM_COND_SIGNAL(&waiter->cond);
        }
        // Note: If it IS a fiber, setting 'signaled = true' is enough.
        // The fiber's yield loop will see it and break out.
    }
    CULV_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
}