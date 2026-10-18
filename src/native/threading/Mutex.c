#include "Mutex.h"
#include "Thread.h"
// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <stdlib.h>
#include <time.h>
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

static constexpr int kMutexMaxSpinCount = 40;

// Address sharding for the parking lot to eliminate false-sharing lock
// contention. 256 buckets padded to Apple Silicon / Modern x86 strict
// cache-line widths.
static constexpr size_t kMutexBucketCount = 256;

#ifndef CPH_MUTEX_CACHELINE_ALIGNED
#    if defined(_MSC_VER)
#        define CPH_MUTEX_CACHELINE_ALIGNED __declspec(align(128))
#    elif defined(__GNUC__) || defined(__clang__)
#        define CPH_MUTEX_CACHELINE_ALIGNED __attribute__((aligned(128)))
#    else
#        define CPH_MUTEX_CACHELINE_ALIGNED
#    endif
#endif

static_assert(ATOMIC_INT_LOCK_FREE == 2, "CulvMutex requires always-lock-free atomics!");
static_assert(ATOMIC_POINTER_LOCK_FREE == 2, "CulvMutex parking lot requires lock-free pointers!");

#ifdef CPH_MUTEX_DEBUG
static constexpr size_t kMutexMaxDebugEdges = 4096;
static constexpr size_t kMutexMaxHeldLocks  = 64;

static culv_platform_mutex_t culv_mutex_debug_graph_lock;
static struct {
    const CPH_CulvMutex *from;
    const CPH_CulvMutex *to;
} culv_mutex_debug_lock_edges[kMutexMaxDebugEdges];
static size_t culv_mutex_debug_lock_edge_count = 0;

static thread_local const CPH_CulvMutex *culv_mutex_debug_held_locks[kMutexMaxHeldLocks];
static thread_local size_t culv_mutex_debug_held_lock_count = 0;

static inline uintptr_t culv_mutex_get_current_id(void) {
    CPH_CulvThread *fiber = CPH_CulvThreadGetCurrent();
    // If we are in a fiber context, the fiber pointer is our unique ID.
    if (fiber) {
        return (uintptr_t)fiber;
    }
    // Fallback for raw OS threads not managed by CulvThread
    return (uintptr_t)CPH_THREAD_PLATFORM_CURRENT_ID();
}

static bool culv_mutex_debug_dfs_check_cycle(const CPH_CulvMutex *current, const CPH_CulvMutex *target) {
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

static void culv_mutex_debug_add_lock_edge(const CPH_CulvMutex *from, const CPH_CulvMutex *to) {
    CPH_THREAD_PLATFORM_MUTEX_LOCK(&culv_mutex_debug_graph_lock);

    for (size_t i = 0; i < culv_mutex_debug_lock_edge_count; i++) {
        if (culv_mutex_debug_lock_edges[i].from == from && culv_mutex_debug_lock_edges[i].to == to) {
            CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&culv_mutex_debug_graph_lock);
            return;
        }
    }

    if (culv_mutex_debug_dfs_check_cycle(to, from)) {
        CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&culv_mutex_debug_graph_lock);
        assert(false && "DEADLOCK DETECTED: Lock order cycle/inversion!");
        abort();
    }

    if (culv_mutex_debug_lock_edge_count < kMutexMaxDebugEdges) {
        culv_mutex_debug_lock_edges[culv_mutex_debug_lock_edge_count].from = from;
        culv_mutex_debug_lock_edges[culv_mutex_debug_lock_edge_count].to   = to;
        culv_mutex_debug_lock_edge_count++;
    }
    CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&culv_mutex_debug_graph_lock);
}

void culv_mutex_debug_clear_owner(CPH_CulvMutex *mod) {
    atomic_store_explicit(&mod->has_owner, false, memory_order_release);
}

void culv_mutex_debug_check_pre_lock(CPH_CulvMutex *mod) {
    uint8_t bits = atomic_load_explicit(&mod->bits, memory_order_relaxed);
    if (bits & kMutexPoisoned) {
        assert(false && "FATAL: Attempting to lock a POISONED mutex!");
        abort();
    }

    if (bits & kMutexLocked) {
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

void culv_mutex_debug_post_lock(CPH_CulvMutex *mod) {
    atomic_store_explicit(&mod->owner, culv_mutex_get_current_id(), memory_order_relaxed);
    atomic_store_explicit(&mod->has_owner, true, memory_order_release);

    for (size_t i = 0; i < culv_mutex_debug_held_lock_count; i++) {
        culv_mutex_debug_add_lock_edge(culv_mutex_debug_held_locks[i], mod);
    }
    if (culv_mutex_debug_held_lock_count < kMutexMaxHeldLocks) {
        culv_mutex_debug_held_locks[culv_mutex_debug_held_lock_count++] = mod;
    }
}

void culv_mutex_debug_pre_unlock(CPH_CulvMutex *mod) {
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

typedef struct CPH_CulvMutexWaiter CPH_CulvMutexWaiter;
static constexpr size_t kMutexWaiterAlignment = 64;
struct CPH_CulvMutexWaiter {
    alignas(kMutexWaiterAlignment)
        // 8-byte members grouped together
        const void *address;
    CPH_CulvThread *fiber;
    CPH_CulvMutexWaiter *next;

    // culv_platform_cond_t size varies by OS, but we keep it here
    culv_platform_cond_t cond;

    // 1-byte atomic at the end
    CPH_MUTEX_ATOMIC(bool) signaled;
};

// Padded 128-byte cacheline bucket to completely eliminate false sharing.
static constexpr size_t kMutexBucketAlignment = 128;
typedef struct {
    alignas(kMutexBucketAlignment) culv_platform_mutex_t mutex;
    CPH_CulvMutexWaiter *head;
    uint8_t _pad[kMutexBucketAlignment - (sizeof(culv_platform_mutex_t) + sizeof(void *))];
} CPH_CulvMutexBucket;

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
alignas(kMutexBucketAlignment) static CPH_CulvMutexBucket culv_mutex_parking_lot[kMutexBucketCount];

static_assert(sizeof(CPH_CulvMutexBucket) == kMutexBucketAlignment, "CulvMutexBucket must be exactly 128 bytes!");

[[gnu::constructor]]
static void culv_mutex_init_parking_lot() {
#pragma unroll 4
    for (size_t i = 0; i < kMutexBucketCount; i++) {
        CPH_THREAD_PLATFORM_MUTEX_INIT(&culv_mutex_parking_lot[i].mutex);
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

    return (size_t)(hash_val % kMutexBucketCount);
}

[[gnu::cold, gnu::noinline, gnu::nonnull(1)]] CPH_MUTEX_INTERNAL void CPH_CulvMutexLockSlow(CPH_CulvMutex *mod) {
    size_t hash    = culv_mutex_hash_address(mod);
    CPH_CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];

    // --- PHASE 1: Adaptive Exponential Backoff Spin ---
    // We increase the number of culv_cpu_relax() calls per iteration.
    // On ARM64/Apple Silicon, 'yield' is very cheap, so we need a significant count.
    size_t backoff_limit = 1;

    for (int i = 0; i < kMutexMaxSpinCount; i++) {
        uint8_t val = atomic_load_explicit(&mod->bits, memory_order_relaxed);

        // If the lock appears free, try to grab it.
        if (!(val & kMutexLocked)) {
            if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, val | kMutexLocked,
                                                      memory_order_acquire, memory_order_relaxed)) {
                culv_mutex_debug_post_lock(mod);
#ifdef CPH_MUTEX_DEBUG
                atomic_fetch_add_explicit(&mod->spin_success_count, 1, memory_order_relaxed);
#endif
                return;
            }
        }

        if (CPH_MUTEX_UNLIKELY(val & kMutexPoisoned)) {
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
        if (!(val & kMutexLocked)) {
            if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, val | kMutexLocked,
                                                      memory_order_acquire, memory_order_relaxed)) {
                culv_mutex_debug_post_lock(mod);
                return;
            }
            continue;
        }

        // Ensure the CULV_MUTEX_HAS_WAITERS bit is set before we actually sleep.
        if (!(val & kMutexHasWaiters)) {
            if (!atomic_compare_exchange_weak_explicit(&mod->bits, &val, val | kMutexHasWaiters,
                                                       memory_order_relaxed,
                                                       memory_order_relaxed)) {
                continue;
            }
        }

#ifdef CPH_MUTEX_DEBUG
        atomic_fetch_add_explicit(&mod->park_count, 1, memory_order_relaxed);
#endif

        CPH_CulvThread *current_thread = CPH_CulvThreadGetCurrent();
        bool is_fiber             = (bool)(current_thread != nullptr && !current_thread->is_main);

        CPH_CulvMutexWaiter node = {
            .address = mod, .next = nullptr, .fiber = (int)is_fiber ? current_thread : nullptr};
        atomic_init(&node.signaled, false);

        if (!is_fiber) {
            CPH_THREAD_PLATFORM_COND_INIT(&node.cond); // Only init OS condvar for OS threads
        }

        CPH_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

        val = atomic_load_explicit(&mod->bits, memory_order_relaxed);
        if (CPH_MUTEX_UNLIKELY(!(val & kMutexLocked) || !(val & kMutexHasWaiters))) {
            CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
            if (!is_fiber) {
                CPH_THREAD_PLATFORM_COND_DESTROY(&node.cond);
            }
            continue;
        }

        node.next    = bucket->head;
        bucket->head = &node;

        if (!is_fiber) {
            // === OS THREAD SLEEP ===
            while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
                CPH_THREAD_PLATFORM_COND_WAIT(&node.cond, &bucket->mutex);
            }
            CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
            CPH_THREAD_PLATFORM_COND_DESTROY(&node.cond);
        } else {
            // === FIBER YIELD ===
            // Fibers don't need the bucket lock to wait!
            // We drop it immediately so other threads/fibers can access the bucket.
            CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);

            while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
                CPH_CulvThreadYield(); // Give CPU back to the scheduler
            }
            // No destruction needed, fiber resumes natively.
        }
    }
}

[[gnu::cold, gnu::noinline, gnu::nonnull(1)]] CPH_MUTEX_INTERNAL void CPH_CulvMutexUnlockSlow(CPH_CulvMutex *mod) {
    uint8_t val = atomic_load_explicit(&mod->bits, memory_order_relaxed);
    for (;;) {
        uint8_t desired = val & ~kMutexLocked;
        if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, desired, memory_order_release,
                                                  memory_order_relaxed)) {
            if (!(val & kMutexHasWaiters)) {
                return;
            }
            break;
        }
    }

    size_t hash    = culv_mutex_hash_address(mod);
    CPH_CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];
    CPH_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

    CPH_CulvMutexWaiter **curr   = &bucket->head;
    CPH_CulvMutexWaiter *to_wake = nullptr;
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
            if (atomic_compare_exchange_weak_explicit(&mod->bits, &val, val & ~kMutexHasWaiters,
                                                      memory_order_relaxed, memory_order_relaxed)) {
                break;
            }
        }
    }

    if (to_wake != nullptr) {
        atomic_store_explicit(&to_wake->signaled, true, memory_order_release);
        if (!to_wake->fiber) {
            // Only hit the kernel if it's an OS thread waiting
            CPH_THREAD_PLATFORM_COND_SIGNAL(&to_wake->cond);
        }
        // Note: If it IS a fiber, setting 'signaled = true' is enough.
        // The fiber's yield loop will see it and break out.
    }
    CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
}

// ============================================================================
// CulvCond Implementation
// ============================================================================

void CPH_CulvCondWait(CPH_CulvCond *condvar, CPH_CulvMutex *mod) {
    size_t hash    = culv_mutex_hash_address(condvar);
    CPH_CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];

    CPH_CulvThread *current_thread = CPH_CulvThreadGetCurrent();
    bool is_fiber             = (bool)(current_thread != nullptr && !current_thread->is_main);

    CPH_CulvMutexWaiter node = {
        .address = condvar, .next = nullptr, .fiber = (int)is_fiber ? current_thread : nullptr};
    atomic_init(&node.signaled, false);

    if (!is_fiber) {
        CPH_THREAD_PLATFORM_COND_INIT(&node.cond);
    }

    // Fast-path bit: let signalers know someone is waiting
    atomic_store_explicit(&condvar->bits, 1, memory_order_relaxed);

    // Step 1: Enqueue our waiter node FIRST
    CPH_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);
    node.next    = bucket->head;
    bucket->head = &node;

    if (!is_fiber) {
        // === OS THREAD PATH ===
        CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
        culv_mutex_unlock(mod); // Unlock user's mutex

        CPH_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);
        while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
            CPH_THREAD_PLATFORM_COND_WAIT(&node.cond, &bucket->mutex);
        }
        CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
        CPH_THREAD_PLATFORM_COND_DESTROY(&node.cond);
    } else {
        // === FIBER PATH ===
        CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex); // Drop bucket lock immediately
        culv_mutex_unlock(mod);            // Unlock user's mutex

        while (!atomic_load_explicit(&node.signaled, memory_order_acquire)) {
            CPH_CulvThreadYield(); // Yield CPU to other fibers
        }
    }

    // Step 4: Re-acquire the user's mutex before returning
    culv_mutex_lock(mod);
}

void CPH_CulvCondSignal(CPH_CulvCond *condvar) {
    // Fast path: if no waiters exist, bail out immediately without taking the bucket lock.
    // (This is completely safe due to the Acquire/Release relationship of the user's Mutex).
    if (atomic_load_explicit(&condvar->bits, memory_order_relaxed) == 0) {
        return;
    }

    size_t hash    = culv_mutex_hash_address(condvar);
    CPH_CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];
    CPH_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

    CPH_CulvMutexWaiter **curr   = &bucket->head;
    CPH_CulvMutexWaiter *to_wake = nullptr;
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
            CPH_THREAD_PLATFORM_COND_SIGNAL(&to_wake->cond);
        }
    }
    CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
}

void CPH_CulvCondBroadcast(CPH_CulvCond *condvar) {
    if (atomic_load_explicit(&condvar->bits, memory_order_relaxed) == 0) {
        return;
    }

    size_t hash    = culv_mutex_hash_address(condvar);
    CPH_CulvMutexBucket *bucket = &culv_mutex_parking_lot[hash];
    CPH_THREAD_PLATFORM_MUTEX_LOCK(&bucket->mutex);

    CPH_CulvMutexWaiter **curr     = &bucket->head;
    CPH_CulvMutexWaiter *wake_list = nullptr;

    while (*curr != nullptr) {
        if ((*curr)->address == condvar) {
            CPH_CulvMutexWaiter *waiter = *curr;
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
        CPH_CulvMutexWaiter *waiter = wake_list;
        wake_list      = waiter->next;

        atomic_store_explicit(&waiter->signaled, true, memory_order_release);
        if (!waiter->fiber) {
            // Only hit the kernel if it's an OS thread waiting
            CPH_THREAD_PLATFORM_COND_SIGNAL(&waiter->cond);
        }
        // Note: If it IS a fiber, setting 'signaled = true' is enough.
        // The fiber's yield loop will see it and break out.
    }
    CPH_THREAD_PLATFORM_MUTEX_UNLOCK(&bucket->mutex);
}