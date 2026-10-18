#include "Thread.h"
// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <stddef.h>
#include <stdint.h> // For uintptr_t
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    define CPH_WIN32_LEAN_AND_MEAN
#    include <windows.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

// Prototypes for functions defined in ThreadSwitch.S
extern void culv_thread_switch(void **old_sp, void *new_sp);
extern void culv_thread_trampoline_asm(void);
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local CPH_CulvThread culv_main_thread;
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static thread_local CPH_CulvThread *culv_current_thread = nullptr;
// NOLINTNEXTLINE(misc-use-internal-linkage)
void CPH_CulvThreadTrampoline(void);
// Real C entry point called by Culverin thread switch assembly
void CPH_CulvThreadTrampoline(void) {
    CPH_CulvThread *self = culv_current_thread;
    if (self->func) {
        self->func(self->arg);
    }
    self->is_finished = true;
    // NOLINTNEXTLINE(altera-unroll-loops)
    while (true) {
        CPH_CulvThreadYield();
    }
}

static inline void culv_thread_swap_teb(CPH_CulvThread *target) {
#if defined(_WIN32)
    NT_TIB *tib                   = (NT_TIB *)NtCurrentTeb();
    culv_current_thread->stack_base  = tib->StackBase;
    culv_current_thread->stack_limit = tib->StackLimit;
    tib->StackBase                = target->stack_base;
    tib->StackLimit               = target->stack_limit;
#else
    (void)target;
#endif
}

static size_t culv_thread_get_page_size(void) {
#if defined(_WIN32)
    SYSTEM_INFO sys_info = {};
    GetSystemInfo(&sys_info);
    return (size_t)sys_info.dwPageSize;
#else
    long size = sysconf(_SC_PAGESIZE);
    // If sysconf fails, it returns -1. We provide a sane 4KB default.
    return (size > 0) ? (size_t)size : 4096;
#endif
}

void CPH_CulvThreadInitMain(void) {
    culv_main_thread.is_finished = false;
    culv_main_thread.is_main     = true;
    culv_main_thread.caller      = nullptr;
#if defined(_WIN32)
    NT_TIB *tib               = (NT_TIB *)NtCurrentTeb();
    culv_main_thread.stack_base  = tib->StackBase;
    culv_main_thread.stack_limit = tib->StackLimit;
#endif
    culv_current_thread = &culv_main_thread;
}

CPH_CulvThread *CPH_CulvThreadCreate(size_t stack_size, CulvThreadFunc func, void *arg) {
    size_t page_size         = culv_thread_get_page_size();
    constexpr size_t CULV_THREAD_ONE_MIB = 1024ULL * 1024ULL;
    if (stack_size == 0) {
        stack_size = CULV_THREAD_ONE_MIB;
    }
    stack_size        = (stack_size + page_size - 1) & ~(page_size - 1);
    size_t total_size = stack_size + ((long long)page_size * 2);

    void *map = nullptr;
#if defined(_WIN32)
    map = VirtualAlloc(nullptr, total_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!map) {
        return nullptr;
    }
    DWORD old = 0;
    VirtualProtect(map, page_size, PAGE_READWRITE | PAGE_GUARD, &old);
#else
    map = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED) {
        return nullptr;
    }
    mprotect(map, page_size, PROT_NONE);
#endif
    constexpr size_t CULV_THREAD_STACK_ALIGNMENT  = 16ULL;
    constexpr size_t CULV_THREAD_STACK_ALIGN_MASK = CULV_THREAD_STACK_ALIGNMENT - 1ULL;
    uintptr_t end_addr                = (uintptr_t)map + total_size;
    uintptr_t thread_addr             = (end_addr - sizeof(CPH_CulvThread)) & ~CULV_THREAD_STACK_ALIGN_MASK;
    CPH_CulvThread *thread                 = nullptr;
    memcpy((void *)&thread, &thread_addr, sizeof(CPH_CulvThread *));

    thread->map_addr    = map;
    thread->map_size    = total_size;
    thread->func        = func;
    thread->arg         = arg;
    thread->is_finished = false;
    thread->is_main     = false;

#if defined(_WIN32)
    uintptr_t limit_addr = (uintptr_t)map + page_size;

    // Copy the bits of the integer directly into the pointer
    memcpy((void *)&thread->stack_base, &end_addr, sizeof(void *));
    memcpy((void *)&thread->stack_limit, &limit_addr, sizeof(void *));
#endif

    uintptr_t stack_pointer = (uintptr_t)thread;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
    static_assert(
        alignof(CPH_CulvThread) % 16 == 0,
        "CulvThread struct alignment must be a multiple of 16 to ensure the stack starts aligned");
    constexpr size_t GPR_SIZE = 8ULL; // Each general-purpose register is 8 bytes
#if defined(_WIN32) && (defined(__x86_64__) || defined(_M_X64))
                                      // Symmetry: 8 (Ret) + 8 (Dummy) + 64 (GPR) + 160 (XMM) = 240
    constexpr size_t WIN64_CTX_SZ = 232ULL; // Everything except the Ret addr
    constexpr size_t WIN64_RET_SZ = 8ULL;
    constexpr size_t WIN64_TOTAL  = WIN64_CTX_SZ + WIN64_RET_SZ; // 240

    // Using the constants defined in the previous working block:
    // NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)
    static_assert(WIN64_TOTAL % 16 == 0,
                  "Windows x64 total stack frame must be a multiple of 16 for movaps alignment");

    static_assert(WIN64_TOTAL == 240,
                  "Windows x64 stack frame size changed; check ThreadSwitch.S synchronization");

    static_assert((WIN64_CTX_SZ + 8) == WIN64_TOTAL,
                  "Windows x64 Context + Return Address mismatch");
    // NOLINTEND(cppcoreguidelines-avoid-magic-numbers, readability-magic-numbers)

    void *dest_ptr         = nullptr;
    uintptr_t val_to_write = 0;

    // 1. Align stack pointer (thread starts 16-aligned)
    // 0x...00 - 240 (0xF0) = 0x...10 (Still 16-aligned!)
    stack_pointer -= WIN64_TOTAL;

    // 2. The 'ret' in culv_thread_switch happens after 'add 160' and 'pop 9' (160+72=232)
    uintptr_t ret_addr_loc = stack_pointer + WIN64_CTX_SZ;
    val_to_write           = (uintptr_t)culv_thread_trampoline_asm;
    memcpy((void *)&dest_ptr, &ret_addr_loc, sizeof(void *));
    memcpy(dest_ptr, &val_to_write, sizeof(uintptr_t));

    // 3. Zero the context
    memcpy((void *)&dest_ptr, &stack_pointer, sizeof(void *));
    memset(dest_ptr, 0, WIN64_CTX_SZ);

    // Save the pointer
    memcpy((void *)&thread->stack_pointer, &stack_pointer, sizeof(void *));
#elif defined(__x86_64__) || defined(_M_X64)
    [[maybe_unused]] constexpr size_t SYSV_GPR_STORAGE = 48ULL; // 6 registers (rbx, rbp, r12-r15)

    static_assert(SYSV_GPR_STORAGE % 8 == 0, "System V GPR storage must be 8-byte multiples");
    // System V ABI (Linux/macOS): Only 6 GPRs are callee-saved
    void *dest_ptr;
    uintptr_t val_to_write;

    // 1. Set Trampoline (Initial RIP)
    stack_pointer -= GPR_SIZE;
    val_to_write = (uintptr_t)culv_thread_trampoline_asm;
    memcpy((void *)&dest_ptr, &stack_pointer, sizeof(void *));
    memcpy(dest_ptr, &val_to_write, GPR_SIZE);

    // 2. Zero the saved register area (RBX, RBP, R12-R15)
    stack_pointer -= SYSV_GPR_STORAGE;
    memcpy((void *)&dest_ptr, &stack_pointer, sizeof(void *));
    memset(dest_ptr, 0, SYSV_GPR_STORAGE);

#elif defined(__aarch64__) || defined(_M_ARM64)
    [[maybe_unused]] constexpr size_t ARM64_CONTEXT_SZ = 160ULL;
    // ARM64: Save area for X19-X30 and D8-D15
    void *dest_ptr;
    uintptr_t val_to_write;

    // 1. Zero out the context save area
    stack_pointer -= ARM64_CONTEXT_SZ;
    memcpy((void *)&dest_ptr, &stack_pointer, sizeof(void *));
    memset(dest_ptr, 0, ARM64_CONTEXT_SZ);

    // 2. Set Trampoline into the Link Register (x30) slot
    // Offset 88 is typically where x30 (LR) is stored in the save block
    uintptr_t lr_offset = 11ULL * GPR_SIZE;
    uintptr_t lr_addr   = stack_pointer + lr_offset;
    val_to_write        = (uintptr_t)culv_thread_trampoline_asm;

    memcpy((void *)&dest_ptr, &lr_addr, sizeof(void *));
    memcpy(dest_ptr, &val_to_write, GPR_SIZE);
#endif

    // Use memcpy to avoid the provenance/optimization warning
    memcpy((void *)&thread->stack_pointer, &stack_pointer, sizeof(void *));

    return thread;
}

void CPH_CulvThreadResume(CPH_CulvThread *target) {
    CPH_CulvThread *self = culv_current_thread;
    target->caller  = self;
    culv_thread_swap_teb(target);
    culv_current_thread = target;
    culv_thread_switch(&self->stack_pointer, target->stack_pointer);
}

void CPH_CulvThreadYield(void) {
    CPH_CulvThread *self   = culv_current_thread;
    CPH_CulvThread *target = self->caller;
    if (!target) {
        return;
    }
    culv_thread_swap_teb(target);
    culv_current_thread = target;
    culv_thread_switch(&self->stack_pointer, target->stack_pointer);
}

void CPH_CulvThreadDestroy(CPH_CulvThread *thread) {
    if (!thread) {
        return;
    }
#if defined(_WIN32)
    VirtualFree(thread->map_addr, 0, MEM_RELEASE);
#else
    munmap(thread->map_addr, thread->map_size);
#endif
}

CPH_CulvThread *CPH_CulvThreadGetCurrent(void) { return culv_current_thread; }