#pragma once
// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <stddef.h>
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

// Note: In C23, bool, true, and false are native keywords. No <stdbool.h> needed.
// nullptr is also a native keyword.

typedef void (*CPH_ThreadFunc)(void *arg);

typedef struct CPH_Thread CPH_Thread;
static constexpr size_t kThreadAlignment = 128;
struct CPH_Thread {
    alignas(kThreadAlignment) void *stack_pointer;
    void *map_addr;
    size_t map_size;
    CPH_ThreadFunc func;
    void *arg;
    CPH_Thread *caller;
#if defined(_WIN32)
    void *stack_base;
    void *stack_limit;
#endif
    bool is_finished;
    bool is_main;
};

#ifdef __cplusplus
extern "C" {
#endif

void CPH_ThreadInitMain(void);
CPH_Thread *CPH_ThreadCreate(size_t stack_size, CPH_ThreadFunc func, void *arg);
void CPH_ThreadResume(CPH_Thread *target);
void CPH_ThreadYield(void);
void CPH_ThreadDestroy(CPH_Thread *thread);
CPH_Thread *CPH_ThreadGetCurrent(void);

#ifdef __cplusplus
} // extern "C"
#endif
