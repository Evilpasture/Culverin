#pragma once
// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <stddef.h>
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

// Note: In C23, bool, true, and false are native keywords. No <stdbool.h> needed.
// nullptr is also a native keyword.

typedef void (*CulvThreadFunc)(void *arg);

typedef struct CPH_CulvThread CPH_CulvThread;
static constexpr size_t kThreadAlignment = 128;
struct CPH_CulvThread {
    alignas(kThreadAlignment) void *stack_pointer;
    void *map_addr;
    size_t map_size;
    CulvThreadFunc func;
    void *arg;
    CPH_CulvThread *caller;
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

void CPH_CulvThreadInitMain(void);
CPH_CulvThread *CPH_CulvThreadCreate(size_t stack_size, CulvThreadFunc func, void *arg);
void CPH_CulvThreadResume(CPH_CulvThread *target);
void CPH_CulvThreadYield(void);
void CPH_CulvThreadDestroy(CPH_CulvThread *thread);
CPH_CulvThread *CPH_CulvThreadGetCurrent(void);

#ifdef __cplusplus
} // extern "C"
#endif
