#pragma once
// NOLINTBEGIN(llvmlibc-restrict-system-libc-headers)
#include <stddef.h>
// NOLINTEND(llvmlibc-restrict-system-libc-headers)

// Note: In C23, bool, true, and false are native keywords. No <stdbool.h> needed.
// nullptr is also a native keyword.

typedef void (*CulvThreadFunc)(void *arg);

typedef struct CulvThread CulvThread;
static constexpr size_t CULV_THREAD_ALIGNMENT = 128;
struct CulvThread {
    alignas(CULV_THREAD_ALIGNMENT) void *stack_pointer;
    void *map_addr;
    size_t map_size;
    CulvThreadFunc func;
    void *arg;
    CulvThread *caller;
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

void culv_thread_init_main(void);
CulvThread *culv_thread_create(size_t stack_size, CulvThreadFunc func, void *arg);
void culv_thread_resume(CulvThread *target);
void culv_thread_yield(void);
void culv_thread_destroy(CulvThread *thread);
CulvThread *culv_thread_get_current(void);

#ifdef __cplusplus
} // extern "C"
#endif
