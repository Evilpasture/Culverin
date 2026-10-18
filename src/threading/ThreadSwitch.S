#include "ThreadAssembly.h"

// ----------------------------------------------------------------------------
// CPH_ThreadSwitch(void **old_sp, void *new_sp)
// ----------------------------------------------------------------------------
    .global CPH_ASM_SYMBOL(CPH_ThreadSwitch)
    .text
    .p2align 4
CPH_ASM_SYMBOL(CPH_ThreadSwitch):
#if defined(CPH_ASM_OS_WINDOWS) && defined(CPH_ASM_ARCH_X64)
    push %rax
    push %rbp; push %rbx; push %rdi; push %rsi
    push %r12; push %r13; push %r14; push %r15
    sub $160, %rsp
    movaps %xmm6,   0(%rsp);  movaps %xmm7,  16(%rsp);  movaps %xmm8,  32(%rsp)
    movaps %xmm9,  48(%rsp);  movaps %xmm10, 64(%rsp);  movaps %xmm11, 80(%rsp)
    movaps %xmm12, 96(%rsp);  movaps %xmm13,112(%rsp);  movaps %xmm14,128(%rsp)
    movaps %xmm15,144(%rsp)
    mov %rsp, (%rcx)
    mov %rdx, %rsp
    movaps 0(%rsp),  %xmm6;   movaps 16(%rsp),  %xmm7;  movaps 32(%rsp),  %xmm8
    movaps 48(%rsp), %xmm9;   movaps 64(%rsp), %xmm10;  movaps 80(%rsp), %xmm11
    movaps 96(%rsp),%xmm12;   movaps 112(%rsp),%xmm13;  movaps 128(%rsp),%xmm14
    movaps 144(%rsp),%xmm15
    add $160, %rsp
    pop %r15; pop %r14; pop %r13; pop %r12
    pop %rsi; pop %rdi; pop %rbx; pop %rbp
    pop %rax
    ret
#elif defined(CPH_ASM_ARCH_X64)
    push %rbp; push %rbx; push %r12; push %r13; push %r14; push %r15
    mov %rsp, (%rdi)
    mov %rsi, %rsp
    pop %r15; pop %r14; pop %r13; pop %r12; pop %rbx; pop %rbp
    ret
#elif defined(CPH_ASM_ARCH_ARM64)
    sub sp, sp, #160
    stp x19, x20, [sp, #0x00]
    stp x21, x22, [sp, #0x10]
    stp x23, x24, [sp, #0x20]
    stp x25, x26, [sp, #0x30]
    stp x27, x28, [sp, #0x40]
    stp x29, x30, [sp, #0x50]
    stp d8,  d9,  [sp, #0x60]
    stp d10, d11, [sp, #0x70]
    stp d12, d13, [sp, #0x80]
    stp d14, d15, [sp, #0x90]
    mov x10, sp
    str x10, [x0]
    mov sp, x1
    ldp x19, x20, [sp, #0x00]
    ldp x21, x22, [sp, #0x10]
    ldp x23, x24, [sp, #0x20]
    ldp x25, x26, [sp, #0x30]
    ldp x27, x28, [sp, #0x40]
    ldp x29, x30, [sp, #0x50]
    ldp d8,  d9,  [sp, #0x60]
    ldp d10, d11, [sp, #0x70]
    ldp d12, d13, [sp, #0x80]
    ldp d14, d15, [sp, #0x90]
    add sp, sp, #160
    ret
#endif
#if defined(CPH_ASM_OS_LINUX)
    .size CPH_ASM_SYMBOL(CPH_ThreadSwitch), .- CPH_ASM_SYMBOL(CPH_ThreadSwitch)
#endif

// ----------------------------------------------------------------------------
// CPH_ThreadTrampolineAsm
// ----------------------------------------------------------------------------
    .global CPH_ASM_SYMBOL(CPH_ThreadTrampolineAsm)
    .text
    .p2align 4
CPH_ASM_SYMBOL(CPH_ThreadTrampolineAsm):
#if defined(CPH_ASM_OS_WINDOWS) && defined(CPH_ASM_ARCH_X64)
    sub $32, %rsp
    call CPH_ASM_SYMBOL(CPH_ThreadTrampoline)
    add $32, %rsp
    ret
#elif defined(CPH_ASM_ARCH_X64)
    sub $8, %rsp
    call CPH_ASM_SYMBOL(CPH_ThreadTrampoline)
    add $8, %rsp
    ret
#elif defined(CPH_ASM_ARCH_ARM64)
    b CPH_ASM_SYMBOL(CPH_ThreadTrampoline)
#endif
#if defined(CPH_ASM_OS_LINUX)
    .size CPH_ASM_SYMBOL(CPH_ThreadTrampolineAsm), .- CPH_ASM_SYMBOL(CPH_ThreadTrampolineAsm)
#endif