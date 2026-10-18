#pragma once

// Architecture detection
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#    define CPH_ASM_ARCH_X64
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
#    define CPH_ASM_ARCH_ARM64
#endif

// OS detection
#if defined(_WIN32)
#    define CPH_ASM_OS_WINDOWS
#elif defined(__APPLE__)
#    define CPH_ASM_OS_APPLE
#else
#    define CPH_ASM_OS_LINUX
#endif

// Direct Apple mangling
#if defined(CPH_ASM_OS_APPLE)
#    define CPH_ASM_SYMBOL(name) _##name
#else
#    define CPH_ASM_SYMBOL(name) name
#endif
