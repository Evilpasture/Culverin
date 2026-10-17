#pragma once

#include <stdint.h>

#ifdef __cplusplus
#    include <format>
#    include <iostream>
#    include <source_location>
#    include <string_view>
#endif

// --- FFI Hooks for Jolt/Culverin ---
#ifdef __cplusplus
extern "C" {
#endif
void culv_jph_trace(const char *inString);
bool culv_jph_assert(const char *inExpression, const char *inMessage, const char *inFile,
                     uint32_t inLine);
#ifdef __cplusplus
}
#endif

// ============================================================================
// CONFIGURATION
// ============================================================================
// Level 0: Assume (Aggressive Release - Elides bounds checks, can cause UB if wrong)
// Level 1: Trap (RelWithDebInfo - Crashes instantly, keeps instruction cache clean)
// Level 2: Full Debug (Debug - Full string formatting and stack tracing)

#if !defined(CULV_ASSERT_LEVEL)
#    if defined(NDEBUG)
#        if defined(CULV_AGGRESSIVE_OPTIMIZATION)
#            define CULV_ASSERT_LEVEL 0
#        else
#            define CULV_ASSERT_LEVEL 1
#        endif
#    else
#        define CULV_ASSERT_LEVEL 2
#    endif
#endif

// --- Backend Crash Logic ---
[[gnu::always_inline]] static inline void culv_internal_break() {
#if __has_builtin(__builtin_debugtrap)
    __builtin_debugtrap();
#elif __has_builtin(__builtin_trap)
    __builtin_trap();
#else
    *(volatile int *)0 = 0; // Absolute fallback
#endif
}

#if CULV_ASSERT_LEVEL == 2
// ============================================================================
// LEVEL 2: FULL DEBUG (Rich Formatting & Callbacks)
// ============================================================================

#    ifdef __cplusplus
// --- C++20 Engine ---
namespace culv::detail {
[[noreturn]] inline void
panic(std::string_view expression, std::string_view message = "",
      const std::source_location loc = std::source_location::current()) noexcept {
    const auto msg =
        std::format("\n[ASSERTION FAILED]\n"
                    "  Expr: {}\n"
                    "  Info: {}\n"
                    "  Loc:  {}:{}\n"
                    "  Func: {}\n",
                    expression, message, loc.file_name(), loc.line(), loc.function_name());

    culv_jph_trace(msg.c_str());

    if (culv_jph_assert(expression.data(), message.data(), loc.file_name(), loc.line())) {
        culv_internal_break();
    }

    std::terminate();
}
} // namespace culv::detail

#        define CULV_ASSERT(expr, ...)                                                             \
            do {                                                                                   \
                if (!(expr)) [[unlikely]] {                                                        \
                    if consteval {                                                                 \
                        throw "Assertion failed at compile time";                                  \
                    } else {                                                                       \
                        ::culv::detail::panic(#expr __VA_OPT__(, )                                 \
                                                  __VA_OPT__(std::format(__VA_ARGS__)));           \
                    }                                                                              \
                }                                                                                  \
            } while (false)

#    else
// --- C23 Engine ---
#        include <stdarg.h>
#        include <stdio.h>
#        include <stdlib.h>

[[gnu::always_inline, gnu::format(printf, 5, 6)]] static inline void
culv_internal_panic_c(const char *expr, const char *file, uint32_t line, const char *func,
                      const char *fmt, ...) {
    char user_msg[512] = {0};
    if (fmt != nullptr) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(user_msg, sizeof(user_msg), fmt, args);
        va_end(args);
    }

    fprintf(stderr,
            "\n[ASSERTION FAILED]\n"
            "  Expr: %s\n"
            "  Info: %s\n"
            "  Loc:  %s:%u\n"
            "  Func: %s\n",
            expr, user_msg, file, line, func);

    if (culv_jph_assert(expr, user_msg, file, line)) {
        culv_internal_break();
    }
    abort();
}

// C23 variadic macro trick: The trailing `nullptr` safely acts as the formatting
// string if __VA_ARGS__ is empty, or as an ignored variadic argument if it's not.
#        define CULV_ASSERT(expr, ...)                                                             \
            do {                                                                                   \
                if (!(expr)) [[unlikely]] {                                                        \
                    culv_internal_panic_c(#expr, __FILE__, __LINE__, __func__,                     \
                                          __VA_OPT__(__VA_ARGS__) __VA_OPT__(, ) nullptr);         \
                }                                                                                  \
            } while (false)

#    endif

#elif CULV_ASSERT_LEVEL == 1
// ============================================================================
// LEVEL 1: TRAP ONLY (RelWithDebInfo)
// ============================================================================
// Safe Release Mode: No string bloat in the binary, keeps the instruction cache
// perfectly clean for hot physics loops, but halts immediately on logic errors.

#    define CULV_ASSERT(expr, ...)                                                                 \
        do {                                                                                       \
            if (!(expr)) [[unlikely]] {                                                            \
                culv_internal_break();                                                             \
            }                                                                                      \
        } while (false)

#else
// ============================================================================
// LEVEL 0: ASSUME (Aggressive Release)
// ============================================================================
// Danger Zone: Elides branches completely. Tells the optimizer to assume the
// condition is ALWAYS true. Yields peak performance but causes UB if violated.

#    if defined(__clang__) || defined(__GNUC__)
#        ifdef __cplusplus
#            define CULV_ASSERT(expr, ...)                                                         \
                do {                                                                               \
                    __builtin_assume(static_cast<bool>(expr));                                     \
                } while (false)
#        else
#            define CULV_ASSERT(expr, ...)                                                         \
                do {                                                                               \
                    __builtin_assume((bool)(expr));                                                \
                } while (false)
#        endif
#    else
// Fallback if __builtin_assume is unavailable
#        define CULV_ASSERT(expr, ...) ((void)0)
#    endif

#endif