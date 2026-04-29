#include "culverin_handler.h"
#include <array>
#include <format>
#include <iostream>
#include <string_view>
#include <ctime>

#ifdef _WIN32
#    include <io.h>
#    include <windows.h>
#    define IS_ATTY _isatty(_fileno(stderr))
#else
#    include <unistd.h>
#    define IS_ATTY isatty(fileno(stderr))
#endif

namespace {
    enum class LogLevel : uint8_t { Info, Warn, Error };

    struct LogTheme {
        std::string_view color;
        std::string_view tag;
    };

     constexpr std::string_view ANSI_RESET = "\033[0m";

    [[nodiscard]]
    constexpr auto get_theme(LogLevel level, bool use_color) noexcept -> LogTheme {
        if (!use_color) {
            switch (level) {
                using enum LogLevel;
                case Error: return { .color = "", .tag = "ERROR" };
                case Warn:  return { .color = "", .tag = "WARN " };
                default:    return { .color = "", .tag = "INFO " };
            }
        }

        switch (level) {
            using enum LogLevel;
            case Error: return { .color = "\033[1;31m", .tag = "ERROR" };
            case Warn:  return { .color = "\033[1;33m", .tag = "WARN " };
            default:    return { .color = "\033[0;36m", .tag = "INFO " };
        }
    }

    [[nodiscard]]
    auto detect_level(std::string_view msg) noexcept -> LogLevel {
        // Use string_view::find instead of .contains for older SDK compatibility if needed, 
        // but .find is perfectly systems-safe.
        if (msg.contains("Error") || msg.contains("fail")) {
            return LogLevel::Error;
        }
        if (msg.contains("Warning")) {
            return LogLevel::Warn;
        }
        return LogLevel::Info;
    }

    /**
     * WORKAROUND: Manual Time Formatting
     * Bypasses the macOS 13.3 'to_chars' floating point SDK bug 
     * by using integer breakdown.
     */
    struct TimeSpec { int h, m, s; };
    [[nodiscard]]
    auto get_local_time() noexcept -> TimeSpec {
        const auto now = std::time(nullptr);
        std::tm tm_info{};
#ifdef _WIN32
        localtime_s(&tm_info, &now);
#else
        localtime_r(&now, &tm_info);
#endif
        return { .h=tm_info.tm_hour, .m=tm_info.tm_min, .s=tm_info.tm_sec };
    }
}

extern "C" {

/**
 * JPH Trace Bridge
 */
void culv_jph_trace(const char* const inString) {
    if (inString == nullptr || *inString == '\0') [[unlikely]] {
        return;
    }

    auto msg = std::string_view{ inString };

    // 1. Clean Jolt's trailing whitespace
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r' || msg.back() == ' ')) {
        msg.remove_suffix(1);
    }

    if (msg.empty()) [[unlikely]] {
        return;
    }

    // 2. Metadata extraction
    const auto ts        = get_local_time();
    const auto level     = detect_level(msg);
    const auto atty      = static_cast<bool>(IS_ATTY);
    const auto theme     = get_theme(level, atty);

    // 3. Stack-allocated formatting (Non-throwing, Zero-allocation)
    std::array<char, 2048> buffer;
    
    // We use integer arguments for the timestamp to avoid triggering 
    // the libc++ floating-point formatter templates.
    const auto result = std::format_to_n(
        buffer.data(), 
        buffer.size(),
        "[{:02d}:{:02d}:{:02d}] {}{}{}{} {}\n",
        ts.h, ts.m, ts.s,
        theme.color, theme.tag, (atty ? ANSI_RESET : ""), 
        (atty ? "" : ":"),
        msg
    );

    // 4. Output safely with signed/unsigned fix
    const auto total_size = static_cast<size_t>(result.size);
    const size_t write_size = (total_size > buffer.size()) ? buffer.size() : total_size;
    
    std::cerr.write(buffer.data(), static_cast<std::streamsize>(write_size));
}

/**
 * JPH Assert Bridge
 */
[[nodiscard]]
auto culv_jph_assert(const char* const inExpression, const char* const inMessage, 
                     const char* const inFile, const uint32_t inLine) -> bool {
    
    const auto atty  = static_cast<bool>(IS_ATTY);
    const auto *const color = atty ? "\033[1;41m" : ""; 
    const auto reset = atty ? ANSI_RESET : "";

    std::array<char, 1024> buffer;
    const auto result = std::format_to_n(
        buffer.data(),
        buffer.size(),
        "\n{}[JOLT ASSERTION FAILURE]{}\n"
        "  Expression: {}\n"
        "  Message:    {}\n"
        "  Location:   {}:{}\n",
        color, reset,
        (inExpression != nullptr) ? inExpression : "N/A",
        (inMessage != nullptr) ? inMessage : "No message provided",
        (inFile != nullptr) ? inFile : "Unknown",
        inLine
    );

    const size_t total_size = static_cast<size_t>(result.size);
    const size_t write_size = (total_size > buffer.size()) ? buffer.size() : total_size;
    
    std::cerr.write(buffer.data(), static_cast<std::streamsize>(write_size));
    std::cerr.flush();

    return true;
}

} // extern "C"