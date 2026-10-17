#include "culverin_handler.h"
#include "CulverinCPP"
#include <array>
#include <chrono>
#include <format>
#include <print>
#include <string_view>

#ifdef _WIN32
#    include <io.h>
#    include <windows.h>
#    define IS_ATTY _isatty(_fileno(stderr))
#else
#    include <unistd.h>
#    define IS_ATTY isatty(fileno(stderr))
#endif

namespace {
enum class LogLevel : CPH::Unsigned8 { Info, Warn, Error };

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
        case Error:
            return {.color = "", .tag = "ERROR"};
        case Warn:
            return {.color = "", .tag = "WARN "};
        default:
            return {.color = "", .tag = "INFO "};
        }
    }

    switch (level) {
        using enum LogLevel;
    case Error:
        return {.color = "\033[1;31m", .tag = "ERROR"};
    case Warn:
        return {.color = "\033[1;33m", .tag = "WARN "};
    default:
        return {.color = "\033[0;36m", .tag = "INFO "};
    }
}

[[nodiscard]]
auto detect_level(std::string_view msg) noexcept -> LogLevel {
    if (msg.contains("Error") || msg.contains("fail")) {
        return LogLevel::Error;
    }
    if (msg.contains("Warning")) {
        return LogLevel::Warn;
    }
    return LogLevel::Info;
}
} // namespace

extern "C" {

/**
 * JPH Trace Bridge
 */
void culv_jph_trace(const char *const inString) {
    if (inString == nullptr || *inString == '\0') [[unlikely]] {
        return;
    }

    auto msg = std::string_view{inString};

    // 1. Clean Jolt's trailing whitespace/newlines
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r' || msg.back() == ' ')) {
        msg.remove_suffix(1);
    }

    if (msg.empty()) [[unlikely]] {
        return;
    }

    // 2. Metadata extraction
    const auto level        = detect_level(msg);
    const auto atty         = static_cast<bool>(IS_ATTY);
    const auto [color, tag] = get_theme(level, atty);

    // libc++'s chrono formatter pulls in floating-point to_chars, which requires macOS 13.4.
    // Keep the project's 13.3 deployment target by formatting the UTC clock directly.
    const auto now_seconds = std::chrono::floor<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch());
    const auto day_start = std::chrono::floor<std::chrono::days>(now_seconds);
    const std::chrono::hh_mm_ss time_of_day(now_seconds - day_start);
    const auto hours   = static_cast<unsigned>(time_of_day.hours().count());
    const auto minutes = static_cast<unsigned>(time_of_day.minutes().count());
    const auto seconds = static_cast<unsigned>(time_of_day.seconds().count());
    const std::array<char, 9> timestamp = {
        static_cast<char>('0' + (hours / 10)), static_cast<char>('0' + (hours % 10)), ':',
        static_cast<char>('0' + (minutes / 10)), static_cast<char>('0' + (minutes % 10)), ':',
        static_cast<char>('0' + (seconds / 10)), static_cast<char>('0' + (seconds % 10)), '\0'};

    // 3. Stack-allocated formatting (Non-throwing, Zero-allocation)
    // 2048 is usually enough for any physics trace or error message
    std::array<char, 2048> buffer;

    // std::format_to_n returns a result object containing a pointer to the end
    // and the total size that would have been written. It does not throw.
    const auto result =
        std::format_to_n(buffer.data(), buffer.size(), "[{}] {}{}{}{} {}\n",
                         std::string_view(timestamp.data(), timestamp.size() - 1), color, tag,
                         (atty ? ANSI_RESET : ""), (atty ? "" : ":"), msg);

    // 4. Output safely
    // Clamp the size to the buffer capacity
    const CPH::SizeType write_size =
        (static_cast<unsigned long>(result.size) > buffer.size()) ? buffer.size() : result.size;
    std::print(stderr, "{}", std::string_view(buffer.data(), write_size));
}

/**
 * JPH Assert Bridge
 */
auto culv_jph_assert(const char *const inExpression, const char *const inMessage,
                     const char *const inFile, const CPH::Unsigned32 inLine) -> bool {

    const auto atty         = static_cast<bool>(IS_ATTY);
    const auto *const color = atty ? "\033[1;41m" : "";
    const auto reset        = atty ? ANSI_RESET : "";

    // Asserts are rare and critical, using a stack buffer again to be safe
    std::array<char, 1024> buffer;
    const auto result =
        std::format_to_n(buffer.data(), buffer.size(),
                         "\n{}[JOLT ASSERTION FAILURE]{}\n"
                         "  Expression: {}\n"
                         "  Message:    {}\n"
                         "  Location:   {}:{}\n",
                         color, reset, (inExpression != nullptr) ? inExpression : "N/A",
                         (inMessage != nullptr) ? inMessage : "No message provided",
                         (inFile != nullptr) ? inFile : "Unknown", inLine);

    const CPH::SizeType write_size = std::min<CPH::SizeType>(result.size, buffer.size());

    // This replaces all three of your lines:
    std::print(stderr, "{}", std::string_view(buffer.data(), write_size));

    return true;
}

} // extern "C"