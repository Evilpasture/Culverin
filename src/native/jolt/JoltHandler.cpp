#include "jolt/JoltHandler.h"
#include "cpp/Cpp.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
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

template <std::size_t Capacity>
class StackBuffer {
public:
    void append(std::string_view text) noexcept {
        const auto remaining = Capacity - size_;
        const auto count = text.size() < remaining ? text.size() : remaining;
        if (count == 0) {
            return;
        }

        std::memcpy(data_.data() + size_, text.data(), count);
        size_ += count;
    }

    void append_unsigned(CPH::Unsigned32 value) noexcept {
        std::array<char, 10> digits{};
        std::size_t count = 0;
        do {
            digits[count++] = static_cast<char>('0' + (value % 10U));
            value /= 10U;
        } while (value != 0);

        while (count != 0) {
            append(std::string_view(&digits[--count], 1));
        }
    }

    [[nodiscard]] auto data() const noexcept -> const char * { return data_.data(); }
    [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }

private:
    std::array<char, Capacity> data_{};
    std::size_t size_ = 0;
};

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
void CPH_JphTrace(const char *const inString) {
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

    // Format UTC as HH:MM:SS without std::format, whose libc++ implementation
    // instantiates floating-point to_chars unavailable at the macOS 13.3 target.
    const auto now_seconds = std::chrono::floor<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch());
    const auto day_start = std::chrono::floor<std::chrono::days>(now_seconds);
    const std::chrono::hh_mm_ss time_of_day(now_seconds - day_start);
    const auto hours   = static_cast<unsigned>(time_of_day.hours().count());
    const auto minutes = static_cast<unsigned>(time_of_day.minutes().count());
    const auto seconds = static_cast<unsigned>(time_of_day.seconds().count());
    const std::array<char, 8> timestamp = {
        static_cast<char>('0' + (hours / 10)), static_cast<char>('0' + (hours % 10)), ':',
        static_cast<char>('0' + (minutes / 10)), static_cast<char>('0' + (minutes % 10)), ':',
        static_cast<char>('0' + (seconds / 10)), static_cast<char>('0' + (seconds % 10))};

    // 3. Assemble in a bounded stack buffer, then write the exact bytes to stderr.
    StackBuffer<2048> buffer;
    buffer.append("[");
    buffer.append(std::string_view(timestamp.data(), timestamp.size()));
    buffer.append("] ");
    buffer.append(color);
    buffer.append(tag);
    buffer.append(atty ? ANSI_RESET : std::string_view{});
    buffer.append(atty ? std::string_view{} : std::string_view{":"});
    buffer.append(" ");
    buffer.append(msg);
    buffer.append("\n");
    (void)std::fwrite(buffer.data(), sizeof(char), buffer.size(), stderr);
}

/**
 * JPH Assert Bridge
 */
auto CPH_JphAssert(const char *const inExpression, const char *const inMessage,
                     const char *const inFile, const CPH::Unsigned32 inLine) -> bool {

    const auto atty         = static_cast<bool>(IS_ATTY);
    const auto *const color = atty ? "\033[1;41m" : "";
    const auto reset        = atty ? ANSI_RESET : "";

    // Asserts are rare and critical; keep their output bounded and allocation-free too.
    StackBuffer<1024> buffer;
    buffer.append("\n");
    buffer.append(color);
    buffer.append("[JOLT ASSERTION FAILURE]");
    buffer.append(reset);
    buffer.append("\n  Expression: ");
    buffer.append((inExpression != nullptr) ? inExpression : "N/A");
    buffer.append("\n  Message:    ");
    buffer.append((inMessage != nullptr) ? inMessage : "No message provided");
    buffer.append("\n  Location:   ");
    buffer.append((inFile != nullptr) ? inFile : "Unknown");
    buffer.append(":");
    buffer.append_unsigned(inLine);
    buffer.append("\n");
    (void)std::fwrite(buffer.data(), sizeof(char), buffer.size(), stderr);

    return true;
}

} // extern "C"