#include "culverin_handler.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#    include <windows.h>
#    define IS_ATTY _isatty(_fileno(stderr))
#else
#    include <unistd.h>
#    define IS_ATTY isatty(fileno(stderr))
#endif

// Helper to get a timestamp string
static void get_timestamp(char *buf, size_t len) {
    time_t now   = time(NULL);
    struct tm *t = localtime(&now);
    strftime(buf, len, "%H:%M:%S", t);
}

void culv_jph_trace(const char *inString) {
    if (!inString || !*inString)
        return;

    char ts[16];
    get_timestamp(ts, sizeof(ts));

    // Determine log level and color based on Jolt's prefix conventions
    const char *color_code = "";
    const char *reset_code = "";
    const char *level_tag  = "INFO";

    // Jolt typically prefixes strings with "Warning:" or "Error:"
    if (strstr(inString, "Error")) {
        level_tag = "ERROR";
        if (IS_ATTY) {
            color_code = "\033[1;31m";
            reset_code = "\033[0m";
        }
    } else if (strstr(inString, "Warning")) {
        level_tag = "WARN ";
        if (IS_ATTY) {
            color_code = "\033[1;33m";
            reset_code = "\033[0m";
        }
    } else {
        if (IS_ATTY) {
            color_code = "\033[0;36m";
            reset_code = "\033[0m";
        }
    }

    // Format: [TIME] [LEVEL] Message
    // We use fprintf(stderr) to ensure it shows up in most Python IDE consoles
    fprintf(stderr, "[%s] %s[%s]%s %s\n", ts, color_code, level_tag, reset_code, inString);

    // Optional: If the string already has a newline at the end (Jolt sometimes adds them),
    // you might want to trim it, but Jolt's Trace strings are usually clean.
}

// Jolt Assert Handler: Captures assertion failures
bool culv_jph_assert(const char *inExpression, const char *inMessage, const char *inFile,
                     uint32_t inLine) {
    fprintf(stderr, "[JPH Assert] Failed: %s\nMessage: %s\nFile: %s:%u\n", inExpression,
            inMessage ? inMessage : "None", inFile, inLine);
    fflush(stderr); // Ensure the log is sent before the breakpoint/crash
    return true;
}