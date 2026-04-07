#include "culverin_handler.h"
#include <stdio.h>
#include <stdarg.h>

// Jolt Trace Handler: Captures internal Jolt logs
void culv_jph_trace(const char *inString) {
    // Jolt has already formatted the message, so we just print it.
    fprintf(stderr, "[JPH] %s\n", inString);
}

// Jolt Assert Handler: Captures assertion failures
bool culv_jph_assert(const char *inExpression, const char *inMessage, const char *inFile, uint32_t inLine) {
    fprintf(stderr, "[JPH Assert] Failed: %s\nMessage: %s\nFile: %s:%u\n", 
            inExpression, inMessage ? inMessage : "None", inFile, inLine);
    fflush(stderr); // Ensure the log is sent before the breakpoint/crash
    return true; 
}