#pragma once

#include <cstdint>

// Jolt's trace/assert handler ABI is C (joltc.h); these two functions are
// registered with JPH_SetTraceHandler / JPH_SetAssertFailureHandler.
extern "C" {
void CPH_JphTrace(const char *inString);
bool CPH_JphAssert(const char *inExpression, const char *inMessage, const char *inFile,
                   std::uint32_t inLine);
}
