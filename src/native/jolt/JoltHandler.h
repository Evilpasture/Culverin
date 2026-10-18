#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void CPH_JphTrace(const char *inString);
bool CPH_JphAssert(const char *inExpression, const char *inMessage, const char *inFile,
                     uint32_t inLine);
#ifdef __cplusplus
}
#endif