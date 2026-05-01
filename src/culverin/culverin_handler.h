#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void culv_jph_trace(const char *inString);
bool culv_jph_assert(const char *inExpression, const char *inMessage, const char *inFile,
                     uint32_t inLine);
#ifdef __cplusplus
}
#endif