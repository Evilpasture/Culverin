// Configure the same precision as the C API before including Jolt headers.
#include "core/JoltConfig.h"

// Jolt/Jolt.h must precede all other Jolt headers.
#include <Jolt/Jolt.h>

// 2. Now you can include the specific headers
#include <Jolt/Renderer/DebugRendererSimple.h>
#include <Jolt/Core/JobSystemWithBarrier.h>
#include <typeinfo>

// Forces RTTI typeinfo symbols for Jolt classes used across the boundary.
void* _jolt_rtti_anchor[] = {
    (void*)&typeid(JPH::DebugRendererSimple),
    (void*)&typeid(JPH::JobSystemWithBarrier),
};