// 1. MUST come first - defines JPH_NAMESPACE_BEGIN, uint32, etc.
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