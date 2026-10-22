# Shared compiler, storage, threading, and ABI contracts used by all native
# domains. The headers carry the contracts; the two translation units carry the
# only global state this layer owns. Neither has a stand-alone binary.
add_library(culverin_native_common INTERFACE)
target_sources(
    culverin_native_common
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Buffer.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Compiler.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Error.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/FloatBits.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Handle.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Profiling.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Threading.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Types.hpp>"
)
target_link_libraries(culverin_native_common INTERFACE culverin_native_support)

# The object half of this layer. Two namespace-scope objects (the Jolt init
# mutex and the error-category list head) are declared in the headers above and
# defined once, here. `inline` variables would also work, but a global that
# exists in every translation unit is exactly what CPH::Global is for: one
# definition, one address, greppable.
add_library(culverin_native_core OBJECT Error.cpp Threading.cpp)
target_link_libraries(culverin_native_core PUBLIC culverin_native_common)
culverin_configure_compile_target(culverin_native_core)
