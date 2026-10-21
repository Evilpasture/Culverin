# Shared compiler, storage, threading, and ABI contracts used by all native
# domains. This is header-only and intentionally has no stand-alone binary.
add_library(culverin_native_common INTERFACE)
target_sources(
    culverin_native_common
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Buffer.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Compiler.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Error.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Handle.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Profiling.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Threading.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Types.hpp>"
)
target_link_libraries(culverin_native_common INTERFACE culverin_native_support)
