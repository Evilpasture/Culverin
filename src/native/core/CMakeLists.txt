# Shared compiler, allocation, threading, and ABI contracts used by all native
# domains. This is header-only and intentionally has no stand-alone binary.
add_library(culverin_native_common INTERFACE)
target_sources(
    culverin_native_common
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Assert.h>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/CompilerSpecifics.h>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/NativeAlloc.h>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Threading.h>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Types.h>"
)
target_link_libraries(culverin_native_common INTERFACE culverin_native_support)
