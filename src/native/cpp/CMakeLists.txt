add_library(culverin_native_cpp INTERFACE)
target_include_directories(
    culverin_native_cpp
    INTERFACE "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/native>"
)
target_sources(
    culverin_native_cpp
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Loop.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Prefetch.hpp>"
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Span.hpp>"
)
