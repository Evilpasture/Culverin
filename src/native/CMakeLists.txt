# Native dependencies and include paths are shared by every native domain.
find_package(Threads REQUIRED)

add_library(culverin_native_support INTERFACE)
target_include_directories(
    culverin_native_support
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_include_directories(
    culverin_native_support
    SYSTEM INTERFACE
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extern/JoltC/include>"
        "$<BUILD_INTERFACE:${JOLT_DIR}>"
)
target_link_libraries(
    culverin_native_support
    INTERFACE ${JOLTC_TARGET} Jolt::Jolt
)

# Header-only utility and shared-contract targets.
add_subdirectory(cpp)
add_subdirectory(core)
add_subdirectory(soft_bodies)

# Compiled native domains.
add_subdirectory(jolt)
add_subdirectory(math)
add_subdirectory(debug)
add_subdirectory(physics)

# Preserve one reusable Python-independent archive while keeping compilation
# partitioned by domain. The Jolt RTTI anchor is extension-link glue, not core.
add_library(
    culverin_core
    STATIC
        threading/Mutex.c
        threading/Mutex.h
        threading/Thread.c
        threading/Thread.h
        threading/ThreadSwitch.S
        threading/ThreadAssembly.h
)
set_source_files_properties(threading/ThreadSwitch.S PROPERTIES LANGUAGE ASM)
target_sources(
    culverin_core
    PRIVATE
        $<TARGET_OBJECTS:culverin_native_jolt>
        $<TARGET_OBJECTS:culverin_native_math>
        $<TARGET_OBJECTS:culverin_native_debug>
        $<TARGET_OBJECTS:culverin_native_physics>
)
target_link_libraries(
    culverin_core
    PUBLIC
        culverin_native_common
        culverin_native_cpp
        culverin_native_soft_bodies
        Threads::Threads
)
culverin_configure_compile_target(culverin_core)
