# Native dependencies and include paths are shared by every native domain.
find_package(Threads REQUIRED)

add_library(culverin_native_support INTERFACE)
target_include_directories(
    culverin_native_support
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
# Jolt::Jolt is the single source of Jolt's include path and compile
# definitions; the JoltC archive that used to sit beside it is gone.
target_link_libraries(
    culverin_native_support
    INTERFACE Jolt::Jolt
)

# Utility, shared-contract and global-state targets.
add_subdirectory(cpp)
add_subdirectory(core)
add_subdirectory(soft_bodies)

# Compiled native domains.
add_subdirectory(jolt)
add_subdirectory(math)
add_subdirectory(debug)
add_subdirectory(physics)
add_subdirectory(ecs)
add_subdirectory(vehicles)
add_subdirectory(characters)
add_subdirectory(ragdolls)

# Preserve one reusable Python-independent archive while keeping compilation
# partitioned by domain. The Jolt RTTI anchor is extension-link glue, not core.
add_library(culverin_core STATIC)
target_sources(
    culverin_core
    PRIVATE
        $<TARGET_OBJECTS:culverin_native_core>
        $<TARGET_OBJECTS:culverin_native_jolt>
        $<TARGET_OBJECTS:culverin_native_math>
        $<TARGET_OBJECTS:culverin_native_debug>
        $<TARGET_OBJECTS:culverin_native_physics>
        $<TARGET_OBJECTS:culverin_native_ecs>
        $<TARGET_OBJECTS:culverin_native_vehicles>
        $<TARGET_OBJECTS:culverin_native_characters>
        $<TARGET_OBJECTS:culverin_native_ragdolls>
        $<TARGET_OBJECTS:culverin_native_soft_bodies>
)
target_link_libraries(
    culverin_core
    PUBLIC
        culverin_native_common
        culverin_native_cpp
        Threads::Threads
)
culverin_configure_compile_target(culverin_core)
