cmake_minimum_required(VERSION 3.16)

# ==============================================================================
# 0. GLOBAL COMPILER & RUNTIME SETTINGS
# ==============================================================================
cmake_policy(SET CMP0091 NEW) 

# FIX 1: Ensure all static libraries are built with -fPIC for Linux shared modules
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force /MD (Dynamic Runtime) globally for Python compatibility
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Force Jolt-specific variables into the cache
set(USE_STATIC_MSVC_RUNTIME OFF CACHE BOOL "Use /MD instead of /MT" FORCE)
set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)

# FIX 2: Disable Jolt features that break old macOS compatibility or slow down builds
set(INTEROP_COMPUTE OFF CACHE BOOL "" FORCE)
set(USE_MTL OFF CACHE BOOL "Disable Metal to fix macOS deployment target issues" FORCE)
set(USE_GL_COMPUTE OFF CACHE BOOL "" FORCE)

add_definitions(-DJPH_DOUBLE_PRECISION)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

project(culverin LANGUAGES C CXX)

# ==============================================================================
# 1. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)

set(GENERATE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(GENERATE_SAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

# Ensure Jolt target inherits the PIC setting and custom defines
if(TARGET Jolt)
    set_property(TARGET Jolt PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET Jolt PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    target_compile_definitions(Jolt PUBLIC JPH_DOUBLE_PRECISION JPH_OBJECT_LAYER_BITS=32)
endif()


# ==============================================================================
# 2. BUILD JOLTC (The C++ Wrapper)
# ==============================================================================
set(JOLTC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/include")
set(JOLTC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/src")

add_library(joltc_double STATIC 
    "${JOLTC_SOURCE_DIR}/joltc.cpp"
    "${JOLTC_SOURCE_DIR}/joltc_assert.cpp"
    "${JOLTC_SOURCE_DIR}/joltc.c"
)

target_compile_features(joltc_double PUBLIC cxx_std_17)
target_include_directories(joltc_double PUBLIC "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}")
target_link_libraries(joltc_double PRIVATE Jolt)

set_property(TARGET joltc_double PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
target_compile_definitions(joltc_double PUBLIC JPH_DOUBLE_PRECISION JPH_OBJECT_LAYER_BITS=32)

# ==============================================================================
# 3. CREATE THE EXTENSION TARGET
# ==============================================================================
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

python_add_library(_culverin_c MODULE 
    src/culverin/culverin.c 
    src/culverin/shadow_sync.c
)

target_include_directories(_culverin_c PRIVATE "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}" ${Python_INCLUDE_DIRS})
target_link_libraries(_culverin_c PRIVATE joltc_double)

target_compile_definitions(_culverin_c PRIVATE 
    JPH_OBJECT_LAYER_BITS=32 
    JPH_DOUBLE_PRECISION
)

if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
endif()

if(MSVC)
    target_link_libraries(_culverin_c PRIVATE Winmm.lib)
    set_property(TARGET _culverin_c PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(_culverin_c PRIVATE -mavx2 -mfma -std=c11)
    else()
        # Ensure stdatomic.h works on standard MSVC
        target_compile_options(_culverin_c PRIVATE /arch:AVX2 /std:c11 /experimental:c11atomics)
    endif()
else()
    target_compile_options(_culverin_c PRIVATE -mavx2 -mfma -std=c11)
endif()

set_target_properties(_culverin_c PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX "")

# ==============================================================================
# 4. INSTALLATION
# ==============================================================================
install(TARGETS _culverin_c DESTINATION .)
install(FILES src/culverin/__init__.py src/culverin/_culverin.py DESTINATION .)