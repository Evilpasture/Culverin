cmake_minimum_required(VERSION 3.16)
project(culverin LANGUAGES C CXX)

# ==============================================================================
# 1. GLOBAL JOLT CONFIGURATION (MUST BE BEFORE ADD_SUBDIRECTORY)
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- THE FIX: Force 32-bit Object Layers Globally ---
# This ensures Jolt Physics signatures match the JoltC wrapper signatures.
set(OBJECT_LAYER_BITS 32 CACHE STRING "Number of bits for Object Layer" FORCE)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

# --- Precision & Hardware Flags ---
option(DOUBLE_PRECISION "Use double precision math" ON)
set(JPH_USE_AVX2 ON CACHE BOOL "" FORCE)
set(JPH_USE_SSE4_2 ON CACHE BOOL "" FORCE)

# --- Build Type Fixes ---
set(JPH_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
set(JPH_OBJECT_LIBRARY OFF CACHE BOOL "" FORCE) 
set(FORCE_MT_RUNTIME OFF CACHE BOOL "" FORCE)

# Stop Jolt's install pollution in the wheel
set(JPH_INSTALL OFF CACHE BOOL "" FORCE)
set(JPH_SAMPLES OFF CACHE BOOL "" FORCE)
set(JPH_UNIT_TESTS OFF CACHE BOOL "" FORCE)

macro(SET_INTERPROCEDURAL_OPTIMIZATION)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endmacro()

# ==============================================================================
# 2. BUILD JOLT PHYSICS (The Engine)
# ==============================================================================
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)

if(EXISTS "${JOLT_DIR}/Build/CMakeLists.txt")
    add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)
else()
    message(FATAL_ERROR "JoltPhysics not found in extern/JoltPhysics.")
endif()

# ==============================================================================
# 3. BUILD JOLTC (The Manual Wrapper Target)
# ==============================================================================
set(JOLTC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/include")
set(JOLTC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/src")

set(JOLTC_SOURCES
    "${JOLTC_SOURCE_DIR}/joltc.cpp"
    "${JOLTC_SOURCE_DIR}/joltc_assert.cpp"
    "${JOLTC_SOURCE_DIR}/joltc.c"
)

if (DOUBLE_PRECISION)
    set(JOLTC_TARGET joltc_double)
else()
    set(JOLTC_TARGET joltc)
endif ()

add_library(${JOLTC_TARGET} STATIC ${JOLTC_SOURCES})

# Force rebuild JoltC if headers change
set_source_files_properties(${JOLTC_SOURCES} PROPERTIES OBJECT_DEPENDS "${JOLTC_INCLUDE_DIR}/joltc.h")

target_include_directories(${JOLTC_TARGET} PUBLIC "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}")
target_link_libraries(${JOLTC_TARGET} PRIVATE Jolt)

# Force definitions to match
target_compile_definitions(${JOLTC_TARGET} PUBLIC JPH_OBJECT_LAYER_BITS=32)
if (DOUBLE_PRECISION)
    target_compile_definitions(${JOLTC_TARGET} PUBLIC JPH_DOUBLE_PRECISION)
endif()

# Propagate Hardware Flags
if (MSVC)
    target_compile_options(${JOLTC_TARGET} PRIVATE /arch:AVX2)
else()
    target_compile_options(${JOLTC_TARGET} PRIVATE -mavx2 -mfma)
endif()

# ==============================================================================
# 4. BUILD CULVERIN EXTENSION
# ==============================================================================
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

python_add_library(_culverin_c MODULE 
    src/culverin/culverin.c 
    src/culverin/shadow_sync.c
)

# Header dependency
set_source_files_properties(src/culverin/culverin.c src/culverin/shadow_sync.c 
    PROPERTIES OBJECT_DEPENDS "${JOLTC_INCLUDE_DIR}/joltc.h"
)

target_include_directories(_culverin_c PRIVATE "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}" ${Python_INCLUDE_DIRS})

target_compile_definitions(_culverin_c PRIVATE 
    MS_WIN64 
    _CRT_SECURE_NO_WARNINGS
    JPH_OBJECT_LAYER_BITS=32
)

if(DOUBLE_PRECISION)
    target_compile_definitions(_culverin_c PRIVATE JPH_DOUBLE_PRECISION)
endif()

if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
endif()

target_link_libraries(_culverin_c PRIVATE ${JOLTC_TARGET} Winmm.lib)

# Match hardware flags
if (MSVC)
    target_compile_options(_culverin_c PRIVATE /arch:AVX2)
else()
    target_compile_options(_culverin_c PRIVATE -mavx2 -mfma)
endif()

set_target_properties(_culverin_c PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX "")

# ==============================================================================
# 5. MSVC COMPATIBILITY FIXES (THE "HAMMER")
# ==============================================================================
if (MSVC)
    # 1. Force /MD (Dynamic Runtime) on ALL targets to match Python
    foreach(t ${JOLTC_TARGET} Jolt _culverin_c)
        if (TARGET ${t})
            set_property(TARGET ${t} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
        endif()
    endforeach()

    # 2. Disable /GL (Whole Program Optimization) to prevent LTCG mismatches during build
    string(REPLACE "/GL" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    string(REPLACE "/GL" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")
    
    # 3. Fix for C++ Exception handling required by Jolt
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()

# ==============================================================================
# 6. INSTALLATION
# ==============================================================================
install(TARGETS _culverin_c DESTINATION .)