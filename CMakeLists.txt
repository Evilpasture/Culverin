cmake_minimum_required(VERSION 3.16)

# ==============================================================================
# 0. INITIALIZATION & POLICIES
# ==============================================================================
cmake_policy(SET CMP0091 NEW)

# Global Settings
set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)
set(USE_STATIC_MSVC_RUNTIME OFF CACHE BOOL "" FORCE)
set(FORCE_MT_RUNTIME OFF CACHE BOOL "" FORCE)

project(culverin LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
# Essential for Linux/macOS shared modules linking to static libs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Define the C++ macro globally
add_definitions(-DJPH_DOUBLE_PRECISION)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

# MSVC Runtime Handling
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    foreach(flag_var
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
        if(${flag_var})
            string(REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
        endif()
    endforeach()
endif()

# ==============================================================================
# 1. HELPER: SIMD FLAGS
# ==============================================================================
# We only apply AVX2 to x86_64. We avoid it on ARM64 (Apple Silicon).
macro(apply_simd_flags TARGET_NAME)
    if(MSVC)
        target_compile_options(${TARGET_NAME} PRIVATE /arch:AVX2 /fp:precise)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        target_compile_options(${TARGET_NAME} PRIVATE -mavx2 -mfma)
    endif()
endmacro()

# ==============================================================================
# 2. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)
add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

if(TARGET Jolt)
    target_compile_definitions(Jolt PUBLIC JPH_DOUBLE_PRECISION JPH_OBJECT_LAYER_BITS=32)
endif()

# ==============================================================================
# 3. BUILD JOLTC
# ==============================================================================
set(JOLTC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/include")
set(JOLTC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/src")
set(JOLTC_TARGET joltc_double)

add_library(${JOLTC_TARGET} STATIC 
    "${JOLTC_SOURCE_DIR}/joltc.cpp"
    "${JOLTC_SOURCE_DIR}/joltc_assert.cpp"
    "${JOLTC_SOURCE_DIR}/joltc.c"
)
target_include_directories(${JOLTC_TARGET} PUBLIC "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}")
target_link_libraries(${JOLTC_TARGET} PRIVATE Jolt)
target_compile_definitions(${JOLTC_TARGET} PUBLIC JPH_OBJECT_LAYER_BITS=32 JPH_DOUBLE_PRECISION)

apply_simd_flags(${JOLTC_TARGET})

# ==============================================================================
# 4. BUILD CULVERIN EXTENSION
# ==============================================================================
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

python_add_library(_culverin_c MODULE 
    src/culverin/culverin.c 
    src/culverin/shadow_sync.c
)

target_include_directories(_culverin_c PRIVATE "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}" ${Python_INCLUDE_DIRS})
target_compile_definitions(_culverin_c PRIVATE 
    _CRT_SECURE_NO_WARNINGS 
    JPH_OBJECT_LAYER_BITS=32 
    JPH_DOUBLE_PRECISION
)

if(WIN32)
    target_compile_definitions(_culverin_c PRIVATE MS_WIN64)
    target_link_libraries(_culverin_c PRIVATE Winmm.lib)
endif()

if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
endif()

target_link_libraries(_culverin_c PRIVATE ${JOLTC_TARGET})
apply_simd_flags(_culverin_c)

set_target_properties(_culverin_c PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX "")

# ==============================================================================
# 5. INSTALLATION
# ==============================================================================
install(TARGETS _culverin_c DESTINATION .)
install(FILES src/culverin/__init__.py src/culverin/_culverin.py DESTINATION .)