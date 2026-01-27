cmake_minimum_required(VERSION 3.16)

# ==============================================================================
# 0. THE NUCLEAR INITIALIZATION
# ==============================================================================
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# --- THE ABI FIX: Force Double Precision into the Cache ---
# Jolt's internal CMakeLists uses the variable 'DOUBLE_PRECISION'.
# We must set it as a CACHE FORCE variable so Jolt cannot overwrite it.
set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)
set(USE_STATIC_MSVC_RUNTIME OFF CACHE BOOL "" FORCE)
set(FORCE_MT_RUNTIME OFF CACHE BOOL "" FORCE)

# Define the C++ macro globally for every target in the project
add_definitions(-DJPH_DOUBLE_PRECISION)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

project(culverin LANGUAGES C CXX)

# Force /MD on all flag strings as a backup
foreach(flag_var
    CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
    CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE)
    if(${flag_var})
        string(REPLACE "/MT" "/MD" ${flag_var} "${${flag_var}}")
    endif()
endforeach()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# ==============================================================================
# 2. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)
add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

# Double-check Jolt target properties
if(TARGET Jolt)
    set_property(TARGET Jolt PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    # Ensure Jolt target itself has the DP define (some versions need this explicitly)
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

# Force definitions for the wrapper
target_compile_definitions(${JOLTC_TARGET} PUBLIC 
    JPH_OBJECT_LAYER_BITS=32 
    JPH_DOUBLE_PRECISION
)

set_property(TARGET ${JOLTC_TARGET} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${JOLTC_TARGET} PRIVATE -mavx2 -mfma)
else()
    target_compile_options(${JOLTC_TARGET} PRIVATE /arch:AVX2)
endif()

# ==============================================================================
# 4. BUILD CULVERIN EXTENSION
# ==============================================================================
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

python_add_library(_culverin_c MODULE 
    src/culverin/culverin.c 
    src/culverin/shadow_sync.c
)

target_include_directories(_culverin_c PRIVATE "${JOLTC_INCLUDE_DIR}" "${JOLT_DIR}" ${Python_INCLUDE_DIRS})

# Ensure the C Extension knows about Double Precision (affects RVec3 size)
target_compile_definitions(_culverin_c PRIVATE 
    MS_WIN64 
    _CRT_SECURE_NO_WARNINGS 
    JPH_OBJECT_LAYER_BITS=32 
    JPH_DOUBLE_PRECISION
)

if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
endif()

target_link_libraries(_culverin_c PRIVATE ${JOLTC_TARGET} Winmm.lib)
set_property(TARGET _culverin_c PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(_culverin_c PRIVATE -mavx2 -mfma -Xclang -std=c11)
else()
    target_compile_options(_culverin_c PRIVATE /arch:AVX2 /std:c11)
endif()

set_target_properties(_culverin_c PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX "")

# ==============================================================================
# 5. INSTALLATION
# ==============================================================================
# We use "." because pyproject.toml already sets the install-dir to "culverin"
install(TARGETS _culverin_c DESTINATION .)

# Install the Python helper files into the package root
install(FILES 
    src/culverin/__init__.py 
    src/culverin/_culverin.py 
    DESTINATION .
)