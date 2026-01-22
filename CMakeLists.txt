cmake_minimum_required(VERSION 3.16)
project(culverin LANGUAGES C CXX)

# --- 1. JOLT CONFIGURATION ---
# We define these globally so all targets see them
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DOUBLE_PRECISION "Use double precision" ON)
set(JPH_USE_AVX2 ON CACHE BOOL "" FORCE)
set(JPH_USE_SSE4_2 ON CACHE BOOL "" FORCE)
set(JPH_STATIC_LIBRARY ON CACHE BOOL "" FORCE)

# Ensure all targets use the same MSVC runtime (/MD)
if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/EHsc>)
endif()

# --- 2. BUILD JOLT PHYSICS ---
# We add Jolt first. Official CMake is in JoltPhysics/Build
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)
if(EXISTS "${JOLT_DIR}/Build/CMakeLists.txt")
    add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)
else()
    message(FATAL_ERROR "JoltPhysics not found in extern/JoltPhysics. Did you forget --recursive?")
endif()

# --- 3. BUILD JOLTC (Manual Target Definition) ---
# We bypass JoltC's internal CMakeLists.txt to avoid the download/macro bugs.
set(JOLTC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/include")
set(JOLTC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltC/src")

set(JOLTC_SOURCES
    "${JOLTC_SOURCE_DIR}/joltc.cpp"
    "${JOLTC_SOURCE_DIR}/joltc_assert.cpp"
    "${JOLTC_SOURCE_DIR}/joltc.c"
)

# Define the target name based on precision
if (DOUBLE_PRECISION)
    set(JOLTC_TARGET joltc_double)
else()
    set(JOLTC_TARGET joltc)
endif ()

add_library(${JOLTC_TARGET} STATIC ${JOLTC_SOURCES})

# Configure JoltC
target_include_directories(${JOLTC_TARGET} PUBLIC 
    "${JOLTC_INCLUDE_DIR}"
    "${JOLT_DIR}"
)
target_link_libraries(${JOLTC_TARGET} PRIVATE Jolt)

# Match precision define
if (DOUBLE_PRECISION)
    target_compile_definitions(${JOLTC_TARGET} PUBLIC JPH_DOUBLE_PRECISION)
endif()

# --- 4. BUILD CULVERIN EXTENSION ---
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
python_add_library(_culverin_c MODULE 
    src/culverin/culverin.c 
    src/culverin/shadow_sync.c
)

# Link everything together
target_link_libraries(_culverin_c PRIVATE ${JOLTC_TARGET} Winmm.lib)

target_include_directories(_culverin_c PRIVATE 
    "${JOLTC_INCLUDE_DIR}" 
    "${JOLT_DIR}" 
    ${Python_INCLUDE_DIRS}
)

target_compile_definitions(_culverin_c PRIVATE 
    MS_WIN64 
    _CRT_SECURE_NO_WARNINGS
)

if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
endif()

# Final Polish
set_target_properties(_culverin_c PROPERTIES 
    OUTPUT_NAME "_culverin_c" 
    PREFIX ""
)

# --- 5. INSTALLATION ---
install(TARGETS _culverin_c DESTINATION .)