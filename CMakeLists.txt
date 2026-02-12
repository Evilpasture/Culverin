cmake_minimum_required(VERSION 3.16)

# 1. THE NUCLEAR INITIALIZATION
# CMP0091 allows us to use CMAKE_MSVC_RUNTIME_LIBRARY
cmake_policy(SET CMP0091 NEW)

# Force Jolt settings into the CACHE so they can't be overridden
set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)
set(GENERATE_SAMPLES OFF CACHE BOOL "" FORCE)
set(GENERATE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME OFF CACHE BOOL "" FORCE)
set(FORCE_MT_RUNTIME OFF CACHE BOOL "" FORCE)
set(USE_VULKAN OFF CACHE BOOL "" FORCE)

project(culverin LANGUAGES C CXX)

# Standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DJPH_DOUBLE_PRECISION)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

# ==============================================================================
# 2. COMPILER SPECIFIC FLAGS
# ==============================================================================
if(MSVC OR (WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang"))
    # --- Windows logic (Clang-CL / MSVC) ---
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    add_compile_options(/arch:AVX2) 
    
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        # Aggressive Clang-CL specific flags
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /clang:-std=c11")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
        # ADDED: -ffast-math and -march=native for Clang-CL
        set(RELEASE_FLAGS "/clang:-O3 /clang:-ffast-math /clang:-march=native /Ob2")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${RELEASE_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
    else()
        # Pure MSVC - use /fp:fast and /Ot (Favor fast code)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:c11 /experimental:c11atomics /fp:fast")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Ox /Ot /Ob2 /Oi")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Ox /Ot /Ob2 /Oi")
    endif()

else()
    # --- LINUX & MACOS ---
    # Global Unix Release Speedups
    set(UNIX_RELEASE_FLAGS "-O3 -ffast-math -fomit-frame-pointer")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" OR CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        # Specific target for x86_64: use FMA and AVX2
        set(UNIX_RELEASE_FLAGS "${UNIX_RELEASE_FLAGS} -mavx2 -mfma -march=native")
        message(STATUS "Culverin: Optimized for x86_64 with AVX2/FMA/Native")
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        # ARM64: Native targeting for M1/M2/M3 or Graviton
        set(UNIX_RELEASE_FLAGS "${UNIX_RELEASE_FLAGS} -march=native")
        message(STATUS "Culverin: Optimized for ARM64 Native")
    endif()

    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${UNIX_RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${UNIX_RELEASE_FLAGS}")
endif()


# ==============================================================================
# 3. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)
add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

# THE CRITICAL STEP: Align all Jolt targets to /MD
if(MSVC OR (WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang"))
    foreach(t Jolt) # Add other Jolt targets here if necessary (e.g. JoltRenderer)
        if(TARGET ${t})
            # 1. Force the Runtime Library to /MD
            set_property(TARGET ${t} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
            
            # 2. Strip broken generator expressions from Jolt (The VS+ClangCL bug)
            if(CMAKE_C_COMPILER_ID MATCHES "Clang")
                set_property(TARGET ${t} PROPERTY COMPILE_OPTIONS "")
                target_compile_options(${t} PRIVATE /arch:AVX2 /std:c++17)
            endif()
        endif()
    endforeach()
endif()

# ==============================================================================
# 4. BUILD JOLTC & EXTENSION
# ==============================================================================
set(JOLTC_TARGET joltc_double)
add_library(${JOLTC_TARGET} STATIC 
    "extern/JoltC/src/joltc.cpp"
    "extern/JoltC/src/joltc_assert.cpp"
    "extern/JoltC/src/joltc.c"
)
target_include_directories(${JOLTC_TARGET} PUBLIC "extern/JoltC/include" "${JOLT_DIR}")
target_link_libraries(${JOLTC_TARGET} PRIVATE Jolt)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
python_add_library(_culverin_c MODULE 
    src/culverin/culverin.c 
    src/culverin/culverin_shadow_sync.c
    src/culverin/culverin_debug_render.c
    src/culverin/culverin_getters.c
    src/culverin/culverin_character.c
    src/culverin/culverin_contact_listener.c
    src/culverin/culverin_parsers.c
    src/culverin/culverin_vehicle.c
    src/culverin/culverin_command_buffer.c
    src/culverin/culverin_tracked_vehicle.c
    src/culverin/culverin_ragdoll.c
    src/culverin/culverin_internal_query.c
    src/culverin/culverin_physics_world_internal.c
    src/culverin/culverin_query_methods.c
    src/culverin/culverin_constraint.c
    src/culverin/culverin_constraint_factory.c
)
add_custom_command(TARGET _culverin_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Extension built at: $<TARGET_FILE:_culverin_c>"
)
if(MSVC)
    # 1. Add /Zi to generate debug info (symbols) even in Release builds
    # 2. Add /DEBUG to the linker to create the .pdb file
    # 3. Add /OPT:REF /OPT:ICF to ensure the Release build remains optimized and small
    target_compile_options(_culverin_c PRIVATE /Zi /std:c11 /experimental:c11atomics /fp:fast)
    target_link_options(_culverin_c PRIVATE /DEBUG /OPT:REF /OPT:ICF)
    
    # Fix for some MSVC headers being stubborn about standards
    target_compile_definitions(_culverin_c PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)
endif()
target_include_directories(_culverin_c PRIVATE 
    "src/culverin"
    "extern/JoltC/include" 
    "${JOLT_DIR}" 
    ${Python_INCLUDE_DIRS}
)

if(WIN32)
    target_link_libraries(_culverin_c PRIVATE Winmm.lib)
endif()

if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
endif()

target_link_libraries(_culverin_c PRIVATE ${JOLTC_TARGET})
set_target_properties(_culverin_c PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX "")

# Final sanity check: ensure our own targets are also /MD
set_property(TARGET ${JOLTC_TARGET} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set_property(TARGET _culverin_c PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

install(TARGETS _culverin_c DESTINATION .)
if(MSVC)
    # We use a Generator Expression to find the exact PDB location
    install(FILES $<TARGET_PDB_FILE:_culverin_c> DESTINATION . OPTIONAL)
endif()
install(FILES src/culverin/__init__.py src/culverin/_culverin.py DESTINATION .)