cmake_minimum_required(VERSION 3.16)

# ==============================================================================
# 0. THE COMPILER ENFORCER
# ==============================================================================
function(enforce_modern_compiler)
    # CMAKE_C_COMPILER_ID is only available AFTER project() or enable_language()
    # So we check the environment or the forced variables
    
    # If the compiler is NOT Clang and NOT GNU, then we trigger the guard
    if(CMAKE_C_COMPILER_ID AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        
        message(STATUS "Detected Compiler ID: ${CMAKE_C_COMPILER_ID}")
        
        message(STATUS "Culverin: MSVC detected. Searching for alternatives...")

        # 2. Try to find Clang or GCC in the system PATH
        find_program(CLANG_PATH NAMES clang clang-cl)
        find_program(GCC_PATH NAMES gcc)

        set(HELP_MSG "\n################################################################\n")
        string(APPEND HELP_MSG " Guard triggered: MSVC (cl.exe) is incapable of compilation.\n")
        string(APPEND HELP_MSG "################################################################\n")
        string(APPEND HELP_MSG " C23 features and Jolt optimization require Clang or GCC.\n\n")

        if(CLANG_PATH OR GCC_PATH)
            string(APPEND HELP_MSG " Alternatives on your computer:\n")
            if(CLANG_PATH)
                string(APPEND HELP_MSG "  - Clang's emplacement: ${CLANG_PATH}\n")
            endif()
            if(GCC_PATH)
                string(APPEND HELP_MSG "  - GCC's emplacement:   ${GCC_PATH}\n")
            endif()
            string(APPEND HELP_MSG "\n Instruction: Decimate your 'build' folder and run:\n")
            string(APPEND HELP_MSG " cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..\n")
        else()
            string(APPEND HELP_MSG " NO SUITABLE COMPILER FOUND in system PATH.\n")
            string(APPEND HELP_MSG " Please download and install LLVM (Clang) from:\n")
            string(APPEND HELP_MSG " https://github.com/llvm/llvm-project/releases\n")
        endif()
        
        string(APPEND HELP_MSG "################################################################\n")
        message(FATAL_ERROR "${HELP_MSG}")
    endif()
endfunction()


# 1. THE NUCLEAR INITIALIZATION
# CMP0091 allows us to use CMAKE_MSVC_RUNTIME_LIBRARY
cmake_policy(SET CMP0091 NEW)

option(ENABLE_SANITIZER "Enable Clang Sanitizers (ASan/TSan/UBSan)" OFF)

# Force Jolt settings into the CACHE so they can't be overridden
set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)
set(GENERATE_SAMPLES OFF CACHE BOOL "" FORCE)
set(GENERATE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME OFF CACHE BOOL "" FORCE)
set(FORCE_MT_RUNTIME OFF CACHE BOOL "" FORCE)
set(USE_VULKAN OFF CACHE BOOL "" FORCE)

# Force RTTI ON for the entire build tree before Jolt is processed
add_definitions(-D_HAS_STATIC_RTTI=1)

# Ensure Jolt doesn't override it internally
set(FORCE_RTTI ON CACHE BOOL "Force RTTI on" FORCE)

project(culverin LANGUAGES C CXX)
enforce_modern_compiler()
message(STATUS "--- COMPILER VERIFICATION ---")
message(STATUS "C Compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Clang Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "-----------------------------")

if(NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-frtti>")
endif()

# Standards
if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS "13.0")
        message(FATAL_ERROR "Culverin: Your compiler is too old! Need GCC 13+ or Clang 17+ for C23.")
    endif()
    
    # GCC 13/14 often prefer c2x, GCC 15+ likes c23
    if(CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL "15.0")
        set(CMAKE_C_STANDARD 23)
    else()
        set(CMAKE_C_STANDARD 2x) 
    endif()
endif()
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DJPH_DOUBLE_PRECISION)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

# ==============================================================================
# 2. COMPILER SPECIFIC FLAGS
# ==============================================================================
# 1. Check if we are using the MSVC-like driver (clang-cl) or pure MSVC
if(MSVC)
    # --- Windows logic (Clang-CL / MSVC) ---
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    
    # Check if this is Clang-CL specifically
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(STATUS "Culverin: Detected Clang-CL (MSVC-like)")
        add_compile_options(/arch:AVX2) 
        
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /clang:-std=c23")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
        
        set(RELEASE_FLAGS " \
            /clang:-O3 \
            /clang:-ffast-math \
            /clang:-march=native \
            /clang:-mavx2 \
            /clang:-fstrict-aliasing \
            /clang:-mllvm /clang:-vectorize-slp-aggressive \
            /clang:-mllvm /clang:-inline-threshold=1000 \
            /Ob2 /Oi /Ot \
        ")
    else()
        message(STATUS "Culverin: Detected Pure MSVC")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /std:clatest /experimental:c11atomics /fp:fast")
        set(RELEASE_FLAGS "/Ox /Ot /Ob2 /Oi")
    endif()

    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")

# 2. Handle vanilla Clang (clang.exe) on Windows or Linux/macOS
else()
    # --- GNU-like Logic (Vanilla Clang or GCC) ---
    message(STATUS "Culverin: Using GNU-like flags (-O3, etc.)")

    set(SAN_FLAGS "")
    if(ENABLE_SANITIZER)
        # Choose ONE: -fsanitize=address OR -fsanitize=thread
        # You generally cannot run ASan and TSan at the same time.
        set(SAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer -fsanitize=undefined")
        message(STATUS "Culverin: SANITIZERS ENABLED: ${SAN_FLAGS}")
    endif()
    
    set(UNIX_RELEASE_FLAGS "-O3 -Wall -Wextra -Wconversion -Wdouble-promotion -Wpadded -ffast-math -fomit-frame-pointer")

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(UNIX_RELEASE_FLAGS "${UNIX_RELEASE_FLAGS} -mavx2 -mfma -march=native")
    endif()

    # Ensure C23/C++17 for vanilla clang
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c2x") # or c23 depending on clang version
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -frtti")

    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${UNIX_RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${UNIX_RELEASE_FLAGS}")
endif()


# ==============================================================================
# 3. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(JOLT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics" ABSOLUTE)
# 1. Add this globally before add_subdirectory(Jolt)
add_definitions(-DJPH_DEBUG_RENDERER)

# 2. CRITICAL: Jolt's CMake has a toggle for this. Force it:
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE ON CACHE BOOL "" FORCE)
add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

# THE CRITICAL STEP: Align all Jolt targets to /MD
if(MSVC OR (WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang"))
    foreach(t Jolt)
        if(TARGET ${t})
            set_property(TARGET ${t} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
            
            if(CMAKE_C_COMPILER_ID MATCHES "Clang")
                set_property(TARGET ${t} PROPERTY COMPILE_OPTIONS "")
                
                if(MSVC)
                    # For clang-cl.exe
                    target_compile_options(${t} PRIVATE /arch:AVX2 /std:c++17)
                else()
                    # For vanilla clang.exe (What you are currently using)
                    # We add -mfma and -mf16c to satisfy the intrinsic requirements
                    target_compile_options(${t} PRIVATE -mavx2 -mfma -mf16c -std=c++17)
                endif()
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
if(MSVC)
    # If using clang-cl or MSVC, tell it to be quiet
    target_compile_options(${JOLTC_TARGET} PRIVATE /W0)
else()
    # If using vanilla clang/gcc, use the "shut up" flag
    target_compile_options(${JOLTC_TARGET} PRIVATE -w)
endif()
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
    src/culverin/culverin_fast_parse.c
)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)

if(ipo_supported)
    message(STATUS "Culverin: LTO/IPO is supported and ENABLED")
    set_property(TARGET _culverin_c PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET ${JOLTC_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(TARGET Jolt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "Culverin: LTO/IPO not supported: ${error}")
endif()
add_custom_command(TARGET _culverin_c POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "Extension built at: $<TARGET_FILE:_culverin_c>"
)
if(MSVC)
    # 1. Add /Zi to generate debug info (symbols) even in Release builds
    # 2. Add /DEBUG to the linker to create the .pdb file
    # 3. Add /OPT:REF /OPT:ICF to ensure the Release build remains optimized and small
    target_compile_options(_culverin_c PRIVATE /Zi /std:clatest /experimental:c11atomics /fp:fast)
    target_link_options(_culverin_c PRIVATE /DEBUG /OPT:REF /OPT:ICF)
    
    # Fix for some MSVC headers being stubborn about standards
    target_compile_definitions(_culverin_c PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)
else() # Vanilla Clang / GNU-like
    message(STATUS "Culverin: Enforcing Strict Physics-Migration Flags")

    # 1. SETUP DEBUG/PDB FLAGS
    # Use -g and -gcodeview for Windows PDB compatibility
    set(DEBUG_FLAGS "-g" "-gcodeview")
    
    # 2. SETUP SANITIZER FLAGS (If enabled)
    set(SAN_FLAGS "")
    if(ENABLE_SANITIZER)
        set(SAN_FLAGS "-fsanitize=address" "-fsanitize=undefined" "-fno-omit-frame-pointer")
    endif()

    # 3. APPLY TO TARGET
    # We add these directly to the target to ensure they aren't overridden by global options
    target_compile_options(_culverin_c PRIVATE 
        ${DEBUG_FLAGS} 
        ${SAN_FLAGS}
        "-std=c2x"
    )

    # 4. THE LINKER IS THE KEY
    if(WIN32)
        target_link_options(_culverin_c PRIVATE 
            "-Wl,-debug"           
            "-Wl,-nodefaultlib:libcmt" 
            ${SAN_FLAGS}           
        )
    else()
        # Mac/Linux don't need those Windows flags!
        target_link_options(_culverin_c PRIVATE ${SAN_FLAGS})
    endif()

    # Performance flags
    set(PERF_FLAGS "-O3" "-ffast-math" "-fomit-frame-pointer")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND PERF_FLAGS "-mavx2" "-mfma" "-march=native")
    endif()

    # Use a list for cleaner appending
    set(STRICT_FLAGS 
        "-Wall" 
        "-Wextra" 
        # "-Wconversion"              # Turns EVERYTHING on (including signs)
        "-Wno-sign-conversion"       # Specifically turns signs OFF
        "-Wno-shorten-64-to-32"      # Turns off the size_t -> uint32_t nag
        "-Wno-double-promotion"      # Since you're sick of the Python "f" noise
        # "-Werror=conversion"         # Treat what's LEFT of conversion as an error
        # "-fmax-errors=5"
    )

    # Apply PERF flags to the whole project
    add_compile_options("${PERF_FLAGS}")
    add_compile_options(-fcolor-diagnostics)
    add_compile_options(-fdiagnostics-color=always)

    # FORCE the strict flags onto your specific module
    # We use BEFORE to ensure they are evaluated early, or 
    # simply ensure no other target_compile_options calls follow this.
    target_compile_options(_culverin_c PRIVATE ${STRICT_FLAGS})

    # Ensure Jolt is treated as SYSTEM to prevent its internal 
    # promotions from triggering your new Werror
    target_include_directories(_culverin_c SYSTEM PRIVATE 
        "extern/JoltC/include" 
        "${JOLT_DIR}"
    )
endif()
target_include_directories(_culverin_c PRIVATE 
    "src/culverin"
    SYSTEM "extern/JoltC/include" 
    SYSTEM "${JOLT_DIR}" 
    ${Python_INCLUDE_DIRS}
)

if(WIN32)
    target_link_libraries(_culverin_c PRIVATE Winmm.lib)
endif()
target_compile_definitions(_culverin_c PRIVATE JPH_DEBUG_RENDERER)
if(Python_VERSION VERSION_GREATER_EQUAL "3.13")
    # Only disable the GIL if the current interpreter actually supports it
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sys; print(int(bool(sys._get_gil_config().get('enabled', True) == False)))"
        OUTPUT_VARIABLE PYTHON_IS_FREE_THREADED
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    if(PYTHON_IS_FREE_THREADED EQUAL 1)
        message(STATUS "Culverin: Free-threading detected. Disabling GIL.")
        target_compile_definitions(_culverin_c PRIVATE Py_GIL_DISABLED=1)
    else()
        message(STATUS "Culverin: Standard Python detected. GIL remains enabled.")
    endif()
endif()

if(APPLE)
    message(STATUS "Culverin: Enforcing Strong Symbol Export")

    # Force RTTI ON for the wrapper so it can see Jolt's RTTI
    target_compile_options(${JOLTC_TARGET} PRIVATE "-frtti")
    target_compile_options(_culverin_c PRIVATE "-frtti")

    target_link_libraries(_culverin_c PRIVATE ${JOLTC_TARGET} Jolt)

    target_link_options(_culverin_c PRIVATE 
        "-Wl,-all_load"
        "-Wl,-export_dynamic"
        "-Wl,-flat_namespace" 
    )
endif()
set_target_properties(_culverin_c PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX "")

# Final sanity check: ensure our own targets are also /MD
set_property(TARGET ${JOLTC_TARGET} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
set_property(TARGET _culverin_c PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

install(TARGETS _culverin_c DESTINATION .)
if(MSVC OR WIN32)
    # We use a Generator Expression to find the exact PDB location
    install(FILES $<TARGET_PDB_FILE:_culverin_c> DESTINATION . OPTIONAL)
endif()
install(FILES src/culverin/__init__.py src/culverin/_culverin.py DESTINATION .)