cmake_minimum_required(VERSION 3.21)

# ==============================================================================
# 0. THE COMPILER ENFORCER
# ==============================================================================
function(enforce_modern_compiler)
    # --- ANTI-MSBUILD / ANTI-VS GENERATOR GUARD ---
    # MSBuild is the bane of build systems. We need a modern/reliable generator (Ninja/Make).
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        message(
            FATAL_ERROR
            "\n ################################################################\n"
            "  GENERATOR ERROR - HALTED: Visual Studio / MSBuild detected.\n"
            " ################################################################\n"
            "  Culverin's build system is incompatible with the \n"
            "  Visual Studio project system's rigid toolchain handling.\n"
            "  - The reason being: The MSBuild generator will default to cl.exe, \n"
            "  which is currently forbidden from compiling this project. \n"
            "\n"
            "  I DEMAND you to use the 'Ninja' generator for Clang to work correctly.\n"
            "\n"
            "  Instruction:\n"
            "  1. Install Ninja (pip install ninja or winget install ninja)\n"
            "  2. Wipe your 'build' folder.\n"
            "  3. Run: cmake -G \"Ninja\" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..\n"
            " ################################################################\n"
        )
    endif()
    if(CMAKE_C_COMPILER_ID AND NOT CMAKE_C_COMPILER_ID MATCHES "Clang|GNU")
        message(STATUS "Detected Compiler ID: ${CMAKE_C_COMPILER_ID}")
        message(STATUS "Culverin: MSVC detected. Searching for alternatives...")

        find_program(CLANG_PATH NAMES clang clang-cl)
        find_program(GCC_PATH NAMES gcc)

        set(HELP_MSG
            "\n################################################################\n"
        )
        string(
            APPEND HELP_MSG
            " Guard triggered: MSVC (cl.exe) is incapable of compilation.\n"
        )
        string(
            APPEND HELP_MSG
            "################################################################\n"
        )
        string(
            APPEND HELP_MSG
            " C23 features and Jolt optimization require Clang or GCC.\n\n"
        )

        if(CLANG_PATH OR GCC_PATH)
            string(APPEND HELP_MSG " Alternatives on your computer:\n")
            if(CLANG_PATH)
                string(
                    APPEND HELP_MSG
                    "  - Clang's emplacement: ${CLANG_PATH}\n"
                )
            endif()
            if(GCC_PATH)
                string(APPEND HELP_MSG "  - GCC's emplacement:   ${GCC_PATH}\n")
            endif()
            string(
                APPEND HELP_MSG
                "\n Instruction: Decimate your 'build' folder and run:\n"
            )
            string(
                APPEND HELP_MSG
                " cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..\n"
            )
        else()
            string(
                APPEND HELP_MSG
                " NO SUITABLE COMPILER FOUND in system PATH.\n"
            )
            string(
                APPEND HELP_MSG
                " Please download and install LLVM (Clang) from:\n"
            )
            string(
                APPEND HELP_MSG
                " https://github.com/llvm/llvm-project/releases\n"
            )
        endif()

        string(
            APPEND HELP_MSG
            "################################################################\n"
        )
        message(FATAL_ERROR "${HELP_MSG}")
    endif()
endfunction()

# 1. THE NUCLEAR INITIALIZATION
option(ENABLE_SANITIZER "Enable Clang Sanitizers (ASan/TSan/UBSan)" OFF)
option(ENABLE_THREAD_SANITIZER "Enable Thread Sanitizer (TSan)" OFF)

set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)
set(GENERATE_SAMPLES OFF CACHE BOOL "" FORCE)
set(GENERATE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(USE_STATIC_MSVC_RUNTIME OFF CACHE BOOL "" FORCE)
set(FORCE_MT_RUNTIME OFF CACHE BOOL "" FORCE)
set(USE_VULKAN OFF CACHE BOOL "" FORCE)

set(FORCE_RTTI ON CACHE BOOL "Force RTTI on" FORCE)

project(culverin LANGUAGES C CXX)
# 1.1 FORCE BUILD TYPE (RelWithDebInfo is the sweet spot for ASan)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES AND ENABLE_SANITIZER)
    message(
        STATUS
        "Culverin: Setting build type to 'RelWithDebInfo' as none was specified."
    )
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING
        "Choose the type of build."
        FORCE
    )
endif()

# 1.2 ENSURE PDBs ARE GENERATED FOR CLANG-CL/MSVC
if(MSVC OR (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    # This ensures .pdb files are created even in Release/RelWithDebInfo
    if(ENABLE_SANITIZER OR ENABLE_THREAD_SANITIZER)
        message(
            STATUS
            "Culverin: Enabling debug information for sanitizers (PDB generation)"
        )
        set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
            "ProgramDatabase"
            CACHE STRING
            ""
            FORCE
        )
    endif()
endif()
if(ENABLE_SANITIZER)
    message(STATUS "Culverin: FORCING Global Sanitizers for all targets")

    # 1. Standard approach (works for most)
    add_compile_options("-fsanitize=address" "-fsanitize=undefined")
    add_link_options("-fsanitize=address" "-fsanitize=undefined")

    # 2. The Sledgehammer (Force it into the global CXX flags for subdirectories)
    set(CMAKE_CXX_FLAGS
        "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined"
    )
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined"
    )

    # 3. Specifically tell Jolt to use them if it has its own toggle
    set(JPH_ADDRESS_SANITIZER ON CACHE BOOL "" FORCE)
endif()
if(ENABLE_THREAD_SANITIZER)
    message(STATUS "Culverin: FORCING Thread Sanitizer for all targets")

    # 1. Standard approach (works for most)
    add_compile_options("-fsanitize=thread")
    add_link_options("-fsanitize=thread")

    # 2. The Sledgehammer (Force it into the global CXX flags for subdirectories)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS
        "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread"
    )

    # 3. Specifically tell Jolt to use them if it has its own toggle
    set(JPH_THREAD_SANITIZER ON CACHE BOOL "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enforce_modern_compiler()
message(STATUS "--- COMPILER VERIFICATION ---")
message(STATUS "C Compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Clang Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "-----------------------------")

if(NOT MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-frtti>")
endif()

# NATIVE C++23 SUPPORT
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# NATIVE C23 SUPPORT
if(MSVC AND CMAKE_C_COMPILER_ID MATCHES "Clang")
    # THE CLANG-CL PROBLEM: MSVC doesn't have a /std:c23 flag, so CMake lacks the mapping
    # for Clang-CL. We must bypass CMake's checking here and manually inject it below.
    message(
        STATUS
        "Culverin: Bypassing native C_STANDARD detection for Clang-CL (Missing CMake mapping)"
    )
else()
    # The normal human being way for GCC and standard Clang (Linux/Mac/MinGW)
    set(CMAKE_C_STANDARD 23)
    set(CMAKE_C_STANDARD_REQUIRED ON)
    set(CMAKE_C_EXTENSIONS ON)
endif()

add_definitions(-DJPH_DOUBLE_PRECISION)
add_definitions(-DJPH_OBJECT_LAYER_BITS=32)

# ==============================================================================
# 2. COMPILER SPECIFIC FLAGS
# ==============================================================================
set(CULVERIN_COMPILE_FLAGS "")
set(CULVERIN_LINK_FLAGS "")

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        message(STATUS "Culverin: Detected Clang-CL (MSVC-like)")

        # Inject C23 directly to the LLVM frontend behind MSVC's back
        add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:/clang:-std=gnu23>"
            "$<$<COMPILE_LANGUAGE:C>:-Wno-c23-extensions>"
            "$<$<COMPILE_LANGUAGE:C>:-Wno-c2x-extensions>"
        )

        add_compile_options(/arch:AVX2)
        set(RELEASE_FLAGS
            "/clang:-O3 /clang:-ffast-math /clang:-fstrict-aliasing /Ob2 /Oi /Ot"
        )
    else()
        message(STATUS "Culverin: Detected Pure MSVC")
        set(RELEASE_FLAGS "/Ox /Ot /Ob2 /Oi")
    endif()

    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RELEASE_FLAGS}")
else()
    message(STATUS "Culverin: Configuring GNU-like flags...")

    set(FLAGS_DEBUG "-g -gcodeview")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND FLAGS_DEBUG "-gcodeview")
    endif()

    set(FLAGS_PERF "-O3" "-ffast-math" "-fomit-frame-pointer")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND FLAGS_PERF "-mavx2" "-mfma")
    endif()

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(
            APPEND FLAGS_PERF
            "SHELL:-mllvm -unroll-threshold=1000"
            "SHELL:-mllvm -inline-threshold=2000"
            "SHELL:-mllvm -unroll-allow-partial"
            "SHELL:-mllvm -force-vector-width=8"
        )
        list(APPEND FLAGS_STRICT "-Wno-c23-extensions" "-Wno-c2x-extensions")
    endif()

    set(FLAGS_STRICT
        "-Wall"
        "-Wextra"
        "-Watomic-implicit-seq-cst"
        "-Wsync-fetch-and-nand-semantics-changed"
        "-Wno-sign-conversion"
        "-Wno-shorten-64-to-32"
        "-Wno-double-promotion"
    )

    list(
        APPEND CULVERIN_COMPILE_FLAGS
        ${FLAGS_DEBUG}
        ${FLAGS_PERF}
        ${FLAGS_STRICT}
    )

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND CULVERIN_COMPILE_FLAGS "-fcolor-diagnostics")
    else()
        list(APPEND CULVERIN_COMPILE_FLAGS "-fdiagnostics-color=always")
    endif()
endif()

# ==============================================================================
# 3. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(
    JOLT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics"
    ABSOLUTE
)
add_definitions(-DJPH_DEBUG_RENDERER)
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE ON CACHE BOOL "" FORCE)

if(DEFINED JOLT_PREBUILT_DIR)
    find_package(
        Jolt
        REQUIRED
        CONFIG
        PATHS "${JOLT_PREBUILT_DIR}"
        NO_DEFAULT_PATH
    )
    # IMPORTED_LOCATION_RELEASE only matches config "Release" exactly.
    # scikit-build-core may use a different config name, so set it unconditionally.
    set_target_properties(
        Jolt::Jolt
        PROPERTIES
            IMPORTED_LOCATION "${JOLT_PREBUILT_DIR}/../../../lib/libJolt.a"
    )
else()
    # Force standard globally BEFORE add_subdirectory
    set(CMAKE_CXX_STANDARD 23)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

    if(MSVC OR (WIN32 AND CMAKE_C_COMPILER_ID MATCHES "Clang"))
        foreach(t Jolt)
            if(TARGET ${t})
                set_property(
                    TARGET ${t}
                    PROPERTY
                        MSVC_RUNTIME_LIBRARY
                            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
                )
                set_target_properties(
                    ${t}
                    PROPERTIES
                        CXX_STANDARD 23
                        CXX_STANDARD_REQUIRED ON
                        CXX_EXTENSIONS OFF
                )
                if(CMAKE_C_COMPILER_ID MATCHES "Clang")
                    set_property(TARGET ${t} PROPERTY COMPILE_OPTIONS "")
                    if(MSVC)
                        target_compile_options(${t} PRIVATE /arch:AVX2 /fp:fast)
                    else()
                        target_compile_options(
                            ${t}
                            PRIVATE
                                -march=x86-64-v3
                                -ffast-math
                                -fomit-frame-pointer
                        )
                    endif()
                endif()
            endif()
        endforeach()
    endif()
endif()

# ==============================================================================
# 4. BUILD JOLTC, MAG-MUTEX & EXTENSION
# ==============================================================================
set(JOLTC_TARGET joltc_double)
add_library(
    ${JOLTC_TARGET}
    STATIC
    "extern/JoltC/src/joltc.cpp"
    "extern/JoltC/src/joltc_assert.cpp"
    "extern/JoltC/src/joltc.c"
)

if(MSVC)
    target_compile_options(${JOLTC_TARGET} PRIVATE /W0)
else()
    target_compile_options(${JOLTC_TARGET} PRIVATE -w)
endif()

target_include_directories(
    ${JOLTC_TARGET}
    PUBLIC "extern/JoltC/include" "${JOLT_DIR}"
)
target_link_libraries(${JOLTC_TARGET} PRIVATE Jolt::Jolt)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

set(MAG_MUTEX_TARGET mag_mutex)
add_library(
    ${MAG_MUTEX_TARGET}
    STATIC
    "extern/mag-mutex/src/mag_mutex.c"
)

if(MSVC)
    target_compile_options(${MAG_MUTEX_TARGET} PRIVATE /W0)
else()
    target_compile_options(${MAG_MUTEX_TARGET} PRIVATE -w)
endif()

target_include_directories(
    ${MAG_MUTEX_TARGET}
    PUBLIC "extern/mag-mutex/include" 
)

python_add_library(
    _culverin_c
    MODULE
    src/culverin/culverin.c
    src/culverin/culverin_shadow_sync.cpp
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
    src/culverin/culverin_arg_indices.c
    src/culverin/culverin_handler.c
)

target_sources(_culverin_c PRIVATE src/culverin/culverin_jph_rtti_shim.cpp)
if(MSVC)
    set_source_files_properties(
        src/culverin/culverin_jph_rtti_shim.cpp
        PROPERTIES COMPILE_OPTIONS "/GR"
    )
else()
    set_source_files_properties(
        src/culverin/culverin_jph_rtti_shim.cpp
        PROPERTIES COMPILE_OPTIONS "-frtti"
    )
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    message(STATUS "Culverin: LTO/IPO is supported and ENABLED")
    set_property(TARGET _culverin_c PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_property(
        TARGET ${JOLTC_TARGET}
        PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE
    )
    if(TARGET Jolt AND NOT DEFINED JOLT_PREBUILT_DIR)
        set_property(TARGET Jolt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
else()
    message(WARNING "Culverin: LTO/IPO not supported: ${error}")
endif()

add_custom_command(
    TARGET _culverin_c
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E echo
        "Extension built at: $<TARGET_FILE:_culverin_c>"
)

if(MSVC)
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        # --- LTO LOGIC ---
        if(NOT ENABLE_SANITIZER AND NOT ENABLE_THREAD_SANITIZER)
            # The compiler still wants -flto=thin
            set(LTO_COMPILE_FLAG "-flto=thin")

            # The linker (lld-link) wants level 2 for ThinLTO
            # Instead of /opt:lldlto=thin, use:
            set(LTO_LINK_FLAG "/opt:lldlto=2")

            target_compile_options(_culverin_c PRIVATE ${LTO_COMPILE_FLAG})
            target_link_options(_culverin_c PRIVATE ${LTO_LINK_FLAG})

            target_compile_options(${JOLTC_TARGET} PRIVATE ${LTO_COMPILE_FLAG})
        endif()
    endif()

    # --- PERFORMANCE & DEBUG FLAGS ---
    target_compile_options(_culverin_c PRIVATE /Zi /fp:fast)
    target_link_options(_culverin_c PRIVATE /DEBUG /OPT:REF /OPT:ICF)
    target_compile_definitions(_culverin_c PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)

    # --- THE MISSING INCLUDES ---
    # We use SYSTEM for Jolt because we don't want to see C++ warnings in our C23 build
    target_include_directories(
        _culverin_c
        SYSTEM
        PRIVATE "extern/JoltC/include" "${JOLT_DIR}"
    )
    target_include_directories(
        ${JOLTC_TARGET}
        SYSTEM
        PRIVATE "extern/JoltC/include" "${JOLT_DIR}"
    )
else()
    # GCC / Standard Clang (Linux/MinGW)
    if(NOT ENABLE_SANITIZER AND NOT ENABLE_THREAD_SANITIZER)
        target_compile_options(_culverin_c PRIVATE -flto=thin)
        target_compile_options(${JOLTC_TARGET} PRIVATE -flto=thin)
    endif()

    if(WIN32)
        set(WIN_LINK_FLAGS "-Wl,-nodefaultlib:libcmt") # CRITICAL: Prevents 3.14 Heap Crashes

        if(ENABLE_SANITIZER OR ENABLE_THREAD_SANITIZER)
            list(APPEND WIN_LINK_FLAGS "-Wl,-debug")
        endif()

        if(NOT ENABLE_SANITIZER AND NOT ENABLE_THREAD_SANITIZER)
            list(APPEND WIN_LINK_FLAGS "-flto=thin")
        endif()

        target_link_options(
            _culverin_c
            PRIVATE ${WIN_LINK_FLAGS} ${CULVERIN_LINK_FLAGS}
        )
    endif()

    target_include_directories(
        _culverin_c
        SYSTEM
        PRIVATE "extern/JoltC/include" "${JOLT_DIR}"
    )
    target_include_directories(
        ${JOLTC_TARGET}
        SYSTEM
        PRIVATE "extern/JoltC/include" "${JOLT_DIR}"
    )
endif()

target_compile_definitions(
    _culverin_c
    PRIVATE $<$<NOT:$<BOOL:${MSVC}>>:_GNU_SOURCE>
)

target_include_directories(
    _culverin_c
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" 
    "src/culverin"
    "extern/mag-mutex/include"
    SYSTEM PRIVATE
    "extern/JoltC/include"
    "${JOLT_DIR}"
    ${Python_INCLUDE_DIRS}
)

target_compile_definitions(
    _culverin_c
    PRIVATE CULVERIN_DOCS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/docs/DOCS.md"
)

if(WIN32)
    target_link_libraries(_culverin_c PRIVATE Winmm.lib)
endif()
target_compile_definitions(_culverin_c PRIVATE JPH_DEBUG_RENDERER)
target_link_libraries(_culverin_c PRIVATE ${MAG_MUTEX_TARGET} ${JOLTC_TARGET} Jolt::Jolt)

if(APPLE)
    message(STATUS "Culverin: Enforcing Strong Symbol Export")
    target_compile_options(${JOLTC_TARGET} PRIVATE "-frtti")
    target_compile_options(_culverin_c PRIVATE "-frtti")
    target_link_options(_culverin_c PRIVATE "-Wl,-all_load")
endif()

set_target_properties(
    _culverin_c
    PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX ""
)

set_property(
    TARGET ${JOLTC_TARGET}
    PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)
set_property(
    TARGET _culverin_c
    PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

install(TARGETS _culverin_c DESTINATION .)
if(MSVC OR WIN32)
    install(FILES $<TARGET_PDB_FILE:_culverin_c> DESTINATION . OPTIONAL)
endif()
install(FILES src/culverin/__init__.py src/culverin/_culverin.py DESTINATION .)
