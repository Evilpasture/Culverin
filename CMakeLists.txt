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

project(culverin LANGUAGES C CXX ASM)
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
message(STATUS "ID:           ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Version:      ${CMAKE_CXX_COMPILER_VERSION}")
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

    set(FLAGS_DEBUG "-g")
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(APPEND FLAGS_DEBUG "-gcodeview")
    endif()

    set(FLAGS_PERF "-O3" "-ffast-math" "-fomit-frame-pointer")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        list(APPEND FLAGS_PERF "-mavx2" "-mfma")
    endif()

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND WIN32)
        # -ffast-math breaks stdatomic memory ordering on MinGW
        set(FLAGS_PERF_C "-O3" "-fomit-frame-pointer")
    else()
        set(FLAGS_PERF_C ${FLAGS_PERF})
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
        "-Wno-sign-conversion"
        "-Wno-shorten-64-to-32"
        "-Wno-double-promotion"
    )

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        list(
            APPEND FLAGS_STRICT
            "-Watomic-implicit-seq-cst"
            "-Wsync-fetch-and-nand-semantics-changed"
            "-Wno-c23-extensions"
            "-Wno-c2x-extensions"
        )
    endif()

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        list(APPEND FLAGS_STRICT "-Wno-psabi") # Suppress AVX parameter passing ABI warnings
    endif()

    list(
        APPEND CULVERIN_COMPILE_FLAGS
        ${FLAGS_DEBUG}
        ${FLAGS_PERF_C}
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

if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND WIN32)
    find_program(GCC_AR NAMES gcc-ar)
    find_program(GCC_RANLIB NAMES gcc-ranlib)
    if(GCC_AR)
        set(CMAKE_AR "${GCC_AR}")
    endif()
    if(GCC_RANLIB)
        set(CMAKE_RANLIB "${GCC_RANLIB}")
    endif()
endif()

add_subdirectory("extern/mag-mutex")

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
    src/culverin/culverin_arg_indices.c
    src/culverin/culverin_handler.c
    src/culverin/culverin_soft_body.c
    src/culverin/culverin_math.cpp
    src/culverin/culverin_math.c
    src/culverin/culverin_ecs.c
    src/culverin/culverin_ship.c
    src/culverin/culverin_physics_world.c
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

# In section 4, guard IPO for GCC on Windows:
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported)
    # GCC LTO on Windows (MinGW/LLD) is broken for Python extensions.
    # The INTERPROCEDURAL_OPTIMIZATION property fights with our -fno-lto workaround
    # and PyInit_ visibility gets lost at link time.
    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" AND WIN32)
        message(STATUS "Culverin: GCC+Windows detected — skipping CMake IPO to avoid PyInit_ visibility loss")
    else()
        message(STATUS "Culverin: LTO/IPO is supported and ENABLED")
        set_property(TARGET _culverin_c PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        set_property(TARGET ${JOLTC_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        if(TARGET Jolt AND NOT DEFINED JOLT_PREBUILT_DIR)
            set_property(TARGET Jolt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        endif()
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

# ==============================================================================
# 4.1 BUILD FAST-CAPI (Via Submodule CMake)
# ==============================================================================
# 1. Use the submodule's own CMake logic
add_subdirectory("extern/fast-capi")

# 2. Extract the target name from the submodule (it's called 'fast_capi_core')
set(FAST_CAPI_TARGET fast_capi_core)

# 3. Apply Culverin's high-performance flags to the submodule target
target_compile_options(${FAST_CAPI_TARGET} PRIVATE ${CULVERIN_COMPILE_FLAGS})

# 4. Enforce IPO/LTO if supported
if(ipo_supported AND NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND WIN32))
    set_property(TARGET ${FAST_CAPI_TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ==============================================================================
# 4.2 BUILD DOCS-EMBEDDER (Via Submodule CMake)
# ==============================================================================
add_subdirectory("extern/docs-embedder")

# Apply Culverin's high-performance flags to the submodule
target_compile_options(docs_embedder PRIVATE ${CULVERIN_COMPILE_FLAGS})

# Enforce IPO/LTO if supported
if(ipo_supported AND NOT (CMAKE_C_COMPILER_ID STREQUAL "GNU" AND WIN32))
    set_property(TARGET docs_embedder PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# --- #embed Fallback Generator for Ancient Compilers which are probably older than me ---
# 1. Ensure the generated directory exists immediately because sometimes it wouldn't work on other environments or compilers
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

# 2. Correct paths relative to the project root (where pyproject.toml lives)
set(EMBED_FILES 
    "PYPROJECT_TOML:pyproject.toml"
    "ALL_DOCS:docs/DOCS.md"
)

foreach(entry ${EMBED_FILES})
    string(REPLACE ":" ";" pair ${entry})
    list(GET pair 0 var_name)
    list(GET pair 1 file_path)
    
    set(absolute_src "${CMAKE_CURRENT_SOURCE_DIR}/${file_path}")
    set(inc_file "${CMAKE_CURRENT_BINARY_DIR}/generated/${var_name}.inc")
    
    # 3. Defensive check: only create the rule if the file actually exists
    if(EXISTS "${absolute_src}")
        add_custom_command(
            OUTPUT "${inc_file}"
            COMMAND ${Python_EXECUTABLE} -c 
            "import sys; data=open(sys.argv[1], 'rb').read(); open(sys.argv[2], 'w').write(','.join(map(str, data)) + ',0')"
            "${absolute_src}" "${inc_file}"
            DEPENDS "${absolute_src}"
            COMMENT "Generating fallback for ${file_path}"
            VERBATIM
        )
        list(APPEND FALLBACK_HEADERS "${inc_file}")
    else()
        message(WARNING "Culverin: Could not find ${absolute_src} for embedding!")
    endif()
endforeach()

# Attach to the target
target_sources(_culverin_c PRIVATE ${FALLBACK_HEADERS})
target_include_directories(_culverin_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

# ==============================================================================
# 5. GLOBAL LTO / IPO CONFIGURATION (Compiler Agnostic)
# ==============================================================================
if(NOT ENABLE_SANITIZER AND NOT ENABLE_THREAD_SANITIZER)
    set(LTO_COMPILE_FLAG "")
    set(LTO_LINK_FLAG "")

    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        # LLVM / Clang-CL specific ThinLTO
        set(LTO_COMPILE_FLAG "-flto=thin")
        if(MSVC)
            set(LTO_LINK_FLAG "/opt:lldlto=2")
        else()
            set(LTO_LINK_FLAG "-flto=thin")
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
        # GCC LTO on Windows often breaks PyInit visibility. 
        # Disable it for now to confirm the fix, or use the "fat" workaround:
        set(LTO_COMPILE_FLAG "-fno-lto" "-fvisibility=hidden") 
        set(LTO_LINK_FLAG "")
    endif()

    if(LTO_COMPILE_FLAG)
        message(STATUS "Culverin: Applying ${CMAKE_C_COMPILER_ID} LTO optimizations")
        target_compile_options(_culverin_c PRIVATE ${LTO_COMPILE_FLAG})
        target_link_options(_culverin_c PRIVATE ${LTO_LINK_FLAG})
        target_compile_options(${JOLTC_TARGET} PRIVATE ${LTO_COMPILE_FLAG})
        
        if(TARGET Jolt AND NOT DEFINED JOLT_PREBUILT_DIR)
            target_compile_options(Jolt PRIVATE ${LTO_COMPILE_FLAG})
        endif()
    endif()
endif()

# ==============================================================================
# 6. PLATFORM & LINKER FLAGS
# ==============================================================================
if(MSVC)
    # --- MSVC / Clang-CL PERFORMANCE & DEBUG FLAGS ---
    target_compile_options(_culverin_c PRIVATE /Zi /fp:fast)
    target_link_options(_culverin_c PRIVATE /DEBUG /OPT:REF /OPT:ICF)
    target_compile_definitions(_culverin_c PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)
else()
    # --- GNU / MinGW LINKER FLAGS ---
    if(WIN32)
        set(WIN_LINK_FLAGS "")
        
        if(CMAKE_C_COMPILER_ID MATCHES "Clang")
            # Only Clang on Windows uses this MSVC-style nodefaultlib format via lld
            list(APPEND WIN_LINK_FLAGS "-Wl,-nodefaultlib:libcmt")
        else()
            # 1. Use LLD if available for speed
            list(APPEND WIN_LINK_FLAGS "-fuse-ld=lld")

            # 2. Static GCC/C++ Runtimes
            list(APPEND WIN_LINK_FLAGS "-static-libgcc")
            list(APPEND WIN_LINK_FLAGS "-static-libstdc++")

            # 3. THE MAGIC TRICK: Link winpthread statically without breaking the CRT
            # This tells the linker: "Start linking statically now"
            list(APPEND WIN_LINK_FLAGS "-Wl,-Bstatic")
            list(APPEND WIN_LINK_FLAGS "-lwinpthread")
            # This tells the linker: "Go back to dynamic linking for everything else"
            list(APPEND WIN_LINK_FLAGS "-Wl,-Bdynamic")

            # 4. Standard Windows Multimedia (already dynamic)
            list(APPEND WIN_LINK_FLAGS "-lwinmm")
        endif()

        if(ENABLE_SANITIZER OR ENABLE_THREAD_SANITIZER)
            list(APPEND WIN_LINK_FLAGS "-Wl,-debug")
        endif()

        target_link_options(
            _culverin_c
            PRIVATE ${WIN_LINK_FLAGS} ${CULVERIN_LINK_FLAGS}
        )
    endif()
endif()

# ==============================================================================
# 7. UNIFIED INCLUDES (DRY)
# ==============================================================================
# We use SYSTEM for Jolt to suppress C++ warnings bleeding into our C23 build
target_include_directories(
    _culverin_c
    SYSTEM PRIVATE "extern/JoltC/include" "${JOLT_DIR}"
)
target_include_directories(
    ${JOLTC_TARGET}
    SYSTEM PRIVATE "extern/JoltC/include" "${JOLT_DIR}"
)

target_compile_definitions(
    _culverin_c
    PRIVATE $<$<NOT:$<BOOL:${MSVC}>>:_GNU_SOURCE>
)

target_include_directories(
    _culverin_c
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" 
    "src/culverin"
    "extern/mag-mutex/include"
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
target_link_libraries(_culverin_c PRIVATE mag_mutex_lib ${JOLTC_TARGET} fast_capi_core Jolt::Jolt docs_embedder)

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

# ==============================================================================
# 8. INSTALLATION
# ==============================================================================
install(TARGETS _culverin_c DESTINATION .)

# Only attempt to evaluate PDB generator expressions on Windows.
# Even with OPTIONAL, Linux linkers will fail the generation phase if they see this GenEx.
if(WIN32 AND (MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
    install(FILES $<TARGET_PDB_FILE:_culverin_c> 
            DESTINATION . 
            OPTIONAL) 
    message(STATUS "Culverin: Windows PDB installation configured.")
else()
    message(STATUS "Culverin: Non-PDB platform/compiler detected, skipping PDB installation.")
endif()


install(FILES src/culverin/__init__.py src/culverin/_culverin.py DESTINATION .)
