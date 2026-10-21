# ==============================================================================
# Culverin — capability-driven build configuration
# ==============================================================================
# One driver model everywhere: the GNU command line. GCC on Linux, Clang on
# Linux/macOS, and on Windows clang.exe FROM THE LLVM INSTALLER, which targets
# the MSVC *ABI* (same CRT as official CPython) while speaking the GNU *flag
# language*. cl.exe and clang-cl are both forbidden.
#
# Corollary: no MSVC-keyed compiler branches anywhere in this file. Compiler
# decisions are made from
#   1. platform          (WIN32 / APPLE / Linux)
#   2. probed capability (check_*_compiler_flag / check_*_source_compiles)
#   3. toolchain feature namespace (e.g. -mllvm flags are Clang's own)
# ABI-level Windows details are delegated to CMake's FRONTEND_VARIANT-aware
# machinery (CMAKE_MSVC_RUNTIME_LIBRARY / CMAKE_MSVC_DEBUG_INFORMATION_FORMAT),
# which CMake translates into GNU-driver flags itself. It simply asks what the
# compiler can do.
# ==============================================================================
cmake_minimum_required(VERSION 3.26)

# ==============================================================================
# 0. THE COMPILER ENFORCER
# ==============================================================================
function(enforce_modern_compiler)
    # --- ANTI-MSBUILD / ANTI-VS GENERATOR GUARD ---
    if(CMAKE_GENERATOR MATCHES "Visual Studio")
        message(
            FATAL_ERROR
            "\n ################################################################\n"
            "  GENERATOR ERROR - HALTED: Visual Studio / MSBuild detected.\n"
            " ################################################################\n"
            "  Culverin's build system is incompatible with the Visual Studio\n"
            "  project system (it defaults to cl.exe, which is forbidden).\n"
            "\n"
            "  Instruction:\n"
            "  1. Install Ninja (pip install ninja or winget install ninja)\n"
            "  2. Wipe your 'build' folder.\n"
            "  3. Run: cmake -G \"Ninja\" -DCMAKE_CXX_COMPILER=clang++ ..\n"
            " ################################################################\n"
        )
    endif()

    # cl.exe (compiler ID "MSVC") and clang-cl (cl-syntax dialect) are
    # forbidden. Every supported toolchain speaks GCC-style flags.
    if(CMAKE_CXX_COMPILER_ID AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        find_program(CLANG_PATH NAMES clang++)
        find_program(GCC_PATH NAMES g++)

        string(
            APPEND HELP_MSG
            "\n################################################################\n"
            " Guard triggered: ${CMAKE_CXX_COMPILER_ID} is incapable of compiling this project.\n"
            "################################################################\n"
            " C++23 and Jolt optimization require the GNU flag dialect.\n"
            " Use clang.exe / clang++.exe (NOT clang-cl!) or GCC:\n"
        )
        if(CLANG_PATH)
            string(APPEND HELP_MSG "  - Clang's emplacement: ${CLANG_PATH}\n")
        endif()
        if(GCC_PATH)
            string(APPEND HELP_MSG "  - GCC's emplacement:   ${GCC_PATH}\n")
        endif()
        if(CLANG_PATH OR GCC_PATH)
            string(
                APPEND HELP_MSG
                "\n Instruction: wipe your 'build' folder and run:\n"
                " cmake -DCMAKE_CXX_COMPILER=clang++ ..\n"
            )
        else()
            string(
                APPEND HELP_MSG
                "\n NO SUITABLE COMPILER FOUND in system PATH.\n"
                " Please download and install LLVM (Clang) from:\n"
                " https://github.com/llvm/llvm-project/releases\n"
            )
        endif()
        string(APPEND HELP_MSG "################################################################\n")
        message(FATAL_ERROR "${HELP_MSG}")
    endif()
endfunction()

# ==============================================================================
# 1. THE NUCLEAR INITIALIZATION
# ==============================================================================
option(ENABLE_SANITIZER "Enable Clang Sanitizers (ASan/UBSan)" OFF)
option(ENABLE_THREAD_SANITIZER "Enable Thread Sanitizer (TSan)" OFF)
option(CPH_AGGRESSIVE_OPTIMIZATION "Enable aggressive __builtin_assume for assertions in Release" OFF)

# Jolt build knobs — read by extern/JoltPhysics/Build/CMakeLists.txt and must be
# forced BEFORE add_subdirectory(Jolt). Every supported compiler speaks the GNU
# flag dialect, so only knobs whose effect reaches the GNU arm are listed here.
set(DOUBLE_PRECISION ON CACHE BOOL "Use double precision math" FORCE)
# ObjectLayer is uint32 across Culverin's handles/settings. This must be a
# Jolt build option (Jolt exports it as a PUBLIC compile definition on the
# Jolt::Jolt target); a global -D would fight Jolt's own =16 default on the
# same command lines.
set(OBJECT_LAYER_BITS 32 CACHE STRING "Jolt: ObjectLayer width" FORCE)
set(GENERATE_SAMPLES OFF CACHE BOOL "" FORCE)
set(GENERATE_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(USE_VULKAN OFF CACHE BOOL "" FORCE)
# Jolt defaults RTTI off (-fno-rtti); our extension needs RTTI across the
# Jolt RTTI. Use Jolt's own option — no source patching anywhere.
set(CPP_RTTI_ENABLED ON CACHE BOOL "Jolt: enable RTTI" FORCE)
# No C++ exception handling is used; keep Jolt in the same no-exceptions mode.
set(CPP_EXCEPTIONS_ENABLED OFF CACHE BOOL "Jolt: disable C++ exceptions" FORCE)
# Third-party -Werror must not gate our superbuild (Jolt adds -Wall -Werror
# on its GNU arm when this is ON).
set(ENABLE_ALL_WARNINGS OFF CACHE BOOL "Jolt: no warning escalation" FORCE)

# --- MSVC-ABI knobs, delegating to CMake (honored only when the compiler ---
# --- targets the MSVC ABI; inert everywhere else) ---
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

project(culverin LANGUAGES CXX)

if(WIN32)
    # Apply the Windows SDK's own switches before any header can include Windows.h.
    add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

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

# Sanitizers carry full debug info; on the MSVC ABI CMake translates this to
# -gcodeview (PDBs). Inert elsewhere.
if(ENABLE_SANITIZER OR ENABLE_THREAD_SANITIZER)
    message(STATUS "Culverin: Enabling debug information for sanitizers")
    set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase" CACHE STRING "" FORCE)
endif()

if(ENABLE_SANITIZER)
    message(STATUS "Culverin: FORCING Global Sanitizers for all targets")

    add_compile_options("-fsanitize=address" "-fsanitize=undefined")
    add_link_options("-fsanitize=address" "-fsanitize=undefined")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize=undefined")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address -fsanitize=undefined")

    set(JPH_ADDRESS_SANITIZER ON CACHE BOOL "" FORCE)
endif()

if(ENABLE_THREAD_SANITIZER)
    # There is no TSan runtime for any Windows target, driver choice notwithstanding.
    if(WIN32)
        message(
            FATAL_ERROR
            "Culverin: Thread Sanitizer is not available on Windows targets.\n"
            "Run the TSan validation on Linux (see check_tsan in build_wheels.yml)."
        )
    endif()

    message(STATUS "Culverin: FORCING Thread Sanitizer for all targets")

    add_compile_options("-fsanitize=thread")
    add_link_options("-fsanitize=thread")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=thread")

    set(JPH_THREAD_SANITIZER ON CACHE BOOL "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
enforce_modern_compiler()

# ==============================================================================
# 1.5 CAPABILITY PROBING — "ask what the compiler can do"
# ==============================================================================
include(CheckCXXCompilerFlag)

# CPH_collect_supported_cxx_flags(<out_var> <flag>...)
# Appends every flag the C++ compiler accepts to <out_var>.
function(CPH_collect_supported_cxx_flags out)
    set(_ok "")
    foreach(flag IN LISTS ARGN)
        string(REGEX REPLACE "[^A-Za-z0-9]" "_" _var "CPH_HAVE_CXX${flag}")
        string(TOUPPER "${_var}" _var)
        check_cxx_compiler_flag("${flag}" ${_var})
        if(${_var})
            list(APPEND _ok "${flag}")
        endif()
    endforeach()
    set(${out} "${_ok}" PARENT_SCOPE)
endfunction()

# ==============================================================================
# ASSERTION LEVEL MAPPING
# ==============================================================================
# Logic:
# - If config is Debug: Level 2 (Full)
# - If Sanitizers are ON: Level 2 (Full) - sanitizers imply a debug context
# - If Aggressive Opt is ON and config is Release: Level 0 (Assume)
# - Default (RelWithDebInfo / Release / MinSizeRel): Level 1 (Trap)
set(CPH_ASSERT_DEFS
    $<$<OR:$<CONFIG:Debug>,$<BOOL:${ENABLE_SANITIZER}>,$<BOOL:${ENABLE_THREAD_SANITIZER}>>:CPH_ASSERT_LEVEL=2>
    $<$<AND:$<CONFIG:Release>,$<BOOL:${CPH_AGGRESSIVE_OPTIMIZATION}>>:CPH_ASSERT_LEVEL=0>
    # Catch-all: optimized builds default to Level 1 (Safe Trap).
    $<$<AND:$<NOT:$<CONFIG:Debug>>,$<NOT:$<BOOL:${ENABLE_SANITIZER}>>,$<NOT:$<BOOL:${ENABLE_THREAD_SANITIZER}>>,$<NOT:$<AND:$<CONFIG:Release>,$<BOOL:${CPH_AGGRESSIVE_OPTIMIZATION}>>>>:CPH_ASSERT_LEVEL=1>
)

message(STATUS "--- COMPILER VERIFICATION ---")
message(STATUS "ID:               ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "Flag dialect:     ${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}")
message(STATUS "CXX Compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "Version:          ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "-----------------------------")

# ==============================================================================
# 2. LANGUAGE STANDARDS + GLOBAL RUNTIME TRAITS
# ==============================================================================
# RTTI everywhere — one probed flag, no dialect branches.
CPH_collect_supported_cxx_flags(CPH_FRTTI "-frtti")
if(CPH_FRTTI)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-frtti>")
endif()

# NATIVE C++23 (CMake maps this to the correct -std= spelling itself)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# JPH_DOUBLE_PRECISION / JPH_OBJECT_LAYER_BITS are intentionally NOT global
# definitions: they are Jolt build options (forced above) that Jolt exports
# as PUBLIC compile definitions on Jolt::Jolt, and every Culverin target
# reaches them through culverin_native_support. src/jolt/Jolt.hpp
# static-asserts the resulting ABI wherever the native API is used.

# ==============================================================================
# 3. COMPILER-SPECIFIC FLAGS (probed where the answer varies)
# ==============================================================================
set(CULVERIN_COMPILE_FLAGS "")
set(CULVERIN_LINK_FLAGS "")

# --- Debug info: CodeView/PDB is the native format on Windows. ---
set(FLAGS_DEBUG "-g")
if(WIN32)
    CPH_collect_supported_cxx_flags(CPH_CODEVIEW "-gcodeview")
    list(APPEND FLAGS_DEBUG ${CPH_CODEVIEW})
endif()

# --- Performance baseline (valid on every supported target) ---
set(FLAGS_PERF "-O3" "-ffast-math" "-fomit-frame-pointer")

# --- x86 SIMD/FMA/compressed-exchange: probe, don't assume the ISA ---
CPH_collect_supported_cxx_flags(CPH_X86_PERF "-mavx2" "-mfma" "-mcx16")
list(APPEND FLAGS_PERF ${CPH_X86_PERF})

# --- LLVM middle-end tuning: these flags live in Clang's own namespace ---
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(
        APPEND FLAGS_PERF
        "SHELL:-mllvm -unroll-threshold=1000"
        "SHELL:-mllvm -inline-threshold=2000"
        "SHELL:-mllvm -unroll-allow-partial"
        "SHELL:-mllvm -force-vector-width=8"
    )
endif()

# --- Warning discipline ---
set(FLAGS_STRICT
    "-Wall"
    "-Wextra"
    "-Wno-sign-conversion"
    "-Wno-shorten-64-to-32"
    "-Wno-double-promotion"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(
        APPEND FLAGS_STRICT
        "-Watomic-implicit-seq-cst"
        "-Wsync-fetch-and-nand-semantics-changed"
        "-Wno-c23-extensions"
        "-Wno-c2x-extensions"
    )
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND FLAGS_STRICT "-Wno-psabi") # Suppress AVX parameter passing ABI warnings
endif()

list(APPEND CULVERIN_COMPILE_FLAGS ${FLAGS_DEBUG} ${FLAGS_PERF} ${FLAGS_STRICT})

# --- Colored diagnostics: ask, don't assume ---
CPH_collect_supported_cxx_flags(CPH_COLOR "-fcolor-diagnostics" "-fdiagnostics-color=always")
if(CPH_COLOR)
    # Take only the FIRST supported spelling; GCC and Clang disagree on names.
    list(GET CPH_COLOR 0 CPH_COLOR_FLAG)
    list(APPEND CULVERIN_COMPILE_FLAGS "${CPH_COLOR_FLAG}")
endif()

# ==============================================================================
# 3.1 SHARED TARGET CONFIGURATION
# ==============================================================================
# Centralize flags shared by the project-owned compile targets. Domain targets
# call this helper in their local CMakeLists.txt; IPO/LTO is applied after all
# targets have been declared.
function(culverin_configure_compile_target target)
    # Most flags are C/C++-specific; keep them off assembly translation units.
    target_compile_options(
        ${target}
        PRIVATE
            "$<$<COMPILE_LANGUAGE:C,CXX>:${CULVERIN_COMPILE_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>"
    )
    target_compile_definitions(${target} PRIVATE ${CPH_ASSERT_DEFS} JPH_DEBUG_RENDERER)

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_compile_definitions(${target} PRIVATE _GNU_SOURCE)
    elseif(WIN32)
        # MSVC CRT: allow POSIX-ish names (target ABI property, not a compiler branch).
        target_compile_definitions(${target} PRIVATE _CRT_DECLARE_NONSTDC_NAMES=1)
    endif()

    set_property(GLOBAL APPEND PROPERTY CULVERIN_COMPILE_TARGETS "${target}")
endfunction()

# ==============================================================================
# 4. BUILD JOLT PHYSICS
# ==============================================================================
get_filename_component(
    JOLT_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/extern/JoltPhysics"
    ABSOLUTE
)
# JPH_DEBUG_RENDERER likewise comes from Jolt's PUBLIC definitions once this
# knob is on (no global -D).
set(DEBUG_RENDERER_IN_DEBUG_AND_RELEASE ON CACHE BOOL "" FORCE)

if(DEFINED JOLT_PREBUILT_DIR)
    find_package(
        Jolt
        REQUIRED
        CONFIG
        PATHS "${JOLT_PREBUILT_DIR}"
        NO_DEFAULT_PATH
    )
    # Use the library location exported by Jolt instead of assuming a Unix
    # archive name. The exported path carries the platform's actual suffix
    # (for example, .lib on Windows and .a on Unix).
    get_target_property(_CULVERIN_JOLT_LOCATION Jolt::Jolt IMPORTED_LOCATION_RELEASE)
    if(NOT _CULVERIN_JOLT_LOCATION)
        get_target_property(_CULVERIN_JOLT_LOCATION Jolt::Jolt IMPORTED_LOCATION)
    endif()
    if(NOT _CULVERIN_JOLT_LOCATION)
        message(FATAL_ERROR "Jolt::Jolt does not export an imported library location")
    endif()
    message(STATUS "Culverin: Using prebuilt Jolt at ${_CULVERIN_JOLT_LOCATION}")
    set_target_properties(
        Jolt::Jolt
        PROPERTIES IMPORTED_LOCATION "${_CULVERIN_JOLT_LOCATION}"
    )
    # A prebuilt Jolt was compiled outside our option control; state the
    # consumer side of the ABI explicitly. The archive MUST have been built
    # with the same settings (double precision, 32-bit object layers, debug
    # renderer) — src/jolt/Jolt.hpp static-asserts the visible result.
    target_compile_definitions(
        Jolt::Jolt
        INTERFACE JPH_DOUBLE_PRECISION JPH_OBJECT_LAYER_BITS=32 JPH_DEBUG_RENDERER
    )
else()
    add_subdirectory("${JOLT_DIR}/Build" jolt_physics_build)

    if(TARGET Jolt)
        # Dynamic CRT to match the extension (translated by CMake only when the
        # target is the MSVC ABI; inert elsewhere).
        set_property(
            TARGET Jolt
            PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
        )
        set_target_properties(
            Jolt
            PROPERTIES
                CXX_STANDARD 23
                CXX_STANDARD_REQUIRED ON
                CXX_EXTENSIONS OFF
        )
        # Uniform optimization for Jolt on every platform, composed from what
        # this compiler actually accepts (--march and -ffast-math probed).
        set_property(TARGET Jolt PROPERTY COMPILE_OPTIONS "")
        CPH_collect_supported_cxx_flags(CPH_JOLT_ARCH "-march=x86-64-v3")
        CPH_collect_supported_cxx_flags(CPH_JOLT_PERF "-ffast-math" "-fomit-frame-pointer")
        target_compile_options(
            Jolt
            PRIVATE
                ${CPH_JOLT_ARCH}
                ${CPH_JOLT_PERF}
                "$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>"
        )
    endif()
endif()

# ==============================================================================
# 5. EXTENSION DOMAINS
# ==============================================================================
# JoltC used to be built here as a static archive (joltc_double) and linked
# into every native target. Nothing under src/ calls the C API any more, so
# the archive, its include directory and the extern/JoltC submodule are gone.
# Jolt itself is still reached through the Jolt::Jolt imported target, whose
# INTERFACE properties carry ${JOLT_DIR} and Jolt's own compile definitions.

# C++ engine targets are declared next to their sources under src/.
add_subdirectory("src")

# Python discovery and extension domains come after the engine core boundary.
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
add_subdirectory("src/python")

# ==============================================================================
# 6. INTERPROCEDURAL OPTIMIZATION (ask CMake whether the toolchain can)
# ==============================================================================
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)
if(ipo_supported AND NOT APPLE AND NOT WIN32)
    message(STATUS "Culverin: LTO/IPO is supported and ENABLED")
    get_property(CULVERIN_COMPILE_TARGETS GLOBAL PROPERTY CULVERIN_COMPILE_TARGETS)
    foreach(target IN LISTS CULVERIN_COMPILE_TARGETS)
        set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endforeach()
    set_property(TARGET culverin_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    if(TARGET Jolt AND NOT DEFINED JOLT_PREBUILT_DIR)
        set_property(TARGET Jolt PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
elseif(APPLE OR WIN32)
    # The macOS and Windows wheel builds use different linkers/driver modes
    # (Apple ld and the MSVC ABI). Keep optimization at -O3 but avoid relying
    # on a linker-plugin configuration that varies between hosted runners.
    message(STATUS "Culverin: Skipping IPO/LTO on ${CMAKE_SYSTEM_NAME}")
else()
    message(WARNING "Culverin: LTO/IPO not supported: ${error}")
endif()

# --- #embed Fallback Generator for Ancient Compilers which are probably older than me ---
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Paths relative to the project root (where pyproject.toml lives)
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

    # Defensive check: only create the rule if the file actually exists
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

# Make the generated headers explicit build dependencies. Their .inc suffix is
# intentionally non-compilable, so listing them as target sources alone does not
# reliably schedule the custom commands with every generator.
if(FALLBACK_HEADERS)
    set_source_files_properties(${FALLBACK_HEADERS} PROPERTIES GENERATED TRUE HEADER_FILE_ONLY TRUE)
    add_custom_target(culverin_embed_fallbacks DEPENDS ${FALLBACK_HEADERS})
    add_dependencies(_culverin_c culverin_embed_fallbacks)
    target_sources(_culverin_c PRIVATE ${FALLBACK_HEADERS})
endif()
target_include_directories(_culverin_c PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")

# ==============================================================================
# 7. GLOBAL LTO FLAGS (compiler-identity keyed: these ARE the toolchain's name)
# ==============================================================================
# Only add explicit linker-plugin flags when CMake's IPO probe succeeded.
if(ipo_supported AND NOT ENABLE_SANITIZER AND NOT ENABLE_THREAD_SANITIZER AND NOT APPLE AND NOT WIN32)
    set(LTO_COMPILE_FLAG "")
    set(LTO_LINK_FLAG "")

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(LTO_COMPILE_FLAG "-flto=thin")
        set(LTO_LINK_FLAG "-flto=thin")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # GCC's LTO fights with PyInit visibility in this build; the IPO pass
        # above is intentionally neutralized for GNU by -fno-lto.
        set(LTO_COMPILE_FLAG "-fno-lto" "-fvisibility=hidden")
        set(LTO_LINK_FLAG "")
    endif()

    if(LTO_COMPILE_FLAG)
        message(STATUS "Culverin: Applying ${CMAKE_CXX_COMPILER_ID} LTO optimizations")
        get_property(CULVERIN_COMPILE_TARGETS GLOBAL PROPERTY CULVERIN_COMPILE_TARGETS)
        foreach(target IN LISTS CULVERIN_COMPILE_TARGETS)
            target_compile_options(
                ${target}
                PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:${LTO_COMPILE_FLAG}>"
            )
        endforeach()
        if(LTO_LINK_FLAG)
            target_link_options(_culverin_c PRIVATE ${LTO_LINK_FLAG})
        endif()

        if(TARGET Jolt AND NOT DEFINED JOLT_PREBUILT_DIR)
            target_compile_options(Jolt PRIVATE ${LTO_COMPILE_FLAG})
        endif()
    endif()
endif()

# ==============================================================================
# 8. PLATFORM LINKER & LIBRARY FLAGS
# ==============================================================================
if(WIN32)
    # GNU-driver spelling of the MSVC-linker switches; CMake itself forwards
    # these to lld-link/link.exe via the driver's -Xlinker mechanism.
    target_link_options(
        _culverin_c
        PRIVATE
            "-Xlinker" "/DEBUG"
            "-Xlinker" "/OPT:REF"
            "-Xlinker" "/OPT:ICF"
    )
endif()

if(APPLE)
    message(STATUS "Culverin: Enforcing Strong Symbol Export")
    get_property(CULVERIN_COMPILE_TARGETS GLOBAL PROPERTY CULVERIN_COMPILE_TARGETS)
    foreach(target IN LISTS CULVERIN_COMPILE_TARGETS)
        target_compile_options(
            ${target}
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-frtti>"
        )
    endforeach()
    target_link_options(_culverin_c PRIVATE "-Wl,-all_load")
endif()

# ==============================================================================
# 9. UNIFIED INCLUDES (DRY)
# ==============================================================================
# Jolt's include path and SYSTEM-ness now arrive with the Jolt::Jolt imported
# target (see src/CMakeLists.txt), so there is no per-target include block here.

target_compile_definitions(
    _culverin_c
    PRIVATE CULVERIN_DOCS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/docs/DOCS.md"
)

if(WIN32)
    target_link_libraries(_culverin_c PRIVATE Winmm.lib)
endif()

set_target_properties(
    _culverin_c
    PROPERTIES OUTPUT_NAME "_culverin_c" PREFIX ""
)

# MSVC-ABI runtime for our two archives. The property is honored only when the
# compiler targets the MSVC ABI; it is inert for GCC/Clang-on-ELF.
set_property(
    TARGET _culverin_c
    PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

# ==============================================================================
# 10. INSTALLATION
# ==============================================================================
install(
    FILES
        src/python/culverin/__init__.py
        src/python/culverin/_culverin.py
        src/python/culverin/__init__.pyi
        src/python/culverin/_culverin_c.pyi
        src/python/culverin/py.typed
    DESTINATION .
)
