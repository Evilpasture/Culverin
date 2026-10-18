add_subdirectory(parsing)

add_library(
    culverin_python_bindings
    OBJECT
        Getters.cpp
        docstrings/Docstrings.cpp
        docstrings/Docstrings.h
        BindingCommon.h
        FastBuild.h
        FastBuildCore.h
        Getters.h
        ModuleState.h
        ModuleParsers.hpp
        PythonDeclarations.h
        PythonSupport.h
)
culverin_configure_python_target(culverin_python_bindings)

# The root module is assembled here, but CPython adapters are compiled into
# separately named object targets that match their feature/domain directories.
python_add_library(_culverin_c MODULE Module.cpp)
culverin_configure_python_target(_culverin_c)
target_link_libraries(
    _culverin_c
    PRIVATE
        culverin_core
        culverin_native_jolt_rtti
        culverin_python_bindings
        culverin_python_parsing
        culverin_python_physics
        culverin_python_characters
        culverin_python_ecs
        culverin_python_math
        culverin_python_ragdolls
        culverin_python_soft_bodies
        culverin_python_vehicles
)

add_custom_command(
    TARGET _culverin_c
    POST_BUILD
    COMMAND
        ${CMAKE_COMMAND} -E echo
        "Extension built at: $<TARGET_FILE:_culverin_c>"
)

install(TARGETS _culverin_c DESTINATION .)

# Install a matching PDB beside the extension on Windows only. CMake selects
# -gcodeview (PDB) for any compiler targeting the MSVC ABI; OPTIONAL tolerates
# its absence elsewhere.
if(WIN32)
    install(FILES $<TARGET_PDB_FILE:_culverin_c> DESTINATION . OPTIONAL)
    message(STATUS "Culverin: Windows PDB installation configured.")
else()
    message(STATUS "Culverin: Non-PDB platform detected, skipping PDB installation.")
endif()
