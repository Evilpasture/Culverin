# Shared compile/link requirements for the Python adapter domains. The object
# targets below are assembled into one public CPython extension in bindings/.
add_library(culverin_python_native_support INTERFACE)
target_include_directories(
    culverin_python_native_support
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
)
target_link_libraries(
    culverin_python_native_support
    INTERFACE Python::Module culverin_native_common
)

function(culverin_configure_python_target target)
    target_link_libraries(${target} PRIVATE culverin_python_native_support)
    culverin_configure_compile_target(${target})

    # CPython and Jolt C-ABI aggregates intentionally rely on omitted trailing
    # fields being zero-initialized; keep -Wextra without flooding C++ builds.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
            ${target}
            PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>"
                "$<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-designated-field-initializers>"
        )
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            ${target}
            PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>"
        )
    endif()
endfunction()

# Each domain directory owns its object target; bindings/ composes them into
# the single extension module after all domain targets have been declared.
add_subdirectory(characters)
add_subdirectory(ecs)
add_subdirectory(math)
add_subdirectory(ragdolls)
add_subdirectory(soft_bodies)
add_subdirectory(vehicles)
add_subdirectory(physics)
add_subdirectory(bindings)
