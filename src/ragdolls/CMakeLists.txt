# Ragdolls: skeleton, settings and the runtime instance. No Python.
add_library(
    culverin_native_ragdolls
    OBJECT
        Ragdoll.cpp
        Ragdoll.hpp
)
target_link_libraries(
    culverin_native_ragdolls
    PRIVATE
        culverin_native_common
        culverin_native_cpp
        culverin_native_physics
)
culverin_configure_compile_target(culverin_native_ragdolls)
