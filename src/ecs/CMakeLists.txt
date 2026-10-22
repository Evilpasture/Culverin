# The ECS registry and its sparse-set storage. Deliberately free of <Python.h>
# and of Jolt: the one physics-aware operation (transform sync) lives in
# src/physics/EcsSync.cpp, the only translation unit that sees both halves.
add_library(
    culverin_native_ecs
    OBJECT
        EcsTypes.hpp
        SparseSet.hpp
        Registry.cpp
        Registry.hpp
)
target_link_libraries(
    culverin_native_ecs
    PRIVATE
        culverin_native_common
        culverin_native_cpp
)
culverin_configure_compile_target(culverin_native_ecs)
