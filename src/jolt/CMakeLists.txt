# Jolt-specific runtime hooks are part of the native archive.
add_library(
    culverin_native_jolt
    OBJECT
        JoltLog.cpp
        JoltLog.hpp
        Jolt.hpp
        JoltConfig.hpp
        JoltUtils.hpp
)
target_link_libraries(
    culverin_native_jolt
    PRIVATE culverin_native_common culverin_native_cpp
)
culverin_configure_compile_target(culverin_native_jolt)
