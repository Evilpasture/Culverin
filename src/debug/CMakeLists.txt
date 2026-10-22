add_library(culverin_native_debug OBJECT DebugRender.cpp DebugRender.hpp)
target_link_libraries(
    culverin_native_debug
    PRIVATE culverin_native_common culverin_native_soft_bodies
)
culverin_configure_compile_target(culverin_native_debug)
