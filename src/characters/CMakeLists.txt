# Characters: Jolt physics and contact handling, no Python.
add_library(
    culverin_native_characters
    OBJECT
        Character.cpp
        Character.hpp
)
target_link_libraries(
    culverin_native_characters
    PRIVATE
        culverin_native_common
        culverin_native_cpp
        culverin_native_physics
)
culverin_configure_compile_target(culverin_native_characters)
