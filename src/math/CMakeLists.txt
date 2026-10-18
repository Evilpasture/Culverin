add_library(culverin_native_math OBJECT Math.cpp MathCore.h)
target_link_libraries(
    culverin_native_math
    PRIVATE culverin_native_common culverin_native_cpp
)
culverin_configure_compile_target(culverin_native_math)
