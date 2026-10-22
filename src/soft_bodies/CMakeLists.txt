# Soft bodies: the settings/topology builder plus the native shadow contract.
add_library(
    culverin_native_soft_bodies
    OBJECT
        SoftBodySettings.cpp
        SoftBodySettings.hpp
)
target_sources(
    culverin_native_soft_bodies
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SoftBodyCore.hpp>"
)
target_link_libraries(
    culverin_native_soft_bodies
    PRIVATE
        culverin_native_common
)
culverin_configure_compile_target(culverin_native_soft_bodies)
