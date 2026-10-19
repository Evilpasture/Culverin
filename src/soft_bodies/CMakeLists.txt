# SoftBodyCore currently defines a native data contract shared by physics and
# the Python adapter; it has no independent translation unit yet.
add_library(culverin_native_soft_bodies INTERFACE)
target_sources(
    culverin_native_soft_bodies
    INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/SoftBodyCore.hpp>"
)
target_link_libraries(culverin_native_soft_bodies INTERFACE culverin_native_common)
