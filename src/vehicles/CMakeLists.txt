# Vehicle controllers: Jolt physics, no Python. A ship owns its own
# PhysicsStepListener, so these run identically headless.
add_library(
    culverin_native_vehicles
    OBJECT
        Ship.cpp
        Ship.hpp
)
target_link_libraries(
    culverin_native_vehicles
    PRIVATE
        culverin_native_common
        culverin_native_cpp
)
culverin_configure_compile_target(culverin_native_vehicles)
