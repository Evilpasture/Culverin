# Vehicle controllers: Jolt physics, no Python. A ship owns its own
# PhysicsStepListener, and a vehicle keeps a Jolt::VehicleConstraint that is
# its own step listener, so these run identically headless.
add_library(
    culverin_native_vehicles
    OBJECT
        Ship.cpp
        Ship.hpp
        Vehicle.cpp
        Vehicle.hpp
)
target_link_libraries(
    culverin_native_vehicles
    PRIVATE
        culverin_native_common
        culverin_native_cpp
        culverin_native_physics
)
culverin_configure_compile_target(culverin_native_vehicles)
