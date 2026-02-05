# Vehicle System

Culverin exposes Jolt's native wheeled and tracked vehicle simulations. These are not simple raycast cars; they simulate engine RPM, transmission gearing, differentials, and suspension dynamics.

## Wheeled Vehicles
Constructing a car requires assembling a drivetrain.

```python
# 1. Define Wheels (Relative to Chassis center)
wheels = [
    {"pos": (-1, -0.5, 1.5), "radius": 0.4, "width": 0.3}, # FL
    {"pos": ( 1, -0.5, 1.5), "radius": 0.4, "width": 0.3}, # FR
    {"pos": (-1, -0.5,-1.5), "radius": 0.4, "width": 0.3}, # BL
    {"pos": ( 1, -0.5,-1.5), "radius": 0.4, "width": 0.3}, # BR
]

# 2. Define Engine & Transmission
engine = culverin.Engine(max_rpm=6000, max_torque=500)
trans = culverin.Automatic(gears=[3.0, 2.0, 1.5, 1.0, 0.8])

# 3. Create
car = world.create_vehicle(chassis_handle, wheels, drive="AWD", engine=engine, transmission=trans)
```

## Tracked Vehicles (Tanks)
Tanks use physical treads simulation.

```python
# Define Wheels (Road wheels + Idler + Sprocket)
wheels = [...] 

# Define Tracks
# indices: Which wheels from the list above are inside this track belt?
# driven_wheel: Which wheel applies the torque?
left_track = {"indices": [0, 1, 2, 3], "driven_wheel": 0}
right_track = {"indices": [4, 5, 6, 7], "driven_wheel": 4}

tank = world.create_tracked_vehicle(chassis, wheels, [left_track, right_track])

# Drive
# left_track_input, right_track_input, brake
tank.set_tank_input(1.0, -1.0, 0.0) # Pivot turn right
```