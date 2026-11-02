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
trans = culverin.Automatic(gears=5)   # gears is a count, not a list

# 3. Create
car = world.create_vehicle(chassis_handle, wheels, drive="AWD", engine=engine, transmission=trans)
```

### Wheel keys

`pos` is required; everything else has a default.

| Key | Default | Meaning |
|---|---|---|
| `pos` | — | Local position relative to the chassis centre |
| `radius` / `width` | `0.4` / `0.2` | Tire dimensions |
| `suspension` | `0.3` | Maximum suspension travel |
| `spring_freq` / `spring_damp` | `1.5` / `0.5` | Suspension spring |
| `brake_torque` / `handbrake_torque` | `1500.0` / `4000.0` | Braking strength |
| `steers` | unset | Whether this wheel steers |

`steers` exists because the default rule is "a wheel steers if its local `z` is
positive". That is fine for a chassis modelled along `+Z`, but a chassis modelled
along `-Z` has its front wheels at negative `z`, and would silently get a car
with no steering wheels at all. Setting `steers: True` on the front pair fixes
it; leaving the key out preserves the old behaviour.

### Engine and transmission are objects, not dicts

`engine=` and `transmission=` take the attribute-carrying objects above
(`culverin.Engine`, `culverin.Automatic`, `culverin.Manual`). Passing a plain
dict is accepted and silently ignored — the parsers read attributes.

The gearbox constructors take a **count**, not a list. `gears=5` selects the
first five of the built-in stack `[2.66, 1.78, 1.30, 1.00, 0.74, 0.50]`; assign
`trans.ratios = [...]` afterwards if you want your own.

| Constructor | Arguments |
|---|---|
| `Engine` | `max_torque=500.0`, `max_rpm=7000.0`, `min_rpm=1000.0`, `inertia=0.5` |
| `Transmission` | `gears=5`, `clutch_strength=2000.0`, `differential_ratio=3.42` |
| `Automatic` | all of the above, plus `shift_up_rpm=5000.0`, `shift_down_rpm=2000.0` |
| `Manual` | `gears=5`, `clutch_strength=5000.0`, `differential_ratio=3.42` |

`Automatic.mode` is `0` and `Manual.mode` is `1`. Be aware that Jolt only reads
`mode` when serialising a wheeled vehicle, so on the wheeled path the two drive
identically; the tracked path does use it.

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