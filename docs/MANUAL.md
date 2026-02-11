# User Manual

## Shapes
| Constant | Data Format | Description |
| :--- | :--- | :--- |
| `SHAPE_BOX` | `(hx, hy, hz)` | Half-extents (from center to edge) |
| `SHAPE_SPHERE` | `radius` | Sphere radius |
| `SHAPE_CAPSULE` | `(radius, height)` | Total height (including caps) |
| `SHAPE_CONVEX_HULL` | `bytes` | Point cloud. Requires `create_convex_hull` |
| `SHAPE_MESH` | `bytes` | Triangle mesh. Requires `create_mesh_body` |

## Ragdolls
Ragdolls are built using a `Skeleton` and `RagdollSettings`.

1.  **Define Skeleton:** Add joints. Parents must be added before children. Call `finalize()`.
2.  **Define Settings:** Map shapes and constraints to joints.
3.  **Instantiate:** Create the physical ragdoll in the world.

## Surface Materials
You can tag bodies with `material_id` integers. These don't just change friction; they are passed through collision events.

```python
# Register materials (Id, Friction, Restitution)
world.register_material(ID_GRASS, 0.6, 0.1)
world.register_material(ID_METAL, 0.2, 0.5)

# When processing contacts:
for event in world.get_contact_events_ex():
    m1 = event['mat1']
    m2 = event['mat2']
    if m1 == ID_METAL or m2 == ID_METAL:
        play_clink_sound()
```

### Stabilizing
When spawning a ragdoll, limbs might overlap slightly. Call `settings.stabilize()` before creation to pre-calculate a valid pose adjustment, or let the physics engine resolve it (energetically) on the first frame.

## Visual Debugging
To see your physics shapes, use `get_debug_data()`. It returns raw vertex data for lines and triangles which can be fed into your rendering engine.

```python
lines, triangles = world.get_debug_data()
# Each vertex is 16 bytes: [float x, y, z, uint32 color]
```
### Possible Limitations:
* **Determinism:** While Jolt is deterministic, Python floating-point order of operations across threads is not guaranteed. Replays are deterministic only if single-threaded or strictly ordered.
* **Large Coordinates:** While Culverin uses Double Precision, rendering engines often use Float32. Large-world jitter may still appear visually if your renderer (not the physics) runs out of precision.