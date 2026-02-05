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

### Stabilizing
When spawning a ragdoll, limbs might overlap slightly. Call `settings.stabilize()` before creation to pre-calculate a valid pose adjustment, or let the physics engine resolve it (energetically) on the first frame.