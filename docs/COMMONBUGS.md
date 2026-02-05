# Every engine has "rules of the road."

*   **The "Flying Through Walls" Fix:** If an object is moving too fast, enable CCD: `world.set_ccd(handle, True)`.
*   **Teleportation vs. Velocity:** Don't use `set_position` every frame to move an object (it breaks collision response). Use `set_linear_velocity` or `apply_impulse`.
*   **The NumPy View Error:** If a user gets a `BufferError: Cannot resize world while views are exported`, explain that they need to delete their local NumPy variable that wraps `world.positions` before the world can expand its capacity.
*   **The "Static Move" Crash:** If you create a body with `MOTION_STATIC` and then try to move it every frame using `set_position`, you will destroy performance and potentially cause internal Jolt errors. If an object is supposed to move (like an elevator or a moving platform), use `MOTION_KINEMATIC`.
*   **The "Frozen" Body:** Jolt automatically "puts to sleep" bodies that haven't moved in a while to save CPU. If you apply a force to a sleeping body, it might not move. Call `world.activate(handle)` to wake it up.
*   **The "Ghost" Collision:** Static bodies **do not** collide with other Static bodies. If you spawn two static walls overlapping, nothing will happen. This is a feature (for performance), not a bug.
*   **The "Jittery" Camera:** If your camera follows a physics object and looks "stuttery," it's because your Physics is 60Hz but your Monitor is 144Hz. Use `world.get_render_state(alpha)` to interpolate positions for the camera.