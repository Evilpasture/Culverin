# Threading & Concurrency

Culverin is designed for Python 3.13t (Free-Threaded).

## The Global Interpreter Lock (GIL)
*   **world.step(dt):** RELEASES the GIL. While Jolt is crunching math, your Python logic thread can run rendering, AI, or networking code in parallel.

## Recommended Loop Pattern
The ideal game loop uses two threads:

1.  **Physics Thread:**
    ```python
    while running:
        # Blocks here, releasing GIL for Render thread
        world.step(1/60) 
        
        # Logic update (Python)
        game_logic.update()
    ```

2.  **Render Thread:**
    ```python
    while running:
        # Calculate interpolation alpha (0.0 to 1.0) based on time
        alpha = accumulator / fixed_dt
        
        # get_render_state creates a packed buffer of [Pos, Rot] 
        # interpolated between Previous and Current physics steps.
        # This creates buttery smooth visuals even at low physics tick rates.
        render_data = world.get_render_state(alpha)
        
        gpu.upload(render_data)
        gpu.draw()
    ```

## Safety Rules
1.  **Read-Only access is always safe.** Reading `world.positions` while `world.step()` is running is safe (you read the state *before* the step completes).
2.  **Mutation queues.** Calling `create_body` or `set_position` from a secondary thread while `step` is running will block the caller until the step finishes.