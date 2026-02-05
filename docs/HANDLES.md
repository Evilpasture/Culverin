# Generational Handles & Memory Access

Culverin uses a **Generational Handle** system to safely manage the lifecycle of physics objects. Unlike C pointers or standard Python object references, handles are safe to store indefinitely. If the underlying physics body is destroyed, the handle becomes "stale" rather than causing a crash or accessing the wrong object.

## The 64-Bit Integer
A handle in Culverin is a standard Python `int`, representing a 64-bit unsigned integer (`uint64`). It packs two pieces of information:

```text
64                               32                                0
|================================|=================================|
|         Generation ID          |           Slot Index            |
|           (32 bits)            |            (32 bits)            |
|================================|=================================|
```

1.  **Slot Index (Low 32 bits):** The specific index in the internal "Sparse" array where bookkeeping for this object lives.
2.  **Generation ID (High 32 bits):** A version counter. Every time a Slot is reused (after an object is destroyed and a new one created in its place), this counter increments.

### Python Bitwise Operations
You can deconstruct a handle using standard Python bitwise operators.

```python
handle = 12884901890  # Example handle

# Extract the Slot Index (Mask the lower 32 bits)
slot_index = handle & 0xFFFFFFFF

# Extract the Generation (Shift right by 32 bits)
generation = handle >> 32

print(f"Slot: {slot_index}, Gen: {generation}")
```

## Safety Mechanism (The "ABA" Problem)

Why do we need the Generation ID?

1.  You create a box. It gets **Slot 5**. You save `handle_A`.
2.  You destroy the box. **Slot 5** is now free.
3.  You create a sphere. It reuses **Slot 5**. You get `handle_B`.

If you try to use `handle_A` to move the box, the engine performs a check:
1.  Extract Slot (5).
2.  Check internal storage: "What is the current generation of Slot 5?"
3.  If the stored generation does not match the generation inside `handle_A`, the handle is **Stale**.
4.  The engine raises a `ValueError` or ignores the command, preventing you from accidentally moving the sphere when you meant to move the box.

## ⚠️ The "Dense" vs. "Sparse" Trap

**Crucial:** The `Slot Index` extracted from the handle is **NOT** the index you use to read from `world.positions` or `world.rotations`.

Culverin uses a **Sparse Set** architecture for performance:
*   **Slots (Sparse):** Stable indices. An object keeps its slot forever.
*   **Data (Dense):** Packed arrays. When an object is deleted, the last object in the array is moved to fill the gap to keep memory contiguous (fast CPU caching).

**Do not do this:**
```python
# ❌ WRONG: This reads the wrong memory address!
# The slot index does not match the position array index.
slot = handle & 0xFFFFFFFF
pos_x = world.positions[slot * 4] 
```

**Do this instead:**
```python
# ✅ CORRECT: Map the Handle to the current Dense Index
dense_index = world.get_index(handle)

if dense_index is not None:
    # Read from the packed shadow buffer
    x = world.positions[dense_index * 4 + 0]
    y = world.positions[dense_index * 4 + 1]
    z = world.positions[dense_index * 4 + 2]
```

### Why use `get_index()`?
Because `world.positions` is a contiguous block of memory, removing an object shifts the indices of other objects. `world.get_index(handle)` performs the O(1) lookup to find where your object currently lives in the memory block.

## Validation Helper
If you just want to know if a handle is valid without reading data, use `is_alive()`:

```python
if world.is_alive(my_handle):
    # Object exists
    pass
else:
    # Object was destroyed
    pass
```

## Summary Table

| Operation | Method / Math | Purpose |
| :--- | :--- | :--- |
| **Get Slot** | `handle & 0xFFFFFFFF` | Internal ID. Useful for hashing or debugging. |
| **Get Generation** | `handle >> 32` | Versioning. Useful for checking reuse count. |
| **Get Memory Index** | `world.get_index(handle)` | **Required** to read `world.positions`, `world.velocities`, etc. |
| **Check Validity** | `world.is_alive(handle)` | Checks if the object still exists. |