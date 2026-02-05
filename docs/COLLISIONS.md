# Collision Filtering
Culverin uses a bitmask system to determine which objects hit each other.

*   **Category:** "What am I?" (A single bit).
*   **Mask:** "What do I hit?" (A collection of bits).

```python
# Constants for clarity
BIT_FLOOR  = 1 << 0
BIT_PLAYER = 1 << 1
BIT_ENEMY  = 1 << 2

# Player: I am a player, I hit the floor and enemies (but not other players)
world.create_body(..., category=BIT_PLAYER, mask=BIT_FLOOR | BIT_ENEMY)

# Trigger Zone: I hit everything
world.create_body(..., is_sensor=True, category=0xFFFF, mask=0xFFFF) # You can omit these because these are defaults
```

### The Trigger Lifecycle
Unlike other engines that only give you "Stay" events, Culverin provides the full lifecycle:
1.  **EVENT_ADDED:** First frame of contact.
2.  **EVENT_PERSISTED:** Every frame of contact after the first.
3.  **EVENT_REMOVED:** The frame objects separate.

### Sensors (Triggers)
To make a "Zone" that detects players but doesn't stop them physically:
1. Set `is_sensor=True`.
2. Set `motion=MOTION_STATIC` (usually).
3. Sensors will trigger `EVENT_ADDED` and `EVENT_REMOVED` but will apply **zero** force to the colliding body.