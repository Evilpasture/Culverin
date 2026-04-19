import numpy as np
import pytest
import struct
import culverin


@pytest.fixture
def registry() -> culverin.Registry:
    """Provides a fresh ECS registry for each test."""
    return culverin.Registry()


def test_entity_creation_and_destruction(registry: culverin.Registry) -> None:
    """Test that entities are created with unique generational handles."""
    e1 = registry.create()
    e2 = registry.create()

    assert e1 != e2, "Entity handles must be unique"

    # Extract index and generation from the 64-bit handle
    # handle = (generation << 32) | index
    idx1 = e1 & 0xFFFFFFFF
    idx2 = e2 & 0xFFFFFFFF

    assert idx1 != idx2, "Entity indices must be distinct"

    registry.destroy(e1)

    # Re-creating an entity should recycle the index but increment the generation
    e3 = registry.create()
    idx3 = e3 & 0xFFFFFFFF
    gen3 = e3 >> 32

    assert idx1 == idx3, "Destroyed indices should be recycled"
    assert gen3 > 1, "Recycled entities must have a higher generation"


def test_component_registration(registry: culverin.Registry) -> None:
    """Test that components can be registered and yield unique IDs."""
    COMP_A = registry.register_component(4)  # 4 bytes
    COMP_B = registry.register_component(16)  # 16 bytes

    assert COMP_A == 0
    assert COMP_B == 1


def test_add_and_has_component(registry: culverin.Registry) -> None:
    """Test adding data to an entity and verifying existence."""
    COMP_VEC3 = registry.register_component(12)  # 3x float32

    ent = registry.create()

    assert not registry.has(ent, COMP_VEC3)

    # Create 12 bytes of float32 data
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    registry.add(ent, COMP_VEC3, data)

    assert registry.has(ent, COMP_VEC3)


def test_component_data_view(registry: culverin.Registry) -> None:
    """Test that added data appears correctly in the contiguous memoryview."""
    COMP_HEALTH = registry.register_component(4)  # 1x float32

    e1 = registry.create()
    e2 = registry.create()

    registry.add(e1, COMP_HEALTH, np.array([100.0], dtype=np.float32).tobytes())
    registry.add(e2, COMP_HEALTH, np.array([85.5], dtype=np.float32).tobytes())

    # Retrieve the dense buffer
    view = registry.get_view(COMP_HEALTH)
    health_array = np.frombuffer(view, dtype=np.float32)

    assert len(health_array) == 2
    assert health_array[0] == 100.0
    assert health_array[1] == 85.5


def test_swap_and_pop_removal(registry: culverin.Registry) -> None:
    """Test the core ECS 'swap-and-pop' logic when a component is removed."""
    COMP_ID = registry.register_component(4)  # 1x uint32

    # Create 3 entities
    entities = [registry.create() for _ in range(3)]

    # Add data: [10, 20, 30]
    for i, ent in enumerate(entities):
        registry.add(ent, COMP_ID, np.array([i * 10 + 10], dtype=np.uint32).tobytes())

    view = registry.get_view(COMP_ID)
    data = np.frombuffer(view, dtype=np.uint32)

    assert len(data) == 3
    assert data[1] == 20  # Middle entity

    # Remove the middle entity's component
    registry.remove(entities[1], COMP_ID)

    # After swap-and-pop, the last element (30) should have moved into the hole (index 1)
    view = registry.get_view(COMP_ID)
    data = np.frombuffer(view, dtype=np.uint32)

    assert len(data) == 2
    assert data[0] == 10
    assert data[1] == 30  # Swap and pop verified

    # Verify handles map array updated
    handles = np.frombuffer(registry.get_entities(COMP_ID), dtype=np.uint64)
    assert handles[0] == entities[0]
    assert handles[1] == entities[2]


def test_destroy_entity_clears_components(registry: culverin.Registry) -> None:
    """Test that destroying an entity automatically strips its components."""
    COMP_A = registry.register_component(4)
    COMP_B = registry.register_component(4)

    ent = registry.create()
    registry.add(ent, COMP_A, b"1234")
    registry.add(ent, COMP_B, b"5678")

    assert len(registry.get_view(COMP_A)) == 4
    assert len(registry.get_view(COMP_B)) == 4

    registry.destroy(ent)

    assert len(registry.get_view(COMP_A)) == 0
    assert len(registry.get_view(COMP_B)) == 0


def test_zero_copy_mutation(registry: culverin.Registry) -> None:
    """Test that modifying the memoryview updates the underlying C-buffer immediately."""
    COMP_POS = registry.register_component(12)  # 3x float32

    ent = registry.create()
    registry.add(ent, COMP_POS, np.array([0.0, 0.0, 0.0], dtype=np.float32).tobytes())

    # Get view and cast to numpy array
    view = registry.get_view(COMP_POS)
    pos_array = np.frombuffer(view, dtype=np.float32).reshape(-1, 3)

    # Mutate via numpy (This modifies C memory directly)
    pos_array[0, 1] += 5.0

    # Request view again to verify changes persisted in the engine
    new_view = registry.get_view(COMP_POS)
    new_pos_array = np.frombuffer(new_view, dtype=np.float32).reshape(-1, 3)

    assert new_pos_array[0, 1] == 5.0


def test_empty_buffers(registry: culverin.Registry) -> None:
    """Ensure querying empty components doesn't crash and returns empty views."""
    COMP_EMPTY = registry.register_component(4)

    view = registry.get_view(COMP_EMPTY)
    entities = registry.get_entities(COMP_EMPTY)

    assert len(view) == 0
    assert len(entities) == 0


def test_error_handling(registry: culverin.Registry) -> None:
    """Test that the C-layer properly traps invalid inputs."""
    COMP_A = registry.register_component(8)
    ent = registry.create()

    # Invalid Component ID
    with pytest.raises(ValueError, match="Invalid component ID"):
        registry.add(ent, 999, b"12345678")

    # Stale Entity
    registry.destroy(ent)
    with pytest.raises(ValueError, match="Invalid or stale entity"):
        registry.add(ent, COMP_A, b"12345678")

    # Buffer Size Mismatch
    ent2 = registry.create()
    with pytest.raises(ValueError, match="Data size mismatch"):
        registry.add(ent2, COMP_A, b"1234")  # Provided 4 bytes, needs 8


def test_add_without_data(registry: culverin.Registry) -> None:
    """Test adding a 'Tag' component with no initial data (zero-initialized)."""
    COMP_TAG = registry.register_component(4)
    ent = registry.create()

    # None is explicitly permitted in the stubs
    registry.add(ent, COMP_TAG, None)

    assert registry.has(ent, COMP_TAG)

    # Verify it zero-initialized the 4 bytes
    view = registry.get_view(COMP_TAG)
    assert bytes(view) == b"\x00\x00\x00\x00"


def test_capacity_expansion(registry: culverin.Registry) -> None:
    """Force the registry to resize its internal buffers."""
    COMP_INT = registry.register_component(4)

    # Create 100 entities and add components
    # (Initial dense_capacity is 64, so this forces a realloc)
    for i in range(100):
        ent = registry.create()
        registry.add(ent, COMP_INT, np.array([i], dtype=np.uint32).tobytes())

    view = registry.get_view(COMP_INT)
    data = np.frombuffer(view, dtype=np.uint32)

    assert len(data) == 100
    assert data[99] == 99


def test_physics_to_ecs_sync_workflow(registry: culverin.Registry) -> None:
    """Demonstrates a real-world usage: Syncing Jolt results to ECS components."""
    # 1. Setup ECS components
    # Transform: 12 bytes (x,y,z float32)
    # Metadata: 8 bytes (uint64 Jolt Handle)
    COMP_TRANS = registry.register_component(12)
    COMP_JOLT = registry.register_component(8)

    # 2. Create entities and "mock" Jolt handles
    for i in range(5):
        ent = registry.create()
        registry.add(ent, COMP_TRANS, np.array([0, i, 0], dtype=np.float32).tobytes())
        registry.add(ent, COMP_JOLT, np.array([i + 1000], dtype=np.uint64).tobytes())

    # 3. The Sync Pass (What you would do in your game loop)
    # Get views of the ECS storage
    transforms = np.frombuffer(registry.get_view(COMP_TRANS), dtype=np.float32).reshape(-1, 3)
    np.frombuffer(registry.get_entities(COMP_JOLT), dtype=np.uint64)

    # In a real game, you'd pull from world.positions[jolt_indices]
    # Here we just perform a bulk operation on the ECS memory
    transforms[:, 0] += 10.0  # Move all entities 10 units on X

    # 4. Verify
    view_verify = np.frombuffer(registry.get_view(COMP_TRANS), dtype=np.float32).reshape(-1, 3)
    assert view_verify[0, 0] == 10.0
    assert view_verify[4, 0] == 10.0


def test_real_physics_integration(registry: culverin.Registry) -> None:
    """
    Test a full frame lifecycle using the optimized C-native sync method.
    1. Update Physics.
    2. Bulk Sync Physics results to ECS components in C.
    """
    import culverin

    # Fresh world with gravity
    world = culverin.PhysicsWorld(settings={"gravity": (0, -10, 0)})

    # 1. Register ECS Components
    COMP_PHYSICS = registry.register_component(8)  # uint64 handle
    COMP_TRANSFORM = registry.register_component(12)  # 3x float32

    # 2. Spawn 10 falling boxes
    # Spread them out so they don't collide
    for i in range(10):
        ent = registry.create()
        handle = world.create_body(pos=(i * 2.0, 10.0, 0.0), motion=culverin.MOTION_DYNAMIC)

        # Add components to ECS
        registry.add(ent, COMP_PHYSICS, np.array([handle], dtype=np.uint64).tobytes())
        registry.add(
            ent, COMP_TRANSFORM, np.array([i * 2.0, 10.0, 0.0], dtype=np.float32).tobytes()
        )

    # Flush creation
    world.step(0)

    # 3. Simulate 10 frames of gravity
    for _ in range(10):
        world.step(1 / 60)

    # 4. OPTIMIZED SYNC PASS
    # This single C call replaces the entire NumPy mapping and manual index-lookup loop.
    # It performs handle validation and precision casting (double -> float) internally.
    registry.sync_from_world(world, COMP_PHYSICS, COMP_TRANSFORM)

    # 5. VERIFICATION
    # Access the ECS buffer to verify the sync worked
    ecs_transforms = np.frombuffer(registry.get_view(COMP_TRANSFORM), dtype=np.float32).reshape(
        -1, 3
    )

    for i in range(10):
        current_y = ecs_transforms[i, 1]
        # Verify the entity has physically moved (fallen) in the ECS storage
        assert current_y < 10.0, f"Entity {i} failed to fall in ECS storage. Y={current_y}"

    print(f"\n[Native ECS Sync] Success: Bulk synced {len(ecs_transforms)} entities.")


def test_multithreaded_ecs_hammer(registry: culverin.Registry) -> None:
    """
    Stresses the ECS under heavy multithreaded contention:
    - Thread 1: Simulates the Physics World.
    - Thread 2: Constantly triggers C-native sync_from_world.
    - Thread 3: Hammers entity creation and destruction (forces reallocs).
    - Thread 4: Mutates component data via NumPy (releases GIL).
    """
    import threading
    import time

    import culverin

    world = culverin.PhysicsWorld()
    COMP_PHYS = registry.register_component(8)
    COMP_POS = registry.register_component(12)

    # Pre-populate some bodies
    target_entity = registry.create()
    phys_handle = world.create_body(pos=(0, 0, 0))
    registry.add(target_entity, COMP_PHYS, np.array([phys_handle], dtype=np.uint64).tobytes())
    registry.add(target_entity, COMP_POS, None)

    stop_event = threading.Event()

    def physics_thread() -> None:
        while not stop_event.is_set():
            world.step(1 / 120)  # High frequency step

    def sync_thread() -> None:
        while not stop_event.is_set():
            # This is the C-native bottleneck we are testing
            registry.sync_from_world(world, COMP_PHYS, COMP_POS)

    def mutation_thread() -> None:
        while not stop_event.is_set():
            view = registry.get_view(COMP_POS)
            if len(view) > 0:
                arr = np.frombuffer(view, dtype=np.float32)
                arr += 1.0  # True parallel math if GIL is disabled

    def spawn_kill_thread() -> None:
        while not stop_event.is_set():
            # Force the Registry to realloc by crossing the 1024 entity boundary
            temp_ents = [registry.create() for _ in range(50)]
            for e in temp_ents:
                registry.add(e, COMP_POS, None)
            for e in temp_ents:
                registry.destroy(e)

    threads = [
        threading.Thread(target=physics_thread),
        threading.Thread(target=sync_thread),
        threading.Thread(target=mutation_thread),
        threading.Thread(target=spawn_kill_thread),
    ]

    for t in threads:
        t.start()
    time.sleep(1.0)
    stop_event.set()
    for t in threads:
        t.join()

    # 2. Check the real handle, not the index 1
    assert registry.is_alive(target_entity), "Target entity should have survived the chaos"

    # 3. Final Integrity Check: Entity index 0 in the Transform view
    # should have been updated by the sync_thread or mutation_thread
    view = registry.get_view(COMP_POS)
    data = np.frombuffer(view, dtype=np.float32)
    assert len(data) >= 3  # At least our target entity
    print(f"\n[Multithreaded ECS] Hammer Success. Entity survived. Data: {data[:3]}")


def test_proxy_len_support(registry: culverin.Registry) -> None:
    """Verify that BufferProxy correctly implements the Sequence protocol (len)."""
    COMP_A = registry.register_component(4)
    for _ in range(5):
        e = registry.create()
        registry.add(e, COMP_A, b"\x00\x00\x00\x00")

    view = registry.get_view(COMP_A)
    # This checks the sq_length slot fix
    assert len(view) == 5 * 4  # 5 entities * 4 bytes

    entities = registry.get_entities(COMP_A)
    assert len(entities) == 5  # 5 uint64 handles


def test_proxy_resizing_guard(registry: culverin.Registry) -> None:
    """
    Ensure the Registry raises BufferError if a realloc is attempted
    while Python holds a memoryview.
    """
    COMP_A = registry.register_component(4)
    # The default dense_capacity in our C code is 64.
    # We fill exactly 64 slots.
    for _ in range(64):
        e = registry.create()
        registry.add(e, COMP_A, b"\x00\x00\x00\x00")

    # Export a view. This increments registry->view_export_count.
    view = registry.get_view(COMP_A)

    # Attempting to add the 65th entity triggers SparseSet_EnsureDenseCapacity.
    # The C code should detect view_export_count > 0 and raise BufferError.
    with pytest.raises(BufferError, match="Cannot resize ECS component while a memoryview is held"):
        e_fail = registry.create()
        registry.add(e_fail, COMP_A, b"\x01\x01\x01\x01")

    # Once the view is deleted and collected, we should be able to resize again.
    del view
    import gc

    gc.collect()  # Force cleanup of the BufferProxy object

    e_success = registry.create()
    # Now it should work because exported_views is back to 0
    registry.add(e_success, COMP_A, b"\x02\x02\x02\x02")
    assert registry.has(e_success, COMP_A)


def test_proxy_readonly_enforcement(registry: culverin.Registry) -> None:
    """Verify that entities are read-only while component data is writable."""
    COMP_A = registry.register_component(4)
    e = registry.create()
    registry.add(e, COMP_A, b"\x00\x00\x00\x00")

    # 1. Component Data should be WRITABLE
    data_view = registry.get_view(COMP_A)
    data_arr = np.frombuffer(data_view, dtype=np.uint8)
    data_arr[0] = 255  # Should succeed
    assert data_arr[0] == 255

    # 2. Entity Handles should be READ-ONLY
    ents_view = registry.get_entities(COMP_A)
    ents_arr = np.frombuffer(ents_view, dtype=np.uint64)

    with pytest.raises(ValueError, match="read-only"):
        ents_arr[0] = 12345  # NumPy raises ValueError when writing to a RO buffer


def test_proxy_ownership_persistence(registry: culverin.Registry) -> None:
    """Verify that a BufferProxy keeps the Registry alive (refcounting)."""
    COMP_A = registry.register_component(4)
    e = registry.create()
    registry.add(e, COMP_A, b"\xde\xad\xbe\xef")

    view = registry.get_view(COMP_A)

    # Delete the local reference to the registry
    # The BufferProxyObject->owner still holds a reference in C
    del registry
    import gc

    gc.collect()

    # The view must still be valid and accessible
    assert bytes(view) == b"\xde\xad\xbe\xef"


def test_proxy_empty_component(registry: culverin.Registry) -> None:
    """Ensure proxies for empty components handle null pointers gracefully."""
    COMP_EMPTY = registry.register_component(100)

    view = registry.get_view(COMP_EMPTY)
    assert len(view) == 0

    # Casting empty view to numpy should yield an empty array, not a crash
    arr = np.frombuffer(view, dtype=np.float32)
    assert arr.size == 0

def test_registry_clear(registry: culverin.Registry) -> None:
    """Test that clearing the registry wipes all entities and data instantly."""
    COMP_A = registry.register_component(4)
    COMP_B = registry.register_component(8)

    # Spawn 100 entities
    entities = [registry.create() for _ in range(100)]
    for e in entities:
        registry.add(e, COMP_A, b"1234")
        registry.add(e, COMP_B, b"56789012")

    assert registry.get_active_count() == 100
    assert registry.get_component_count(COMP_A) == 100

    # The wipe
    registry.clear()

    # Verify Registry stats are reset
    assert registry.get_active_count() == 0
    assert registry.get_component_count(COMP_A) == 0
    assert registry.get_component_count(COMP_B) == 0

    # Verify old handles are entirely dead (generation bumped)
    for e in entities:
        assert not registry.is_alive(e)


def test_registry_clear_guard(registry: culverin.Registry) -> None:
    """Ensure clearing the registry is blocked if memoryviews are active."""
    COMP_A = registry.register_component(4)
    e = registry.create()
    registry.add(e, COMP_A, b"1234")

    # Export view
    view = registry.get_view(COMP_A)

    with pytest.raises(BufferError, match="Cannot clear registry"):
        registry.clear()

    del view
    import gc
    gc.collect()

    # Now it should work
    registry.clear()
    assert registry.get_active_count() == 0


def test_single_component_get(registry: culverin.Registry) -> None:
    """Test retrieving raw bytes for a single entity's component."""
    COMP_ID = registry.register_component(4)
    ent = registry.create()
    registry.add(ent, COMP_ID, b"\xAA\xBB\xCC\xDD")

    data = registry.get(ent, COMP_ID)
    assert isinstance(data, bytes)
    assert data == b"\xAA\xBB\xCC\xDD"

    # Test invalid cases
    ent2 = registry.create()
    assert registry.get(ent2, COMP_ID) is None  # Entity exists, but lacks component
    
    registry.destroy(ent)
    assert registry.get(ent, COMP_ID) is None  # Entity is dead


def test_stats_getters(registry: culverin.Registry) -> None:
    """Test that ECS statistics track correctly."""
    assert registry.get_active_count() == 0

    COMP_ID = registry.register_component(4)
    assert registry.get_component_count(COMP_ID) == 0

    ent1 = registry.create()
    ent2 = registry.create()

    assert registry.get_active_count() == 2

    registry.add(ent1, COMP_ID, b"1234")
    assert registry.get_component_count(COMP_ID) == 1

    registry.add(ent2, COMP_ID, b"5678")
    assert registry.get_component_count(COMP_ID) == 2

    registry.remove(ent1, COMP_ID)
    assert registry.get_component_count(COMP_ID) == 1

    registry.destroy(ent2)
    assert registry.get_active_count() == 1  # ent1 is still alive here!
    
    registry.destroy(ent1)                   # Kill ent1
    assert registry.get_active_count() == 0  # NOW it is 0
    assert registry.get_component_count(COMP_ID) == 0


def test_full_transform_sync(registry: culverin.Registry) -> None:
    """Test syncing both Position AND Rotation from the physics engine."""
    world = culverin.PhysicsWorld()

    COMP_PHYS = registry.register_component(8)    # uint64 handle
    COMP_POS = registry.register_component(12)    # 3x float32
    COMP_ROT = registry.register_component(16)    # 4x float32

    ent = registry.create()
    phys_handle = world.create_body(pos=(1.0, 2.0, 3.0), rot=(0.0, 1.0, 0.0, 0.0))
    
    registry.add(ent, COMP_PHYS, np.array([phys_handle], dtype=np.uint64).tobytes())
    registry.add(ent, COMP_POS, None)
    registry.add(ent, COMP_ROT, None)

    # Sync Both
    registry.sync_from_world(world, COMP_PHYS, COMP_POS, COMP_ROT)

    # Verify Position
    assert (pos_data := registry.get(ent, COMP_POS))
    pos_floats = struct.unpack("3f", pos_data)
    assert pos_floats[0] == 1.0
    assert pos_floats[1] == 2.0
    assert pos_floats[2] == 3.0

    # Verify Rotation
    assert(rot_data := registry.get(ent, COMP_ROT))
    rot_floats = struct.unpack("4f", rot_data)
    assert rot_floats[0] == 0.0
    assert rot_floats[1] == 1.0
    assert rot_floats[2] == 0.0
    assert rot_floats[3] == 0.0


def test_partial_transform_sync(registry: culverin.Registry) -> None:
    """Test syncing ONLY position, skipping rotation by passing -1."""
    world = culverin.PhysicsWorld()

    COMP_PHYS = registry.register_component(8)
    COMP_POS = registry.register_component(12)

    ent = registry.create()
    phys_handle = world.create_body(pos=(10.0, 20.0, 30.0))
    
    registry.add(ent, COMP_PHYS, np.array([phys_handle], dtype=np.uint64).tobytes())
    registry.add(ent, COMP_POS, None)

    # Sync using -1 to ignore rotation
    registry.sync_from_world(world, COMP_PHYS, COMP_POS, -1)

    assert (pos_data := registry.get(ent, COMP_POS))
    pos_floats = struct.unpack("3f", pos_data)
    assert pos_floats[0] == 10.0