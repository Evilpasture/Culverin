import numpy as np
import pytest
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
    COMP_B = registry.register_component(16) # 16 bytes
    
    assert COMP_A == 0
    assert COMP_B == 1

def test_add_and_has_component(registry: culverin.Registry) -> None:
    """Test adding data to an entity and verifying existence."""
    COMP_VEC3 = registry.register_component(12) # 3x float32
    
    ent = registry.create()
    
    assert not registry.has(ent, COMP_VEC3)
    
    # Create 12 bytes of float32 data
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
    registry.add(ent, COMP_VEC3, data)
    
    assert registry.has(ent, COMP_VEC3)

def test_component_data_view(registry: culverin.Registry) -> None:
    """Test that added data appears correctly in the contiguous memoryview."""
    COMP_HEALTH = registry.register_component(4) # 1x float32
    
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
    COMP_ID = registry.register_component(4) # 1x uint32
    
    # Create 3 entities
    entities = [registry.create() for _ in range(3)]
    
    # Add data: [10, 20, 30]
    for i, ent in enumerate(entities):
        registry.add(ent, COMP_ID, np.array([i * 10 + 10], dtype=np.uint32).tobytes())
        
    view = registry.get_view(COMP_ID)
    data = np.frombuffer(view, dtype=np.uint32)
    
    assert len(data) == 3
    assert data[1] == 20 # Middle entity
    
    # Remove the middle entity's component
    registry.remove(entities[1], COMP_ID)
    
    # After swap-and-pop, the last element (30) should have moved into the hole (index 1)
    view = registry.get_view(COMP_ID)
    data = np.frombuffer(view, dtype=np.uint32)
    
    assert len(data) == 2
    assert data[0] == 10
    assert data[1] == 30 # Swap and pop verified
    
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
    COMP_POS = registry.register_component(12) # 3x float32
    
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
        registry.add(ent2, COMP_A, b"1234") # Provided 4 bytes, needs 8

def test_add_without_data(registry: culverin.Registry) -> None:
    """Test adding a 'Tag' component with no initial data (zero-initialized)."""
    COMP_TAG = registry.register_component(4)
    ent = registry.create()
    
    # None is explicitly permitted in the stubs
    registry.add(ent, COMP_TAG, None)
    
    assert registry.has(ent, COMP_TAG)
    
    # Verify it zero-initialized the 4 bytes
    view = registry.get_view(COMP_TAG)
    assert bytes(view) == b'\x00\x00\x00\x00'

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