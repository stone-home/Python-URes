import pytest
from collections import OrderedDict
from ures.memory.manager import MemoryManager  # Using manager.py
from ures.memory.type import MemoryBlock       # Using type.py
from ures.memory.sequence import AbsSequence, MemorySequence # Using sequence.py


class TestMemoryManager:
    """Tests for the MemoryManager class using real components."""

    def test_initialization(self, empty_manager: MemoryManager):
        """Test that a new manager starts with an empty OrderedDict."""
        assert isinstance(empty_manager.memory, OrderedDict) # Check type
        assert len(empty_manager.memory) == 0 # Check emptiness

    def test_add_single(self, empty_manager: MemoryManager, memory_block_A: MemoryBlock):
        """Test adding a single memory block."""
        manager = empty_manager
        block_id = id(memory_block_A) # Manager uses id() as key

        assert not manager.exist(memory_block_A)
        manager.add(memory_block_A) #
        assert manager.exist(memory_block_A) #
        assert len(manager.memory) == 1
        assert manager.memory[block_id] == memory_block_A

    def test_add_multiple(self, empty_manager: MemoryManager, memory_block_A: MemoryBlock, memory_block_B: MemoryBlock):
        """Test adding multiple memory blocks maintains order (if using OrderedDict)."""
        manager = empty_manager
        id_A = id(memory_block_A)
        id_B = id(memory_block_B)

        manager.add(memory_block_A)
        manager.add(memory_block_B)

        assert len(manager.memory) == 2
        assert list(manager.memory.keys()) == [id_A, id_B] # Check order if important
        assert manager.memory[id_A] == memory_block_A
        assert manager.memory[id_B] == memory_block_B

    def test_add_duplicate(self, empty_manager: MemoryManager, memory_block_A: MemoryBlock):
        """Test adding the same block instance multiple times overwrites."""
        manager = empty_manager
        id_A = id(memory_block_A)

        manager.add(memory_block_A)
        assert len(manager.memory) == 1
        # Adding the same instance uses the same id(), overwriting the entry
        manager.add(memory_block_A)
        assert len(manager.memory) == 1
        assert manager.memory[id_A] == memory_block_A

    def test_exist(self, populated_manager: MemoryManager, memory_block_A: MemoryBlock, memory_block_B: MemoryBlock, memory_block_C: MemoryBlock):
        """Test the exist method for present and absent blocks."""
        assert populated_manager.exist(memory_block_A) # Should exist
        assert populated_manager.exist(memory_block_B) # Should exist
        assert not populated_manager.exist(memory_block_C) # C was not added

    def test_remove_existing(self, populated_manager: MemoryManager, memory_block_A: MemoryBlock, memory_block_B: MemoryBlock):
        """Test removing an existing block using the block object."""
        manager = populated_manager
        id_A = id(memory_block_A)

        assert manager.exist(memory_block_A)
        assert len(manager.memory) == 2

        removed_block = manager.remove(memory_block_A) # Calls remove_by_id(id(memory_block_A))

        assert removed_block == memory_block_A
        assert not manager.exist(memory_block_A)
        assert id_A not in manager.memory
        assert len(manager.memory) == 1
        # Ensure B is still there
        assert manager.exist(memory_block_B)

    def test_remove_non_existing(self, populated_manager: MemoryManager, memory_block_C: MemoryBlock):
        """Test removing a block that is not in the manager yields None."""
        manager = populated_manager
        initial_len = len(manager.memory)

        removed_block = manager.remove(memory_block_C) #

        assert removed_block is None # remove_by_id returns None if key not found
        assert len(manager.memory) == initial_len # Size should not change

    def test_remove_by_id_existing(self, populated_manager: MemoryManager, memory_block_B: MemoryBlock):
        """Test removing an existing block using its ID."""
        manager = populated_manager
        id_B = id(memory_block_B)

        assert manager.exist(memory_block_B)
        initial_len = len(manager.memory)

        removed_block = manager.remove_by_id(id_B) #

        assert removed_block == memory_block_B
        assert not manager.exist(memory_block_B)
        assert id_B not in manager.memory
        assert len(manager.memory) == initial_len - 1

    def test_remove_by_id_non_existing(self, populated_manager: MemoryManager):
        """Test removing by a non-existent ID yields None."""
        manager = populated_manager
        non_existent_id = 12345 # Assume this ID is not in the manager
        initial_len = len(manager.memory)

        removed_block = manager.remove_by_id(non_existent_id) #

        assert removed_block is None # Returns None if key not found
        assert len(manager.memory) == initial_len

    def test_gen_sequence_default(self, populated_manager: MemoryManager, memory_block_A: MemoryBlock, memory_block_B: MemoryBlock):
        """Test gen_sequence uses MemorySequence by default and deep copies blocks."""
        manager = populated_manager
        original_blocks = list(manager.memory.values())
        id_A_orig = id(memory_block_A)

        # Act
        sequence_obj = manager.gen_sequence() # Should use MemorySequence

        # Assert Type
        assert isinstance(sequence_obj, MemorySequence) # Check default type used

        # Assert Deep Copy
        sequence_blocks = sequence_obj._memory_blocks # Access internal list (implementation detail)
        assert sequence_blocks is not original_blocks # Should be a new list instance
        assert len(sequence_blocks) == len(original_blocks)

        # Modify original block A after sequence generation
        original_block_A_instance = manager.memory[id_A_orig]
        original_comment = original_block_A_instance.comment # Assuming comment property exists
        original_block_A_instance.comment = "MODIFIED AFTER COPY"

        # Find the corresponding block in the sequence list (assuming address is unique identifier)
        sequence_block_A_instance = next((b for b in sequence_blocks if b.address == memory_block_A.address), None)

        assert sequence_block_A_instance is not None # Should find the block
        assert sequence_block_A_instance is not original_block_A_instance # Should be different instance
        assert sequence_block_A_instance.comment == original_comment # Copied block's comment unchanged

    def test_gen_sequence_custom_processor(self, populated_manager: MemoryManager, memory_block_A: MemoryBlock):
        """Test gen_sequence correctly uses a provided custom sequence processor class."""
        manager = populated_manager
        original_blocks = list(manager.memory.values())

        # Define a simple mock custom processor inline or import
        class CustomSequenceProcessor(AbsSequence):
            initialized_with_blocks = None
            def __init__(self, memory_blocks): # Overriding init to capture blocks
                super().__init__(memory_blocks)
                # Store a copy for checking, prevent modification by storing ids/addresses if needed
                self.initialized_with_blocks = [b.address for b in memory_blocks] # Example: store addresses
            def preprocess(self): # Provide minimal implementation
                 yield from [] # pragma: no cover


        # Act: Pass the *class* of the custom processor
        sequence_obj = manager.gen_sequence(sequence_processor=CustomSequenceProcessor) #

        # Assert Type
        assert isinstance(sequence_obj, CustomSequenceProcessor)

        # Assert it was initialized with the correct (copied) blocks
        assert sequence_obj.initialized_with_blocks is not None
        assert len(sequence_obj.initialized_with_blocks) == len(original_blocks)
        # Check if the addresses match (or compare objects if __eq__ is reliable)
        assert set(sequence_obj.initialized_with_blocks) == set(b.address for b in original_blocks)
        # Check deep copy again (optional but good)
        assert sequence_obj._memory_blocks[0] is not original_blocks[0] # Compare first instance
