import unittest
import pytest
from ures.data_structure.bi_directional_links import BiDirection, NonCircularBiDirection


class TestBiDirection(unittest.TestCase):

    def test_node_initialization(self):
        """Test that a new node initializes correctly with self-referential pointers."""
        node = BiDirection("Test")
        self.assertEqual(node.value, "Test")
        self.assertIs(node.prev, node)
        self.assertIs(node.next, node)

    def test_insert_after(self):
        """Test that insert_after correctly links nodes in the forward direction."""
        node1 = BiDirection("A")
        node2 = BiDirection("B")
        node1.insert_after(node2)
        self.assertIs(node1.next, node2)
        self.assertIs(node2.prev, node1)
        # Ensure circular linking: node2.next should be node1's old next (which is node1)
        self.assertIs(node2.next, node1)

    def test_insert_before(self):
        """Test that insert_before correctly links nodes in the backward direction."""
        node1 = BiDirection("A")
        node2 = BiDirection("B")
        node1.insert_before(node2)
        self.assertIs(node1.prev, node2)
        self.assertIs(node2.next, node1)
        # Ensure circular linking: node2.prev should be node1 (initially node1.prev was node1)
        self.assertIs(node2.prev, node1)

    def test_remove(self):
        """Test that a node is properly removed from the chain."""
        node1 = BiDirection("A")
        node2 = BiDirection("B")
        node3 = BiDirection("C")
        node1.insert_after(node2)
        node2.insert_after(node3)
        # Remove node2
        node2.remove()
        # After removal, node1's next should be node3 and node3's prev should be node1.
        self.assertIs(node1.next, node3)
        self.assertIs(node3.prev, node1)
        # Also, node2's pointers should point to itself.
        self.assertIs(node2.next, node2)
        self.assertIs(node2.prev, node2)

    def test_search_found(self):
        """Test searching for an existing value in the chain."""
        node1 = BiDirection("A")
        node2 = BiDirection("B")
        node3 = BiDirection("C")
        node1.insert_after(node2)
        node2.insert_after(node3)
        found_node = node1.search("C")
        self.assertIsNotNone(found_node)
        self.assertEqual(found_node.value, "C")

    def test_search_not_found(self):
        """Test searching for a value that does not exist returns None."""
        node1 = BiDirection("A")
        node2 = BiDirection("B")
        node1.insert_after(node2)
        result = node1.search("Z")
        self.assertIsNone(result)

    def test_equality(self):
        """Test that the equality method compares nodes based on their unique IDs."""
        node1 = BiDirection("A")
        node2 = BiDirection("A")
        self.assertTrue(node1 == node1)
        self.assertFalse(node1 == node2)

    def test_str_and_repr(self):
        """Test the __str__ and __repr__ methods for expected output."""
        node = BiDirection("Hello")
        self.assertEqual(str(node), "Hello")
        self.assertEqual(repr(node), "BiDirection(Hello)")


# --------------------- Tests for NonCircularBiDirection Node ---------------------


class TestNodeInitialization:
    """Tests for node creation and initial state."""

    def test_node_creation_value(self):
        """Test if a node is created with the correct value."""
        node = NonCircularBiDirection(10)
        assert node.value == 10

    def test_node_creation_initial_links(self):
        """Test if a new node has None for prev and next links."""
        node = NonCircularBiDirection("test")
        assert node.prev is None
        assert node.next is None


class TestNodeInsertion:
    """Tests for node insertion methods (insert_after, insert_before)."""

    def test_insert_after_single_node(self):
        """Test inserting a node after a standalone node."""
        node1 = NonCircularBiDirection(1)
        node2 = NonCircularBiDirection(2)
        node1.insert_after(node2)
        assert node1.next is node2
        assert node2.prev is node1
        assert node1.prev is None
        assert node2.next is None

    def test_insert_after_in_middle(self):
        """Test inserting a node between two existing nodes using insert_after."""
        node1 = NonCircularBiDirection(1)
        node3 = NonCircularBiDirection(3)
        # Manually link node1 and node3 first
        node1._next = node3
        node3._prev = node1

        node2 = NonCircularBiDirection(2)
        node1.insert_after(node2)  # Insert 2 after 1

        assert node1.next is node2
        assert node2.prev is node1
        assert node2.next is node3
        assert node3.prev is node2
        assert node1.prev is None
        assert node3.next is None

    def test_insert_before_single_node(self):
        """Test inserting a node before a standalone node."""
        node1 = NonCircularBiDirection(1)
        node2 = NonCircularBiDirection(2)
        node2.insert_before(node1)  # Insert 1 before 2
        assert node1.next is node2
        assert node2.prev is node1
        assert node1.prev is None
        assert node2.next is None

    def test_insert_before_in_middle(self):
        """Test inserting a node between two existing nodes using insert_before."""
        node1 = NonCircularBiDirection(1)
        node3 = NonCircularBiDirection(3)
        # Manually link node1 and node3 first
        node1._next = node3
        node3._prev = node1

        node2 = NonCircularBiDirection(2)
        node3.insert_before(node2)  # Insert 2 before 3

        assert node1.next is node2
        assert node2.prev is node1
        assert node2.next is node3
        assert node3.prev is node2
        assert node1.prev is None
        assert node3.next is None

    def test_insert_after_updates_original_next_prev(self):
        """Verify that the original next node's prev pointer is updated."""
        node1 = NonCircularBiDirection(1)
        node3 = NonCircularBiDirection(3)
        node1.insert_after(node3)  # 1 -> 3

        node2 = NonCircularBiDirection(2)
        node1.insert_after(node2)  # 1 -> 2 -> 3

        assert node3.prev is node2  # Check original next node's prev pointer

    def test_insert_before_updates_original_prev_next(self):
        """Verify that the original previous node's next pointer is updated."""
        node1 = NonCircularBiDirection(1)
        node3 = NonCircularBiDirection(3)
        node1.insert_after(node3)  # 1 -> 3

        node2 = NonCircularBiDirection(2)
        node3.insert_before(node2)  # 1 -> 2 -> 3

        assert node1.next is node2  # Check original previous node's next pointer


class TestNodeRemoval:
    """Tests for node removal."""

    @pytest.fixture
    def three_node_list(self):
        """Fixture to create a list of three linked nodes: 1 <-> 2 <-> 3."""
        node1 = NonCircularBiDirection(1)
        node2 = NonCircularBiDirection(2)
        node3 = NonCircularBiDirection(3)
        node1.insert_after(node2)
        node2.insert_after(node3)
        # node1 <-> node2 <-> node3
        return node1, node2, node3

    def test_remove_single_node(self):
        """Test removing the only node in a list."""
        node = NonCircularBiDirection(100)
        node.remove()
        assert node.prev is None
        assert node.next is None
        # Note: The node object still exists, but its links are severed.

    def test_remove_first_node(self, three_node_list):
        """Test removing the first node from a list."""
        node1, node2, node3 = three_node_list
        head = node1  # Keep track of the potential new head

        node1.remove()

        # Check removed node's links
        assert node1.prev is None
        assert node1.next is None

        # Check remaining list structure
        assert node2.prev is None  # node2 is now the head
        assert node2.next is node3
        assert node3.prev is node2
        assert node3.next is None

    def test_remove_last_node(self, three_node_list):
        """Test removing the last node from a list."""
        node1, node2, node3 = three_node_list

        node3.remove()

        # Check removed node's links
        assert node3.prev is None
        assert node3.next is None

        # Check remaining list structure
        assert node1.next is node2
        assert node2.prev is node1
        assert node2.next is None  # node2 is now the tail

    def test_remove_middle_node(self, three_node_list):
        """Test removing a node from the middle of a list."""
        node1, node2, node3 = three_node_list

        node2.remove()

        # Check removed node's links
        assert node2.prev is None
        assert node2.next is None

        # Check remaining list structure
        assert node1.next is node3
        assert node3.prev is node1
        assert node1.prev is None
        assert node3.next is None


class TestNodeSearch:
    """Tests for the search method."""

    @pytest.fixture
    def three_node_list(self):
        """Fixture to create a list of three linked nodes: 1 <-> 2 <-> 3."""
        node1 = NonCircularBiDirection(1)
        node2 = NonCircularBiDirection(2)
        node3 = NonCircularBiDirection(3)
        node1.insert_after(node2)
        node2.insert_after(node3)
        return node1, node2, node3

    def test_search_finds_value_from_start(self, three_node_list):
        """Test finding a value present later in the list, starting search from head."""
        node1, node2, node3 = three_node_list
        found_node = node1.search(3)
        assert found_node is node3
        assert found_node.value == 3

    def test_search_finds_value_from_middle(self, three_node_list):
        """Test finding a value present later in the list, starting search from middle."""
        node1, node2, node3 = three_node_list
        found_node = node2.search(3)
        assert found_node is node3
        assert found_node.value == 3

    def test_search_finds_current_node_value(self, three_node_list):
        """Test finding the value of the node where the search starts."""
        node1, node2, node3 = three_node_list
        found_node = node2.search(2)
        assert found_node is node2
        assert found_node.value == 2

    def test_search_value_not_found(self, three_node_list):
        """Test searching for a value not present in the list."""
        node1, node2, node3 = three_node_list
        found_node = node1.search(99)
        assert found_node is None

    def test_search_on_single_node_list_found(self):
        """Test search on a list with only one node (value found)."""
        node = NonCircularBiDirection(5)
        found_node = node.search(5)
        assert found_node is node

    def test_search_on_single_node_list_not_found(self):
        """Test search on a list with only one node (value not found)."""
        node = NonCircularBiDirection(5)
        found_node = node.search(10)
        assert found_node is None

    def test_search_from_start_of_list(self, three_node_list):
        """Test that search only proceeds forward (using next)."""
        node1, node2, node3 = three_node_list
        found_node = node3.search(1)  # Try to find node1 starting from node3
        assert found_node is node1  # Should not find it by going backwards


class TestNodeRepresentations:
    """Tests for __str__ and __repr__ methods."""

    def test_str_representation(self):
        """Test the __str__ method."""
        node = NonCircularBiDirection(123)
        assert str(node) == "123"
        node_str = NonCircularBiDirection("hello")
        assert str(node_str) == "hello"

    def test_repr_representation_single_node(self):
        """Test the __repr__ method for a single node."""
        node = NonCircularBiDirection(10)
        expected_repr = "Node(10| Prev:None| Next:None)"
        assert repr(node) == expected_repr

    def test_repr_representation_linked_nodes(self):
        """Test the __repr__ method for linked nodes."""
        node1 = NonCircularBiDirection(1)
        node2 = NonCircularBiDirection(2)
        node3 = NonCircularBiDirection(3)
        node1.insert_after(node2)
        node2.insert_after(node3)

        assert repr(node1) == "Node(1| Prev:None| Next:2)"
        assert repr(node2) == "Node(2| Prev:1| Next:3)"
        assert repr(node3) == "Node(3| Prev:2| Next:None)"
