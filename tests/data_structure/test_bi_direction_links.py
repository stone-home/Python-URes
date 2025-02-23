import unittest
from ures.data_structure.bi_directional_links import BiDirection


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
