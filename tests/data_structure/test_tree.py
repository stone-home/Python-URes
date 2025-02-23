import pytest
import unittest
from typing import Iterator
from unittest.mock import patch
from ures.data_structure import TreeNode


class TestTreeNodeproperties:
    @pytest.fixture(scope="function")
    def instantiate_tree_node(self):
        return TreeNode("new node")

    def test_parent(self, instantiate_tree_node):
        assert instantiate_tree_node.parent is None

    def test_children(self, instantiate_tree_node):
        assert instantiate_tree_node.children == {}

    def test_value(self, instantiate_tree_node):
        assert instantiate_tree_node.value == "new node"

    def test_id(self, instantiate_tree_node):
        assert isinstance(instantiate_tree_node.id, str)


class TestTreeNodeMethods:
    @pytest.fixture(scope="function")
    def instantiate_tree_node(self):
        return TreeNode("new node")

    @pytest.fixture(scope="function")
    def child1(self):
        return TreeNode("child node")

    @pytest.fixture(scope="function")
    def child2(self):
        return TreeNode("child node2")

    def test_add_child(self, instantiate_tree_node):
        child = TreeNode("child node")
        instantiate_tree_node.add_child(child)
        assert child in instantiate_tree_node.children.values()
        assert child.parent == instantiate_tree_node

    def test_ignore_duplcate_child(self, instantiate_tree_node):
        child = TreeNode("child node")
        instantiate_tree_node.add_child(child)
        instantiate_tree_node.add_child(child)
        assert len(instantiate_tree_node.children) == 1

    def test_remove_child(self, instantiate_tree_node, child1):
        instantiate_tree_node.add_child(child1)
        assert child1 in instantiate_tree_node.children.values()
        assert child1.parent == instantiate_tree_node
        instantiate_tree_node.remove_child(child1)
        assert child1 not in instantiate_tree_node.children.values()
        assert child1.parent is None

    def test_set_parent(self, instantiate_tree_node, child1):
        child1.set_parent(instantiate_tree_node)
        assert child1.parent == instantiate_tree_node
        child2 = TreeNode("child node2")
        child2.set_parent(instantiate_tree_node)
        assert child1.parent == instantiate_tree_node
        assert child2.parent == instantiate_tree_node
        assert instantiate_tree_node.children == {}

    def test_backward_stack(self, instantiate_tree_node, child1, child2):
        instantiate_tree_node.add_child(child1)
        child1.add_child(child2)
        backward_stack = child2.backward_stack()
        assert isinstance(backward_stack, Iterator)
        backward_stack_list = list(backward_stack)
        for node in backward_stack_list:
            assert isinstance(node, TreeNode)
        assert len(backward_stack_list) == 3

    def test_backward_stack_root(self, instantiate_tree_node):
        input_len = 10
        current = instantiate_tree_node
        for index in range(input_len):
            child = TreeNode(f"child node {index}")
            current.add_child(child)
            current = child
        backward_stack = current.backward_stack()
        assert len(list(backward_stack)) == input_len + 1

    @patch.object(TreeNode, "_dfs")
    def test_forward_stack_single_layer(
        self, mock_dfs, instantiate_tree_node, child1, child2
    ):
        x = instantiate_tree_node.forward_stack()
        mock_dfs.assert_called_once()
        assert x == []

    def test_dfs_algorith(self, instantiate_tree_node, child1, child2):
        instantiate_tree_node.add_child(child1)
        instantiate_tree_node.add_child(child2)
        all_paths = []
        instantiate_tree_node._dfs(instantiate_tree_node, [], all_paths)
        assert len(all_paths) == 2
        for path in all_paths:
            for element in path:
                assert isinstance(element, TreeNode)

    def test_dfs_stack(self, instantiate_tree_node, child1, child2):
        instantiate_tree_node.add_child(child1)
        child1.add_child(child2)
        all_path = []
        instantiate_tree_node._dfs(instantiate_tree_node, [], all_path)
        assert isinstance(all_path, list)
        assert len(all_path) == 1

    def test_dfs_configuration(self, instantiate_tree_node, child1, child2):
        instantiate_tree_node.add_child(child1)
        child1.add_child(child2)
        all_path = []
        instantiate_tree_node._dfs(instantiate_tree_node, [], all_path, attr="value")
        assert len(all_path) == 1
        assert all_path[0] == ["new node", "child node", "child node2"]


class TestTreeNode(unittest.TestCase):

    def test_node_initialization(self):
        """Test that a new TreeNode initializes correctly."""
        node = TreeNode("root")
        self.assertEqual(node.value, "root")
        self.assertIsNone(node.parent)
        self.assertEqual(len(node.children), 0)
        self.assertIsInstance(node.id, str)

    def test_add_child_and_parent_relationship(self):
        """Test adding a child updates both the parent's children and the child's parent."""
        root = TreeNode("root")
        child = TreeNode("child")
        root.add_child(child)
        self.assertIn(child.id, root.children)
        self.assertIs(child.parent, root)

    def test_remove_child(self):
        """Test that removing a child updates the parent's children and resets the child's parent."""
        root = TreeNode("root")
        child = TreeNode("child")
        root.add_child(child)
        root.remove_child(child)
        self.assertNotIn(child.id, root.children)
        self.assertIsNone(child.parent)

    def test_set_parent_updates_relationship(self):
        """Test that setting a parent removes the node from any previous parent's children."""
        root1 = TreeNode("root1")
        root2 = TreeNode("root2")
        child = TreeNode("child")
        child.set_parent(root1)
        self.assertIs(child.parent, root1)
        # Now set a new parent
        child.set_parent(root2)
        self.assertIs(child.parent, root2)
        self.assertNotIn(child.id, root1.children)

    def test_backward_stack(self):
        """Test that backward_stack returns the correct path from node to root."""
        root = TreeNode("root")
        child = TreeNode("child")
        grandchild = TreeNode("grandchild")
        child.set_parent(root)
        grandchild.set_parent(child)
        path = list(grandchild.backward_stack())
        # Expected order: grandchild, child, root
        expected_values = ["grandchild", "child", "root"]
        self.assertEqual([node.value for node in path], expected_values)

    def test_forward_stack_without_attr(self):
        """Test that forward_stack returns all paths from the current node to leaves without attribute extraction."""
        root = TreeNode("root")
        child1 = TreeNode("child1")
        child2 = TreeNode("child2")
        grandchild = TreeNode("grandchild")
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(grandchild)
        # Expected two paths: [root, child1, grandchild] and [root, child2]
        paths = root.forward_stack()
        # Convert each path to list of node values
        paths_values = [[node.value for node in path] for path in paths]
        expected_paths = [
            ["root", "child1", "grandchild"],
            ["root", "child2"],
        ]
        # Since order may vary, compare sorted lists.
        self.assertEqual(sorted(paths_values), sorted(expected_paths))

    def test_forward_stack_with_attr(self):
        """Test that forward_stack returns attribute values when 'attr' keyword is provided."""
        root = TreeNode("root")
        child = TreeNode("child")
        root.add_child(child)
        paths = root.forward_stack(attr="value")
        # Expected path: [root.value, child.value]
        self.assertEqual(paths, [["root", "child"]])

    def test_is_leaf(self):
        """Test the is_leaf property."""
        node = TreeNode("leaf")
        self.assertTrue(node.is_leaf)
        child = TreeNode("child")
        node.add_child(child)
        self.assertFalse(node.is_leaf)
        self.assertTrue(child.is_leaf)
