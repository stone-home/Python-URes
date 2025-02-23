from __future__ import annotations
import uuid
from typing import Any, AnyStr, Dict, Iterator, List, Optional


class TreeNode:
    def __init__(self, value: Any):
        """
        Initialize a TreeNode instance.

        Args:
            value (Any): The value to be stored in the node.

        Returns:
            None

        Example:
            >>> node = TreeNode("root")
            >>> node.value
            'root'
        """
        self._parent: Optional[TreeNode] = None
        self._children: Dict[AnyStr, TreeNode] = {}
        self._value: Any = value
        self._id = uuid.uuid4().hex

    @property
    def parent(self) -> Optional[TreeNode]:
        """
        Get the parent node of this TreeNode.

        Returns:
            Optional[TreeNode]: The parent node if it exists; otherwise, None.

        Example:
            >>> root = TreeNode("root")
            >>> child = TreeNode("child")
            >>> child.set_parent(root)
            >>> child.parent is root
            True
        """
        return self._parent

    @property
    def children(self) -> Dict[AnyStr, TreeNode]:
        """
        Get the dictionary of child nodes.

        Returns:
            Dict[AnyStr, TreeNode]: A dictionary mapping each child's unique ID to its TreeNode instance.

        Example:
            >>> root = TreeNode("root")
            >>> child = TreeNode("child")
            >>> root.add_child(child)
            >>> list(root.children.values())[0].value
            'child'
        """
        return self._children

    @property
    def is_leaf(self) -> bool:
        """
        Determine if the node is a leaf (i.e., has no children).

        Returns:
            bool: True if the node has no children; otherwise, False.

        Example:
            >>> node = TreeNode("leaf")
            >>> node.is_leaf
            True
        """
        return len(self.children) == 0

    @property
    def value(self) -> Any:
        """
        Retrieve the value stored in the node.

        Returns:
            Any: The value of the node.

        Example:
            >>> node = TreeNode(10)
            >>> node.value
            10
        """
        return self._value

    @property
    def id(self) -> str:
        """
        Get the unique identifier of the node.

        Returns:
            str: A unique hexadecimal string identifier for the node.

        Example:
            >>> node = TreeNode("example")
            >>> isinstance(node.id, str)
            True
        """
        return self._id

    def add_child(self, child: TreeNode):
        """
        Add a child node to the current node.

        This method adds the given child to the node's children dictionary (using the child's ID as key)
        and sets the current node as the parent of the child.

        Args:
            child (TreeNode): The child node to add.

        Returns:
            None

        Example:
            >>> root = TreeNode("root")
            >>> child = TreeNode("child")
            >>> root.add_child(child)
            >>> child.parent is root
            True
        """
        if child.id not in self.children.keys():
            self._children[child.id] = child
            child.set_parent(self)

    def remove_child(self, child: TreeNode):
        """
        Remove a child node from the current node.

        This method removes the specified child node from the current node's children and clears the
        child's parent reference.

        Args:
            child (TreeNode): The child node to remove.

        Returns:
            None

        Example:
            >>> root = TreeNode("root")
            >>> child = TreeNode("child")
            >>> root.add_child(child)
            >>> root.remove_child(child)
            >>> child.parent is None
            True
        """
        if child.id in self.children.keys():
            self._children.pop(child.id)
            child.set_parent(None)

    def set_parent(self, parent: Optional[TreeNode]):
        """
        Set the parent of the current node.

        If the node already has a parent, it will be removed from that parent's children before setting
        the new parent.

        Args:
            parent (Optional[TreeNode]): The new parent node. If None, the node will have no parent.

        Returns:
            None

        Example:
            >>> root = TreeNode("root")
            >>> child = TreeNode("child")
            >>> child.set_parent(root)
            >>> child.parent is root
            True
        """
        if parent is not None:
            if isinstance(self.parent, TreeNode):
                self.parent.remove_child(self)
        self._parent = parent

    def backward_stack(self) -> Iterator[TreeNode]:
        """
        Generate an iterator for the path from the current node to the root.

        The iterator yields nodes starting with the current node and then each successive parent until
        no further parent exists.

        Returns:
            Iterator[TreeNode]: An iterator over the nodes from the current node up to the root.

        Example:
            >>> root = TreeNode("root")
            >>> child = TreeNode("child")
            >>> child.set_parent(root)
            >>> [node.value for node in child.backward_stack()]
            ['child', 'root']
        """
        current = self
        while current is not None:
            yield current
            current = current.parent

    def forward_stack(self, **kwargs) -> List[List[Any]]:
        """
        Get all forward paths from the current node to each leaf node.

        This method performs a depth-first search (DFS) to compute every possible path from the current node
        to all leaf nodes. If an optional attribute key is provided via kwargs, the method returns that attribute
        for each node in the path; otherwise, it returns the node itself.

        Keyword Args:
            attr (str, optional): The attribute name to extract from each node. Defaults to None.

        Returns:
            List[List[Any]]: A list of paths, where each path is a list of nodes or attribute values from the
                             current node to a leaf node.

        Example:
            >>> root = TreeNode("root")
            >>> child1 = TreeNode("child1")
            >>> child2 = TreeNode("child2")
            >>> root.add_child(child1)
            >>> root.add_child(child2)
            >>> paths = root.forward_stack(attr="value")
            >>> sorted(paths)
            [['child1', 'root'], ['child2', 'root']]  # Order may vary
        """
        all_paths = []
        self._dfs(self, [], all_paths, **kwargs)
        return all_paths

    def _dfs(self, node: TreeNode, current_path: list, all_paths: list, **kwargs):
        """
        Recursively perform depth-first search (DFS) to find all paths from the given node to leaf nodes.

        This internal helper method accumulates paths by traversing each branch of the tree.
        If a keyword argument 'attr' is provided, it appends the value of that attribute for each node.

        Args:
            node (TreeNode): The current node in the DFS traversal.
            current_path (list): The path taken to reach the current node.
            all_paths (list): A list to store all complete paths.
            **kwargs: Optional keyword arguments.
                - attr (str, optional): The attribute name to use for each node in the path.

        Returns:
            None

        Example:
            >>> # Typically used internally by forward_stack.
            ... pass
        """
        assert isinstance(node, TreeNode)
        _attr = kwargs.get("attr", None)
        _key = node if _attr is None else getattr(node, _attr)
        current_path.append(_key)

        if not node.children:
            all_paths.append(list(current_path))
        else:
            for child in node.children.values():
                self._dfs(child, current_path, all_paths, **kwargs)
        current_path.pop()
