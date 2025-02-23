from __future__ import annotations
import uuid
from typing import Any, Optional


class BiDirection:
    def __init__(self, value: Any):
        """
        Create a bi-directional linked node.

        Initializes a node with the given value and sets its previous and next pointers to itself,
        forming a circular structure when isolated.

        Args:
            value (Any): The value to store in the node.

        Returns:
            None

        Example:
            >>> node = BiDirection("A")
            >>> node.value
            'A'
            >>> node.prev is node
            True
            >>> node.next is node
            True
        """
        self._prev: Optional[BiDirection] = self
        self._next: Optional[BiDirection] = self
        self._value: Any = value
        self._id = uuid.uuid4().hex

    @property
    def prev(self) -> BiDirection:
        """
        Get the previous node in the linked structure.

        Returns:
            BiDirection: The previous node.

        Example:
            >>> node = BiDirection("A")
            >>> node.prev is node
            True
        """
        return self._prev

    @property
    def next(self) -> BiDirection:
        """
        Get the next node in the linked structure.

        Returns:
            BiDirection: The next node.

        Example:
            >>> node = BiDirection("A")
            >>> node.next is node
            True
        """
        return self._next

    @property
    def value(self) -> Any:
        """
        Retrieve the value stored in the node.

        Returns:
            Any: The node's value.

        Example:
            >>> node = BiDirection(123)
            >>> node.value
            123
        """
        return self._value

    @property
    def id(self) -> str:
        """
        Get the unique identifier of the node.

        Returns:
            str: A hexadecimal string representing the node's unique ID.

        Example:
            >>> node = BiDirection("A")
            >>> isinstance(node.id, str)
            True
        """
        return self._id

    def insert_after(self, node: BiDirection) -> None:
        """
        Insert a node immediately after the current node.

        Adjusts pointers so that the new node is placed between the current node and its next node.

        Args:
            node (BiDirection): The node to be inserted after the current node.

        Returns:
            None

        Example:
            >>> node1 = BiDirection("A")
            >>> node2 = BiDirection("B")
            >>> node1.insert_after(node2)
            >>> node1.next is node2
            True
            >>> node2.prev is node1
            True
        """
        node._prev = self
        node._next = self._next
        self._next._prev = node
        self._next = node

    def insert_before(self, node: BiDirection) -> None:
        """
        Insert a node immediately before the current node.

        Adjusts pointers so that the new node is placed between the current node's previous node and the current node.

        Args:
            node (BiDirection): The node to be inserted before the current node.

        Returns:
            None

        Example:
            >>> node1 = BiDirection("A")
            >>> node2 = BiDirection("B")
            >>> node1.insert_before(node2)
            >>> node1.prev is node2
            True
            >>> node2.next is node1
            True
        """
        node._prev = self._prev
        node._next = self
        self._prev._next = node
        self._prev = node

    def remove(self) -> None:
        """
        Remove the current node from the linked structure.

        Adjusts the previous and next nodes to bypass the current node and resets the current node's
        pointers to point to itself.

        Returns:
            None

        Example:
            >>> node1 = BiDirection("A")
            >>> node2 = BiDirection("B")
            >>> node1.insert_after(node2)
            >>> node2.remove()
            >>> node1.next is node1
            True
        """
        self._prev._next = self._next
        self._next._prev = self._prev
        self._prev = self
        self._next = self

    def search(self, value: Any) -> Optional[BiDirection]:
        """
        Search for a node with the specified value in the linked structure.

        Starting from the current node, traverse through the chain until the value is found or the search
        returns to the starting node.

        Args:
            value (Any): The value to search for.

        Returns:
            Optional[BiDirection]: The node with the matching value, or None if not found.

        Example:
            >>> node1 = BiDirection("A")
            >>> node2 = BiDirection("B")
            >>> node3 = BiDirection("C")
            >>> node1.insert_after(node2)
            >>> node2.insert_after(node3)
            >>> found = node1.search("C")
            >>> found.value
            'C'
            >>> not_found = node1.search("D")
            >>> not_found is None
            True
        """
        node = self
        while node.value != value and node.next != self:
            node = node.next
        return node if node.value == value else None

    def __eq__(self, other: BiDirection) -> bool:
        """
        Check equality between two nodes based on their unique IDs.

        Args:
            other (BiDirection): Another node to compare with.

        Returns:
            bool: True if both nodes have the same unique ID, False otherwise.

        Example:
            >>> node1 = BiDirection("A")
            >>> node2 = BiDirection("A")
            >>> node1 == node1
            True
            >>> node1 == node2
            False
        """
        return self.id == other.id

    def __str__(self) -> str:
        """
        Get the string representation of the node.

        Returns:
            str: A string representing the node's value.

        Example:
            >>> node = BiDirection("Hello")
            >>> str(node)
            'Hello'
        """
        return str(self.value)

    def __repr__(self) -> str:
        """
        Get a detailed string representation of the node.

        Returns:
            str: A string in the format "BiDirection(<value>)" representing the node.

        Example:
            >>> node = BiDirection("World")
            >>> repr(node)
            'BiDirection(World)'
        """
        return f"BiDirection({self.value})"
