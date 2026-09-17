"""Tree and bi-directional list structures used across URes.

Examples:
    >>> from ures.data_structure import TreeNode
    >>> root = TreeNode("root")
    >>> root.value
    'root'
"""
from .tree import TreeNode
from .bi_directional_links import BiDirection, NonCircularBiLink

__all__ = ["TreeNode", "BiDirection", "NonCircularBiLink"]
