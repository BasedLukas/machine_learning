import pytest

from games.tree import Tree

def test_tree():
    args = { 
        "n_daughters": 3,
        "depth": 8,
        "min_depth": 3,
        "sd": 0.1
    }
    tree = Tree(**args)
    assert (tree.n_nodes, tree.n_leaves) == tree.traverse(), tree.traverse() 