from typing import List, Optional, Tuple
import random


class Node:
    def __init__(self, value: float, sd:float, daughters: Optional[List["Node"]] = None):
        self._value = value
        self.daughters = daughters
        self.sd = sd

    # @property
    def value(self):
        """Value of the node drawn from a normal distribution centred around self._value"""
        return random.gauss(self._value, self.sd)

class Tree:
    def __init__(self, n_daughters:int, depth:int, min_depth:int, sd:float = 0.1)->Node:
        """
        n_daughters: Number of daughters per node, deterministic
        depth: Stochastic, but on average 5
        min_depth: 2
        sd: the standard deviation of the normal distribution from which the value of a node is drawn
        """
        self.n_daughters = n_daughters
        self.depth = depth
        self.min_depth = min_depth
        self.sd = sd
        self.max_depth = depth *3
        self.n_nodes = 0
        self.n_leaves = 0
        self.root = self._create_node(0,0)


    def _create_node(self, current_depth: int, value: float) -> Node:
        """recursively creates a node and its daughters"""
        self.n_nodes += 1
        if current_depth == self.max_depth or current_depth >= self.min_depth and random.random() > (0.5 - current_depth / self.depth):
            self.n_leaves += 1
            return Node(value, self.sd)

        values = self._generate_values()
        daughters = [self._create_node(current_depth + 1, value) for value in values]
        return Node(value, self.sd, daughters)

    def _generate_values(self) -> List[float]:
        """generates values for daughters of a node, summing to 0"""
        values = [random.uniform(-1, 1) for _ in range(self.n_daughters)]
        avg = sum(values) / self.n_daughters
        # Adjust the values to ensure they sum to 0
        return [value - avg for value in values]

    def traverse(self)->Tuple[int, int]:
        """returns the number of nodes and leaf nodes in the tree"""
        nodes_visited, leaf_nodes_visited = 0, 0
        def _traverse(node: Node):
            nonlocal nodes_visited, leaf_nodes_visited
            nodes_visited += 1
            if node.daughters is None:
                leaf_nodes_visited += 1
                return
            for daughter in node.daughters:
                _traverse(daughter)
        _traverse(self.root)
        return nodes_visited, leaf_nodes_visited


    def ideal_path(self)->Tuple[List[int], float]:
        """returns the path from the root to any leaf with the highest possibele total value, and the value"""
        all_values = []
        all_paths = []
        def _traverse(node: Node, path: List[float], value:float):
            if node.daughters is None:
                all_paths.append(path[:])
                all_values.append(value)
                return
            for i, daughter in enumerate(node.daughters):
                _traverse(daughter, path + [i], value+daughter._value)
        

        _traverse(self.root, [], self.root._value)
        index_max_value = all_values.index(max(all_values))
        best_path = all_paths[index_max_value]
        return best_path, max(all_values)


if __name__ == "__main__":
    args = { 
        "n_daughters": 3,
        "depth": 8,
        "min_depth": 3,
        "sd": 0.1
    }
    tree = Tree(**args)
    nodes, leaves = tree.traverse()
    best_path, best_score = tree.ideal_path()
    print(f"Nodes: {nodes}, leaves: {leaves}")
    print(f"Best path: {best_path}, best score: {best_score}")


