class DictTree(dict):
    """
    A tree datastructure in which the list of child trees per node is represented in a dict (and every node has a key).
    """
    def __init__(self):
        super().__init__()

    def children(self):
        """
        Returns the children as a list.
        """
        return [ch for ch in self.values()]

    def isLeaf(self):
        return len(self) == 0

    def isUnary(self):
        return len(self) == 1

    def isBinary(self):
        return len(self) == 2

    def depth(self):
        """
        The depth of the tree (longest path from root to a leaf).
        If the tree contains a circle - which it really shouldn't - this function will cause infinite recursion.
        """
        ch = self.children()
        if ch:
            return max(c.depth() for c in ch)
        else:
            return 0