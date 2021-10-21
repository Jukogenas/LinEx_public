class ReshapeCache(dict):
    def __init__(self):
        super().__init__()

    def cache(self, fromTree: 'ExpressionTree', reshape: 'Reshape', result, verbosity=2, indent=""):
        """
        The result can be:
            a) 'direct', indicating that the reshape can be applied without using other root reshapes first.
                Child reshapes must be considered to get all possibilities (but those are cached, too, so it's fine).
            b) 'indirect', indicating that the reshape is not applicable like this or simply through child reshapes,
                but POTENTIALLY after other root-level reshapes.
            d) 'futile', indicating that there is no tree in reach of this tree to which the reshape could apply.
        # TODO: Actually save it across runs in the way I advertise.
        """
        treeHash = fromTree.expStrLenient()
        if not treeHash in self:
            if verbosity > 3: print(f"{indent}Encountered a new tree: {str(fromTree)}")
            self[treeHash] = dict()
            # self[treeHash]['_tree'] = fromTree # No longer need to save the exact tree here at all.
        self[treeHash][reshape] = result

    def contains(self, tree: 'ExpressionTree', reshape: 'Reshape'):
        treeHash = tree.expStrLenient()
        if not treeHash in self:
            return False
        return reshape in self[treeHash]

    def lookup(self, tree: 'ExpressionTree', reshape: 'Reshape'):
        # TODO potentially count how often each tree is encountered to determine what is worth caching across calls.
        treeHash = tree.expStrLenient()
        if not treeHash in self:
            return 'unknown Tree'
        if not reshape in self[treeHash]:
            return 'undetermined'
        return self[treeHash][reshape]