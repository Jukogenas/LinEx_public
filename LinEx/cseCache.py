from collections import Counter

from LinEx.equivalencyCache import EquivalencyCache
from LinEx.solution import Solution
from expressions.expressiontree import ExpressionTree


class CSECache():
    """
    Caches Partial solutions in the form of: How often they appear in the overall solution, what KernelTree solves them
    best, and what ExpTree that kernelTree applies to.
    The key to cache with is the expString of the ExpressionTree inside, INCLUDING variable names.
    """
    # CSECaches share a lot of things as class variables:
    cseNumerator = 0 # Ensures unique IDs across instances.
    solutions = dict() # ExpTree -> (SubtreeID, Solution) Mapping
    graphHint = dict() # ExpTree -> Debug graph node mapping (only '' entries if graphing is disabled)

    @classmethod
    def reset(cls):
        cls.cseNumerator = 0 # Ensures unique IDs across instances.
        cls.solutions = dict() # ExpTree -> (SubtreeID, Solution) Mapping
        cls.graphHint = dict() # ExpTree -> Debug graph node mapping (only '' entries if graphing is disabled)

    def __init__(self, shapeCache: EquivalencyCache):
        self.counter = Counter()
        self.shapeCache = shapeCache

    def __contains__(self, item):
        if not isinstance(item, ExpressionTree):
            return False
        tree = self.shapeCache.getRepresentative(item)
        return tree.expStrNamed() in CSECache.solutions

    def countOf(self, tree: ExpressionTree):
        tree = self.shapeCache.getRepresentative(tree)
        return self.counter[tree.expStrNamed()]

    def register(self, solution: Solution, recursive=False):
        """
        Usually, when a tree is newly registered that means the subtrees have just been considered. So it should NOT be
        set to recursive. The exception occurs when a solution is loaded from cache (because it doesn't directly consider
        the subtrees in that case).
        """
        assert solution.complete # Caching an incomplete solution would be very bad.
        
        treeRep = self.shapeCache.getRepresentative(solution.expTree)
        if self.isUndesirable(treeRep):
            return 0 # No use doing CSE for leaves. And transposing is (nearly) free in numpy; don't want to cse for that!
        assert treeRep not in self

        etHash = treeRep.expStrNamed()
        assert self.counter[etHash] == 0
        self.counter[etHash] = 1
        CSECache.solutions[etHash] = f"SubTree{self.cseNumerator}", solution
        CSECache.graphHint[etHash] = ""
        CSECache.cseNumerator += 1

        if recursive:
            for key, child in solution.kernelTree.kernel.getChildTrees(solution.expTree).items():
                childRep = self.shapeCache.getRepresentative(child)
                if childRep in self: # It should be, because children are worked on first.
                    # The graph hint has to stay as it is, although that's not ideal.
                    self.increment(childRep, recursive=recursive)
                elif not childRep.isLeaf():
                    # TODO: I can't assign a graph hint to this one in here ...
                    subSolution = Solution(child, solution.kernelTree[key], self).markComplete()
                    self.register(subSolution, recursive=recursive)

    def isUndesirable(self, treeRep):
        # Filter cases that aren't worth applying CSE to:
        return treeRep.isLeaf() or (treeRep.name == "T" and treeRep.isUnary() and treeRep.left.isLeaf())

    def increment(self, tree: ExpressionTree, recursive=True):
        """
        Increases the count of this specific ExpressionTree, remembers both trees (assuming it didn't know them yet) or
        asserts that they are indeed what has previously been hashed under the same expression.
        If the count rises to exactly 2, the KernelTree in question will automatically be told that it is now a cse
        candidate.
        
        Make sure to pass a representant, not just any tree.

        Returns (the cost, the original solution, the sub tree name) as a tuple.
        Cost:
         Either the full cost of the tree (if the count was 0 before but a solution was already stored)
          or a nominal cost for copying the SubTree result (based on its size, applies if new count is 2)
          or 0 (no cost if new count is greater than 2 because a copy of the result was already around).
        Original Solution:
         The KernelTree that Calculates the original expression.
        SubTreeName:
         This should be used as the name of the new leaf node.
        """
        treeRep = self.shapeCache.getRepresentative(tree)
        if self.isUndesirable(treeRep):
            return 0 # No use doing CSE for leaves.

        etHash = treeRep.expStrNamed()
        assert etHash in CSECache.solutions # It may not be in counter, but counter defaults to 0 anyway.
        newValue = self.counter[etHash] + 1
        self.counter[etHash] = newValue
        solName, solution = CSECache.solutions[etHash]
        solution = solution.copy4Cache(cseCache=self) # The cache probably changed since it was saved.
        kernelTree = solution.kernelTree

        if recursive:
            for key, child in kernelTree.kernel.getChildTrees(solution.expTree).items():
                childRep = self.shapeCache.getRepresentative(child)
                assert self.isUndesirable(childRep) or childRep in self # It should be, because if this tree was incremented now, it was registered earlier.
                self.increment(childRep, recursive=recursive)

        # Calculate the cost based on how often the subexpression now occurs.
        # TODO: Copy not at 2, but at "now more often than parent" (although new or old? root parent should be 1 but is 0 ...)
        #  Also, recursive stuff now matters. In fact, even newly registering one can mean I copy down below, so there
        #  has to be a register cost ...
        if newValue == 1:
            cost, _ = kernelTree.toCode(solution.expTree, cseCache=self) # Need to calculate it, because this is the first time.
        elif newValue == 2:
            cost = solution.expTree.copyCost() # Need to copy for reuse of the solution.
        elif newValue > 2:
            cost = 0 # It is already copied.
        else:
            assert False
        return solName, solution, cost

    def setGraphHint(self, tree: ExpressionTree, graph):
        treeRep = self.shapeCache.getRepresentative(tree)
        assert graph != '' # That would be the same as setting no hint, which would be done if graphing is disabled.
        etHash = treeRep.expStrNamed()
        if etHash not in CSECache.solutions:
            print(f"Warning: Tried to set graph hint for unknown tree {str(treeRep)}.")
        elif self.counter[etHash] != 1:
            print("Warning: Can only set graph hint for trees that have *just* reached a count of 1.")
        else:
            CSECache.graphHint[treeRep.expStrNamed()] = graph

    def getGraphHint(self, tree: ExpressionTree):
        treeRep = self.shapeCache.getRepresentative(tree)
        return CSECache.graphHint.get(treeRep.expStrNamed(), f"No graphhint for {str(treeRep)}")

    def decrement(self, tree: ExpressionTree, recursive=True, howMuch = 1):
        """
        Decreases the counter corresponding to this exact expressionTree. Returns the new count.
        If the new value is exactly 1, the tree will automatically be told that it is no longer a cse candidate.
        If recursive, child trees (not immediate children but the descendants that are used in the top-level solution kernel)
        will also be decremented.
        This is not used when finding solutions - just copy the counts when the search branches. Instead, this is used
        to generate the code once at the end for a finished solution.

        Returns the new count.
        """
        treeRep = self.shapeCache.getRepresentative(tree)
        if treeRep.isLeaf():
            return 0 # No use doing CSE for leaves.

        etHash = treeRep.expStrNamed()
        if self.counter[etHash] <= 0:
            return 0
        newCount = self.counter[etHash] - howMuch
        assert newCount >= -1 # -1 is used as a marker for "already generated with CSE in mind" in the Kerneltree.
        self.counter[etHash] = newCount

        # See to children, if desired:
        if recursive:
            for key, child in CSECache.solutions[etHash][1].kernelTree.kernel.getChildTrees(treeRep).items():
                childRep = self.shapeCache.getRepresentative(child)
                self.decrement(childRep, recursive=recursive, howMuch=howMuch)

        return newCount

    def nameOf(self, tree: ExpressionTree):
        tree = self.shapeCache.getRepresentative(tree)
        assert tree in self
        return CSECache.solutions[tree.expStrNamed()][0]

    def softCopy(self) -> 'CSECache':
        cpy = CSECache(self.shapeCache)
        cpy.counter = self.counter.copy() # Need to copy because this part will be separately modified.
        # The class variables are deliberately class variables to work the same across instances, so no copying there.
        return cpy