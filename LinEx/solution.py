from LinEx.kernelTree import KernelTree
from expressions.expressiontree import ExpressionTree


class Solution():
    # Because fiddling with long tuples manually gets really old.
    def __init__(self, expTree: ExpressionTree = None, kernelTree: KernelTree = None,
                 cseCache: 'CSECache' = None, cost = 0):
        self.expTree = expTree
        self.kernelTree = kernelTree
        self.cseCache = cseCache
        self.cost = cost

        self.complete = False
        self.failReason = "" # We did not fail yet.
        self.graphRoot = "" # This is for drawing the debug graph; remembers the node where this solution originated.

    def copy4Cache(self, newExpTree: 'ExpressionTree' = None, cseCache: 'CSECache' = None):
        """
        Makes a SHALLOW copy of this solution, but uses the passed cseCache, if any, as replacements.
        The reason is that when a solution is cached, its own CSECache must not change (it may be required when later
        generating code from that original solution) but the cached version will need a new CSECache (because it will be
        used in a different branch of the solving process where the count of existing subexpressions differs).
        """
        if newExpTree is None:
            newExpTree = self.expTree
        if cseCache is None:
            cseCache = self.cseCache

        cpy = Solution(expTree=newExpTree, kernelTree=self.kernelTree, cseCache=cseCache, cost=self.cost)
        cpy.failReason = self.failReason
        cpy.graphRoot = self.graphRoot
        cpy.complete = self.complete
        
        assert cpy.complete # Why would I cache an incomplete one?
        cpy.recalculateCost() # Because the use of CSE can change the cost.
        
        return cpy

    def costsMatch(self):
        return self.cost == self.generateCode()[0]

    def recalculateCost(self):
        self.cost = self.generateCode()[0]

    def isValid(self):
        return self.failReason == "" and self.kernelTree is not None and self.kernelTree.matches(self.expTree)

    def markComplete(self) -> 'Solution':
        if self.failReason:
            print("[WARNING] Tried to mark a solution as finished, but ")
        else:
            self.complete = True
        return self

    def markFailed(self, reason: str) -> 'Solution':
        self.complete = False
        self.failReason = reason
        return self

    def generateCode(self, orientedVectors = True, codeIndent="\t"):
        """
        Checks that the solutions is complete, valid, and has the predicted costs.
        Returns a tuple: (cost: number, definitions: str, main code block: str)
        """
        assert self.complete and self.isValid()
        cost, codeLines = self.kernelTree.toCode(self.expTree, self.cseCache, orientedVectors=orientedVectors)
        codeDefs = "\n".join(self.kernelTree.getDefs())
        codeBlock = codeIndent + f"\n{codeIndent}".join(codeLines)
        return cost, codeDefs, codeBlock

    def __str__(self) -> str:
        return f"[Solution using {self.kernelTree.kernel}]"

    # TODO: Initialize solution objects with just the expTree, and have them automatically track a "todo" dict that lists
    #  the (sub)trees that still need to be worked on (and where in the kernelTree those solutions would go).
    #  The 'finished' attribute can then be reworked into a function that checks whether nothing is left to do.
    #  Issue: May have to rebuild the whole expTree every time a part of it changes, even a part low down, because
    #  otherwise it breaks the caching of hashes.
    #  Hm ... would clearing the hash cache of all parent trees be sufficient?
    #  There is, however, on other thing: Solutions are varname-specific, kerneltrees aren't.
    # -> It might then be possible or even sensible to merge solutionTree and kernelTree.
    #    Although that would be a lot of work, since kernelTrees are used all over.