from LinEx.equivalencyCache import EquivalencyCache
from expressions.DictTree import DictTree
from expressions.dimension import Dimension


class KernelTree(DictTree):
    """
    A combination of kernels. The root kernel is to be applied to the root of an ExpressionTree, the child kernels
    are named based on what subtree they should apply to.
    Note: Multiple leaves with the same name only necessarily refer to the same variable if they appear in the same
    kernel. Not across the whole tree.
    Also NOTE: Whether or not CSE can be used is NOT an attribute of the KernelTree. After all A*B and C*D both use
    the SAME KernelTree ({} M* {}), but they don't actually calculate the same subexpression. What's relevant is whether
    the exact expression a KT has to calculate is a CSE candidate.
    """
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    @staticmethod
    def _fixSubtreeNames(lines):
        lineID = 0
        processed = []

        # Rename and sort subtrees to avoid gaps
        while lines:
            newLines = []
            leafSubTrees = []
            # Filter for lines that have no unresolved dependencies:
            for name, code in lines:
                if 'SubTree' in code:
                    newLines.append((name, code))
                else:
                    leafSubTrees.append((name, code))
            assert len(leafSubTrees) > 0 # Otherwise we'd have a looping dependency.
            # For the newly filtered lines, change the names.
            for oldname, code in leafSubTrees:
                newname = f"tmp_{lineID}"
                lineID += 1
                processed.append((newname, code))
                # Change the old name wherever it occurred:
                for i in range(len(newLines)):
                    othername, othercode = newLines[i]
                    newLines[i] = othername, othercode.replace(oldname, newname)

            lines = newLines

        # Change the finished tuples into single lines:
        for i in range(len(processed)-1):
            name, code = processed[i]
            processed[i] = f"{name} = {code}"
        processed[-1] = f"return {processed[-1][1]}" # The final one doesn't need a name, it's the return value.

        return processed

    def toCode(self, tree: 'ExpressionTree', cseCache: 'CSECache', orientedVectors=True):
        """
        Returns a pair: cost, code (code can be multiline in case of CSE).
        Note that the cost always refers to the estimated cost of the actual code that you get. This may not always be
        representative of the overall cost (in particular, calling this on a part of a larger tree will give an accurate
        return, but WITHOUT using CSE based on expression in the larger tree, because those don't exist in the smaller one.)
        In fact, the cost may be a little bigger than required because the subtree will think it is "In charge" of doing
        CSE for candidate subexpressions (even when, within the subtree, they aren't actually subexpressions).
        In short: CSE makes this complicated.
        TODO: If 'orientedVectors' ever influences the cost ... honestly, just don't. It would mess thing up. Including the
         increment cost in cseCache, for one.
        """
        cost, lines, _ = self._toCode(tree, cseCache.softCopy(), orientedVectors=orientedVectors)
        lines[-1] = ("return", lines[-1])
        lines = self._fixSubtreeNames(lines)
        return cost, lines

    def _toCode(self, tree: 'ExpressionTree', cseCache: 'CSECache', orientedVectors=True):
        """
        The return is a tuple: cost, codeLines, bindStrength
        codelines: The lines of code (only 1 if no cse is used) that calculate the expression. The final line is the one
        that should be included in the parent calculation, the previous lines calculate subexpressions.
        All but the final line come in a pair: The ID of the subtree and the actual code. (Last line just has code.)
        """
        if self.isLeaf():
            if tree.isLeaf():
                return 0, [KernelTree.leafToCode(tree, orientedVectors)], float('inf')
            else:
                raise ValueError("Can't generate leaf code for a none-leaf expression. This is an internal error, not a user error.")

        # Check CSE opportunities:
        cseCount = cseCache.countOf(tree)
        if cseCount < 0:
            # Negative values indicate that I already generated the CSE code (so no the solution can be used for free).
            cseName = cseCache.nameOf(tree)
            return 0, [cseName], float('inf') # Inf because the CSE effectively results in brackets, anyway.
        elif cseCount > 1:
            # Tell the children that they will occur (cseCount -1) times less often in the new expression:
            cseCache.decrement(tree, howMuch=cseCount-1, recursive=True)

        # implied else. If we got to this point, we need to actually generate new code.

        # Get the relevant subtrees and calculate child solutions:
        childTrees = self.kernel.getChildTrees(tree)
        childSolutions = dict()
        codeLines = []
        childShapesSymbolic = dict()
        childShapesNumeric = dict()
        childrenCost = 0

        for key, child in childTrees.items():
            # Note for debugging: If self[key] is None, that means there is a subtree key that was not assigned a
            # kernelTree to take care of it.
            childCost, childLines, bindStrength = self[key]._toCode(child, cseCache, orientedVectors)

            # This final code line is placed right inside the rest, potentially needing brackets. The other lines are added in front.
            childSolutions[key] = childLines[-1] if bindStrength > self.kernel.bindStrengths['inner'] else f"({childLines[-1]})"
            codeLines = childLines[:-1] + codeLines # TODO Is the order correct?
            childShapesSymbolic[(key, 'rows')] = Dimension.toCode(child.shape[0])
            childShapesSymbolic[(key, 'cols')] = Dimension.toCode(child.shape[1])
            childrenCost += childCost
            childShapesNumeric[key] = child.numericSize()

        # Generate the expression that calculates the root kernel from the child returns:
        rootCode = self.kernel.toCode(childSolutions, childShapesSymbolic)
        rootCost = self.kernel.getCost(childShapesNumeric)

        # And either return it as the last line, or (for CSE) give it a name and return that as the last line.
        if cseCount > 1:
            cseCache.decrement(tree, howMuch=2, recursive=False) # Mark the tree as already "done" for CSE.

            # Cost includes the cost of having to save the extra variable:
            totalCost = rootCost + childrenCost + tree.copyCost()

            # Finalize code and return:
            cseName = cseCache.nameOf(tree)
            codeLines.append((cseName, rootCode))
            codeLines.append(cseName)
            return totalCost, codeLines, float('inf') # Inf because the CSE is as strong as explicit brackets.
        else:
            # cseCount == 1; all other options have been handled.
            #   Except the outermost layer which is 0 because the outermost expression is never counted for CSE (no point).
            codeLines.append(rootCode)
            return rootCost + childrenCost, codeLines, self.kernel.bindStrengths['outer']

    def getDefs(self):
        if self.isLeaf():
            return set()
        defs = self.kernel.preDefs
        for child in self.values():
            defs |= child.getDefs()

        return defs

    def matches(self, expTree, verbosity: int = 2):
        if self.isLeaf():
            return expTree.isLeaf()

        if not self.kernel.matches(expTree, verbosity=verbosity):
            return False

        for key, child in self.kernel.getChildTrees(expTree).items():
            if not self[key].matches(child):
                return False

        return True

    def __bool__(self):
        """
        Always returns true. This is important, because an empty dict (the superclass) would be False, whereas a leaf
        KernelTree is very much a "real" tree. This has actual consequences in the overall program execution, where a
        value of "False" would denote that a calculation resulted in NO valid tree.
        """
        return True

    @staticmethod
    def leafToCode(leaf, orientedVectors):
        if leaf.name.startswith('Var_'):
            (dim0, dim1) = Dimension.toCode(leaf.shape[0]), Dimension.toCode(leaf.shape[1])
            if leaf.isScalar() or not leaf.isNumeric():
                return leaf.name[4:]
            if leaf.isVector():
                if orientedVectors:
                    return f"np.full(({dim0}, 1), {leaf.name[4:]})"
                else:
                    return f"np.full({dim0}, {leaf.name[4:]})"
            if leaf.isCoVector():
                if orientedVectors:
                    return f"np.full((1, {dim1}), {leaf.name[4:]})"
                else:
                    return f"np.full({dim1}, {leaf.name[4:]})"
            if leaf.isMatrix():
                return f"np.full(({dim0}, {dim1}), {leaf.name[4:]})"
        elif leaf.isDelta():
            if leaf.isScalar():
                return '1.0'
            elif leaf.isMatrix():
                return f'np.eye({Dimension.toCode(leaf.shape[0])})'
        # TODO: Case for inputting dimension code ref; will be required in reshapes. Need a "dim" ExpTree node, too.
        #  ... Not sure reshapes can currently read dims. Should be able, though.
        # If this here happens, it usually means the KernelTree did not fit the Passed expression.
        raise ValueError("Can't generate leaf code for a none-leaf expression. This is an internal error, not a user error.")