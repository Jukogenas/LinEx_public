import expressions.expressiontree as expt # I do it this way to avoid circular import disaster.
import expressions.parser
from testing.dotGraph import DotGraph

class SymbolTree:
    def __init__(self, name, left=None, right=None, attributes=None, forbiddenAttr=None, checkLeafName=False):
        if not isinstance(name, str):
            raise TypeError("The name of a SymbolTree must be a string!")
        self.name = name
        if attributes:
            if isinstance(attributes, str):
                attributes = {attributes} # To make passing a single attribute more convenient.
        else:
            attributes = set()
        self.attributes = attributes
        self.forbiddenAttr = forbiddenAttr if forbiddenAttr else set()
        self.left = left
        self.right = right
        self.children = []
        if self.left:
            self.children += [self.left]
        if self.right:
            self.children += [self.right]
        self.checkLeafName = checkLeafName
        
        # This will be for later:
        self.shortDescCashe = dict()

    @staticmethod
    def fromExpTree(tree: 'ExpressionTree', forbiddenAttr=None, checkLeafName=False):
        left = None
        if tree.left:
            left = SymbolTree.fromExpTree(tree.left, checkLeafName=checkLeafName)
        right = None
        if tree.right:
            right = SymbolTree.fromExpTree(tree.right, checkLeafName=checkLeafName)

        if forbiddenAttr is None:
            forbiddenAttr = set()
        return SymbolTree(name = tree.name, left=left, right=right, attributes=tree.attributes,
                          forbiddenAttr=forbiddenAttr, checkLeafName=checkLeafName)

    def isLeaf(self):
        return not self.children

    # TODO: This function may eventually be performance-critical. At that point, I may need to remove all the debug outputs.
    #  One simple solution would be to return 0 for True or error codes and wrap it in ony of two corresponding functions.
    #  Couldn't return all the details, though. Maybe just drop (most of) the verbosity once sufficiently tested?
    def _matches(self, other, namedLeaves: dict, verbosity: int = 2, indent = "") -> bool:
        """
        other should be an ExpressionTree, but I can't ensure that. (I need to import this class in expressiontree.py,
        so it would be a circular import).
        TODO There is a tricky circular-import thing I've done elsewhere, but I only
         learned about that after writing this function ...
        """
        childIndent = f"{indent}│\t"

        if verbosity >= 3:
            print(f"{indent}Trying to match {self}")

        # Check attributes
        if not self.couldMatch(other, verbosity, indent):
            return False

        # It is a leaf, so the name is not important (just stands for a subtree)
        if self.isLeaf():
            # If the settings indicate that the name must match exactly, check that here:
            if self.checkLeafName:
                if self.name == other.name:
                    if verbosity >= 3: print(f"{indent}{self.name} was the expected leaf name -> Match successful.")
                    return True
                else:
                    if verbosity >= 3: print(f"{indent}{self.name} was not the expected leaf name -> Match failed.")
                    return False

            # If we encountered another subtree with this name, test for equality with the previous subtree!
            elif self.name in namedLeaves.keys():
                if namedLeaves.get(self.name) == other: # TODO: Compare DEFAULT trees.
                    if verbosity >= 3: print(f"{indent}{self.name} successfully matched to previously associated subtree -> Match successful.")
                    return True
                else:
                    if verbosity >= 3: print(f"{indent}{self.name} previously referred to a different subtree -> Match failed.")
                    return False
            # If not, add this one for future reference (so that the next occurrence can compare)
            else:
                namedLeaves[self.name] = other
                if verbosity >= 3: print(f"{indent}Matched {self.name} as a new subtree {str(other)} -> Match successful.")
                return True

        # Wrong operation in root node -> no match
        if not self.name == other.name:
            if verbosity >= 3: print(f"{indent}{self.name} expected but {other.name} found -> Match failed.")
            return False

        # Check child requirements (existence/nonexistence and subtree match if applicable)
        if self.left:
            if not other.left:
                if verbosity >= 3: print(f"{indent}Expected left child for {self.name}, but didn't find one -> Match failed.")
                return False
            # Updates namedLeaves as a desired side-effect.
            if not self.left._matches(other.left, namedLeaves, verbosity=verbosity, indent=childIndent):
                if verbosity >= 3: print(f"{indent}Left child failed the match.")
                return False
        elif other.left:
            if verbosity >= 3: print(f"{indent}Unexpected left child in {self.name} -> Match failed.")
            return False # Theoretically, this should never happen (because leaves are already handled in 'couldMatch').

        if self.right:
            if not other.right:
                if verbosity >= 3: print(f"{indent}Expected right child for {self.name}, but didn't find one -> Match failed.")
                return False
            # Updates namedLeaves as a desired side-effect.
            if not self.right._matches(other.right, namedLeaves, verbosity=verbosity, indent=childIndent):
                if verbosity >= 3: print(f"{indent}Right child failed the match.")
                return False
        elif other.right:
            if verbosity >= 3: print(f"{indent}Unexpected right child in {self.name} -> Match failed.")
            return False

        if verbosity >= 3: print(f"{indent}All conditions cleared -> Match successful.")
        return True

    def matches(self, other, verbosity: int = 2, indent ="") -> bool:
        return self._matches(other, dict(), verbosity = verbosity, indent= indent)

    def couldMatch(self, other, verbosity: int = 2, indent =""):
        """
        Checks whether the root node attributes work out,
        because no legitimate reshaping operation can ever change them.
        """
        # Note: The number of children need NOT necessarily be equal.
        # For example, inv(X) -> solve(X, eye) [unary to binary] is possible.
        # However, a leaf can only ever map to a leaf - as far as I am currently aware.

        # Check whether we are dealing with a real tree here:
        if other == 'timeout!':
            return False

        if other.isLeaf() and not self.isLeaf():
            if verbosity >= 3: print(f"{indent}Trying to match operation onto a leaf -> Match failed.")
            return False

        if not self.attributes.issubset(other.attributes):
            missing = self.attributes.difference(other.attributes)
            if verbosity >= 3: print(f"{indent}Missing attribute(s) {missing} in {str(other)} -> Match failed.")
            return False

        if self.forbiddenAttr.intersection(other.attributes):
            forbidden = self.forbiddenAttr.intersection(other.attributes)
            if verbosity >= 3: print(
                f"{indent}Forbidden attribute(s) {forbidden} in {str(other)} -> Match failed.")
            return False

        return True

    def inputTreeDict(self, tree: 'ExpressionTree') -> dict:
        """
        Returns a dictionary that maps the names of leaves in this SymbolTree to the corresponding subTrees in the
        given ExpressionTree.
        """
        if self.isLeaf():
            return {self.name : tree}
        elif self.left and not self.right:
            return self.left.inputTreeDict(tree.left)
        else:
            return {**self.left.inputTreeDict(tree.left), **self.right.inputTreeDict(tree.right)}

    def toExpressionTree(self, inputTreeDict):
        """
        Creates a new ExpressionTree with the structure of this SymbolTree and the indicated subtrees in place of the
        SymbolTree's leaves.
        This is what you'd use to generate a modified ExpressionTree were the children (not immediate children but the
        successors returned from 'inputTreeDict') have been reshaped.

        Note: It is important to make NEW nodes for the ones that have been touched, not reuse old ones.
        Among other things, this makes sure nothing is cached inside those nodes.
        If a node is NOT touched, it must be a leaf of the operation. In that case, it can be preserved.

        Note also: calling toExpressionTree(inputTreeDict(tree)) should result in an ExpressionTree that is
        identical to 'tree' (but without cached information).
        """
        if self.isLeaf():
            if self.name.startswith('Scalar_'):
                # It is a new constant leaf, which we now create:
                return expt.ExpressionTree(self.name.replace("Scalar", "Var"), shape=(1, 1), attributes={'scalar'})
            # TODO: Maybe I will eventually need others, like 'eye'.
            else:
                # Otherwise it's the name of an input tree, which we now load:
                return inputTreeDict[self.name]

        # if not, find the children ...
        l = None
        if self.left:
            l = self.left.toExpressionTree(inputTreeDict)
        r = None
        if self.right:
            r = self.right.toExpressionTree(inputTreeDict)

        # ... and make a new tree with the desired operation. (Properties will propagate automatically in the constructor)
        newTree = expt.ExpressionTree(self.name, l, r)

        return newTree

    def shortDesc(self, noteNegativeAttribues=True) -> str:
        if noteNegativeAttribues in self.shortDescCashe:
            return self.shortDescCashe[noteNegativeAttribues]
        
        s = self.name + "["
        
        # Write all present and potentially all forbidden attribues:
        attributeSets = [("", self.attributes)]
        if noteNegativeAttribues:
            attributeSets.append(("-", self.forbiddenAttr))
        
        # Attributes (sorting is important to make sure hashes are consistent):
        if self.attributes:
            shortSortedProps = [expt.shortProp[p] for p in sorted(self.attributes)]
            s += ", ".join(shortSortedProps)

        if self.attributes and self.forbiddenAttr:
            s += ', '
            
        # Forbidden attributes:
        if self.forbiddenAttr:
            shortSortedProps = [expt.shortProp[p] for p in sorted(self.forbiddenAttr)]
            s += "-" + ", -".join(shortSortedProps)

        s += "]"
        self.shortDescCashe[noteNegativeAttribues] = s
        return s

    def __str__(self):
        primary = f"[SymbolTree {self.shortDesc()}"

        if self.isLeaf():
            note = "(leaf)"
        elif len(self.children) == 1:
            note = f"(unary: {self.left.shortDesc()})"
        else:
            note = f"(binary: {self.left.shortDesc()}, {self.right.shortDesc()})"

        if self.checkLeafName:
            note += " !!!NAMES MATTER!!! "

        return f"{primary} {note}]"

    def copy(self):
        return SymbolTree(
            self.name,
            self.left.copy() if self.left else None,
            self.right.copy() if self.right else None,
            self.attributes.copy(),
            self.forbiddenAttr.copy()
        )

    def toExpressionString(self):
        """
        This function is NOT efficient. It is intended for use in debug output, like the graph.
        """
        selfLeafNames = self.inputTreeDict(self)
        # Now turn the leaves from SymbolTrees into ExpressionTrees:
        for key, tree in selfLeafNames.items():
            expLeaf = expressions.parser.Parser(tree.shortDesc(False)).parse()
            selfLeafNames[key] = expLeaf
        resTree = self.toExpressionTree(selfLeafNames)
        return self._replaceAttributes(resTree).toExpressionString(True, False) # Because the ExpressionTree infers things that are not actually required.

    def _replaceAttributes(self, tree: 'ExpressionTree'):
        """
        Evil hack for debug purposes. Replaces the attributes in the expTree with those in this tree.
        This works INPLACE and results in an invalid ExpressionTree. DO NOT EVER use this on a tree you still want to
        use for actual solving.
        """
        tree.attributes = self.attributes
        if tree.left and self.left:
            self.left._replaceAttributes(tree.left)
        if tree.right and self.right:
            self.right._replaceAttributes(tree.right)
        return tree


    ## Functions to help construct Symboltrees ##
    def substitute(self, original: str, substitute: str):
        cpy = self.copy()
        cpy.substituteNoCopy(original, substitute)
        return cpy

    def substituteNoCopy(self, original: str, substitute: str):
        if self.name == original:
            self.name = substitute
        if self.left:
            self.left.substituteNoCopy(original, substitute)
        if self.right:
            self.right.substituteNoCopy(original, substitute)

    def combine(self, subTreeLinks: dict):
        """
        Replaces leaves in this SymbolTree with the trees in the passed dictionary, matched by name.
        Note that this modifies the original tree.
        Note that variable names will be changed by adding a number for the tree they are from to avoid name collision.
        The default names will remain for the
        """
        if self.isLeaf():
            return
        if self.left:
            if self.left.isLeaf() and self.left.name in subTreeLinks:
                self.left = subTreeLinks[self.left.name]
        if self.right:
            if self.right.isLeaf() and self.right.name in subTreeLinks:
                self.right = subTreeLinks[self.right.name]

    def toGraph(self) -> DotGraph:
        graph = DotGraph("SymbolTree")
        self._addToGraph(graph)
        return graph

    def _addToGraph(self, graph: DotGraph) -> int:
        uid = graph.addNode(attributes={'label': self.shortDesc(), 'shape': 'oval', 'style': 'filled', 'color': 'lightblue'})

        # Add child nodes:
        if self.left:
            leftID = self.left._addToGraph(graph)
            graph.addEdge(uid, leftID)
        if self.right:
            rightID = self.right._addToGraph(graph)
            graph.addEdge(uid, rightID)

        return uid

# Some useful trees for later (Not currently in use) #
ATA = SymbolTree('M*', left=SymbolTree('T', left=SymbolTree('A')), right=SymbolTree('A'))  # A^T * A
AAT = SymbolTree('M*', left=SymbolTree('A'), right=SymbolTree('T', left=SymbolTree('A')))  # A * A^T
ADAT = SymbolTree('M*',
                   left=SymbolTree('M*', SymbolTree('A'), SymbolTree('D', attributes={'pos_semi_def'})),
                   right=SymbolTree('T', left=SymbolTree('A')))  # (A * D[psd]) * A^T
ATDA = SymbolTree('M*',
                   left=SymbolTree('M*', SymbolTree('T', left=SymbolTree('A')), SymbolTree('D', attributes={'pos_semi_def'})),
                   right=SymbolTree('A'))  # (A^T * D[psd]) * A

# Now put them into handy list. These propagations are "complex" in the sense that all simple ones (that only need to
# consider depth 1) are already handled directly in the ExpTree constructor along with all other processing.
complexPropagations = [
    ({'pos_semi_def'}, [ATA, AAT, ADAT, ATDA]),
]