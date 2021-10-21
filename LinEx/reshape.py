from LinEx.symboltree import SymbolTree

class Reshape():
    """
    A reshape is an operation that:
        a) Does not alter the mathematical result of the tree it is applied to and
        b) has a clearly defined root operation that it causes once applied.
            (That is so I can swiftly consider reshapes that I'd want rather than going through all of them!)
    """

    def __init__(self, requires: SymbolTree, target: SymbolTree, skipIf=lambda tree: False):
        """
        Note: Giving target SymbolTrees attributes is not required. They will propagate from the input tree.
        Just make sure to add required attributes to 'requires' so that they can be considered while matching.
        """
        self.requires = requires
        self.target = target
        self.skipIf = skipIf

    def matches(self, tree: 'ExpressionTree', verbosity=2, indent="\t") -> bool:
        if self.skipIf(tree):
            if verbosity > 3:
                print(f"Skipped {str(tree)} because it fulfilled skipping criteria of {str(self)}")
                return False # The Generator should just assume that it can't be applied.
        return self.requires.matches(tree, verbosity=verbosity, indent=indent)


    def apply(self, tree: 'ExpressionTree') -> 'ExpressionTree':
        """
        Transforms the input tree into the corresponding output tree and returns that.
        Note that this assumes the reshape action is applicable, which should be tested first.
        This does not modify the original tree, but it doesn't create a full copy, either.
        It keeps unchanged nodes in the tree and just rebuilds the structure on top from there.

        Note: If the target tree was given explicit attributes, those are ONLY specifically added in case of the tree
        root. All layers in between that and the children (if any) will have to make do with the automatically inferred
        attributes. Even so, EXPCICITLY ADDING ATTRIBUTES TO THE TARGET TREE IS A GOOD THING, because when constructing
        the .reverse() reshape, those become the gating attributes (since the target tree will then be the source tree).
        """
        readOnlyChildDict = self.requires.inputTreeDict(tree)
        newTree = self.target.toExpressionTree(inputTreeDict=readOnlyChildDict)
        # Add attributes that the new tree must have to avoid having to recalculate them:
        newTree.attributes.update(tree.attributes)
        newTree.attributes.update(self.target.attributes)

        return newTree

    def __str__(self) -> str:
        return f"Reshape {str(self.requires)} -> {str(self.target)}"

    def __hash__(self) -> int:
        # This is relatively limited, but it should do the trick. Could always include deeper levels and skipIf if desired.
        return str(self).__hash__()

    ## Functions to help easily construct Reshapes:
    def reverse(self, skipIf = 'same') -> 'Reshape':
        if skipIf == 'same':
            skipIf = self.skipIf
        elif skipIf == 'reverse':
            skipIf = lambda tree: not self.skipIf(tree)
        return Reshape(requires=self.target, target=self.requires, skipIf=skipIf)

    def copy(self) -> 'Reshape':
        return Reshape(
            requires=self.requires.copy(),
            target=self.target.copy(),
            skipIf=self.skipIf
        )

    def substitute(self, original: str, substitute: str, skipIf = 'same') -> 'Reshape':
        cpy = self.copy()
        cpy.requires = self.requires.substitute(original, substitute)
        cpy.target = self.target.substitute(original, substitute)
        if skipIf != 'same':
            cpy.skipIf = skipIf

        return cpy

    def substituteNoCopy(self, original: str, substitute: str, skipIf ='same'):
        self.requires.substituteNoCopy(original, substitute)
        self.target.substituteNoCopy(original, substitute)
        if skipIf != 'same':
            self.defaultTest = skipIf