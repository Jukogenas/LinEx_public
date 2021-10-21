from expressions.expressiontree import ExpressionTree
from LinEx.symboltree import SymbolTree

# The cost factors represent a rough cost (per element) for operations where the exact cost may be hard or impossible to
# determine.
# Balancing them against each other is the equivalent of saying "how much worse" an exp calculation (for example) is
# than a simple multiply or add.
# TODO: Possibly add a pre-run configuration step that tries to guess these factors for the specific machine.
#  In case of things like ^, it will have to be averaged across different size inputs.
costFactors = dict()
costFactors['lenient'] = 1 # For things like copying.
costFactors['medium'] = 5 # For things like sin, cos, ... # TODO: Could I find a precise value for those, right?
costFactors['harsh'] = 10 # For things like sqrt, exp, log

class Calculation():
    def __init__(self, requires: SymbolTree, costFunction, codeFunction, bindStrengths = None, preDefs = None, defaultDim=100):
        """
        requires: A SymbolTree that defines what the calculating kernel can work on.
        costFunction: A function (dict of child shapes) -> flop estimate
        codeFunction: A function (dict of child codes) -> valid python code.
        bindStrengths: How strongly this kernel binds its operands. A dict (string -> number)
            bindStrengths['inner']: How strong a binding this requires of child code pieces. If they don't have it, they
            get brackets.
            bindStrengths['outer']: How strong the resulting code will be bound. Used for the next layer to potentially
            add brackets.

            For example, functions would get {'inner': -inf, 'outer': inf}, because they use brackets anyway. This is also
            the default value.

            You can alternatively enter something other than a dict. In that case, 'inner' and 'outer' are both set to
            that value.
        preDefs: Code that should be included above the main function, such as imports and definitions.
        """
        self.requires = requires
        self.costFunction = costFunction
        self.codeFunction = codeFunction
        if bindStrengths is None:
            bindStrengths = {'inner': -float('inf'), 'outer': float('inf')}
        if not isinstance(bindStrengths, dict):
            bindStrengths = {'inner': bindStrengths, 'outer': bindStrengths}
        self.bindStrengths = bindStrengths
        if preDefs is None:
            preDefs = set()
        self.preDefs = preDefs

        self.costEstimate = self._roughCostEstimate(defaultDim)

    def matches(self, other, verbosity: int = 2, indent="\t") -> bool:
        return self.requires.matches(other, verbosity=verbosity, indent=indent)

    def getChildTrees(self, expTree: ExpressionTree):
        return self.requires.inputTreeDict(expTree)

    def reconstructExpTree(self, childTrees: dict):
        return self.requires.toExpressionTree(childTrees)

    def getCost(self, childShapes):
        return self.costFunction(childShapes)

    def _roughCostEstimate(self, defaultDim = 100):
        fakeShapes = fakedict(fakedict(defaultDim))
        return self.getCost(fakeShapes)

    def getCostEstimate(self):
        return self.costEstimate

    def toCode(self, childCodes, childDims):
        return self.codeFunction(childCodes, childDims)

    def __str__(self):
        return f"[Calculation on {str(self.requires)}]"

    def __combine(self, subKernels: dict):
        """
        Returns a new kernel made up of applying this one and using the kernels in the passed dict for the leaf nodes,
        mapped by their names.
        The code and cost functions are combined as well as the symboltree.
        """
        # This function has conceptual issues. See "Thoughts on caching solutions" for more info.
        # TODO: Either solve them or remove this.
        pass

class fakedict():
    def __init__(self, constValue):
        self.constValue = constValue

    def __getitem__(self, key):
        return self.constValue
