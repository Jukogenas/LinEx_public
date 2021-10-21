from testing.dotGraph import DotGraph
from LinEx.calculation import Calculation
from expressions.expressiontree import ExpressionTree

# TODO: Support for marking CSE and cached results.

class DebugDrawer(DotGraph):
    strAttributes = {"style": "filled", "color": "aquamarine3", "shape": "note"}
    expAttributes = {"style": "filled", "color": "antiquewhite"}
    knlAttributes = {"style": "filled", "color": "beige", "shape": "box"}

    # TODO: self.counter could be removed; I integrated that functionality in the parent class.
    def __init__(self, name = "LinEx_debug_graph"):
        super().__init__(name)
        self.counter = 0

    def nodeFromString(self, string: str, root: str = None, edgeLabel=""):
        uid = str(self.counter)
        self.counter += 1
        self.addNode(uid, {**DebugDrawer.strAttributes, **{"label": string}})
        if root is not None:
            self.addEdge(root, uid, {"label": edgeLabel})
        return uid

    def nodeFromExpression(self, tree: ExpressionTree, root: str = None, edgeLabel=""):
        uid = str(self.counter)
        self.counter += 1
        self.addNode(uid, {**DebugDrawer.expAttributes, **{"label": tree.toExpressionString(True, False)}})
        if root is not None:
            self.addEdge(root, uid, {"label": edgeLabel})
        return uid

    def nodeFromKernel(self, kernel: Calculation, root: str = None, edgeLabel=""):
        uid = str(self.counter)
        self.counter += 1
        label = kernel.requires.toExpressionString()
        self.addNode(uid, {**DebugDrawer.knlAttributes, **{"label": label}})
        if root is not None:
            self.addEdge(root, uid, {"label": edgeLabel})
        return uid

    def dotURL(self, attributes:dict = None):
        if attributes is None:
            attributes = {}
        return super().dotURL(attributes=attributes)