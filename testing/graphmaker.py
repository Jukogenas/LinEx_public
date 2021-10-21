from expressions import Parser
from expressions.tree_backend import TreeGenerator
from expressions.expressiontree import ExpressionTree
from LinEx.linex import LinExGenerator as LinexGenerator
from LinEx.reshapeList import assoMatmulLR

def graphReshapeSingle(inp="(A*B)*C", rs=assoMatmulLR):
	dimsAndProps = False
	
	et1 = Parser(inp).parse()
	tg1 = TreeGenerator(et1)
	assert rs.requires.matches(et1)
	et2 = rs.apply(et1)
	tg2 = TreeGenerator(et2)
	print("Dot 1:")
	print(tg1.generate(dimsAndProps=dimsAndProps))
	print()
	print("Dot 2:")
	print(tg2.generate(dimsAndProps=dimsAndProps))
	print()
	print("URL1:", tg1.getImageURL(dimsAndProps=dimsAndProps))
	print("URL2:", tg2.getImageURL(dimsAndProps=dimsAndProps))

def simpleDotNodeOfTree(et: 'ExpressionTree'):
	uid = simpleDotNodeOfTree.counter
	simpleDotNodeOfTree.counter += 1
	return uid, f'\t"node{uid}"[label="{et.toExpressionString(writeNames=True, writeDims=False, properties=False)}" fillcolor="beige" style="filled"]\n'

simpleDotNodeOfTree.counter = 0

def graphReshapeNeighborhood(inp="(A+B)*C"):
	et = Parser(inp).parse()
	LG = LinexGenerator(et)
	
	treesTodo = {et}
	treesVisited = set()
	treeDotMap = dict()
	edges = []
	
	# Add the initial node
	treeDotMap[et] = simpleDotNodeOfTree(et)
	
	# Explore all the top-level reshape options
	while treesTodo:
		tree = treesTodo.pop()

		treesVisited.add(tree)
		LG.exploreTheNeighborhood(tree)

		# TODO: ReshapeCache doesn't care about variable names (that's one of the main points), but I do in this graph. Maybe just do it manually?

		for reshape in LG.reshapeCache[tree]:
			targetTree = LG.reshapeCache[tree][reshape]
			if not isinstance(targetTree, ExpressionTree):
				continue # There is no direct link here.
			if not targetTree in treeDotMap:
				treeDotMap[targetTree] = simpleDotNodeOfTree(targetTree)
			edges.append((treeDotMap[tree][0], treeDotMap[targetTree][0]))
			if not targetTree in treesVisited:
				treesTodo.add(targetTree)
				
	# Write the actual dot code ...
	dotStr = 'digraph g {\n\tgraph [rankdir = "TD"];\n'
	# ... for nodes ...
	for _, dotstuff in treeDotMap.items():
		dotStr += dotstuff[1]
	
	# ... and for edges:
	for a, b in edges:
		dotStr += f'\t"node{a}"->"node{b}"\n'
	
	dotStr += '}'
	print(dotStr)

# FOR LATER (this is my modified neighborhood reshape demo:
"""
digraph g {
	graph [rankdir = "LR"];
	"node0"[label="((A + B) M* C)" fillcolor="beige" style="filled"]
	"node1"[label="((A M* C) + (B M* C))" fillcolor="beige" style="filled"]
	"node2"[label="((B M* C) + (A M* C))" fillcolor="beige" style="filled"]
	"node3"[label="((B + A) M* C)" fillcolor="beige" style="filled"]
	"node0"->"node1" [label="1"]
	"node1"->"node2" [label="2"]
	"node1"->"node0" [label="1"]
	"node2"->"node1" [label="2"]
	"node2"->"node3" [label="1"]
	"node3"->"node2" [label="1"]
	"node0"->"node3" [label="2" color="gray"]
	"node3"->"node0" [label="2" color="gray"]
}
"""


# TODO: option to select specific test via console arguments?
if __name__ == "__main__":
	# graphReshapeSingle()
	graphReshapeNeighborhood()