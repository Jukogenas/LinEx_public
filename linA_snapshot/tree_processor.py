from collections import defaultdict
from linA_snapshot.tensortree import TensorTree

"""
Part 0: Helper functions=======================================================
"""

"""
Generates the expression String of a node (copy from the expression generator)
"""


def generate(node):
	if hasattr(node, 'exprString'):
		return node.exprString
	s = ""
	if node.left and node.right:
		s += "(" + generate(node.left)
		s += str(node.name)
		s += generate(node.right) + ")"
	elif node.left:
		# special case T
		if node.name == "T":
			s += f"{generate(node.left)}'"
		else:
			s += f"{node.name}({generate(node.left)})"
	else:
		s += str(node.name)
	node.exprString = s
	return s


"""
Adds pointers to parent nodes to all nodes in the tree and retuns a set with
the tree's leaves.
"""


def reverseOrganizeTree(tree, parent):
	tree.parent = parent
	if not tree.left and not tree.right:
		return {tree}
	# implicit "else"
	left = reverseOrganizeTree(tree.left, tree) if tree.left else set()
	right = reverseOrganizeTree(tree.right, tree) if tree.right else set()

	return left | right


"""
Takes a set of nodes and returns the set of their parent nodes. Nodes that
have no parents add nothing to the set.
"""


def getHigherLayer(nodeSet):
	res = {node.parent for node in nodeSet}
	res.discard(None)  # Added by nodes with no parent
	return res


"""
Part 1: Elimination of common subexpressions.==================================
Takes a Tensortree; splits it into multiple Trees with named results.
"""


def splitSubtrees(tree):
	from copy import copy
	subTreeList = []
	subTreeInfo = {}

	# Search the bottommost layer and save reverse pointers on the way.
	stage = reverseOrganizeTree(tree, None)

	# Fill generate() cache for leaves
	for leaf in stage:
		generate(leaf)
	# Go up one layer: Common leaves are just variables and not worth unifying.
	stage = getHigherLayer(stage)

	# Work with the result until nothing is left to do:
	while stage:
		# Group nodes in the layer that create the same expression:
		stageDict = defaultdict(list)
		for node in [n for n in stage if all(hasattr(c, "exprString") for c in n.children)]:
			stageDict[generate(node)].append(node)

		# per group, eliminate unnecessary copies. Add one instance to the result dict; replace the entry in the parent.
		# TODO (optional): allow both directions for commutative operation.
		for nodeList in stageDict.values():
			if len(nodeList) > 1:  # unique subtrees can stay
				subTreeName = f'SubTree{len(subTreeList)+1}'
				subTreeList.append((subTreeName, copy(nodeList[0])))  # any one will do; they are the same
				subTreeInfo[subTreeName] = (len(nodeList), nodeList, copy(nodeList[0]))
				# Truncate the trees in their original locations:
				for node in nodeList:
					node.name = "Var_" + subTreeName
					node.left = None
					node.right = None
					node.children = []

		stage = getHigherLayer(stage)

	# now find all useless subtrees
	uselessSubTrees = set()
	for name, subTree in reversed(subTreeList):  			# for every subtree (big to small)
		size, _, _ = subTreeInfo[name]
		for c in subTree.children:
			if c.name.startswith("Var_SubTree"):			# check if a child is a subtree
				name2 = c.name[4:]
				size2, nodeList2, subTree2 = subTreeInfo[name2]
				if size2 == size:  							# if it occurs as often it is a useless subtree
					for node in nodeList2:					# inline old subtree in all places
						node.name = subTree2.name
						node.left = subTree2.left
						node.right = subTree2.right
						node.children = subTree2.children
					uselessSubTrees.add(name2)

	# do not return useless subtrees
	subTreeList = [e for e in subTreeList if e[0] not in uselessSubTrees]

	# reenumarate subTrees to remove gaps left by uselessSubTrees
	for i, (oldName, subTree) in enumerate(subTreeList):
		_, nodeList, _ = subTreeInfo[oldName]
		subTreeName = f'SubTree{i+1}'
		for node in nodeList:
			node.name = "Var_" + subTreeName
		subTreeList[i] = (subTreeName, subTree)

	# Add the parent tree to the dict:
	subTreeList.append(('result', tree))

	return subTreeList


# TODO: Cleanup this giant comment and the one a little below. Maybe make a nice, fused version. Or put most of it in a doc.
"""
Part 1: Optimize Broadcasting.=================================================
Takes a Tensortree; moves generation of higher dimension objects (vectors, amtrices)
up in the tree as far as possible to keep calculating only with scalars for as
long as possible.

Afterwards, all nodes in the tree will have an attribute "needsBroadcast" set
to True iff it does not yet match the dimensions that it ought to. Depending
on the backend, these may need to be "fixed" when generating the code.

---
The idea is to broadcast only when needed (and as high in the tree as needed).
Essentially, we determine when all entries in a matrix or vector are the same
(of scalars this is always true) and in such cases simplify the calculation.
For example, [complicated scalar] * [huge vector of 2s] is better written as
[huge vector of [complicated scalar] * [2]]. There is no need to multiply a 
huge number of times.

There are also cases in which only one operand having equal entries simplifies
the caculation. For example: 
	"v.T * vector(a)" is simply "a*sum(v)" and
	"v * vector(a).T" is "matrix(Every column is a*v)"
	
Ideally, these simplifications can even prevent unecessary vectors from being
created in the first place (although I'm not sure how often that will happen.)
														
Also, there are cases where simplification is NOT based on all values being the same.
Example:
	Eye * v is simply diag(v)
And other cases where we have matrices that are not trivial but that could be 
simplified (e.g. diag(v) * diag(w) is diag(v .* w)).
Oh, also tr(diag(v)) is sum(v), and I think there's mores stuff like that.

Many (all?) of those come down to a matrix being diagonal, so that is also 
worth tracking.
(There are likely more complicated cases, too, but those can come later.)

Some issues have also been removed by the tree creation itself (like double
transpose and many others). That said, not all of the simplifications in the 
tree are necessarily performance enhancements.

Finally, there is a group of simplifications that would work in theory but can
never actually occur. Example: 'vector(a).T * vector(b)' = a * b * (len of vectors)
This can never occur, because: If the dimension is not fixed, as in the example,
there is no way to know the result. That would be an invalid input. But if the
dimension is known, that means it was set by a variable array or matrix. Since
we don't know anything about its contends (all elements being the same, it being
diagonal etc.) there is nothing we can do with it.

"""

""" These functions handily test for a specific set of flags """
def basicallyAScalar(node):
	"""
	Returns whether this node's value can be represented as a single scalar (independently of what it really represents)
	"""
	return node.broadcastFlag in ['scalar', 'scalar2covector', 'scalar2vector', 'scalar2matrix', 'scalar2diag']

""" These functions transform nodes """
def reduceToScalar(node):
	"""
	The node will become a scalar and will NOT be broadcast. Its parent node must take over responsibility for
	whatever broadcasting was originally intended.
	"""
	if basicallyAScalar(node): # In the actual 'scalar' case there is no real change.
		node.upper = []
		node.lower = []
		node.broadcastFlag = 'scalar'
	else:
		print("[Warning] While optimizing, tried to squish something into a scalar that wouldn't fit.")

def reduceToVector(node):
	"""
	The node will become a vector and will NOT be broadcast. Its parent node must take over responsibility for
	whatever broadcasting was originally intended.
	"""
	if node.isMatrix() and node.broadcastFlag in ['scalar2diag', 'vector2diag', 'vector']:
		# node.upper stays as it is
		node.lower = []
		node.broadcastFlag = 'vector'
	else:
		print(f"[Warning] While optimizing, tried to squish something ({node.broadcastFlag}) into a vector that would not fit!")

def reduceToCoVector(node):
	"""
	The node will become a vector and will NOT be broadcast. Its parent node must take over responsibility for
	whatever broadcasting was originally intended.
	"""
	if node.isMatrix() and node.upper == node.lower and node.broadcastFlag in ['scalar2diag', 'vector2diag', 'covector']:
		# node.upper stays as it is
		node.upper = []
		node.broadcastFlag = 'covector'
	else:
		print("[Warning] While optimizing, tried to squish something into a covector that would not fit!")

def setToRetainDimension(node, reference):
	"""
	The node will be set to the reference node, but will retain its own (intended) dimension.
	The broadcastingFlag is retained in any event, because the original 'setTo' method was never intended for it.
	"""
	# TODO: Remove wrapper once new version is confirmed to be working.
	#upper, lower = node.upper, node.lower
	#node.setTo(reference) # broadcastFlag is retained anyway
	#node.upper, node.lower = upper, lower
	node.setToRetainDimension(reference)

def processByTags(expectation, node1, node2):
	"""
	This functions helps when checking for more or less specific tags of two nodes (generally those that will be
	processed together in the next operation).
	expectation can be a tag or a list of tags.
	The return is False, False if neither node has a tag corresponding to (one of those in) expectation, or a pair
	(nodeA, nodeB) where nodeA is (the) one that does correspond to it. There is no guarantee regarding the tag of the
	second node.

	The main purpose is to easily perform order-agnostic checks.
	"""
	if not isinstance(expectation, list):
		expectation = [expectation]

	if node1.broadcastFlag in expectation:
		return node1, node2
	elif node2.broadcastFlag in expectation:
		return node2, node1
	else:
		return False, False

# Now we get to the actual processing.
#
# The important point during the optimizations is to make sure all cases work individually, that
# way they can be added, changed or removed without much fuss. The "complicated"
# combinations are solved simply by reapplying this all the way up the tree.
# To ensure proper order and compatibility, here is a basic code of conduct:
# 	- the function call always ends with a tree that is semantically equal to
# 	the input tree. In particular, it must be a valid tree. Making a mess and
# 	fixing it in a later call to a parent node is not enough.
# 	- when called on a node, the function may only modify the subtree of
# 	which node is the root.
# 	- when a node is processed, all subnodes must already have been processed.
# 	- after a node has been processed, the node.broadcastFlag has been set
# 	to one of the following values:
# 		### Info only, no broadcast ###
# 		- 'none': The node should remain as it is. We know nothing useful about it.
# 		- 'scalar': The node is and should be a scalar. No broadcasting needed, but it may be good to know.
# 		- 'vector': The node is and should be a column vector. No broadcasting needed, but it may be good to know.
# 		- 'covector': The node is and should be a row vector. No broadcasting needed, but it may be good to know.
#
# 		### No further optimization possible, but must broadcast in backend ###
# 		- 'scalar2vector': For further calculation, this mus be cast to a column vector. Its other dimension will not
# 			be expanded, even if it is something other than 1. This is the result of optimizations in the parent node.
# 		- 'scalar2covector' as the above, but expanding along the other dimension
#
# 		# Special cases for calculations (a "fake broadcasts") #
# 		- 'reduce-multiply-child-dims': Indicates that the result should be the child result multiplied by
# 			the child dimensions. This is for cases like sum and trace performed on a child with a
# 			scalar2* tag (although that tag can then be removed, because the broadcast is never necessary).
# 				-> In unary cases, multiplies with both child dimensions (for sum etc.)
# 				-> in binary cases, multiplies with the "middle" dimension (the one that disappears; makes sense (only?) for * )
#
# 		### Info for further optimization and/or broadcast ###
# 		- 'scalar2matrix': The node can be stored as a scalar, but represents a larger matrix (or vector) with all entries the same.
# 			The resulting shape is whatever its upper and lower values suggest.
# 		- 'scalar2diag': The node can be stored as a scalar, but represents a diagonal matrix (all values on the diagonal
# 			are that scalar, the rest 0).
# 		- 'vector2diag': The node can be stored as a vector, but represents a matrix with that diagonal.
# 			I use a column vector to store it in that case.
#
#
# 		NOT READY:
# 		 - flag for vector2matrix and covector2matrix for "stacking"?
#		 - Note: The inclusion would also require adding these to existing subcases, because I sometimes make assumptions
#			about what flag options are possible (based on the flags that exist currently). Example: v2d .* other
#
# 		 ALSO: Several operations here are associative, but the tree structure defines a specific order. Sometimes a
# 		 different order would allow more optimizations, but reordering the entire tree would be difficult and messy ...
#
# The procedure for adding a case is:
# 	- test for when it occurs (what operation? What restrictions on operands?)
# 	- move / change / modify nodes as required | change operation if necessary
# 	- propagate broadcastFlag upwards if and as applicable
# 	- Reduce child size as much as possible.
# 	- Make sure the node always gets some broadcastFlag so the process can propagate further up the tree.
# 	Useful info: node.setTo does not override the broadcastFlag

def broadcastProcessLeaf(node, debugOut=False):
	operation = node.name

	# Initialize
	if hasattr(node, "broadcastFlag"):
		return  # This was relevant when the Tensortree was allowed to set flags (we removed that, but the case here doesn't hurt).
	elif node.isScalar():
		node.broadcastFlag = 'scalar'
	elif node.isVector():
		if node.isNumeric():
			node.broadcastFlag = 'scalar2vector'
		else:
			node.broadcastFlag = 'vector'
	elif node.isCoVector():
		if node.isNumeric():
			node.broadcastFlag = 'scalar2covector'
		else:
			node.broadcastFlag = 'covector'
	elif node.isMatrix():
		if node.isNumeric():
			node.broadcastFlag = 'scalar2matrix'
		elif node.isDelta(): # input was 'eye'
			node.broadcastFlag = 'scalar2diag' # The backend was taught not to prematurely expand it.
			# node.name = 'Var_1' # This is an alternative, but sadly it causes issues in simplifyOperation (.^eye is not a neutral operation, but .^Var_1 is)

	if not hasattr(node, 'broadcastFlag'): # Ensure that there is at least *some* flag so the process can propagate up.
		node.broadcastFlag = 'none' # a matrix full of arbitrary entries gets here, for example.

	if debugOut:
		print("\tProcessed leaf", operation, "and assignd flag", node.broadcastFlag)

# TODO: It may be simpler to read and understand a version in which every outside check tests for a combination of
# all required triggers (maybe make own function that can be called with swapped child order for simplicity).
# There will be extra checks, yes, bit with elif that's not so bad. And the code might become cleaner.
def broadcastProcessNonLeaf(node, verbose=False, debugOut = False):
	"""
	This is where all the various cases go; there will be many, but the comments will give some orientation.
	The two obvious ways to keep order are
		a) by operation (this is what I do, because it groups similar actions together and can best reuse code)
		b) by child flags (would have fewer outer cases, but be a mess inside each one)
			However, One *could* optimize by catching "boring" cases (such as all children being flagged with 'none') 
			early and skip all the checks for them.
	"""

	if hasattr(node, 'broadcastFlag'):
		if debugOut: # TODO: hunt for the cause of extra calls (at least when using the normal generator?)
			print("[Warning] ", node, " (", node.name, ", ", node.broadcastFlag ,") would have been processed more than once!")
			print("\t(If this happens precisely once at the end, it's nothing to worry about.)")
		return # Avoid processing nodes more than once

	if debugOut:
		if node.isBinary():
			print("Now processing: ", node.left.broadcastFlag, " ", node.name, " ", node.right.broadcastFlag)
		elif node.isUnary():
			print("Now processing: ", node.name, " (", node.left.broadcastFlag, ") ")
		else:
			print("Now processing: ", node.name)

	# ----- Unary Operation -----
	# Case: Transpose is pointless ...
	if node.name == 'T':
		# ... because it does nothing
		if node.left.broadcastFlag in ['scalar', 'scalar2matrix', 'scalar2diag', 'vector2diag']:
			node.broadcastFlag = node.left.broadcastFlag
			node.setTo(node.left) # skip transpose
			if verbose: print("\tRemoved pointless transpose for symmetric matrix.")

		# ... because it can be handled by the broadcast itself.
		elif node.left.broadcastFlag in ['scalar2vector', 'scalar2covector']:
			node.broadcastFlag = 'scalar2vector' if node.left.broadcastFlag == 'scalar2covector' else 'scalar2covector'
			node.setTo(node.left) # skip transpose
			node.lower, node.upper = node.upper, node.lower # Swap dimensions to match the flag
			if verbose: print("\tRemoved transpose: Can simply broadcast along the other dimension.")


	# Case: Extract diagonal from matrix ...
	elif node.name == 'diag2' and node.isVector():
		# ... where all entries are already identical (simply broadcast them to the desired shape).
		if basicallyAScalar(node.left):
			setToRetainDimension(node, node.left) # Skip the diagonal extraction step ...
			node.broadcastFlag = 'scalar2vector' # ... and just broadcast a little less in the first place.
			if verbose: print(f"\tReplaced a 'diag' (scalar2matrix -> vector) with a different flag (scalar2vector).")

		# ... that was already a diagonal matrix (simply use the diagonal vector as it is).
		# NOTE: the 'scalar2diag' case is handled above, NOT here, because it can be compressed more that way.
		elif node.left.broadcastFlag == 'vector2diag':
			setToRetainDimension(node, node.left) # Skip the diagonal extraction step ...
			node.broadcastFlag = 'vector' # ... and just don't broadcast in the first place.
			if verbose: print(f"\tRemoved a 'diag' (vector2matrix -> vector) that did nothing.")

	# Case: Manual Vector -> Diagonal matrix
	elif node.name == 'diag' and node.isMatrix():
		# Remove pointless transpose
		if node.left.name == 'T':
			node.left.setTo(node.left.left)
			if verbose: print(f"\tSkipped 'transpose' before 'diag' (no effect).")

		# Handle the broadcast as a flag rather than an explicit node
		node.broadcastFlag = node.left.broadcastFlag # Save the flag for now and ...
		setToRetainDimension(node, node.left) # ... skip the explicit "diag".

		if basicallyAScalar(node):
			node.broadcastFlag = 'scalar2diag'
			if verbose: print(f"\tReplaced a 'diag' (scalar -> vector -> matrix) with a broadcasting flag.")
		elif node.broadcastFlag in ['vector', 'covector']: # Which one it was won't matter
			node.broadcastFlag = 'vector2diag'
			if verbose: print(f"\tReplaced a 'diag' (vector -> matrix) with a broadcasting flag.")
		else: # This should never happen unless we made a mistake, because diag(vector) has no other valid inputs.
			if debugOut: print("Encountered a 'diag' (v->M), but it doesn't fit any of the possible cases! Help!")

	# Case: elementwise operation on one child (things that do nothing exciting but need to propagate the flags)
	elif node.name in ['u-', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'exp', 'log', 'tanh', 'abs', 'sign', 'relu']:
		# ... but all (nonzero) entries are actually identical
		if basicallyAScalar(node.left):
			node.broadcastFlag = node.left.broadcastFlag
			reduceToScalar(node.left)
			if verbose: print(f"\tDelayed S2M broadcast for unary operation {node.name}: All relevant entries were identical.")

		# ... but only the diagonal is interesting
		if node.left.broadcastFlag == 'vector2diag':
			node.broadcastFlag = 'vector2diag'
			reduceToVector(node.left)
			if verbose: print(f"\tDelayed V2M broadcast for unary operation {node.name}: Only diagonal is relevant.")

	# Theoretical Case: sum using multiplication with dimensions
	# ... I have thus far been unable to think of a case where this would happen with our currently allowed inputs.
	#		If all entries inside are the same, we don't know the dimension.

	# Case: Sum or trace over diagonal matrix (can only be vector2diag because scalar2diag would have no certain dimension)
	elif node.name in ['sum', 'tr'] and node.left.broadcastFlag == 'vector2diag':
		node.name = 'sum' # In case it was trace
		node.broadcastFlag = 'scalar'
		node.left.broadcastFlag = 'vector'

	# ----- Binary Operation -----
	# Case: Elementwise operations ...
	# (For numpy, these will often work without any explicit broadcast at all. But that may not always be the case.)
	elif node.name in {'t+', 't-', './', 't/', '.^'} \
			or (node.name == 't*' and node.left.lower == node.right.lower and
            node.left.upper == node.right.upper): # This last one is the ".*" case, even though it doesn't look like it yet. Check the TensorTree constructor.

		# ... where both nodes are the same sort of "almost-scalar" (but not true scalars or nothing would need to be done).
		if node.left.broadcastFlag in ['scalar2covector', 'scalar2vector', 'scalar2matrix', 'scalar2diag'] \
				and node.left.broadcastFlag == node.right.broadcastFlag:

			# Make sure we're NOT talking about .^ with diagonal matrices, because there the zeros actually matter (0**0 == 1)
			if node.left.broadcastFlag != 'scalar2diag' or node.name != '.^':
				node.broadcastFlag = node.left.broadcastFlag
				reduceToScalar(node.left)
				reduceToScalar(node.right)
				if verbose: print(f"\tDelayed broadcast for binary operation {node.name}: All relevant entries were identical.")
				return

		# ... mult. where one node has the same scalar in every element (but the other doesn't) -> use scalar mult.
		scalarNode, otherNode = processByTags(['scalar2covector', 'scalar2vector', 'scalar2matrix'], node.left, node.right)
		if scalarNode and node.name == 't*':
			# Note: We know both have the same shape, so the other is marked 'covector' / 'vector' / 'matrix'.
			node.broadcastFlag = otherNode.broadcastFlag
			reduceToScalar(scalarNode)
			if verbose: print(
				f"\tExpressed elementwise multiplication as scalar multiplication: All relevant entries were identical.")
			return

		# ... where one node is 'none' but the other is a diagonal matrix.
		# TODO: With the way our optimization works, I can not propagate the information that I only need the diagonal
		# DOWNWARD in the tree. That would require a second flag ("trimFlag" or something) and a reverse pass.
		noneNode, otherNode = processByTags('none', node.left, node.right)
		if noneNode and node.name in ['t*'] and otherNode.broadcastFlag in ['scalar2diag', 'vector2diag']:
			# TODO: for some other operations (./, t/, .^) this also works in certain configurations. It still needs more
			# work, though, in particular becasue the TensorTree and the backends don't support all of what I'd like to do.
			#
			#if node.right == otherNode and node.name in ['./', 't/', '.^']:
			#	if verbose: print("Your calculation may divide by zero. Is this intended?")
			#	# Because there are multiply ways the user may want to handle division by zero (exception; nan; = 0; = 1)
			#	# There is no way I can know how to optimize this. Better leave it as it is.
			#elif node.name == '.^':
			#	pass # I haven't found anything useful I can do or say in this case.
			#else:

			# also Todo: is matrix -> extract diag -> make to diag matrix really better than eye.*matrix?
			# It needs fewer operations, but it may or may not perform worse on memory allocation.
			# Perhaps more importantly, though, this way the broadcast can be propagated further!

			# First of all, extract the diagonal from the matrix, because we need nothing else.
			diagWrapper = TensorTree("diag", left=noneNode, upper = noneNode.upper, lower = noneNode.lower)
			if noneNode == node.left:
				node.left = diagWrapper
			else:
				node.right = diagWrapper
			node.children.remove(noneNode)
			node.children.append(diagWrapper)
			diagWrapper.broadcastFlag = 'vector2diag' # Tell it to return to its original size, but as a diagonal matrix.

			if verbose:
				print(f"\tElementwise operation {node.name} where one matrix was diagonal was simplified to only use both diagonals.")

			# ... and that's it, actually, because we've now reduced the calculation to the case seen below.

		# ... where both nodes are diagonal matrices, but at least one with the 'vector2diag' flag (if both are 'scalar2diag' see above).
		# That means that, unless we're using .^, we can ignore anything outside of the diagonal.
		v2dNode, otherNode = processByTags('vector2diag', node.left, node.right)
		if v2dNode and otherNode.broadcastFlag in ['scalar2diag', 'vector2diag'] and node.name != '.^': # Again, .^ is an issue.
			if node.name in ['./', 't/'] and node.right == otherNode:
				if verbose: print("Your calculation may divide by zero. Is this intended?")
				# Because there are multiply ways the user may want to handle division by zero (exception; treat as 0, treat as 0/0 = 1, ...)
				# There is no way I can know how to optimize this. Better leave it as it is.
			else:
				node.broadcastFlag = 'vector2diag'
				reduceToVector(v2dNode)
				# Now squish the node as much as possible:
				if otherNode.broadcastFlag == 'scalar2diag':
					reduceToScalar(otherNode)
				else:
					reduceToVector(otherNode)

				if verbose:
					print(f"\tSimplified elementwise {node.name} of diagonal matrices as elementwise multiplication of their diagonals.")

	# Case: (Matrix) Multiplication ... this one is the most complicated and has a lot of different subcases.
	elif node.name in 't*':
		# Case: Either one is a *true* scalar ...
		scalarNode, otherNode = processByTags('scalar', node.left, node.right)
		if scalarNode:
			# ... and the other *can be represented by* a scalar:
			if basicallyAScalar(otherNode):
				node.broadcastFlag = otherNode.broadcastFlag
				#reduceToScalar(scalarNode) # Not required - it is already an actual scalar
				reduceToScalar(otherNode)
				if verbose: print("\tSimplified a scalar multiplication: No need to broadcast first.")

			# ... and the other is vector2diag
			if otherNode.broadcastFlag == 'vector2diag':
				node.broadcastFlag = 'vector2diag'
				#reduceToScalar(scalarNode) # Not required - it is already an actual scalar
				reduceToVector(otherNode)
				if verbose: print("\tSimplified a scalar multiplication: No need to broadcast first.")

		# Case: Multiplying two diagonal matrices ...
		elif node.left.broadcastFlag in ['scalar2diag', 'vector2diag'] \
				and node.right.broadcastFlag in ['scalar2diag', 'vector2diag']:

			# ... and one of the operands has unknown diagonal entries (solve as elementwise multiply for diagonal vectors)
			vectorNode, otherNode = processByTags(['vector2diag'], node.left, node.right)
			if vectorNode: # Is at least one of them a 'vector2diag'?
				node.broadcastFlag = 'vector2diag'
				reduceToVector(vectorNode)
				if otherNode.broadcastFlag == 'scalar2diag':
					otherNode.broadcastFlag = 'scalar2vector' # Only broadcast just as much as needed.
					otherNode.lower = []
					if otherNode.name == 'delta':
						otherNode.name = "Var_1" # There is no such thing as a delta vector in our backends.
				else: # Must also be 'vector2diag'
					reduceToVector(otherNode)
				node.name = ".*"
				if verbose: print("\tSimplified multiplication of diagonal matrices as elementwise multiplication of their diagonals.")

			# ... and all diagonal entries of each are the same:
			else:
				node.broadcastFlag = 'scalar2diag'
				reduceToScalar(node.left)
				reduceToScalar(node.right)
				if verbose: print(f"\tSimplified multiplication of diagonal matrices with the same entry everywhere.")

			return

		# Case: Multiplying a diagonal matrix with a non-diagonal matrix ...
		s2dNode, otherNode = processByTags('scalar2diag', node.left, node.right)
		if s2dNode and not otherNode.isScalar(): # I *think* the not scalar test is actually unnecessary because it was handled.
			node.broadcastFlag = otherNode.broadcastFlag
			reduceToScalar(s2dNode)

		# Case: covector * diagonal matrix (skip expanding the matrix)
		elif node.left.broadcastFlag == ['scalar2covector', 'covector'] and node.right.broadcastFlag in ['scalar2diag', 'vector2diag']:
			node.broadcastFlag = 'covector'
			reduceToCoVector(node.right)
			if node.left.broadcastFlag == 'covector':
				node.name  = ".*"
			else: # scalar2covector -> use scalar mult
				reduceToScalar(node.left)
			if verbose: print("\tSimplified multiplication of a covector and a diagonal matrix.")

		# Case: diagonal matrix * vector (skip expanding the matrix)
		elif node.right.broadcastFlag in  ['scalar2vector', 'vector'] and node.left.broadcastFlag in ['scalar2diag', 'vector2diag']:
			node.broadcastFlag = 'vector'
			reduceToVector(node.left)
			if node.right.broadcastFlag == 'vector':
				node.name  = ".*"
			else: # scalar2vector -> use scalar mult
				reduceToScalar(node.right)
			if verbose: print("\tSimplified multiplication of a diagonal matrix and a vector.")

		# Theoretical Case: Inner product between scalar2vector and scalar2covector
			# Cannot exist, because we could not know their length. At least I can't think of a case where that would work.
		# Same for scalar2(co)vector and scalar2matrix

		# Case: Inner product ...
		if node.left.broadcastFlag in ['covector', 'scalar2covector'] \
				and node.right.broadcastFlag in ['vector', 'scalar2vector']:

			# ... of two effective scalars (-> simply multiply them and the vector length)
			# I suspect that this case is not actually possible (once again because the dimensions would be unknown)
			if node.left.broadcastFlag == 'scalar2covector' and node.right.broadcastFlag == 'scalar2vector':
				node.broadcastFlag = 'reduce-multiply-child-dims'
				# Can't fully scalarize the children because their dimensions are needed in the calculation. Only set the flag.
				# Potential problem: this may confuse "isScalar()" checks.
				node.left.broadcastFlag = 'scalar'
				node.right.broadcastFlag = 'scalar'
				if verbose: print("\tSimplified inner product of constant vectors.")

			# ... of one effective scalar and an unknown vector (-> result is vector sum times that scalar)
			scalarableNode, otherNode = processByTags(['scalar2covector', 'scalar2vector'], node.left, node.right)
			if scalarableNode:
				node.broadcastFlag = 'scalar'
				reduceToScalar(scalarableNode)
				# Put a "sum" node in between this node and the vector child.
				sumNode = TensorTree("sum", left=otherNode)
				node.children.remove(otherNode)
				node.children.append(sumNode)
				if node.left == otherNode:
					node.left = sumNode
				else:
					node.right = sumNode
				if verbose: print("\tSimplified inner product where one vector was a constant using sum.")

			# ... of two arbitrary vectors, in which case we can do nothing.

		# Case: Outer product with the same result in every entry (-> only calculate once and broadcast afterwards)
		# TODO: Sadly, this may only trigger with correct brackets (consider A*vector(2)*vector(3)'*B)
		elif node.left.broadcastFlag == 'scalar2vector' and node.right.broadcastFlag == 'scalar2covector':
			node.broadcastFlag = 'scalar2matrix'
			reduceToScalar(node.left)
			reduceToScalar(node.right)
			if verbose: print("\tSimplified outer product: All entries are identical.")

		# TODO: outer product of all same value per row/col (will require vector stacking support in backend)

	# ----- Cleanup and Fallback -----:
	# If nothing interesting happened at all, we still need to set the flag to something before moving on.
	# This also catches cases that are disabled or not implemented yet.
	if not hasattr(node, "broadcastFlag"):
		if node.isScalar():
			node.broadcastFlag = 'scalar'
		elif node.isVector():
			node.broadcastFlag = 'vector'
		elif node.isCoVector():
			node.broadcastFlag = 'covector'
		else:
			node.broadcastFlag = 'none' # For example a matrix that we know nothing about.

	if debugOut:
		print("\tNode was assigned flag", node.broadcastFlag)

def simplifyOperation(node, verbose=False):
	"""
	Reduces the operation of this node by removing pointless calculations like
	multiplication with 1.
	Make sure to run broadcastProcessNoneLeaf(node) before this!

	This function is not recursive.
	It only looks far enough down the tree to see whether the operation of the
	CURRENT node can be simplified.

	It is also very deliberately NOT just using the simplification in the
	tensortree class (which seems pretty complicated, more geared towards
	logical simplicity than performance simplicity, and DOES have a recursive
	approach that is ill suited to what I do here and seems to traverse the
	tree much more than required. Also, It's hard for me to verify that it is
	indeed correct (although I suppose it must be)).
	Instead, this function will follow the single-application, modular and
	leave-everything-in-working-order philosophy of the broadcastOptimizer below.

	Note that those simplifications which operate with broadcastingFlags belong in the broadcastProcessNonLeaf function.
	This here is for cleanup afterwards, using the results of those operations. BroadcastingFlags no longer play a role.
	"""
	eliminated = None
	operation = node.name

	# Case: double transpose or scalar transpose
	if node.name == 'T':
		if node.left.name == 'T':
			node.setTo(node.left.left)
			if verbose: print("Eliminated double transpose")

	# Case: transpose before action that doesn't care
	elif node.name in ['diag', 'sum', 'norm1', 'norm2'] and node.left.name == 'T':
		eliminated = node.left
		operation = node.left.name
		node.left.setTo(node.left.left)

	# Case: Addition with neutral element.
	elif node.name in ["t+", "t-"]:  # not u-!
		if node.left.name == "Var_0":
			eliminated = node.left
			setToRetainDimension(node, node.right)
		elif node.right.name == "Var_0":
			eliminated = node.right
			setToRetainDimension(node, node.left)

	# Case: Elementwise Multiplication with neutral element.
	elif node.name in ["t*"] and node.left.upper == node.right.upper and node.left.lower == node.right.lower:
		if node.left.isOne():
			eliminated = node.left
			setToRetainDimension(node, node.right)
		elif node.right.isOne():
			eliminated = node.right
			setToRetainDimension(node, node.left)

	# Case: Elementwise Division/Power with neutral element on the right.
	elif node.name in ["./", ".^"] and node.right.name == "Var_1":
			eliminated = node.right
			setToRetainDimension(node, node.left)

	# Case: Multiplication with neutral element.
	elif node.name in ["t*"]:
		# Scalar 1 with anything
		if node.left.isScalar() and node.left.isOne():
			eliminated = node.left
			setToRetainDimension(node, node.right)
		elif node.right.isScalar() and node.right.isOne():
			eliminated = node.right
			setToRetainDimension(node, node.left)
		elif node.left.isScalar() and node.left.isDelta(): # scalar delta -> 1.0
			eliminated = node.left
			setToRetainDimension(node, node.right)
		elif node.right.isScalar() and node.right.isDelta():  # scalar delta -> 1.0
			eliminated = node.right
			setToRetainDimension(node, node.left)

		# matrix-mult. with eye:
		elif (node.left.upper != node.right.upper or node.left.lower != node.right.lower) and \
			not (node.left.isScalar() or node.right.isScalar()): # Neither is a scalar; the multiplication is not elementwise
			if node.left.isMatrix() and node.left.isDelta(): # And left is eye
				eliminated = node.left
				setToRetainDimension(node, node.right)
			elif node.right.isMatrix() and node.right.isDelta(): # And right is eye
				eliminated = node.right
				setToRetainDimension(node, node.left)

	if eliminated and verbose:
		print(f"Eliminated a neutral element in {operation}: {eliminated.name}.")

def optimizeBroadcasts(tree, verbose=False, debugOut = False):
	# Search the bottommost layer and save reverse pointers on the way.
	stage = reverseOrganizeTree(tree, None)

	if debugOut:
		print("Processing ", len(stage), " leaves")
	for node in stage:
		broadcastProcessLeaf(node, debugOut=debugOut)

	stage = getHigherLayer(stage)
	while stage:
		for node in stage:
			if all(hasattr(c, "broadcastFlag") for c in node.children):
				# Do not simplifyOperation() here! If you want to do it before, do it before optimizations first begin!
				broadcastProcessNonLeaf(node, verbose=verbose, debugOut=debugOut)
				simplifyOperation(node, verbose=verbose)
		stage = getHigherLayer(stage)
