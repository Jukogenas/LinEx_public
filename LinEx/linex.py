#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on So Nov 29 14:30 2020

Copyright:  Konstantin Wiedom (konstantin.wiedom@uni-jena.de)
            Paul Gerhardt Rump (paul.gerhardt.rump@uni-jena.de)

@author: Konstantin Wiedom, Paul Gerhardt Rump
@email: konstantin.wiedom@uni-jena.de, paul.gerhardt.rump@uni-jena.de
"""
import keyword # for marking python reserved words as variables
import time

from LinEx.equivalencyCache import EquivalencyCache
from LinEx.solution import Solution
from LinEx.cseCache import CSECache
from LinEx.kernelTree import KernelTree
from LinEx.calculation import Calculation
from LinEx.reshapeList import allReshapes
from LinEx.calculations_numpy import AllKernels as numpykernels
from expressions.expressiontree import ExpressionTree
from expressions.expressiontree import ExpressionForester as Reshaper


class LinExGenerator:
    def __init__(self, exptree, cse=True, kernels=numpykernels, reshapePreFilter=True, # Core Settings
                 preferFitting=True, preferCheap=True, pruningTolerance=0, # Run and heuristics settings
                 orientedVectors=True # Code settings
                 ):
        self.exptree = exptree
        self.finalTree = False
        self.cse = cse
        self.useSolutionCache = True # Maybe make it adjustable later. Although disabling it is silly.
        self.pruningTolerance = pruningTolerance
        self.orientedVectors = orientedVectors

        ## Set kernels, optionally sorted by (estimated) cheapest first:
        self.kernels = kernels
        if preferCheap:
            self.kernels = sorted(self.kernels, key=lambda kernel: kernel.getCostEstimate())
        self.preferFitting = preferFitting # This will require a stable re-sorting for every tree node.

        ## Set Reshapes:
        if reshapePreFilter:
            reshapes = LinExGenerator.reshapePrefilter(exptree, allReshapes)
        else:
            reshapes = allReshapes
        Reshaper.setAvailableReshapes(reshapes)
        
        # Prepare for (potentially) time-restricted generation:
        self.genStartTime = None
        self.genEndTime = float('inf')
        
        self.solutionCache = dict() # Mapping of (Hash of tree, sizes matter) -> Solution Object
        self.shapeCache = EquivalencyCache() # For specific (named) expTrees and their equivalencies; only within current run.
        # The CSECache is no longer a single object, there are versions that branch along with the possible solutions.

    def applyPythonVars(self, node):
        for c in node.children():
            self.applyPythonVars(c)
        if node.name.startswith('Var_') and not node.isNumeric():
            node.name += "_" if keyword.iskeyword(node.name[4:]) else ""

    @classmethod
    def reshapePrefilter(cls, tree, reshapeOptions, verbosity=2):
        """
        This is a potentially significant reduction of the search space, but not without cost.
        It doesn't consider reshapes that introduce new operations, like A+A -> 2 S* A, which introduces S*.
        """
        if verbosity > 2: print(f"Began reshape prefiltering with {len(reshapeOptions)} options.")
        # Find all operators present in the tree
        operatorSet = set()
        treeList = [tree]
        while treeList:
            sometree = treeList.pop()
            if not sometree.isLeaf():
                treeList += sometree.children()
                operatorSet.add(sometree.name)

        if verbosity > 3: print(
            f"\tTree had {len(operatorSet)} different operators.")
        # Keep only reshapes that have all their operators present somewhere.
        reshapes = []
        for option in reshapeOptions:
            option_operatorSet = set()
            treeList = [option.requires]
            while treeList:
                sometree = treeList.pop()
                if not sometree.isLeaf():
                    treeList += sometree.children
                    option_operatorSet.add(sometree.name)
            if option_operatorSet.issubset(operatorSet):
                reshapes.append(option)

        if verbosity > 2:
            print(f"Finished reshape prefiltering with {len(reshapes)} options (discarded {len(reshapeOptions)-len(reshapes)}).")
        return reshapes
    
    def generate(self, indent='    ', docString=True, forceVectorOrientation=True, # Code output settings
                 timeout=30, # Heuristics and runtime settings
                 verbosity: int = 2, commentindent = "" # Debug settings
                 ):
        """
        Timeout is in minutes (but you may enter fractions). Note that the program is not guaranteed to finish after
        *exactly* the given amount of time. It only checks for timeouts every once in a while.
        Also note that because of the way the program works and because much of the speed relies on caching (with
        initially empty caches for the most part), letting the program run for the fraction of the time required to find
        the best solution will not necessarily yield a solution that is proportionally good. In fact, early interrupts
        may force the program to terminate without having found any solution (in which case it will return the trivial
        solution that doesn't use any reshapes).
        """
        self.genStartTime = time.time()
        if timeout and timeout > 0:
            self.genEndTime = self.genStartTime + timeout * 60
    
        self.applyPythonVars(self.exptree)
        varPairs = sorted(self.exptree.allVars())
        vars = [v[1] for v in varPairs]

        solution = self.toKernelTree(self.exptree, verbosity=verbosity, indent=indent)
        if not solution.complete:
            solution = self.trivialSolution(self.exptree, verbosity=verbosity, indent=indent)
            CSECache.reset()
        
        cost, defBlock, codeBlock = solution.generateCode(orientedVectors=self.orientedVectors, codeIndent=indent)

        # Required definitions
        if defBlock:
            s = "\n"+ defBlock + "\n\n"
        else:
            s = ""
        # Function header
        s += f'''def rename_this_function({", ".join(vars)}):\n'''

        # Collect some data about variables for later:
        matrices, vectors, covectors, scalars = [], [], [], []
        for type, name in varPairs:
            if type == 'matrix':
                matrices.append(name)
            elif type == 'vector':
                vectors.append(name)
            elif type == 'covector':
                covectors.append(name)
            elif type == 'scalar':
                scalars.append(name)
            # Else should never happen; see allVars function in ExpressionTree.

        # Docstring
        if docString:
            s += f'{indent}"""\n'
            s += f'{indent}Generated with LinEx from input:\n{indent}{indent}{getattr(self.exptree, "inputString", "")}\n'

            if matrices: s += f'{indent}Matrices:\n{indent}{indent}{", ".join(matrices)}\n'
            if vectors: s += f'{indent}Column vectors:\n{indent}{indent}{", ".join(vectors)}\n'
            if covectors: s += f'{indent}Row vectors:\n{indent}{indent}{", ".join(covectors)}\n'
            if scalars: s += f'{indent}Scalars:\n{indent}{indent}{", ".join(scalars)}\n'

            s += f'{indent}Matching matrix and vector dimensions:\n'
            for dim in self.exptree.getDims():
                s += f"{indent}{indent}{str(dim)} == {dim.allNames()}\n"
            s += f'{indent}"""\n'

        if forceVectorOrientation:
            if self.orientedVectors:
                for v in vectors:
                    s += f"{indent}{v} = {v}.reshape(-1,1)\n"
                for c in covectors:
                    s += f"{indent}{c} = {c}.reshape(1,-1)\n"
            else:
                for v in vectors:
                    s += f"{indent}{v} = {v}.reshape(-1,)\n"
                for c in covectors:
                    s += f"{indent}{c} = {c}.reshape(-1,)\n"

        # And finally the actual code:
        s += codeBlock
        # TODO maybe display cost? On website?
        CSECache.reset() # TODO: Make the relevant things not class variables (without making the copying expensive).
        return s

    def toCodeDebug(self, tree: ExpressionTree, costLimit = float('inf'), timeout=1,
                    verbosity: int = 2, indent ="", graph=True):
        self.generationStart = time.process_time()
        self.solutionTrace = []
        

        self.genStartTime = time.time() # Note that this is not the same as the other time (this one uses system time, not process time).
        if timeout and timeout > 0:
            self.genEndTime = self.genStartTime + timeout * 60
        
        from testing.debugDrawer import DebugDrawer # Local import to avoid plantuml dependency for other functions.

        self.graph = DebugDrawer()
        if graph:
            graph="youAreTheTrueRoot"

        solution = self.toKernelTree(tree, costLimit=costLimit,
                                     verbosity=verbosity, indent=indent, tracing=True, graph=graph)
        if not solution.complete:
            if verbosity > 1: "Failed to find any complete solution within the time and cost limit. Resorting to trivial solution without reshapes."
            CSECache.reset()
            solution = self.trivialSolution(tree, tracing=True)

        self.finalTree = solution.kernelTree  # For display on website if desired. No overhead if not used for that.
        res = solution.generateCode(orientedVectors=self.orientedVectors)

        # TODO: Handle output in case of multiline code! (With indentation, I guess.)
        print()
        print(indent+"Here's a summary of the process (top-level kernels and reshapes only; repetitions aren't a sign of error!):")
        print(indent+"\t\tTime (s)\tCost\t\t\tCode")
        for i, s in enumerate(self.solutionTrace):
            print(indent+f"{i+1}\t\t{s[0]:.2f}\t\t{s[1]}\t\t{s[2]}")
        print()

        # These two solely exists so I can get at them quickly with the debugger:
        debugShortcutRSCache = Reshaper.reshapeCache
        debugShortcutDotStr = self.graph.fullDotString({"ratio": "auto"})

        CSECache.reset() # TODO: Make the relevant things not class variables (without making the copying expensive).
        return res, self.graph.dotURL({"ratio": "auto"}) #"layout": "neato" (also try twopi)

    def applyCSE(self, treeRep: ExpressionTree, cseCache: CSECache, verbosity: int = 2, indent ="", graph ="") -> Solution:
        _, solution, cost = cseCache.increment(treeRep)
        solution.cost = cost # Since I just recalculated it, anyway ...
        assert solution.isValid()

        count = cseCache.countOf(treeRep)
        # If the tree was already part of this current solution, use it.
        if count > 1:
            if verbosity > 2: print(f"{indent}It is possible to eliminate {str(treeRep)} as a common subexpression.")
            # DEBUG: Write some hints as sanity checks while debugging (they are not used in the program):
            solution.kernelTree.cseHint = True
            treeRep.cseHint = True

        # Either way, point to where the solution was most recently calculated (previous branch if count == 1, else
        # our own branch).
        if graph:
            # Hint to wherever this was most recently calculated.
            destination = cseCache.getGraphHint(treeRep)
            cseCache.setGraphHint(treeRep, graph) # This is what the next solutions will point to.
            color = "aquamarine4"
            label = f"CSE ({count-1})"
            self.graph.addEdge(graph, destination, {"color": color, "style": "dashed", "label": label})
            solution.graphRoot = cseCache.getGraphHint(treeRep) # In case I ever need it elsewhere.

        return solution

    def trivialSolutionAlternatice(self, tree: ExpressionTree, cseCache: CSECache = None,
                           verbosity: int = 2, indent="", tracing=False) -> Solution:
        # Clear things that would get in the way (but restore them later in case I want to use them):
        reshapes = Reshaper.availableReshapes
        Reshaper.setAvailableReshapes([])
        endTime = self.genEndTime
        self.genEndTime = None
        shapeCache = self.shapeCache
        self.shapeCache = EquivalencyCache()
        
        solution = self.toKernelTree(tree, cseCache=cseCache, verbosity=verbosity, indent=indent, tracing=tracing)
        
        Reshaper.setAvailableReshapes(reshapes)
        self.genEndTime = endTime
        self.shapeCache = shapeCache
        
        return solution

    def trivialSolution(self, tree: ExpressionTree, cseCache: CSECache = None,
                        verbosity: int = 2, indent="", tracing=False) -> Solution:
        """
        Calculates the trivial solution (the one that doesn't use reshapes).
        """
        
        childIndent = f"{indent}│\t"
        
        if self.cse and cseCache is None:
            cseCache = CSECache(self.shapeCache)
    
        # Catch leaves right here (end of recursion):
        if tree.isLeaf():
            if verbosity > 3: print(f"{indent}Encountered a leaf: {str(tree)}")
            # A leaf needs no kernel; the 'leaf' comment could be anything:
            return Solution(expTree=tree, kernelTree=KernelTree(kernel='leaf'), cseCache=CSECache(self.shapeCache),
                            cost=0).markComplete()
        elif verbosity > 2:
            print(f"{indent}Calculating trivial solution for {str(tree)}")
    
        # CSE: If the tree is an exact match (incl. varnames) to another in my expression, I drop in the original solution.
        treeRep = self.shapeCache.getRepresentative(tree)
        if self.cse and treeRep in cseCache:
            solution = self.applyCSE(treeRep, cseCache, graph='')
            return solution  # This cost is impossible to beat; no need to keep searching.
    
        # Or just use the first fitting kernel  (pre-filtering is optional; might help a little bit) ...
        # TODO: One issue here is that solve is a kernel for @ (involving inv), and is more expensive than a pure @.
        #  Consequently, the preferred kernel will be @ + explicit inv later, which is not ideal at all.
        kernels = filter(lambda kernel: kernel.requires.name == tree.name, self.kernels)
        for kernel in kernels:
            if not kernel.matches(tree):
                continue
        
            solutionTree = KernelTree(kernel=kernel)
        
            # Calculate child solutions:
            for key, child in kernel.getChildTrees(tree).items():
                childSolution = self.trivialSolution(child, cseCache=cseCache, verbosity=verbosity, indent=childIndent)
                cseCache = childSolution.cseCache  # Update CSE info
                solutionTree[key] = childSolution.kernelTree
            
                cseCache = childSolution.cseCache  # Update with child info (if not self.cse, this will simply reference None).
            
                if not childSolution.complete:
                    if verbosity > 3: print(f"{indent}Failure: Child {str(child)} failed.")
                    return Solution(expTree=tree, cseCache=cseCache, cost=float('inf')).markFailed(
                        childSolution.failReason)
        
            solution = Solution(expTree=tree, kernelTree=solutionTree, cseCache=cseCache, cost=0)
            solution.markComplete()
            solution.recalculateCost()
        
            assert solution.isValid()  # DEBUG
        
            if tracing:
                self.solutionTrace.append(
                    (time.process_time() - self.generationStart, solution.cost, solution.generateCode()[2]))
        
            if self.cse and cseCache.countOf(treeRep) == 0:
                cseCache.register(solution)

            CSECache.reset()
            return solution
        else:
            CSECache.reset()
            # There was no fitting kernel? Then I guess we have no solution.
            return Solution(tree, cost=float('inf')).markFailed("Initial Non-solution")
        
    def toKernelTree(self, tree: ExpressionTree, parentShape: ExpressionTree = None,
                     cseCache: CSECache = None, costLimit = float('inf'),
                     verbosity: int = 2, indent = "", tracing=False, graph="", graphEdge="") -> Solution:
        """
        Returns (cost, KernelTree, ExpTree that fits KernelTree), so make sure to unwrap it!
        """
        childIndent = f"{indent}│\t"

        tree = tree.rootSimplify() # Avoid working on a tree in which a part can be remade into the whole.
        
        if self.cse and cseCache is None:
            cseCache = CSECache(self.shapeCache)
        if tree == parentShape:
            return Solution(expTree=tree, cost=float('inf')).markFailed("Would have been a miniature loop.")

        # Add to the debug graph for visual analysis:
        if graph == 'youAreTheTrueRoot':
            graph = self.graph.nodeFromExpression(tree)
        elif graph:
            graph = self.graph.nodeFromExpression(tree, graph, edgeLabel=graphEdge)

        # Catch leaves right here (end of recursion):
        if tree.isLeaf():
            if verbosity > 3: print(f"{indent}Encountered a leaf: {str(tree)}")
            # A leaf needs no kernel; the 'leaf' comment could be anything:
            return Solution(expTree=tree, kernelTree=KernelTree(kernel='leaf'), cseCache=cseCache, cost=0).markComplete()
        elif verbosity > 2: print(f"{indent}Searching solution for {str(tree)}")

        # CSE: If the tree is an exact match (incl. varnames) to another in my expression, I drop in the original solution.
        # I filter transposing because it is always cheaper (at least in numpy) than copying a matrix. So why even bother.
        if self.cse and not cseCache.isUndesirable(tree) and tree in cseCache:
            # NOTE: Getting here only means we've seen this tree before. It may or may not still exist in the same
            # solution 'branch' where we are.
            solution = self.applyCSE(tree, cseCache, graph=graph)
            return solution # This cost is impossible to beat; no need to keep searching.

        # Alternatively, if I ever (even in another 'branch' of the solver) encountered a matching tree with the same
        # variable dimensions and properties (not necessarily names) I can use that original solution as a guaranteed best:
        sizedHash = tree.expStrShaped()
        if sizedHash in self.solutionCache.keys() and self.useSolutionCache:
            # TODO: Generalize this cache by applying the shapeCache first. Would need ability to retarget trees.
            #  That wouldn't be too difficult, though: The rep and the tree in the solution share a shape, and the rep
            #  and the input tree share variables. Use together to retarget the solution tree to the current vars.
            # Without this change, the cache only recognizes trees that are in their best shape.
            if verbosity > 2: print(f"{indent}Used cached code for {str(tree)}")
            solution = self.solutionCache[sizedHash]

            if self.cse:
                # Need to inform the solution of the new (potentially changed) CSE situation:
                solution = solution.copy4Cache(newExpTree=tree, cseCache=cseCache)
                # Also, need to inform the cache that it can now use this solution elsewhere:
                if solution.expTree not in cseCache:
                    cseCache.register(solution, recursive=True)
                    if graph:
                        cseCache.setGraphHint(solution.expTree, graph)
                # If the solution WAS in the cseCache, then the cseCache has already handled it and we never should have
                #  gotten here.
            else:
                solution.expTree = tree # Only need to swap out leaf names. No need to copy in this case (there is no CSECache to confuse).
            assert solution.isValid() # DEBUG
            

            if graph:
                self.graph.addEdge(graph, solution.graphRoot, {"color": "aquamarine3", "style": "dashed", "label": "Cached"})
            return solution
        else:
            # In this case, we have nothing to go on, so we start with the triviel solution (no reshapes):
            bestSoFar = Solution(tree, cost=float('inf')).markFailed("Initial Non-solution")

        # TODO: Support for kernels that write the solution into one of the inputs. Would possibly need to copy.

        # TODO: constant folding. If a tree is marked as constant, just
        #  use the first encountered solution and evaluate it. Cost 0 (ish).
        #  Possibly make a "to executable" function here. Can also be used in tests.
        #  Difficulty: Inputting the solution in code in cases where it isn't a pseudo_scalar.

        # TODO: I could introduce special cases to deal with puer + or M* trees without a full search, but integrating
        #  that with CSE, pruning etc. would be difficult.
        # Try all kernels.
        bestSoFar = self.__tryAllKernels(tree, bestSoFar, tree, cseCache, costLimit, verbosity, indent, tracing, graph, graphEdge)
        cseCache = bestSoFar.cseCache # Update with child counts.

        # We have tried every possible solution within the cost limit. Whatever we have now is the best we'll get.
        if not bestSoFar.complete:
            if verbosity > 2: print(f"{indent}Found no solution within the cost/time limit.")
        else:
            assert bestSoFar.isValid() # DEBUG - I check in the relevant other places, so it *should* never break here.
            # Cache the solution for this tree (and all with the same shapes):
            self.solutionCache[bestSoFar.expTree.expStrShaped()] = bestSoFar
            if self.cse and bestSoFar.expTree not in cseCache:
                cseCache.register(bestSoFar)
                if graph:
                    cseCache.setGraphHint(bestSoFar.expTree, graph)
            if verbosity > 2: print(f"{indent}Calculated best possible solution for {str(tree)} (Cost {bestSoFar.cost}).")

        return bestSoFar
    
    def __tryAllKernels(self, tree: ExpressionTree, bestSoFar: Solution, parentTree: ExpressionTree,
                        cseCache: CSECache = None, costLimit = float('inf'),
                        verbosity: int = 2, indent = "", tracing=False, graph="", graphEdge="") -> (Solution, set):
        """
        Goes through all available kernels and all ways to make them fit to the given tree to find the best solution.
        """
        childIndent = f"{indent}│\t"
        
        kernels = self.kernels
        if self.preferFitting:
            kernels = sorted(self.kernels, key=lambda kernel: 0 if kernel.requires.name == tree.name else 1)
    
        thereWasAKernel = False
        for kernel in kernels:
            if verbosity > 2: print(f"{indent}Trying to apply {str(kernel)}.")
            thereWasAWay = False
            for treeShape in Reshaper.tryModifyTree(tree, kernel.requires, shapeCache=self.shapeCache,
                                                    forceEndTime=self.genEndTime,
                                                    verbosity=verbosity, indent=childIndent):
                if not kernel.matches(treeShape, verbosity=0, indent=childIndent):
                    if verbosity > 3: print(
                        f"Skipping tree shape {str(tree)}: Input variables did not match the required pattern.")
                    continue  # This can happen (see tryModifyTree description string for the reason)
                    
                thereWasAKernel = True
                solution = self.__useKernel(treeShape, kernel, parentTree,
                                            cseCache=cseCache.softCopy() if self.cse else None, costLimit=costLimit,
                                            verbosity=verbosity, indent=childIndent, tracing=tracing, graph=graph)
                
                if solution.complete:
                    assert solution.isValid()  # DEBUG
                    if solution.cost < bestSoFar.cost:
                        thereWasAWay = True
                        costLimit = solution.cost
                        bestSoFar = solution
                
            if verbosity > 2:
                if thereWasAWay:
                    print(f"{indent}Conclusion: Kernel applicable.")
                else:
                    print(f"{indent}Conclusion: Kernel NOT applicable.")
    
        if not thereWasAKernel:
            if time.time() >= self.genEndTime:
                if tracing:
                    self.solutionTrace.append((time.process_time() - self.generationStart, "Time limit exceeded!", ' '))
                if not bestSoFar.complete:
                    bestSoFar.markFailed("Time limit exceeded!")
                if verbosity > 1: print(f'Ran out of time while solving {tree}!')
            elif verbosity > 1: print(f'[ERROR] Missing Kernel: There was no way to solve {str(tree)}!')
            
        return bestSoFar

    def __useKernel(self, tree: ExpressionTree, kernel: Calculation, parentTree: ExpressionTree,
                    cseCache:CSECache, costLimit = float('inf'),
                    verbosity: int = 3, indent="", tracing = False, graph="") -> Solution:
        childIndent = f"{indent}│\t"

        childTrees = kernel.getChildTrees(tree)
        childShapesNumeric = dict()
        
        # Add pruning tolerance: We explore slightly more expensive calculations hoping that CSE will offset the cost.
        pruningTolerance = self.pruningTolerance
        if 'cse_cost' == pruningTolerance and self.cse:
            pruningTolerance = tree.copyCost()
        elif pruningTolerance == '-full': # Set's costLimit to 0. For debugging purposes.
            pruningTolerance = - costLimit
        elif pruningTolerance == '-half':
            pruningTolerance = - costLimit/2 # Set's costLimit to 1/2 the usual. For debugging purposes.
        elif isinstance(pruningTolerance, str):
            if verbosity > 2: print(f"Unkown pruning tolerance: {pruningTolerance}")
            pruningTolerance = 0
                    

        # Evaluate the cost of the current kernel:
        for key, child in childTrees.items():
            childShapesNumeric[key] = child.numericSize()
        cost = kernel.getCost(childShapesNumeric)

        parentGraph = graph
        if graph:
            graph = self.graph.nodeFromKernel(kernel, graph, edgeLabel=str(cost))

        solutionTree = KernelTree(kernel=kernel)
        childTreesModified = dict()

        # Calculate best child solution:
        for key, child in childTrees.items():
            # Abort if the solution would be no improvement:
            if cost > costLimit + pruningTolerance:
                if verbosity > 3: print(f"{indent}Abandoned potential solution: Cost {cost} exceeded the set limit.")
                # I don't know the total overall cost in here - but it would be useful for tracing ...
                if tracing:
                    self.solutionTrace.append((time.process_time() - self.generationStart, '>'+str(cost), 'Aborted: Cost too great!'))
                if graph:
                    self.graph.nodeFromString(f"Aborted: Too expensive! (>{cost})", graph, edgeLabel=key)
                return Solution(expTree=tree, kernelTree=solutionTree, cseCache=cseCache, cost=cost).markFailed(f"Cost exceeded ({cost})")

            childSolution = self.toKernelTree(child, parentTree, cseCache=cseCache, costLimit=costLimit - cost,
                                              verbosity=verbosity, indent=childIndent, graph=graph, graphEdge=key)
            cost += childSolution.cost
            solutionTree[key] = childSolution.kernelTree
            childTreesModified[key] = childSolution.expTree

            cseCache = childSolution.cseCache  # Update with child info (if not self.cse, this will simply reference None).

            if not childSolution.complete:
                if verbosity > 3: print(f"{indent}Failure: Child {str(child)} failed.")
                return Solution(expTree=tree, cseCache=cseCache, cost=cost).markFailed(childSolution.failReason)

        modifiedTree = kernel.reconstructExpTree(childTreesModified)
        solution = Solution(expTree=modifiedTree, kernelTree=solutionTree, cseCache=cseCache, cost=cost).markComplete()
        assert solution.isValid() # DEBUG
        solution.graphRoot = parentGraph

        # DEBUG help - TODO: Disable this, the graph is more informative, anyway:


        solution.recalculateCost() # TODO: This here is a cheat; remove once properly fixed:

        if tracing:
            # TODO: Tracing only captures the outermost kernels (but all used shapes for those).
            assert solution.costsMatch()
            self.solutionTrace.append((time.process_time() - self.generationStart, cost, solution.generateCode()[2]))

        return solution
