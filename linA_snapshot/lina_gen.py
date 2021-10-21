#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on So Nov 29 14:30 2020

Copyright:  Konstantin Wiedom (konstantin.wiedom@uni-jena.de)
            Paul Gerhardt Rump (paul.gerhardt.rump@uni-jena.de)

@author: Konstantin Wiedom, Paul Gerhardt Rump
@email: konstantin.wiedom@uni-jena.de, paul.gerhardt.rump@uni-jena.de
"""

from linA_snapshot.tree_processor import splitSubtrees
from linA_snapshot.tree_processor import optimizeBroadcasts
from linA_snapshot.dimension import merge0

# for marking python reserved words as variables
import keyword

class NumpyGenerator():
    def __init__(self, exptree, cse=True, optimize=True, orientedVectors=True, verbose=True):
        self.exptree = exptree
        self.inputString = getattr(exptree, "inputString", '')
        self.dimTable = self.exptree.inferDimension() # must be done BEFORE optimization
        if optimize:
            optimizeBroadcasts(self.exptree, verbose=verbose)
        self.exptree = self.exptree.expressionTree()
        self.exptree.prepareForEval()
        self.cse = cse
        self.optimize = optimize
        self.orientedVectors = orientedVectors
        self.additionalDefinitions = {"import numpy as np": "import numpy as np\n"}

        op1Prefix = {
            'exp': 'np.exp',
            'log': 'np.log',
            'sin': 'np.sin',
            'cos': 'np.cos',
            'tan': 'np.tan',
            'arcsin': 'np.arcsin',
            'arccos': 'np.arccos',
            'arctan': 'np.arctan',
            'tanh': 'np.tanh',
            'diag': 'np.diagflat' if self.orientedVectors else 'np.diag',
            'diag2': 'np.diagonal+Reshape' if self.orientedVectors else 'np.diag', # siehe unten
            'sum': 'np.sum',
            'tr': 'np.trace',
            'inv': 'np.linalg.inv',
            'softmax': 'softmax',
            'det': 'np.linalg.det',
            # 'norm1': 'np.linalg.norm',
            # 'norm2': 'np.linalg.norm',
            # 'relu': 'np.maximum',
            'abs': 'np.abs',
            'sign': 'np.sign',
        }
        self.op1PrefixKeys = list(op1Prefix)

        op2Prefix = {
            'M*': 'np.dot',
            'O*': 'np.dot' if self.orientedVectors else 'np.multiply.outer',
            #'^': 'np.linalg.matrix_power'
        }
        op2Infix = {
            '+': '+',
            '-': '-',
            'S*': '*',
            '.*': '*',
            'S/': '/',
            './': '/',
            #'^': '**',
            '.^': '**'
        }
        # maybe make lookup on fname,left.type,right.type
        self.lookup = {
            "T": lambda left: f"{self.toPyStr(left,4)}.T" if left.isMatrix() or orientedVectors and not left.isScalar() else f'{self.toPyStr(left,4)}',
            "u-": lambda left: f"-{self.toPyStr(left,4)}",
            "norm1": lambda left: f"np.sum(np.abs({self.toPyStr(left)}))" if left.isMatrix() else f"np.linalg.norm({self.toPyStr(left)}, 1)",
            "norm2": lambda left: f"np.linalg.norm({self.toPyStr(left)}" + (", 'fro')" if left.isMatrix() else ')'),
            "relu": lambda left: f"np.maximum({self.toPyStr(left)}, 0)",
        }
        self.lookup.update({k: self.op1PrefixFun(v) for (k, v) in op1Prefix.items()})
        self.lookup.update({k: self.op2PrefixFun(v) for (k, v) in op2Prefix.items()})
        self.lookup.update({k: self.op2InfixFun(v) for (k, v) in op2Infix.items()})
        self.lookup["M*"] = lambda left, right: f"{self.toPyStr(left,4)}.dot({self.toPyStr(right)})"
        self.lookup["inv"] = lambda left: f"np.linalg.solve({self.toPyStr(left)},np.eye({self.dimString(left)[0]}))"
        if self.orientedVectors:
            self.lookup["diag2"] = lambda left: f"np.diagonal({self.toPyStr(left)}).reshape((-1,1))"
        def power(left,right):
            if (left.isMatrix() and right.isScalar()):
                return f"np.linalg.matrix_power({self.toPyStr(left)},{self.toPyStr(right)})"
            elif (left.isScalar() and right.isScalar()):
                return f"{self.toPyStr(left)}**{self.toPyStr(right,3)}"
            raise ValueError("^ is only used for Matrix^Scalar or Scalar^Scalar; use .^ for everything else")
        self.lookup["^"] = power
        
        def softmax(left):
            self.additionalDefinitions["softmax"]=\
'''def softmax(X):
{indent}"""
{indent}Calculates the rowwise softmax and adds minimal constant to make log() safe.
{indent}"""
{indent}e_x = np.exp(X - np.repeat(np.amax(X, axis=1)[:, np.newaxis], X.shape[1], axis=1))
{indent}repeated_row_sums = np.repeat(np.sum(e_x, axis=1)[:, np.newaxis], X.shape[1], axis=1)
{indent}return e_x / repeated_row_sums + np.finfo(X.dtype).eps
'''
            return f"softmax({self.toPyStr(left)})"
        self.lookup["softmax"]=softmax

        def v_softmax(left):
            self.additionalDefinitions["v_softmax"]=\
'''def v_softmax(v):
{indent}"""
{indent}Calculates the softmax of a vector and adds minimal constant to make log() safe.
{indent}"""
{indent}e_x = np.exp(v - np.max(v))
{indent}return e_x / np.sum(e_x) + np.finfo(v.dtype).eps
'''            
            return f"v_softmax({self.toPyStr(left)})"
        self.lookup["v_softmax"]=v_softmax

    def applyPythonVars(self, node):
        for c in node.children:
            self.applyPythonVars(c)
        if node.name.startswith('Var_') and not node.isNumeric():
            node.name = f'Var_{self.markPythonVars(node.name[4:])}'

    def markPythonVars(self, v):
        return v + '_' if keyword.iskeyword(v) else v

    def generate(self, indent='    ', docString=True, inline=False, forceVectorOrientation=True):
        self.applyPythonVars(self.exptree)
        vars = sorted(self.exptree.allVars())
        indent2 = indent
        if not inline:
            dims = sorted(map(sorted, merge0(self.dimTable.values())))
            s = f'''def rename_this_function({", ".join(vars)}):\n''' # add , **args for simplifyed testbench
            if docString:
                s += f'{indent}"""\n'
                s += f'{indent}Generated with LinA from input:\n{indent}{indent}{self.inputString}\n'

                matrices, vectors, scalars = [], [], []
                for v in vars:
                    if 'a' <= v[0] <= 'g': scalars.append(v)
                    if 'h' <= v[0] <= 'z': vectors.append(v)
                    if 'A' <= v[0] <= 'Z': matrices.append(v)

                if matrices: s += f'{indent}Matrices:\n{indent}{indent}{", ".join(matrices)}\n'
                if vectors: s += f'{indent}Vectors:\n{indent}{indent}{", ".join(vectors)}\n'
                if scalars: s += f'{indent}Scalars:\n{indent}{indent}{", ".join(scalars)}\n'

                equalDims = [dimList for dimList in dims if len(dimList)>=2]
                if(equalDims): 
                    s += f'{indent}Matching matrix and vector dimensions:\n'
                    for dimList in equalDims:
                        dimList = [dim[:-5] + ('.shape[0]' if dim[-5:] == '_rows' else '.shape[1]') for dim in dimList]
                        s += indent + indent + " == ".join(dimList) + "\n"
                s += f'{indent}"""\n'
        else: s, indent = '', ''

        if forceVectorOrientation:
            vectors = [v for v in vars if 'h' <= v[0] <= 'z']
            marker = '1' if self.orientedVectors else ''
            if vectors: 
                for v in vectors:
                    s += f"{indent}{v} = {v}.reshape(-1,{marker})\n"

        returnOrResult = 'rename_this_variable =' if inline else 'return'
        if self.cse:
            subtrees = splitSubtrees(self.exptree)
            for var, tree in subtrees[:-1]:  # handle result later
                s += f'''{indent}{var} = {self.toPyStr(tree)}\n'''
            s += f'''{indent}{returnOrResult} {self.toPyStr(subtrees[-1][1])}'''
        else:
            s += f'''{indent}{returnOrResult} {self.toPyStr(self.exptree)}'''

        if self.additionalDefinitions:
            s = "\n".join(v.format(indent=indent2) for v in self.additionalDefinitions.values()) +"\n"+ s
        return s

    def dimString(self, t):  # generate A.shape[0] from A_rows
        dim0 = next(iter(self.dimTable[t.upper[0]]), None) if t.upper else '1'
        dim1 = next(iter(self.dimTable[t.lower[0]]), None) if t.lower else '1'
        if dim0 != '1': dim0 = dim0[:-5] + ('.shape[0]' if dim0[-5:] == '_rows' else '.shape[1]')
        if dim1 != '1': dim1 = dim1[:-5] + ('.shape[0]' if dim1[-5:] == '_rows' else '.shape[1]')
        return (dim0, dim1)

    def op1PrefixFun(self, fname):
        return lambda left: f"{fname}({self.toPyStr(left)})"

    def op2PrefixFun(self, fname):
        return lambda left, right: f"{fname}({self.toPyStr(left)},{self.toPyStr(right)})"

    def op2InfixFun(self, fname):
        binding = {
            '+':0,'-':0,'*':1,'/':1,'**':2
        }.get(fname, fname)
        extra = fname in ('-','/','**')
        return lambda left, right: f"{self.toPyStr(left,binding)}{fname}{self.toPyStr(right,binding+extra)}"

    def needsBraces(self, name, level):
        if name.startswith('Var_'):
            return False
        elif name in self.op1PrefixKeys or name in ('norm1','norm','relu','M*','O*','T','u-'):
            return False
        elif name in '+-':
            return level >= 1
        elif name in ('.*','S*','./','S/'):
            return level >= 2
        elif name in ('.^','^'):
            return level >= 3
        else:
            return level >= 4

    def broadcast(self, node):
        (dim0, dim1) = self.dimString(node) # If you encounter a KeyError here, optimizations probably removed something that was important to inferdimensions. Again.
        flag = node.broadcastFlag
        node.broadcastFlag = 'none' # Prevent getting into this again (endless recursion)

        # Now broadcast:
        if flag == 'scalar2vector':
            if self.orientedVectors:
                return f"np.full(({dim0}, 1), {self.toPyStr(node)})"
            else:
                return f"np.full({dim0}, {self.toPyStr(node)})"

        elif flag == 'scalar2covector':
            if self.orientedVectors:
                return f"np.full((1, {dim1}), {self.toPyStr(node)})"
            else:
                return f"np.full({dim1}, {self.toPyStr(node)})"

        elif flag == 'scalar2matrix':
            return f"np.full(({dim0}, {dim1}), {self.toPyStr(node)})"

        elif flag == 'scalar2diag':
            substring = self.toPyStr(node)
            if substring == '1':
                return f"np.eye({dim0})"
            elif substring.startswith("np.eye("): # Apparently it was never "squished down" in the first place.
                return substring
            else:
                return f"np.diag(np.full({dim0}, {substring}))"

        elif flag == 'vector2diag':
            return f"np.diagflat({self.toPyStr(node)})"

        elif flag == 'reduce-multiply-child-dims':
            if node.isUnary():
                (childdim0, childdim1) = self.dimString(node.left)
                return f"({childdim0}*{childdim1}*{self.toPyStr(node.left)})"
                # Todo: These brackets may be unecessary in some cases.
            elif node.isBinary():
                (_, ldim1) = self.dimString(node.left)
                (rdim0, _) = self.dimString(node.right)
                if ldim1 == rdim0:
                    return f"({ldim1}*{self.toPyStr(node)})"
                else:
                    print(f"[Warning] Performed 'reduce-multiply-child-dims' on binary node, but dimensions don't match: {node.name}.")
            else:
                print(f"[Warning] Performed 'reduce-multiply-child-dims' broadcast on an unexpected node: {node.name}.")


    def toPyStr(self, node, binding=0):
        if self.optimize:
            # Avoid broadcasts in child nodes that numpy handles automatically:
            if node.name in ['+', '-', '.*', './']:
                for n in node.children:
                    if n.broadcastFlag in ['scalar2covector', 'scalar2vector', 'scalar2matrix']:
                        # 'scalar2diag' must NOT be removed. Consider A + eye != A + 1 (calculated as A + matrix(1))
                        # Also note: The case that both operands will be reduced (breaking the dimension) should never
                        # occur, because in those cases the optimization would already have propagated up the tree.
                        n.broadcastFlag = 'scalar'

            # Broadcast this node:
            if hasattr(node, "broadcastFlag") and node.broadcastFlag not in ['none', 'scalar', 'vector', 'covector']:
                return self.broadcast(node) # broadcast current node if required

        if not node.children:
            if node.name.startswith('Var_'):
                (dim0, dim1) = self.dimString(node)
                if self.optimize or node.isScalar() or not node.isNumeric(): # If we optimize, that code will deal with broadcasts.
                    return node.name[4:]
                if node.isVector():
                    if self.orientedVectors:
                        return f"np.full(({dim0}, 1), {node.name[4:]})"
                    else:
                        return f"np.full({dim0}, {node.name[4:]})"
                if node.isCoVector():
                    if self.orientedVectors:
                        return f"np.full((1, {dim1}), {node.name[4:]})"
                    else:
                        return f"np.full({dim1}, {node.name[4:]})"
                if node.isMatrix(): return f"np.full(({dim0}, {dim1}), {node.name[4:]})"
            elif node.isDelta():
                if node.isScalar(): return '1.0'
                elif node.isMatrix():
                    return f'np.eye({self.dimString(node)[0]})'
                # Vectors filled with 1 may later also be required because of broadcasting:
                #elif node.isVector():
                #    if self.orientedVectors:
                #        return f"np.ones((1, {self.dimString(node)[0]}))"
                #    else:
                #        return f"np.ones({self.dimString(node)[0]})"
            assert(False)
        if node.isUnary():
            res = self.lookup[node.name](node.left)
            return f'({res})' if self.needsBraces(node.name,binding) else res
        if node.isBinary(): 
            res = self.lookup[node.name](node.left, node.right)
            return f'({res})' if self.needsBraces(node.name,binding) else res