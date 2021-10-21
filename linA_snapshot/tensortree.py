# -*- coding: utf-8 -*-
"""
Copyright: Soeren Laue (soeren.laue@uni-jena.de)

The core TensorTree module.
"""

import numbers
import copy
import re

class TensorTree:
    def __init__(self, name, left = None, right = None,
                 upper = [], lower = [], attributes = None):
        self.name = name
        self.upper = upper.copy()
        self.lower = lower.copy()
        self.attributes = attributes if attributes else set()
        self.left = left.copy() if left else None
        self.right = right.copy() if right else None
        self.children = []
        if self.left:
            self.children += [self.left]
        if self.right:
            self.children += [self.right]
        processed = False

        if self.name in ['delta', 'deltaT']:
            processed = True

        if self.name == 'Var_eye':
            self.name = 'delta'
            processed = True

        if self.name.startswith('Var_'):
            processed = True

        if self.name == 't+' or self.name == 't-':
            assert(self.isBinary())
            assert(set(self.left.upper) == set(self.right.upper))
            assert(set(self.left.lower) == set(self.right.lower))

            self.upper = self.left.upper.copy()
            self.lower = self.left.lower.copy()
            processed = True

        if self.name in ['t*', 't/']:
            assert(self.isBinary())
            processed = True

        #IMPORTANT, because not all u- are detected properly!
        if self.name == '-' and self.isUnary():
            self.name = 'u-'

        if self.name in ['+', '-', '.*', './']:
            if self.isBinary():
                if  (len(self.left.upper) == len(self.right.upper)) and \
                    (len(self.left.lower) == len(self.right.lower)):

                    matchIndex(self.left, self.left.upper, self.right, self.right.upper)
                    matchIndex(self.left, self.left.lower, self.right, self.right.lower)

                    self.upper = self.left.upper.copy()
                    self.lower = self.left.lower.copy()
                    changeIndicesForJoinSubtrees(self.left, self.right)
                    if self.name in ['+', '-']:
                        self.name = 't' + self.name
                    else:
                        self.name = 't' + self.name[1]
                    processed = True

                    assert(len(self.left.upper) == len(self.right.upper))
                    assert(len(self.left.lower) == len(self.right.lower))

        if self.name == 'u-':
            if self.isUnary():
                self.upper = self.left.upper.copy()
                self.lower = self.left.lower.copy()
                processed = True

        if self.name == 'T':
            if self.isUnary():
                self.lower = self.left.upper.copy()
                self.upper = self.left.lower.copy()
                processed = True

        if self.name == '*' and self.isBinary():
            # check if scalar mult
            if self.left.isScalar() or self.right.isScalar():
                self.name = 's*'
            else:
                # check for outer product
                if self.left.isVector() and self.right.isCoVector():
                    oldIndexLeft = self.left.upper[0]
                    oldIndexRight = self.right.lower[0]
                    if oldIndexRight == oldIndexLeft:
                        newIndex = max(self.left.maxIndex(), self.right.maxIndex()) + 1
                        self.right.changeIndex(oldIndexRight, newIndex)
                    processed = True
                elif len(self.left.lower) == 1 and len(self.right.upper) == 1:
                    oldIndexLeft = self.left.lower[0]
                    oldIndexRight = self.right.upper[0]
                    if oldIndexRight == oldIndexLeft:
                        newIndex = oldIndexLeft
                    else:
                        if self.right.maxIndex() < oldIndexLeft:
                            newIndex = oldIndexLeft
                        else:
                            newIndex = max(self.left.maxIndex(), self.right.maxIndex()) + 1

                    self.left.changeIndex(oldIndexLeft, newIndex)
                    self.right.changeIndex(oldIndexRight, newIndex)
                    # in case of matrices involved make sure the other
                    # indices are different
                    commonIndex = set(self.left.upper).intersection(set(self.right.lower))
                    for c in commonIndex:
                        newIndex = max(self.left.maxIndex(), self.right.maxIndex()) + 1
                        self.left.changeIndex(c, newIndex)
                    processed = True

                if processed:
                    self.upper = self.left.upper.copy()
                    self.lower = self.right.lower.copy()
                    changeIndicesForJoinSubtrees(self.left, self.right)
                    self.name = 't*'
                    assert(self.isBinary())

        if self.name == 's*':
            # scalar mult
            # needs to come after '*' case
            assert(self.isBinary())
            assert(self.left.isScalar() or self.right.isScalar())

            # make sure scalar is in left child
            if not self.left.isScalar():
                tmp = self.left
                self.left = self.right
                self.right = tmp
                self.children = [self.left, self.right]

            self.upper = self.right.upper.copy()
            self.lower = self.right.lower.copy()
            changeIndicesForJoinSubtrees(self.left, self.right)
            self.name = 't*'
            processed = True

        if self.name == '/':
            if self.isBinary() and self.right.isScalar():
                self.upper = self.left.upper.copy()
                self.lower = self.left.lower.copy()
                changeIndicesForJoinSubtrees(self.left, self.right)
                self.name = 't/'
                processed = True

        if self.name == '^':  # scalar^scalar or matrix^scalar == matrix*matrix*matrix...
            if self.isBinary() and \
                self.left.isScalar() and self.right.isScalar():
                changeIndicesForJoinSubtrees(self.left, self.right)
                processed = True

            elif self.isBinary() and (len(self.left.upper) == len(self.left.lower)):
                self.upper = self.left.upper.copy()
                self.lower = self.left.lower.copy()
                changeIndicesForJoinSubtrees(self.left, self.right)
                processed = True

                print(self.lower,self.upper)
                assert(len(self.left.upper) == len(self.left.lower))

        if self.name == '.^':  # scalar^something or something^scalar or something^same_dims
            if self.isBinary() and (self.right.isScalar() or self.left.isScalar()):
                if self.right.isScalar():
                    self.upper = self.left.upper.copy()
                    self.lower = self.left.lower.copy()
                if self.left.isScalar():
                    self.upper = self.right.upper.copy()
                    self.lower = self.right.lower.copy()
                changeIndicesForJoinSubtrees(self.left, self.right)
                processed = True
            
            # same_dims
            elif  (len(self.left.upper) == len(self.right.upper)) and \
                (len(self.left.lower) == len(self.right.lower)):

                matchIndex(self.left, self.left.upper, self.right, self.right.upper)
                matchIndex(self.left, self.left.lower, self.right, self.right.lower)

                self.upper = self.left.upper.copy()
                self.lower = self.left.lower.copy()
                changeIndicesForJoinSubtrees(self.left, self.right)
                processed = True

                assert(len(self.left.upper) == len(self.right.upper))
                assert(len(self.left.lower) == len(self.right.lower))
            

        if self.name in ['norm1', 'norm2', 'sum']:
            if self.isUnary():
                processed = True

        if self.name in ['log', 'exp', 'sin', 'cos', 'tan', 'abs', 'sign',
                         'relu', 'tanh', 'arcsin', 'arccos', 'arctan', 'no_op']:
            if self.isUnary():
                self.lower = self.left.lower.copy()
                self.upper = self.left.upper.copy()
                processed = True

        if self.name in ['tr', 'det', 'logdet', 'sum', 'norm1', 'norm2']: # Some of these can be simplified in scalar cases (see prepareforeval)
            if self.isUnary() and \
                (self.left.isMatrix() or self.left.isScalar()):

                processed = True

        if self.name in ['inv']:
            if self.isUnary() and \
                (self.left.isMatrix() or self.left.isScalar()):
                self.lower = self.left.upper.copy()
                self.upper = self.left.lower.copy()
                if self.isScalar():
                    self.setTo(TensorTree('t/',TensorTree('Var_1', upper = self.upper, lower = self.lower),\
                        self.left, 
                        self.upper, self.lower))
                processed = True

        if self.name in ['softmax']:
            if self.isUnary() and self.left.isMatrix():
                self.lower = self.left.lower.copy()
                self.upper = self.left.upper.copy()

                processed = True
            if self.isUnary() and (self.left.isVector() or self.left.isCoVector()):
                self.lower = self.left.lower.copy()
                self.upper = self.left.upper.copy()
                self.name = "v_softmax"
                processed = True
            

        if self.name in ['vector', 'matrix']:
            if self.isUnary() and self.left.isScalar():
                if not (self.lower or self.upper):
                    index = self.left.maxIndex()
                    self.upper = [index + 1]
                    if self.name == 'vector':
                        self.lower = []
                    else:
                        self.lower = [index + 2]
                # if self.left.isNumeric():
                #     self.setTo(TensorTree('Var_' + str(self.left.getNumeric()), \
                #         upper = self.upper, lower = self.lower))
                # else:
                    # change to Scalar s* Vec_1 (s* -> gets converted to t* automatically)
                self.setTo(TensorTree('s*', self.left, \
                        TensorTree('Var_1', upper = self.upper, lower = self.lower),
                        self.upper, self.lower))
                processed = True

        if self.name == 'diag':
            if self.isUnary() and \
                (self.left.isVector() or self.left.isCoVector() \
                 or self.left.isMatrix):

                index = self.left.maxIndex()
                if self.left.isVector():
                    self.upper = self.left.upper
                    self.lower = [index + 1]
                elif self.left.isCoVector():
                    self.upper = [index + 1]
                    self.lower = self.left.lower
                elif self.left.isMatrix():
                    self.name = 'diag2'
                    self.upper = self.left.upper
                processed = True

        # mulitply for derivative
        if self.name in ['R*', 'R/']:
            assert(self.isBinary())

            remove1 = set(self.left.upper).intersection(set(self.right.lower))
            remove2 = set(self.left.lower).intersection(set(self.right.upper))
            remove = remove1.union(remove2)
            upper = set(self.left.upper).union(set(self.right.upper))
            lower = set(self.left.lower).union(set(self.right.lower))
            upper = upper.difference(remove)
            lower = lower.difference(remove)

            self.upper = list(upper)
            self.lower = list(lower)
            if self.name == 'R*':
                # maybe switch children
                if remove1 or (self.right.isScalar()):
                    tmp = self.left
                    self.left = self.right
                    self.right = tmp
                    self.children = [self.left, self.right]

                self.name = 't*'
            else:
                self.name = 't/'

                # might need to change A ./v to A * (1 ./ v)
                if not len(self.left.upper) == len(self.right.upper) or \
                    not len(self.left.lower) == len(self.right.lower):
                    if self.right.isCoVector():
                        T = TensorTree('T', TensorTree('Var_1', [], [], \
                        self.right.lower, self.right.upper))
                    else:
                        T = TensorTree('Var_1', [], [], \
                        self.right.upper, self.right.lower)

                    self.right = TensorTree('t/', T, \
                        self.right, self.right.upper, self.right.lower)
                    self.name = 't*'
                    self.children = [self.left, self.right]

            processed = True
		
        if not processed:
            operator = self.name
            rightOp = self.right.getType() if self.right else None
            leftOp = self.left.getType() if self.left else "missing argument"
            message = self.assembleSemanticMessage(operator, leftOp, rightOp)
            raise SemanticException(message)

    def assembleSemanticMessage(self, operator, left, right):
        message = f'Cannot {operator} {left} and {right}.'
        opMapBinary = {'-': 'subtract', '+': 'add', '*': 'multiply',
                       '/': 'divide', '.*': 'elementwise multiply',
                       './': 'elementwise divide'}
        opMapMatrix = {'det': 'determinant', 'logdet': 'log-determinant',
                       'tr': 'trace', 'adj': 'adjoint', 'inv': 'inverse'}

        if operator in opMapBinary:
            message = f'Cannot {opMapBinary[operator]} ({operator}) {left} and {right}.'

        elif operator == '^':
            message = f'Cannot raise a {left} to the power of a {right}. ' \
                'Exponent can only be a scalar. Use .^ instead.'

        elif operator == '.^':
            message = f'Cannot elementwise raise a {left} to the power of a {right}. ' \
                f'Exponent can only be a scalar or {left}.'

        elif operator in opMapMatrix:
            message = f'Cannot compute {opMapMatrix[operator]} ({operator}) of a {left}. It only works for matrices.'

        elif operator in ['matrix', 'vector']:
            message = f'Cannot form a {operator} of a {left}. It only works for scalars.'

        elif operator == 'diag':
            message = f'Cannot form a diagonal matrix ({operator}) of a {left}. It only works for vectors.'

        elif operator == 'norm2':
            message = f'Cannot compute the 2-norm ({operator}) of a {left}. It only works for vectors and matrices.'

        elif operator == 'norm1':
            message = f'Cannot compute the 1-norm ({operator}) of a {left}. It only works for vectors and matrices.'

        elif operator == 'sum':
            message = f'Cannot compute the sum ({operator}) of a {left}. It only works for vectors and matrices.'

        elif right is None:
            message = f'Cannot evaluate ({operator}) of a {left}.'
        else:
            message = f'Cannot use ({operator}) with {left} and {right}.'

        return message


    def copy(self):
        t = copy.copy(self)
        if hasattr(self, 'dimTable'):
            t.dimTable = self.dimTable
        t.upper = copy.copy(self.upper)
        t.lower = copy.copy(self.lower)
        t.children = []
        for c in self.children:
            t.children.append(c.copy())
        if len(t.children) >= 1:
            t.left = t.children[0]
        else:
            t.left = None
        if len(t.children) >= 2:
            t.right = t.children[1]
        else:
            t.right = None
        return t

    def setTo(self, target):
        self.name = target.name
        self.upper = target.upper
        self.lower = target.lower
        self.attributes = target.attributes
        self.left = target.left
        self.right = target.right
        self.children = target.children

    def setToRetainDimension(self, target):
        """
        As setTo, but retains the node's original dimensions. This is useful during optimization when a node can be
        dropped and replaced with broadcasting (in which case the end result has unchanged dimensions).
        """
        self.name = target.name
        # Skip dimension setting
        self.attributes = target.attributes
        self.left = target.left
        self.right = target.right
        self.children = target.children

    def setSymmetric(self):
        self.attributes.add("symmetric")

    def isSymmetric(self):
        return "symmetric" in self.attributes

    def add(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        return TensorTree('+', self, t)

    def __add__(self, t):
        return self.add(t)

    def mult(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        if self.isScalar() or t.isScalar():
            return TensorTree('*', self, t)
        else:
            return TensorTree('.*', self, t)

    def __mul__(self, t):
        return self.mult(t)

    def pow(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        return TensorTree('.^', self, t)

    def __pow__(self, t):
        return self.pow(t)

    def neg(self):
        return TensorTree('-', self)

    def __neg__(self):
        return self.neg()

    def dot(self, t):
        return TensorTree('*', self, t)

    def outer(self, t):
        return TensorTree('*', self, t.T())

    def div(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        if t.isScalar():
            return TensorTree('/', self, t)
        else:
            return TensorTree('./', self, t)

    def __truediv__(self, t):
        return self.div(t)

    def sub(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        return TensorTree('-', self, t)

    def __sub__(self, t):
        return self.sub(t)

    def __rmul__(self, t):
        return self.mult(Scalar(t))

    def __radd__(self, t):
        return self.add(Scalar(t))

    def T(self):
        return TensorTree('T', self)

    def exp(self):
        return TensorTree('exp', self)

    def log(self):
        return TensorTree('log', self)

    def sin(self):
        return TensorTree('sin', self)

    def cos(self):
        return TensorTree('cos', self)

    def inv(self):
        return TensorTree('inv', self)

    def softmax(self):
        return TensorTree('softmax', self)

    def det(self):
        return TensorTree('det', self)

    def sum(self):
        return TensorTree('sum', self)

    def norm2(self):
        return TensorTree('norm2', self)


    def nodeIndices(self):
        return set(self.upper + self.lower)

    def childIndices(self):
        s = set()
        for c in self.children:
            s |= c.nodeIndices()
        return s

    def allIndices(self):
        s = set()
        return self.__allIndices(s)

    def __allIndices(self, s):
        s |= set(self.upper)
        s |= set(self.lower)
        for c in self.children:
            s = c.__allIndices(s)
        return s

    def internalIndices(self):
        return self.allIndices() - set(self.upper) - set(self.lower)

    def internalIndices2(self):
        return self.childIndices() - set(self.upper) - set(self.lower)

    def otherChildrenContain(self, matchIndex, i, j):
        for k in range(len(self.children)):
            if k == i or k == j:
                continue
            if matchIndex in self.children[k].upper or \
               matchIndex in self.children[k].lower:
                return True
        return False

    def otherChildrenContainUpper(self, matchIndex, i, j):
        for k in range(len(self.children)):
            if k == i or k == j:
                continue
            if matchIndex in self.children[k].upper:
                return True
        return False

    def otherChildrenContainLower(self, matchIndex, i, j):
        for k in range(len(self.children)):
            if k == i or k == j:
                continue
            if matchIndex in self.children[k].lower:
                return True
        return False

    def checkChildConsistency(self):
        if len(self.children) > 2:
            print(self.prettyString())
        assert len(self.children) <= 2
        if len(self.children) >= 1:
            assert(self.left)
        if len(self.children) == 2:
            assert(self.right)

        if self.left:
            if not self.left == self.children[0]:
                print(self.prettyString())
            assert(self.left == self.children[0])

            self.left.checkChildConsistency()
        if self.right:
            if not self.right == self.children[1]:
                print(self.prettyString())
            assert(self.right == self.children[1])
            self.right.checkChildConsistency()

    def checkIndexConsistency(self):
        for c in self.children:
            c.checkIndexConsistency()

        if self.name == "t+" or self.name == "t-":
            for c in self.children:
                if not set(self.upper) == set(c.upper) or not \
                    set(self.lower) == set(c.lower):
                    print(self.prettyString())
                assert(set(self.upper) == set(c.upper))
                assert(set(self.lower) == set(c.lower))

        if self.name == "t*" or self.name == "t/":
            assert(len(self.children) == 2)
            upperSet = set(self.left.upper) | set(self.right.upper)
            lowerSet = set(self.left.lower) | set(self.right.lower)
            cancelSet1 = set(self.left.upper) & set(self.right.lower)
            cancelSet2 = set(self.left.lower) & set(self.right.upper)
            cancelSet = cancelSet1 | cancelSet2
            upperSet = upperSet.difference(cancelSet)
            lowerSet = lowerSet.difference(cancelSet)
            if not set(self.upper) == upperSet or not set(self.lower) == lowerSet:
                print(self.prettyString())
            assert(set(self.upper) == upperSet)
            assert(set(self.lower) == lowerSet)


    def checkConsistency(self):
        self.checkChildConsistency()
        self.checkIndexConsistency()

    def isZero(self):
        if self.isNumeric():
            return float(self.name[4:])==0
            #return self.getNumeric() == 0
        else:
            return False

    def isOne(self):
        if self.isNumeric():
            return float(self.name[4:])==1
        elif self.name == "T":
                return self.left.isOne()
        else:
            return False

    def isDelta(self):
#        return self.name.startswith('delta')
        return self.name == 'delta'

    def isDeltaT(self):
#        return self.name.startswith('delta')
        return self.name == 'deltaT'

    def isScalar(self):
        return self.upper == [] and self.lower == []

    def isVector(self):
        return len(self.upper) == 1 and self.lower == []

    def isCoVector(self):
        return self.upper == [] and len(self.lower) == 1

    def isMatrix(self):
        return len(self.upper) == 1 and len(self.lower) == 1

    def isUnary(self):
        return self.left and not self.right

    def isBinary(self):
        if self.left and self.right:
            assert(len(self.children) == 2)
        return self.left and self.right

    def isNumeric(self, ignoreUnaryMinus = False):
        # check also for unary minus
        if not ignoreUnaryMinus and self.name == 'u-':
            return self.left.isNumeric()
        if not self.name.startswith('Var_'):
            return False
        s = self.name[4:]
        return re.fullmatch(r"^(-|\+)?\d+(.\d+)?((e|E)(-|\+)?\d+)?$",s)
        #return s.replace('.','',1).isdigit()  # a bit faster and less Exceptions in debugging

    def isConstant(self):
        if self.isNumeric():
            return True
        else:
            return False

    def isTensorMultIKJL(self):
        if self.name == 't*' and  \
            self.left.isMatrix() and self.right.isMatrix() and \
            len(self.upper) == 2 and len(self.lower) == 2:
            if self.left.upper[0] == self.upper[0] and \
                self.left.lower[0] == self.lower[1] and \
                self.right.upper[0] == self.upper[1] and \
                self.right.lower[0] == self.lower[0]:
                return True
        return False

    def isTensorMultIJK(self):
        if self.name == 't*' and  \
            self.left.isMatrix() and self.right.isVector() and \
            len(self.upper) == 2 and len(self.lower) == 1:
            if self.left.upper[0] == self.upper[0] and \
                self.left.lower[0] == self.lower[0] and \
                self.right.upper[0] == self.upper[1]:
                return True
        return False

    def isTensorMultIKJ(self):
        if self.name == 't*' and  \
            self.left.isMatrix() and self.right.isCoVector() and \
            len(self.upper) == 1 and len(self.lower) == 2:
            if self.left.upper[0] == self.upper[0] and \
                self.left.lower[0] == self.lower[1] and \
                self.right.lower[0] == self.lower[0]:
                return True
        return False

    def isTensorMultJKI(self):
        if self.name == 't*' and  \
            len(self.left.upper) == 0 and len(self.left.lower) == 2 and \
            self.right.isVector() and \
            len(self.upper) == 1 and len(self.lower) == 2:
            if self.left.lower[0] == self.lower[0] and \
                self.left.lower[1] == self.lower[1] and \
                self.right.upper[0] == self.upper[0]:
                return True
        return False

    def isPointwiseOp(self):
        if self.isBinary():
            if self.left.isScalar() and self.right.isScalar():
                return True
        if self.isScalar():
            return False
        if not self.isBinary():
            return False
        if self.left.isScalar():
            return False
        if self.right.isScalar():
            return False
        # the and really needs to be there, otherwise it will
        # cause some error somewhere else when simplifying
        if self.upper == self.left.upper == self.right.upper and \
            self.lower == self.left.lower == self.right.lower:
            return True
        return False

    def getPointwiseOpIndices(self):
        lI = set()
        if self.isBinary():
            lI = set(self.upper).intersection(set(self.left.upper)).intersection(set(self.right.upper))
            lI |= set(self.lower).intersection(set(self.left.lower)).intersection(set(self.right.lower))
        return lI

    def makePointwiseOp(self, T, indices = None):
        # might need to change index
        common1 = set(self.upper).intersection(set(T.lower))
        common2 = set(self.lower).intersection(set(T.upper))
        if indices:
            common1 = common1.intersection(indices)
            common2 = common2.intersection(indices)
        # no clue why this was here
#        common3 = set(self.upper).intersection(set(T.upper))
#        common4 = set(self.lower).intersection(set(T.lower))
#        if common1 or common3:
            # might need to change index for self.lower
#            if self.isMatrix() and T.upper:
#                self.changeIndex(self.lower[0], T.upper[0])
#        if common2 or common4:
            # might need to change index for self.upper
#            if self.isMatrix() and T.lower:
#                self.changeIndex(self.upper[0], T.lower[0])
#        print('make')
#        print(self.prettyString())
#        print(T.prettyString())
#        print('common1', common1)
#        print('common2', common2)
        if common1 or common2:
            if self.isVector() or self.isCoVector():
                return TensorTree('T', self)
            elif self.isMatrix():
                if common1 and common2:
                    return TensorTree('T', self)
                else:
                    # need to do some kind of half transpose
                    if common1:
                        newIndex = max(self.maxIndex(), T.maxIndex()) + 1
                        copySelf = self.copy()
                        copySelf.changeIndex(self.upper[0], newIndex)
                        t = TensorTree('delta', [], [], [], [newIndex] + self.upper)
                        return TensorTree('t*', copySelf, t, [], self.lower + self.upper)
                    else:
                        newIndex = max(self.maxIndex(), T.maxIndex()) + 1
                        copySelf = self.copy()
                        copySelf.changeIndex(self.lower[0], newIndex)
                        t = TensorTree('delta', [], [], [newIndex] + self.lower, [])
                        return TensorTree('t*', copySelf, t, self.upper + self.lower, [])
            else:
                print('what is this?')
                assert(False)
        else:
            return self

    def orderMatters(self):
        assert(self.name == 't*')
        # pointwise mult
        if self.isPointwiseOp():
            return False
        # scalar mult
        if self.left.isScalar() or self.right.isScalar():
            return False
        return True

    #TODO Test
    def getType(self):
        if self.isScalar():
            return "scalar"
        elif self.isVector():
            return "column vector"
        elif self.isCoVector():
            return "row vector"
        elif self.isMatrix():
            return "matrix"
        else:
            raise Exception("Who am i?")

    def getOrder(self):
        return len(self.upper) + len(self.lower)

    def getNumeric(self):
        if self.name == 'u-':
            return -1*(self.left.getNumeric())
        f = float(self.name[4:])
        return int(f) if  f==int(f) else f
        #s = self.name[4:]
        # maybe its an int
        #floatS = float(s)
        #intS = int(floatS)
        #if intS == floatS:
        #    return intS
        #return float(s)


    def changeIndex(self, indexOld, indexNew):
        # first need to check that indexNew is not in self yet, otherwise we
        # might get crashes
        if indexOld == indexNew:
            return
        if indexNew in self.allIndices():
            tmpIndex = self.maxIndex() + 1
            self._changeIndex(indexNew, tmpIndex)
        self._changeIndex(indexOld, indexNew)

    def _changeIndex(self, indexOld, indexNew):
        upper = []
        for i in self.upper:
            if i == indexOld:
                i = indexNew
            upper += [i]
        self.upper = upper

        lower = []
        for i in self.lower:
            if i == indexOld:
                i = indexNew
            lower += [i]
        self.lower = lower

        for c in self.children:
            c._changeIndex(indexOld, indexNew)

    def maxIndex(self):
        maxIndex = 0
        if self.lower:
            maxIndex = max(maxIndex, max(self.lower))
        if self.upper:
            maxIndex = max(maxIndex, max(self.upper))
        if self.right:
            maxIndex = max(maxIndex, self.right.maxIndex())
        if self.left:
            maxIndex = max(maxIndex, self.left.maxIndex())
        return maxIndex

    def hasIndex(self, index):
        if index in self.lower or index in self.upper:
            return True
        isThere = False
        if self.left:
            isThere = isThere or self.left.hasIndex(index)
        if self.right:
            isThere = isThere or self.right.hasIndex(index)
        return isThere

    def uniqueSubtreeIndex(self):
        newIndex = self.maxIndex() + 1
        self.__uniqueSubtreeIndex(newIndex)

    def __uniqueSubtreeIndex(self, newIndex):
        for c in self.children:
            childInternalIndices = set(c.internalIndices())
            childIndices = set(c.upper + c.lower)
            ownIndices = set(self.upper + self.lower)
            toChange = ownIndices.intersection(childInternalIndices) - childIndices
            for i in toChange:
                c.changeIndex(i, newIndex)
                newIndex += 1

        for c in self.children:
            newIndex = c.__uniqueSubtreeIndex(newIndex)
        return newIndex



    def swapChildren(self):
        assert(self.isBinary())
        tmp = self.left
        self.left = self.right
        self.right = tmp
        self.children = [self.left, self.right]

    def contains(self, name):
        for c in self.children:
            found = c.contains(name)
            if found:
                return found

        if self.name == name:
            return True
        return False


    def symbolTable(self):
        _symbolTable = {}
        for v in self.children:
            childSymbolTable = v.symbolTable()
            both = set(childSymbolTable.keys()).intersection(set(_symbolTable.keys()))
            for v in both:
                if not childSymbolTable[v] == _symbolTable[v]:
                    assert(False)
            _symbolTable.update(childSymbolTable)
        if self.name.startswith('Var_') and not self.isNumeric():
            if self.isScalar():
                typeStr = 'scalar'
            elif self.isVector() or self.isCoVector():
                typeStr = 'vector'
            elif self.isMatrix():
                typeStr = 'matrix'
            else:
                assert(False)
            _symbolTable[self.name[4:]] = typeStr
        return _symbolTable


    def inferDimension(self):
        dimTable = {}
        self.__inferDimensionPlus(dimTable)
        if hasattr(self, 'dimTable'):
            for index in self.dimTable:
                if index in dimTable:
                    dimTable[index] |= self.dimTable[index]
                else:
                    dimTable[index] = self.dimTable[index]

        # same dimensions coming from delta tensors
        l = self.equalDims()
        # need to join sets with same variables but different indices coming
        # from different subtrees
        indexJoin = {}
        for i in dimTable:
            indexJoin[i] = {i}

        for (i, j) in l:
            if not i in indexJoin:
                indexJoin[i] = {i}
            if not j in indexJoin:
                indexJoin[j] = {j}
            indexJoin[i] |= indexJoin[j]
            indexJoin[j] = indexJoin[i]
            
        for i in dimTable:
            for j in dimTable:
                if i == j:
                    continue
                if dimTable[i].intersection(dimTable[j]):
                    indexJoin[i] |= {j}
                    indexJoin[j] |= {i}
        for i in indexJoin:
            for j in indexJoin[i]:
                if i in dimTable and j in dimTable:
                    dimTable[i] |= dimTable[j]
                elif j in dimTable:
                    dimTable[i] = dimTable[j]
                elif i in dimTable:
                    dimTable[j] = dimTable[i]
        return dimTable

    def equalDims(self):
        l = []
        for child in self.children:
            l += child.equalDims()

        # Eye is square
        if self.isDelta() and self.isMatrix():
            l += [(self.upper[0], self.lower[0])]

        elif self.name == "diag" and self.isMatrix():
            l += [(self.upper[0], self.lower[0])]

        elif self.name == "diag2":
            l += [(self.left.upper[0], self.left.lower[0])]

        # Only square matrices can be multiplied with themselves
        elif self.name == "^" and self.left.isMatrix():
            l += [(self.left.upper[0],self.left.lower[0])]

        return l

    def __inferDimensionPlus(self, dimTable):
        for child in self.children:
            child.__inferDimensionPlus(dimTable)

        if not self.children:
#            assert(len(self.upper) <= 1)
#            assert(len(self.lower) <= 1)
            if self.name.startswith('Var_'):
                if not self.isNumeric():
                    if self.upper:
                        if not self.upper[0] in dimTable:
                            dimTable[self.upper[0]] = set()
                        dimTable[self.upper[0]] |= set([self.name[4:] + '_rows'])
                    if self.lower:
                        if not self.lower[0] in dimTable:
                            dimTable[self.lower[0]] = set()
                        dimTable[self.lower[0]] |= set([self.name[4:] + '_cols'])

    def prepareForExpressionTree(self):
        for c in self.children:
            c.prepareForExpressionTree()

        # if self.name == 't*':
        #     # vector(1)' * vector = sum(vector)
        #     if self.left.isCoVector() and \
        #         self.right.isVector() and self.isScalar():
        #         if self.left.isOne():
        #             self.setTo(TensorTree('sum', self.right))
        #             return
        #         elif self.right.isOne():
        #             self.setTo(TensorTree('sum', self.left))
        #             return
        if self.isDelta() and len(self.upper) == 2 and len(self.lower) == 2:
            left = TensorTree('delta',
                              upper=[self.upper[0]],
                              lower=[self.lower[1]])
            right = TensorTree('delta',
                               upper=[self.upper[1]],
                               lower=[self.lower[0]])
            self.setTo(TensorTree('t*', left, right,
                                  self.upper, self.lower))
            return
#            # vector' * vector = norm2(vector)^2
#            if self.left.isCoVector() and \
#                self.right.isVector() and self.isScalar():
#                if self.left.name == 'T':
#                    if self.left.left.isEqual(self.right):
#                        self.setTo(TensorTree('^', \
#                            TensorTree('norm2', self.right), Scalar(2)))
#                        return


    def inOrderTree(self):
        for child in self.children:
            child.inOrderTree()

        if self.name == 't*':
            assert(self.isBinary())
            if self.left.isScalar():
                return
            if self.right.isScalar():
                self.swapChildren()
                return
            # left and right are both not scalars now
            if self.left.lower == self.right.lower and \
               self.left.upper == self.right.upper:
                return
            # matrix * delta == trace, if corresponding indices match
            # or matrix A * matrix B == trace (A'*B)
            commonIndex1 = set(self.left.lower) & set(self.right.upper)
            commonIndex2 = set(self.left.upper) & set(self.right.lower)
            if self.left.isMatrix and self.right.isMatrix and \
                commonIndex1 and commonIndex2 and self.isScalar():
                if self.left.isDelta():
                    return
                elif self.right.isDelta():
                    return
                # A * B = tr(A'*B)
                return
            commonIndex = set(self.left.upper) & set(self.right.lower)
            # standard multiply but swap children
            if commonIndex:
                self.swapChildren()
            commonIndex = set(self.left.lower) & set(self.right.upper)
            # standard multiply
            if commonIndex:
                return
            # outer product
            if self.left.isVector() and self.right.isCoVector():
                return
            # outer product but swap children
            if self.left.isCoVector() and self.right.isVector():
                self.swapChildren()
                return
            # diag(vector) times matrix
            if self.left.isMatrix() and self.right.isVector() and \
                self.left.upper == self.right.upper:
                self.swapChildren()
                return
            # diag(vector) times matrix
            if self.right.isMatrix() and self.left.isVector() and \
                self.left.upper == self.right.upper:
                return
            # matrix times diag(covector)
            if self.left.isMatrix() and self.right.isCoVector() and \
                self.left.lower == self.right.lower:
                return
            # matrix times diag(covector)
            if self.right.isMatrix() and self.left.isCoVector() and \
                self.left.lower == self.right.lower:
                self.swapChildren()
                return


    def removeTranspose(self):
        for c in self.children:
            c.removeTranspose()

        if self.name == 'T':
            if self.isScalar():
                self.setTo(self.left)
            elif self.isMatrix() and self.left.isNumeric():
                self.setTo(TensorTree(self.left.name, \
                    upper = self.upper, lower = self.lower))
            elif self.left.isDelta() or self.left.isDeltaT():
                self.setTo(TensorTree(self.left.name, \
                    upper = self.upper, lower = self.lower))
            elif self.isMatrix and self.left.isSymmetric():
                self.setTo(TensorTree(self.left.name, \
                    upper = self.upper, lower = self.lower,
                    attributes = self.attributes))

    def expressionTree(self, partialMult = False):
        e = self.copy()
        try:
#            e.reorderTree()
            e.prepareForExpressionTree()
            e.__toExpressionTree(partialMult)
            e.removeTranspose()
        except SemanticException as expt:
            n = len(e.upper) + len(e.lower)
            s = ''
            if n == 1:
                s = '1st'
            elif n == 2:
                s = '2nd'
            elif n == 3:
                s = '3rd'
            else:
                s = str(n) + 'th'
            raise SemanticException('This ' + s + ' order tensor cannot be displayed as a matrix.')

        e.removeUnnecessaryTranspose()
        return e

    def __toExpressionTree(self, partialMult):
        for child in self.children:
            child.__toExpressionTree(partialMult)

        if self.name in ['t+', 't-']:
            self.name = self.name[1]
            return
        if self.name == 't*':
            assert(self.isBinary())
            if self.left.isScalar():
                self.name = 'S*'
                return
            if self.right.isScalar():
                self.swapChildren()
                self.name = 'S*'
                return
            # left and right are both not scalars now
            if self.left.lower == self.right.lower and \
               self.left.upper == self.right.upper:
                self.name = '.*'
                return
            # matrix * delta == trace, if corresponding indices match
            # or matrix A * matrix B == trace (A'*B)
            commonIndex1 = set(self.left.lower) & set(self.right.upper)
            commonIndex2 = set(self.left.upper) & set(self.right.lower)
            if self.left.isMatrix() and self.right.isMatrix() and \
                commonIndex1 and commonIndex2 and self.isScalar():
                if self.left.isDelta():
                    self.setTo(TensorTree('tr', self.right))
                    return
                elif self.right.isDelta():
                    self.setTo(TensorTree('tr', self.left))
                    return
                # A * B = tr(A'*B)
                self.setTo(TensorTree('tr', \
                    TensorTree('*', self.left, self.right)))
                self.__toExpressionTree(partialMult)
                return
            # matrix * delta == diag(matrix), if corresponding indices match
            if self.isVector() or self.isCoVector():
                if self.left.isDelta() and self.right.isMatrix():
                    self.setTo(TensorTree('diag', self.right))
                    return
                if self.left.isMatrix() and self.right.isDelta():
                    self.setTo(TensorTree('diag', self.left))
                    return
            commonIndex = set(self.left.upper) & set(self.right.lower)
            # standard multiply but swap children
            if commonIndex:
                self.swapChildren()
            commonIndex = set(self.left.lower) & set(self.right.upper)
            # standard multiply
            if commonIndex:
                if len(self.upper) <= 1 and len(self.lower) <= 1:
                    self.name = 'M*'
                    return
            # outer product
            if self.left.isVector() and self.right.isCoVector():
                self.name = 'O*'
                return
            # outer product but swap children
            if self.left.isCoVector() and self.right.isVector():
                self.swapChildren()
                self.name = 'O*'
                return
            # diag(vector) times matrix
            if self.left.isMatrix() and self.right.isVector() and \
                self.left.upper == self.right.upper:
                self.swapChildren()
                if self.right.isDelta():
                    upper = self.upper
                    lower = self.lower
                    self.setTo(TensorTree('diag', self.left))
                    self.upper = upper
                    self.lower = lower
                else:
                    if partialMult:
                        self.name = 'partialMultL'
                    else:
                        self.name = 'M*'
                        self.left.setTo(TensorTree('diag', self.left))
                return
            # diag(vector) times matrix
            if self.right.isMatrix() and self.left.isVector() and \
                self.left.upper == self.right.upper:
                if self.right.isDelta():
                    upper = self.upper
                    lower = self.lower
                    self.setTo(TensorTree('diag', self.left))
                    self.upper = upper
                    self.lower = lower
                else:
                    if partialMult:
                        self.name = 'partialMultL'
                    else:
                        self.name = 'M*'
                        self.left.setTo(TensorTree('diag', self.left))
                return
            # matrix times diag(covector)
            if self.left.isMatrix() and self.right.isCoVector() and \
                self.left.lower == self.right.lower:
                if self.left.isDelta():
                    upper = self.upper
                    lower = self.lower
                    self.setTo(TensorTree('diag', self.right))
                    self.upper = upper
                    self.lower = lower
                else:
                    if partialMult:
                        self.name = 'partialMultR'
                    else:
                        self.name = 'M*'
                        self.right.setTo(TensorTree('diag', self.right))
                return
            # matrix times diag(covector)
            if self.right.isMatrix() and self.left.isCoVector() and \
                self.left.lower == self.right.lower:
                self.swapChildren()
                if self.left.isDelta():
                    upper = self.upper
                    lower = self.lower
                    self.setTo(TensorTree('diag', self.right))
                    self.upper = upper
                    self.lower = lower
                else:
                    if partialMult:
                        self.name = 'partialMultR'
                    else:
                        self.name = 'M*'
                        self.right.setTo(TensorTree('diag', self.right))
                return

            for i in range(2):
                if self.isTensorMultIKJL():
                    self.right.setTo(TensorTree('T', self.right))
                    if partialMult:
                        self.name = 'tensorMultIKJL'
                    else:
                        self.name = 'kron'
                        self.swapChildren()
                    return
                if self.isTensorMultIJK():
                    if partialMult:
                        self.name = 'tensorMultIJK'
                    else:
                        self.name = 'kron'
                        self.right.setTo(TensorTree('T', self.right))
                        self.swapChildren()
                    return
                if self.isTensorMultIKJ():
                    if partialMult:
                        self.name = 'tensorMultIKJ'
                    else:
                        self.name = 'kron'
                        self.right.setTo(TensorTree('T', self.right))
                        self.swapChildren()
                    return
                if self.isTensorMultJKI():
                    if partialMult:
                        self.name = 'tensorMultJKI'
                    else:
                        self.name = 'kron'
                    return

                self.swapChildren()

            # seems we cannot transform TensorTree into ExpressionTree
            raise SemanticException('')

        if self.name == 't/':
            assert(self.isBinary())
            if self.right.isScalar():
                self.name = 'S/'
                return
            # right is no scalar -> can only be ./
            if self.left.lower == self.right.lower and \
               self.left.upper == self.right.upper:
                self.name = './'
                return
            assert(False)

    def removeUnnecessaryTranspose(self):
        for c in self.children:
            c.removeUnnecessaryTranspose()
        if self.name == 'T' and self.left.name == 'T':
            self.setTo(self.left.left)
        if self.name == 'diag' and self.left.name == 'T':
            self.left.setTo(self.left.left)
        if self.name == 'diag2' and self.left.name == 'T':
#            self.left.setTo(self.left.left)
            pass
        if self.name == 'sum' and self.left.name == 'T':
            self.left.setTo(self.left.left)
        if self.name == 'norm2' and self.left.name == 'T':
            self.left.setTo(self.left.left)
        if self.name == 'norm1' and self.left.name == 'T':
            self.left.setTo(self.left.left)
        if self.name == 'T' and self.isScalar():
            self.setTo(self.left)

    def replace(self, s, t):
        for c in self.children:
            c.replace(s, t)

        if self.name == s:
            assert(self.upper == t.upper)
            assert(self.lower == t.lower)
            self.setTo(t)

    def rename(self, s, t):
        for c in self.children:
            c.rename(s, t)

        if self.name == s:
            self.name = t

    def substitute(self, s, t):
        assert len(s.upper) == len(t.upper)
        assert len(t.lower) == len(t.lower)
        assert len(s.children) == 0

        forbiddenIndices = self.allIndices()
        self.__substitute(s, t, forbiddenIndices)

    def __substitute(self, s, t, forbiddenIndices):
        for c in self.children:
            c.__substitute(s, t, forbiddenIndices)

        if self.name == s.name:
            tNew = t.copy()
            for i, index in enumerate(self.upper):
                tNew.changeIndex(tNew.upper[i], index)
            for i, index in enumerate(self.lower):
                tNew.changeIndex(tNew.lower[i], index)

            newIndex = -1
            if forbiddenIndices:
                newIndex = max(forbiddenIndices)
                tIndices = tNew.allIndices()
                if tIndices:
                    newIndex = max(newIndex, max(tIndices))
            newIndex += 1
            # need to change all forbidden indices in t
            changeIndices = forbiddenIndices - set(self.upper) - set(self.lower)
            changeIndices = changeIndices.intersection(tNew.internalIndices())
            for index in changeIndices:
                tNew.changeIndex(index, newIndex)
                forbiddenIndices.add(newIndex)
                newIndex += 1
            assert(self.upper == tNew.upper)
            assert(self.lower == tNew.lower)
            self.setTo(tNew)

    def prepareForEval(self):
        for c in self.children:
            c.prepareForEval()
        if self.name == 'logdet':
            if self.left.isScalar():
                self.name = 'log'
            else: 
                self.setTo(TensorTree('log', self))
                self.left.name = 'det'
        # elif self.name == 'adj':
        #     self.setTo(TensorTree('s*', TensorTree('det', self.left), \
        #         TensorTree('inv', self.left)))
        #     self.name = 'S*'
        elif self.name in ['tr', 'det', 'sum'] and self.left.isScalar():
            self.setTo(self.left)
        elif self.name in ['norm1', 'norm2'] and self.left.isScalar():
            self.name = 'abs'

    def inferDimMap(self, mapsTo):
        # infers the numerical dimension of the current variable mapping
        # also performs consistency checks
        symbolicDims = self.inferDimension()
        dimMap = {}
        for d in symbolicDims:
            dimSet = list(symbolicDims[d])
            numD = None
            for symD in dimSet:
                newNumD = None
                if symD[-5:] == '_rows':
                    newNumD = mapsTo[symD[:-5]].shape[0]
                elif symD[-5:] == '_cols':
                    newNumD = mapsTo[symD[:-5]].shape[1]
                if not newNumD:
                        raise ValueError('dimension mismatch')

                if numD:
                    if not numD == newNumD:
                        raise ValueError('dimension mismatch')
                numD = newNumD
            dimMap[d] = numD

        return dimMap

    def checkDims(self, mapsTo):
        self.checkTypes(mapsTo)
        dimMap = self.inferDimMap(mapsTo)
        return dimMap

    def allVars(self, s = None):
        if s is None: s=set()
        for c in self.children:
            s |= c.allVars(s)
        if self.name.startswith('Var_') and not self.isNumeric():
            s |= {self.name[4:]}
        return s

    def simpleString(self):
        s = 'TensorTree(' + "'" + str(self.name) + "'"
        for child in self.children:
            s += ', ' + child.simpleString()
        s += ', ' + str(self.upper) + ', ' + str(self.lower) + ')'
        return s

    def prettyString(self):
        return '\n'.join(self.__prettyListString())

    def __prettyListString(self):
        childStr = []
        maxChildrenHeight = 0
        childWidth = []
        for child in self.children:
            s = child.__prettyListString()
            childStr += [s]
            maxChildrenHeight = max(len(s), maxChildrenHeight)
            childWidth += [len(s[0])]

        for i in range(len(childStr)):
            s = childStr[i]
            for j in range(len(s), maxChildrenHeight):
                s.append(' ' * childWidth[i])

        childrenStr = []
        for i in range(maxChildrenHeight):
            lineS = ''
            for s in childStr:
                if lineS:
                    lineS += '  '
                lineS += s[i]
            childrenStr += [lineS]

        if childrenStr:
            childrenWidth = len(childrenStr[0])
        else:
            childrenWidth = 0

        leftOffset = 0
        if len(childWidth) > 0:
            leftOffset = childWidth[0]

        rightOffset = 0
        if len(childWidth) > 0:
            rightOffset = childWidth[-1]

        nameStr = '' + str(self.name) + ''
        if hasattr(self, 'broadcastFlag'):
                nameStr += f" ({self.broadcastFlag})"
        if "symmetric" in self.attributes:
            nameStr += ', s'
        nameWidth = len(nameStr)
        upperStr = ' ' * nameWidth + str(self.upper) + ' '
        upperWidth = len(upperStr)
        lowerStr = ' ' * nameWidth + str(self.lower) + ' '
        lowerWidth = len(lowerStr)
        maxWidth = max([nameWidth, upperWidth, lowerWidth])
        nameStr += ' ' * (maxWidth - nameWidth)
        upperStr += ' ' * (maxWidth - upperWidth)
        lowerStr += ' ' * (maxWidth - lowerWidth)

        totalWidth = max(childrenWidth, maxWidth)

        leftAdd = int((totalWidth - maxWidth) / 2)
        rightAdd = totalWidth - leftAdd - maxWidth

        totalS =  [' ' * (int(totalWidth / 2) - 1) + '|' \
                   + ' ' * (totalWidth - int(totalWidth / 2))]
        totalS += [' ' * leftAdd + upperStr + ' ' * rightAdd]
        totalS += [' ' * leftAdd + nameStr + ' ' * rightAdd]
        totalS += [' ' * leftAdd + lowerStr + ' ' * rightAdd]
        if self.left or self.right:
            totalS += [' ' * int(leftOffset / 2 - 1) \
                       + '-' * (int(totalWidth / 2) - 1 - int(leftOffset / 2 - 1)) + '+' \
                       + '-' * (totalWidth - int(totalWidth / 2) - int(rightOffset / 2)) \
                       + ' ' * int(rightOffset / 2)]

        totalS += childrenStr
        return totalS
    
    def prepareForSimplify(self):
        """
        if self.name == "norm2":
            if self.left.isVector():
                t = TensorTree('*', \
                    TensorTree('T', self.left), self.left)
            elif self.left.isCoVector():
                t = TensorTree('*', \
                    self.left, TensorTree('T', self.left))
            elif self.left.isMatrix():
                t = TensorTree('sum', TensorTree('.*', \
                    self.left, self.left))
            else:
                assert(False)
            self.setTo(TensorTree('^', t, Scalar(0.5)))

        if self.name == "norm1":
            self.setTo(TensorTree('sum', TensorTree('abs', self.left)))
        """
        """
        if self.name == "sum":
            if self.left.isVector():
                self.setTo(TensorTree('*', \
                    TensorTree('T', Vector(1)), self.left))
            elif self.left.isCoVector():
                self.setTo(TensorTree('*', \
                    self.left, Vector(1)))
            elif self.left.isMatrix():
                self.setTo(TensorTree('*', \
                    TensorTree('T', Vector(1)), \
                    TensorTree('*', self.left, Vector(1))))
        """
        #elif self.name == "tr":
        #    # transform trace into delta mult
        #    self.setTo(TensorTree('t*', self.left, \
        #        TensorTree('delta', [], [], \
        #            self.left.lower,
        #            self.left.upper)))
        #elif self.name == "diag":
        #    # transform diag into delta mult
        #    self.setTo(TensorTree('t*', self.left, \
        #        TensorTree('delta', [], [], \
        #            self.upper,
        #            self.lower),
        #            self.upper, self.lower))

        for c in self.children:
            c.prepareForSimplify()

    def prepareForDiff(self):
        if self.name == 'adj':
            self.setTo(TensorTree('s*', TensorTree('det', self.left), \
                TensorTree('inv', self.left)))

        for c in self.children:
            c.prepareForDiff()

    def removeMatrixFunctions(self):
        if self.name == "det" and self.left.isScalar():
            self.setTo(self.left)
        if self.name == "logdet" and self.left.isScalar():
            self.setTo(TensorTree('log', self.left))
        if self.name == "inv" and self.left.isScalar():
            self.setTo(TensorTree('/', Scalar(1), self.left))
        for c in self.children:
            c.removeMatrixFunctions()


    def removeZero(self):
        if self.left:
            self.left.removeZero()
        if self.right:
            self.right.removeZero()

        if self.name == 't+':
            if self.left.isZero():
                self.setTo(self.right)
                return
            if self.right.isZero():
                 self.setTo(self.left)
                 return
        if self.name == 't-':
            if self.left.isZero():
                self.setTo(TensorTree('u-', self.right))
                return
            if self.right.isZero():
                self.setTo(self.left)
                return

        if self.name == 'T' or self.name == 'u-':
            if self.left.isZero():
                self.name = 'Var_0'
                self.left = []
                self.right = []
                self.children = []
        if self.name == 't*':
            if self.left.isZero() or self.right.isZero():
                self.name = 'Var_0'
                self.left = []
                self.right = []
                self.children = []
        if self.name == 't/':
            if self.left.isZero():
                self.name = 'Var_0'
                self.left = []
                self.right = []
                self.children = []
        if self.name == '^':
            if self.right.isZero() and not self.left.isZero():
                self.name = 'Var_1'
                self.left = []
                self.right = []
                self.children = []
        return

    def removeOne(self):
        if self.left:
            self.left.removeOne()
        if self.right:
            self.right.removeOne()

        oneChild = []
        if self.left and self.left.isOne():
            oneChild = self.left
            otherChild = self.right
        if self.right and self.right.isOne():
            oneChild = self.right
            otherChild = self.left

        if self.name == 't*':
            if oneChild:
              # Scalar 1 * TT -> TT
              if oneChild.isScalar():
                  self.setTo(otherChild)
                  return
              # Var_1 .* TT -> TT
              if oneChild.upper == otherChild.upper and \
                 oneChild.lower == otherChild.lower:
                  self.setTo(otherChild)
                  return
              # Vector(1) * Matrix == diag(Vector(1)) * Matrix == Matrix
              if oneChild.isVector() and \
                 otherChild.isMatrix() and \
                 self.isMatrix():
                  self.setTo(otherChild)
              # Matrix * CoVector(1) == Matrix * diag(Vector(1)) == Matrix
              if oneChild.isCoVector() and \
                 otherChild.isMatrix() and \
                 self.isMatrix():
                  self.setTo(otherChild)

        if self.name in ['^', '.^']:
            # TT .^ 1 -> TT
            if self.right.isOne() and self.right.isScalar():
                self.setTo(self.left)
                return

    def transposeVector(self, index):
        if self.isVector() and self.upper[0] == index:
            self.setTo(TensorTree('T', self))
            return
        if self.isCoVector() and self.lower[0] == index:
            self.setTo(TensorTree('T', self))
            return

        if index in self.upper:
            self.upper = list(set(self.upper) - set([index]))
            self.lower.append(index)
        elif index in self.lower:
            self.lower = list(set(self.lower) - set([index]))
            self.upper.append(index)

        for c in self.children:
            c.transposeVector(index)


    def removeDelta(self):
        if self.left:
            self.left.removeDelta()
        if self.right:
            self.right.removeDelta()

        deltaChild = None
        if self.right and (self.right.isDelta() or self.right.isDeltaT()):
            deltaChild = self.right
            otherChild = self.left
        if not (deltaChild and (len(deltaChild.upper) == len(deltaChild.lower))):
            if self.left and (self.left.isDelta() or self.left.isDeltaT()):
                deltaChild = self.left
                otherChild = self.right

        if self.name == 'T':
            if deltaChild:
                self.name = deltaChild.name
                self.left = []
                self.right = []
                self.children = []
                return

        if self.name == 't*':
            # delta * delta == diag(delta) == vec_1, if corresponding indices match
            if (self.isVector() or self.isCoVector()) and \
                self.left.isDelta() and self.right.isDelta():
                if self.left.isMatrix() or self.right.isMatrix():
                    self.setTo(TensorTree('Var_1', upper=self.upper, lower=self.lower))
                    return

            if deltaChild:
                changeFrom = []
                changeTo = []
                # scalar delta
                if not deltaChild.upper and not deltaChild.lower:
                    self.setTo(otherChild)

                # 4th order tensor delta, no transpose
                if len(deltaChild.upper) == 2 and len(deltaChild.lower) == 2 \
                    and not deltaChild.name == 'deltaT':
                    if not deltaChild.lower[0] in self.lower:
                        changeFrom = deltaChild.lower[0]
                        changeTo = deltaChild.upper[1]
                        otherChild.changeIndex(changeFrom, changeTo)
                        del(deltaChild.lower[0])
                        del(deltaChild.upper[1])
                    elif not deltaChild.upper[0] in self.upper:
                        changeFrom = deltaChild.upper[0]
                        changeTo = deltaChild.lower[1]
                        otherChild.changeIndex(changeFrom, changeTo)
                        del(deltaChild.upper[0])
                        del(deltaChild.lower[1])

                # matrix delta
                if len(deltaChild.upper) == 1 and len(deltaChild.lower) == 1:
                    if deltaChild.lower[0] in self.lower:
                        changeTo = deltaChild.lower
                        changeFrom = deltaChild.upper
                    if changeFrom and changeFrom[0] in self.upper:
                        changeTo = []
                        changeFrom = []
                    if changeTo and changeTo[0] in otherChild.nodeIndices():
                        changeTo = []
                        changeFrom = []
                    if deltaChild.upper[0] in self.upper:
                        changeTo = deltaChild.upper
                        changeFrom = deltaChild.lower
                    if changeFrom and changeFrom[0] in self.lower:
                        changeTo = []
                        changeFrom = []
                    if changeTo and changeTo[0] in otherChild.nodeIndices():
                        changeTo = []
                        changeFrom = []
                    if changeTo and changeFrom:
                        otherChild.changeIndex(changeFrom[0], changeTo[0])
                        otherChild.upper = self.upper.copy()
                        otherChild.lower = self.lower.copy()
                        self.setTo(otherChild)
                        return

                # transpose operator on vector
                if len(deltaChild.lower) == 2 and not deltaChild.upper and \
                    otherChild.isVector():
                    for i in range(2):
                        if deltaChild.lower[i] in self.lower:
                            changeTo = [deltaChild.lower[i]]
                            changeFrom = [deltaChild.lower[(i+1) % 2]]
                        if changeFrom and changeFrom[0] in self.lower:
                            changeTo = []
                            changeFrom = []
                        if changeTo and changeFrom:
                            otherChild.changeIndex(changeFrom[0], changeTo[0])
                            self.setTo(TensorTree('T', otherChild))
                            return

                # transpose operator on covector
                if len(deltaChild.upper) == 2 and not deltaChild.lower and \
                    otherChild.isCoVector():
                    for i in range(2):
                        if deltaChild.upper[i] in self.upper:
                            changeTo = [deltaChild.upper[i]]
                            changeFrom = [deltaChild.upper[(i+1) % 2]]
                        if changeFrom and changeFrom[0] in self.upper:
                            changeTo = []
                            changeFrom = []
                        if changeTo and changeFrom:
                            otherChild.changeIndex(changeFrom[0], changeTo[0])
                            self.setTo(TensorTree('T', otherChild))
                            return
                """
                # transpose operator on vector, but not right here
                if len(deltaChild.lower) == 2 and not deltaChild.upper and \
                    len(otherChild.upper) > 1:
                    for i in range(2):
                        if deltaChild.lower[i] in self.lower:
                            changeTo = [deltaChild.lower[i]]
                            changeFrom = [deltaChild.lower[(i+1) % 2]]
                        if changeFrom and changeFrom[0] in self.lower:
                            changeTo = []
                            changeFrom = []
                        if changeTo and changeFrom:
                            otherChild.transposeVector(changeFrom[0])
                            otherChild.changeIndex(changeFrom[0], changeTo[0])
                            self.setTo(otherChild)
                            return

                # transpose operator on covector, but not right here
                if len(deltaChild.upper) == 2 and not deltaChild.lower and \
                    len(otherChild.lower) > 1:
                    for i in range(2):
                        if deltaChild.upper[i] in self.upper:
                            changeTo = [deltaChild.upper[i]]
                            changeFrom = [deltaChild.upper[(i+1) % 2]]
                        if changeFrom and changeFrom[0] in self.upper:
                            changeTo = []
                            changeFrom = []
                        if changeTo and changeFrom:
                            otherChild.transposeVector(changeFrom[0])
                            otherChild.changeIndex(changeFrom[0], changeTo[0])
                            self.setTo(otherChild)
                            return
                """
                # 4th order tensor transposed delta
                if len(deltaChild.upper) == 2 and len(deltaChild.lower) == 2 \
                    and deltaChild.name == 'deltaT':
                    if otherChild.isMatrix():
                        if otherChild.upper[0] in deltaChild.lower and \
                            otherChild.lower[0] in deltaChild.upper:
                            assert(otherChild.upper[0] == deltaChild.lower[0])
                            assert(otherChild.lower[0] == deltaChild.upper[0])
                            assert(not otherChild.upper[0] == otherChild.lower[0])
                            otherChild.changeIndex(otherChild.upper[0], \
                                deltaChild.lower[1])
                            otherChild.changeIndex(otherChild.lower[0], \
                                deltaChild.upper[1])
                            self.setTo(TensorTree('T', otherChild))

    def deltaT2delta(self):
        for c in self.children:
            c.deltaT2delta()

        if self.name == 't*':
            for c in self.children:
                if c.isDelta() and (len(c.upper) == 2 or len(c.lower) == 2):
                    deltaTChild = c

                    upperIndices = []
                    lowerIndices = []
                    for c2 in self.children:
                        if c2 == deltaTChild:
                            continue
                        else:
                            upperIndices += c2.upper
                            lowerIndices += c2.lower
                    for i in range(2):
                        if deltaTChild.upper:
                            index = deltaTChild.upper[1 - i]
                        else:
                            index = deltaTChild.lower[1 - i]
                        if index in upperIndices and index in lowerIndices:
                            if deltaTChild.upper:
                                deltaTChild.upper = [deltaTChild.upper[i]]
                                deltaTChild.lower = [index]
                            else:
                                deltaTChild.lower = [deltaTChild.lower[i]]
                                deltaTChild.upper = [index]
                            break

    def keyOrderAdd(self):
        return self.name

    def keyOrderMult(self):
        if self.isScalar():
            s = '0' + self.name
        else:
            s = '1'# + self.name
        return s

    def reorderTree(self, reorderMult = False):
        for c in self.children:
            c.reorderTree(reorderMult)
        if self.name == 't+':
            self.children.sort(key = TensorTree.keyOrderAdd)
        if reorderMult and self.name == 't*':
            self.children.sort(key = TensorTree.keyOrderMult)

        if len(self.children) > 0:
            self.left = self.children[0]
        if len(self.children) > 1:
            self.right = self.children[1]

        if not reorderMult and self.name == 't*':
            if self.orderMatters():
                commonIndex = set(self.left.lower) & set(self.right.upper)
                # standard multiply
                if commonIndex:
                    return
                commonIndex = set(self.left.upper) & set(self.right.lower)
                # standard multiply but swap children
                if commonIndex:
                    self.swapChildren()
                    return
                # outer product but swap children
                if self.left.isCoVector() and self.right.isVector():
                    self.swapChildren()
                    return

    def distributivUnfold(self):
        if self.left:
            self.left.distributivUnfold()
        if self.right:
            self.right.distributivUnfold()
        if self.name == 't*':
            if self.right.name == 't+':
                index = self.left.nodeIndices()
                index = index.difference(self.nodeIndices())
                if index or self.left.isScalar():
                    self.setTo(TensorTree('t+', TensorTree('t*', \
                               self.left, self.right.left, \
                               self.upper, self.lower), \
                               TensorTree('t*', \
                               self.left, self.right.right, \
                               self.upper, self.lower), \
                               self.upper, self.lower))
                    return
            if self.left.name == 't+':
                index = self.right.nodeIndices()
                index = index.difference(self.nodeIndices())
                if index or self.right.isScalar():
                    self.setTo(TensorTree('t+', TensorTree('t*', \
                               self.left.left, self.right, \
                               self.upper, self.lower), \
                               TensorTree('t*', \
                               self.left.right, self.right, \
                               self.upper, self.lower), \
                               self.upper, self.lower))
                    return

    def flattenAdd(self):
        for c in self.children:
            c.flattenAdd()
        if self.name == 't+':
            # Attention here! We modify self.children!
            i = 0
            while i < len(self.children):
                child = self.children[i]
                if child.name == 't+':
                    moveForeward = len(child.children)
                    # might need to rename indices in child.children
                    ownIndices  = self.internalIndices()
                    childIndices = child.internalIndices()
                    changeIndices = childIndices - ownIndices
                    newIndex = self.maxIndex() + 1
                    for index in changeIndices:
                        child.changeIndex(index, newIndex)
                        newIndex += 1
                    tmp = self.children[0:i]
                    tmp += child.children
                    tmp += self.children[i+1:]
                    self.children = tmp
                    i += moveForeward - 1
                i += 1

    def unflattenAdd(self):
        for c in self.children:
            c.unflattenAdd()
        if self.name == 't+':
            # Attention here! We modify self.children!
            while len(self.children) > 2:
                t = TensorTree('t+', self.children[0], self.children[1],
                               self.upper, self.lower)
                self.children = self.children[1:]
                self.children[0] = t
                self.left = self.children[0]
                self.right = self.children[1]

    def flattenMult(self):
        for c in self.children:
            c.flattenMult()
        if self.name == 't*':
            # Attention here! We modify self.children!
            i = 0
            while i < len(self.children):
                child = self.children[i]
                if child.name == 't*':
                    moveForeward = len(child.children)
                    # might need to rename indices in child.children
                    ownIndices  = self.childIndices()
                    childIndices = child.internalIndices2()
                    changeIndices = childIndices & ownIndices
                    newIndex = self.maxIndex() + 1
                    for index in changeIndices:
                        child.changeIndex(index, newIndex)
                        newIndex += 1
                    tmp = self.children[0:i]
                    tmp += child.children
                    tmp += self.children[i+1:]
                    self.children = tmp
                    i += moveForeward - 1
                i += 1

    def unflattenMult(self):
        if self.name == 't*':
            while True:
                hasChanged = False
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    # numeric scalar
                    if leftChild.isScalar() and leftChild.isNumeric():
                        scalarChild = leftChild.copy()
                        self.children = self.children[:i] + self.children[i+1:]
                        self.setTo(TensorTree('t*', scalarChild, self,
                                    self.upper, self.lower))
                    i += 1

                hasChanged = False
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    # nonnumeric scalar
                    if leftChild.isScalar():
                        scalarChild = leftChild.copy()
                        self.children = self.children[:i] + self.children[i+1:]
                        self.setTo(TensorTree('t*', scalarChild, self,
                                    self.upper, self.lower))
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # .*
                        if leftChild.upper == rightChild.upper and \
                           leftChild.lower == rightChild.lower:
                            self.children[i] = TensorTree('t*', leftChild, rightChild, \
                              leftChild.upper, leftChild.lower)
                            self.children = self.children[:j] + self.children[j+1:]
                            hasChanged = True
                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # vector^T * vector * matrix -> (vector^T .* vector^T) * matrix
                        if leftChild.isCoVector():
                            matchIndex = leftChild.lower[0]
                            if leftChild.lower == rightChild.upper and \
                               leftChild.upper == rightChild.lower and \
                               self.otherChildrenContainUpper(matchIndex, i, j):
                                self.children[i] = TensorTree('t*', leftChild, \
                                  TensorTree('T', rightChild), \
                                  leftChild.upper, leftChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # vector^T * vector * matrix -> matrix * (vector .* vector)
                        if leftChild.isCoVector():
                            matchIndex = leftChild.lower[0]
                            if leftChild.lower == rightChild.upper and \
                               leftChild.upper == rightChild.lower and \
                               self.otherChildrenContainLower(matchIndex, i, j):
                                self.children[i] = TensorTree('t*', \
                                  TensorTree('T', leftChild), \
                                  rightChild, \
                                  rightChild.upper, rightChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * vector
                        if rightChild.isVector():
                            matchIndex = rightChild.upper[0]
                            if matchIndex in leftChild.lower and \
                               not self.otherChildrenContain(matchIndex, i, j):
                                lowerIndex = leftChild.lower.copy()
                                lowerIndex.remove(matchIndex)
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper, lowerIndex)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # vector^T * matrix
                        if rightChild.isCoVector():
                            matchIndex = rightChild.lower[0]
                            if matchIndex in leftChild.upper and \
                              not self.otherChildrenContain(matchIndex, i, j):
                                  upperIndex = leftChild.upper.copy()
                                  upperIndex.remove(matchIndex)
                                  self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                    upperIndex, leftChild.lower)
                                  self.children = self.children[:j] + self.children[j+1:]
                                  hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * delta / higher order tensor A -> tr(matrix*A)
                        if rightChild.isMatrix():
                            matchIndex1 = rightChild.upper[0]
                            matchIndex2 = rightChild.lower[0]
                            if matchIndex1 in leftChild.lower and \
                               matchIndex2 in leftChild.upper and \
                               not self.otherChildrenContain(matchIndex1, i, j) and \
                               not self.otherChildrenContain(matchIndex2, i, j):
                                indexUpper = copy.copy(leftChild.upper)
                                indexUpper.remove(matchIndex2)
                                indexLower = copy.copy(leftChild.lower)
                                indexLower.remove(matchIndex1)
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  indexUpper, indexLower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * matrix
                        if rightChild.isMatrix() and leftChild.isMatrix():
                            matchIndex = rightChild.upper[0]
                            if matchIndex == leftChild.lower[0] and \
                               not leftChild.upper[0] == rightChild.lower[0] and \
                               not self.otherChildrenContain(matchIndex, i, j):

                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper, rightChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1
                """
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * some tensor
                        if leftChild.isMatrix() and rightChild.upper:
                            matchIndex = rightChild.upper[0]
                            if matchIndex in leftChild.lower and \
                               not self.otherChildrenContain(matchIndex, i, j):
                                lowerIndex = set(leftChild.lower) | set(rightChild.lower)
                                lowerIndex = list(lowerIndex - set([matchIndex]))
                                upperIndex = set(leftChild.upper) | set(rightChild.upper)
                                upperIndex = list(upperIndex - set([matchIndex]))
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  upperIndex, lowerIndex)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1
                """
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix .* (actually diag) vector
                        if rightChild.isVector():
                            matchIndex = rightChild.upper[0]
                            if matchIndex in leftChild.upper and \
                               not self.otherChildrenContainUpper(matchIndex, i, j):
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper, leftChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix .* (actually diag) CoVector
                        if rightChild.isCoVector():
                            matchIndex = rightChild.lower[0]
                            if matchIndex in leftChild.lower and \
                               not self.otherChildrenContainLower(matchIndex, i, j):
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper, leftChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1
                """
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * matrix
                        if rightChild.isMatrix() and leftChild.isMatrix():
                            matchIndex = rightChild.upper[0]
                            if matchIndex == leftChild.lower[0] and \
                               not leftChild.upper[0] == rightChild.lower[0] and \
                               not self.otherChildrenContain(matchIndex, i, j):

                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper, rightChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1
                """
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * matrix, upper equal  -> third order tensor
                        if leftChild.isMatrix() and rightChild.isMatrix() and \
                            not leftChild.isDelta() and not rightChild.isDelta():
                            matchIndex = leftChild.upper[0]
                            if matchIndex == rightChild.upper[0] and \
                               not matchIndex == leftChild.lower[0] and \
                               not matchIndex == rightChild.lower[0] and \
                               not self.otherChildrenContain(matchIndex, i, j):
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper, leftChild.lower + rightChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # matrix * matrix, lower equal  -> third order tensor
                        if leftChild.isMatrix() and rightChild.isMatrix() and \
                            not leftChild.isDelta() and not rightChild.isDelta():
                            matchIndex = leftChild.lower[0]
                            if matchIndex == rightChild.lower[0] and \
                               not matchIndex == leftChild.upper[0] and \
                               not matchIndex == rightChild.upper[0] and \
                               not self.otherChildrenContain(matchIndex, i, j):
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper + rightChild.upper, leftChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1

                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 2:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 2:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # if nothing else has worked out
                        # last chance: outer products
                        if not leftChild.isDelta() and not rightChild.isDelta():
                            allIndices = leftChild.upper + leftChild.lower + \
                                rightChild.upper + rightChild.lower
                            if len(allIndices) == len(set(allIndices)):
                                self.children[i] = TensorTree('t*', leftChild, rightChild, \
                                  leftChild.upper + rightChild.upper, \
                                  leftChild.lower + rightChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1


                if not hasChanged:
                    break

            # update left and right back to correct children
            if len(self.children) > 0:
                self.left = self.children[0]
            else:
                self.left = []
            if len(self.children) > 1:
                self.right = self.children[1]
            else:
                self.right = []
      #      assert(len(self.children) <= 2)
        for c in self.children:
            c.unflattenMult()


    def groupScalarMult(self):
        if self.name == 't*':
            leftTreeChildren = []
            rightTreeChildren = []
            for c in self.children:
                if c.isScalar():
                    leftTreeChildren += [c]
                else:
                    rightTreeChildren += [c]

            if leftTreeChildren and rightTreeChildren:
                if len(leftTreeChildren) > 1:
                    leftTree = self.copy()
                    leftTree.children = leftTreeChildren
                    leftTree.upper = []
                    leftTree.lower = []
                else:
                    leftTree = leftTreeChildren[0]

                if len(rightTreeChildren) > 1:
                    rightTree = self.copy()
                    rightTree.children = rightTreeChildren
                else:
                    rightTree = rightTreeChildren[0]

                self.children = [leftTree, rightTree]

            self.updateChildren()
        for c in self.children:
            c.groupScalarMult()

    def updateChildren(self):
        # update left and right back to correct children
        if len(self.children) > 0:
            self.left = self.children[0]
        else:
            self.left = []
        if len(self.children) > 1:
            self.right = self.children[1]
        else:
            self.right = []


    def groupDivide(self):
        if self.name == 't*' and self.isScalar():
            leftTreeChildren = []
            rightTreeChildren = []
            allScalar = True
            for c in self.children:
                if not c.isScalar():
                    allScalar = False
                (child, scalar) = c.splitTreePowConst()
                if scalar.isNumeric() and scalar.getNumeric() < 0:
                    rightTreeChildren += [TensorTree('^', child, \
                        Scalar(-scalar.getNumeric()))]
                else:
                    leftTreeChildren += [c]
            if allScalar and rightTreeChildren:
                # do we have something above the fraction slash?
                if leftTreeChildren:
                    if len(leftTreeChildren) > 1:
                        leftTree = self.copy()
                        leftTree.children = leftTreeChildren
                        leftTree.upper = []
                        leftTree.lower = []
                    else:
                        leftTree = leftTreeChildren[0]
                    leftTree.updateChildren()
                else:
                    leftTree = Scalar(1)

                if len(rightTreeChildren) > 1:
                    rightTree = self.copy()
                    rightTree.children = rightTreeChildren
                else:
                    rightTree = rightTreeChildren[0]
                rightTree.updateChildren()

                self.name = 't/'
                self.children = [leftTree, rightTree]
                self.updateChildren()

        for c in self.children:
            c.groupDivide()

    def pow2Divide(self):
        for c in self.children:
            c.pow2Divide()

        if self.name == '^':
            (child, scalar) = self.splitTreePowConst()
            if scalar.isNumeric() and scalar.getNumeric() < 0:
                rightTree = TensorTree('^', child, \
                    Scalar(-scalar.getNumeric()))
                self.setTo(TensorTree('t/', Scalar(1), rightTree))

    def constantFold(self):
        for c in self.children:
            c.constantFold()
        if self.name in ['t+', 't*', '^']: # or self.name == 't/':
            hasChanged = True
            while hasChanged:
                hasChanged = False
                for i in range(len(self.children) - 1):
                    leftChild = self.children[i]
                    if leftChild.isNumeric():
                        for j in range(i+1, len(self.children)):
                            rightChild = self.children[j]
                            if rightChild.isNumeric() and \
                               leftChild.upper == rightChild.upper and \
                               leftChild.lower == rightChild.lower :
                                if self.name == 't+':
                                    self.children[i] = TensorTree('Var_' + \
                                                     str(leftChild.getNumeric() + \
                                                     rightChild.getNumeric()), \
                                                     upper = leftChild.upper, \
                                                     lower = leftChild.lower)
                                    self.children = self.children[:j] + self.children[j+1:]
                                    hasChanged = True
                                if self.name == 't*':
                                    self.children[i] = TensorTree('Var_' + \
                                                     str(leftChild.getNumeric() * \
                                                     rightChild.getNumeric()), \
                                                     upper = leftChild.upper, \
                                                     lower = leftChild.lower)
                                    self.children = self.children[:j] + self.children[j+1:]
                                    hasChanged = True
                                if self.name == '^' and rightChild.getNumeric() >= 0:
                                    a = leftChild.getNumeric()
                                    b = rightChild.getNumeric()
                                    if b <= 5 and a <= 1000:
                                        if 0 <= a or (-1000 <= a and b == int(b)):
                                            self.children[i] = TensorTree('Var_' + \
                                                         str(leftChild.getNumeric() ** \
                                                         rightChild.getNumeric()), \
                                                         upper = leftChild.upper, \
                                                         lower = leftChild.lower)
                                            self.children = self.children[:j] + self.children[j+1:]
                                            hasChanged = True
                                if self.name == 't/':
                                    self.children[i] = TensorTree('Var_' + \
                                                     str(leftChild.getNumeric() / \
                                                     rightChild.getNumeric()), \
                                                     upper = leftChild.upper, \
                                                     lower = leftChild.lower)
                                    self.children = self.children[:j] + self.children[j+1:]
                                    hasChanged = True
                                break

                            if rightChild.isNumeric() and \
                               (leftChild.isScalar() or rightChild.isScalar()):
                                if leftChild.isScalar():
                                    upper = rightChild.upper
                                    lower = rightChild.lower
                                else:
                                    upper = leftChild.upper
                                    lower = leftChild.lower
                                self.children[i] = TensorTree('Var_' + \
                                                 str(leftChild.getNumeric() * \
                                                 rightChild.getNumeric()), \
                                                 upper = upper, \
                                                 lower = lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True
                                break

                        if hasChanged:
                            break
                    if hasChanged:
                        break
            if len(self.children) == 1:
                self.setTo(self.children[0])
            # update left and right back to correct children
            if len(self.children) > 0:
                self.left = self.children[0]
            else:
                self.left = []
            if len(self.children) > 1:
                self.right = self.children[1]
            else:
                self.right = []

    def distributivFold(self):
        for c in self.children:
            c.distributivFold()
        if self.name == 't+':
            hasChanged = True
            while hasChanged:
                hasChanged = False
                for i in range(len(self.children) - 1):
                    leftChild = self.children[i]
                    for j in range(i+1, len(self.children)):
                        rightChild = self.children[j]
                        (equal, leftScalar, rightScalar, commonTree) = leftChild.isEqualModScalar(rightChild)
                        if equal:
                            self.children[i] = TensorTree('t*', \
                                TensorTree('t+', leftScalar, rightScalar,
                                           commonTree.upper,
                                           commonTree.lower),
                                commonTree,
                                upper = leftChild.upper,
                                lower = leftChild.lower)
                            self.children = self.children[:j] + self.children[j+1:]
                            self.constantFold()
                            hasChanged = True
                            break

                        if hasChanged:
                            break
                    if hasChanged:
                        break
            if len(self.children) == 1:
                self.setTo(self.children[0])
            # update left and right back to correct children
            if len(self.children) > 0:
                self.left = self.children[0]
            else:
                self.left = []
            if len(self.children) > 1:
                self.right = self.children[1]
            else:
                self.right = []

    def distributivUnfoldPow(self):
        if self.name == '^' and self.left.name == 't*':
            allScalar = True
            for c in self.left.children:
                if not c.isScalar():
                    allScalar = False
            if allScalar:
                newChildren = []
                for c in self.left.children:
                    newChildren += [TensorTree('^', c, self.right)]
                self.name = 't*'
                self.children = newChildren
                self.updateChildren()

        for c in self.children:
            c.distributivUnfoldPow()

    def distributivFoldPow(self):
        for c in self.children:
            c.distributivFoldPow()
        if self.name == 't*':
            hasChanged = True
            while hasChanged:
                hasChanged = False
                for i in range(len(self.children) - 1):
                    leftChild = self.children[i]
                    for j in range(i+1, len(self.children)):
                        rightChild = self.children[j]
                        (equal, leftPow, rightPow, commonTree) = leftChild.isEqualModPow(rightChild)
                        if equal and commonTree.isScalar():
                            self.children[i] = TensorTree('^', \
                                commonTree,
                                TensorTree('t+', leftPow, rightPow,
                                           commonTree.upper,
                                           commonTree.lower),
                                upper = leftChild.upper,
                                lower = leftChild.lower)
                            self.children = self.children[:j] + self.children[j+1:]
                            self.constantFold()
                            hasChanged = True
                            break

                        if hasChanged:
                            break
                    if hasChanged:
                        break
            if len(self.children) == 1:
                self.setTo(self.children[0])
            # update left and right back to correct children
            if len(self.children) > 0:
                self.left = self.children[0]
            else:
                self.left = []
            if len(self.children) > 1:
                self.right = self.children[1]
            else:
                self.right = []

    def logDet2logdet(self):
        for c in self.children:
            c.logDet2logdet()
        if self.name == 'log' and self.left.name == 'det':
            self.name = 'logdet'
            self.left.setTo(self.left.left)

    def binaryMinus2ScalarMult(self):
        for c in self.children:
            c.binaryMinus2ScalarMult()
        if self.name == 't-':
            assert(self.isBinary())
            self.right = TensorTree('t*', Scalar(-1), self.right, \
                          self.right.upper, self.right.lower)
            self.children[1] = self.right
            self.name = 't+'

    def scalarMult2UnaryMinus(self):
        for c in self.children:
            c.scalarMult2UnaryMinus()
        if self.name == 't*':
            if self.left.isScalar() and self.left.name == 'Var_-1':
                assert(self.isBinary())
                self.setTo(TensorTree('-', self.right))

    def unaryMinus2ScalarMult(self):
        for c in self.children:
            c.unaryMinus2ScalarMult()
        if self.name == 'u-':
            assert(self.isUnary())
            self.setTo(TensorTree('t*', Scalar(-1), self.left, \
                       self.left.upper, self.left.lower))

    def binaryMinus2Unary(self):
        for c in self.children:
            c.binaryMinus2Unary()
        if self.name == 't-':
            assert(self.isBinary())
            self.right = TensorTree('-', self.right)
            self.children[1] = self.right
            self.name = 't+'

    def unaryMinus2Binary(self):
        for c in self.children:
            c.unaryMinus2Binary()
        if self.name == 't+':
            assert(self.isBinary())
            if self.left.name == 'u-' and self.right.name == 'u-':
                self.left.setTo(self.left.left)
                self.right.setTo(self.right.left)
                self.setTo(TensorTree('u-', self))
                return
            if self.right.name == 'u-':
                self.name = 't-'
                self.right.setTo(self.right.left)
            if self.left.name == 'u-':
                self.name = 't-'
                self.left.setTo(self.left.left)
                self.swapChildren()

    def divide2Pow(self):
        for c in self.children:
            c.divide2Pow()
        if self.name == 't/' and self.right.isScalar():
            assert(self.isBinary())
            self.right = TensorTree('^', self.right, Scalar(-1), \
                          self.right.upper, self.right.lower)
            self.children[1] = self.right
            self.name = 't*'

    def moveUnaryMinusUp(self):
        for c in self.children:
            c.moveUnaryMinusUp()
        if self.name in ['t*', 't/']:
            if self.left.name == 'u-' and self.right.name == 'u-':
                self.setTo(TensorTree(self.name,
                           self.left.left, self.right.left,
                           self.upper, self.lower))
                return
            if self.left.name == 'u-':
                self.setTo(TensorTree('u-', TensorTree(self.name,
                           self.left.left, self.right,
                           self.upper, self.lower), [],
                           self.upper, self.lower))
                return
            if self.right.name == 'u-':
                self.setTo(TensorTree('u-', TensorTree(self.name,
                           self.left, self.right.left,
                           self.upper, self.lower), [],
                           self.upper, self.lower))
                return

    def negativeScalar2UnaryMinus(self):
        for c in self.children:
            c.negativeScalar2UnaryMinus()
        if self.isNumeric(ignoreUnaryMinus = True):
            s = self.getNumeric()
            if s < 0:
                self.setTo(TensorTree('u-', TensorTree('Var_' + \
                    str(-s), \
                    upper = self.upper, \
                    lower = self.lower)))

    def propagateTransposeDown(self):
#        while self.name == 'T' and self.isScalar():
#            self.setTo(self.left)
        while self.name == 'T' and self.left.name == 'T':
            self.setTo(self.left.left)
        if self.name == 'T':
            if self.left.isUnary():
                if self.left.name == 'diag2':
                    pass
                else:
                    self.name, self.left.name = self.left.name, self.name
                    if self.name in ['tr', 'det', 'logdet', 'norm2', 'norm1']:
                        self.left.upper = copy.copy(self.left.left.lower)
                        self.left.lower = copy.copy(self.left.left.upper)
                    elif not self.name in ['inv', 'adj']:
                        self.left.upper, self.left.lower = \
                            self.left.lower, self.left.upper
            if self.left.isBinary():
                self.setTo(TensorTree(self.left.name, \
                    TensorTree('T', self.left.left), \
                    TensorTree('T', self.left.right), \
                    self.upper, self.lower))
        for c in self.children:
            c.propagateTransposeDown()

    def propagateTransposeUp(self):
        if not self.children:
            return
        for c in self.children:
            c.propagateTransposeUp()
        while self.name == 'T' and self.left.name == 'T':
            self.setTo(self.left.left)
        if self.isUnary() and self.left.name == 'T':
            if self.name == 'diag2':
                pass
            else:
                self.name, self.left.name = self.left.name, self.name
                if self.left.name in ['tr', 'det', 'logdet', 'norm2', 'norm1']:
                    self.left.upper = []
                    self.left.lower = []
                elif not self.left.name in ['inv', 'adj']:
                    self.left.upper, self.left.lower = \
                        self.left.lower, self.left.upper
        if self.isBinary():
            if self.left.name == 'T' and \
                self.right.name == 'T':
                # remove transposes
                self.left.setTo(self.left.left)
                self.right.setTo(self.right.left)

                self.upper, self.lower = self.lower, self.upper
                self.setTo(TensorTree('T', self))
                return

            transposeChild = None
            scalarChild = None
            if self.left.name == 'T' and self.right.isScalar():
                transposeChild = self.left
                scalarChild = self.right
            if self.right.name == 'T' and self.left.isScalar():
                transposeChild = self.right
                scalarChild = self.left
            if transposeChild and scalarChild:
                # remove transposes
                transposeChild.setTo(transposeChild.left)

                self.upper, self.lower = self.lower, self.upper
                self.setTo(TensorTree('T', self))


    def removeInvInv(self):
        for c in self.children:
            c.removeInvInv()

        if self.name == 'inv' and self.left.name == "inv":
            self.setTo(self.left.left)

    def removeXInvX(self):
        for c in self.children:
            c.removeXInvX()
        """
        if self.name == 't*' and not self.isPointwiseOp():
            if self.left.name == "inv" and self.left.left.isEqual(self.right):
                self.setTo(TensorTree('delta', [], [], self.upper, self.lower))
            elif self.right.name == "inv" and self.right.left.isEqual(self.left):
                self.setTo(TensorTree('delta', [], [], self.upper, self.lower))
        """
        if self.name == "t*":
            while True:
                hasChanged = False
                i = 0
                while not hasChanged and \
                    i < len(self.children) and len(self.children) > 1:
                    leftChild = self.children[i]
                    j = 0
                    while not hasChanged and \
                        j < len(self.children) and len(self.children) > 1:
                        if i == j:
                            j += 1
                            continue
                        rightChild = self.children[j]

                        # inv(T) * T
                        if (leftChild.name == 'inv' and \
                            rightChild.isEqual(leftChild.left)) or \
                            (rightChild.name == 'inv' and \
                            leftChild.isEqual(rightChild.left)):
                            # make sure no other term has this index
                            if leftChild.lower and rightChild.upper:
                                if leftChild.lower == rightChild.upper:
                                    matchIndex == leftChild.lower[0]
                                    if not self.otherChildrenContain(matchIndex, i, j):
                                        self.children[i] = TensorTree('delta', \
                                            [], [], leftChild.upper, rightChild.lower)
                                        self.children = self.children[:j] + self.children[j+1:]
                                        hasChanged = True
                            else:
                                self.children[i] = TensorTree('delta', \
                                    [], [], leftChild.upper, rightChild.lower)
                                self.children = self.children[:j] + self.children[j+1:]
                                hasChanged = True

                        j += 1
                    i += 1
                if not hasChanged:
                    break
            if len(self.children) == 1:
                self.setTo(self.children[0])

    def removeXDivX(self):
        for c in self.children:
            c.removeXDivX()

        if self.name == 't/':
            if self.left.isEqual(self.right):
                self.setTo(TensorTree('Var_1', [], [], self.upper, self.lower))

    def moveScalarLeft(self):
        for c in self.children:
            c.moveScalarLeft()
        if self.name == 't*' and not self.left.isScalar() and \
            self.right.name == 't*' and self.right.left.isScalar():
            tmp = self.left.copy()
            self.left.setTo(self.right.left)
            self.right.left.setTo(tmp)
            self.right.upper = self.upper
            self.right.lower = self.lower

    def rotateRight(self):
        for c in self.children:
            c.rotateRight()
        if self.name == 't*' and self.left.name == 't*':
            pass

    def rotateScalarLeft(self):
        if self.name == 't*' and self.right.name == 't*' and \
            self.left.isScalar() and self.right.left.isScalar():
            self.left.setTo(TensorTree('t*', self.left, self.right.left))
            self.right.setTo(self.right.right)
        for c in self.children:
            c.rotateScalarLeft()

    def foldPow(self):
        for c in self.children:
            c.foldPow()
        if self.name == '^' and self.left.name == '^':
            self.right.setTo(TensorTree('*', self.left.right, self.right))
            self.left.setTo(self.left.left)

    def collectDivide(self):
        for c in self.children:
            c.collectDivide()

        if self.name == 't*' and self.isPointwiseOp():
            if self.right.name == 't/':
                self.swapChildren()
                self.name = 't/'
                self.left.name = 't*'
                tmp = self.left.right.copy()
                self.left.right.setTo(self.right)
                self.right.setTo(tmp)
            elif self.left.name == 't/':
                self.name = 't/'
                self.left.name = 't*'
                tmp = self.left.right.copy()
                self.left.right.setTo(self.right)
                self.right.setTo(tmp)



    def isEqual(self, target):
        if not self.name == target.name:
            return False
        if not (len(self.upper) == len(target.upper) and \
                len(self.lower) == len(target.lower)):
            return False
        if not len(self.children) == len(target.children):
            return False
        for i in range(len(self.children)):
            if not self.children[i].isEqual(target.children[i]):
                return False
        return True

    def splitConstTimesTree(self):
        if self.name == 't*':
          # TODO fix this
    #      assert(self.isBinary())
            if self.left.isNumeric():
                if len(self.children) == 2:
                    return (self.left, self.right)
                else:
                    t = self.copy()
                    t.children = t.children[1:]
                    if len(t.children) >= 1:
                        t.left = t.children[0]
                    if len(t.children) >= 2:
                        t.right = t.children[1]
                    return (self.left, t)
            else:
                return (Scalar(1), self)
        else:
            return (Scalar(1), self)

    def splitTreePowConst(self):
        if self.name == '^':
          # TODO fix this
    #      assert(self.isBinary())
            if self.right.isNumeric():
                if len(self.children) == 2:
                    return (self.left, self.right)
                else:
                    t = self.copy()
                    t.children = t.children[1:]
                    if len(t.children) >= 1:
                        t.left = t.children[0]
                    if len(t.children) >= 2:
                        t.right = t.children[1]
                    return (self.left, t)
            else:
                return (self, Scalar(1))
        else:
            return (self, Scalar(1))

    def isEqualModScalar(self, target):
        (leftScalar, leftT) = self.splitConstTimesTree()
        (rightScalar, rightT) = target.splitConstTimesTree()
        if leftT.isEqual(rightT):
            return (True, leftScalar, rightScalar, leftT)

        return (False, None, None, None)

    def isEqualModPow(self, target):
        (leftT, leftPow) = self.splitTreePowConst()
        (rightT, rightPow) = target.splitTreePowConst()
        if leftT.isEqual(rightT):
            return (True, leftPow, rightPow, leftT)

        return (False, None, None, None)




    def simplify(self, b = True, distributivUnfold = False):
        self.prepareForSimplify()
        self.removeMatrixFunctions()
        self.foldPow()
        self.constantFold()
        self.removeZero()
        self.removeOne()
        self.checkConsistency()
        self.removeDelta()
        self.checkConsistency()
        self.removeUnnecessaryTranspose()
        if b:
            self.propagateTransposeDown()
        self.removeTranspose()
        self.moveUnaryMinusUp()
        self.binaryMinus2ScalarMult()
        self.unaryMinus2ScalarMult()

        self.removeInvInv()
        self.removeXInvX()
        self.removeXDivX()
        self.removeDelta()
        self.checkConsistency()

        self.divide2Pow()
        self.distributivUnfoldPow()

        if distributivUnfold:
            self.distributivUnfold()
        self.flattenAdd()
        self.reorderTree()
#        self.rotateRight()
        self.distributivFold()
        self.constantFold()
        self.foldPow()
        self.constantFold()
        # removeZero not correct for flattened add
#        self.removeZero()
        self.removeOne()
        self.removeDelta()
#        print('before', self.prettyString())
        self.reorderTree()
#        self.distributivUnfoldPow()
        t = self.copy()
        try:
            self.flattenMult()
            self.reorderTree(reorderMult = True)
#            self.constantFold()
#            print('before', self.prettyString())
            self.deltaT2delta()
#            print('after', self.prettyString())
            self.removeXInvX()
            self.distributivFoldPow()
            self.groupScalarMult()
            self.groupDivide()
            self.pow2Divide()
            self.unflattenMult()
            self.unflattenAdd()
            self.checkConsistency()
        except AssertionError:
            print('could not unflatten tree')
            print('restored to original tree')
            print(t.prettyString())
            self.setTo(t)
#        print('after', self.prettyString())

        self.unflattenAdd()
        self.checkConsistency()
        if b:
            self.propagateTransposeDown()
        self.removeTranspose()
        self.checkConsistency()

        self.removeInvInv()
        self.removeXInvX()
        self.removeXDivX()

#        self.reorderTree()
        self.collectDivide()
        self.distributivFold()
        self.constantFold()
        self.removeZero()
        self.removeOne()
        self.removeDelta()

#        self.reorderTree()
        self.moveScalarLeft()
        self.rotateScalarLeft()
        self.constantFold()
        self.removeZero()
        self.removeOne()

        if b:
            self.propagateTransposeUp()
        self.removeUnnecessaryTranspose()
        self.scalarMult2UnaryMinus()
        self.negativeScalar2UnaryMinus()
        self.moveUnaryMinusUp()
        self.constantFold()
        self.removeZero()
        self.removeOne()
        self.scalarMult2UnaryMinus()
        self.unaryMinus2Binary()
        self.logDet2logdet()
#        self.reorderTree()

        self.removeXDivX()


class Scalar(TensorTree):
    def __init__(self, name):
        TensorTree.__init__(self, 'Var_' + str(name))

class Vector(TensorTree):
    def __init__(self, name):
        TensorTree.__init__(self, 'Var_' + str(name), upper = [0])

class Matrix(TensorTree):
    def __init__(self, name):
        TensorTree.__init__(self, 'Var_' + str(name), upper = [0], lower = [1])

def matchIndex(leftTree, leftList, rightTree, rightList):
    assert(leftTree)
    assert(rightTree)
    assert(len(leftList) == len(rightList))
    for i in range(len(rightList)):
        if not rightTree.hasIndex(leftList[i]):
            rightTree.changeIndex(rightList[i], leftList[i])
        else:
            # need to change both indeces
            newIndex = max(leftTree.maxIndex(), rightTree.maxIndex()) + 1
            leftTree.changeIndex(leftList[i], newIndex)
            rightTree.changeIndex(rightList[i], newIndex)

def changeIndicesForJoinSubtrees(leftTree, rightTree):
    # make it symmetric, it is better and correct
    changeIndicesForJoinSubtrees2(leftTree, rightTree)
    changeIndicesForJoinSubtrees2(rightTree, leftTree)

def changeIndicesForJoinSubtrees2(leftTree, rightTree):
    assert(leftTree)
    assert(rightTree)

    leftInternalIndices = leftTree.internalIndices()
    rightAllIndices = rightTree.allIndices()

    changeIndices = leftInternalIndices.intersection(rightAllIndices)
    if changeIndices:
        maxIndex = max(leftTree.maxIndex(), rightTree.maxIndex())
        newIndex = maxIndex

        for index in changeIndices:
            newIndex = newIndex + 1
            leftTree.changeIndex(index, newIndex)




class SemanticException(Exception):
    def __init__(self, message='', position=None):
        self.message = message + (f" @pos:{position}" if position else '')
    def __str__(self):
        return self.message