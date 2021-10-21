# -*- coding: utf-8 -*-
"""
Copyright: Soeren Laue (soeren.laue@uni-jena.de)

Edited by Konstantin Wiedom and Paul Gerhardt Rump for the LinA project: https://lina.ti2.uni-jena.de/
Further edited by Paul Gerhardt Rump for the LinEx project (paul.gerhardt.rump@uni-jena.de)

The core ExpressionTree module.
(Used to be TensorTree, but tensors (of degree > 2) do not appear in the project anymore.)
"""

import numbers

import copy
import os.path
import pickle
import re
import time

from LinEx.equivalencyCache import EquivalencyCache
from LinEx.orderedSet import OrderedSet
from expressions.dimension import Dimension

from LinEx.reshapeCache import ReshapeCache
from LinEx.reshapeList import allReshapes, simplifyReshapes
from LinEx.symboltree import SymbolTree

shortProp = {
    # ... based on the shape:
    'scalar' : 's',
    'vector' : 'v',
    'covector' : 'cv',
    'matrix' : 'm',
    'square_matrix' : 'sq',
    # ... based on entries being zero:
    'diagonal' : 'd',
    'triangle_upper' : 'tu',
    'triangle_lower' : 'tl',
    # ... based on values being the same:
    'symmetric' : 'sym',
    'pseudo_scalar' : 'ps', # What I mean here is that all values are the same (except those that are forced to 0 by another tag)
    # ... for recognition of constants and neutral elements (internal use only):
    'constant' : 'c',  # like eye, vector(5) etc. - anything that could be folded during generation time.
    'zero': '0',
    'one' : '1',
    # ... other (these are pretty complicated, but important for matrix factorization):
    'pos_def' : 'pd',
    'pos_semi_def' : 'psd',
    'neg_def' : 'nd',
    'neg_semi_def' : 'nsd',
}

allAttributes=set(shortProp.keys())


class ExpressionTree:
    # Fair warning: I currently assume that "shape" is precisely a 2-tuple. Flat dimensions should set as 1.
    # Higher tensors are not currently supported.
    def __init__(self, name, left: 'ExpressionTree' = None, right: 'ExpressionTree' = None, shape: tuple = (),
                 attributes: set = None, highLevelProps = False):
        """
        highLevelProps indicates whether compliacated propagations (including some that internally create other trees)
            should be determined. The parser should set this to true for new trees, but otherwise it should be false.
            Reshaping operations retain properties to the root of their tree, so thy don't have to re-derive.
        """
        self.name = name
        self.shape = shape
        self.attributes = attributes if attributes else set()
        del attributes # This way, I immediately notice if I try to change "attributes" rather than "self.attributes".

        self.left = left if left else None
        self.right = right if right else None
        if right and not left:
            raise SyntaxError(f"Tree {self} had a right but no left child!")
        
        # A few things that will be populated later in separate functions:
        self.varSet = None # Will be a set of variables in this tree.
        self.expString = None # Will be a dict that contains the various hashes I use
        
        self.userNamedDims = self.__linkDimensionsByUserRequest()
        
        ## Processing part 1 - detect the precise operation ##
        # Classify multiplication
        if self.name == '*':
            if not self.isBinary():
                raise SemanticException(self)

            # check if scalar mult
            if self.left.isScalar() or self.right.isScalar():
                self.name = 'S*'

            # It is some kind of M*. If either one is eye (diagonal and one), I can remove it right here:
            elif {'diagonal', 'one'}.issubset(self.left.attributes):
                self.setTo(self.right)
                return
            elif {'diagonal', 'one'}.issubset(self.right.attributes):
                self.setTo(self.left)
                return
            else:
                self.name = 'M*'

        # Classify *other* multiplication.
        elif self.name == 't*':
            raise SyntaxError("'t*' encountered, but that is not supported!")

        # Classify division
        elif self.name == '/':
            if not self.isBinary():
                raise SemanticException(self)
            if self.right.isScalar():
                self.name = 'S/'
            else:
                raise SemanticException(self) #TODO make ./

        elif self.name == 't/':
            raise SyntaxError("'t/' encountered, but that is not supported!")

        # Remove pointless scalar functions:
        elif self.name in ['tr', 'det', 'sum', 'det'] and self.isUnary() and self.left.isScalar():
            self.setTo(self.left)
            return # self.left already went through all the steps.

        # Simplify norm for scalars:
        elif self.name in ['norm1', 'norm2'] and self.isUnary() and self.left.isScalar():
            self.name = 'abs'

        # Simplify inverse for scalars:
        elif self.name in ['inv']:
            if not self.isUnary():
                raise SemanticException(self)
            if self.left.isScalar():
                self.setTo(ExpressionTree('S/', left=Scalar('1', attributes={'constant'}), right=self.left))

        # Substitute 'logdet':
        elif self.name == 'logdet':
            if self.left.isScalar():
                self.name = 'log'
            else:
                self.setTo(ExpressionTree('log', self))
                self.left.name = 'det'

        ## Processing part 2 - propagate shapes, properties, etc. ##
        if self.name in ['delta']: # TODO is suspect 'eye' might currently be the only delta there is.
            self.attributes.add('constant')
            self.attributes.add('one')
            # Shape must have been set on creation.

        elif self.name == 'Var_eye':
            self.name = 'delta' # TODO: From past experience, I'd say keeping an explicit "eye" might actually be good ...
            self.attributes.update({'square_matrix', 'diagonal', 'constant', 'pseudo_scalar', 'pos_def', 'triangle_upper',
			                                 'triangle_lower', 'symmetric', 'one'})
            # Shape must have been set on creation.

        elif self.name.startswith('Var_'):
            # Relevant attributes were already added on creation. See {@code ident()} in parser.py.
            # So was the shape. However, I still need to recognize 0 and 1.
            try:
                if float(self.name[4:])==0:
                    self.attributes.add('zero')
                if float(self.name[4:])==1:
                    self.attributes.add('one')
            except ValueError:
                pass # This is a variable and can be left alone.

        elif self.name == 'M*':
            if not self.isBinary():
                raise SemanticException(self)

            # Link colliding dimensions, unless they are already linked or we have an outer product (They are literal 1).
            if (not self.left.shape[1] == self.right.shape[0]) and (not self.left.isVector()):
                self.left.shape[1].link(self.right.shape[0])
            self.shape = self.left.shape[0], self.right.shape[1]

            # Certain properties can be propagated if both children have them:
            self.attributes.update(
                {'matrix', 'square_matrix', 'diagonal', 'triangle_upper', 'triangle_lower',
                 'constant', 'pseudo_scalar'}
                .intersection(self.left.attributes)
                .intersection(self.right.attributes)
            )

            # If either operand is zero, the result will be zero:
            if 'zero' in self.left.attributes or 'zero' in self.right.attributes:
                self.attributes.add('zero')

            # There are different cases for the shapes (matrix*matrix is covered in the general case):
            if 'matrix' in self.left.attributes and 'vector' in self.right.attributes: # M*v
                self.attributes.add('vector')
            elif 'covector' in self.left.attributes and 'matrix' in self.right.attributes: # c*M
                self.attributes.add('covector')
            elif 'covector' in self.left.attributes and 'vector' in self.right.attributes: # Inner product (c*v)
                self.attributes.add('scalar')
            elif 'vector' in self.left.attributes and 'covector' in self.right.attributes: # Outer product (v*c)
                self.attributes.add('matrix')
                # TODO: Outer products can cause symmetry, square matrix, pos_def etc. under the right circumstances.

            # TODO: pseudo_scalar can also propagate, but the resulting scalar is not trivial to determine anymore.
            # TODO: 'one' propagated weirdly. The shape and which kind of 'one' it is may matter.

            # 'M*' retains symmetry only under specific conditions.
            # One case that I can reasonably test is both inputs being the same:
            # Sadly, 'mathEquals' is pretty expensive, so I use exact equals for now. Will miss cases, obviously.
            if 'symmetric' in self.left.attributes and self.left == self.right:
                self.attributes.add('symmetric')

            # Check for special case: Outer product
            if self.left.isVector() and self.right.isCoVector():
                self.attributes.update({'one'}
                                       & self.left.attributes
                                       & self.right.attributes
                                       )

        elif self.name == 'S/':
            # Shape is based on the left child.
            self.shape = self.left.shape

            if 'zero' in self.right.attributes:
                print("Detected devision by zero. Execution is likely to result in an error.")

            # Certain properties can be propagated from the left child:
            self.attributes.update(
                {'scalar', 'vector', 'covector', 'matrix', 'square_matrix',
                 'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric',
                 'zero'
                 } & self.left.attributes
            )

            # Result will be one if both are one:
            if 'one' in self.left.attributes and 'one' in self.right.attributes:
                self.attributes.add('one')
                # I do not (yet?) track whether both are the same constant and even each other out.

            # Propagate or flip definity based on the scalar. Note that the scalar can't be 0 (division by 0).
            if 'pos_def' in self.right.attributes or 'pos_semi_def' in self.right.attributes:
                self.attributes.update(
                    {'pos_def', 'pos_semi_def', 'neg_def', 'neg_semi_def'} & self.left.attributes
                )
            if 'neg_def' in self.right.attributes or 'neg_semi_def' in self.right.attributes:
                self.attributes.update(flippedDefinities(self.left.attributes & {'pos_def', 'pos_semi_def', 'neg_def', 'neg_semi_def'}))

        elif self.name == 'u-':
            if self.isUnary():
                self.shape = self.left.shape

                # Certain properties can be propagated if the child has them:
                self.attributes.update(
                    {'scalar', 'vector', 'covector', 'matrix', 'square_matrix',
                     'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric', 'constant', 'zero'}
                        .intersection(self.left.attributes)
                )
                # TODO: I *could* track 'negative_one' to propagate 'one' past this.

                # pd, psd, nd and nsd are flipped (proof: the sign is a constant that can be pulled out).
                self.attributes.update(flippedDefinities({'pos_def', 'neg_def', 'pos_semi_def', 'neg_semi_def'}
                                                         & self.left.attributes))
            else:
                raise SemanticException(self, message='(Should have been unary!)')

        elif self.name in ['+', '-', '.*', './']:
            if not self.isBinary():
                raise SemanticException(self)

            # Make sure all Dimensions that aren't flat match:
            self.linkChildShapes()
            self.shape = self.left.shape

            # Certain properties can be propagated if both children have them:
            self.attributes.update(
                {'scalar', 'vector', 'covector', 'matrix', 'square_matrix',
                 'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric', 'constant', 'pseudo_scalar', 'zero'}
                    .intersection(self.left.attributes)
                    .intersection(self.right.attributes)
            )

            # TODO: There is much more that can be propagated for the specific cases:
            if self.name == '+':
                self.attributes.update(
                    {'pos_def', 'pos_semi_def', 'neg_def', 'neg_semi_def'}
                    .intersection(self.left.attributes)
                    .intersection(self.right.attributes)
                )
            elif self.name == '-':
                # TODO: Propagate 'one' and 'zero'. But remember that 'one' can be an eye or a matrix full of one. (Check diagonal tag)
                pass
            elif self.name == '.*':
                # Propagate if both children have these:
                self.attributes.update(
                    {'pos_semi_def', 'pos_def', 'neg_def', 'neg_semi_def', 'one'}
                    & self.left.attributes
                    &  self.right.attributes
                )

                # Propagate if either child has these:
                tmpSet = {'zero', 'diagonal', 'triangle_upper', 'triangle_lower'}
                self.attributes.update(
                    (tmpSet & self.left.attributes)
                     |
                    (tmpSet &  self.right.attributes)
                )

                # Special: opposed triangles make a diagonal.
                if ('triangle_lower' in self.left.attributes and 'triangle_upper' in self.right.attributes) or \
                        ('triangle_upper' in self.left.attributes and 'triangle_lower' in self.right.attributes):
                    self.attributes.add('diagonal')
            elif self.name == './':
                if 'zero' in self.right.attributes or ('scalar' not in self.right.attributes and
                        {'diagonal', 'triangle_upper', 'triangle_lower'}.issubset(self.right.attributes)):
                    print("Detected devision by zero. Execution is likely to result in an error.")

                # Result will be one if both are one (note that the zero case makes this impossible):
                elif 'one' in self.left.attributes and 'one' in self.right.attributes:
                    self.attributes.add('one')
                    # I do not (yet?) track whether both are the same constant and even each other out.

                # Definity is a whole mess. I'll just not touch that, thank you.
                # TODO: I guess I could add it if the right child is 1, but that's very specific.

        elif self.name == 'T':
            if self.isUnary():
                self.shape = self.left.shape[1], self.left.shape[0]
                # Though they are various cases in which transposing can be skipped, that's now the job of the
                # reshaper, not the constructor.
                
                # Retain the following properties:
                self.attributes.update(
                    {'scalar', 'matrix', 'diagonal', 'square_matrix',
                     'symmetric', 'constant', 'zero', 'one', 'pseudo_scalar',
                     'pos_def', 'pos_semi_def', 'neg_def', 'neg_semi_def'}
                     & self.left.attributes
                )

                # Certain properties cause other properties when transposed:
                if 'vector' in self.left.attributes:
                    self.attributes.add('covector')
                if 'covector' in self.left.attributes:
                    self.attributes.add('vector')
                if 'triangle_upper' in self.left.attributes:
                    self.attributes.add('triangle_lower')
                if 'triangle_lower' in self.left.attributes:
                    self.attributes.add('triangle_upper')
            else:
                raise SemanticException(self, message='(Should have been unary!)')

        elif self.name == 'S*':
            # make sure the/a scalar is the left child
            if not self.left.isScalar():
                self.left, self.right = self.right, self.left

            self.shape = self.right.shape

            # Certain properties can be propagated if the non-scalar has them:
            self.attributes.update(
                {'scalar', 'vector', 'covector', 'matrix', 'square_matrix',
                 'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric', 'pseudo_scalar'}
                    .intersection(self.right.attributes)
            )

            # Certain properties can be propagated if both children have them:
            self.attributes.update({'constant', 'one'} & self.left.attributes & self.right.attributes)
            # ... or either one:
            if 'zero' in self.left.attributes or 'zero' in self.right.attributes:
                self.attributes.add('zero')

            # Propagation of definity is a little more involved:
            definities = {'pos_def', 'pos_semi_def', 'neg_def', 'neg_semi_def'} & self.right.attributes
            if 'pos_def' in self.left.attributes: # ... preserve them as is.
                self.attributes.update(definities)
            if 'pos_semi_def' in self.left.attributes: # ... preserve them as is, but only the 'semi' ones.
                self.attributes.update({'pos_semi_def', 'neg_semi_def'} & definities)
            if 'neg_def' in self.left.attributes: # ... flip them.
                self.attributes.update(flippedDefinities(definities))
            if 'neg_semi_def' in self.left.attributes: # ... flip them, but only keep the 'semi' ones.
                self.attributes.update(flippedDefinities({'pos_semi_def', 'neg_semi_def'} & definities))

        elif self.name == '^':  # scalar^scalar or matrix^scalar == matrix*matrix*matrix... (square matrix only!)
            if self.isBinary():
                if self.left.isScalar() and self.right.isScalar():
                    self.shape = 1, 1
                    self.attributes.add('scalar') # All the implied attributes follow automatically; see end of function.
                elif self.left.isMatrix() and self.right.isScalar():
                    # The left operand must be a square matrix for this to work at all:
                    self.left.attributes.add('square_matrix')
                    self.left.shape[0].link(self.left.shape[1])

                    self.attributes.add('square_matrix')
                    self.shape = self.left.shape

                    # Certain properties can be propagated from the left operand:
                    self.attributes.update(
                        {'matrix', 'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric', 'constant'}
                            .intersection(self.left.attributes)
                    ) # TODO pseudo_scalar? psd? zero and one?
                else:
                    raise SemanticException(self, message="Did you mean .^ instead of ^?")
            else:
                raise SemanticException(self)

        elif self.name == '.^':  # scalar^something or something^scalar or something^same_dims
            # TODO pseudo_scalar? psd? zero and one?
            if not self.isBinary():
                raise SemanticException(self)

            # No matter which case, some things can be propagated right away:
            self.attributes.update(
                {'constant', 'pseudo_scalar'}
                    .intersection(self.left.attributes)
                    .intersection(self.right.attributes)
            )

            # --- Other .^ scalar (broadcast) ---
            if self.right.isScalar():
                self.shape = self.left.shape

                # Certain properties can be propagated from the non-scalar operand:
                self.attributes.update(
                    {'scalar', 'vector', 'covector', 'matrix', 'square_matrix',
                     'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric'}
                        .intersection(self.left.attributes)
                )

            # --- scalar (broadcast) .^ other ---
            elif self.left.isScalar():
                self.shape = self.right.shape

                # Certain properties can be propagated from the non-scalar operand (but NOT the same as above!)
                self.attributes.update(
                    {'scalar', 'vector', 'covector', 'matrix', 'square_matrix', 'symmetric'} # scalar ^ 0 is usually 1, so 0s are destroyed!
                        .intersection(self.right.attributes)
                )

            # --- neither is a scalar (dimensions must match) ---
            else:
                self.linkChildShapes()
                self.shape = self.left.shape

                # Certain properties can be propagated if both operands have them:
                self.attributes.update(
                    {'scalar', 'vector', 'covector', 'matrix', 'square_matrix',
                     'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric'}
                        .intersection(self.left.attributes)
                        .intersection(self.right.attributes)
                )

        elif self.name in ['norm1', 'norm2', 'sum']:
            if self.isUnary():
                self.shape = 1, 1
                self.attributes.add('scalar') # Related properties will be inferred later.
                self.attributes.update({'constant', 'zero'} & self.left.attributes)
            else:
                raise SemanticException(self)

        elif self.name in ['tr', 'det']:
            if self.isUnary() and self.left.isMatrix():
                self.left.attributes.add('square_matrix')
                self.left.shape[0].link(self.left.shape[1])
                self.shape = 1, 1

                self.attributes.add('scalar') # Related properties will be inferred later.
                self.attributes.update({'constant', 'zero'} & self.left.attributes)

                if self.name == 'det' and {'one', 'diagonal'}.issubset(self.left.attributes):
                    self.attributes.add('one') # Read: det(eye) == 1
            else:
                raise SemanticException(self)

        elif self.name in ['log', 'exp', 'cos', 'arccos']: # these do not preserve 0
            if self.isUnary():
                self.shape = self.left.shape

                # Certain properties can be propagated from the non-scalar operand (but NOT the same as above!)
                self.attributes.update(
                    {'scalar', 'vector', 'covector', 'matrix', 'square_matrix', 'symmetric', 'constant', 'pseudo_scalar'}
                        .intersection(self.left.attributes)
                )
            else:
                raise SemanticException(self)

        elif self.name in ['sin', 'cos', 'tan', 'abs', 'sign', 'relu', 'tanh', 'arcsin', 'arctan']: # Preserve 0.
            if self.isUnary():
                self.shape = self.left.shape

                # Certain properties can be propagated from the non-scalar operand (but NOT the same as above!)
                self.attributes.update(
                    {'scalar', 'vector', 'covector', 'matrix', 'square_matrix', 'symmetric', 'constant', 'pseudo_scalar',
                     'diagonal', 'triangle_upper', 'triangle_lower', 'zero'}
                        .intersection(self.left.attributes)
                )
            else:
                raise SemanticException(self)

        elif self.name in ['inv']:
            if self.isUnary() and self.left.isMatrix():
                # To invert a matrix, it must be square.
                self.left.attributes.add('square_matrix')
                self.left.shape[0].link(self.left.shape[1])
                self.attributes.add('square_matrix')
                self.shape = self.left.shape

                self.attributes.update(
                    {'matrix', 'constant', 'diagonal', 'triangle_upper', 'triangle_lower', 'symmetric'}
                        .intersection(self.left.attributes)
                )
            else:
                raise SemanticException(self)

        elif self.name in ['softmax']:
            if self.isUnary() and self.left.isMatrix():
                self.shape = self.left.shape

                self.attributes.update(
                    {'matrix', 'square_matrix', 'constant', 'symmetric', 'pseudo_scalar'}
                        .intersection(self.left.attributes)
                )
            else:
                raise SemanticException(self)

        elif self.name in ['vector', 'matrix']:
            # TODO: Change to generic 'broadcast' operation.
            if self.isUnary() and self.left.isScalar():
                # The following ugly multiplication will be cleaned up later in the process:
                if self.name == 'vector':
                    if shape == ():
                        shape = Dimension('unnamed_rows'), 1
                    shaped1 = ExpressionTree.quickExpLeaf('1', shape, {'vector', 'constant', 'pseudo_scalar', 'one'})
                else:
                    if shape == ():
                        shape = Dimension('unnamed_rows'), Dimension('unnamed_cols')
                    shaped1 = ExpressionTree.quickExpLeaf('1', shape, {'matrix', 'constant', 'pseudo_scalar', 'one'})
                self.setTo(self.left * shaped1)
            else:
                raise SemanticException(self)

        elif self.name == 'diag':
            if self.isUnary():
                self.attributes.update({'constant', 'pseudo_scalar', 'zero', 'one'} & self.left.attributes)

                if self.left.isVector():
                    self.shape = self.left.shape[0], self.left.shape[0]
                    self.attributes.update({'matrix', 'square_matrix', 'diagonal'})
                elif self.left.isCoVector():
                    self.shape = self.left.shape[1], self.left.shape[1]
                    self.attributes.update({'matrix', 'square_matrix', 'diagonal'})
                elif self.left.isMatrix():
                    self.name = 'diag2' # TODO: for prettification, this should be done in part1.
                    self.left.attributes.add('square_matrix') # Assume the child is square.
                    self.left.shape[0].link(self.left.shape[1])
                    self.attributes.add('vector')
                    self.shape = self.left.shape[0], 1
                else:
                    raise SemanticException(self)
            else:
                raise SemanticException(self)

        elif self.name == 'diag2':
            if not self.isUnary() or not self.left.isMatrix():
                raise SemanticException(self)
            self.left.attributes.add('square_matrix')
            self.left.shape[0].link(self.left.shape[1])
            self.attributes.add('vector')
            self.attributes.update({'constant', 'pseudo_scalar', 'zero', 'one'} & self.left.attributes)
            self.shape = self.left.shape[0], 1

        else:
            raise SemanticException(self, message="(No processing instructions for operator!)")

        ## Part 3: Clean up, propagate more properties, handle errors.
        self.removeUnnecessaryTranspose() # Does not propagate down the tree anymore! This is deliberate.
        # TODO: Maybe call other simplifications here instead of in code gen?
        
        ## Add properties that have requirements deeper in the tree (that is, grandchildren and beyond) ##
        # These involve reshaping, but at least the reshapes are already cached for later.
        if highLevelProps:
            if self.isBinary() and self.left.mathEquals(ExpressionTree('T', left=self.right, highLevelProps=False), verbosity=1):
                self.attributes.add('pos_semi_def') # If left mathEquals right.T, then we have an A.T * A type situation.

        ## Add inferrable properties if they haven't been added yet. NOTE: Order matters (so things can cascade) ##
        if 'scalar' in self.attributes:
            self.shape = 1, 1
            self.attributes.update({'diagonal', 'pseudo_scalar'}) # both triangle versions will automatically follow below.

        if 'diagonal' in self.attributes:
            self.attributes.update({'symmetric', 'triangle_upper', 'triangle_lower'})

        elif {'triangle_upper', 'triangle_lower'}.issubset(self.attributes) or \
                {'triangle_upper', 'symmetric'}.issubset(self.attributes) or \
                {'symmetric', 'triangle_lower'}.issubset(self.attributes):
            self.attributes.add('diagonal')

        if {'pos_def', 'neg_def'}.issubset(self.attributes):
            raise SemanticException(message=f"The tree {str(self)} was marked as positive definite and negative definite!")

        if 'pos_def' in self.attributes:
            self.attributes.add('pos_semi_def')
        elif 'neg_def' in self.attributes:
            self.attributes.add('neg_semi_def')

        if 'pos_semi_def' in self.attributes or 'neg_semi_def' in self.attributes:
            self.attributes.add('symmetric') # 'square_matrix' will follow from that.

        if {'square_matrix', 'pseudo_scalar'} in self.attributes:
            self.attributes.add('symmetric')

        if 'symmetric' in self.attributes and not 'scalar' in self.attributes:
            self.attributes.add('square_matrix')

        ## Part 4: Error checking (note that SemanticExceptions are now thrown above when and where they occur!) ##
        if len(self.shape) != 2:
            raise SemanticException(message=f"The node {str(self)} is missing shape information. This is an internal error.")

        # The following will warn me when I use undefined attributes (in practice mostly caused by typos):
        undefinedAttributes = self.attributes.difference(allAttributes)
        if undefinedAttributes:
            raise SyntaxError(f"{str(self)} was given unknown properties: {undefinedAttributes}!")

        # The following will warn me in case of no or multiple base attributes:
        baseAttributes = self.attributes & {'scalar', 'vector', 'covector', 'matrix'}
        if len(baseAttributes) == 0:
            raise SyntaxError(f"{str(self)} was mistakenly given no base properties!")
        elif len(baseAttributes) > 1:
            raise SyntaxError(f"{str(self)} was mistakenly given multiple base properties: {baseAttributes}!")

        # The following catches cases of properties that can only have been assigned my mistake.
        if not self.attributes & {'matrix', 'scalar'}:
            if self.attributes & {'diagonal', 'pos_def', 'pos_semi_def', 'neg_def', 'neg_semi_def', 'triangle_upper,'
                                                'triangle_lower', 'symmetric'}:
                raise SemanticException(self, message=f"(Contains attributes that only a matrix or scalar should have!)")

        if 'matrix' not in self.attributes:
            if self.attributes & {'square_matrix'}:
                raise SemanticException(self, message="(Contains attributes that only a matrix should have!)")

        if 'square_matrix' in self.attributes:
            # Just in case I didn't (which can happen; e.g. construct X[d] without specifying that it's square):
            self.shape[0].link(self.shape[1])

        if 'zero' in self.attributes or 'one' in self.attributes:
            if not 'pseudo_scalar' in self.attributes:
                raise SemanticException(self, message="(Zero or One should be a pseudo_scalar!)")

        if 'zero' in self.attributes and {'pos_def', 'neg_def'} & self.attributes:
            raise SemanticException(self, message="(A matrix with only zero entries CAN NOT be strictly definite.)")

        if len(self.attributes & {'pos_semi_def', 'neg_semi_def'}) > 1:
            raise SemanticException(self, "(contains contradictory attributes!)")

    def linkChildShapes(self):
        l, r = self.left, self.right
        if len(l.shape) != len(r.shape):
            raise SemanticException(self, message="(Shape mismatch!)")
        for x, y in zip(l.shape, r.shape):
            # Either they are both already 1, or they need to be linked.
            if x == 1:
                if y != 1:
                    raise SemanticException(self, message="(Shape mismatch!)")
            elif y == 1:
                if x != 1:
                    raise SemanticException(self, message="(Shape mismatch!)")
            else:
                x.link(y)

    def numericSize(self):
        return tuple(Dimension.toNumber(x) for x in self.shape)

    def copyCost(self):
        """
        The cost of copying this value, simplified as the amount of scalar entries in this scalar/vector/matrix.
        """
        a, b = self.numericSize()
        return a*b

    def children(self):
        if self.right:
            return [self.left, self.right]
        if self.left:
            return [self.left]
        return []

    def depth(self) -> int:
        if self.isLeaf():
            return 0
        elif self.isUnary():
            return 1 + self.left.depth()
        else:
            return 1 + max(self.left.depth(), self.right.depth())

    # This is a deep copy.
    def copy(self):
        t = copy.copy(self)
        t.attributes = self.attributes.copy()
        t.shape = self.shape # Deliberately not a copy - the dimensions remain the same, and that's as it should be.
        if self.left:
            t.left = self.left.copy()
        else:
            t.left = None
        if self.right:
            t.right = self.right.copy()
        else:
            t.right = None
        return t

    def shallowcopy(self):
        t = copy.copy(self)
        t.attributes = self.attributes.copy()
        t.shape = self.shape # Deliberately not a copy - the dimensions remain the same.
        if self.left:
            t.left = self.left # No copy, because shallow.
        else:
            t.left = None
        if self.right:
            t.right = self.right # No copy, because shallow.
        else:
            t.right = None
        return t

    def setTo(self, target):
        self.name = target.name
        self.shape = target.shape
        self.attributes = target.attributes
        self.left = target.left
        self.right = target.right

    def setToRetainDimension(self, target):
        """
        As setTo, but retains the node's original dimensions. This is useful during optimization when a node can be
        dropped and replaced with broadcasting (in which case the end result has unchanged dimensions).
        """
        s = self.shape
        self.setTo(target)
        self.shape = s

    #<editor-fold defaultstate="collapsed" desc="Convenience Constructors and growth of Trees">
    def add(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        return ExpressionTree('+', self, t)

    def __add__(self, t):
        return self.add(t)

    def mult(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        if self.isScalar() or t.isScalar():
            return ExpressionTree('S*', self, t)
        else:
            return ExpressionTree('*', self, t)

    def __mul__(self, t):
        return self.mult(t)

    def pow(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        return ExpressionTree('.^', self, t)

    def __pow__(self, t):
        return self.pow(t)

    def neg(self):
        return ExpressionTree('u-', self)

    def __neg__(self):
        return self.neg()

    def dot(self, t):
        return ExpressionTree('*', self, t)

    def div(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        if t.isScalar():
            return ExpressionTree('S/', self, t)
        else:
            return ExpressionTree('./', self, t)

    def __truediv__(self, t):
        return self.div(t)

    def sub(self, t):
        if isinstance(t, numbers.Number):
            t = Scalar(t)
        return ExpressionTree('-', self, t)

    def __sub__(self, t):
        return self.sub(t)

    def __rmul__(self, t):
        return self.mult(Scalar(t))

    def __radd__(self, t):
        return self.add(Scalar(t))

    def T(self):
        return ExpressionTree('T', self)

    def exp(self):
        return ExpressionTree('exp', self)

    def log(self):
        return ExpressionTree('log', self)

    def sin(self):
        return ExpressionTree('sin', self)

    def cos(self):
        return ExpressionTree('cos', self)

    def inv(self):
        return ExpressionTree('inv', self)

    def softmax(self):
        return ExpressionTree('softmax', self)

    def det(self):
        return ExpressionTree('det', self)

    def sum(self):
        return ExpressionTree('sum', self)

    def norm2(self):
        return ExpressionTree('norm2', self)
    #</editor-fold>

    # def isZero(self):
    #     if self.isNumeric():
    #         return float(self.name[4:])==0
    #         #return self.getNumeric() == 0
    #     else:
    #         return False

    # TODO: Superseded by the new property tags.
    def isOne(self):
        if self.isNumeric():
            return float(self.name[4:])==1
        elif self.name == "T":
                return self.left.isOne()
        else:
            return False

    def isDelta(self):
        return self.name == 'delta'

    def isScalar(self):
        return self.shape[0] == 1 and self.shape[1] == 1

    def isVector(self):
        return isinstance(self.shape[0], Dimension) and self.shape[1] == 1

    def isCoVector(self):
        return isinstance(self.shape[1], Dimension) and self.shape[0] == 1

    def isMatrix(self):
        return isinstance(self.shape[0], Dimension) and isinstance(self.shape[1], Dimension)

    def isLeaf(self):
        return (not self.left) and (not self.right)

    def isUnary(self):
        return self.left and not self.right

    def isBinary(self):
        return self.left and self.right

    numberPattern = re.compile(r"^([-+])?\d+(.\d+)?(([eE])([-+])?\d+)?$")
    def isNumeric(self, ignoreUnaryMinus = False):
        # check also for unary minus
        if not ignoreUnaryMinus and self.name == 'u-':
            return self.left.isNumeric()
        if not self.name.startswith('Var_'):
            return False
        return ExpressionTree.numberPattern.fullmatch(self.name[4:])

    def isVariable(self):
        return self.isLeaf() and (not self.isNumeric()) and not (self.isDelta())

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
            raise Exception("Who am I?")

    def getOrder(self):
        if self.isScalar():
            return 0
        if self.isVector():
            return 1
        return len(self.shape) # Note: This will return 2 for covectors, which may or may not be desired.

    def getNumeric(self):
        if self.name == 'u-':
            return '-'+(self.left.getNumeric())
        return self.name[4:]
        #s = self.name[4:]
        # maybe its an int
        #floatS = float(s)
        #intS = int(floatS)
        #if intS == floatS:
        #    return intS
        #return float(s)

    def swapChildren(self):
        assert(self.isBinary())
        self.left, self.right = self.right, self.left

    def contains(self, name) -> bool:
        for c in self.children():
            if c.contains(name):
                return True

        if self.name == name:
            return True
        return False

    def removeUnnecessaryTranspose(self):
        # If self.name == 'T', it is already and trivially handled in the constructor. This is for T children.
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
        if self.name == 'tr' and self.left.name == 'T':
            self.left.setTo(self.left.left)

    def replace(self, s, t):
        for c in self.children():
            c.replace(s, t)

        if self.name == s:
            assert(self.upper == t.upper)
            assert(self.lower == t.lower)
            self.setTo(t)

    def rename(self, s, t):
        for c in self.children():
            c.rename(s, t)

        if self.name == s:
            self.name = t

    def symbolTable(self):
        # TODO: Unused. Update or discard.
        _symbolTable = {}
        for v in self.children():
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
        
    def __linkDimensionsByUserRequest(self):
        """
        Links dimensions throughout the subtrees if the user specified that they should be the same.
        Dimensions that represent the same leaf are already the exact same object.
        Dimensions that should be linked based on operations and properties will be linked throughout the constructor
        wherever required.
        """
        if self.isBinary():
            leftDims = self.left.userNamedDims
            rightDims = self.right.userNamedDims
            selfDims = leftDims.copy()
            # We assume that the two individual sets have already been linked wherever necessary.
            for lalias, ldim in leftDims.items():
                for ralias, rdim in rightDims.items():
                    if lalias == ralias:
                        ldim.link(rdim)
                    else:
                        selfDims[ralias] = rdim
            return selfDims
        elif self.isUnary():
            return self.left.userNamedDims
        else: # Leaf
            dimsWithAlias = [(dim.getAlias(), dim) for dim in self.shape if 1 != dim and dim.getAlias()]
            if len(dimsWithAlias) == 2 and dimsWithAlias[0][0] == dimsWithAlias[1][0]:
                dimsWithAlias[0][1].link(dimsWithAlias[1][1])
            # TODO: I could replace all dim objects with reps while I'm here, but it's not really necessary.
            return dict(dimsWithAlias)
            
    def getDims(self, userNamedOnly=False) -> set:
        dimSet = set()
        if self.left:
            dimSet = self.left.getDims(userNamedOnly)
            if self.right:
                dimSet |= self.right.getDims(userNamedOnly)
        else:
            # Only leaves can have new dim objects; the rest of the tree inherits them.
            for dim in self.shape:
                if 1 == dim:
                    continue # We don't care about scalar dims.
                elif dim.getAlias() or not userNamedOnly:
                    dimSet.add(dim.getRepresentative())
        
        return dimSet
    
    def replaceDimsWithReps(self):
        """
        Minor maintenance. Not really required, but maybe I will bother to do it at some point.
        The parser does it when the tree is finished. Other than that, it's barely worth it.
        """
        if self.left:
            self.left.replaceDimsWithReps()
        if self.right:
            self.right.replaceDimsWithReps()
        
        h, w = self.shape
        if 1 != h:
            h = h.getRepresentative()
        if 1 != w:
            w = w.getRepresentative()
        self.shape = h, w
       
    def simpleString(self):
        s = 'TensorTree(' + "'" + str(self.name) + "'"
        for child in self.children():
            s += ', ' + child.simpleString()
        s += ', ' + str(self.shape[0]) + ', ' + str(self.shape[1]) + ')'
        return s

    def prettyString(self):
        return '\n'.join(self.__prettyListString())

    def __prettyListString(self):
        childStr = []
        maxChildrenHeight = 0
        childWidth = []
        for child in self.children():
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

        nameStr = self.shortDesc()
        nameWidth = len(nameStr)
        upperStr = ' ' * nameWidth + str(self.shape[0]) + ' '
        upperWidth = len(upperStr)
        lowerStr = ' ' * nameWidth + str(self.shape[1]) + ' '
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

    def varName(self) -> str:
        if self.name.startswith("Var_"):
            return self.name[4:]
        else:
            return self.name

    def shortDesc(self, writeName=True, writeDims=False, properties='minimalistic') -> str:
        props = self.attributes.copy()
        # Filter properties that obviously follow anyway (order matters!):
        if properties == 'minimalistic':
            if 'symmetric' in props:
                props.discard('square_matrix')
            if 'pos_semi_def' in props:
                props.discard('symmetric')
            if 'neg_def' in props:
                props.discard('pos_semi_def')
            elif 'pos_def' in props:
                props.discard('neg_semi_def')
            if 'diagonal' in props:
                props.difference_update({'symmetric', 'triangle_upper', 'triangle_lower'})
            if 'scalar' in props:
                props.difference_update({'pseudo_scalar', 'diagonal'})

        parts = []
        # Now begin!
        if writeName:
            parts.append(self.varName())

        if writeDims:
            if writeName:
                parts.append(f"({Dimension.toRef(self.shape[0])}, {Dimension.toRef(self.shape[1])})")
            else:
                parts.append(f"({Dimension.toNumber(self.shape[0])}, {Dimension.toNumber(self.shape[1])})")

        if properties and props:
            # Sorting is important to make sure hashes are consistent.
            shortSortedProps = [shortProp[p] for p in sorted(props)]
            parts.append(f"[{', '.join(shortSortedProps)}]")

        return "".join(parts)

    def __str__(self):
        primary = f"ExpressionTree {self.shortDesc()}"
        if self.isLeaf():
            note = "(leaf)"
        elif self.right:
            note = f"(binary: {self.left.shortDesc()}, {self.right.shortDesc()})"
        else:
            note = f"(unary: {self.left.shortDesc()})"
        return primary + " " + note
    
    def rootSimplify(self):
        """
        Remove obviously complicated and unhelpful bits at the root of the tree, such as double transpose.
        Beyond speeding things up a little bit, this also avoids the issue of the shapeCache remembering these old forms
        and indefinitely suggesting the bad form as a reshape option for the good form.
        """
        tree = self
        for reshape in simplifyReshapes:
            if reshape.matches(tree):
                tree = reshape.apply(tree)
        return tree
    
    def fullSimplify(self):
        tree = self.rootSimplify()
        if tree.left:
            tree.left = tree.left.rootSimplify()
        if tree.right:
            tree.right = tree.right.rootSimplify()
        return tree

    @DeprecationWarning
    def __lt__(self, other: 'ExpressionTree'):
        """
        Deprecated: It should still work, but was discontinued after I cut the entire "default tree" stuff.

        A tree A is "smaller than" another tree B if, assuming that both are children of the same parent node,
        with that node's operation being associative, A "should be" the left child.
        That is, tree comparison defines the order of child nodes in the default tree.

        NOTE (again) that in cases where
         a) the two nodes are not children of the same parent node
         or
         b) the two nodes are children of a parent node with a non-associative operation
        the return value of this function is absolutely meaningless.
        """
        # A scalar should be left of a non-scalar:
        if 'scalar' in self.attributes and not 'scalar' in other.attributes:
            return True
        # If only one matrix is diagonal, it should be on the left.
        elif 'diagonal' in self.attributes and not 'diagonal' in other.attributes:
            return True
        # If only one matrix is constant, it should be on the left.
        elif 'constant' in self.attributes and not 'constant' in other.attributes:
            return True
        # TODO: Add cases until every associative operation can be strictly sorted (assuming arbitrary, non-equal child trees)

        # If one is a leaf and the other not, put that leaf on the left:
        elif self.isLeaf() and not other.isLeaf():
            return True

        # Note: I will NOT use the names of variables to sort alphabetically - nor is that required.
        #  The reason for both is that variable names DO NOT influence the default tree. Varnames are mathematically
        #  inconsequential.
        # However, I *can* sort non-leaf nodes based on their names (which are operators and thus mathematically consequential).
        elif not (self.isLeaf() or other.isLeaf()):
            return self.name < other.name

        # ... that's it. self is not "smaller than" other.
        return False

    @DeprecationWarning
    def __ge__(self, other):
        return not self < other

    def allVars(self):
        if self.varSet is None:
            self.varSet = set()
            for c in self.children():
                self.varSet.update(c.allVars())
            if self.name.startswith('Var_') and not self.isNumeric():
                type = "???"
                if self.isMatrix():
                    type = "matrix"
                elif self.isVector():
                    type = "vector"
                elif self.isCoVector():
                    type = "covector"
                else:
                    type = "scalar"

                self.varSet.add((type, self.name[4:]))
        return self.varSet

    def toExpressionString(self, writeNames=True, writeDims=False, properties='minimalistic', knownVars=None):
        # Caching (separately for hashMode True and False) directly in the node
        if self.expString is None:
            self.expString = dict()

        # NOTE: The internal calls MUST NOT cache. That is because subtrees may have fewer variables.
        # Ex: "A" becomes "Var1", "B" becomes "Var1", but "A+B" must NOT become "Var1 + Var1", it must be "Var1 + Var2",
        #  which only works if the function is called on the whole tree! Loading and combining the subtree results would
        #  not be the same. This is unfortunate but can't be helped at this time.
        if (writeNames, writeDims, properties) not in self.expString:
            self.expString[(writeNames, writeDims, properties)] = self._toExpressionString(writeNames=writeNames, writeDims=writeDims, properties=properties, knownVars=knownVars)
        return self.expString[(writeNames, writeDims, properties)]

    # TODO: Set brackets only where required. Although it doesn't really hurt the caching as it is.
    def _toExpressionString(self, writeNames=True, writeDims=True, properties='minimalistic', knownVars=None):
        if knownVars is None:
            knownVars = dict()
        #Binary
        if self.left and self.right:
            l = self.left._toExpressionString(writeNames=writeNames, writeDims=writeDims, properties=properties, knownVars=knownVars)
            r = self.right._toExpressionString(writeNames=writeNames, writeDims=writeDims, properties=properties, knownVars=knownVars)
            s = f"({l} {str(self.name)} {r})"
        #Unary
        elif self.left:
            # special case T
            if self.name == "T":
                s = f"{self.left._toExpressionString(writeNames=writeNames, writeDims=writeDims, properties=properties, knownVars=knownVars)}'"
            else:
                s = f"{self.name}({self.left._toExpressionString(writeNames=writeNames, writeDims=writeDims, properties=properties, knownVars=knownVars)})"
        # Leaf
        else:
            # Write properties and dims only on the first occurrence (as in the input string).
            if self.name not in knownVars:
                if writeNames:
                    knownVars[self.name] = self.varName()
                    s = self.shortDesc(writeName=writeNames, writeDims=writeDims, properties=properties)
                else:
                    knownVars[self.name] = "V" + str(len(knownVars))
                    s = f"{knownVars[self.name]}{self.shortDesc(writeName=writeNames, writeDims=writeDims, properties=properties)}"
            else:
                s = knownVars[self.name]

        return s

### Hashing and Equality ###
    def __hash__(self) -> int:
        """
        Generates a hash based on the EXACT tree, preserving even variable names.

        Note that the tree should not be changed after the hash has been calculated, or things will go wrong.
        You can make new nodes and reuse old child nodes within them; that's fine (and intended; see reshape.py).
        """
        return hash(self.expStrNamed())

    # Convenience wrappers for hash string functions. These are used for different things.
    def expStrNamed(self) -> str:
        """
        Returns the expression as a string. Uses variable names. Does not use shapes or properties (although nothing would
        break if it did).
        This is used for caching subexpressions for CSE, since the exact variables must be the same.
        """
        return self.toExpressionString(writeNames=True, writeDims=False, properties=False)

    def expStrLenient(self) -> str:
        """
        Returns the expression as a string. Variable names are removed (although if a variable appears more than once it
        will get the same codename every time). Shape is disregarded and only properties are used to identify the variables.
        This is used for caching reshapes, since only the mathematical attributes of the variables matter.
        """
        return self.toExpressionString(False, False, properties='full')

    def expStrShaped(self) -> str:
        """
        Returns the expression as a string. Variable names are removed (although if a variable appears more than once it
        will get the same codename every time). Shape and properties are used to identify the variables.
        This is used for caching solutions, since the best solution for a tree will only be guaranteed to remain
        the best solution for a different tree if variable dimensionality does NOT change.
        """
        return self.toExpressionString(False, True, properties='full')

    def __eq_Alternative(self, other: 'ExpressionTree', ignoreLeafNames = False, ignoreDims = False):
        # I though this one might actually be better, but it seems to be VERY significantly worse.
        #  Possibly due to bad string concatenation? But even with that improved a little it still sucks.
        """
        Note: This may behave weirdly when comparing ExpTrees where a variable with the same name reappears, but uses
        different shapes/properties. Of course, that's not something LinEx ever does, so I'm not going to fix it.
        """

        if not hasattr(other, 'toExpressionString'):
            return False

        return (self.toExpressionString(not ignoreLeafNames, not ignoreDims, 'minimalistic') ==
                other.toExpressionString(ignoreLeafNames, not ignoreDims, 'minimalistic'))

    def __eq__(self, other: 'ExpressionTree', ignoreLeafNames = False, ignoreDims = False):
        """
        Note: This may behave weirdly when comparing ExpTrees where a variable with the same name reappears, but uses
        different shapes/properties. Of course, that's not something LinEx ever does, so I'm not going to fix it.
        """

        if not isinstance(other, ExpressionTree):
            return False

        # Check content
        if not (((ignoreLeafNames and self.isLeaf()) or self.name == other.name) and
               self.shape[0] == other.shape[0] and
               self.shape[1] == other.shape[1] and
               self.attributes == other.attributes):
            return False

        # Check children:
        if self.isLeaf():
            return other.isLeaf()
        if self.isUnary():
            return other.isUnary() and self.left == other.left
        if self.isBinary():
            return other.isBinary() and self.left == other.left and self.right == other.right
        
    def mathEquals(self, other, shapeCache: EquivalencyCache = None, timeLimit=.5, verbosity=4, indent=""):
        """
        Unlike the __eq__ function, which returns whether the tree results in the same expression, this tests
        whether the mathematical result of the tree is the same. This is done by trying to reshape the other tree into
        this tree and additionally making sure all leaves match (by their names, not dimensions).
        """
        # If we have a cache, that cache MIGHT know that these trees are equal.
        if shapeCache is not None:
            if shapeCache.getRepresentative(self) == shapeCache.getRepresentative(other):
                return True
            # If they are not known to be equal, that does not mean they definitely aren't!
        
        symbolicSelf = SymbolTree.fromExpTree(self, checkLeafName=True)
        for res in ExpressionForester.tryModifyTree(other, symbolicSelf, forceEndTime=time.time()+timeLimit*60,
                                                    verbosity=verbosity, indent=indent):
            if symbolicSelf.matches(res):
                if shapeCache is not None:
                    shapeCache.link(self, other)
                return True # No need to care about other options to reach the same shape.

        return False

    def isTooComplicated(self):
        """
        A tree is 'too complicated' when it stacks operations in a useless way. For example, a double transpose might still
        be useful to enable some other reshapes that rely on transposition, but a triple transpose has never helped anyone.
        
        TODO: The issue is that together with other useless operations (multiplying by 1, for example), the program could
         stack interleaved operations indefinitely, even though they would still be perfectly useless.
         I have yet to find a really good (meaning: not expensive) way to filter that.
        """
        trippleTranspose = SymbolTree('T', SymbolTree('T', SymbolTree('T', SymbolTree('X'))))
        return trippleTranspose.matches(self) or \
               (self.left and self.left.isTooComplicated())or \
               (self.right and self.right.isTooComplicated())

    @staticmethod
    def quickExpLeaf(name, shape, props, highLevelProps=False):
        return ExpressionTree('Var_' + name, shape=shape, attributes=props, highLevelProps=highLevelProps)
    
    def isPureTree(self, operationsAllowed=None, attributesRequired=None, attributesForbidden=None):
        """
        Determines whether:
            - every operation in the tree is one of the allowed operations AND
            - every node in this tree has ALL of the required attributes AND
            - no node in this tree has ANY of the forbidden attributes.
        """
        
        # Make sets if I haven't passed them as sets:
        if isinstance(operationsAllowed, str):
            operationsAllowed = {operationsAllowed}
        elif operationsAllowed is None:
            operationsAllowed = set()
        if isinstance(attributesRequired, str):
            attributesRequired = {attributesRequired}
        elif attributesRequired is None:
            attributesRequired = set()
        if isinstance(attributesForbidden, str):
            attributesForbidden = {attributesForbidden}
        elif attributesForbidden is None:
            attributesForbidden = set()
        # Check this tree:
        if (self.name in operationsAllowed
                and attributesRequired.issubset(self.attributes)
                and not attributesForbidden.intersection(self.attributes)):
            # Check children:
            return all(ch.isPureTree for ch in self.children())
        else:
            return False
            
### Convenience Constructor(s) ###
class Scalar(ExpressionTree):
    def __init__(self, name, attributes=None):
        if not attributes:
            attributes = set()
        attributes.update({'scalar', 'pseudo_scalar','diagonal', 'triangle_upper', 'triangle_lower'})
        ExpressionTree.__init__(self, 'Var_' + str(name), shape=(1,1), attributes=attributes)

class Vector(ExpressionTree):
    def __init__(self, name, attributes=None):
        if not attributes:
            attributes = set()
        attributes.update({'vector'})
        if 'constant' in attributes:
            dim = Dimension('unnamed_rows') # Numeric values and deltas don't have a name to refer to.
        else:
            dim = Dimension(origin=name+'_rows')
        ExpressionTree.__init__(self, 'Var_' + str(name), shape=(dim, 1), attributes=attributes)

class Matrix(ExpressionTree):
    def __init__(self, name, attributes=None):
        if 'constant' in attributes:
            shape = Dimension('unnamed_rows'), Dimension('unnamed_cols')
        else:
            shape = Dimension(origin=name+'_rows'), Dimension(origin=name+'_cols')

        # In case of square matrices, the two dimensions will be linked as normal in the super constructor.

        if not attributes:
            attributes = set()
        attributes.add('matrix')
        ExpressionTree.__init__(self, 'Var_' + str(name), shape=shape, attributes=attributes)

### Static functions ###
def flippedDefinities(s:set) -> set:
    res = s.copy()
    if 'pos_def' in res:
        res.remove('pos_def')
        res.add('neg_def')
    if 'pos_semi_def' in res:
        res.remove('pos_semi_def')
        res.add('neg_semi_def')
    if 'neg_def' in res:
        res.remove('neg_def')
        res.add('pos_def')
    if 'neg_semi_def' in res:
        res.remove('neg_semi_def')
        res.add('pos_semi_def')
    return res

class SemanticException(Exception):
    def __init__(self, node: ExpressionTree = None, message='', position=None):
        if node is not None:
            operator = node.name
            rightOp = node.right.getType() if node.right else None
            leftOp = node.left.getType() if node.left else "missing argument"
            message = SemanticException.assembleSemanticMessage(operator, leftOp, rightOp) + ' ' + message
        self.message = message + (f" @pos:{position}" if position else '')

    def __str__(self):
        return self.message

    @staticmethod
    def assembleSemanticMessage(operator, left, right):
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

class ExpressionForester():
    availableReshapes = allReshapes
    
    reshapeCache = ReshapeCache() # For abstracted applicability of reshapes; useful across runs

    @classmethod
    def pickleCache(cls, path='./reshapeCache.pyc'):
        with open(path, 'w') as file:
            pickle.dump(cls.reshapeCache, file)
        print(f"Saved cache as {path}.")

    @classmethod
    def unpickleCache(cls, path='./reshapeCache.pyc'):
        if not os.path.isfile(path):
            print(f"Could not load cache: {path} does not exist.")
            return
        with open(path) as file:
            cls.reshapeCache = pickle.load(file)
        print(f"Loaded cache from {path}.")

    @classmethod
    def setAvailableReshapes(cls, reshapes):
        cls.availableReshapes = reshapes
        
    # These are the current versions. Note that together with the shape Cache, they have the potential issue of
    #  skipping some shapes (if the first destination tree didn't consider those).
    @classmethod
    def tryModifyTree(cls, tree, destination: SymbolTree, shapeCache: EquivalencyCache = None,
                      forceEndTime=None,
                      verbosity: int = 3, indent=""):
        """
        Tries every possible (and sensible) reshaping option to reach the target tree.
        Returns a reshaped tree.
        The input tree will NOT be modified.

        Note that there is one thing about reshapes that this function can not currently handle: If multiple child trees
        have to be (effectively) the same, this function will currently (TODO?) also return versions where that is not so.
        Those have to be filtered externally.
         -> Simplest (but not ideal) solution: Test every value before yield and make sure it really matches. Skip otherwise.
            This can just as well be done outside the function, though.
        
        The argument rootShapesThusFar ensures that subtrees aren't reshaped into a shape that could already have been
        achieved at this or a higher level (since that would have been the more sensible option all along).
        Example: A'B -> (B'A)' is valid, but then doing B'A -> (A'B)' on the child level would yield (A'B)'' in total,
        which is trivially more complicated than where we started out.
        """
        if forceEndTime and time.time() >= forceEndTime:
            return 'timeout!'
        
        childIndent = f"{indent}│\t"
        if shapeCache is None:
            shapeCache = EquivalencyCache()
        
        # Before anything else, filter leaves (they can't be reshaped, anyway - either they fit, or not.)
        # Similarly, if all we're *looking for* is a leaf, then the matched tree will be reshaped later, no need to go
        # through variations now.
        if tree.isLeaf() or destination.isLeaf():
            if destination.matches(tree, verbosity=verbosity-1, indent=childIndent):
                if verbosity > 2: print(f"{indent}Accepted {str(tree)} as {str(destination)}!")
                yield tree
            else:
                if verbosity > 2: print(f"{indent}The tree {str(tree)} did not fit {str(destination)}!")
            return
        else:
            if verbosity > 2 and destination.matches(tree): print(f"{indent}Unmodified tree works!")
            if verbosity > 2: print(f"{indent}Searching for ways to reshape {str(tree)} into {str(destination)}.")

        ## If we got here, reshaping this tree could be very costly. Meaning we might have cached it:
        if tree in shapeCache:
            yield from shapeCache.getAllShapes(tree)
            return

        ## If we got here, this tree was not cached and we have to do all the hard work:
        #  Part 1 - Are the attributes correct?
        if not destination.couldMatch(tree, verbosity=verbosity, indent=childIndent):
            # Avoid a whole lot of work if there is no way to get it right.
            if verbosity > 2: print(f"{indent}Conclusion: Decided it was trivially impossible.")
            return

        #  Part 2 - Get the root operation to work:
        # >> Looking for all possible reshapes with the correct result. Includes circles, which must be filtered.
        rootShapesThusFar = {tree}
        rootShapes2Consider = list(rootShapesThusFar)
        
        solutions = 0
        while rootShapes2Consider:
            option = rootShapes2Consider.pop()
            # >> Queue trees that can be reached with a single top level reshape:
            for nextOption in cls._singleRootReshape(option, shapeCache=shapeCache, forceEndTime=forceEndTime,
                                                     verbosity=verbosity, indent=childIndent):
                #if nextOption.isTooComplicated(): # Not needed (I currently don't allow double transpose)
                #    continue
                if nextOption not in rootShapesThusFar:
                    rootShapesThusFar.add(nextOption) # Avoid considering trees more than once
                    shapeCache.link(nextOption, tree)
                    rootShapes2Consider.append(nextOption)

            # Check whether the reshape actually resulted in the desired root:
            if option.name == destination.name:
                # Part 3 - find all ways (if any) to make the child nodes fit.
                childOptions = 0
                for result in cls._tryModifyChildren(option, destination, shapeCache=shapeCache, forceEndTime=forceEndTime,
                                                     verbosity=verbosity, indent=childIndent):
                    if destination.matches(result): # This is necessary (see 'tryModifyTree' documentation for details).
                        solutions += 1
                        childOptions += 1
                        if verbosity > 2: print(f"{indent}Found solution #{solutions}!")
                        yield result
                    
                if childOptions == 0 and verbosity > 3:
                    print(f"{indent}Root worked, but there was no way to also satisfy the child nodes!")
            else: # the tree is only a stepping stone but no potential solution itself.
                if verbosity > 3: print(f"{indent}Result was no match, but will try reshaping further from here.")

        if verbosity > 2: print(f"{indent}Conclusion: Encountered {solutions} solution(s)!")

        return

    @classmethod
    def _singleRootReshape(cls, tree, shapeCache: EquivalencyCache, forceEndTime,
                           verbosity: int = 2, indent=""):
        """
        Generates all ExpressionTrees that can be reached from the input tree by applying a single reshape operation on
        the root node. This may require arbitrarily many reshapes on the child nodes.

        The input tree itself is NOT among the generated returns.
        """
        childIndent = f"{indent}│\t"

        if verbosity > 3: print(f"{indent}Now reshaping from {str(tree)}:")

        # Try all reshapes, starting with those that already have the correct operation in the root.
        for potentialReshape in sorted(ExpressionForester.availableReshapes, key=lambda reshape: 0 if reshape.requires.name == tree.name else 1):
            cachedResult = cls.reshapeCache.lookup(tree, potentialReshape)

            # If either the tree or the specific reshape aren't cached yet - so we have to calculate and add them.
            if cachedResult in ['unknown Tree', 'undetermined', 'direct']:
                if verbosity > 2: print(f"{indent}Considering {str(potentialReshape)}.")

                # Rule out hopeless cases rather than doing the work:
                if 'direct' != cachedResult and not potentialReshape.requires.couldMatch(tree):
                    cls.reshapeCache.cache(tree, potentialReshape, result='futile')
                    continue

                # If we got here, there is some hope at least. Maybe child reshapes can make this reshape work.
                optionCount = 0
                req = potentialReshape.requires
                if req.couldMatch(tree) and \
                        (not req.left or req.left.couldMatch(tree)) and \
                        (not req.right or req.right.couldMatch(tree)):
                    waysToMakeItWork = cls._tryModifyChildren(tree, potentialReshape.requires, shapeCache=shapeCache,
                                                              forceEndTime=forceEndTime,
                                                              verbosity=verbosity, indent=childIndent)

                    if forceEndTime and time.time() >= forceEndTime:
                        return 'timeout!' # Yes, this has to be here instead of at the top of the function.
                    
                    for wayToMakeItWork in waysToMakeItWork:
                        if not potentialReshape.matches(wayToMakeItWork):
                            continue  # This can happen (see tryModifyTree description string)
                        option = potentialReshape.apply(wayToMakeItWork)
                        optionCount += 1
                        yield option
                
                if optionCount > 0:
                    if verbosity > 2: print(f"{indent}Conclusion: Discovered {optionCount} options overall!")
                    cls.reshapeCache.cache(tree, potentialReshape, 'direct') # Not that it makes much of a difference.
                else:
                    # If it didn't work even though we had no other reshapes blacklisted, then it can't work at all.
                    if verbosity > 2: print(f"{indent}Conclusion: Impossible! (Result cached)")
                    cls.reshapeCache.cache(tree, potentialReshape, 'indirect')

            # Skip reshapes that have been marked as impossible to reach by a single reshape step:
            elif cachedResult in ['futile', 'indirect']:
                if verbosity > 3: print(f"{indent}Skipping {str(potentialReshape)}: It was marked as '{cachedResult}'.")
                # In practice, I no longer bother to actually determine 'futile'.
                # It still just comes down to not bothering to try in either case.
                continue
            else:
                raise ValueError(f"Invalid return from the ReshapeCache: {str(cachedResult)}. (This is an internal error.)")
            
    @classmethod
    def _tryModifyChildren(cls, tree, destination: SymbolTree, shapeCache: EquivalencyCache,
                           forceEndTime,
                           verbosity: int = 3, indent=""):
        """
        Generates all shapes (left, right) or single shapes (if unary) of children of this tree that would fit the
        respective children of the destination tree. In the binary case, all possible combinations are generated.
        The return comes in the form of the rebuilt tree, i. e. the new root node is passed back, not the child(ren).

        The originally passed tree is not modified. Child shapes that would be equivalent to a shape that could have
        been achieved higher in the tree are skipped (because they would be trivially more expensive).
        """
        if forceEndTime and time.time() >= forceEndTime:
            return 'timeout!'
        
        if not destination.left:
            if verbosity > 2: print("Info: Asked to reshape children of a tree when there was nothing to change.")
            return tree

        elif not destination.right:
            if tree.isUnary():
                for l in cls.tryModifyTree(tree.left, destination.left, shapeCache=shapeCache,
                                           forceEndTime=forceEndTime,
                                           verbosity=verbosity, indent=indent):
                    # This function can't remove properties, so I can just copy them over rather than recalculating:
                    et = ExpressionTree(tree.name, left=l, highLevelProps=False)
                    et.attributes.update(tree.attributes)
                    yield et
            return

        elif destination.left and destination.right:
            if tree.isBinary():
                rightList = cls.tryModifyTree(tree.right, destination.right, shapeCache=shapeCache,
                                               forceEndTime=forceEndTime,
                                               verbosity=verbosity, indent=indent)

                if not rightList:
                    # If there is no option for the right child, then there is no option for a pair of children.
                    return

                for l in cls.tryModifyTree(tree.left, destination.left, shapeCache=shapeCache,
                                           forceEndTime=forceEndTime,
                                           verbosity=verbosity, indent=indent):
                    
                    # Build every possible pair:
                    for r in rightList:
                        # This function can't remove properties, so I can just copy them over rather than recalculating:
                        et = ExpressionTree(tree.name, left=l, right=r, highLevelProps=False)
                        et.attributes.update(tree.attributes)
                        yield et
            return

    # These are replacer versions for the three functions above. I don't have time to make them work, but I suspect they
    # may eventually lead to a better solution.
    @classmethod
    def allTreeShapes(cls, tree, depth=None, shapeCache: EquivalencyCache = None,
                      verbosity: int = 3, indent=""):
        """
        Tries every possible reshaping option to generate all equivalent trees that differ within the given depth.
        For example, with a depth of two, all shapes that differ in the root or the root's children will
        be returned, but trees that differ only below that won't be considered separately.

        Note that reshapes can still be applied lower down in the tree to make the top reshapes possible.
        """
        childIndent = f"{indent}│\t"
        if depth is None:
            depth = tree.depth()
        if shapeCache is None:
            shapeCache = EquivalencyCache()

        # Before anything else, filter leaves (they can't be reshaped, anyway - either they fit, or not.)
        # Similarly, if all we're *looking for* is a leaf, then the matched tree will be reshaped later, no need to go
        # through variations now.
        if tree.isLeaf() or depth < 1:
            if verbosity > 2: print(f"{indent}{str(tree)} did not need further reshaping at this time.")
            yield tree
            return
        else:
            if verbosity > 2: print(f"{indent}Searching for ways to reshape {str(tree)} within depth {str(depth)}.")

        ## If we got here, reshaping this tree could be very costly. We might have cached it:
        if tree in shapeCache:
            yield from shapeCache.getAllShapes(tree)  # TODO: What if it was originally cashed with less depth?
            return

        #  Part 2 - Get the root operation to work:
        # >> Looking for all possible reshapes with the correct result. Includes circles, which must be filtered.
        rootShapes2Consider = [tree]
        rootShapesThusFar = OrderedSet(rootShapes2Consider)  # The order is optional, but helps with debugging.

        solutions = 0
        while rootShapes2Consider:
            option = rootShapes2Consider.pop()
            # >> Queue trees that can be reached with a single top level reshape:
            for nextOption in cls._allRootShapes(option, shapeCache=shapeCache,
                                                 verbosity=verbosity, indent=childIndent):
                if nextOption not in rootShapesThusFar:
                    rootShapesThusFar.add(nextOption)  # Avoid considering trees more than once
                    shapeCache.link(nextOption, tree)
                    rootShapes2Consider.append(nextOption)
    
            # Part 3 - find all ways (if any) to reshape the children.
            childOptions = 0
            for result in cls._allChildShapes(option, depth, shapeCache=shapeCache,
                                              verbosity=verbosity, indent=childIndent):
                solutions += 1
                childOptions += 1
                if verbosity > 2: print(f"{indent}Found shape #{solutions}!")
                yield result

        if verbosity > 2: print(f"{indent}Conclusion: Encountered {solutions} shape(s)!")

        return

    @classmethod
    def _allRootShapes(cls, tree, shapeCache: EquivalencyCache, verbosity: int = 2, indent=""):
        """
        Generates all ExpressionTrees that can be reached from the input tree by applying a single reshape operation on
        the root node. This may require arbitrarily many reshapes on the child nodes.

        The input tree itself is NOT among the generated returns.
        """
        childIndent = f"{indent}│\t"

        if verbosity > 3: print(f"{indent}Now reshaping from {str(tree)}:")

        # Try all reshapes, starting with those that already have the correct operation in the root.
        for potentialReshape in sorted(ExpressionForester.availableReshapes,
                                       key=lambda reshape: 0 if reshape.requires.name == tree.name else 1):
            cachedResult = cls.reshapeCache.lookup(tree, potentialReshape)
    
            # If either the tree or the specific reshape aren't cached yet - so we have to calculate and add them.
            if cachedResult in ['unknown Tree', 'undetermined', 'direct']:
                if verbosity > 2: print(f"{indent}Considering {str(potentialReshape)}.")
        
                # Rule out hopeless cases rather than doing the work:
                if 'direct' != cachedResult and not potentialReshape.requires.couldMatch(tree):
                    cls.reshapeCache.cache(tree, potentialReshape, result='futile')
                    continue
        
                # If we got here, there is some hope at least. Maybe child reshapes can make this reshape work.
                optionCount = 0
        
                req = potentialReshape.requires
                if req.couldMatch(tree) and \
                        (not req.left or req.left.couldMatch(tree)) and \
                        (not req.right or req.right.couldMatch(tree)):
                    # Note: Depth should ultimately only be the depth of the potential reshape. The issue, however,
                    #  is that I cache the results, so they will later also have to work for other depths ...
                    #  TODO: Could use the maximum depth of all loaded kernels and reshapes.
                    waysToMakeItWork = cls._allChildShapes(tree, depth=None, shapeCache=shapeCache,
                                                           verbosity=verbosity, indent=childIndent)
                    for wayToMakeItWork in waysToMakeItWork:
                        if not potentialReshape.matches(wayToMakeItWork):
                            continue  # This one didn't help ...
                        option = potentialReshape.apply(wayToMakeItWork)
                        optionCount += 1
                        yield option
        
                if optionCount > 0:
                    if verbosity > 2: print(f"{indent}Conclusion: Discovered {optionCount} options overall!")
                    cls.reshapeCache.cache(tree, potentialReshape,
                                           'direct')  # Not that it makes much of a difference.
                else:
                    # If it didn't work even though we had no other reshapes blacklisted, then it can't work at all.
                    if verbosity > 2: print(f"{indent}Conclusion: Impossible! (Result cached)")
                    cls.reshapeCache.cache(tree, potentialReshape, 'indirect')
    
            # Skip reshapes that have been marked as impossible to reach by a single reshape step:
            elif cachedResult in ['futile', 'indirect']:
                if verbosity > 3: print(
                    f"{indent}Skipping {str(potentialReshape)}: It was marked as '{cachedResult}'.")
                # In practice, I no longer bother to actually determine 'futile'.
                # It still just comes down to not bothering to try in either case.
                continue
            else:
                raise ValueError(
                    f"Invalid return from the ReshapeCache: {str(cachedResult)}. (This is an internal error.)")

    @classmethod
    def _allChildShapes(cls, tree, depth: int, shapeCache: EquivalencyCache,
                        verbosity: int = 3, indent=""):
        """
        Generates all shapes (left, right) or single shapes (if unary) of children of this tree that would fit the
        respective children of the destination tree. In the binary case, all possible combinations are generated.
        The return comes in the form of the rebuilt tree, i. e. the new root node is passed back, not the child(ren).

        The originally passed tree is not modified. Child shapes that would be equivalent to a shape that could have
        been achieved higher in the tree are skipped (because they would be trivially more expensive).
        """
        if depth is None:
            depth = tree.depth()
        if depth < 1:
            if verbosity > 2: print(f"{indent}Info: Depth is too small; not reshaping children.")
            return tree

        if tree.isUnary():
            for l in cls.allTreeShapes(tree.left, depth=depth - 1, shapeCache=shapeCache,
                                       verbosity=verbosity, indent=indent):
                # This function can't remove properties, so I can just copy them over rather than recalculating:
                et = ExpressionTree(tree.name, left=l, highLevelProps=False)
                et.attributes.update(tree.attributes)
                yield et

        elif tree.isBinary():
            rightList = [r for r in cls.allTreeShapes(tree.right, depth=depth - 1, shapeCache=shapeCache,
                                                      verbosity=verbosity, indent=indent)]
    
            if not rightList:
                return  # If there is no option for the right child, then there is no option for a pair of children.
    
            for l in cls.allTreeShapes(tree.left, depth=depth - 1, shapeCache=shapeCache,
                                       verbosity=verbosity, indent=indent):
        
                # Build every possible pair:
                for r in rightList:
                    # This function can't remove properties, so I can just copy them over rather than recalculating:
                    et = ExpressionTree(tree.name, left=l, right=r, highLevelProps=False)
                    et.attributes.update(tree.attributes)
                    yield et
        return