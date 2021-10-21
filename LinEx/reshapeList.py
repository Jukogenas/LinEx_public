from LinEx.symboltree import SymbolTree
from LinEx.reshape import Reshape

# Note: 'skipIf' defaults to 'lambda tree: False', meaning reshapes are usually not skipped. As it should be.
# When deriving reshapes from other reshapes (using .reverse() or .substitute()) the default is to keep the old 'skipIf'.

# Also, it may be a good idea to explicity write down operations that DON'T have a given property to differentiate them
# from ones that I simply haven't bothered to add yet.

allReshapes = []
simplifyReshapes = [] # These should be applied as early as possible to simplify the tree before doing difficult work.

#Commutativity ========================================================================================================
## Template ##
commutativity = Reshape(
    requires=SymbolTree('OPERATION', SymbolTree('X'), SymbolTree('Y')), # X+Y
    target=SymbolTree('OPERATION', SymbolTree('Y'), SymbolTree('X')), # Y+X
    skipIf=lambda tree: tree.right == tree.left # Technically, could skip if they are mathematically equal, but that is an expensive check!
)

## + ##
allReshapes.append(commutativity.substitute('OPERATION', '+')) # Commutativity of sum
allReshapes.append(commutativity.substitute('OPERATION', 'S*')) # Commutativity of S*
allReshapes.append(commutativity.substitute('OPERATION', '.*')) # Commutativity of .*

# Commutativity of matrix product between a diagonal and a square matrix:
commuDiagMatProd = Reshape(
    requires=SymbolTree('M*', attributes={'matrix'},
                        left=SymbolTree('X', attributes={'square_matrix'}),
                        right=SymbolTree('Y', attributes={'diagonal'})), # X*Y[d]
    target=SymbolTree('M*', attributes={'matrix'},
                      left=SymbolTree('Y', attributes={'diagonal'}),
                      right=SymbolTree('X', attributes={'square_matrix'})), # Y*X
)
allReshapes.append(commuDiagMatProd)
allReshapes.append(commuDiagMatProd.reverse())

## Not commutative: -, M* (general case), S/, ./, every non-binary operation ##

#Associativity ========================================================================================================
## Templates ##
associativityLR = Reshape(
    requires=SymbolTree('OUTER', SymbolTree('X'), SymbolTree('INNER', SymbolTree('Y'), SymbolTree('Z'))), # X + (Y+Z)
    target=SymbolTree('INNER', SymbolTree('OUTER', SymbolTree('X'), SymbolTree('Y')), SymbolTree('Z')), # (X+Y) + Z
)

## + ##
# + with +
assoSumOtherLR = associativityLR.substitute('OUTER', '+')
assoSumSumLR = assoSumOtherLR.substitute('INNER', '+')
allReshapes.append(assoSumSumLR)
allReshapes.append(assoSumSumLR.reverse())

## M*, S* and .* are associative ###
for m in ["M*", "S*", ".*"]:
    temp = associativityLR.substitute("OUTER", m)
    temp.substituteNoCopy("INNER", m)
    allReshapes.append(temp)
    allReshapes.append(temp.reverse())

#Pseudo-Associativity =================================================================================================
# + with - on the right/inside (but NOT with - on the left/outside):
assoSumDifLR = assoSumOtherLR.substitute("Inner", "-")
allReshapes.append(assoSumDifLR)
allReshapes.append(assoSumDifLR.reverse())

# Mixing S* and M* (.* can't be added in that mix): Because S* is already commutative, avoid issues by assuming it's on the left.
assoMatMulScalarMul = Reshape(
    requires=SymbolTree('S*', # a[s] S* (Y M* Z)
                        left=SymbolTree('a', attributes={'scalar'}),
                        right=SymbolTree('M*', SymbolTree('Y'), SymbolTree('Z'))),
    target=SymbolTree('M*', # (a S* Y) * Z
                      left=SymbolTree('S*', SymbolTree('a', attributes={'scalar'}), SymbolTree('Y')),
                      right=SymbolTree('Z')),
)
allReshapes.append(assoMatMulScalarMul)
allReshapes.append(assoMatMulScalarMul.reverse())

# For various multiplications, scalar division can be applied to any operand.
moveScalarDivInRight = Reshape(
    requires=SymbolTree("S/", # (X*Y) S/ a
                        left=SymbolTree("MUL", left=SymbolTree('X'), right=SymbolTree("Y")),
                        right=SymbolTree('a', attributes={'scalar'})
    ),
    target=SymbolTree("MUL", # X * (Y S/ a[s])
                      left=SymbolTree("X"),
                      right=SymbolTree("S/", left=SymbolTree("Y"), right=SymbolTree("a", attributes={'scalar'}))
    )
)
moveScalarDivInLeft = Reshape(
    requires=SymbolTree("S/", # (X*Y) S/ a
                        left=SymbolTree("MUL", left=SymbolTree('X'), right=SymbolTree("Y")),
                        right=SymbolTree('a', attributes={'scalar'})
    ),
    target=SymbolTree("MUL", # (X S/ a[s]) * Y
                      left=SymbolTree("S/", left=SymbolTree("X"), right=SymbolTree("a", attributes={'scalar'})),
                      right=SymbolTree("Y")
    )
)
for m in ["M*", "S*", ".*"]:
    temp = moveScalarDivInRight.substitute("MUL", m)
    allReshapes.append(temp)
    allReshapes.append(temp.reverse())
    temp = moveScalarDivInLeft.substitute("MUL", m)
    allReshapes.append(temp)
    allReshapes.append(temp.reverse())

# Specifically for M*, I also have to add the option to draw S/ to the left, because M* is not commutative.
moveScalarDivInLeft = Reshape(
    requires=SymbolTree("S/", # (X*Y) S/ a
                        left=SymbolTree("M*", left=SymbolTree('X'), right=SymbolTree("Y")),
                        right=SymbolTree('a', attributes={'scalar'})
    ),
    target=SymbolTree("M*", # (X S/ a[s]) *Y
                      left=SymbolTree("S/", left=SymbolTree('X'), right=SymbolTree("a", attributes={'scalar'})),
                      right=SymbolTree("Y")
    )
)
allReshapes.append(moveScalarDivInLeft)
allReshapes.append(moveScalarDivInLeft.reverse())

#Distributivity =======================================================================================================
## Templates ##
distributivityLeftOut = Reshape(
    requires=SymbolTree('MUL', SymbolTree('X'), SymbolTree('ADD', SymbolTree('Y'), SymbolTree('Z'))), # X(Y+Z)
    target=SymbolTree('ADD', SymbolTree('MUL', SymbolTree('X'), SymbolTree('Y')), SymbolTree('MUL', SymbolTree('X'), SymbolTree('Z'))), # XY+XZ
)

distributivityRightOut = Reshape(
    requires=SymbolTree('MUL', SymbolTree('ADD', SymbolTree('X'), SymbolTree('Y')), SymbolTree('Z')), # (X+Y)Z
    target=SymbolTree('ADD', SymbolTree('MUL', SymbolTree('X'), SymbolTree('Z')), SymbolTree('MUL', SymbolTree('Y'), SymbolTree('Z'))), # XZ+YZ
)

## There are a lot of combinations, and some operations only work distributively on the right ##
distributiveAdds = ['+', '-']
distributiveMulsLeft = ['M*', '.*', 'S*']
distributiveMulsRight = ['M*', 'S/', './'] # .* and S* are commutative, so there's no need to add the other direction.

for addOperation in distributiveAdds:
    # LEFT: A*X + A*Y = A*(X+Y)
    temp = distributivityLeftOut.substitute('ADD', addOperation)
    for mulOperation in distributiveMulsLeft:
        final = temp.substitute('MUL', mulOperation)
        allReshapes.append(final)
        allReshapes.append(final.reverse())
    # RIGHT: X*A + Y*A = (X+Y)*A
    temp = distributivityRightOut.substitute('ADD', addOperation)
    for mulOperation in distributiveMulsRight:
        final = temp.substitute('MUL', mulOperation)
        allReshapes.append(final)
        allReshapes.append(final.reverse())

#Neutral Elements =====================================================================================================
# Note: I do not add the reverse reshapes, because introducing neutral elements can be stacked and is rarely helpful.
## 0 is neutral in sum ##
removeNeutralZeroSum = Reshape(
    requires=SymbolTree('+', SymbolTree('X'), SymbolTree('Z', attributes={'zero'})), # X+0
    target=SymbolTree('X'), # X
)
simplifyReshapes.append(removeNeutralZeroSum)
allReshapes.append(removeNeutralZeroSum) # Because + is commutative, this also captures 0+X

## 0 is neutral in Diff (on the right) or is a simple negation (on the left) ##
removeNeutralZeroDiff = removeNeutralZeroSum.substitute('+', '-') # X-0 -> X
simplifyReshapes.append(removeNeutralZeroSum)
allReshapes.append(removeNeutralZeroDiff)

neutralZeroToUMinus = Reshape(
    requires=SymbolTree('-', SymbolTree('Z', attributes={'zero'}), SymbolTree('X')), # 0-X
    target=SymbolTree('u-', SymbolTree('X')), # -X
)
simplifyReshapes.append(neutralZeroToUMinus)
allReshapes.append(neutralZeroToUMinus)

## eye (diagaonl, one) is neutral for MatMul ##
eyeMatMulLeft =  Reshape(
    requires=SymbolTree('M*',
                        left=SymbolTree('eye', attributes={'diagonal', 'one'}),
                        right=SymbolTree('X')),
    target=SymbolTree('X')
)
eyeMatMulRight =  Reshape(
    requires=SymbolTree('M*',
                        left=SymbolTree('X'),
                        right=SymbolTree('eye', attributes={'diagonal', 'one'})),
    target=SymbolTree('X')
)
simplifyReshapes.append(eyeMatMulLeft)
allReshapes.append(eyeMatMulLeft)
simplifyReshapes.append(eyeMatMulRight)
allReshapes.append(eyeMatMulRight)

## Scalar one is the neutral element for S* and S/ (because S* is commutative, the flipped case is covered) ##
neutralScalarOne = Reshape(
    requires=SymbolTree("OP", left=SymbolTree('a', attributes={'scalar', 'one'}), right=SymbolTree('X')),
    target=SymbolTree('X')
)
neutralScalarMul = neutralScalarOne.substitute('OP', 'S*')
neutralScalarDiv = neutralScalarOne.substitute('OP', 'S/')
simplifyReshapes.append(neutralScalarMul)
allReshapes.append(neutralScalarMul)
simplifyReshapes.append(neutralScalarDiv)
allReshapes.append(neutralScalarDiv)

## Full matrix one is the neutral element for .* and .*
nonFullMatrix = {'diagonal', 'triangle_upper', 'triangle_lower'}

neutralScalarOne = Reshape(
    requires=SymbolTree("OP",
                        left=SymbolTree('X'),
                        right=SymbolTree('Y', attributes={'matrix', 'one'}, forbiddenAttr=nonFullMatrix)),
    target=SymbolTree('X')
)
neutralScalarMatMul = neutralScalarOne.substitute('OP', '.*')
neutralScalarMatDiv = neutralScalarOne.substitute('OP', './')
simplifyReshapes.append(neutralScalarMatMul)
allReshapes.append(neutralScalarMatMul)
simplifyReshapes.append(neutralScalarMatDiv)
allReshapes.append(neutralScalarMatDiv)

#Neutral Operations ===================================================================================================
## Summing a scalar ##
dontSumScalar = Reshape(
    requires=SymbolTree('sum', SymbolTree('X', attributes={'scalar'})), # sum(X[s])
    target=SymbolTree('X', attributes={'scalar'}), # X[s]
)
dontTraceScalar = dontSumScalar.substitute('sum', 'tr') # Same with trace

simplifyReshapes.append(dontSumScalar)
allReshapes.append(dontSumScalar)
simplifyReshapes.append(dontTraceScalar)
allReshapes.append(dontTraceScalar)

#Negation  ============================================================================================================
## u- with u- ##
removeDoubleNegation = Reshape(
    requires=SymbolTree('u-', SymbolTree('u-', SymbolTree('X'))), # --X
    target=SymbolTree('X'), # X
)
simplifyReshapes.append(removeDoubleNegation)
allReshapes.append(removeDoubleNegation)

## u- with + ##
pushMinusDown = Reshape(
    requires=SymbolTree('u-', SymbolTree('+', SymbolTree('X'), SymbolTree('Y'))), # - (X+Y)
    target=SymbolTree('+', SymbolTree('u-', SymbolTree('X')), SymbolTree('u-', SymbolTree('Y'))), #(-X) + (-Y)
)
allReshapes.append(pushMinusDown)
allReshapes.append(pushMinusDown.reverse())

## u- with - ##
flipDiff = Reshape(
    requires=SymbolTree('u-', SymbolTree('-', SymbolTree('X'), SymbolTree('Y'))), # - (X-Y)
    target=SymbolTree('-', SymbolTree('X'), SymbolTree('Y')), # X-Y
)
allReshapes.append(flipDiff)
# The reverse would be infinitely stackable (and is never easier to calculate).

wrappedDoubleNegation = Reshape(
    requires=SymbolTree('-', SymbolTree('X'), SymbolTree('u-', SymbolTree('Y'))), # X-(-Y)
    target=SymbolTree('+', SymbolTree('X'), SymbolTree('Y')), # X + Y
)
allReshapes.append(wrappedDoubleNegation)
# allReshapes.append(wrappedDoubleNegation.reverse()) # I don't think this will ever be useful, since it's twice the cost.

## u- is S* with -1 ##
# This enables pulling u- out of T and gives it something like associativity and commutativity.
uMinusMult = Reshape(
    requires=SymbolTree('u-', SymbolTree('X')), # -X
    target=SymbolTree('S*', SymbolTree('u-', SymbolTree('Scalar_1')), SymbolTree('X')), # -1 S* X
)
# Sadly, this causes infinite recursion. I really need to fix that bit.
# allReshapes.append(uMinusMult)
# allReshapes.append(uMinusMult.reverse())

#Transposing  =========================================================================================================
## double Transpose ##
# If a matrix is symmetric, I already have a rule that can eradicate transposition (see below).
removeDoubleTranspose = Reshape(
    requires=SymbolTree('T', SymbolTree('T', SymbolTree('X', forbiddenAttr={'symmetric'})), forbiddenAttr={'symmetric'}), # X''
    target=SymbolTree('X', forbiddenAttr={'symmetric'}), # X
)
simplifyReshapes.append(removeDoubleTranspose)
allReshapes.append(removeDoubleTranspose) # X'' -> X
# allReshapes.append(removeDoubleTranspose.reverse()) # X -> X''

## T and M* ##
transposeMatMultIn = Reshape(
    requires=SymbolTree('T', SymbolTree('M*', SymbolTree('Y'), SymbolTree('X'))), # (Y*X)'
    target=SymbolTree('M*', SymbolTree('T', SymbolTree('X')), SymbolTree('T', SymbolTree('Y'))), # X' * Y'
)

allReshapes.append(transposeMatMultIn)
allReshapes.append(transposeMatMultIn.reverse())

# Swap matrix mult. with inner transpose without introducing double transpose first:
#  No longer required now that I allow (individual) applicatiosn of double transposition. It snuck them in, anyway. TODO remove.
transposeMatMultSwap = Reshape(
    requires=SymbolTree('M*', SymbolTree('T', SymbolTree('X')), SymbolTree('Y')), # X' * Y
    target=SymbolTree('T', SymbolTree('M*', SymbolTree('T', SymbolTree('Y')), SymbolTree('X'))), # (Y'*X)'
)
# allReshapes.append(transposeMatMultSwap)
# The reverse is covered by 'transposeMatMultIn' + 'removeDoubleTranspose'
# I also don't add [Y*X' -> (X*Y')'], because together with the rest that allows X -> X''.
# To recognize the common subexpression in X'Y and Y'X I only really need one, anyway.

## T and almost every other binary operation ##
transposeInBin = Reshape(
    requires=SymbolTree('T', SymbolTree('OP', SymbolTree('X'), SymbolTree('Y'))), # (Y*X)'
    target=SymbolTree('OP', SymbolTree('T', SymbolTree('X')), SymbolTree('T', SymbolTree('Y'))), # X' * Y'
)
for op in ['+', '-', './', 'S*', 'S/']:
    # Note: In case of scalars (S* and S/) the unnecessary transpose can be removed by the corresponding rule.
    temp = transposeInBin.substitute('OP', op)
    allReshapes.append(temp)
    allReshapes.append(temp.reverse())

## T and certain unary operations ##
transposeInUn = Reshape(
    requires=SymbolTree('T', SymbolTree('OP', SymbolTree('X'))), # op(X)'
    target=SymbolTree('OP', SymbolTree('T', SymbolTree('X'))), # op(X')
)

for op in ['inv', 'u-', 'relu', 'exp', 'log', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'tanh', 'abs', 'sign']:
    temp = transposeInUn.substitute('OP', op)
    # allReshapes.append(temp) # That's not really useful. In particular, CSE will want to use the form where T is outside.
    allReshapes.append(temp.reverse())

## unnecessary transpose (operation doesn't care) ##
transposeIgnore = Reshape(
    requires=SymbolTree('OP', SymbolTree('T', SymbolTree('X'))), # op(X')
    target=SymbolTree('OP', SymbolTree('X')), # op(X)
)

for op in ['det', 'sum', 'tr', 'diag', 'diag2']:
    temp = transposeInBin.substitute('OP', op)
    simplifyReshapes.append(temp)
    allReshapes.append(temp)
    # The reverse would add unnecessary bloat ad infinitum.

## unnecessary transpose (variable wouldn't change) ##
symmetricTranspose = Reshape(
    requires=SymbolTree('T', left=SymbolTree('X', attributes={'symmetric'})), # This includes scalars, diagonal etc.
    target=SymbolTree('X', attributes={'symmetric'})
)
simplifyReshapes.append(symmetricTranspose)
allReshapes.append(symmetricTranspose)

#Inversion ============================================================================================================
detOfInverse = Reshape(
    requires=SymbolTree('det', SymbolTree('inv', SymbolTree('X')), attributes={'scalar'}),
    target=SymbolTree('S/', SymbolTree('Scalar_1'), SymbolTree('det', SymbolTree('X')), attributes={'scalar'}),
)
allReshapes.append(detOfInverse)
# The reverse could be infinitely stacked (and isn't useful).

inverseScalar = Reshape(
    requires=SymbolTree('inv', attributes={'scalar'}, left=SymbolTree('X', attributes={'scalar'})),
    target=SymbolTree('S/', left=SymbolTree('Scalar_1'), right=SymbolTree('X')),
)
allReshapes.append(inverseScalar)

#Determinant ==========================================================================================================
# TODO: det(A[tu]) and det(A[tl]).

#Other ================================================================================================================
logExp = Reshape(
    requires=SymbolTree('log', left=SymbolTree('exp', SymbolTree('X'))),
    target=SymbolTree('X'),
)
allReshapes.append(logExp)
simplifyReshapes.append(logExp)

# TODO I'm still missing lots and lots of the reshapes that LinA essentially used. I always meant to add them and somehow never did.