from LinEx.calculation import Calculation, costFactors
from LinEx.symboltree import SymbolTree

#=== Notes ===#
# Don't forget to add new kernels to the list at the bottom, otherwise it won't be used!
# binStrength: 3 for multiplication, 1 for additon. Leave default for functions, float('inf') for atoms.
# 5 for power (**)
# 1024 for transpose (as a means to say: Not infinite but pretty high.
# Note that testing whether shapes match (such as in a*b) is NOT the kernel's job. The EspTree does that.

# Remember that adding top-level attribute requirements where applicable helps performance by ruling out
# impossibly kernels early on. Even if 'the property is there anyway', adding the requirement helps!

#=== Cost simplifiers ===#

#=== Unary Operations ===#
uminus = Calculation(
    SymbolTree('u-', SymbolTree('X')),
    lambda sizes: sizes['X'][0]*sizes['X'][1],
    lambda codes, dims: f"-{codes['X']}", # Need info about X to check whether brackets are required.
    bindStrengths=1
)

sum = Calculation(
    SymbolTree('sum', SymbolTree('X')),
    lambda sizes: sizes['X'][0]*sizes['X'][1] - 1,
    lambda codes, dims: f"np.sum({codes['X']})",
    preDefs={'import numpy as np'}
)

trace = Calculation(
    SymbolTree('tr', SymbolTree('X', attributes={'square_matrix'})),
    lambda sizes: sizes['X'][0] - 1,
    lambda codes, dims: f"np.trace({codes['X']})",
    preDefs={'import numpy as np'}
)

# - Other Matrix fun - #
transp = Calculation(
    SymbolTree('T', SymbolTree('X')),
    lambda sizes: 1, # The cost can be effectively 0 or higher. Numpy only marks unless it really needs to save the result.
    lambda codes, dims: f"{codes['X']}.T",
    bindStrengths=1024 # Not inf, because it's weaker than variables and brackets. But stronger than anything else.
)

diagV = Calculation(
    SymbolTree('diag', attributes={'square_matrix'}, left=SymbolTree('X', attributes={'vector'})),
    lambda sizes: sizes['X'][0],
    lambda codes, dims: f"np.diagflat({codes['X']})", # TODO: Without oriented vectors, np.diag is sufficient.
    preDefs={'import numpy as np'}
)

diagCV = Calculation(
    SymbolTree('diag', attributes={'square_matrix'}, left=SymbolTree('X', attributes={'covector'})),
    lambda sizes: sizes['X'][0],
    lambda codes, dims: f"np.diagflat({codes['X']})", # Without oriented vectors, np.diag is sufficient, but this still works.
    preDefs={'import numpy as np'}
)

diagM = Calculation(
    SymbolTree('diag2', attributes={'vector'}, left=SymbolTree('X', attributes={'square_matrix'})),
    lambda sizes: sizes['X'][0],
    lambda codes, dims: f"np.diagonal({codes['X']}).reshape((-1,1))", # Without oriented vectors, np.diag is sufficient, but this should still work.
    preDefs={'import numpy as np'}
)

det = Calculation(
    SymbolTree('det', SymbolTree('X')),
    lambda sizes: sizes['X'][0]**3, # n^3 (matrix is square). TODO: Assumes underlying Gauss alg. Is that correct?
    # Read here: https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html
    lambda codes, dims: f"np.linalg.det({codes['X']})",
    preDefs={'import numpy as np'}
)

invMatrix = Calculation(
    SymbolTree('inv', SymbolTree('X', attributes={'square_matrix'})),
    lambda sizes: 2*sizes['X'][0]**3, # n^3 (matrix is square). TODO: Assumes underlying Gauss alg. Is that correct?
    lambda codes, dims: f"np.linalg.inv({codes['X']})",
    preDefs={'import numpy as np'}
)

invSolveVector = Calculation( # X^-1 M* v as solve(X, v)
    SymbolTree("M*", attributes={'vector'},
            left=SymbolTree('inv', attributes={'square_matrix'}, left=SymbolTree('X', attributes={'square_matrix'})),
            right=SymbolTree('v', attributes={'vector'})
    ),
    lambda sizes: (sizes['X'][0])**3+(sizes['X'][0])**2, # TODO: Uses lapack _gesv; read up on that for a better estimate.
    lambda codes, dims: f"np.linalg.solve({codes['X']}, {codes['v']})", # TODO: Have solve as a thing in the tree, use reshapes to put the right stuff in?
    preDefs={'import numpy as np'}
)

invSolveMatrix = Calculation( # X^-1 M* Y as solve(X, Y)
    SymbolTree("M*", attributes={'matrix'},
            left=SymbolTree('inv', attributes={'square_matrix'}, left=SymbolTree('X', attributes={'square_matrix'})),
            right=SymbolTree('Y', attributes={'matrix'})
    ),
    lambda sizes: (sizes['X'][0])**2 * (sizes['X'][0]+sizes['Y'][0]), # TODO: Uses lapack _gesv; read up on that for a better estimate.
    lambda codes, dims: f"np.linalg.solve({codes['X']}, {codes['Y']})", # TODO: Have solve as a thing in the tree, use reshapes to put the right stuff in?
    preDefs={'import numpy as np'}
)

norm1Matrix = Calculation( # Abs sum
    SymbolTree('norm1', SymbolTree('X', attributes={'matrix'})),
    lambda sizes: 2*sizes['X'][0]*sizes['X'][1] -1, # abs + sum for all elements.
    lambda codes, dims: f"np.sum(np.abs({codes['X']}))",
    preDefs={'import numpy as np'}
)

norm1vector = Calculation( # Abs sum
    SymbolTree('norm1', SymbolTree('X', forbiddenAttr={'matrix'})),
    lambda sizes: 2*sizes['X'][0]*sizes['X'][1] -1, # abs + sum for all elements.
    lambda codes, dims: f"np.linalg.norm({codes['X']}, 1)",
    preDefs={'import numpy as np'}
)

norm2Matrix = Calculation( # Frobenius Norm
    SymbolTree('norm2', SymbolTree('X', attributes={'matrix'})),
    lambda sizes: 2*sizes['X'][0]*sizes['X'][1] -1 + costFactors['harsh'], # Multiply with self + sum for all elements. Sqrt.
    lambda codes, dims: f"np.linalg.norm({codes['X']}, 'fro')",
    preDefs={'import numpy as np'}
)

norm2vector = Calculation( # Euclidean Norm
    SymbolTree('norm2', SymbolTree('X', forbiddenAttr={'matrix'})),
    lambda sizes: 2*sizes['X'][0]*sizes['X'][1] -1 + costFactors['harsh'], # Multiply with self + sum for all elements. Sqrt.
    lambda codes, dims: f"np.linalg.norm({codes['X']})", # 2-norm is default.
    preDefs={'import numpy as np'}
)

hardRelu = Calculation(
    SymbolTree('relu', SymbolTree('X')),
    lambda sizes: sizes['X'][0]*sizes['X'][1], # Maximum is an if, though, right? That can be worse than add or mult.
    lambda codes, dims: f"np.maximum({codes['X']}, 0)",
    preDefs={'import numpy as np'}
)

# Elementwise Functions
exp = Calculation(
    SymbolTree('exp', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['harsh'],
    lambda codes, dims: f"np.exp({codes['X']})",
    preDefs={'import numpy as np'}
)

log = Calculation(
    SymbolTree('log', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['harsh'],
    lambda codes, dims: f"np.log({codes['X']})",
    preDefs={'import numpy as np'}
)

sin = Calculation(
    SymbolTree('sin', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.sin({codes['X']})",
    preDefs={'import numpy as np'}
)

cos = Calculation(
    SymbolTree('cos', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.cos({codes['X']})",
    preDefs={'import numpy as np'}
)

tan = Calculation(
    SymbolTree('tan', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.tan({codes['X']})",
    preDefs={'import numpy as np'}
)

arcsin = Calculation(
    SymbolTree('arcsin', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.arcsin({codes['X']})",
    preDefs={'import numpy as np'}
)

arccos = Calculation(
    SymbolTree('arccos', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.arccos({codes['X']})",
    preDefs={'import numpy as np'}
)

arctan = Calculation(
    SymbolTree('arctan', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.arctan({codes['X']})",
    preDefs={'import numpy as np'}
)

tanh = Calculation(
    SymbolTree('tanh', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1] * costFactors['medium'],
    lambda codes, dims: f"np.tanh({codes['X']})",
    preDefs={'import numpy as np'}
)

abs = Calculation(
    SymbolTree('abs', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1],
    lambda codes, dims: f"np.abs({codes['X']})",
    preDefs={'import numpy as np'}
)

sign = Calculation(
    SymbolTree('sign', SymbolTree('X')),
    lambda sizes: sizes['X'][0] * sizes['X'][1],
    lambda codes, dims: f"np.sign({codes['X']})",
    preDefs={'import numpy as np'}
)


#=== Binary Operations ===#

# - Addition - #
add = Calculation(
    SymbolTree('+', SymbolTree('X'), SymbolTree('Y')),
    lambda sizes: sizes['X'][0]*sizes['X'][1],
    lambda codes, dims: f"{codes['X']} + {codes['Y']}",
    bindStrengths=1
)

# - Subtraction - #
sub = Calculation(
    SymbolTree('-', SymbolTree('X'), SymbolTree('Y')),
    lambda sizes: sizes['X'][0]*sizes['X'][1],
    lambda codes, dims: f"{codes['X']} - {codes['Y']}",
    bindStrengths=1
)

# - Multiplication - #
dotMult = Calculation(
    SymbolTree('.*', SymbolTree('X'), SymbolTree('Y')),
    lambda sizes: sizes['X'][0]*sizes['X'][1],
    lambda codes, dims: f"{codes['X']} * {codes['Y']}",
    bindStrengths=3
)

scalarMult = Calculation(
    SymbolTree('S*', SymbolTree('a', attributes={'scalar'}), SymbolTree('Y')),
    lambda sizes: sizes['Y'][0]*sizes['Y'][1],
    lambda codes, dims: f"{codes['a']} * {codes['Y']}",
    bindStrengths=3
)

# Matrix multiplication. M*v and c*M is separated so it can be preferred in the cost heuristic (judging kernels by probable cost).
matMult = Calculation(
    SymbolTree('M*', SymbolTree('X', attributes={'matrix'}), SymbolTree('Y', attributes={'matrix'})),
    lambda sizes: sizes['X'][0]* (2*sizes['X'][1]-1) *sizes['Y'][1],
    lambda codes, dims: f"{codes['X']} @ {codes['Y']}",
    bindStrengths=3
)

matVecMult = Calculation(
    SymbolTree('M*', SymbolTree('X', attributes={'matrix'}), SymbolTree('Y', attributes={'vector'})),
    lambda sizes: sizes['X'][0]* (2*sizes['X'][1]-1),
    lambda codes, dims: f"{codes['X']} @ {codes['Y']}",
    bindStrengths=3
)

covevMatMult = Calculation(
    SymbolTree('M*', SymbolTree('X', attributes={'covector'}), SymbolTree('Y', attributes={'matrix'})),
    lambda sizes: (2*sizes['X'][1]-1) *sizes['Y'][1],
    lambda codes, dims: f"{codes['X']} @ {codes['Y']}",
    bindStrengths=3
)

innerMatMult = Calculation(
    SymbolTree('M*', SymbolTree('X', attributes={'covector'}), SymbolTree('Y', attributes={'vector'})),
    lambda sizes: 2*sizes['X'][1]-1,
    lambda codes, dims: f"{codes['X']} @ {codes['Y']}",
    bindStrengths=3
)

# - Division - #
dotDiv = Calculation(
    SymbolTree('./', SymbolTree('X'), SymbolTree('Y')),
    lambda sizes: sizes['X'][0]*sizes['X'][1],
    lambda codes, dims: f"{codes['X']} / {codes['Y']}",
    bindStrengths=3
)

scalarDiv = Calculation(
    SymbolTree('S/', SymbolTree('Y'), SymbolTree('a', attributes={'scalar'})),
    lambda sizes: sizes['Y'][0]*sizes['Y'][1],
    lambda codes, dims: f"{codes['Y']} / {codes['a']}",
    bindStrengths=3
)

# - Power - #
scalarPower = Calculation(
    SymbolTree('^', SymbolTree('a', attributes={'scalar'}), SymbolTree('b', attributes={'scalar'})),
    lambda sizes: costFactors['harsh'],
    lambda codes, dims: f"{codes['a']} ** {codes['b']}",
    bindStrengths=5
)

#=== Complete List ===#
AllKernels = [
                uminus, sum, trace,
                transp, diagV, diagCV, diagM, det, invMatrix, invSolveVector, invSolveMatrix,
                norm1vector, norm1Matrix, norm2vector, norm2Matrix, hardRelu,
                exp, log, sin, cos, tan, arcsin, arccos, arctan, tanh, abs, sign,
                add, sub,
                dotMult, scalarMult, matMult, matVecMult, covevMatMult, innerMatMult,
                dotDiv, scalarDiv,
                scalarPower,
              ]