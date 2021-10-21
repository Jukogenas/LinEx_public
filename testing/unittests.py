import unittest
from expressions.parser import Parser

# TODO: Stupidly, the framework seems to block debugger breakpoints. I may want to use something else.

class ExpressionTreeTests(unittest.TestCase):

    def test_mathEquals(self):
        etA = Parser("A").parse()
        # Leaves
        self.assertTrue(etA.mathEquals(etA)) # Exactly the same tree
        self.assertTrue(etA.mathEquals(Parser("A").parse())) # Exactly the same tree but represented by a different object
        self.assertFalse(etA.mathEquals(Parser("X").parse())) # Differently named leaf. Should NOT be equal.

        # Elementary operations, no reshape:
        aPlusB = etA + Parser("B").parse()
        self.assertTrue(aPlusB.mathEquals(aPlusB))
        del etA, aPlusB

        # The actually interesting bits, but still in small pieces
        aAB = Parser("a*(A+B)").parse()
        AaaB = Parser("A*a+a*B").parse()
        self.assertTrue(aAB.mathEquals(AaaB))

        del aAB, AaaB

        # TODO: A few big and complicated examples.

if __name__ == '__main__':
    unittest.main()