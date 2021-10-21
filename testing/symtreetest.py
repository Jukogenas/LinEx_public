import unittest

from expressions import Parser
from LinEx.symboltree import SymbolTree


class MyTestCase(unittest.TestCase):

    def test_symtreeMatch(self):
        expTree = Parser("A*B+C").parse()
        # Check for t+ of any two things:
        symtree = SymbolTree("t+", left=SymbolTree('X'), right=SymbolTree('Y'))
        self.assertTrue(symtree.matches(expTree, verbosity=3))
        print()

    def test_symtreeNonMatchSame(self):
        expTree = Parser("A*B+C").parse()
        # Check for t+ of the same two things:
        symtree = SymbolTree("t+", left=SymbolTree('X'), right=SymbolTree('X'))
        self.assertFalse(symtree.matches(expTree, verbosity=3))
        print()

    def test_symtreeNonMatchOperation(self):
        expTree = Parser("A*B+C").parse()
        # Check for t+ of the same two things:
        symtree = SymbolTree("t*", left=SymbolTree('X'), right=SymbolTree('X'))
        self.assertFalse(symtree.matches(expTree, verbosity=3))
        print()

    def test_symtreeNonMatchProperty(self):
        expTree = Parser("A*B+C").parse()
        # Check for t+ of the same two things:
        symtree = SymbolTree("X", attributes={'symmetric'})
        self.assertFalse(symtree.matches(expTree, verbosity=3))
        print()

    def test_symmetryCreation(self):
        # A*A^T and A^T*A results in a symmetric matrix.
        symtree = SymbolTree("X", attributes={'symmetric'})
        self.assertTrue(symtree.matches(Parser("A*A'").parse(), verbosity=3))
        self.assertTrue(symtree.matches(Parser("A'*A").parse(), verbosity=3))
        # It doesn't work for two different matrices:
        self.assertFalse(symtree.matches(Parser("A'*B").parse(), verbosity=3))
        self.assertFalse(symtree.matches(Parser("A*B'").parse(), verbosity=3))
        print()

if __name__ == '__main__':
    unittest.main()
