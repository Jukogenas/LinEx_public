#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:27:56 2018

@author: julien
@email: julien.klaus@uni-jena.de

This version has been slightly adjusted by Paul Gerhardt Rump (paul.gerhardt.rump@uni-jena.de) for the LinEx project.
"""

import string

from expressions.dimension import Dimension
from expressions.scanner import Scanner, KEYWORD
from expressions.expressiontree import ExpressionTree, Scalar, Matrix, SemanticException, allAttributes

ATTRIBUTE_SHORT = {
    's', 'v', 'cv', 'm', 'sq',
    'd', 'tu', 'tl', 'sym', 'ps', 'p', 'pd', 'psd', 'n', 'nd', 'nsd',
	#'c', 'z', 'o', # These should not be entered!
}

ATTRIBUTE = allAttributes.copy()

ATTRIBUTE_SHORT_MAP = {
	's': 'scalar',
	'v': 'vector',
	'cv': 'covector',
	'm': 'matrix',
	'sq': 'square_matrix',
	'd': 'diagonal',
	'tu': 'triangle_upper',
	'tl': 'triangle_lower',
	'sym': 'symmetric',
	'ps': 'pseudo_scalar',
	'c': 'constant', # Should not appear in the input string, but whatever.
	'p': 'pose_def',
	'pd': 'pos_def',
	'psd': 'pos_semi_def',
	'n': 'neg_def',
	'nd': 'neg_def',
	'nsd': 'neg_semi_def',
	# 'c': 'constant',
	# 'z': 'zero',
	# 'o': 'one',
}

class Parser(object):
	def __init__(self, input, geno=False):
		self.input = input
		self.geno = geno
		self.vars = dict()
		self.comments = [] # Unused?

	def tryMakeTree(self, position, *args, **kwargs):
		try:
			return ExpressionTree(*args, **kwargs)
		except IllegalOperandException as e:
			raise SemanticException(message=e.message, position=position)

	def parse(self):
		# start the parsing process
		self.scanner = Scanner(self.input, self.geno)
		self.getSym()
		if self.input.strip() == "":
			raise SyntaxException(f"Empty expression.", self.position)
		self.tensortree = self.expr()
		# is there something we do not know?
		if self.identifier is not None:
			raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)

		self.tensortree.inputString = self.input
		self.tensortree.replaceDimsWithReps() # Minor cleanup. Not strictly necessary.
		return self.tensortree.fullSimplify()

	# Expr = Term ([+-] Term)*
	def expr(self):
		tt = self.term()
		while self.description == "plus" or self.description == "minus":
			ident = self.identifier
			position = self.position
			self.getSym()
			tt = self.tryMakeTree(position, name=ident, left=tt, right=self.term(), highLevelProps=True)
		return tt

	# Term = -? Factor ((\*|/|.\*|./) -? Factor)*
	def term(self):
		# optional minus
		if self.description == "minus":
			pos = self.position
			self.getSym()
			tt = self.tryMakeTree(pos, "u-", left=self.factor(), highLevelProps=True)
		else:
			tt = self.factor()
		# look for multops
		while self.description == "times" or self.description == "div" or \
				self.description == "ptimes" or self.description == "pdiv":
			ident = self.identifier
			position = self.position
			self.getSym()
			if self.description == "minus":
				position2 = self.position
				self.getSym()
				tt = self.tryMakeTree(position, ident, left=tt,
				                      right=self.tryMakeTree(position2, "u-", left=self.factor(), highLevelProps=True)
				                      , highLevelProps=True)
			else:
				tt = self.tryMakeTree(position, ident, left=tt, right=self.factor(), highLevelProps=True)
		return tt

	# Factor = Atom '* ((^|.^) Factor)*
	def factor(self):
		tt = self.atom()
		while self.description == "apostrophe":
			position = self.position
			self.getSym()
			tt = self.tryMakeTree(position, "T", tt, highLevelProps=True)
		while self.description == "ppow" or self.description == "pow":
			ident = self.identifier
			position = self.position
			self.getSym()
			tt = self.tryMakeTree(position, ident, left=tt, right=self.factor(), highLevelProps=True)
		return tt

	# Atom = (FNumber|Ident|Functioncall(Expression)|Bracket Term)
	# Btw. "Atom" is not a fitting name because this is not a minimal building block (it contains calls back to Expression).
	# Better consider this a SubExpression or something like that.
	def atom(self):
		# fnumber
		if self.description == "plus" or self.description == "minus" or self.description == "number":
			tt = self.fnumber()
		# variable or function:
		elif self.description == "ident":
			if self.identifier in KEYWORD:
				tt = self.function()
			else:
				tt = self.ident()
		# bracket term
		elif self.description == "lrbracket":
			self.getSym()
			tt = self.expr()
			if self.description == "rrbracket":
				self.getSym()
			else:
				if self.identifier is None or self.identifier == '\n':
					raise SyntaxException("Unexpected ending - bracket remained unclosed.", self.position)
				raise SyntaxException(f"Unexpected symbol '{self.identifier}' - bracket remained unclosed.",
				                      self.position)
		else:
			if self.identifier is None or self.identifier == '\n':
				raise SyntaxException("Unexpected ending.", self.position)
			raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)
		return tt

	def function(self):
		# Check the function name for validity:
		if self.identifier in self.vars:  # Will only catch this if variable was PREVIOUSLY read tho.
			raise SyntaxException(f"Cannot use '{self.identifier}' as variable name and operator name.",
								  self.position)
		if self.identifier in KEYWORD:
			ident = self.identifier
		else:
			#              ident = "op_" + self.identifier
			raise SyntaxException(f"Function '{self.identifier}' not defined.", self.position)
		self.getSym()
		# Grab the arguments:
		if self.description == "lrbracket":
			position = self.position
			colStart = position['column']
			self.getSym()
			tt = self.expr()
			if self.description == "rrbracket":
				colEnd = self.position['column']
				position['length'] = colEnd - colStart + 1
				self.getSym()
			else:
				if self.identifier is None or self.identifier == '\n':
					raise SyntaxException("Unexpected ending - bracket remained unclosed.", self.position)
				raise SyntaxException(f"Unexpected symbol '{self.identifier}' - bracket remained unclosed.",
									  self.position)
		else:
			if self.identifier is None or self.identifier == '\n':
				raise SyntaxException("Unexpected ending - function ended without argument block.", self.position)
			raise SyntaxException(f"Unexpected symbol '{self.identifier}' - function ended without argument block.",
								  self.position)

		# In case of 'vector' and 'matrix', give the user the opportunity to specify a length:
		shape = () # Default; the ExpTree constructor can deal with it.
		if ident == 'vector':
			if self.description == 'lrbracket':
				shape, enteredCols = self._readShape('unnamed', (Dimension("unnamed_rows"), 1))
				if enteredCols:
					raise SemanticException(message="Can't specify the width of a vector!", position=self.position)
		elif ident == 'matrix':
			if self.description == 'lrbracket':
				shape, enteredCols = self._readShape('unnamed', (Dimension("unnamed_rows"), Dimension("unnamed_cols")))
				if not enteredCols:
					raise SemanticException(message="Tried to construct a matrix without columns!", position=self.position)
				# TODO: In case of 'matrix' I could even allow making it triagonal and stuff like that.
		return self.tryMakeTree(position, ident, left=tt, shape=shape, highLevelProps=True)

	# FNumber = [+-]? number (by itself always evaluates to a scalar!)
	def fnumber(self):
		number = ""
		if self.description == "plus" or self.description == "minus":
			number += self.identifier
			self.getSym()
		if self.description == "number":
			number += self.identifier
			self.getSym()
		tt = Scalar(number, attributes={'constant'})
		return tt

	def ident(self):
		# Save things and move on (some cases need the next symbol):
		identStr, pos = self.identifier, self.position
		self.getSym()

		# Eye need no further checks:
		if identStr.lower() == "eye":
			return Matrix("eye", attributes={'square_matrix', 'diagonal', 'constant', 'pseudo_scalar', 'pos_def', 'triangle_upper',
			                                 'triangle_lower', 'symmetric'})

		# What if I encounter the same name repeatedly?
		if identStr in self.vars:
			if self.description == 'lsbracket':
				raise SyntaxException(f"Encountered variable '{identStr}' a second time with shape or property information. \n"
									  f"Please make sure to only input this information when the variable first occurs.")
			return self.vars[identStr]

		# See whether the user noted additional properties:
		shape, props = self.propertiesAndShape(identStr)

		# And finally construct the thing:
		tt = ExpressionTree.quickExpLeaf(identStr, shape, props, highLevelProps=True)
		self.vars[identStr] = tt
		return tt

	# Properties = \[ property (, property)* \]
	# Note that one 'property' can be the shape, written as \( rows?, cols?\).
	# rows and cols can be a number or a name for the dimension. It can even be 'name = value' to indicate both.
	# If they are missing, the name of the dimension will be based on the name of the variable.
	# Btw. the outer commata are technically optional; spaces itself can adequately separate properties.
	def propertiesAndShape(self, varname):
		# Prepare defaults
		props = set()
		shape = Dimension(varname+'_rows'), Dimension(varname+'_cols')
		userEnteredShape = False
		userEnteredShape2 = False

		# Read info.
		while self.description in ['lrbracket', 'lsbracket']:
			# It's a shape block. There can only be one, though!
			if self.description == 'lrbracket':
				if userEnteredShape:
					raise SyntaxException(f"Encountered shape information for variable '{varname}' more than once!")
				else:
					userEnteredShape = True
					shape, userEnteredShape2 = self._readShape(varname, shape)
			# It's a property block. If multiple are encountered, just bundle them together.
			else:
				props.update(self._readProps())

		# Clean up shape (that is: Interpret what the user meant by combining dimensions and properties):
		# 1. Add whichever shape tag corresponds to the entered dimensions, assuming the user entered any:
		if userEnteredShape:
			# A: The user entered both (or at least indicated both exist with a comma), so everything's clearly defined:
			if userEnteredShape2:
				if shape[0] == 1 and shape[1] == 1:
					props.add('scalar')
				elif shape[0] == 1:
					props.add('covector')
				elif shape[1] == 1:
					props.add('vector')
				else:
					props.add('matrix')
			# B: The user only entered one shape, so the only sensible thing is to assume we have a column vector:
			else:
				if shape[0] == 1:
					props.add('scalar')
				else:
					props.add('vector')

		# 2. If we still don't have any shape info, add the default based on the name:
		baseProps = props.intersection({'scalar', 'vector', 'covector', 'matrix'})
		if len(baseProps) == 0:
			props.add(standardType(varname))

		# 3. But if we have multiple, tell the user something is wrong:
		elif len(baseProps) > 1:
			raise SyntaxException(f"Variable {varname} was given (explicitly or by its shape) contradictory shape properties: {str(props)}",
								  position=self.position)

		# 4. Now we know that there's precisely one. Make sure everything is set accordingly:
		if 'scalar' in props:
			shape = 1, 1
		elif 'vector' in props:
			shape = shape[0], 1
		elif 'covector' in props:
			shape = 1, shape[1]
		# nothing to change in matrix case.

		return shape, props

	def _readShape(self, varname, shape):
		encounteredWidthData = False

		self.getSym() # Skip '('
		# First dim:
		name, number = self.getShapeEntry()
		if '' != name:
			shape[0].setAlias(name)
		if 1 == number:
			shape = 1, shape[1]
		elif 0 != number:
			shape[0].setNumeric(number)
		# Intermediary comma (required if entering second dim):
		if self.description == 'comma':
			self.getSym() # Skip ','
			encounteredWidthData = True
			# Second dim:
			name, number = self.getShapeEntry()
			if '' != name:
				shape[1].setAlias(name)
			if 1 == number:
				shape = shape[0], 1
			elif 0 != number:
				shape[1].setNumeric(number)

		# Closing bracket:
		if self.description == 'rrbracket':
			self.getSym()  # Skip ')'
		else:
			raise SyntaxException(f'Expected ")", but encountered {self.identifier}', self.position)

		return shape, encounteredWidthData

	def _readProps(self):
		props = set()
		self.getSym()  # skip '['

		while self.description != 'rsbracket':
			# If it's a property, identify it and add it to the list.
			if self.description == 'ident':
				# Expand short attribute name, if that's what it is.
				identifier = ATTRIBUTE_SHORT_MAP.get(self.identifier, self.identifier)
				# If it's still not a valid attribute, it doesn't belong here!
				if not identifier in ATTRIBUTE:
					raise SyntaxException(f"Unexpected symbol in attribute brackets: {self.identifier}.", self.position)
				# If it cleared that, everything is fine!
				props.add(identifier)
				self.getSym()
			# If it's a comma, skip it.
			elif self.description == 'comma':
				self.getSym()
				continue
			else:
				raise SyntaxException(f"Unexpected symbol in attribute brackets: {self.identifier}.", self.position)

		self.getSym()  # skip ']'
		return props

	# Shapeentry = name | numeric | name = numeric
	def getShapeEntry(self):
		name, number = '', 0
		# Check for a name:
		if self.description == 'ident':
			name = self.identifier
			if name.startswith("unnamed"):
				name = "_"+name
			self.getSym()
			# Check whether the user also added a numeric value to the dimension:
			if self.description == 'colon':
				self.getSym()
				if self.description == 'number':
					number = int(self.identifier)
					self.getSym()
				else:
					raise SyntaxException(f"Expected a number after '{name}:' in shapestring.", self.position)
		# Maybe a numeric value is all we get?
		elif self.description == 'number':
			number = int(self.identifier)
			self.getSym()
		# Maybe the user didn't specify this shape at all?
		elif self.description in ['comma', 'rrbracket', 'rsbracket']: # This is the next symbol after the shapestring.
			return name, number
		else:
			raise SyntaxException(f"Unexpected symbol '{self.identifier}' in shapestring.", self.position)

		return name, number

	def getSym(self):
		(self.description, self.identifier, self.position) = self.scanner.getSym()
		if self.identifier:
			self.position['length'] = len(self.identifier)
		self.position['printLine'] = self.geno

	def expectedKeyword(self, keywordList):
		scanner = Scanner(self.input, geno=True)
		(desc, ident, _) = scanner.getSym()
		while ident is not None:
			if desc == "ident":
				if ident in keywordList:
					return False
			(desc, ident, _) = scanner.getSym()
		return True

def standardType(symbol):
	if symbol == 'eye':
		return 'eye'
	elif symbol[0] in string.ascii_lowercase[:7]:
		return "scalar"
	elif symbol[0] in string.ascii_lowercase[7:]:
		return "vector"
	elif symbol[0] in string.ascii_uppercase:
		return "matrix"
	else:
		return "scalar"

class IllegalOperandException(Exception):
	# so far only used in tensortreebuilder
	def __init__(self, message=""):
		super().__init__(message)
		self.message = message

class SyntaxException(Exception):
	def __init__(self, message, position=None):
		if position is None:
			position = {}
			position['column'] = 0
			position['line'] = 0
		self.message = message + (f" @pos:{position['column']}" if position else '')

	def __str__(self):
		return f"Syntax {str(self.message)}"
