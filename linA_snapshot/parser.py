#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:27:56 2018

@author: julien
@email: julien.klaus@uni-jena.de
"""

import string

from linA_snapshot.scanner import Scanner, KEYWORD
from linA_snapshot.tensortree import TensorTree, Scalar, Vector, Matrix, SemanticException


class Parser(object):
   def __init__(self, input, geno=False):
      self.input = input
      self.geno = geno
      # get the right variable types
      self.variables = self.getVariables(self.input)
      self.parameter = dict()
      self.attributes = dict()
      self.vars = dict()
      # we keep the order of declaration
      self.paramList = []
      self.varList = []
      self.st = list()
      self.constraints = []
      self.min = None
      self.comments = []

   def call_TT(self, position, *args, **kwargs):
      try:
         return TensorTree(*args, **kwargs)
      except IllegalOperandException as e:
         raise SemanticException(e.message, position)

   def parse(self):
      # start the parsing process
      self.scanner = Scanner(self.input, self.geno)
      self.getSym()
      if self.input.strip() == "":
            raise SyntaxException(f"Empty expression.", self.position)
            # TODO: decide me!
         #                return None
      self.tensortree = self.expr()
      # if there is something we do not know
      if self.identifier is not None:
         raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)

      self.tensortree.inputString = self.input
      return self.tensortree

   def checkAttribute(self, name, param):
      if not param:
         raise SemanticException(f"A variable cannot be '{self.identifier}'.", self.position)
      else:
         if not self.parameter[name] == "matrix":
            raise SemanticException(f"A {self.parameter[name]} cannot be '{self.identifier}'.", self.position)

   def variable(self):
      if self.descriptionEqual("ident") and self.identifier in ["variables", "variable"]:
         param = False
      else:
         param = True
      self.getSym()
      while self.descriptionEqual("newline"):
         self.getSym()
         while self.descriptionEqual("newline"):
            self.getSym()
         # optional newline
         if self.descriptionEqual("type"):
            while self.descriptionEqual("type"):
               varType = self.identifier
               self.getSym()
               if self.descriptionEqual("ident"):
                  if self.identifier in KEYWORD:
                     raise SyntaxException(f"Cannot use '{self.identifier}' as variable name.", self.position)
                  name = self.identifier

                  if name in self.parameter or name in self.vars:
                     errorMsg = 'Variable'
                     if param:
                        errorMsg = 'Parameter'
                     raise SemanticException(f"{errorMsg} '{name}' has already been declared.", self.position)
                  if param:
                     self.parameter[name] = varType
                     self.paramList.append(name)
                  else:
                     self.vars[name] = varType
                     self.varList.append(name)
                  self.getSym()
                  while self.descriptionEqual("attribute"):
                     self.checkAttribute(name, param)
                     if name in self.attributes:
                        self.attributes[name].append(self.identifier)
                     else:
                        self.attributes[name] = [self.identifier]
                     self.getSym()
                  while self.descriptionEqual("comma"):
                     self.getSym()
                     if self.descriptionEqual("ident"):
                        if self.identifier in KEYWORD:
                           raise SyntaxException(f"Cannot use '{self.identifier}' as variable name.", self.position)
                        name = self.identifier

                        if name in self.parameter or name in self.vars:
                           errorMsg = 'Variable'
                           if param:
                              errorMsg = 'Parameter'
                           raise SemanticException(f"{errorMsg} '{name}' has already been declared.", self.position)
                        if param:
                           self.paramList.append(name)
                           self.parameter[name] = varType
                        else:
                           self.varList.append(name)
                           self.vars[name] = varType
                        self.getSym()
                        while self.descriptionEqual("attribute"):
                           self.checkAttribute(name, param)
                           if name in self.attributes:
                              self.attributes[name].append(self.identifier)
                           else:
                              self.attributes[name] = [self.identifier]
                           self.getSym()
                     else:
                        raise SyntaxException("Expected a variable after ','.", self.position)
               else:
                  raise SyntaxException("Expected at least one variable.", self.position)
               while self.descriptionEqual("newline"):
                  self.getSym()
         else:
            raise SyntaxException("Expected variable type ('matrix', 'vector', or 'scalar') and at least one variable.",
                                  self.position)


   def expr(self):
      tt = self.term()
      while self.descriptionEqual("plus") or self.descriptionEqual("minus"):
         ident = self.identifier
         position = self.position
         self.getSym()
         tt = self.call_TT(position, name=ident, left=tt, right=self.term())
      return tt

   def term(self):
      desc = self.description
      ident = self.identifier
      # optional minus
      if self.descriptionEqual("minus"):
         position = self.position
         self.getSym()
      if desc == "minus":
         tt = self.call_TT(position, "-", left=self.factor())
      else:
         tt = self.factor()
      # look for multops
      while self.descriptionEqual("times") or self.descriptionEqual("div") or \
              self.descriptionEqual("ptimes") or self.descriptionEqual("pdiv"):
         desc = self.description
         ident = self.identifier
         position = self.position
         self.getSym()
         if self.descriptionEqual("minus"):
            position2 = self.position
            self.getSym()
            tt = self.call_TT(position, ident, left=tt,
                              right=self.call_TT(position2, "-", left=self.factor()))
         else:
            tt = self.call_TT(position, ident, left=tt, right=self.factor())
      return tt

   def factor(self):
      tt = self.atom()
      while self.descriptionEqual("apostrophe"):
         position = self.position
         self.getSym()
         tt = self.call_TT(position, "T", tt)
      while self.descriptionEqual("ppow") or self.descriptionEqual("pow"):
         ident = self.identifier
         position = self.position
         self.getSym()
         tt = self.call_TT(position, ident, left=tt, right=self.factor())
      return tt

   def atom(self):
      tt = None
      # fnumber
      if self.descriptionEqual("plus") or self.descriptionEqual("minus") or self.descriptionEqual("number"):
         tt = self.fnumber()
      # variable
      elif self.descriptionEqual("ident"):
         if self.identifier in KEYWORD:
            raise SyntaxException(f"Function '{self.identifier}' is missing '('.", self.position)
         tt = self.ident()
      # function
      elif self.descriptionEqual("function"):
         if self.identifier in self.variables.keys():
            raise SyntaxException(f"Cannot use '{self.identifier}' as variable name and operator name.",
                                  self.position)
         if self.identifier in KEYWORD:
            ident = self.identifier
         else:
            #              ident = "op_" + self.identifier
            raise SyntaxException(f"Function '{self.identifier}' not defined.", self.position)
         self.getSym()
         if self.descriptionEqual("lrbracket"):
            position = self.position
            colStart = position['column']
            self.getSym()
            tt = self.expr()
            if self.descriptionEqual("rrbracket"):
               colEnd = self.position['column']
               position['length'] = colEnd - colStart + 1
               self.getSym()
               tt = self.call_TT(position, ident, left=tt)
            else:
               if self.identifier is None or self.identifier == '\n':
                  raise SyntaxException("Unexpected ending.", self.position)
               raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)
         else:
            if self.identifier is None or self.identifier == '\n':
               raise SyntaxException("Unexpected ending.", self.position)
            raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)
      # bracket term
      elif self.descriptionEqual("lrbracket"):
         self.getSym()
         tt = self.expr()
         if self.descriptionEqual("rrbracket"):
            self.getSym()
         else:
            if self.identifier is None or self.identifier == '\n':
               raise SyntaxException("Unexpected ending.", self.position)
            raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)
      else:
         if self.identifier is None or self.identifier == '\n':
            raise SyntaxException("Unexpected ending.", self.position)
         raise SyntaxException(f"Unexpected symbol '{self.identifier}'.", self.position)
      return tt

   def fnumber(self):
      number = ""
      if self.descriptionEqual("plus") or self.descriptionEqual("minus"):
         number += self.identifier
         self.getSym()
      if self.descriptionEqual("number"):
         number += self.identifier
         self.getSym()
      tt = Scalar(number)
      return tt

   def ident(self):
      variable_type = self.vars.get(self.identifier)
      if variable_type is None:
         variable_type = self.parameter.get(self.identifier)
      if variable_type is None:
         variable_type = self.variables.get(self.identifier)
      if self.identifier.lower() == "eye":
         tt = Matrix("eye")
      elif variable_type.lower() == "scalar":
         tt = Scalar(self.identifier)
      elif variable_type.lower() == "vector":
         tt = Vector(self.identifier)
      elif variable_type.lower() == "matrix":
         tt = Matrix(self.identifier)
      elif variable_type.lower() == "symmetric matrix":
         tt = Matrix(self.identifier)
         tt.setSymmetric()
      else:
         raise SyntaxException(f"Variable '{self.identifier}' has not been declared.", self.position)
      self.getSym()
      return tt

   def getVariableTypes(self):
      return self.variables

   def setVariableTypes(self, variables):
      self.variables = variables

   def getTensorTree(self):
      return self.tensortree

   def getSym(self):
      (self.description, self.identifier, self.position) = self.scanner.getSym()
      if self.identifier:
         self.position['length'] = len(self.identifier)
      self.position['printLine'] = self.geno

   def descriptionEqual(self, symbol):
      return self.description == symbol

   def getVariables(self, input):
      variables = []
      scanner = Scanner(input, self.geno)
      (desc, ident, _) = scanner.getSym()
      while ident is not None:
         if desc == "ident":
            if not ident == "eye":
               variables.append((ident, standardType(ident)))
         (desc, ident, _) = scanner.getSym()
      return dict(variables)

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

def parseExpression(txt, symblTbl={}):
   p = Parser(txt)
   var = p.getVariableTypes()
   for v in var.keys():
      if v in symblTbl:
         var[v] = symblTbl[v]
      else:
         var[v] = standardType(v)
   p.setVariableTypes(var)
   t = p.parse()
   return t, var


def getVars(txt, symblTbl={}, geno=False):
   p = Parser(txt, geno)
   var = p.getVariableTypes()
   for v in var.keys():
      if v in symblTbl:
         var[v] = symblTbl[v]
      else:
         var[v] = standardType(v)
   p.setVariableTypes(var)
   return var


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