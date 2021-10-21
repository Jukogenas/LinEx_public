#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 09:06:30 2018

@author: julien
@email: julien.klaus@uni-jena.de
"""

import string

# allowed character
ALPHA = list(string.ascii_letters)

# allowed functions
KEYWORD = ["sin", "cos", "tan", "exp", "log", "tanh",
           "sign", "abs", "det", "logdet",
           "matrix", "vector", "diag", "sum", "inv",
           "norm1", "norm2", "tr",
           "arcsin", "arccos", "arctan", "relu", "softmax"]

# allowed numbers
DIGIT = [str(i) for i in range(10)]

# allowed symbols
SYMBOL = {'(': 'lrbracket',
          ')': 'rrbracket',
          '[': 'lsbracket',
          ']': 'rsbracket',
          '*': 'times',
          '+': 'plus',
          ',': 'comma',
          '-': 'minus',
          '/': 'div',
          '^': 'pow',
          '.': 'dot',
          '\'': 'apostrophe',
          '.^': 'ppow',
          '.*': 'ptimes',
          './': 'pdiv',
          ':': 'colon',
          }

COMPARE = {'<': 'less',
           '>': 'greater',
           '=': 'equal',
           '<=': 'lesseq',
           '>=': 'greatereq',
           '==': 'equiv'
           }


class Input(object):
    def __init__(self, input):
        self.index = 0
        self.length = len(input)
        self.input = input
        # current position
        self.column = 0

    def next(self):
        if self.index < self.length:
            literal = self.input[self.index]
            self.index += 1
            self.column += 1
            return literal
        else:
            return None


class Scanner(object):
    def __init__(self, input, geno=False):
        # create new input object and remove trailing whitespaces
        # and change lineend to \n (Linux and Windows compatible)
#        self.input = Input(input.replace("\t", " "*4).replace("\r", ""))
        self.input = Input(input.replace("\r", ""))
        self.current = self.input.next()
        self.line = 1
        self.geno = geno # TODO: This is part of something that was removed. May get rid of the remains later.
        self.comments = []

    def getSym(self):
        identifier = ""
        description = ""

        # blank
        while self.current in [" ", "\t"]:
            description = "blank"
            identifier = self.current
            self.current = self.input.next()

        position = {}
        position['column'] = self.getColumn()
        position['line'] = self.getLine()
        position['printLine'] = self.geno

        if self.geno:
            # skip comment
            if self.current == "#":
                if self.getColumn() == 1 or description == "blank":
                    self.current = self.input.next()
                    comment = ""
                    while self.current != "\n":
                        comment += self.current
                        self.current = self.input.next()
                    self.comments.append((self.getLine(), comment))
                else:
                    raise ScannerException(f"Leave a space before '#' or put it on a new line to start a comment.", position)

        if self.current == "\n":
            description = "newline"
            self.lastLine = self.line
            self.lastColumn = self.getColumn()
            self.line += 1
            identifier = self.current
            self.resetColumn()
            self.current = self.input.next()
            # cannot mark newline char, mark the one before
            position['column'] -= 1

        elif self.current in DIGIT:
            description = "number"
            identifier = ""
            while self.current in DIGIT:
                identifier += self.current
                self.current = self.input.next()
            if self.current == ".":
                self.current = self.input.next()
                if self.current in ["*", "/", "^"]:
                    self.current = "." + self.current
                else:
                    identifier += '.'

                if self.current in DIGIT:
                    while self.current in DIGIT:
                        identifier += self.current
                        self.current = self.input.next()

            if self.current == "e" or self.current == "E":
                identifier += self.current
                self.current = self.input.next()
                if self.current == "+" or self.current == "-":
                    identifier += self.current
                    self.current = self.input.next()
                if self.current in DIGIT:
                    while self.current in DIGIT:
                        identifier += self.current
                        self.current = self.input.next()
                else:
                    position['length'] = len(identifier)
                    raise ScannerException(f"This is not a valid number.", position)

        # detect variables (can also be function names; the parser decides that)
        elif self.current in ALPHA:
            description = "ident"
            identifier = self.current
            self.current = self.input.next()
            while self.current in ALPHA + DIGIT:
                identifier += self.current
                self.current = self.input.next()

        # symbols
        elif self.current in SYMBOL.keys():
            description = SYMBOL.get(self.current)
            identifier = self.current
            # pointwise operations
            if identifier == '.':
                self.current = self.input.next()
                if self.current in ['^', '*', '/']:
                    identifier += self.current
                    description = SYMBOL.get(self.current)
                    self.current = self.input.next()
            else:
                self.current = self.input.next()

        elif self.geno and self.current in COMPARE:
            description = "compare"
            identifier = self.current
            if identifier in ["<", ">", "="]:
                self.current = self.input.next()
                if self.current == "=":
                    identifier += self.current
                    description = "compare"
                    self.current = self.input.next()
            else:
                self.current = self.input.next()

        # at the end it should be None
        elif self.current is None:
            description = "None"
            identifier = self.current

        # unrecognized symbol
        else:
            description = "error"
            identifier = self.current
            position['length'] = len(identifier)
            raise ScannerException(f"Symbol '{identifier}' is not allowed.", position)

        # return the (key, value, position) triple
        return (description, identifier, position)

    def getColumn(self):
        return self.input.column

    def resetColumn(self):
        self.input.column = 0

    def getLine(self):
        return self.line


# Class for the exceptions
class ScannerException(Exception):
    def __init__(self, message, column):
        self.message = message
        self.column = column

    def __str__(self):
        return str("Scanner exception at column %i: %s" % (self.column, self.message))