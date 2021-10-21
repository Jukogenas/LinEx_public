#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 09:31:24 2020

@author: julien
@email: julien.klaus@uni-jena.de
"""

# TODO: I've integrated this functionality directly into the ExpTree, since it is used for hashing. This file is now obsolete.
class ExpressionGenerator():
    def __init__(self, exptree):
        self.exptree = exptree

    def generate(self, hashMode=False):
        return self._generate(self.exptree, hashMode=hashMode)

    def _generate(self, node, hashMode=False, knownVars=None):
        # Caching (separately for hashMode True and False) directly in the node
        if not hasattr(node, "expString"):
            node.expString = dict()
        if hashMode in node.expString:
            return node.expString[hashMode]

        # Calculation if not already cached
        s = ""
        if knownVars is None:
            knownVars = dict()
        #Binary
        if node.left and node.right:
            s += "(" + self._generate(node.left, hashMode=hashMode, knownVars=knownVars)
            s += str(node.name)
            s += self._generate(node.right, hashMode=hashMode, knownVars=knownVars) + ")"
        #Unary
        elif node.left:
            # special case T
            if node.name == "T":
                s += f"{self._generate(node.left, hashMode=hashMode, knownVars=knownVars)}'"
            else:
                s += f"{node.name}({self._generate(node.left, hashMode=hashMode)})"
        # Leaf (not using Varnames)
        elif hashMode:
            # In hashMode, I only care about the properties, not the name!
            # That said, if the same name Reappears, that is significant!
            if node.name not in knownVars:
                knownVars[node.name] = "V" + str(len(knownVars)) +":"
            s += knownVars[node.name] + str(node.attributes)
        # Leaf (using Varnames)
        else:
            s +=  str(node.name)

        node.expString[hashMode] = s
        return s
