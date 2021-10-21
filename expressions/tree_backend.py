#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: paul rump
@email: paul.gerhardt.rump@uni-jena.de
"""
from expressions.expressiontree import ExpressionTree
import plantuml # This is non-standard, but then this file is for debug/display only

 # TODO: Replace this class with a function in the ExpressionTree class that uses the DotGraph class.

def set_label(s: set):
    return " | ".join(s)

def node_label(node: ExpressionTree, dimsAndProps = True):
    if dimsAndProps:
        return '"{<name>' + node.shortDesc(writeDims=True).replace('(', '|(').replace('[', '|').replace(']', '') + '}"'
    else:
        # Replace the internal names with user friendly ones (since this is the "simple" viewing mode).
        if node.name in ["M*", "S*"]:
            simpleName = "*"
        elif node.name.startswith("Var_"):
            simpleName = node.name[4:]
        else:
            simpleName = node.name
        
        return '"' + simpleName + '"'

class TreeGenerator():
    def __init__(self, exptree):
        self.exptree = exptree
        self.counter = 0

    def getImageURL(self, dimsAndProps = True):
        PlantUML = plantuml.PlantUML(url='http://www.plantuml.com/plantuml/svg/')
        return PlantUML.get_url(self.generate(dimsAndProps=dimsAndProps))

    def generate(self, dimsAndProps = True):
        _, string = self._generate(self.exptree, dimsAndProps)
        string = '''digraph g {
    graph [rankdir = "TD"];
    ''' + string + '}'

        return string

    def _generate(self, node: 'ExpressionTreee', dimsAndProps = True):
        uid = self.counter
        self.counter+=1
        # 'antiquewhite' and 'azure' are also interesting node colors, if required:
        s = f'\t"node{uid}"[label={node_label(node, dimsAndProps)} fillcolor="antiquewhite" style="filled"'
        if dimsAndProps:
            s += ' shape="record"'
        s += ']\n'
        for child in node.children():
            childDot = self._generate(child, dimsAndProps)
            s += childDot[1]
            s += f'\t\t"node{uid}"->"node{childDot[0]}"\n'

        if hasattr(node, 'cseOrigin'):
            pass # TODO: CSE, but it no longer works like that which makes things much more complicated.
                 #  Will need to pass the actual CSE dict.

        return uid, s
