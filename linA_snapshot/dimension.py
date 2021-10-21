#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def merge0(lists):
    newsets, sets = [set(lst) for lst in lists if lst], []
    while len(sets) != len(newsets):
        sets, newsets = newsets, []
        for aset in sets:
            for eachset in newsets:
                if not aset.isdisjoint(eachset):
                    eachset.update(aset)
                    break
            else:
                newsets.append(aset)
    return newsets


def makedata(var, rng, dimsizes,min=-1,max=1):
    if 'a' <= var[0] <= 'g': return rng.uniform(min, max)
    elif 'h' <= var[0] <= 'z': return rng.uniform(min, max, size=(dimsizes.get(var + '_rows', 1),1))
    elif 'A' <= var[0] <= 'Z': return rng.uniform(min, max, size=(dimsizes.get(var + '_rows', 1), dimsizes.get(var + '_cols', 1)))


def generate_test_data(extTree, seed, minDims=2, maxDims=10, verbose=False):
    from numpy.random import default_rng
    dims = merge0(extTree.inferDimension().values())
    if verbose: print(dims)
    varnames = sorted(extTree.allVars())
    if verbose: print(varnames)
    rng = default_rng(seed)
    dimsizes = {}
    for group in dims:
        size = rng.integers(minDims, maxDims, endpoint=True)
        dimsizes.update({dim: size for dim in group})
    data = {var: makedata(var, rng, dimsizes) for var in varnames}
    #data.update((dimsizes))
    return data
