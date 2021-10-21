from LinEx.orderedSet import OrderedSet


class EquivalencyCache(dict):
    """
    TODO: This class is now more general than it used to be. It doesn't only apply to ExpressionTrees.
    This cache takes trees and uses the (named) hash to map them to  a representative tree (a mathematically equivalent
    tree that has been chosen as the 'default' version of that expression). The representative itself is mapped to a set
    of all trees that map to it (plus itself), which at once identifies it and serves as a cache of all relevant* equivalent
    forms of a tree.
    It is possible that a tree doesn't map directly to it's representative. In that case, it maps to the next tree in a
    chain that will eventually lead to the representative. When the cache comes upon such a situation, it will automatically
    update the link to be more direct.
    
    *relevant: See tryModifyTree(...).
    TODO Potential issue: The depth of a SymbolTree defines how far down the tree can be
     modified for the modification to still be 'relevant'. For kernels of varying depth, there might be difficulties when
     the initial kernel or reshape (that leads to this cache being populated) isn't of the maximum depth. I may then have to also
     save the depth and repopulate the cache when a deeper kernel comes along.
    """
    def __init__(self):
        super().__init__()
    
    def register(self, obj):
        if obj in self:
            raise KeyError(f"[Eq.cache] The supposedly new object {str(obj)} is already known!")
        self[obj] = OrderedSet([obj]) # The order is not required, but makes debugging much less tedious.
            
    def getRepresentative(self, obj, registerIfUnknown=False):
        original = obj
        # For new trees, add if desired and just return the tree itself:
        #  Note: Once a tree is registered, the reshaping functions assume that they don't have to work on it anymore.
        if obj not in self:
            if registerIfUnknown:
                self.register(obj)
            return obj
        # Follow the chain of links to the last tree (the one that points to the set of trees):
        steps = 0
        previous = obj
        maxIter = len(self)
        while not isinstance(obj, OrderedSet):
            steps += 1
            previous = obj
            obj = self[obj]
            # Error in case of circle (should never happen, but we'll se ...)
            assert steps <= maxIter
        # Remove unnecessary chains ('previousTree' is now the representative):
        if steps > 1:
            self[original] = previous
            
        return previous
        
    def getAllShapes(self, obj, registerIfMissing:bool = True):
        if obj in self:
            rep = self.getRepresentative(obj)
            return self[rep]
        elif registerIfMissing:
            self.register(obj)
            return self[obj]
        else:
            raise KeyError(f"[Eq.cache] Trying to retrieve equivalencies of {obj}, but that object is not known!")
    
    def link(self, newObj, knownObj, mergeIfKnown: bool = True):
        knownRep = self.getRepresentative(knownObj, registerIfUnknown=True)
        if newObj in self:
            if self.getRepresentative(newObj) == knownRep:
                pass
                # print(f"[Eq.cache] {newObj} and {knownRep} were already linked!")
            elif mergeIfKnown:
                self.merge(newObj, knownObj)
            else:
                print(f"[Eq.cache] object {newObj} wasn't new! Leaving the mapping as it was.")
        else:
            self[newObj] = knownObj
            self[knownRep].add(newObj)
            
    def merge(self, obj1, obj2):
        # Get Reps
        obj1 = self.getRepresentative(obj1, registerIfUnknown=True)
        obj2 = self.getRepresentative(obj2, registerIfUnknown=True)
        # Update set of 1, make 2 point to 1:
        theSet = self[obj1]
        for item in self[obj2]:
            theSet.add(item) # Sadly, orderedSet doesn't seem to have an update function.
        self[obj2] = obj1 # This increases chain length by 1, but it will be cleaned when encountered later.