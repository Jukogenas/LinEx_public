#!/usr/bin/env python3
# -*- coding: utf-8 -*-
class Dimension():
    """
    Dimensions are now objects (formerly they were simple integers) to avoid the trouble and confusion of having to
    change their numbers whenever trees were combined (because individual trees all began counting at 0).

    Dimensions of the same variable (occurring more than once) are the same object (handled by the parser).
    Of all identical dimensions, only one, the representant, is guaranteed to have the most up-to-date info about the
    name(s) and numeric sizes of the dimensions. All others simply link to their representant (or an intermediate dim
    one step closer to the representant) in a sort of star pattern. When requested, they query their representant for the
    necessary information and update their link to point directly to the rep in case it didn't.
    
    When two dimensions must be linked (because the user requested it or an operation requires it), both query their
    representants and let *them* link. One of the two becomes the new representant and inherits the merged data, the
    other points to that one (and with it all those that had it as the rep, now with a chain length increased by 1).
    
    If the data can't be merged (for example, if two dimensions are numerically different but should be equal), the
    attempt terminates in an error.
    
    Reshaping only very rarely creates new leaves (and thus new dimensions), so keeping the old, useless pointer objects
    in place isn't too much trouble. Going through the tree and setting everything to reps would work, but is not really
    necessary.
    """
    dimCounter = 0
    
    def __init__(self, origin: str, userAlias: str = '', numeric = 0):
        if userAlias:
            nickName = userAlias
        elif origin:
            nickName = origin
        else:
            nickName = ""
            
        self.linked = (userAlias, nickName, numeric, [origin])
        
        # The following is definitely not threadsafe (does python have atomic ints?):
        self.ID = Dimension.dimCounter
        Dimension.dimCounter += 1

    def __hash__(self):
        return self.ID

    def __eq__(self, other: 'Dimension'):
        return isinstance(other, Dimension) and \
               self.getRepresentative().ID == other.getRepresentative().ID

    def getData(self):
        return self.getRepresentative().linked
    
    def getRepresentative(self):
        original = self
        rep = self
        steps = 0
        maxIter = Dimension.dimCounter
        while not isinstance(rep.linked, tuple):
            rep = rep.linked
            assert steps <= maxIter
        
        # Remove unnecessary chains:
        if steps > 1:
            self.linked = rep
    
        return rep

    @classmethod
    def mergeData(cls, data1: tuple, data2: tuple):
        alias1, nickName1, numeric1, origins1 = data1
        alias2, nickName2, numeric2, origins2 = data2
        alias, nickName, numeric, origins = "", "", 0, list(set(origins1 + origins2))
        
        # Alias: If they have one, use that. If they have different ones, that's an error.
        if alias1:
            if alias2 and alias2 != alias1:
                raise DimensionMismatch(data1, data2, "Dimensions should match but have different userAliases:")
            else:
                alias = alias1
        else:
            alias = alias2
            
        # Nickname: Any is fine, but the alias is preferred.
        if alias:
            nickName = alias
        elif nickName1:
            nickName = nickName1
        else:
            nickName = nickName2
            
        # Numeric: Works like alias does.
        if numeric1:
            if numeric2 and numeric2 != numeric1:
                raise DimensionMismatch(data1, data2, "Dimensions should match but have different userAliases:")
            else:
                numeric = numeric1
        else:
            numeric = numeric2
            
        # Origins: Nothing to do. I already merged them all into one list without duplicates.
        
        return alias, nickName, numeric, origins
        

    def link(self, other: 'Dimension'):
        myRep = self.getRepresentative()
        otherRep = other.getRepresentative()
        
        if myRep.ID == otherRep.ID:
            return # We were already linked.
        
        myData = myRep.linked
        otherData = otherRep.linked
        
        # This step will fail if we're trying to link differently named or numerically different dims:
        newData = Dimension.mergeData(myData, otherData)
        
        myRep.linked = newData
        otherRep.linked = myRep # They point to us now. Chain lengths increase by 1, but that will be fixed.

    def allNames(self, includeAlias: bool = False):
        data = self.getData()
        names = data[3]
        if includeAlias and data[0]:
            names.append(data[0])
        return " == ".join(name for name in sorted(names) if not name.startswith("unnamed"))

    def setNickname(self, s):
        rep = self.getRepresentative()
        a, _, c, d = rep.linked # The second entry is the old nickname.
        rep.linked = a, s, c, d

    # TODO: I would prefer if alias / numeric value was set once on __init__ and then left alone.
    #  Alas, that is not how I made the parser do it (and I don't have time to fiddle with that).
    def setAlias(self, s):
        rep = self.getRepresentative()
        _, b, c, d = rep.linked # The first entry is the old alias.
        rep.linked = s, b, c, d

    def setNumeric(self, n):
        rep = self.getRepresentative()
        a, b, _, d = rep.linked # The third entry is the old numeric value.
        rep.linked = a, b, n, d

    def getAlias(self):
        return self.getData()[0]

    def getNumericValue(self, defVal=100):
        numeric = self.getData()[2]
        if numeric:
            return numeric
        else:
            return defVal

    def anyRef(self):
        """
        Return an arbitrary way to refer to this dimension for later use in code. Can't use the alias.
        This will generally be the origin, if it had one with a name.
        It will be an arbitrary linked name if not.
        It will be the numeric value (default 100) if it doesn't have one of those either.
        """
        origins = [origin for origin in self.getData()[3] if not origin.startswith("unnamed")]
        if origins:
            return origins[0]
        else:
            return str(self.getNumericValue())

    def anyCode(self):
        """
        Returns a string to reference this dimension in code, based on the origin or the numeric value.
        """
        return self.anyRef().replace('_rows', '.shape[0]').replace('_cols', '.shape[1]')

    def __str__(self):
        """
        Write a string representation of this dimension in the form of "name : value"
        """
        data = self.getData()
        if data[0]:
            name = data[0] # Prefer user alias
        else:
            name = data[1] # Other name if the user didn't give one
            
        val = self.getNumericValue()

        # If there was no nickname, simply use the value alone.
        if name:
            return f"{name}: {val}"
        else:
            return val

    @staticmethod
    def toRef(dim):
        if isinstance(dim, Dimension):
            return dim.anyRef()
        else:
            return dim

    @staticmethod
    def toCode(dim):
        if isinstance(dim, Dimension):
            return dim.anyCode()
        else:
            return dim

    @staticmethod
    def toNumber(dim):
        if isinstance(dim, Dimension):
            return dim.getNumericValue()
        else:
            return dim

class DimensionMismatch(Exception):
    def __init__(self, dim1, dim2, message=''):
        self.message = message + f" ({str(dim1)}, {str(dim2)})"

    def __str__(self):
        return self.message