# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2017, Acoular Development Team.
#------------------------------------------------------------------------------

from hashlib import md5

def digest( obj, name='digest'):
    str_ = []
    for do_ in obj.trait(name).depends_on:
        vobj = obj
        try:
            for i in do_.split('.'):               
                vobj = list(vobj.get(i.rstrip('[]')).values())[0]
            str_.append(str(vobj).encode("UTF-8"))
        except:
            pass
    return '_' + md5(''.encode("UTF-8").join(str_)).hexdigest()
