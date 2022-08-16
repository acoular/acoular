# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------

from hashlib import md5

def digest( obj, name='digest'):
    str_ = [str(obj.__class__).encode("UTF-8")]
    for do_ in obj.trait(name).depends_on:
        vobj = obj
        try:
            for i in do_.split('.'):               
                vobj = list(vobj.trait_get(i.rstrip('[]')).values())[0]
            str_.append(str(vobj).encode("UTF-8"))
        except:
            pass
    return '_' + md5(''.encode("UTF-8").join(str_)).hexdigest()

def ldigest(l):
    str_ = []
    for i in l:
        str_.append(str(i.digest).encode('UTF-8'))
    return '_' + md5(''.encode("UTF-8").join(str_)).hexdigest()


