"""
internal.py

Part of the acoular library: several classes for the implementation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2011, all rights reserved
ennes.sarradj@gmx.de
"""
from hashlib import md5

def digest( obj, name='digest'):
    str_ = []
    for do_ in obj.trait(name).depends_on:
        vobj = obj
        try:
            for i in do_.split('.'):               
                vobj = vobj.get(i.rstrip('[]')).values()[0]
            str_.append(str(vobj))
        except:
            pass
    return '_' + md5(''.join(str_)).hexdigest()
