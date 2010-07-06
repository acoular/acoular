"""
internal.py

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""
import hashlib

def digest( obj, name='digest'):
    str_ = []
    for do_ in obj.trait(name).depends_on:
        vobj = obj
        try:
            for i in do_.split('.'):               
                vobj = vobj.get(i).values()[0]
            str_.append(str(vobj))
        except:
            pass
    return '_' + hashlib.md5(''.join(str_)).hexdigest()
