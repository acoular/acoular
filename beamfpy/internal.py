"""
internal.py (c) Ennes Sarradj 2008, all rights reserved
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
