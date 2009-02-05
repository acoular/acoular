"""
internal.py (c) Ennes Sarradj 2008, all rights reserved
"""
import md5

def digest( object, name='digest'):
    s = []
    for do in object.trait(name).depends_on:
        v = object
        try:
            for x in do.split('.'):               
                v = v.get(x).values()[0]
            s.append(str(v))
        except:
            pass
    return '_'+md5.new(''.join(s)).hexdigest()
