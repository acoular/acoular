# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""Implements a digest function for caching of traits based on a unique identifier."""

from hashlib import md5


def digest(obj, name='digest'):
    """Generate a unique digest for the given object based on its traits."""
    str_ = [str(obj.__class__).encode('UTF-8')]
    for do_ in obj.trait(name).depends_on:
        vobj = obj
        try:
            for i in do_.split('.'):
                vobj = list(vobj.trait_get(i.rstrip('[]')).values())[0]
            str_.append(str(vobj).encode('UTF-8'))
        except:  # noqa: E722
            pass
    return '_' + md5(b''.join(str_)).hexdigest()


def ldigest(obj_list):
    """Generate a unique digest for a list of objects based on their traits."""
    str_ = []
    for i in obj_list:
        str_.append(str(i.digest).encode('UTF-8'))
    return '_' + md5(b''.join(str_)).hexdigest()
