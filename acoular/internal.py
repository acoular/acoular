# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""Internal helpers for caching, compilation, and iterator cleanup."""

from contextlib import contextmanager
from functools import wraps
from hashlib import blake2b


@contextmanager
def _closing_iterator(iterator):
    """Close upstream generators without requiring close() on ordinary iterators."""
    try:
        yield iterator
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            close()


def lazyjit(decorator):
    """Delay applying a decorator until the wrapped function is first called."""

    def decorate(func):
        compiled = None

        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal compiled
            if compiled is None:
                compiled = decorator(func)
            return compiled(*args, **kwargs)

        return wrapper

    return decorate


def digest(obj, name='digest'):
    """Generate a unique digest for the given object based on its traits."""
    str_ = [str(obj.__class__).encode('UTF-8')]
    for do_ in obj.trait(name).depends_on:
        vobj = obj
        try:
            for i in do_.split('.'):
                vobj = next(iter(vobj.trait_get(i.rstrip('[]')).values()))
            str_.append(str(vobj).encode('UTF-8'))
        except (AttributeError, KeyError, StopIteration, TypeError):
            continue
    return '_' + blake2b(b''.join(str_), digest_size=16).hexdigest()


def ldigest(obj_list):
    """Generate a unique digest for a list of objects based on their traits."""
    str_ = []
    for i in obj_list:
        str_.append(str(i.digest).encode('UTF-8'))
    return '_' + blake2b(b''.join(str_), digest_size=16).hexdigest()
