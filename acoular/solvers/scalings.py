# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Problem scaling transforms for inverse methods.

.. autosummary::
    :toctree: generated/

    register_problem_scaling
    get_problem_scaling
    available_problem_scalings
    no_scaling
    unit_l2_scaling

Custom scaling transforms can be registered either as decorators or by passing
an existing function explicitly. A transform receives an array and an optional
axis, and returns the scaled array together with the scale factor used by the
problem to recover the absolute source strengths.

Decorator registration:

>>> import numpy as np
>>> from acoular.solvers.scalings import get_problem_scaling, register_problem_scaling
>>> @register_problem_scaling('doc_half')
... def half_scaling(array, axis=None):
...     return array / 2.0, 2.0
>>> scaled, scale = get_problem_scaling('doc_half')(np.array([2.0, 4.0]))
>>> scaled
array([1., 2.])
>>> scale
2.0

Explicit function registration:

>>> def tenth_scaling(array, axis=None):
...     return array / 10.0, 10.0
>>> register_problem_scaling('doc_tenth', tenth_scaling) is tenth_scaling
True
>>> get_problem_scaling('doc_tenth')(np.array([10.0, 20.0]))[0]
array([1., 2.])

Registered names are stable and cannot be overwritten:

>>> register_problem_scaling('doc_half', half_scaling) # doctest: +ELLIPSIS
Traceback (most recent call last):
...
ValueError: Problem scaling 'doc_half' is already registered.
"""

import scipy.linalg as spla

_PROBLEM_SCALING_TRANSFORMS = {}


def register_problem_scaling(name, func=None):
    """Register a problem scaling transform.

    Registered names are stable identifiers and cannot be overwritten. The
    function can be used as a decorator or with an explicit callable.

    Parameters
    ----------
    name : str
        Name under which the scaling transform is registered.
    func : callable, optional
        Scaling transform to register. If omitted, a decorator is returned.

    Returns
    -------
    callable
        The registered scaling transform if *func* is given, otherwise a
        decorator that registers the decorated function.

    Raises
    ------
    TypeError
        If *name* is not a string or if *func* is not callable.
    ValueError
        If *name* is already registered.
    """
    if not isinstance(name, str):
        msg = 'Problem scaling name must be a string.'
        raise TypeError(msg)

    def decorator(transform):
        if not callable(transform):
            msg = 'Problem scaling transform must be callable.'
            raise TypeError(msg)
        if name in _PROBLEM_SCALING_TRANSFORMS:
            msg = f'Problem scaling {name!r} is already registered.'
            raise ValueError(msg)
        _PROBLEM_SCALING_TRANSFORMS[name] = transform
        return transform

    if func is None:
        return decorator
    return decorator(func)


def available_problem_scalings():
    """Return the names of all registered problem scaling transforms.

    Returns
    -------
    tuple of str
        Names of all registered problem scaling transforms.
    """
    return tuple(_PROBLEM_SCALING_TRANSFORMS)


def get_problem_scaling(name):
    """Return a registered problem scaling transform.

    Parameters
    ----------
    name : str
        Name of the requested scaling transform.

    Returns
    -------
    callable
        Scaling transform registered under *name*.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    try:
        return _PROBLEM_SCALING_TRANSFORMS[name]
    except KeyError:
        available = ', '.join(available_problem_scalings())
        msg = f'Unknown problem scaling {name!r}. Available scalings are: {available}.'
        raise ValueError(msg) from None


@register_problem_scaling('none')
def no_scaling(array, axis=None):  # noqa: ARG001
    """Return the input unchanged.

    Parameters
    ----------
    array : array-like
        Input array to be scaled.
    axis : int or None, optional
        Accepted for API compatibility with other scaling transforms. Ignored
        by this transform.

    Returns
    -------
    array-like
        Unchanged input array.
    float
        Scale factor ``1.0``.
    """
    return array, 1.0


@register_problem_scaling('unit_l2')
def unit_l2_scaling(array, axis=None):
    """Scale an input by its L2 norm.

    Parameters
    ----------
    array : array-like
        Input array to be scaled.
    axis : int or None, optional
        Axis along which the L2 norm is computed. If ``None``, the norm of the
        flattened input is used.

    Returns
    -------
    array-like
        Input array divided by its L2 norm.
    float or array-like
        L2 norm used as scale factor.
    """
    scale = spla.norm(array, axis=axis)
    return array / scale, scale
