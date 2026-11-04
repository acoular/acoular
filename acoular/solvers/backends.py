# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""Solver backend registry for concrete solver implementations.

Registers backend implementations per solver family (e.g. 'nnls', 'fista_lasso'),
distinguishing backends that are registered from backends that are currently
importable, so optional dependencies like PyLops can be registered without being
installed.
"""

import importlib.util

_SOLVER_BACKENDS = {}  # {family: {backend_name: {'func': callable, 'dependency': str or None}}}


def register_solver_backend(family, backend_name, func, dependency=None):
    """Register a backend implementation for a solver family.

    Parameters
    ----------
    family : str
        Solver family/objective name, e.g. 'nnls', 'fista_lasso'.
    backend_name : str
        Name of the backend implementation, e.g. 'sklearn', 'pylops'.
    func : callable
        Backend function with signature (solver, problem, dictionary_matrix, data, start_value).
    dependency : str, optional
        Name of the external package this backend requires, if any. Registration
        succeeds even if this package is not installed; only calling the backend
        function later requires it.

    Raises
    ------
    TypeError
        If *family*/*backend_name* are not strings, or *func* is not callable.
    ValueError
        If (*family*, *backend_name*) is already registered.
    """
    if not isinstance(family, str) or not isinstance(backend_name, str):
        msg = 'Solver family and backend name must be strings.'
        raise TypeError(msg)
    if not callable(func):
        msg = 'Solver backend implementation must be callable.'
        raise TypeError(msg)
    family_backends = _SOLVER_BACKENDS.setdefault(family, {})
    if backend_name in family_backends:
        msg = f'Backend {backend_name!r} for {family!r} is already registered.'
        raise ValueError(msg)
    family_backends[backend_name] = {'func': func, 'dependency': dependency}
    return func


def registered_solver_backends(family):
    """Names registered for *family*, regardless of dependency availability."""
    return tuple(_SOLVER_BACKENDS.get(family, {}))


def _dependency_available(dependency):
    return dependency is None or importlib.util.find_spec(dependency) is not None


def available_solver_backends(family):
    """Names registered for *family* whose dependency is currently importable."""
    return tuple(
        name for name, meta in _SOLVER_BACKENDS.get(family, {}).items() if _dependency_available(meta['dependency'])
    )


def get_solver_backend(family, backend_name):
    """Return the registered backend function for (*family*, *backend_name*).

    Raises
    ------
    ValueError
        If no backend is registered under that name for that family.
    """
    try:
        return _SOLVER_BACKENDS[family][backend_name]['func']
    except KeyError:
        available = ', '.join(registered_solver_backends(family))
        msg = f'Unknown backend {backend_name!r} for {family!r}. Registered backends: {available}.'
        raise ValueError(msg) from None


def solver_backend_info(family, backend_name):
    """Metadata for one registered backend: its dependency name and availability.

    Raises
    ------
    ValueError
        If no backend is registered under that name for that family.
    """
    try:
        meta = _SOLVER_BACKENDS[family][backend_name]
    except KeyError:
        available = ', '.join(registered_solver_backends(family))
        msg = f'Unknown backend {backend_name!r} for {family!r}. Registered backends: {available}.'
        raise ValueError(msg) from None
    return {'dependency': meta['dependency'], 'available': _dependency_available(meta['dependency'])}
