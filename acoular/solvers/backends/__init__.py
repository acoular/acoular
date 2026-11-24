"""Backend registry and per-library backend implementations for inverse-problem solvers.

.. autosummary::
    :toctree: generated/

    sklearn_backends
    scipy_backends
    pylops_backends
"""

from . import pylops_backends, scipy_backends, sklearn_backends  # noqa: F401
from .backends import (
    available_solver_backends,
    get_solver_backend,
    register_solver_backend,
    registered_solver_backends,
    solver_backend_info,
)

__all__ = [
    'available_solver_backends',
    'get_solver_backend',
    'register_solver_backend',
    'registered_solver_backends',
    'solver_backend_info',
]
