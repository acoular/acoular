# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""PyLops least-squares solver implementation.

This module provides a configurable PyLops-based least-squares solver for
inverse methods such as :class:`acoular.fbeamform.BeamformerCMF`.

.. autosummary::
    :toctree: ../../generated/

    PylopsLeastSquaresSolver
    CallbackBase
    IntermediateResultCallback
    SolutionResidualCallback
    SolutionResidualL1Callback
    RelativeResidualCallback
    AbsoluteResidualCallback
"""

import numpy as np
from traits.api import Enum, Instance, List, Union

from acoular.configuration import config
from acoular.solvers.base import LeastSquaresSolver

if config.have_pylops:
    import pylops


if config.have_pylops:

    class PylopsLeastSquaresSolver(LeastSquaresSolver):
        """Least-squares solver backed by PyLops sparsity optimizers.

        The backend is selected by :attr:`method`:

        - ``'ISTA'`` and ``'FISTA'`` use class-based PyLops solvers from
          :mod:`pylops.optimization.sparsity`
        - ``'Split_Bregman'`` uses
          :class:`pylops.optimization.cls_sparsity.SplitBregman`

        All backend-specific solver arguments are forwarded from :attr:`options`.
        Typical keys are ``eps``, ``niter``, and ``show`` for ISTA/FISTA and
        ``RegsL1``, ``niter_outer``, and ``show`` for Split Bregman.
        """

        #: Solver backend to use.
        method = Enum('ISTA', 'FISTA', 'Split_Bregman')

        #: Optional callback(s) for the underlying PyLops optimization routine.
        callback = Union(
            None,
            Instance(pylops.optimization.callback.Callbacks),
            List(Instance(pylops.optimization.callback.Callbacks)),
        )

        def solve(self, A, y, x, index):  # noqa: N803
            """Solve one inverse problem for a single frequency bin.

            Parameters
            ----------
            A : array_like
                Sensing matrix for the selected frequency (unnormalized).
            y : array_like
                Measurement vector (unscaled).
            x : array_like
                Initial estimate of the solution vector.
            index : int
                Frequency index used as key in :attr:`output`.

            Returns
            -------
            array_like
                Estimated solution vector (rescaled to original units).

            Notes
            -----
            :attr:`options` is forwarded directly to the selected backend
            ``solve`` method. Callback configuration is handled via
            :attr:`callback`.

            The solver applies column normalization using :attr:`norms` and
            measurement scaling using :attr:`unit` internally. The returned
            solution is automatically rescaled. Callbacks receive the internal
            solver state, which includes ``norms`` and ``unit`` attributes for
            manual rescaling if needed.
            """
            # Apply normalization and scaling
            A_normalized = A / self.norms  # noqa: N806
            y_scaled = np.asarray(y).ravel() * self.unit
            x0_vec = np.asarray(x).ravel()

            operator = pylops.MatrixMult(A_normalized)
            solver_options = dict(self.options)
            callbacks = None
            if self.callback is not None:
                callbacks = self.callback if isinstance(self.callback, list) else [self.callback]

            if self.method == 'ISTA':
                model_cls = pylops.optimization.sparsity.ISTA
                model_instance = model_cls(operator, callbacks=callbacks)
                # Store norms and unit on solver instance for callback access
                model_instance.norms = self.norms
                model_instance.unit = self.unit
                xhat, ntotal, cost = model_instance.solve(y=y_scaled, x0=x0_vec, **solver_options)
            elif self.method == 'FISTA':
                model_cls = pylops.optimization.sparsity.FISTA
                model_instance = model_cls(operator, callbacks=callbacks)
                # Store norms and unit on solver instance for callback access
                model_instance.norms = self.norms
                model_instance.unit = self.unit
                xhat, ntotal, cost = model_instance.solve(y=y_scaled, x0=x0_vec, **solver_options)
            elif self.method == 'Split_Bregman':
                # Split_Bregman requires RegsL1 as a positional argument
                model_cls = pylops.optimization.cls_sparsity.SplitBregman
                # Extract alpha and create regularization operator
                alpha = solver_options.pop('alpha', 0.0)
                num_points = A.shape[1]
                iop = alpha * pylops.Identity(num_points)
                model_instance = model_cls(operator, callbacks=callbacks)
                # Store norms and unit on solver instance for callback access
                model_instance.norms = self.norms
                model_instance.unit = self.unit
                xhat, ntotal, cost = model_instance.solve(y=y_scaled, RegsL1=[iop], x0=x0_vec, **solver_options)
            else:
                msg = f'Unsupported method: {self.method}'
                raise ValueError(msg)

            self.output = {index: {'ntotal': ntotal, 'cost': cost}}
            # Rescale solution to original units
            return np.asarray(xhat).squeeze() / self.norms / self.unit

    class CallbackBase(pylops.optimization.callback.Callbacks):
        """Base callback to monitor optimization convergence.

        Parameters
        ----------
        tol : float, optional
            Convergence threshold for callback-specific stopping criterion.
        """

        def __init__(self, tol=1e-10):
            self.tol = tol
            self.cost = []
            self.stop = False

    class IntermediateResultCallback(pylops.optimization.callback.Callbacks):
        """Store intermediate solution estimates during optimization.

        Intermediate results are automatically rescaled to original absolute
        values using the solver's ``norms`` and ``unit`` attributes.

        Parameters
        ----------
        n_iter : int or sequence of int or None, optional
            If an integer is provided, store all iterates up to and including
            this iteration index. If a sequence is provided, store only these
            explicit iteration indices. If None, store all iterations.
        """

        def __init__(self, n_iter=None):
            super().__init__()
            self.x_steps = []
            self.i_steps = []
            self.x_by_iter = {}
            self.n_iter = n_iter
            self._iter_max = None
            self._iter_set = None

            if n_iter is None:
                return

            if np.isscalar(n_iter):
                self._iter_max = int(n_iter)
                return

            n_iter_array = np.asarray(n_iter)
            if n_iter_array.ndim != 1:
                msg = 'n_iter must be an int, a 1D sequence of ints, or None.'
                raise ValueError(msg)
            self._iter_set = {int(i) for i in n_iter_array}

        def on_step_end(self, solver, x):
            """Store current solution estimate for configured iterations.

            The stored solution is automatically rescaled to original absolute
            values using ``solver.norms`` and ``solver.unit`` if available.
            """
            iiter = solver.iiter

            if self._iter_set is not None:
                should_store = iiter in self._iter_set
            elif self._iter_max is None:
                should_store = True
            else:
                should_store = iiter <= self._iter_max

            if should_store:
                xcopy = x.copy()
                # Rescale to original absolute values
                norms = getattr(solver, 'norms', np.ones_like(x))
                unit = getattr(solver, 'unit', 1.0)
                xcopy = xcopy / norms / unit

                self.i_steps.append(iiter)
                self.x_steps.append(xcopy)
                self.x_by_iter[iiter] = xcopy

    class SolutionResidualCallback(CallbackBase):
        """Monitor relative solution change in L2 norm."""

        def on_run_begin(self, solver, x):  # noqa: ARG002
            """Initialize solution state at first iteration."""
            self.x0 = x.copy()

        def on_step_end(self, solver, x):
            """Update relative L2 solution residual and stopping flag."""
            if solver.iiter <= 1:
                self.xold = self.x0
                return
            sres = np.linalg.norm(x - self.xold) / np.linalg.norm(x)
            self.cost.append(sres)
            self.xold = x
            if sres < self.tol:
                print(f'Converged at iteration {solver.iiter} with sres={sres:.2e} < tol={self.tol:.2e}')
                self.stop = True

    class SolutionResidualL1Callback(CallbackBase):
        """Monitor relative solution change in L1 norm."""

        def on_run_begin(self, solver, x):  # noqa: ARG002
            """Initialize solution state at first iteration."""
            self.x0 = x.copy()

        def on_step_end(self, solver, x):
            """Update relative L1 solution residual and stopping flag."""
            if solver.iiter <= 1:
                self.xold = self.x0
                return
            sres1 = np.linalg.norm(x - self.xold, ord=1) / np.linalg.norm(self.xold, ord=1)
            self.cost.append(sres1)
            self.xold = x
            if sres1 < self.tol:
                print(f'Converged at iteration {solver.iiter} with sres1={sres1:.2e} < tol={self.tol:.2e}')
                self.stop = True

    class RelativeResidualCallback(CallbackBase):
        """Monitor relative data residual ``||A x - y|| / ||y||``."""

        def __init__(self, tol=1e-10, return_db=True):
            super().__init__(tol=tol)
            self.return_db = return_db

        def on_run_begin(self, solver, x):  # noqa: ARG002
            """Store norm of measurement vector before optimization starts."""
            self.y_norm = np.linalg.norm(solver.y)

        def on_step_end(self, solver, x):
            """Update relative data residual and stopping flag."""
            if solver.iiter <= 1:
                return
            res = np.linalg.norm(solver.Op @ x - solver.y)
            rres = res / self.y_norm

            rres_value = 20 * np.log10(rres) if self.return_db else rres

            self.cost.append(rres_value)
            if rres < self.tol:
                tol_db = 20 * np.log10(self.tol) if self.return_db else self.tol
                print(f'Converged at iteration {solver.iiter} with rres={rres_value:.2e} < tol={tol_db:.2e}')
                self.stop = True

    class AbsoluteResidualCallback(CallbackBase):
        """Monitor absolute data residual ``||A x - y||``."""

        def on_step_end(self, solver, x):
            """Update absolute data residual and stopping flag."""
            if solver.iiter <= 1:
                return
            res = np.linalg.norm(solver.Op @ x - solver.y)
            self.cost.append(res)
            if res < self.tol:
                print(f'Converged at iteration {solver.iiter} with res={res:.2e} < tol={self.tol:.2e}')
                self.stop = True

    class PositivityCallback(pylops.optimization.callback.Callbacks):
        """Project negative entries to zero after each optimization step."""

        def on_step_end(self, solver, x):  # noqa: ARG002
            """Apply positivity projection to the current iterate."""
            x[x < 0] = 0

else:

    class PylopsLeastSquaresSolver(LeastSquaresSolver):
        """Fallback class used when PyLops is not available."""

        method = Enum('ISTA', 'FISTA', 'Split_Bregman')
        callback = Union(None)

        def solve(self, A, y, x, index):  # noqa: N803, ARG002
            """Raise an error because PyLops is required."""
            msg = 'Cannot import Pylops package. No Pylops installed.'
            raise ImportError(msg)
