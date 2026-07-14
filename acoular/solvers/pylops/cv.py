# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Cross-validation wrappers for PyLops solver backends.

.. autosummary::
    :toctree: ../../generated/

    ISTACV
"""

import numpy as np
import sklearn
from traits.api import Bool, Dict, Float, Instance, Int, List, Union

from acoular.configuration import config
from acoular.solvers.base import LeastSquaresSolver

from .solver import PylopsLeastSquaresSolver

if config.have_pylops:
    import pylops


if config.have_pylops:

    class ISTACV(LeastSquaresSolver):
        """Cross-validated regularization wrapper for a PyLops solver.

        The class performs grid-search cross-validation over regularization
        values for the configured :attr:`solver` (``'ISTA'`` or ``'FISTA'``)
        and then solves the full inverse problem with the selected value.
        """

        #: Underlying PyLops solver instance to cross-validate.
        solver = Instance(PylopsLeastSquaresSolver, ())

        #: Number of regularization values in the CV grid.
        num_grid = Int(20)

        #: Ratio between minimum and maximum tested regularization.
        reg_ratio = Float(1e-3)

        #: Random seed for CV fold generation.
        seed = Int(42)

        #: Number of cross-validation folds.
        cv = Int(5)

        #: Optional options dictionary for CV runs.
        cv_options = Union(None, Dict)

        #: Optional callback(s) for CV runs.
        cv_callback = Union(
            None,
            Instance(pylops.optimization.callback.Callbacks),
            List(),
        )

        #: If True, reuse previous solution as initial guess in regularization grid.
        warm_start = Bool(True)

        def _validate_solver_method(self):
            """Ensure ISTACV is used only with ISTA/FISTA solvers."""
            if self.solver.method not in ('ISTA', 'FISTA'):
                msg = "ISTACV supports only solver methods 'ISTA' and 'FISTA'."
                raise ValueError(msg)

        def _train_once(self, A_train, y_train, x0, reg_value, callbacks):  # noqa: N803
            """Run one training solve for a given regularization value."""
            self._validate_solver_method()
            original_options = dict(self.solver.options)
            solver_options = dict(self.cv_options) if self.cv_options is not None else dict(original_options)
            solver_options['eps'] = reg_value

            original_callback = self.solver.callback

            try:
                self.solver.options = solver_options
                if not callbacks:
                    self.solver.callback = None
                elif len(callbacks) == 1:
                    self.solver.callback = callbacks[0]
                else:
                    self.solver.callback = callbacks
                # Propagate norms and unit to wrapped solver
                self.solver.norms = self.norms
                self.solver.unit = self.unit
                xhat = self.solver.solve(A=A_train, y=y_train, x=x0, index=-1)
            finally:
                self.solver.options = original_options
                self.solver.callback = original_callback

            return np.asarray(xhat).ravel()

        def grid_search(self, A, y, x0, reg):  # noqa: N803
            """Perform cross-validated regularization search."""
            y = np.asarray(y).ravel()
            n_samples = y.size
            kf = sklearn.model_selection.KFold(n_splits=self.cv, shuffle=True, random_state=self.seed)

            mean_val_losses = []
            reg = list(reg)
            x0 = np.asarray(x0).copy()

            for reg_value in reg:
                fold_losses = []
                for train_idx, val_idx in kf.split(np.arange(n_samples)):
                    a_train = A[train_idx, :]
                    y_train = y[train_idx]
                    a_val = A[val_idx, :]
                    y_val = y[val_idx]

                    callbacks = []
                    if self.cv_callback is not None:
                        if isinstance(self.cv_callback, list):
                            callbacks.extend(self.cv_callback)
                        else:
                            callbacks.append(self.cv_callback)

                    xhat = self._train_once(a_train, y_train, x0, reg_value, callbacks)
                    resid = (a_val @ xhat) - y_val
                    fold_losses.append(np.vdot(resid, resid).real / len(val_idx))

                if self.warm_start:
                    x0 = xhat
                mean_val_losses.append(np.asarray(fold_losses).mean())

            losses = np.asarray(mean_val_losses)
            return reg[int(np.argmin(losses))], losses

        def solve(self, A, y, x, index):  # noqa: N803
            """Solve one inverse problem after cross-validating regularization."""
            self._validate_solver_method()
            reg_max = np.max(np.abs(np.asarray(A).T @ np.asarray(y).ravel()))
            reg_values = np.geomspace(reg_max, reg_max * self.reg_ratio, self.num_grid)
            reg_best, losses = self.grid_search(A, y, x, reg_values)

            # Update solver options with best regularization value
            options = dict(self.solver.options)
            options['eps'] = reg_best
            self.solver.options = options

            # Propagate norms and unit to wrapped solver
            self.solver.norms = self.norms
            self.solver.unit = self.unit

            xhat = self.solver.solve(A=A, y=y, x=x, index=index)

            output_dict = {
                'reg': reg_values,
                'mean_val_loss': losses,
                'reg_best': reg_best,
                'solver_method': self.solver.method,
                'solver_output': self.solver.output.get(index, {}) if self.solver.output else {},
            }
            self.output = {index: output_dict}
            return xhat

else:

    class ISTACV(LeastSquaresSolver):
        """Fallback class used when PyLops is not available."""

        solver = Instance(PylopsLeastSquaresSolver, ())

        def solve(self, A, y, x, index):  # noqa: N803, ARG002
            """Raise an error because PyLops is required."""
            msg = 'Cannot import Pylops package. No Pylops installed.'
            raise ImportError(msg)
