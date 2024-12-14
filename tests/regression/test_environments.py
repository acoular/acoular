# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements testing of environments and flows."""

import acoular as ac
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from tests.cases.test_environments_cases import Environments, Flows

GRID = ac.RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=0.5, z_max=0.9, increment=0.2)
MICS = ac.MicGeom(pos_total=((0.5, 0.5, 0), (0, 0, 0), (-0.5, -0.5, 0)))


@pytest.mark.parametrize('grid', [GRID.pos], ids=['RectGrid3D'])
@parametrize_with_cases('flow', cases=Flows)
def test_flow(snapshot, grid, flow):
    """Performs snapshot testing with snapshot fixture from pytest-regtest.

    To overwrite the collected snapshots, run:

    ```bash
    pytest -v --regtest-reset tests/regression/test_environments.py::test_flow
    ```

    Parameters
    ----------
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    grid : numpy.ndarray
        Grid points to test
    flow : instance of acoular.environment.FlowField
        FlowField to test (cases from Flows)
    """
    result = np.array([np.vstack(flow.v(x)) for x in grid.T])
    snapshot.check(result, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize('mics', [MICS.pos, 0.0], ids=['mics-arr', 'mics-scalar'])
@pytest.mark.parametrize('grid', [GRID.pos], ids=['RectGrid3D'])
@parametrize_with_cases('env', cases=Environments)
def test_environment(snapshot, grid, mics, env):
    """Performs snapshot testing with snapshot fixture from pytest-regtest.

    To overwrite the collected snapshots, run:

    ```bash
    pytest -v --regtest-reset tests/regression/test_environments.py::test_environment
    ```

    Parameters
    ----------
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    grid : numpy.ndarray
        Grid points to test
    mics : numpy.ndarray or scalar
        Microphone positions to test
    env : instance of acoular.environment.Environment
        Environment to test (cases from Environments)
    """
    result = np.vstack((env._r(grid, mics).T, env._r(grid).T))
    snapshot.check(result, rtol=1e-5, atol=1e-8)
