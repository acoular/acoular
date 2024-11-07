# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements testing of environments and helpers."""

import acoular as ac
import numpy as np
import pytest

from tests.setup import test_config

# if this flag is set to True
WRITE_NEW_REFERENCE_DATA = False
# results are generated for comparison during testing.
# Should always be False. Only set to True if it is necessary to
# recalculate the data due to intended changes of the Beamformers.

GRID = ac.RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=0.5, z_max=0.9, increment=0.2)
MICS = ac.MicGeom(mpos_tot=((0.5, 0.5, 0), (0, 0, 0), (-0.5, -0.5, 0)))


@pytest.mark.parametrize(
    'flow',
    [
        ac.SlotJet(v0=70.0, origin=(-0.7, 0, 0.7)),
        ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7)),
        ac.RotatingFlow(v0=70.0, rpm=1000.0),
    ],
)
def test_flow_results(flow):
    gc = GRID.gpos
    name = test_config.reference_data / f'{flow.__class__.__name__}.npy'
    actual_data = np.array([np.vstack(flow.v(x)) for x in gc.T])
    if WRITE_NEW_REFERENCE_DATA:
        np.save(name, actual_data)
    ref_data = np.load(name)
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    'env',
    [
        ac.Environment(),
        ac.UniformFlowEnvironment(ma=0.3),
        ac.GeneralFlowEnvironment(ff=ac.OpenJet(v0=70.0, origin=(-0.7, 0, 0.7))),
    ],
)
def test_env_results(env):
    mc = MICS.mpos
    gc = GRID.gpos
    name = test_config.reference_data / f'{env.__class__.__name__}.npy'
    # stack all results
    actual_data = np.vstack((env._r(gc, mc).T, env._r(gc).T))
    if WRITE_NEW_REFERENCE_DATA:
        np.save(name, actual_data)
    ref_data = np.load(name)
    np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)
