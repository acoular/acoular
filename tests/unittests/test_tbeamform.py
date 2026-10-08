"""Tests for the time-domain beamformers with moving focus."""

import acoular as ac

import numpy as np
import pytest


@pytest.mark.parametrize(('x', 'sign'), [(5.0, 1.0), (-5.0, -1.0)])
def test_macostheta_sign_matches_moving_point_source(x, sign):
    """The radial Mach number must be positive for an approaching source.

    This is the convention of :class:`~acoular.sources.MovingPointSource`
    (vector from the source to the microphone, see #570/#571).
    """
    c = 343.0
    v = np.array([-50.0, 0.0, 0.0])  # source moves in -x direction
    mics = ac.MicGeom(pos_total=np.zeros((3, 1)))  # single microphone at the origin
    grid = ac.RectGrid(x_min=0, x_max=0, y_min=0, y_max=0, z=0, increment=1)
    steer = ac.SteeringVector(grid=grid, mics=mics, env=ac.Environment(c=c))
    bf = ac.BeamformerTimeTraj(steer=steer)

    tpos = np.array([[x], [0.0], [10.0]])  # grid point = source position
    rm = np.linalg.norm(tpos - mics.pos, axis=0)[:, np.newaxis]  # shape (n_grid, n_mics)
    expected = sign * np.linalg.norm(v) / c * abs(x) / np.hypot(x, 10.0)

    np.testing.assert_allclose(bf._get_macostheta(v, tpos, rm), [[expected]], rtol=1e-12)
