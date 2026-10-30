"""Tests for the time-domain beamformers."""

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


# --- spatial microphone weighting (weights='power') ---

FS = 8000
MICS = ac.MicGeom(pos_total=np.array([[0.1, -0.3, 0.6, -0.9], [0.0, 0.2, 0.0, -0.1], [0.0, 0.0, 0.0, 0.0]]))
GRID = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=1.0, increment=0.2)
DATA = np.random.default_rng(1).normal(size=(2048, MICS.num_mics))
# trajectory must cover all blocks, i.e. at least ceil(2048 / 256) * 256 samples = 0.256 s
TRAJ = ac.Trajectory(points={0.0: (-1.0, 0.0, 0.5), 0.4: (1.0, 0.0, 0.5)})


def _steer():
    return ac.SteeringVector(grid=GRID, mics=MICS)


def test_const_power_weight_shape():
    """Weights are one factor per microphone, normalised to mean 1."""
    w = ac.tbeamform.const_power_weight(ac.BeamformerTime(steer=_steer()))
    assert w.shape == (MICS.num_mics,)
    np.testing.assert_allclose(w.mean(), 1.0)


FIXED = [ac.BeamformerTime, ac.BeamformerTimeSq, ac.BeamformerCleant, ac.BeamformerCleantSq]
TRAJS = [ac.BeamformerTimeTraj, ac.BeamformerTimeSqTraj, ac.BeamformerCleantTraj, ac.BeamformerCleantSqTraj]
CASES = [pytest.param(c, {}, id=c.__name__) for c in FIXED] + [
    pytest.param(c, {'trajectory': TRAJ, 'precision': p}, id=f'{c.__name__}-{p}') for c in TRAJS for p in (64, 32)
]


@pytest.mark.parametrize(('cls', 'kwargs'), CASES)
def test_power_weights_equal_preweighted_data(cls, kwargs):
    """weights='power' must give the same result as weighting the input once beforehand.

    Several blocks (num=256) are processed so that weighting the overlapping
    buffer region more than once would be detected.
    """
    steer = _steer()
    w = ac.tbeamform.const_power_weight(ac.BeamformerTime(steer=steer)).reshape(-1)
    raw = ac.TimeSamples(data=DATA, sample_freq=FS)
    pre = ac.TimeSamples(data=DATA * w, sample_freq=FS)

    res = ac.tools.return_result(cls(source=raw, steer=steer, weights='power', **kwargs), num=256)
    ref = ac.tools.return_result(cls(source=pre, steer=steer, weights='none', **kwargs), num=256)

    assert res.dtype == ref.dtype
    rtol = 1e-4 if kwargs.get('precision') == 32 else 1e-10
    np.testing.assert_allclose(res, ref, rtol=rtol, atol=rtol * np.abs(ref).max())
