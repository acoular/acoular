# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for internal CMF problem-assembly helpers in BeamformerCMF."""

from unittest.mock import patch

import acoular as ac

import numpy as np


def test_sensing_matrix_and_csm_consistency_at_scale():
    """Structural invariants should hold regardless of array size, not just for the 2-mic case."""
    rng = np.random.default_rng(0)
    nc = 5
    mics = ac.MicGeom(pos_total=rng.uniform(-0.3, 0.3, size=(3, nc)))
    grid = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.5, increment=0.2)
    steer = ac.SteeringVector(grid=grid, mics=mics)

    raw = rng.normal(size=(nc, nc)) + 1j * rng.normal(size=(nc, nc))
    csm = raw + raw.conj().T  # random Hermitian CSM
    freq_data = ac.PowerSpectraImport(csm=csm[np.newaxis, :, :], frequencies=np.array([1000.0]))

    for r_diag in (False, True):
        bf = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=r_diag)
        A = bf._calc_sensing_matrix(1000.0)
        R = bf._vectorize_csm(csm)
        assert A.shape[0] == R.shape[0]
        assert A.shape[1] == grid.size
        assert not np.isnan(A).any()
        assert not np.isnan(R).any()


def test_vectorize_csm_values():
    """Verify _vectorize_csm produces exact correct values, for both r_diag settings."""
    mics = ac.MicGeom(pos_total=np.zeros((3, 2)))
    grid = ac.RectGrid(x_min=0, x_max=1, y_min=0, y_max=0, z=0.5, increment=1)
    steer = ac.SteeringVector(grid=grid, mics=mics)

    csm = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])  # non-trivial: nonzero off-diagonal and imaginary
    freq_data = ac.PowerSpectraImport(csm=csm[np.newaxis, :, :], frequencies=np.array([1000.0]))

    bf_full = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=False)
    bf_diag = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=True)

    np.testing.assert_allclose(bf_full._vectorize_csm(csm).ravel(), [2, 1, 3, -1])
    np.testing.assert_allclose(bf_diag._vectorize_csm(csm).ravel(), [1, -1])


def test_calc_sensing_matrix_values():
    """Verify _calc_sensing_matrix produces exact correct values, for both r_diag settings."""
    mics = ac.MicGeom(pos_total=np.zeros((3, 2)))
    grid = ac.RectGrid(x_min=0, x_max=1, y_min=0, y_max=0, z=0.5, increment=1)
    steer = ac.SteeringVector(grid=grid, mics=mics)
    freq_data = ac.PowerSpectraImport(csm=np.eye(2, dtype='complex128')[np.newaxis, :, :], frequencies=np.array([1000.0]))

    bf_full = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=False)
    bf_diag = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=True)

    h = np.array([[1 + 1j, 0.5 - 0.5j], [0 + 1j, 1 + 0j]])  # fixed, known steering matrix
    with patch.object(ac.SteeringVector, 'transfer', return_value=h.T):
        A_full = bf_full._calc_sensing_matrix(1000.0)
        A_diag = bf_diag._calc_sensing_matrix(1000.0)

    np.testing.assert_allclose(A_full, [[2, 0.5], [1, 0.5], [1, 1], [-1, -0.5]])
    np.testing.assert_allclose(A_diag, [[1, 0.5], [-1, -0.5]])
