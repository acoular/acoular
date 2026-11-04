# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for internal CMF problem-assembly helpers in BeamformerCMF."""

from unittest.mock import patch

import acoular as ac

import numpy as np


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
    grid = ac.RectGrid(x_min=0, x_max=2, y_min=0, y_max=0, z=0.5, increment=1)  # 3 grid points, != nc mics
    steer = ac.SteeringVector(grid=grid, mics=mics)
    freq_data = ac.PowerSpectraImport(
        csm=np.eye(2, dtype='complex128')[np.newaxis, :, :], frequencies=np.array([1000.0])
    )

    bf_full = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=False)
    bf_diag = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=True)

    h = np.array([[1 + 1j, 0.5 - 0.5j, 2 + 0j], [0 + 1j, 1 + 0j, 0 - 0j]])  # nc=2, num_points=3
    with patch.object(ac.SteeringVector, 'transfer', return_value=h.T):
        A_full = bf_full._calc_sensing_matrix(1000.0)
        A_diag = bf_diag._calc_sensing_matrix(1000.0)

    np.testing.assert_allclose(A_full, [[2, 0.5, 4], [1, 0.5, 0], [1, 1, 0], [-1, -0.5, 0]])
    np.testing.assert_allclose(A_diag, [[1, 0.5, 0], [-1, -0.5, 0]])
