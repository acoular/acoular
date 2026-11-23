# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for internal CMF problem-assembly helpers in BeamformerCMF."""

from unittest.mock import patch

import acoular as ac

import numpy as np
import pytest


def test_vectorize_csm_values():
    """Verify _vectorize_csm produces exact correct values, for both r_diag settings."""
    mics = ac.MicGeom(pos_total=np.zeros((3, 2)))
    grid = ac.RectGrid(x_min=0, x_max=1, y_min=0, y_max=0, z=0.5, increment=1)
    steer = ac.SteeringVector(grid=grid, mics=mics)

    csm = np.array([[2 + 0j, 1 - 1j], [1 + 1j, 3 + 0j]])  # non-trivial: nonzero off-diagonal and imaginary
    freq_data = ac.PowerSpectraImport(csm=csm[np.newaxis, :, :], frequencies=np.array([1000.0]))

    bf_full = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=False)
    bf_diag = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=True)

    sqrt2 = np.sqrt(2.0)
    np.testing.assert_allclose(bf_full._vectorize_csm(csm).ravel(), [2, sqrt2, 3, -sqrt2])
    np.testing.assert_allclose(bf_diag._vectorize_csm(csm).ravel(), [sqrt2, -sqrt2])


def test_build_dictionary_values():
    """Verify _build_dictionary produces exact correct values, for both r_diag settings."""
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
        dictionary_full = bf_full._build_dictionary(1000.0)
        dictionary_diag = bf_diag._build_dictionary(1000.0)

    sqrt2 = np.sqrt(2.0)
    np.testing.assert_allclose(
        dictionary_full, [[2, 0.5, 4], [sqrt2, 0.5 * sqrt2, 0], [1, 1, 0], [-sqrt2, -0.5 * sqrt2, 0]]
    )
    np.testing.assert_allclose(dictionary_diag, [[sqrt2, 0.5 * sqrt2, 0], [-sqrt2, -0.5 * sqrt2, 0]])


@pytest.mark.parametrize('r_diag', [False, True])
def test_cmf_residual_norm(r_diag):
    """The reduced real system preserves the Frobenius norm of H Q H^H - C."""
    mics = ac.MicGeom(pos_total=np.array([[0, 0.1, 0], [0, 0, 0.2], [0, 0, 0]]))
    grid = ac.RectGrid(x_min=0, x_max=1, y_min=0, y_max=0, z=0.5, increment=1)
    steer = ac.SteeringVector(grid=grid, mics=mics)
    samples = np.array([[1 + 1j, 2], [0.5j, 1 - 1j], [2 - 1j, -0.5]])
    csm = samples @ samples.conj().T
    freq_data = ac.PowerSpectraImport(csm=csm[np.newaxis, :, :], frequencies=np.array([1000.0]))
    beamformer = ac.BeamformerCMF(freq_data=freq_data, steer=steer, r_diag=r_diag)

    h = steer.transfer(1000.0).T
    x = np.array([0.4, 1.7])
    residual = h @ np.diag(x) @ h.conj().T - csm
    if r_diag:
        np.fill_diagonal(residual, 0)

    dictionary = beamformer._build_dictionary(1000.0)
    data = beamformer._vectorize_csm(csm).ravel()
    np.testing.assert_allclose(np.linalg.norm(residual, 'fro'), np.linalg.norm(dictionary @ x - data))
