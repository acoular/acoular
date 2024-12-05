# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements snapshot testing of spectra classes."""

import acoular as ac
import numpy as np
import pytest


@pytest.fixture(
    params=[
        {'cached': False},  # default (no cache)
        {'cached': False, 'precision': 'complex64', 'overlap': '50%', 'block_size': 256, 'window': 'Hanning'},
    ],
    ids=['default', 'custom'],
    scope='module',
)
def freq_data(request, regression_source_case):
    """
    Fixture to create a :class:`acoular.freq_data.PowerSpectra` instance for testing.

    The fixture is parametrized to test the object with default and custom configuration.

    Parameters
    ----------
    request : pytest request
        Request fixture (to inject parameters)
    regression_source_case : fixture
        regression_source_case fixture (see conftest.py)
    """
    return ac.PowerSpectra(source=regression_source_case.calib, **request.param)


@pytest.mark.parametrize('ind', [16, 32])
def test_csm(freq_data, snapshot, ind):
    """Performs snapshot testing for the cross spectral matrix.

    To overwrite the snapshot, run:
    ```bash
    pytest -v --regtest-reset tests/regression/test_spectra.py::test_csm
    ```

    Parameters
    ----------
    freq_data : instance of acoular.spectra.PowerSpectra
        PowerSpectra instance to be tested (see freq_data fixture)
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    ind : tuple
        frequency indices to test
    """
    snapshot.check(
        freq_data.csm[ind, :, :].astype(np.complex64), rtol=5e-5, atol=1e-8
    )  # uses numpy.testing.assert_allclose


@pytest.mark.parametrize('ind', [16, 32])
def test_eva(freq_data, snapshot, ind):
    """Performs snapshot testing for the eigenvalues.

    To overwrite the snapshot, run:
    ```bash
    pytest -v --regtest-reset tests/regression/test_spectra.py::test_eva
    ```

    Parameters
    ----------
    freq_data : instance of acoular.spectra.PowerSpectra
        PowerSpectra instance to be tested (see freq_data fixture)
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    ind : tuple
        frequency indices to test
    """
    snapshot.check(freq_data.eva[ind, -5:].astype(np.float32), rtol=5e-5, atol=1e-8)


@pytest.mark.parametrize('ind', [16, 32])
def test_eve(freq_data, snapshot, ind):
    """Performs snapshot testing for the eigenvectors.

    To overwrite the snapshot, run:
    ```bash
    pytest -v --regtest-reset tests/regression/test_spectra.py::test_eve
    ```

    Parameters
    ----------
    freq_data : instance of acoular.spectra.PowerSpectra
        PowerSpectra instance to be tested (see freq_data fixture)
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    ind : tuple
        frequency indices to test
    """
    snapshot.check(np.abs(freq_data.eve[ind, :, -5:].astype(np.complex64)), rtol=5e-5, atol=1e-8)
