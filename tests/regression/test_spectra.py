# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements snapshot testing of spectra classes."""

import acoular as ac
import pytest


@pytest.fixture(
    params=[
        {},  # default
        {'precision': 'complex64', 'overlap': '50%', 'block_size': 256, 'window': 'Hanning'},
    ],
    ids=['default', 'custom'],
)
def freq_data(request, regression_source_case):
    """
    Fixture to create a beamformer instance for testing.

    Parameters
    ----------
    request : pytest request
        Request fixture (to inject parameters)
    regression_source_case : fixture
        regression_source_case fixture (see conftest.py)
    """
    return ac.PowerSpectra(source=regression_source_case.source, **request.param)


@pytest.mark.parametrize('ind', [(16, 32)])
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
    ac.config.global_caching = 'none'  # we do not cache results!
    snapshot.check(freq_data.csm[ind, :, :], rtol=5e-5, atol=1e-8)  # uses numpy.testing.assert_allclose


@pytest.mark.parametrize('ind', [(16, 32)])
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
    ac.config.global_caching = 'none'
    snapshot.check(freq_data.eva[ind, :], rtol=5e-5, atol=1e-8)


@pytest.mark.parametrize('ind', [(16, 32)])
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
    ac.config.global_caching = 'none'
    snapshot.check(freq_data.eve[ind, :, :], rtol=5e-5, atol=1e-8)