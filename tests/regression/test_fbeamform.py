# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements snapshot testing of frequency beamformers."""

import acoular as ac
import pytest


@pytest.fixture(
    params=[
        (ac.BeamformerBase, {}),  # default # TODO: check every possible combination?
        (ac.BeamformerBase, {'r_diag': False, 'precision': 'float32'}),
    ],
    ids=['BeamformerBase_r_diag-true_float64', 'BeamformerBase_r_diag-false_float32'],
)
def beamformer(request, source_case):
    """
    Fixture to create a beamformer instance for testing.

    Parameters
    ----------
    request : pytest request
        Request fixture (to inject parameters)
    source_case : fixture
        source_case fixture (see conftest.py)
    """
    beamformer_cls, params = request.param  # TODO: explain request fixture
    return beamformer_cls(
        freq_data=source_case.freq_data, steer=source_case.steer, **params
    )  # TODO: use cached freq_data result (test before)


@pytest.mark.parametrize(
    'f, num',
    [
        (1000, 1),
        (8000, 1),
    ],  # TODO: is there a reason why we use an octave band? -> computational cost may be too high with many parameters tested
)
def test_beamformer(beamformer, snapshot, f, num):
    """Performs snapshot testing with snapshot fixture from pytest-regtest

    To overwrite the snapshot, run:
    ```bash
    pytest -v --regtest-reset tests/regression/test_fbeamform.py
    ```

    Parameters
    ----------
    beamformer : instance of acoular.fbeamform.BeamformerBase
        Beamformer instance to be tested (see beamformer fixture)
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    f : int
        Frequency to test
    num : int
        Bandwidth to test (1: octave)

    """
    ac.config.global_caching = 'none'  # we do not cache results!
    result = beamformer.synthetic(
        f, num
    )  # TODO: grid is large (13x13) -> wouldn't a single source be enough on a 3x3 grid? (lower storage requirements)
    snapshot.check(result, rtol=5e-5, atol=5e-8)  # uses numpy.testing.assert_allclose
