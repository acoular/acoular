# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements snapshot testing of frequency beamformers."""


import acoular as ac
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases
from test_fbeamform_cases import Beamformer
import matplotlib.pyplot as plt
from pathlib import Path

# create plot directory inside directory if needed
plot_path = Path(__file__).parent / 'plots'
plot_path.mkdir(exist_ok=True)


def get_Lp_ref(freq_data, f, num):
    return np.round(ac.L_p(
        np.mean(
            ac.synthetic(
                freq_data.csm, freq_data.fftfreq(), f, num
                )[0].diagonal()
                ).real), 1)

# TODO: grid is large (13x13) -> wouldn't a single source be enough on a 3x3 grid? (lower storage requirements)
@pytest.mark.parametrize(
    'f, num',
    [
        pytest.param(1000, 1, id='1kHz_oct'),
        #pytest.param(8000, 1, id='8kHz_oct'),
    ],  # TODO: is there a reason why we use an octave band? -> computational cost may be too high with many parameters tested
)
@parametrize_with_cases("beamformer", cases=Beamformer)
def test_beamformer(snapshot, beamformer, f, num):
    """Performs snapshot testing with snapshot fixture from pytest-regtest

    To overwrite the snapshot, run:
    ```bash
    pytest -v --regtest-reset tests/regression/test_fbeamform.py
    ```

    Parameters
    ----------
    snapshot : pytest-regtest snapshot fixture
        Snapshot fixture to compare results
    beamformer : instance of acoular.fbeamform.BeamformerBase
        Beamformer instance to be tested
    f : int
        Frequency to test
    num : int
        Bandwidth to test (1: octave)

    """
    if isinstance(beamformer, ac.BeamformerCMF) and beamformer.method in ['FISTA', 'Split_Bregman']:
        pytest.skip("This is a current issue with Pylops. See: https://github.com/acoular/acoular/issues/203")

    # get result
    ac.config.global_caching = 'none'  # we do not cache results!
    result = beamformer.synthetic(f, num)

    # results need to be plausible before they are saved!
    Lp_ref = get_Lp_ref(beamformer.freq_data, f, num)
    assert ac.L_p(result.sum()) > (Lp_ref - 20) # we should ensure here that the
    assert ac.L_p(result.sum()) < (Lp_ref + 20)

    # compare / save snapshot
    #snapshot.check(result, rtol=5e-5, atol=5e-8) # uses numpy.testing.assert_allclose


@pytest.mark.parametrize(
    'f, num',
    [
        pytest.param(1000, 1, id='1kHz_oct'),
        #pytest.param(8000, 1, id='8kHz_oct'),
    ],  # TODO: is there a reason why we use an octave band? -> computational cost may be too high with many parameters tested
)
@parametrize_with_cases("beamformer", cases=Beamformer)
def test_plot_beamformer(beamformer, f, num, request):
    #ac.config.global_caching = 'none'  # we do not cache results!
    result = beamformer.synthetic(
        f, num
    )
    fname = request.node.name
    plt.figure()
    Lm = ac.L_p(result)
    plt.imshow(Lm.T, origin='lower', extent=beamformer.steer.grid.extend(), 
        vmin=Lm.max()-20, vmax=Lm.max())
    plt.colorbar()
    # create directory if needed
    plt.savefig(plot_path/f'{fname}.png')
    # use pytest node name and frequency for filename
    #plt.savefig(f'{beamformer.__class__.__name__}_{f}Hz_{num}band.png')
