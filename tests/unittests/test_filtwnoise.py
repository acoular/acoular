# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Tests for FiltWNoiseGenerator class."""

from pathlib import Path

import numpy as np
import pytest
from acoular import FiltWNoiseGenerator
from numpy import array
from numpy.random import RandomState

data_path = Path(__file__).parent.parent / 'data'
ma_coeff = np.load(data_path / 'ma_coeff.npy')
ar_coeff = np.load(data_path / 'ar_coeff.npy')


def test_filtwnoise_no_coefficients():
    """Test that white noise and filtered white noise is equal if no coefficients are specified."""
    fwn = FiltWNoiseGenerator(sample_freq=100, num_samples=400, seed=1)
    wn_signal = RandomState(seed=1).standard_normal(400)
    assert wn_signal.sum() == fwn.signal().sum()


@pytest.mark.parametrize(
    ('ar', 'ma'), [(array([]), ma_coeff), (ar_coeff, array([])), (ar_coeff, ma_coeff)], ids=['MA', 'AR', 'ARMA']
)
def test_filtwnoise_signal_length(ar, ma):
    """Test that signal retains correct length after filtering.

    ar: numpy.ndarray
        AR coefficients
    ma: numpy.ndarray
        MA coefficients
    """
    fwn = FiltWNoiseGenerator(sample_freq=100, num_samples=400, seed=1, ar=ar, ma=ma)
    assert fwn.signal().shape[0] == fwn.num_samples
