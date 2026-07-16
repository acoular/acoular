"""Pytest fixtures shared across the ported notch-filter tests."""

from tests.unittests.notch.notch_helpers import generate_tonal_signal

import pytest


@pytest.fixture
def standard_params():
    """Standard test parameters for signal generation."""
    return {
        'sample_freq': 16000,
        'duration': 1.0,
        'f0': 100,
        'snr_db': 20,
    }


@pytest.fixture
def standard_tonal_signal(standard_params):
    """Single-channel tonal signal with harmonics [1, 2, 3]."""
    return generate_tonal_signal(
        f0=standard_params['f0'],
        harmonics=[1, 2, 3],
        duration=standard_params['duration'],
        sample_freq=standard_params['sample_freq'],
        num_channels=1,
        snr_db=standard_params['snr_db'],
    )


@pytest.fixture
def multi_channel_signal(standard_params):
    """Four-channel tonal signal with harmonics [1, 2, 3]."""
    return generate_tonal_signal(
        f0=standard_params['f0'],
        harmonics=[1, 2, 3],
        duration=standard_params['duration'],
        sample_freq=standard_params['sample_freq'],
        num_channels=4,
        snr_db=standard_params['snr_db'],
    )
