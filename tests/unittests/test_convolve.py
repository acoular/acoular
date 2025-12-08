# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Test cases for all convolve classes."""

import numpy as np
import pytest
from acoular import TimeConvolve, TimeSamples, tools
from pytest_cases import parametrize


@parametrize('extend_signal', [True, False], ids=['extend_signal-True', 'extend_signal-False'])
def test_time_convolve(time_data_source, extend_signal):
    """Compare results of timeconvolve with numpy convolve.

    Parameters
    ----------
    time_data_source : instance of acoular.sources.TimeSamples
        TimeSamples instance to be tested (see time_data_source fixture in conftest.py)
    """
    sig = tools.return_result(time_data_source)
    nc = time_data_source.num_channels
    kernel = np.random.rand(20 * nc).reshape(20, nc)
    conv = TimeConvolve(kernel=kernel, source=time_data_source, extend_signal=extend_signal)
    res = tools.return_result(conv)
    for i in range(time_data_source.num_channels):
        ref = np.convolve(np.squeeze(conv.kernel[:, i]), np.squeeze(sig[:, i]))
        np.testing.assert_allclose(np.squeeze(res[:, i]), ref[: res.shape[0]], rtol=1e-5, atol=1e-8)


@parametrize(
    'randn_params,error_msg',
    [
        ((10, 2, 2), 'Only one or two dimensional kernels accepted'),
        ((10, 3), 'Number of kernels must be either'),
    ],
    ids=['3d-kernel', 'mismatched-channels'],
)
def test_time_convolve_kernel_validation_errors(randn_params, error_msg):
    """
    Test TimeConvolve kernel validation errors.

    Covers the _validate_kernel() method:
    - Raises ValueError for kernels with more than 2 dimensions
    - Raises ValueError when kernel channel count doesn't match source
    """
    source = TimeSamples(sample_freq=51200, data=np.random.randn(100, 2))
    kernel = np.random.randn(*randn_params)
    conv = TimeConvolve(kernel=kernel, source=source)

    with pytest.raises(ValueError, match=error_msg):
        _ = tools.return_result(conv)
