# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Test cases for all convolve classes."""

import numpy as np
from acoular import TimeConvolve, tools
from pytest_cases import parametrize


@parametrize('extend_sample', [True, False], ids=['extend_sample-True', 'extend_sample-False'])
def test_time_convolve(time_data_source, extend_sample):
    """Compare results of timeconvolve with numpy convolve.

    Parameters
    ----------
    time_data_source : instance of acoular.sources.TimeSamples
        TimeSamples instance to be tested (see time_data_source fixture in conftest.py)
    """
    sig = tools.return_result(time_data_source)
    nc = time_data_source.num_channels
    kernel = np.random.rand(20 * nc).reshape(20, nc)
    conv = TimeConvolve(kernel=kernel, source=time_data_source, extend_sample=extend_sample)
    res = tools.return_result(conv)
    for i in range(time_data_source.num_channels):
        ref = np.convolve(np.squeeze(conv.kernel[:, i]), np.squeeze(sig[:, i]))
        np.testing.assert_allclose(np.squeeze(res[:, i]), ref[: res.shape[0]], rtol=1e-5, atol=1e-8)
