import acoular as ac
import numpy as np
import pytest


def create_source(numsamples, sample_freq=64):
    data = ac.WNoiseGenerator(
        sample_freq=sample_freq,
        numsamples=numsamples,
    ).signal()[:, np.newaxis]

    return ac.TimeSamples(
        data=data,
        sample_freq=sample_freq,
    )


class TestFFT:
    @pytest.mark.parametrize('num', [8, 16])
    def test_equal_in_out(self, num):
        source = create_source(numsamples=num)
        data = next(source.result(num))
        data_comp = next(ac.IRFFT(source=ac.RFFT(source=source)).result(num))
        np.testing.assert_allclose(data, data_comp)
