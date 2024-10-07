import unittest

import acoular as ac
import numpy as np
import pytest

rng = np.random.RandomState(2)
# test array
# np.abs(np.imag(ps.csm)).sum() + np.real(ps.csm).sum()
csm_sum = 0.5613882842200368

mg = ac.MicGeom(mpos_tot=rng.normal(0, 1, 3 * 2).reshape(3, 2))
sig = ac.WNoiseGenerator(seed=1, numsamples=1010, sample_freq=1000)
p = ac.PointSource(signal=sig, loc=(0, 0, 0), mics=mg)
ps = ac.PowerSpectra(source=p, block_size=128, window='Hanning', cached=False)
fft = ac.FFTSpectra(source=p, window='Hanning', block_size=128)


def create_source(numsamples, sample_freq=64):
    data = ac.WNoiseGenerator(
        sample_freq=sample_freq,
        numsamples=numsamples,
    ).signal()[:, np.newaxis]

    return ac.TimeSamples(
        data=data,
        sample_freq=sample_freq,
    )


class TestPowerSpectra(unittest.TestCase):
    def test_calc_csm(self):
        """test that csm result has not changed over different releases."""
        test_csm_sum = np.abs(np.imag(ps.csm)).sum() + np.real(ps.csm).sum()
        self.assertAlmostEqual(test_csm_sum, csm_sum)


class TestRFFT:
    def test_reflect_source_changes(self):
        source = create_source(numsamples=8)
        fft = ac.RFFT(source=source)
        digest = fft.digest
        freqs = fft.freqs
        # change source sample_freq
        source.sample_freq *= 2
        assert fft.digest != digest
        assert not np.array_equal(fft.freqs, freqs)

    def test_digest_changes(self):
        source = create_source(numsamples=8)
        fft = ac.RFFT(source=source)
        digest = fft.digest
        fft.scaling = 'energy'
        assert fft.digest != digest
        digest = fft.digest
        fft.precision = 'complex64'
        assert fft.digest != digest
        digest = fft.digest
        fft.block_size = 16
        assert fft.digest != digest
        digest = fft.digest
        fft.window = 'Hanning'
        assert fft.digest != digest
        digest = fft.digest
        fft.overlap = '50%'
        assert fft.digest != digest
        digest = fft.digest

    @pytest.mark.parametrize('num', [8, 16])
    def test_equal_fft_ifft(self, num):
        source = create_source(numsamples=num)
        data = next(source.result(num))
        data_comp = next(ac.IRFFT(source=ac.RFFT(block_size=num, source=source)).result(num))
        np.testing.assert_allclose(data, data_comp)

    @pytest.mark.parametrize('window', ['Rectangular', 'Hanning', 'Hamming', 'Blackman', 'Bartlett'])
    def test_equal_power(self, window):
        block_size = 128
        num = 2
        source = create_source(numsamples=block_size * 2)
        fft = ac.RFFT(block_size=block_size, source=source, window=window)
        ap = ac.AutoPowerSpectra(source=fft)
        cp = ac.CrossPowerSpectra(source=fft)
        ps = ac.PowerSpectra(source=source, cached=False, block_size=block_size, window=window)
        ap_data = next(ap.result(num))
        cp_data = next(cp.result(num))
        np.testing.assert_allclose(ap_data, cp_data)
        np.testing.assert_allclose(ap_data.mean(axis=0).reshape(-1), ps.csm[:, :, :].reshape(-1))

    @pytest.mark.parametrize('num', [1, 4, 7, 8, 9, 16])
    def test_irfft_buffer(self, num):
        block_size = 8
        numsamples = 64
        source = create_source(numsamples=numsamples)
        fft = ac.IRFFT(source=ac.RFFT(block_size=block_size, source=source))
        data = ac.tools.return_result(source, num=num)
        data_comp = ac.tools.return_result(fft, num=num)
        np.testing.assert_allclose(data, data_comp)

    @pytest.mark.parametrize('precision', ['64', '32'])
    def test_precision(self, precision):
        """
        test if the desired precision is reflected in the output
        """
        precision_mapping = {
            ('64', 'real'): 'float64',
            ('32', 'real'): 'float32',
            ('64', 'complex'): 'complex128',
            ('32', 'complex'): 'complex64',
        }
        numsamples = 8
        source = create_source(numsamples=numsamples)
        fft = ac.RFFT(source=source, block_size=numsamples, precision=precision_mapping[(precision, 'complex')])
        data = next(fft.result(1))
        assert data.dtype == precision_mapping[(precision, 'complex')]
        ap = ac.CrossPowerSpectra(source=fft, precision=precision_mapping[(precision, 'complex')])
        data = next(ap.result(1))
        assert data.dtype == precision_mapping[(precision, 'complex')]
        ap = ac.AutoPowerSpectra(source=fft, precision=precision_mapping[(precision, 'real')])
        data = next(ap.result(1))
        assert data.dtype == precision_mapping[(precision, 'real')]
        irfft = ac.IRFFT(source=fft, precision=precision_mapping[(precision, 'real')])
        data = next(irfft.result(1))
        assert data.dtype == precision_mapping[(precision, 'real')]


if __name__ == '__main__':
    unittest.main()
