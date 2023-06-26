
from os import path
import numpy as np
import acoular
import unittest

rng = np.random.RandomState(2)
# test array
#np.abs(np.imag(ps.csm)).sum() + np.real(ps.csm).sum()
csm_sum = 0.5613882842200368
fft_sum = 0.3690552444079589+0.05264645224899272j

mg = acoular.MicGeom( mpos_tot=rng.normal(0,1,3*2).reshape(3,2))
sig = acoular.WNoiseGenerator(seed=1,numsamples=1010,sample_freq=1000)
p = acoular.PointSource(signal=sig,loc=(0,0,0),mics=mg)
ps = acoular.PowerSpectra( source=p, block_size=128, window='Hanning',cached=False )
fft = acoular.FFTSpectra(source=p,window='Hanning',block_size=128)

class Test_PowerSpectra(unittest.TestCase):

    def test_calc_csm(self):
        """ test that csm result has not changed over different releases"""
        test_csm_sum = np.abs(np.imag(ps.csm)).sum() + np.real(ps.csm).sum()
        self.assertAlmostEqual(test_csm_sum,csm_sum)

class Test_FFTSpectra(unittest.TestCase):

    def test_calc_fft(self):
        """ test that fft result has not changed over different releases"""
        test_fft_sum = 0
        for temp in fft.result():
            test_fft_sum += temp.sum()
        self.assertAlmostEqual(test_fft_sum,fft_sum)


if __name__ == '__main__':
    unittest.main()

