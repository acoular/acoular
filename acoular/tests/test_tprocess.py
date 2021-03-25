import unittest
import numpy as np
from acoular import (
    config,
    TimeConvolve,
    WNoiseGenerator,
    PointSource,
    MicGeom,
    tools
)

config.global_caching = "none"


class TprocessTest(unittest.TestCase):
    """
    A simple test case that verifies that the results of tprocess are not changing across different versions of code.
    """

    def test_timeconvolve(self):
        """compare results of timeconvolve with numpy convolve"""
        # Parameters
        NSAMPLES = 25
        N1 = WNoiseGenerator(sample_freq=1000, numsamples=NSAMPLES, seed=1)
        MGEOM = MicGeom(mpos_tot=[[1], [1], [1]])
        P1 = PointSource(signal=N1, mics=MGEOM)
        KERNEL = np.random.rand(20)
        CONV = TimeConvolve(kernel=KERNEL, source=P1)

        SIG = tools.return_result(P1, num=NSAMPLES)
        RES = tools.return_result(CONV, num=100)
        
        for i in range(P1.numchannels):
            REF = np.convolve(np.squeeze(KERNEL), np.squeeze(SIG[:,i]))
            np.testing.assert_allclose(np.squeeze(RES[:,i]), REF, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
