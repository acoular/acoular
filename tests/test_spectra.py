# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

import unittest

import acoular as ac
import numpy as np


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
        csm_sum = 0.8815407250342017
        source = create_source(numsamples=128)
        ps = ac.PowerSpectra(source=source, block_size=128, window='Hanning', cached=False)
        test_csm_sum = np.abs(np.imag(ps.csm)).sum() + np.real(ps.csm).sum()
        self.assertAlmostEqual(test_csm_sum, csm_sum)


if __name__ == '__main__':
    unittest.main()
