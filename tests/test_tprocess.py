import unittest
from os.path import join
import re
import numpy as np
from acoular import (
    config,
    TimeConvolve,
    WNoiseGenerator,
    PointSource,
    MicGeom,
    MaskedTimeSamples,
    tools
)
from acoular.tprocess import *

WRITE_NEW_REFERENCE_DATA = False

config.global_caching = "none"
datafile = join('..','..','examples','example_data.h5')
t1 = MaskedTimeSamples(name=datafile)
t1.start = 0 # first sample, default
t1.stop = 500 # last valid sample = 15999
invalid = list(range(4,64)) # list of invalid channels 
t1.invalid_channels = invalid # use four channels

# these are tested 
test_list = (
    "TimeInOut()",
    "TimePower()",
    "TimeAverage()",
    "TimeCumAverage()",
    "TimeReverse()",
    "Filter()",
    "FiltFiltOctave(band = 100.0, fraction = 'Third octave')",
    "FiltFiltOctave()",
    "FiltOctave(band = 100.0, fraction = 'Third octave')",
    "FiltOctave()",
    "TimeExpAverage(weight = 'F')",
    "TimeExpAverage(weight = 'S')",
    "TimeExpAverage(weight = 'I')",
    "FiltFreqWeight(weight = 'A')",
    "FiltFreqWeight(weight = 'C')",
    "FiltFreqWeight(weight = 'Z')",
    "OctaveFilterBank()"
)

def fname(s):
    """converts string to file name removing unsafe characters"""
    s1 = re.sub(r'[,.()=]', '_', s)
    return re.sub(r'[/\\:*?"<>|\' ]', '', s1)

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

    #@unittest.skip
    def test_tprocess_results(self):
        """compare results with reference results"""
        for s in test_list:
            b = eval(s)
            with self.subTest(s):
                name = join('reference_data',f'{fname(s)}.npy')
                b.source = t1
                # compute with block size 64 and add some extra 
                actual_data = tools.return_result(b, nmax=70, num=64)
                if WRITE_NEW_REFERENCE_DATA:
                    np.save(name,actual_data)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

if __name__ == "__main__":
    unittest.main()
