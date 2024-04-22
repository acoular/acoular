import unittest
from os.path import join
import numpy as np
from acoular import __file__ as bpath, config, WNoiseGenerator, PointSource, MicGeom

config.global_caching = "none"

# if this flag is set to True, new data will be simulated and
WRITE_NEW_REFERENCE_DATA = False
# new source results are generated for comparison during testing. Should always be False. Only set to
# true, if it is necessary to recalculate the data, due to wanted changes of the sources.

# Parameters
SFREQ = 1000
SEED = 1
NSAMPLES = 100
N1 = WNoiseGenerator(sample_freq=SFREQ, numsamples=NSAMPLES, seed=SEED)
MGEOM = MicGeom(mpos_tot=[[1],[1],[1]])


def get_source_result(Source, num=32):
    """
    returns the result for a given source

    Parameters
    ----------
    source : cls
        source class that is tested.
    num : int, optional
        number of samples to return. The default is 32.

    Returns
    -------
    array
        first block returned by the source result() function.

    """
    if Source == PointSource:
        source = Source(signal=N1, mics=MGEOM)

    return next(source.result(num)).astype(np.float32)


class SourcesTest(unittest.TestCase):
    """
    A simple test case that verifies that the results of sources are not changing across different versions of code.
    """

    sources = [PointSource]

    def test_sources_result(self):
        """compare results of sources against previous
        results from .npy file"""
        for source in self.sources:
            with self.subTest(source.__name__):
                name = join("reference_data", f"{source.__name__}.npy")
                actual_data = get_source_result(source)
                if WRITE_NEW_REFERENCE_DATA:
                    np.save(name, actual_data)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
