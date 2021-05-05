from acoular import *
import numpy as np
from acoular import __file__ as bpath
from os import path
import unittest

# load some array geometry
micgeofile = path.join(path.split(bpath)[0], 'xml', 'array_64.xml')
MICGEOM = MicGeom(from_file=micgeofile) 

NUM_SAMPLES = 7
BLOCK_SIZE = 5

class TimeCacheTest(unittest.TestCase):
    GLOBAL_CACHING_COFIGS= ['individual','all','none','readonly','overwrite']
    def test_valid_cache_result(self):
        """
        manually create an incomplete cash file and then read it 
        """
        for conf in self.GLOBAL_CASHING_COFIGS:
            config.global_caching= conf
            with self.subTest(conf):
                sig = WNoiseGenerator(numsamples=NUM_SAMPLES)
                ps = PointSource(signal=sig, mics=MICGEOM)
                tc = TimeCache(source=ps)
                for i,(block_c, block_nc) in enumerate(zip(tc.result(BLOCK_SIZE), ps.result(BLOCK_SIZE))):
                    np.testing.assert_array_almost_equal(block_c,block_nc)
                    if i == 0: break
      
                for (block_c, block_nc) in zip(tc.result(BLOCK_SIZE), ps.result(BLOCK_SIZE)):
                    np.testing.assert_array_almost_equal(block_c,block_nc)


if __name__=="__main__":
    unittest.main()
