from acoular import *
import numpy as np
from acoular import __file__ as bpath
from os import path
import unittest

# load some array geometry
micgeofile = path.join(path.split(bpath)[0], 'xml', 'array_64.xml')
micgeom = MicGeom(from_file=micgeofile)

num_samples = 7
block_size = 5

class TimeCacheTest(unittest.TestCase):
    global_caching_configs= ['individual','all','none','readonly','overwrite']
    def test_valid_cache_result(self):
        """
        manually create an incomplete cash file and then read it 
        """
        for conf in self.global_caching_configs:
            config.global_caching= conf
            with self.subTest(conf):
                sig = WNoiseGenerator(numsamples=num_samples)
                ps = PointSource(signal=sig, mics=micgeom)
                tc = TimeCache(source=ps)
                for i,(block_c, block_nc) in enumerate(zip(tc.result(block_size), ps.result(block_size))):
                    np.testing.assert_array_almost_equal(block_c,block_nc)
                    if i == 0: break
      
                for (block_c, block_nc) in zip(tc.result(block_size), ps.result(block_size)):
                    np.testing.assert_array_almost_equal(block_c,block_nc)


if __name__=="__main__":
    unittest.main()
