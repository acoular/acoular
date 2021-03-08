# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2021, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements testing of frequency beamformers.
"""

import unittest

from os.path import join

import numpy as np
#acoular imports
import acoular
acoular.config.global_caching = 'none' # to make sure that nothing is cached

from acoular import Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional, BeamformerDamasPlus, BeamformerGIB, SteeringVector,Environment

# if this flag is set to True
WRITE_NEW_REFERENCE_DATA = False
# new beamformer results are generated for comparison during testing. 
# Should always be False. Only set to True if it is necessary to 
# recalculate the data due to intended changes of the Beamformers.

#load exampledata
datafile = join('..','..','examples','example_data.h5')
calibfile = join('..','..','examples','example_calib.xml')
micgeofile = join( '..','xml','array_56.xml')

#frequencies to test
cfreqs = 1000,8000

#calc all values from example with low resolution
t1 = MaskedTimeSamples(name= datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999
t1.calib = Calib(from_file=calibfile)
m = MicGeom(from_file=micgeofile)
g = RectGrid(x_min=-0.2, x_max=-0.0, y_min=-0.3, y_max=0.2, z=0.68,
             increment=0.1 )
env=Environment(c=346.04)
st = SteeringVector(grid=g, mics=m, env=env)
f = PowerSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=128, #FFT-parameters
               cached = False )  #cached = False

bb = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
be = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)

#frequency beamformers to test
bbase = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
bc = BeamformerCapon(freq_data=f, steer=st, cached=False)
beig = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)
bm = BeamformerMusic(freq_data=f, steer=st, n=6, cached = False)
bd = BeamformerDamas(beamformer=bb, n_iter=10, cached = False)
bdp = BeamformerDamasPlus(beamformer=bb, n_iter=100, cached = False)
bo = BeamformerOrth(beamformer=be, eva_list=list(range(38,54)), cached = False)
bs = BeamformerCleansc(freq_data=f, steer=st, r_diag=True, cached = False)
bcmf = BeamformerCMF(freq_data=f, steer=st, method='LassoLarsBIC', cached = False)
bl = BeamformerClean(beamformer=bb, n_iter=10, cached = False)
bf = BeamformerFunctional(freq_data=f, steer=st, r_diag=False, gamma=3, cached = False)
bgib = BeamformerGIB(freq_data=f, steer=st, method= 'LassoLars', n=2, cached = False)

fbeamformers = (bbase, bc, beig, bm, bl, bo, bs, bd, bcmf, bf, bdp, bgib)

class acoular_beamformer_test(unittest.TestCase):  
    
    def test_beamformer_freq_results(self):
        for b in fbeamformers:
            with self.subTest(b.__class__.__name__):
                name = join('reference_data',f'{b.__class__.__name__}.npy')
                # stack all frequency band results together
                actual_data = np.array([b.synthetic(cf,1) for cf in cfreqs],dtype=np.float32)
                if WRITE_NEW_REFERENCE_DATA:
                    np.save(name,actual_data)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=0)

if __name__ == '__main__':
    unittest.main() #exit=False




