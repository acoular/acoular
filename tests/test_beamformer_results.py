# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
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
BeamformerFunctional, BeamformerDamasPlus, BeamformerGIB, BeamformerGridlessOrth,\
SteeringVector, Environment, BeamformerSODIX

# copy CMF classes as workaround so that reference data name is unique later
class BeamformerCMFLassoLarsBIC (BeamformerCMF):
    pass

class BeamformerCMFNNLS (BeamformerCMF):
    pass

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

# values from example 1
t1 = MaskedTimeSamples(name=datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999
invalid = [1,7] # list of invalid channels (unwanted microphones etc.)
t1.invalid_channels = invalid 
t1.calib = Calib(from_file=calibfile)
m = MicGeom(from_file=micgeofile)
m.invalid_channels = invalid
g = RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68,
             increment=0.05 )
env=Environment(c=346.04)
st = SteeringVector(grid=g, mics=m, env=env)
f = PowerSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=128, #FFT-parameters
               cached = False )  

# produces a tuple of beamformer objects to test
# because we need new objects for each test we have to call this more than once
def fbeamformers():
    bb = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
    be = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)

    #frequency beamformers to test
    bbase = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
    bc = BeamformerCapon(freq_data=f, steer=st, cached=False)
    beig = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)
    bm = BeamformerMusic(freq_data=f, steer=st, n=6, cached = False)
    bd = BeamformerDamas(beamformer=bb, n_iter=10, cached = False)
    bdp = BeamformerDamasPlus(beamformer=bb, n_iter=100, cached = False)
    bo = BeamformerOrth(freq_data=f, steer=st, r_diag=True, eva_list=list(range(38,54)), cached = False)
    bs = BeamformerCleansc(freq_data=f, steer=st, r_diag=True, cached = False)
    bcmflassobic = BeamformerCMFLassoLarsBIC(freq_data=f, steer=st, method='LassoLarsBIC', cached = False)
    bcmfnnls = BeamformerCMFNNLS(freq_data=f, steer=st, method='NNLS', cached = False)
    bl = BeamformerClean(beamformer=bb, n_iter=10, cached = False)
    bf = BeamformerFunctional(freq_data=f, steer=st, r_diag=False, gamma=3, cached = False)
    bgib = BeamformerGIB(freq_data=f, steer=st, method= 'LassoLars', n=2, cached = False)
    bgo = BeamformerGridlessOrth(freq_data=f, steer=st, r_diag=False, n=1,  shgo={'n':16}, cached = False)
    bsodix = BeamformerSODIX(freq_data=f, steer=st,max_iter=10, cached = False)
    return (bbase, bc, beig, bm, bl, bo, bs, bd, bcmflassobic, bcmfnnls, bf, bdp, bgib, bgo,bsodix)

class acoular_beamformer_test(unittest.TestCase):

    def test_beamformer_freq_results(self):
        # we expect the results to computed
        acoular.config.global_caching = 'none'
        for b in fbeamformers():
            with self.subTest(b.__class__.__name__+" global_caching = none"):
                name = join('reference_data',f'{b.__class__.__name__}.npy')
                # stack all frequency band results together
                actual_data = np.array([b.synthetic(cf,1) for cf in cfreqs],dtype=np.float32)
                if WRITE_NEW_REFERENCE_DATA:
                    np.save(name,actual_data)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=5e-5, atol=5e-8)
        # we expect the results to be computed and written to cache
        acoular.config.global_caching = 'individual'
        for b in fbeamformers():
            b.cached = True
            with self.subTest(b.__class__.__name__+" global_caching = individual"):
                name = join('reference_data',f'{b.__class__.__name__}.npy')
                actual_data = np.array([b.synthetic(cf,1) for cf in cfreqs],dtype=np.float32)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=5e-5, atol=5e-8)
        # we expect the results to be read from cache
        acoular.config.global_caching = 'all'
        for b in fbeamformers():
            b.cached = True
            with self.subTest(b.__class__.__name__+" global_caching = all"):
                name = join('reference_data',f'{b.__class__.__name__}.npy')
                actual_data = np.array([b.synthetic(cf,1) for cf in cfreqs],dtype=np.float32)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=5e-5, atol=5e-8)
        # we expect the cached results to be overwritten
        acoular.config.global_caching = 'overwrite'
        for b0,b1 in zip(fbeamformers(),fbeamformers()):
            b0.cached = True
            b1.cached = True
            with self.subTest(b0.__class__.__name__+" global_caching = overwrite"):
                if hasattr(b0,'beamformer'): # BeamformerClean, BeamformerDamas, BeamformerDamasplus 
                                            # do not pass because the .beamformer result is not take from cache
                    continue                         # nor recalculated   
                b0.result[:] = 0
                self.assertFalse(np.any(b0.result))
                name = join('reference_data',f'{b1.__class__.__name__}.npy')
                actual_data = np.array([b1.synthetic(cf,1) for cf in cfreqs],dtype=np.float32)
                ref_data = np.load(name)
                np.testing.assert_allclose(actual_data, ref_data, rtol=5e-5, atol=5e-8)

    def test_beamformer_caching(self):
        # within each subcase, we need new beamformer objects because result is not updated when
        # global_caching or cached changes
        with self.subTest("global_caching = 'none'"):
            acoular.config.global_caching = 'none'
            b0 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            b1 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            self.assertNotEqual(id(b0.result),id(b1.result))
        with self.subTest("global_caching = 'individual'"):
            acoular.config.global_caching = 'individual'
            b0 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            b1 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            self.assertEqual(id(b0.result),id(b1.result))
            b1 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
            self.assertNotEqual(id(b0.result),id(b1.result))
        with self.subTest("global_caching = 'all'"):
            acoular.config.global_caching = 'all'
            b0 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            b1 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
            self.assertEqual(id(b0.result),id(b1.result))
        with self.subTest("global_caching = 'readonly'"):
            acoular.config.global_caching = 'readonly'
            b0 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            b1 = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = True)
            self.assertEqual(id(b0.result),id(b1.result))

class Test_PowerSpectra(unittest.TestCase):

    def test_csm(self):
        """ test that csm result has not changed over different releases"""
        name = join('reference_data',f'{f.__class__.__name__}_csm.npy')
        # test only two frequencies
        actual_data = np.array(f.csm[(16,32),:,:],dtype=np.complex64)
        if WRITE_NEW_REFERENCE_DATA:
            np.save(name,actual_data)
        ref_data = np.load(name)
        np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)

    def test_ev(self):
        """ test that eve and eva result has not changed over different releases"""
        name = join('reference_data',f'{f.__class__.__name__}_ev.npy')
        # test only two frequencies
        actual_data = np.array((f.eve*f.eva[:,:,np.newaxis])[(16,32),:,:],dtype=np.complex64)
        if WRITE_NEW_REFERENCE_DATA:
            np.save(name,actual_data)
        ref_data = np.load(name)
        np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)


class TestSteerFormulation(unittest.TestCase):

    def test_all_steer_formulation(self):
        """ tests all variants of beamformerFreq subroutines
        """
        st = SteeringVector(grid=g, mics=m, env=env)
        b0 = BeamformerBase(freq_data=f, steer=st, cached = False)
        b1 = BeamformerEig(freq_data=f, steer=st, n=54, cached = False)
        for ki,kind in enumerate(('classic', 'inverse','true level', 'true location')):
            st.steer_type = kind
            for b in (b0,b1):
                for dr in (True,False):
                    b.r_diag = dr
                    with self.subTest(f"{b.__class__.__name__} r_diag:{dr} steer:{kind}"):
                        name = join('reference_data',f"{b.__class__.__name__}{dr}{ki+1}.npy")
                        actual_data = np.array([b.synthetic(cf,1) for cf in cfreqs],dtype=np.float32)
                        if WRITE_NEW_REFERENCE_DATA:
                            np.save(name,actual_data)
                        ref_data = np.load(name)
                        np.testing.assert_allclose(actual_data, ref_data, rtol=1e-5, atol=1e-8)


if __name__ == '__main__':
    unittest.main() #exit=False




