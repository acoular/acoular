#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:58:04 2019

@author: newuser
"""
#standard testing suite from python
import unittest

#acoular imports
import acoular

from acoular import L_p, Calib, MicGeom, PowerSpectra, Environment, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional, BeamformerDamasPlus, BeamformerGIB, SteeringVector

from numpy import zeros, empty
from os import path
import tables

#load numerical values from Examples
h5file_num = tables.open_file('reference_data/Example1_numerical_values_testsum.h5', 'r')

mpos_num = h5file_num.get_node('/mpos_values').read()
grid_pos_num = h5file_num.get_node('/grid_pos_values').read()
transfer_num = h5file_num.get_node('/transfer_values').read()
csm_num = h5file_num.get_node('/csm_values').read()
eve_num = h5file_num.get_node('/eva_values').read()
eva_num = h5file_num.get_node('/eve_values').read()

d={}
for b in ('bb', 'bc', 'be', 'bm', 'bl', 'bo', 'bs', 'bd', 'bcmf', 'bf', 'bdp', 'bgib'):
    d[b+'num'] = h5file_num.get_node('/'+str(b)+'_values').read()


#load exampledata
datafile = '../../examples/example_data.h5'
calibfile = '../../examples/example_calib.xml'
micgeofile = path.join( path.split(acoular.__file__)[0],'xml','array_56.xml')

#calc all values from example 1
cfreq = 4000
t1 = MaskedTimeSamples(name= datafile)
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
               cached = False )  #cached = False


bb = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
bc = BeamformerCapon(freq_data=f, steer=st, cached=False)
be = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)
bm = BeamformerMusic(freq_data=f, steer=st, n=6, cached = False)
bd = BeamformerDamas(beamformer=bb, n_iter=100, cached = False)
bdp = BeamformerDamasPlus(beamformer=bb, n_iter=100, cached = False)
bo = BeamformerOrth(beamformer=be, eva_list=list(range(38,54)), cached = False)
bs = BeamformerCleansc(freq_data=f, steer=st, r_diag=True, cached = False)
bcmf = BeamformerCMF(freq_data=f, steer=st, method='LassoLarsBIC', cached = False)
bl = BeamformerClean(beamformer=bb, n_iter=100, cached = False)
bf = BeamformerFunctional(freq_data=f, steer=st, r_diag=False, gamma=4, cached = False)
bgib = BeamformerGIB(freq_data=f, steer=st, method= 'LassoLars', n=10, cached = False)

class acoular_test(unittest.TestCase):  
    
    #test if microfon positions are correct
    def test_mic_positions(self):
        self.assertAlmostEqual(m.mpos.sum()/mpos_num.sum(),1,3) 
    
    #test if grid points are correct
    def test_grid_positions(self):
        self.assertAlmostEqual(g.gpos.sum()/grid_pos_num.sum(),1,3) 
        
    #test steering vector calculation
    def test_steering(self):
        ind=0
        checktrans=0
        for freq in f.fftfreq()[1:-1]:
            checktrans=checktrans+st.transfer(freq)
            ind+=1
        self.assertAlmostEqual(checktrans.sum()/transfer_num.sum(),1,3)             
        
    #test if csm values are correct
    @unittest.skip
    def test_csm_calculation(self):
        self.assertAlmostEqual(f.csm[:].sum()/csm_num.sum(),1,3)    
        
    #test eve/eva
    @unittest.skip    
    def test_eigenvalue_calculation(self):
        self.assertAlmostEqual(f.eva[:].sum()/eve_num.sum(),1,3)
        print(f.eve[:].sum(),eva_num)      
        self.assertAlmostEqual(f.eve[:].sum()/eva_num.sum(),1,3) 
     
    #test beamformer results
    @unittest.skip  
    def test_beamformer_calculation(self):
        for beam,bfname in zip((bb, bc, be, bm, bl, bo, bs, bd, bcmf, bf, bdp, bgib),('bb', 'bc', 'be', 'bm', 'bl', 'bo', 'bs', 'bd', 'bcmf', 'bf', 'bdp', 'bgib')):   
            with self.subTest(bfname):
                self.assertAlmostEqual(beam.synthetic(cfreq,1).sum()/d[bfname+'num'].sum(),1,3)      
            
if "__main__" == __name__:
    unittest.main() #exit=False


                            
        