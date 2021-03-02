#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:30:41 2019

@author: newuser
"""

#standart testing suite from python
import unittest

#acoular imports
import acoular
acoular.config.global_caching="none" # to make sure that nothing is cached

from acoular import L_p, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional, BeamformerDamasPlus, BeamformerGIB, SteeringVector,Environment

from os import path
import tables


#load exampledata
datafile = '../../examples/example_data.h5'
calibfile = '../../examples/example_calib.xml'
micgeofile = path.join( path.split(acoular.__file__)[0],'xml','array_56.xml')

#frequencies to test
cfreqs = 1000,8000

#load numerical values from datafile
h5file_num = tables.open_file('reference_data/Beamforer_numerical_values.h5', 'r')

res_num = h5file_num.get_node('/timebf_values').read()
resq_num = h5file_num.get_node('/timebfsq_values').read()

bfdata={}
for b in ('bb', 'bc', 'be', 'bm', 'bl', 'bo', 'bs', 'bd', 'bcmf', 'bf', 'bdp', 'bgib'):
    for cfreq_num in cfreqs:
        bfdata[b+'_num_'+str(cfreq_num)] = h5file_num.get_node('/'+b+'_'+str(cfreq_num)+'_values').read()


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

#frequency beamformer test
bb = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
bc = BeamformerCapon(freq_data=f, steer=st, cached=False)
be = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)
bm = BeamformerMusic(freq_data=f, steer=st, n=6, cached = False)
bd = BeamformerDamas(beamformer=bb, n_iter=10, cached = False)
bdp = BeamformerDamasPlus(beamformer=bb, n_iter=100, cached = False)
bo = BeamformerOrth(beamformer=be, eva_list=list(range(38,54)), cached = False)
bs = BeamformerCleansc(freq_data=f, steer=st, r_diag=True, cached = False)
bcmf = BeamformerCMF(freq_data=f, steer=st, method='LassoLarsBIC', cached = False)
bl = BeamformerClean(beamformer=bb, n_iter=10, cached = False)
bf = BeamformerFunctional(freq_data=f, steer=st, r_diag=False, gamma=3, cached = False)
bgib = BeamformerGIB(freq_data=f, steer=st, method= 'LassoLars', n=2, cached = False)
bbase = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)
beig = BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54, cached = False)

#timebeamformer test
bt = BeamformerTime(source=t1, steer=st)
ft = FiltFiltOctave(source=bt, band=1000)
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt, naverage = 1024)
res= next(avgt.result(1))
#squared
fi = FiltFiltOctave(source=t1, band=4000)
bts = BeamformerTimeSq(source=fi, steer=st, r_diag=True)
avgts = TimeAverage(source=bts, naverage = 1024)
resq= next(avgts.result(1))


class acoular_beamformer_test(unittest.TestCase):  
    
    #@unittest.skip('test time bf first')
    def test_beamformer_freq_results(self):
        #test all fbeamformers for 1000 and 8000 hertz
        for cfreq in cfreqs:     
            for beam,bfname in zip((bbase, bc, beig, bm, bl, bo, bs, bd, bcmf, bf, bdp, bgib),('bb', 'bc', 'be', 'bm', 'bl', 'bo', 'bs', 'bd', 'bcmf', 'bf', 'bdp', 'bgib')):   
                for i in range(len(beam.synthetic(cfreq,1).flatten())):
                    self.assertAlmostEqual((beam.synthetic(cfreq,1).flatten()[i]+1)/(bfdata[bfname+'_num_'+str(cfreq)][i]+1),1,2)##,1,3)#.flatten()   
   
    def test_beamformer_time_results(self):
        #test beamformertime
        for i in range(len(res.flatten())):
            self.assertAlmostEqual(res.flatten()[i]/res_num.flatten()[i],1,2)
        #test beamformer time squared    
        for i in range(len(resq.flatten())):
            self.assertAlmostEqual(resq.flatten()[i]/resq_num.flatten()[i],1,2) 
        
if "__main__" == __name__:
    unittest.main() #exit=False




