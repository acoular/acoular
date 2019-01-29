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

from acoular import L_p, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional, BeamformerDamasPlus, BeamformerGIB, SteeringVector

from numpy import zeros, empty
from os import path

import h5py

datafile = '../../examples/example_data.h5'
calibfile = '../../examples/example_calib.xml'
micgeofile = path.join( path.split(acoular.__file__)[0],'xml','array_56.xml')

#calc all values from example 1
cfreqs = 1000,8000
t1 = MaskedTimeSamples(name= datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999

t1.calib = Calib(from_file=calibfile)
m = MicGeom(from_file=micgeofile)

g = RectGrid(x_min=-0.2, x_max=-0.0, y_min=-0.3, y_max=0.2, z=0.68,
             increment=0.1 )

st = SteeringVector(grid=g, mpos=m, c=346.04 )

f = PowerSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=32, #FFT-parameters
               cached = False )  #cached = False

#frequency beamformer test
bb = BeamformerBase(freq_data=f, steer_obj=st, r_diag=True, cached = False)
bc = BeamformerCapon(freq_data=f, steer_obj=st, cached=False)
be = BeamformerEig(freq_data=f, steer_obj=st, r_diag=True, n=54, cached = False)
bm = BeamformerMusic(freq_data=f, steer_obj=st, n=6, cached = False)
bd = BeamformerDamas(beamformer=bb, n_iter=10, cached = False)
bdp = BeamformerDamasPlus(beamformer=bb, n_iter=100, cached = False)
bo = BeamformerOrth(beamformer=be, eva_list=list(range(38,54)), cached = False)
bs = BeamformerCleansc(freq_data=f, steer_obj=st, r_diag=True, cached = False)
bcmf = BeamformerCMF(freq_data=f, steer_obj=st, method='LassoLarsBIC', cached = False)
bl = BeamformerClean(beamformer=bb, n_iter=10, cached = False)
bf = BeamformerFunctional(freq_data=f, steer_obj=st, r_diag=False, gamma=3, cached = False)
bgib = BeamformerGIB(freq_data=f, steer_obj=st, method= 'LassoLars', n=2, cached = False)

#timebeamformer test
bt = BeamformerTime(source=t1, steer_obj=st)
ft = FiltFiltOctave(source=bt, band=1000)
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt, naverage = 1024)
res= avgt.result(1)


#squared
fi = FiltFiltOctave(source=t1, band=4000)
bts = BeamformerTimeSq(source=fi, steer_obj=st, r_diag=True)
avgts = TimeAverage(source=bts, naverage = 1024)
resq= avgts.result(1)


class acoular_beamformer_test(unittest.TestCase):  
    
    def test_beamformer_results(self):
        for cfreq in cfreqs:     
            for beam,bfname in zip((bb, bc, be, bm, bl, bo, bs, bd, bcmf, bf, bdp, bgib),('bb', 'bc', 'be', 'bm', 'bl', 'bo', 'bs', 'bd', 'bcmf', 'bf', 'bdp', 'bgib')):   
                self.assertAlmostEqual(beam.synthetic(cfreq,1).sum()/d[bfname+'num'].sum(),1,3)   
    







