# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
from os.path import join
import numpy as np
#acoular imports
import acoular
acoular.config.global_caching = 'none' # to make sure that nothing is cached

from acoular import Calib, MicGeom, PowerSpectra, RectGrid, BeamformerBase,\
SteeringVector, Environment, MaskedTimeSamples

#load exampledata
datafile = join('..','example_data.h5')
micgeofile = join( '..','..','acoular','xml','array_56.xml')

# values from example 1
t1 = MaskedTimeSamples(name=datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999
m = MicGeom(from_file=micgeofile)
g = RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68,
             increment=0.01 )
env = Environment(c=346.04)
st = SteeringVector(grid=g, mics=m, env=env, steer_type='classic')
f = PowerSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=4096, #FFT-parameters
               cached = False )  
b = BeamformerBase(freq_data=f, steer=st, r_diag=True, cached = False)

csm = np.array(b.freq_data.csm[20], dtype='complex128')
param_steer_type, steer_vector = b._beamformer_params()
sln = b.sig_loss_norm()
svt = steer_vector(8000.)
eva = b.freq_data.eva[20]
eve = b.freq_data.eve[20]

def test_beamformerFreq():
    acoular.fastFuncs.beamformerFreq(param_steer_type, False, sln, svt, csm)

def test_beamformerFreqE():
    acoular.fastFuncs.beamformerFreq(param_steer_type, False, sln, svt, (eva[30:],eve[:,30:]))

def test_csm():
    t1.stop=-1
    f.calc_csm()
