# -*- coding: utf-8 -*-
"""
Example 6 for acoular library

demonstrates different steering vectors in acoular,
and CSM diagonal removal 
with same setup as in example 1

uses measured data in file example_data.h5
calibration in file example_calib.xml
microphone geometry in array_56.xml (part of acoular)


Copyright (c) 2006-2017 The Acoular developers.
All rights reserved.
"""
from __future__ import print_function

# imports from acoular
import acoular
from acoular import L_p, Calib, MicGeom, EigSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, BeamformerDamas, BeamformerFunctional, PointSpreadFunction, BeamformerClean, BeamformerDamasPlus

# other imports
from os import path
from pylab import figure, subplot, imshow, show, colorbar, title, suptitle

# files
datafile = 'example_data.h5'
calibfile = 'example_calib.xml'
micgeofile = path.join( path.split(acoular.__file__)[0],'xml','array_56.xml')

#octave band of interest
cfreq = 4000

#===============================================================================
# first, we define the time samples using the MaskedTimeSamples class
# alternatively we could use the TimeSamples class that provides no masking
# of channels and samples
#===============================================================================
t1 = MaskedTimeSamples(name=datafile)
t1.start = 0 # first sample, default
t1.stop = 16000 # last valid sample = 15999
invalid = [1,7] # list of invalid channels (unwanted microphones etc.)
t1.invalid_channels = invalid 

#===============================================================================
# calibration is usually needed and can be set directly at the TimeSamples 
# object (preferred) or for frequency domain processing at the PowerSpectra 
# object (for backwards compatibility)
#===============================================================================
t1.calib = Calib(from_file=calibfile)

#===============================================================================
# the microphone geometry must have the same number of valid channels as the
# TimeSamples object has
#===============================================================================
m = MicGeom(from_file=micgeofile)
m.invalid_channels = invalid

#===============================================================================
# the grid for the beamforming map; a RectGrid3D class is also available
# (the example grid is very coarse)
#===============================================================================
g = RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68,
             increment=0.05)

#===============================================================================
# for frequency domain methods, this provides the cross spectral matrix and its
# eigenvalues and eigenvectors, if only the matrix is needed then class 
# PowerSpectra can be used instead
#===============================================================================
f = EigSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=128, #FFT-parameters
               ind_low=7, ind_high=15, precision='complex128') #to save computational effort, only
               # frequencies with index 1-30 are used
csm = f.csm[:]
eva = f.eva[:]

f32 = EigSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=128, #FFT-parameters
               ind_low=7, ind_high=15, precision='complex64') #to save computational effort, only
csm32 = f32.csm[:]
eva32 = f32.eva[:]

psf32 = PointSpreadFunction(grid=g, mpos=m, c=346.04, precision='float32')
psf32Res = psf32.psf[:]

psf64 = PointSpreadFunction(grid=g, mpos=m, c=346.04, precision='float64')
psf64Res = psf64.psf[:]

bb32 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, precision='float32')
bb32Res = bb32.synthetic(cfreq,1)

bb64 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, precision='float64')
bb64Res = bb64.synthetic(cfreq,1)

bf = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, gamma = 60, precision='float32')
bfRes = bf.synthetic(cfreq,1)

# 32 Bit PSF precision
bd3232 = BeamformerDamas(beamformer=bb32, n_iter=100, psf_precision='float32')
bd3232Res = bd3232.synthetic(cfreq,1)
bc3232 = BeamformerClean(beamformer=bb32, psf_precision='float32')
bc3232Res = bc3232.synthetic(cfreq,1)
bdp3232 = BeamformerDamasPlus(beamformer=bb32, n_iter=100, psf_precision='float32')
bdp3232Res  = bdp3232.synthetic(cfreq,1)

#64 Bit
bd3264 = BeamformerDamas(beamformer=bb32, n_iter=100, psf_precision='float64')
bd3264Res = bd3264.synthetic(cfreq,1)
bc3264 = BeamformerClean(beamformer=bb32, psf_precision='float64')
bc3264Res = bc3264.synthetic(cfreq,1)
bdp3264 = BeamformerDamasPlus(beamformer=bb32, n_iter=100, psf_precision='float64')
bdp3264Res  = bdp3264.synthetic(cfreq,1)

# 32 Bit PSF precision
bd6432 = BeamformerDamas(beamformer=bb64, n_iter=100, psf_precision='float32')
bd6432Res = bd6432.synthetic(cfreq,1)
bc6432 = BeamformerClean(beamformer=bb64, psf_precision='float32')
bc6432Res = bc6432.synthetic(cfreq,1)
bdp6432 = BeamformerDamasPlus(beamformer=bb64, n_iter=100, psf_precision='float32')
bdp6432Res  = bdp6432.synthetic(cfreq,1)

#64 Bit
bd6464 = BeamformerDamas(beamformer=bb64, n_iter=100, psf_precision='float64')
bd6464Res = bd6464.synthetic(cfreq,1)
bc6464 = BeamformerClean(beamformer=bb64, psf_precision='float64')
bc6464Res = bc6464.synthetic(cfreq,1)
bdp6464 = BeamformerDamasPlus(beamformer=bb64, n_iter=100, psf_precision='float64')
bdp6464Res  = bdp6464.synthetic(cfreq,1)



#===============================================================================
# plot result maps for different beamformers in frequency domain
#===============================================================================
i1 = 1
for b in (bb32, bd3232, bc3232, bdp3232, bd3264, bc3264, bdp3264,
          bb64, bd6432, bc6432, bdp6432, bd6464, bc6464, bdp6464):
    subplot(2, 7, i1)
    i1 += 1
    res = b.synthetic(cfreq,1)
    mx = L_p(res.max())
    imshow(L_p(res.T), vmax=mx, vmin=mx-15, 
           interpolation='nearest', extent=g.extend())
    print(b.steer)
    colorbar()
    title(b.__class__.__name__ + b.precision,fontsize='small')


show()
