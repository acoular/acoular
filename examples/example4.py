# -*- coding: utf-8 -*-
"""
Example 4 for acoular library

demonstrates different beamformers in frequency domain,
- setting of steering vector type,
- disabling of result caching,
- persistence (saving of configured beamformers), see example5 for 
  the second part (loading)

uses measured data in file example_data.h5
calibration in file example_calib.xml
microphone geometry in array_56.xml (part of acoular)


Copyright (c) 2006-2015 The Acoular developers.
All rights reserved.
"""

# imports from acoular
import acoular
from acoular import L_p, Calib, MicGeom, EigSpectra, \
RectGrid3D, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, \
BeamformerCapon, BeamformerMusic, BeamformerDamas
# other imports
from os import path
from pylab import figure, subplot, imshow, show, colorbar, title
from cPickle import dump

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
# the grid for the beamforming map; the RectGrid3D class is used to produce a
# grid in a plane perpendicular to the array
# (the example grid is very coarse)
#===============================================================================
g = RectGrid3D(x_min=-0.6, x_max=+0.0, y_min=0.0, y_max=0.0, \
    z_min=0.3, z_max=0.9, increment=0.05)

#===============================================================================
# for frequency domain methods, this provides the cross spectral matrix and its
# eigenvalues and eigenvectors, if only the matrix is needed then class 
# PowerSpectra can be used instead
#===============================================================================
f = EigSpectra(time_data=t1, 
               window='Hanning', overlap='50%', block_size=128, #FFT-parameters
               ind_low=1, ind_high=15) #to save computational effort, only
               # frequencies with index 1-30 are used


#===============================================================================
# different beamformers in frequency domain
#===============================================================================
# first, some simple delay and sum beamformers, 
# but with different steering vector types
#===============================================================================
bb0 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='true level') # this is the default
bb1 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='true location') # gives better results for 3D location at low freqs.
bb2 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='classic') # classical formulation (e.g. Johnson/Dudgeon)
bb3 = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='inverse') # 'inverse' formulation (e.g. Brooks 2001)

#===============================================================================
# second, the same with CleanSC
#===============================================================================
bs0 = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='true level')
bs1 = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='true location')
bs2 = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='classic')
bs3 = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    steer='inverse')


#===============================================================================
# third, caching is disabled here for BeamformerEig to save cache storage
# (the results are not needed anyway, exept one time for BeamformerOrth)
#===============================================================================
be = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, n=-1, \
    cached=False) 
bo = BeamformerOrth(beamformer=be, eva_list=range(38,54))

#===============================================================================
# save all beamformers in an external file
# important: import dump from cPickle !!!
#===============================================================================
all_bf = (bb0, bb1, bb2, bb3, bs0, bs1, bs2, bs3, bo)
fi = open('all_bf.sav','w')
dump(all_bf,fi,-1) # uses newest pickle protocol -1 (default = 0)
fi.close()

#===============================================================================
# plot result maps for different beamformers in frequency domain
#===============================================================================
figure(1)
i1 = 1 #no of subplot
for b in all_bf:
    subplot(3,4,i1)
    i1 += 1
    map = b.synthetic(cfreq,1)[:,0,:]
    mx = L_p(map.max())
    imshow(L_p(map.T), vmax=mx, vmin=mx-15, 
           interpolation='nearest', extent=(g.x_min,g.x_max,g.z_min,g.z_max),
           origin='lower')
    colorbar()
    title(b.__class__.__name__+'\n '+b.steer, size=10)
show()
