# -*- coding: utf-8 -*-
"""
Example 1 for acoular library

demonstrates different features of acoular,

uses measured data in file example_data.h5
calibration in file example_calib.xml
microphone geometry in array_56.xml (part of acoular)

Copyright (c) 2006-2015 The Acoular developers.
All rights reserved.
"""

# imports from acoular
import acoular
from acoular import L_p, Calib, MicGeom, EigSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, BeamformerTimeSq, TimeAverage, \
TimeCache, BeamformerTime, TimePower, BeamformerCMF, \
BeamformerCapon, BeamformerMusic, BeamformerDamas, BeamformerClean, \
BeamformerFunctional

# other imports
from numpy import zeros
from os import path
from pylab import figure, subplot, imshow, show, colorbar, title

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
               ind_low=7, ind_high=15) #to save computational effort, only
               # frequencies with index 1-30 are used


#===============================================================================
# different beamformers in frequency domain
#===============================================================================
bb = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04)
bc = BeamformerCapon(freq_data=f, grid=g, mpos=m, c=346.04, cached=False)
be = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, n=54)
bm = BeamformerMusic(freq_data=f, grid=g, mpos=m, c=346.04, n=6)
bd = BeamformerDamas(beamformer=bb, n_iter=100)
bo = BeamformerOrth(beamformer=be, eva_list=range(38,54))
bs = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04)
bcmf = BeamformerCMF(freq_data=f, grid=g, mpos=m, c=346.04, \
    method='LassoLarsBIC')
bl = BeamformerClean(beamformer=bb, n_iter=100)
bf = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, \
    gamma=4)

#===============================================================================
# plot result maps for different beamformers in frequency domain
#===============================================================================
figure(1)
i1 = 1 #no of subplot
for b in (bb, bc, be, bm, bl, bo, bs, bd, bcmf, bf):
    subplot(3,4,i1)
    i1 += 1
    map = b.synthetic(cfreq,1)
    mx = L_p(map.max())
    imshow(L_p(map.T), vmax=mx, vmin=mx-15, 
           interpolation='nearest', extent=g.extend())
    colorbar()
    title(b.__class__.__name__)

#===============================================================================
# delay and sum beamformer in time domain
# processing chain: beamforming, filtering, power, average
#===============================================================================
bt = BeamformerTime(source=t1, grid=g, mpos=m, c=346.04)
ft = FiltFiltOctave(source=bt, band=cfreq)
pt = TimePower(source=ft)
avgt = TimeAverage(source=pt, naverage = 1024)
cacht = TimeCache( source = avgt) # cache to prevent recalculation

#===============================================================================
# delay and sum beamformer in time domain with autocorrelation removal
# processing chain: zero-phase filtering, beamforming+power, average
#===============================================================================
fi = FiltFiltOctave(source=t1, band=cfreq)
bts = BeamformerTimeSq(source = fi,grid=g, mpos=m, r_diag=True,c=346.04)
avgts = TimeAverage(source=bts, naverage = 1024)
cachts = TimeCache( source = avgts) # cache to prevent recalculation

#===============================================================================
# plot result maps for different beamformers in time domain
#===============================================================================
i2 = 2 # no of figure
for b in (cacht, cachts):
    # first, plot time-dependent result (block-wise)
    figure(i2)
    i2 += 1
    res = zeros(g.size) # init accumulator for average
    i3 = 1 # no of subplot
    for r in b.result(1):  #one single block
        subplot(4,4,i3)
        i3 += 1
        res += r[0] # average accum.
        map = r[0].reshape(g.shape)
        mx = L_p(map.max())
        imshow(L_p(map.T), vmax=mx, vmin=mx-15, 
               interpolation='nearest', extent=g.extend())
        title('%i' % ((i3-1)*1024))
    res /= i3-1 # average
    # second, plot overall result (average over all blocks)
    figure(1)
    subplot(3,4,i1)
    i1 += 1
    map = res.reshape(g.shape)
    mx = L_p(map.max())
    imshow(L_p(map.T), vmax=mx, vmin=mx-15, 
           interpolation='nearest', extent=g.extend())
    colorbar()
    title(('BeamformerTime','BeamformerTimeSq')[i2-3])
show()
