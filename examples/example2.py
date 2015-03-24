# -*- coding: utf-8 -*-
"""
Example 2 for acoular library

demonstrates use of acoular for a point source moving on a circle trajectory

uses synthesized data

Copyright (c) 2006-2015 The Acoular developers.
All rights reserved.
"""

import acoular
print acoular.__file__

from os import path
import sys
from numpy import empty, clip, sqrt, arange, log10, sort, array, pi, zeros, \
hypot, cos, sin, linspace, hstack, cross, dot, newaxis
from numpy.linalg import norm
from acoular import td_dir, L_p, TimeSamples, Calib, MicGeom, PowerSpectra, \
RectGrid, BeamformerBase, BeamformerEig, BeamformerOrth, BeamformerCleansc, \
MaskedTimeSamples, FiltFiltOctave, Trajectory, BeamformerTimeSq, TimeAverage, \
BeamformerTimeSqTraj, \
TimeCache, FiltOctave, BeamformerTime, TimePower, IntegratorSectorTime, \
PointSource, MovingPointSource, SineGenerator, WNoiseGenerator, Mixer, WriteWAV

from pylab import subplot, imshow, show, colorbar, plot, transpose, figure, \
psd, axis, xlim, ylim, title, suptitle

#===============================================================================
# some important definitions
#===============================================================================

freq = 6144.0*3/128.0 # frequency of interest (114 Hz)
sfreq = 6144.0/2 # sampling frequency (3072 Hz)
c0 = 343.0 # speed of sound
r = 3.0 # array radius
R = 2.5 # radius of source trajectory
Z = 4 # distance of source trajectory from 
rps = 15.0/60. # revolutions per second
U = 3.0 # total number of revolutions

#===============================================================================
# construct the trajectory for the source
#===============================================================================

tr = Trajectory()
tr1 = Trajectory()
tmax = U/rps
delta_t = 1./rps/16.0 # 16 steps per revolution
for t in arange(0, tmax*1.001, delta_t):
    i = t* rps * 2 * pi #angle
    # define points for trajectory spline
    tr.points[t] = (R*cos(i), R*sin(i), Z) # anti-clockwise rotation
    tr1.points[t] = (R*cos(i), R*sin(i), Z) # anti-clockwise rotation

#===============================================================================
# define circular microphone array
#===============================================================================

m = MicGeom()
# set 28 microphone positions
m.mpos_tot = array([(r*sin(2*pi*i+pi/4), r*cos(2*pi*i+pi/4), 0) \
    for i in linspace(0.0, 1.0, 28, False)]).T

#===============================================================================
# define the different source signals
#===============================================================================

nsamples = long(sfreq*tmax)
n1 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples)
s1 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq)
s2 = SineGenerator(sample_freq=sfreq, numsamples=nsamples, freq=freq, \
    phase=pi)

#===============================================================================
# define the moving source and one fixed source
#===============================================================================

p0 = MovingPointSource(signal=s1, mpos=m, trajectory=tr1)
#t = p0 # use only moving source
p1 = PointSource(signal=n1, mpos=m,  loc=(0,R,Z))
t = Mixer(source = p0, sources = [p1,]) # mix both signals
#t = p1 # use only fix source

# uncomment to save the signal to a wave file
#ww = WriteWAV(source = t)
#ww.channels = [0,14]
#ww.save()

#===============================================================================
# fixed focus frequency domain beamforming
#===============================================================================

f = PowerSpectra(time_data=t, window='Hanning', overlap='50%', block_size=128, \
    ind_low=1,ind_high=30) # CSM calculation 
g = RectGrid(x_min=-3.0, x_max=+3.0, y_min=-3.0, y_max=+3.0, z=Z, increment=0.3)
b = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=c0)
map1 = b.synthetic(freq,3)

#===============================================================================
# fixed focus time domain beamforming
#===============================================================================

fi = FiltFiltOctave(source=t, band=freq, fraction='Third octave')
bt = BeamformerTimeSq(source=fi, grid=g, mpos=m, r_diag=True, c=c0)
avgt = TimeAverage(source=bt, naverage=int(sfreq*tmax/16)) # 16 single images
cacht = TimeCache(source=avgt) # cache to prevent recalculation
map2 = zeros(g.shape) # accumulator for average
# plot single frames
figure(1)
i = 0
for res in cacht.result(1):
    res0 = res[0].reshape(g.shape)
    map2 += res0 # average
    i += 1  
    subplot(4,4,i)
    mx = L_p(res0.max())
    imshow(L_p(transpose(res0)), vmax=mx, vmin=mx-10, interpolation='nearest',\
        extent=g.extend(), origin='lower')
    colorbar()
map2 /= i
suptitle('fixed focus')

#===============================================================================
# moving focus time domain beamforming
#===============================================================================

# new grid needed, the trajectory starts at origin and is oriented towards +x
# thus, with the circular movement assumed, the center of rotation is at (0,2.5)
g1 = RectGrid(x_min=-3.0, x_max=+3.0, y_min=-1.0, y_max=+5.0, z=0, \
    increment=0.3)# grid point of origin is at trajectory (thus z=0)
# beamforming with trajectory (rvec axis perpendicular to trajectory)
bts = BeamformerTimeSqTraj(source=fi, grid=g1, mpos=m, trajectory=tr, \
    rvec = array((0,0,1.0)))
avgts = TimeAverage(source=bts, naverage=int(sfreq*tmax/16)) # 16 single images
cachts = TimeCache(source=avgts) # cache to prevent recalculation
map3 = zeros(g1.shape) # accumulator for average
# plot single frames
figure(2)
i = 0
for res in cachts.result(1):
    res0 = res[0].reshape(g1.shape)
    map3 += res0 # average
    i += 1  
    subplot(4,4,i)
    mx = L_p(res0.max())
    imshow(L_p(transpose(res0)), vmax=mx, vmin=mx-10, interpolation='nearest',\
        extent=g1.extend(), origin='lower')
    colorbar()
map3 /= i
suptitle('moving focus')

#===============================================================================
# compare all three results
#===============================================================================

figure(3)
subplot(1,3,1)
mx = L_p(map1.max())
imshow(L_p(transpose(map1)), vmax=mx, vmin=mx-10, interpolation='nearest',\
    extent=g.extend(), origin='lower')
colorbar()
title('frequency domain\n fixed focus')
subplot(1,3,2)
mx = L_p(map2.max())
imshow(L_p(transpose(map2)), vmax=mx, vmin=mx-10, interpolation='nearest',\
    extent=g.extend(), origin='lower')
colorbar()
title('time domain\n fixed focus')
subplot(1,3,3)
mx = L_p(map3.max())
imshow(L_p(transpose(map3)), vmax=mx, vmin=mx-10, interpolation='nearest',\
    extent=g.extend(), origin='lower')
colorbar()
title('time domain\n moving focus')


show()


