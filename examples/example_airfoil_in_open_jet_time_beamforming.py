# -*- coding: utf-8 -*-
"""
Example "Airfoil in open jet -- Time Beamforming".
acoular example_airfoil_in_open_jet_beamforming.py without frequency beamformers
and only time beamformers

Demonstrates different features of Acoular.

Uses measured data in file example_data.h5,
calibration in file example_calib.xml,
microphone geometry in array_56.xml (part of Acoular).

Copyright (c)  Acoular Development Team.
All rights reserved.
"""

from os import path
import acoular
from pylab import figure, imshow, colorbar, show, subplot,title, tight_layout
from numpy import zeros


micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_56.xml')
datafile = 'example_data.h5'
calibfile = 'example_calib.xml'
cfreq = 4000

mg = acoular.MicGeom( from_file=micgeofile )
ts = acoular.MaskedTimeSamples( name=datafile )

ts.start = 0
ts.stop = 16000
invalid = [1,7]
ts.invalid_channels = invalid
mg.invalid_channels = invalid
rg = acoular.RectGrid( x_min=-0.6, x_max=0.0, y_min=-0.3, y_max=0.3, z=0.68, \
increment=0.05 )
env = acoular.Environment(c = 346.04)
st = acoular.SteeringVector(grid=rg, mics=mg, env=env)

#===============================================================================
# delay and sum beamformer in time domain
# processing chain: beamforming, filtering, power, average
#===============================================================================
bt = acoular.BeamformerTime(source=ts, steer=st)
ft = acoular.FiltFiltOctave(source=bt, band=cfreq)
pt = acoular.TimePower(source=ft)
avgt = acoular.TimeAverage(source=pt, naverage = 1024)
cacht = acoular.TimeCache( source = avgt) # cache to prevent recalculation

#===============================================================================
# delay and sum beamformer in time domain with autocorrelation removal
# processing chain: zero-phase filtering, beamforming+power, average
#===============================================================================
fi = acoular.FiltFiltOctave(source=ts, band=cfreq)
bts = acoular.BeamformerTimeSq(source=fi, steer=st, r_diag=True)
avgts = acoular.TimeAverage(source=bts, naverage = 1024)
cachts = acoular.TimeCache( source = avgts) # cache to prevent recalculation

#===============================================================================
# clean deconvolution in time domain
# processing chain: zero-phase filtering, clean in time domain, power, average
#===============================================================================
fct = acoular.FiltFiltOctave(source=ts, band=cfreq)
bct = acoular.BeamformerCleant(source=fct, steer=st, n_iter=20,damp=.7)
ptct = acoular.TimePower(source=bct)
avgct = acoular.TimeAverage(source=ptct, naverage = 1024)
cachct = acoular.TimeCache( source = avgct) # cache to prevent recalculation

#===============================================================================
# clean deconvolution in time domain
# processing chain: zero-phase filtering, clean in time domain with 
# autocorrelation removal, average
#===============================================================================
fcts = acoular.FiltFiltOctave(source=ts, band=cfreq)
bcts = acoular.BeamformerCleantSq(source=fcts, steer=st, n_iter=20,damp=.7,r_diag=True)
avgcts = acoular.TimeAverage(source=bcts, naverage = 1024)
cachcts = acoular.TimeCache( source = avgcts) # cache to prevent recalculation

#===============================================================================
# plot result maps for different beamformers in time domain
#===============================================================================
i2 = 1 # no of figure
i1 = 1 #no of subplot
for b in (cacht, cachts, cachct, cachcts):
    # first, plot time-dependent result (block-wise)
    figure(i2,(7,7))
    i2 += 1
    res = zeros(rg.size) # init accumulator for average
    i3 = 1 # no of subplot
    for r in b.result(1):  #one single block
        subplot(4,4,i3)
        i3 += 1
        res += r[0] # average accum.
        map = r[0].reshape(rg.shape)
        mx = acoular.L_p(map.max())
        imshow(acoular.L_p(map.T), vmax=mx, vmin=mx-15, origin='lower',
               interpolation='nearest', extent=rg.extend())
        title('%i' % ((i3-1)*1024))
    res /= i3-1 # average
    tight_layout()

    # second, plot overall result (average over all blocks)
    figure(10)
    subplot(4,4,i1)
    i1 += 1
    map = res.reshape(rg.shape)
    mx = acoular.L_p(map.max())
    imshow(acoular.L_p(map.T), vmax=mx, vmin=mx-15, origin='lower',
           interpolation='nearest', extent=rg.extend())
    colorbar()
    title(('BeamformerTime','BeamformerTimeSq','BeamformerCleant',
           'BeamformerCleantSq')[i2-2])
tight_layout()
# only display result on screen if this script is run directly
if __name__ == '__main__': show()
