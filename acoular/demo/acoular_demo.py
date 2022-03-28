#! /usr/bin/env python
# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Demo for Acoular

Generates a test data set for three sources, analyzes them and generates a
map of the three sources.
 
The simulation generates the sound pressure at 64 microphones that are
arrangend in the 'array64' geometry, which is part of the package. The sound
pressure signals are sampled at 51200 Hz for a duration of 1 second.

Source location (relative to array center) and levels:

====== =============== ======
Source Location        Level 
====== =============== ======
1      (-0.1,-0.1,0.3) 1.0 Pa
2      (0.15,0,0.3)    0.7 Pa 
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======
"""

def run():

    from os import path
    from acoular import __file__ as bpath, MicGeom, WNoiseGenerator, PointSource,\
     Mixer, WriteH5, TimeSamples, PowerSpectra, RectGrid, SteeringVector,\
     BeamformerBase, L_p
    from pylab import figure, plot, axis, imshow, colorbar, show
    
    # set up the parameters
    sfreq = 51200 
    duration = 1
    nsamples = duration*sfreq
    micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
    h5savefile = 'three_sources.h5'
    
    # generate test data, in real life this would come from an array measurement
    mg = MicGeom( from_file=micgeofile )
    n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
    n2 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.7 )
    n3 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5 )
    p1 = PointSource( signal=n1, mics=mg,  loc=(-0.1,-0.1,0.3) )
    p2 = PointSource( signal=n2, mics=mg,  loc=(0.15,0,0.3) )
    p3 = PointSource( signal=n3, mics=mg,  loc=(0,0.1,0.3) )
    pa = Mixer( source=p1, sources=[p2,p3] )
    wh5 = WriteH5( source=pa, name=h5savefile )
    wh5.save()
    
    # analyze the data and generate map
    
    ts = TimeSamples( name=h5savefile )
    ps = PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
    
    rg = RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
    increment=0.01 )
    st = SteeringVector(grid=rg, mics=mg)
    
    bb = BeamformerBase( freq_data=ps, steer=st )
    pm = bb.synthetic( 8000, 3 )
    Lm = L_p( pm )
    
    # show map
    imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), \
    interpolation='bicubic')
    colorbar()
    
    # plot microphone geometry
    figure(2)
    plot(mg.mpos[0],mg.mpos[1],'o')
    axis('equal')
    
    show()

if __name__ == "__main__":
    run()
