# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Generates a test data set for three sources.
 
The simulation generates the sound pressure at 64 microphones that are
arrangend in the 'array64' geometry which is part of the package. The sound
pressure signals are sampled at 51200 Hz for a duration of 1 second.

Source location (relative to array center) and levels:

====== =============== ======
Source Location        Level 
====== =============== ======
1      (-0.1,-0.1,0.3) 1 Pa
2      (0.15,0,0.3)    0.7 Pa 
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======
"""

from os import path
from acoular import __file__ as bpath, MicGeom, WNoiseGenerator, PointSource, Mixer, WriteH5

sfreq = 51200
duration = 1
nsamples = duration*sfreq
micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
h5savefile = 'three_sources.h5'

m = MicGeom(from_file=micgeofile)
n1 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=1)
n2 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.7)
n3 = WNoiseGenerator(sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.5)
p1 = PointSource(signal=n1, mics=m,  loc=(-0.1,-0.1,0.3))
p2 = PointSource(signal=n2, mics=m,  loc=(0.15,0,0.3))
p3 = PointSource(signal=n3, mics=m,  loc=(0,0.1,0.3))
p = Mixer(source=p1, sources=[p2,p3])
wh5 = WriteH5(source=p, name=h5savefile)
wh5.save()
