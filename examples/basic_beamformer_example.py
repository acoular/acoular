# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""Generates an test data set for three sources.
 
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
import acoular
from pylab import figure, plot, axis, imshow, colorbar, show

micgeofile = path.join(path.split(acoular.__file__)[0],'xml','array_64.xml')
datafile = 'three_sources.h5'

mg = acoular.MicGeom( from_file=micgeofile )
ts = acoular.TimeSamples( name='three_sources.h5' )
ps = acoular.PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
rg = acoular.RectGrid( x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=0.3, \
increment=0.01 )
bb = acoular.BeamformerBase( freq_data=ps, grid=rg, mpos=mg )
pm = bb.synthetic( 8000, 3 )
Lm = acoular.L_p( pm )
imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), \
interpolation='bicubic')
colorbar()
figure(2)
plot(mg.mpos[0],mg.mpos[1],'o')
axis('equal')
show()
