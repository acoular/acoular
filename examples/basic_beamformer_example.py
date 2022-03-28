# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""

Loads the three sources test data set, analyzes them and generates a
map of the three sources. 

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
st = acoular.SteeringVector( grid = rg, mics=mg )
bb = acoular.BeamformerBase( freq_data=ps, steer=st )
pm = bb.synthetic( 8000, 3 )
Lm = acoular.L_p( pm )
imshow( Lm.T, origin='lower', vmin=Lm.max()-10, extent=rg.extend(), \
interpolation='bicubic')
colorbar()
figure(2)
plot(mg.mpos[0],mg.mpos[1],'o')
axis('equal')
show()
