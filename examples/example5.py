# -*- coding: utf-8 -*-
"""
Example 5 for acoular library

demonstrates different beamformers in frequency domain,
- persistence (loading of configured beamformers), see example 4 for 
  the first part (saving)

uses measured data in file example_data.h5
calibration in file example_calib.xml
microphone geometry in array_56.xml (part of acoular)


Copyright (c) 2006-2015 The Acoular developers.
All rights reserved.
"""

# imports from acoular
from acoular import L_p
# other imports
from pylab import figure, subplot, imshow, show, colorbar, title
from cPickle import load

#octave band of interest
cfreq = 4000

#===============================================================================
# load all beamformers in an external file
# important: import load from cPickle !!!
#===============================================================================
fi = open('all_bf.sav','r')
all_bf = load(fi)
fi.close()

#===============================================================================
# plot result maps for different beamformers in frequency domain
#===============================================================================
figure(1)
i1 = 1 #no of subplot
for b in all_bf:
    subplot(3,4,i1)
    i1 += 1
    g = b.grid
    map = b.synthetic(cfreq,1)[:,0,:]
    mx = L_p(map.max())
    imshow(L_p(map.T), vmax=mx, vmin=mx-15, 
           interpolation='nearest', extent=(g.x_min,g.x_max,g.z_min,g.z_max),
           origin='lower')
    colorbar()
    title(b.__class__.__name__+'\n '+b.steer, size=10)
show()

