# -*- coding: utf-8 -*-
"""
Example "3D beamforming" for Acoular library.

Demonstrates a 3D beamforming setup with point sources.

Simulates data on 64 channel array,
subsequent beamforming with CLEAN-SC on 3D grid.

Copyright (c) 2019 Acoular Development Team.
All rights reserved.
"""
from os import path

# imports from acoular

from acoular import __file__ as bpath, L_p, MicGeom, PowerSpectra,\
RectGrid3D, BeamformerBase, BeamformerCleansc, \
SteeringVector, WNoiseGenerator, PointSource, SourceMixer

# other imports
from numpy import mgrid, arange, array, arccos, pi, cos, sin, sum
import mpl_toolkits.mplot3d
from pylab import figure, show, scatter, subplot, imshow, title, colorbar,\
xlabel, ylabel

#===============================================================================
# First, we define the microphone geometry.
#===============================================================================

micgeofile = path.join(path.split(bpath)[0],'xml','array_64.xml')
# generate test data, in real life this would come from an array measurement
m = MicGeom( from_file=micgeofile )

#===============================================================================
# Now, the sources (signals and types/positions) are defined.
#===============================================================================
sfreq = 51200 
duration = 1
nsamples = duration*sfreq

n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
n2 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=2, rms=0.5 )
n3 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=3, rms=0.25 )
p1 = PointSource( signal=n1, mics=m,  loc=(-0.1,-0.1,0.3) )
p2 = PointSource( signal=n2, mics=m,  loc=(0.15,0,0.17) )
p3 = PointSource( signal=n3, mics=m,  loc=(0,0.1,0.25) )
pa = SourceMixer( sources=[p1,p2,p3])


#===============================================================================
# the 3D grid (very coarse to enable fast computation for this example)
#===============================================================================

g = RectGrid3D(x_min=-0.2, x_max=0.2, 
               y_min=-0.2, y_max=0.2, 
               z_min=0.1, z_max=0.36, 
               increment=0.02)

#===============================================================================
# The following provides the cross spectral matrix and defines the CLEAN-SC beamformer.
# To be really fast, we restrict ourselves to only 10 frequencies
# in the range 2000 - 6000 Hz (5*400 - 15*400)
#===============================================================================

f = PowerSpectra(time_data=pa, 
                 window='Hanning', 
                 overlap='50%', 
                 block_size=128, 
                 ind_low=5, ind_high=16)
st = SteeringVector(grid=g, mics=m, steer_type='true location') 
b = BeamformerCleansc(freq_data=f, steer=st)

#===============================================================================
# Calculate the result for 4 kHz octave band
#===============================================================================

map = b.synthetic(4000,1)

#===============================================================================
# Display views of setup and result.
# For each view, the values along the repsective axis are summed.
# Note that, while Acoular uses a left-oriented coordinate system, 
# for display purposes, the z-axis is inverted, plotting the data in
# a right-oriented coordinate system.
#===============================================================================

fig=figure(1,(8,8))

# plot the results

subplot(221)
map_z = sum(map,2)
mx = L_p(map_z.max())
imshow(L_p(map_z.T), vmax=mx, vmin=mx-20, origin='lower', interpolation='nearest', 
       extent=(g.x_min, g.x_max, g.y_min, g.y_max))
xlabel('x')
ylabel('y')
title('Top view (xy)' )

subplot(223)
map_y = sum(map,1)
imshow(L_p(map_y.T), vmax=mx, vmin=mx-20, origin='upper', interpolation='nearest', 
       extent=(g.x_min, g.x_max, -g.z_max, -g.z_min))
xlabel('x')
ylabel('z')
title('Side view (xz)' )

subplot(222)
map_x = sum(map,0)
imshow(L_p(map_x), vmax=mx, vmin=mx-20, origin='lower', interpolation='nearest', 
       extent=(-g.z_min, -g.z_max,g.y_min, g.y_max))
xlabel('z')
ylabel('y')
title('Side view (zy)' )
colorbar()


# plot the setup

ax0 = fig.add_subplot((224), projection='3d')
ax0.scatter(m.mpos[0],m.mpos[1],-m.mpos[2])
source_locs=array([p1.loc,p2.loc,p3.loc]).T
ax0.scatter(source_locs[0],source_locs[1],-source_locs[2])
ax0.set_xlabel('x')
ax0.set_ylabel('y')
ax0.set_zlabel('z')
ax0.set_title('Setup (mic and source positions)')

# only display result on screen if this script is run directly
if __name__ == '__main__': show()


