#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""

Loads the example data set, sets diffrent Sectors for intergration.
Shows acoular Sector und Sound Pressure level Integration functionality.

"""

from os import path
from numpy import array,arange
import acoular
from pylab import figure, plot, imshow, colorbar, show,xlim, ylim,legend,cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle

#fft block size
block = 128

#example data 
micgeofile = path.join( path.split(acoular.__file__)[0],'xml','array_56.xml')
mg = acoular.MicGeom( from_file=micgeofile )
ts = acoular.TimeSamples( name='example_data.h5' )
ps = acoular.PowerSpectra( time_data=ts, block_size=128, window='Hanning' )
rg = acoular.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68,
             increment=0.02)
st = acoular.SteeringVector( grid = rg, mics=mg )
f = acoular.PowerSpectra(time_data=ts,block_size=block)
bf  = acoular.BeamformerBase(freq_data = f,steer= st)


#Integrate function can deal with multiple methods for integration:

#1. a circle containing of three values: x-center, y-center and radius
circle = array([-0.3,-0.1, 0.05])

#2. a rectangle containing of 4 values: lower corner(x1, y1) and upper corner(x2, y2).
rect  =  array([-0.5,   -0.15, -0.4 , 0.15])

#3. a polygon containing of vector tuples: x1,y1,x2,y2,...,xi,yi
poly = array([ -0.25, -0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.25, -0.3, -0.2])

#4 alternative: define those sectors as Classes
circle_sector = acoular.CircSector(x=-0.3,y= -0.1, r= 0.05)

rect_sector = acoular.RectSector(x_min=-0.5,x_max=-0.4, y_min=-0.15, y_max= 0.15)

#list of points containing x1,y1,x2,y2,...,xi,yi
poly_sector =  acoular.PolySector(edges=[ -0.25, -0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.25, -0.3, -0.2])

#multisector allows to sum over multiple different sectors
multi_sector = acoular.MultiSector( sectors = [circle_sector,rect_sector,poly_sector])

#calculate the discrete frequencies for the integration
fftfreqs = arange(block/2+1)*(51200/block)


# two integration variants exist (with same outcome):
# 1. use acoulars integrate function    

#integrate SPL values from beamforming results using the shapes
levels_circ = acoular.integrate(bf.result, rg, circle)
levels_rect = acoular.integrate(bf.result, rg, rect)
levels_poly = acoular.integrate(bf.result, rg, poly)

#integrate SPL values from beamforming results using sector classes
levels_circ_sector = acoular.integrate(bf.result, rg, circle_sector)
levels_rect_sector = acoular.integrate(bf.result, rg, rect_sector)
levels_poly_sector = acoular.integrate(bf.result, rg, poly_sector)
levels_multi_sector = acoular.integrate(bf.result, rg, multi_sector)

# 2. use beamformers integrate function (does not require explicit assignment 
# of grid object)

#integrate SPL values from beamforming results using the shapes
levels_circ = bf.integrate(circle)
levels_rect = bf.integrate(rect)
levels_poly = bf.integrate(poly)

#integrate SPL values from beamforming results using sector classes
levels_circ_sector = bf.integrate(circle_sector)
levels_rect_sector = bf.integrate(rect_sector)
levels_poly_sector = bf.integrate(poly_sector)
levels_multi_sector = bf.integrate(multi_sector)


#plot map and sectors
figure()
map = bf.synthetic(2000,1)
mx = acoular.L_p(map.max())
imshow(acoular.L_p(map.T), origin='lower', vmin=mx-15,interpolation='nearest', extent=rg.extend(),cmap=cm.hot_r)
colorbar()
circle1 = plt.Circle((-0.3,0.1), 0.05, color='k', fill=False)
plt.gcf().gca().add_artist(circle1)
polygon = Polygon(poly.reshape(-1,2), color='k', fill=False)
plt.gcf().gca().add_artist(polygon)
rect = Rectangle((-0.5,-0.15),0.1,0.3,linewidth=1,edgecolor='k',facecolor='none')
plt.gcf().gca().add_artist(rect)


#plot from shapes
figure()
plot(fftfreqs,acoular.L_p(levels_circ))
plot(fftfreqs,acoular.L_p(levels_rect))
plot(fftfreqs,acoular.L_p(levels_poly))
xlim([2000,20000])
ylim([10,80])
legend(['Circle','Rectangle','Polygon'])

#plot from sector classes
figure()
plot(fftfreqs,acoular.L_p(levels_circ_sector))
plot(fftfreqs,acoular.L_p(levels_rect_sector))
plot(fftfreqs,acoular.L_p(levels_poly_sector))
plot(fftfreqs,acoular.L_p(levels_multi_sector))
xlim([2000,20000])
ylim([10,80])
legend(['Circle Sector','Rectangle Sector','Polygon Sector','Multisector'])

show()