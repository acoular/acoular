# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
"""
beamfpy.py: classes for calculations in the time domain

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming
 
(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

# imports from other packages
from numpy import array, arange, sort, r_
from scipy.interpolate import splprep, splev
from traits.api import HasPrivateTraits, Float, \
Property, cached_property, property_depends_on, Dict, Tuple
from traitsui.api import View, Item
from traitsui.menu import OKCancelButtons

# beamfpy imports
from .internal import digest


class Trajectory( HasPrivateTraits ):
    """
    describes the trajectory from sampled points
    does spline interpolation of positions between samples
    """
    # dictionary; keys: time, values: sampled (x, y, z) positions along the 
    # trajectory
    points = Dict(key_trait = Float, value_trait = Tuple(Float, Float, Float), 
        desc = "sampled positions along the trajectory")
    
    # t_min, t_max tuple
    interval = Property()
    
    # spline data, internal use
    tck = Property()
    
    # internal identifier
    digest = Property( 
        depends_on = ['points[]'], 
        )

    traits_view = View(
        [Item('points', style='custom')
        ], 
        title='Grid center trajectory', 
        buttons = OKCancelButtons
        )

    @cached_property
    def _get_digest( self ):
        return digest(self)
        
    @property_depends_on('points[]')
    def _get_interval( self ):
        return sort(self.points.keys())[r_[0, -1]]

    @property_depends_on('points[]')
    def _get_tck( self ):
        t = sort(self.points.keys())
        xp = array([self.points[i] for i in t]).T
        k = min(3, len(self.points)-1)
        tcku = splprep(xp, u=t, s=0, k=k)
        return tcku[0]
    
    def location(self, t, der=0):
        """ returns (x, y, z) for t, x, y and z have the same shape as t """
        return splev(t, self.tck, der)
    
    def traj(self, t_start, t_end=None, delta_t=None, der=0):
        """
        python generator, yields locations along the trajectory
        x.traj(0.1)  every 0.1s within self.interval
        x.traj(2.5, 4.5, 0.1)  every 0.1s between 2.5s and 4.5s
        x.traj(0.1, der=1)  1st derivative every 0.1s within self.interval
        """
        if not delta_t:
            delta_t = t_start
            t_start, t_end = self.interval
        if not t_end:
            t_end = self.interval[1]
        for t in arange(t_start, t_end, delta_t):
            yield self.location(t, der)
        
