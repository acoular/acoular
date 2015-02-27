# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements support for two- and threedimensional grids

.. autosummary::
    :toctree: generated/

    Grid
    RectGrid
    RectGrid3D

"""

# imports from other packages
from numpy import mgrid, s_, array, arange
from traits.api import HasPrivateTraits, Float, Property, CArray, \
property_depends_on, cached_property, on_trait_change
from traitsui.api import View

from .internal import digest

class Grid( HasPrivateTraits ):
    """
    Virtual base class for grid geometries.
    
    Defines the common interface for all grid classes and
    provides facilities to query grid properties and related data. This class
    may be used as a base for specialized grid implementaions. It should not
    be used directly as it contains no real functionality.
    """

    #: Overall number of grid points. Readonly, is set automatically when
    #: other grid defining properties are set
    size = Property(
        desc="overall number of grid points")

    #: Shape of grid. Readonly, gives the shape as tuple, useful for cartesian
    #: grids
    shape = Property(
        desc="grid shape as tuple")

    # internal identifier
    digest = Property

    # no view necessary
    traits_view = View()

    def _get_digest( self ):
        return ''

    # 'digest' is a placeholder for other properties in derived classes,
    # necessary to trigger the depends on mechanism
    @property_depends_on('digest')
    def _get_size ( self ):
        return 1

    # 'digest' is a placeholder for other properties in derived classes
    @property_depends_on('digest')
    def _get_shape ( self ):
        return (1, 1)

    def pos ( self ):
        """
        Calculates grid co-ordinates.
        
        Returns
        -------
        array of floats of shape (3, :attr:`size`)
            The grid point x, y, z-coordinates in one array.
        """
        return array([[0.], [0.], [0.]])

# This is not needed in the base class (?)
#    def extend (self) :
#        """
#        returns the x, y extension of the grid,
#        useful for the imshow function from pylab
#        """
#        pos = self.pos()
#        return (min(pos[0,:]), max(pos[0,:]), min(pos[1,:]), max(pos[1,:]))


class RectGrid( Grid ):
    """
    Provides a cartesian 2D grid for the beamforming results.
    
    The grid has square or nearly square cells and is on a plane perpendicular
    to the z-axis. It is defined by lower and upper x- and  y-limits and the 
    z co-ordinate.
    """
    
    #: The lower x-limit that defines the grid, defaults to -1.
    x_min = Float(-1.0,
        desc="minimum  x-value")

    #: The upper x-limit that defines the grid, defaults to 1.
    x_max = Float(1.0,
        desc="maximum  x-value")

    #: The lower y-limit that defines the grid, defaults to -1.
    y_min = Float(-1.0,
        desc="minimum  y-value")

    #: The upper y-limit that defines the grid, defaults to 1.
    y_max = Float(1.0,
        desc="maximum  y-value")

    #: The z co-ordinate that defines the grid, defaults to 1.
    z = Float(1.0,
        desc="position on z-axis")

    #: The cell side length for the grid, defaults to 0.1.
    increment = Float(0.1,
        desc="step size")

    #: Number of grid points along x-axis, readonly.
    nxsteps = Property(
        desc="number of grid points along x-axis")

    #: Number of grid points along y-axis, readonly.
    nysteps = Property(
        desc="number of grid points along y-axis")

    # internal identifier
    digest = Property(
        depends_on = ['x_min', 'x_max', 'y_min', 'y_max', 'z', 'increment']
        )

    traits_view = View(
            [
                ['x_min', 'y_min', '|'],
                ['x_max', 'y_max', 'z', 'increment', 'size~{Grid size}', '|'],
                '-[Map extension]'
            ]
        )

    @property_depends_on('nxsteps, nysteps')
    def _get_size ( self ):
        return self.nxsteps*self.nysteps

    @property_depends_on('nxsteps, nysteps')
    def _get_shape ( self ):
        return (self.nxsteps, self.nysteps)

    @property_depends_on('x_min, x_max, increment')
    def _get_nxsteps ( self ):
        i = abs(self.increment)
        if i != 0:
            return int(round((abs(self.x_max-self.x_min)+i)/i))
        return 1

    @property_depends_on('y_min, y_max, increment')
    def _get_nysteps ( self ):
        i = abs(self.increment)
        if i != 0:
            return int(round((abs(self.y_max-self.y_min)+i)/i))
        return 1

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def pos ( self ):
        """
        Calculates grid co-ordinates.
        
        Returns
        -------
        array of floats of shape (3, :attr:`~Grid.size`)
            The grid point x, y, z-coordinates in one array.
        """
        bpos = mgrid[self.x_min:self.x_max:self.nxsteps*1j, \
                     self.y_min:self.y_max:self.nysteps*1j, \
                     self.z:self.z+0.1]
        bpos.resize((3, self.size))
        return bpos

    def index ( self, x, y ):
        """
        Queries the indices for a grid point near a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        x, y : float
            The co-ordinates for which the indices are queried.

        Returns
        -------
        2-tuple of integers
            The indices that give the grid point nearest to the given x, y
            co-ordinates from an array with the same shape as the grid.            
        """
        if x < self.x_min or x > self.x_max:
            raise ValueError, "x-value out of range"
        if y  <  self.y_min or y > self.y_max:
            raise ValueError, "y-value out of range"
        xi = int((x-self.x_min)/self.increment+0.5)
        yi = int((y-self.y_min)/self.increment+0.5)
        return xi, yi

    def indices ( self, x1, y1, x2, y2=None ):
        """
        Queries the indices for a subdomain in the grid.
        
        Allows either rectangular or circular subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        x1, x2, y1, y2 : float
            If all four paramters are given, then a rectangular sector is
            assumed that is given by two corners (x1, y1) and (x2, y2). If
            only three parameters are given, then a circular sector is assumed
            that is given by its center (x1, y1) and the radius x2.

        Returns
        -------
        2-tuple of arrays of integers or of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        # only 3 values given -> use x,y,radius method
        if y2 is None: 
            xpos = self.pos()
            xis = []
            yis = []
            dr2 = (xpos[0, :]-x1)**2 + (xpos[1, :]-y1)**2
            # array with true/false entries
            inds = dr2 <= x2**2 
            for np in arange(self.size)[inds]: # np -- points in x2-circle
                xi, yi = self.index(xpos[0, np], xpos[1, np])
                xis += [xi]
                yis += [yi]
            if not (xis and yis): # if no points in circle, take nearest one
                return self.index(x1, y1)
            else:
                return array(xis), array(yis)
        else: # rectangular subdomain - old functionality
            xi1, yi1 = self.index(min(x1, x2), min(y1, y2))
            xi2, yi2 = self.index(max(x1, x2), max(y1, y2))
            return s_[xi1:xi2+1], s_[yi1:yi2+1]

    def extend (self) :
        """
        The extension of the grid in pylab.imshow compatible form.

        Returns
        -------
        4-tuple of floats
            The extent of the grid as a tuple of x_min, x_max, y_min, y_max)
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)

class RectGrid3D( RectGrid):
    """
    Provides a cartesian 3D grid for the beamforming results.
    
    The grid has cubic or nearly cubic cells. It is defined by lower and upper 
    x-, y- and  z-limits.
    """

    #: The lower z-limit that defines the grid, defaults to -1.
    z_min = Float(-1.0,
        desc="minimum  z-value")

    #: The upper z-limit that defines the grid, defaults to 1.
    z_max = Float(1.0,
        desc="maximum  z-value")

    #: Number of grid points along x-axis, readonly.
    nzsteps = Property(
        desc="number of grid points alog x-axis")

    #: Respective increments in x,y, and z-direction (in m), defaults 
    #: to :attr:`~RectGrid.increment` for all three (whichever of the two
    #: increment parameters is set last replaces the other). 
    increment3D = CArray( dtype=float, shape=(3, ),
                         desc="3D step sizes")
    def _increment3D_default(self): 
        return array([self.increment,self.increment,self.increment])
    
    @on_trait_change('increment')
    def reset_increment3D(self): 
        self.increment3D = array([self.increment,self.increment,self.increment])
     
    # internal identifier
    digest = Property(
        depends_on = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', \
        'increment3D']
        )

    # increment3D omitted in view for easier handling, can be added later
    traits_view = View(
            [
                ['x_min', 'y_min', 'z_min', '|'],
                ['x_max', 'y_max', 'z_max', 'increment', \
                'size~{Grid size}', '|'],
                '-[Map extension]'
            ]
        )

    @property_depends_on('nxsteps, nysteps, nzsteps')
    def _get_size ( self ):
        return self.nxsteps*self.nysteps*self.nzsteps

    @property_depends_on('nxsteps, nysteps, nzsteps')
    def _get_shape ( self ):
        return (self.nxsteps, self.nysteps, self.nzsteps)
    
    @property_depends_on('x_min, x_max, increment3D')
    def _get_nxsteps ( self ):
        i = abs(self.increment3D[0])
        if i != 0:
            return int(round((abs(self.x_max-self.x_min)+i)/i))
        return 1

    @property_depends_on('y_min, y_max, increment3D')
    def _get_nysteps ( self ):
        i = abs(self.increment3D[1])
        if i != 0:
            return int(round((abs(self.y_max-self.y_min)+i)/i))
        return 1
        
    @property_depends_on('z_min, z_max, increment3D')
    def _get_nzsteps ( self ):
        i = abs(self.increment3D[2])
        if i != 0:
            return int(round((abs(self.z_max-self.z_min)+i)/i))
        return 1

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def pos ( self ):
        """
        Calculates grid co-ordinates.
        
        Returns
        -------
        array of floats of shape (3, :attr:`~Grid.size`)
            The grid point x, y, z-coordinates in one array.
        """
        bpos = mgrid[self.x_min:self.x_max:self.nxsteps*1j, \
                     self.y_min:self.y_max:self.nysteps*1j, \
                     self.z_min:self.z_max:self.nzsteps*1j]
        bpos.resize((3, self.size))
        return bpos

    def index ( self, x, y, z ):
        """
        Queries the indices for a grid point near a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        x, y, z : float
            The co-ordinates for which the indices is queried.

        Returns
        -------
        3-tuple of integers
            The indices that give the grid point nearest to the given x, y, z
            co-ordinates from an array with the same shape as the grid.            
        """
        if x < self.x_min or x > self.x_max:
            raise ValueError, "x-value out of range %f (%f, %f)" % \
                (x,self.x_min,self.x_max)
        if y < self.y_min or y > self.y_max:
            raise ValueError, "y-value out of range %f (%f, %f)" % \
                (y,self.y_min,self.y_max)
        if z < self.z_min or z > self.z_max:
            raise ValueError, "z-value out of range %f (%f, %f)" % \
                (z,self.z_min,self.z_max)
        xi = round((x-self.x_min)/self.increment3D[0])
        yi = round((y-self.y_min)/self.increment3D[1])
        zi = round((z-self.z_min)/self.increment3D[2])
        return xi, yi, zi

    def indices ( self, x1, y1, z1, x2, y2, z2 ):
        """
        Queries the indices for a subdomain in the grid.
        
        Allows box-shaped subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        x1, y1, z1, x2, y2, z2 : float
            A box-shaped sector is assumed that is given by two corners
            (x1,y1,z1) and (x2,y2,z2). 

        Returns
        -------
        3-tuple of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        xi1, yi1, zi1 = self.index(min(x1, x2), min(y1, y2), min(z1, z2))
        xi2, yi2, zi2 = self.index(max(x1, x2), max(y1, y2), max(z1, z2))
        return s_[xi1:xi2+1], s_[yi1:yi2+1], s_[zi1:zi2+1]
