# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2020, Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements support for two- and threedimensional grids

.. autosummary::
    :toctree: generated/

    Grid
    RectGrid
    RectGrid3D
    Sector
    RectSector
    CircSector
    PolySector
    MultiSector

"""

# imports from other packages
from numpy import mgrid, s_, array, arange, isscalar, absolute, ones, argmin,\
zeros, where, bool_, nonzero
from traits.api import HasPrivateTraits, Float, Property, Any, \
property_depends_on, cached_property, Bool, List, Instance
from traits.trait_errors import TraitError
#from matplotlib.path import Path
from scipy.spatial import Delaunay
from .internal import digest


def in_hull(p, hull, border= True, tol = 0 ):
    """
    test if points in `p` are in `hull`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    
    if border:
        return hull.find_simplex(p,tol = tol)>=0
    else:
        return hull.find_simplex(p,tol = tol)>0
        
def in_poly(p,poly, border= True, tol = 0 ):
    """
    test if points in `p` are in `poly`
    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `poly` is the `MxK` array of the coordinates of `M` points in `K`dimensions
    
    """  
    n = len(poly)
    inside = zeros(len(p[:,0]),bool_)
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        idx = nonzero((p[:,1] > min(p1y,p2y)) & (p[:,1] <= max(p1y,p2y)) & (p[:,0] <= max(p1x,p2x)))[0]
        if p1y != p2y:
            xints = (p[:,1][idx]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
        if p1x == p2x:
            inside[idx] = ~inside[idx]
        else:
            idxx = idx[p[:,0][idx] <= xints]
            inside[idxx] = ~inside[idxx]    
        p1x,p1y = p2x,p2y
    return inside



class Grid( HasPrivateTraits ):
    """
    Virtual base class for grid geometries.
    
    Defines the common interface for all grid classes and
    provides facilities to query grid properties and related data. This class
    may be used as a base for specialized grid implementaions. It should not
    be used directly as it contains no real functionality.
    """

    #: Overall number of grid points. Readonly; is set automatically when
    #: other grid defining properties are set
    size = Property(desc="overall number of grid points")

    #: Shape of grid. Readonly, gives the shape as tuple, useful for cartesian
    #: grids
    shape = Property(desc="grid shape as tuple")

    #: Grid positions as (3, :attr:`size`) array of floats, without invalid
    #: microphones; readonly.
    gpos = Property(desc="x, y, z positions of grid points")

    # internal identifier
    digest = Property

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

    @property_depends_on('digest')
    def _get_gpos( self ):
        return array([[0.], [0.], [0.]])
    
    def pos ( self ):
        """
        Calculates grid co-ordinates. 
        Deprecated; use :attr:`gpos` attribute instead.
        
        Returns
        -------
        array of floats of shape (3, :attr:`size`)
            The grid point x, y, z-coordinates in one array.
        """
        return self.gpos# array([[0.], [0.], [0.]])
    
    def subdomain (self, sector) :
        """
        Queries the indices for a subdomain in the grid.
        
        Allows arbitrary subdomains of type :class:`Sector`
        
        Parameters
        ----------
        sector : :class:`Sector`
            Sector describing the subdomain.

        Returns
        -------
        2-tuple of arrays of integers or of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        
        xpos = self.gpos
        # construct grid-shaped array with "True" entries where sector is
        xyi = sector.contains(xpos).reshape(self.shape)
        # return indices of "True" entries
        return where(xyi)


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

    @property_depends_on('x_min, x_max, y_min, y_max, increment')
    def _get_gpos ( self ):
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
            raise ValueError("x-value out of range")
        if y  <  self.y_min or y > self.y_max:
            raise ValueError("y-value out of range")
        xi = int((x-self.x_min)/self.increment+0.5)
        yi = int((y-self.y_min)/self.increment+0.5)
        return xi, yi

    def indices ( self, *r):
        """
        Queries the indices for a subdomain in the grid.
        
        Allows either rectangular, circular or polygonial subdomains.
        This can be used to mask or to query results from a certain 
        sector or subdomain.
        
        Parameters
        ----------
        x1, y1, x2, y2, ... : float
            If three parameters are given, then a circular sector is assumed
            that is given by its center (x1, y1) and the radius x2.
            If four paramters are given, then a rectangular sector is
            assumed that is given by two corners (x1, y1) and (x2, y2). 
            If more parameters are given, the subdomain is assumed to have
            polygonial shape with corners at (x_n, y_n).

        Returns
        -------
        2-tuple of arrays of integers or of numpy slice objects
            The indices that can be used to mask/select the grid subdomain from 
            an array with the same shape as the grid.            
        """
        
        if len(r) == 3: # only 3 values given -> use x,y,radius method
            xpos = self.gpos
            xis = []
            yis = []
            dr2 = (xpos[0, :]-r[0])**2 + (xpos[1, :]-r[1])**2
            # array with true/false entries
            inds = dr2 <= r[2]**2 
            for np in arange(self.size)[inds]: # np -- points in x2-circle
                xi, yi = self.index(xpos[0, np], xpos[1, np])
                xis += [xi]
                yis += [yi]
            if not (xis and yis): # if no points in circle, take nearest one
                return self.index(r[0], r[1])
            else:
                return array(xis), array(yis)
        elif len(r) == 4: # rectangular subdomain - old functionality
            xi1, yi1 = self.index(min(r[0], r[2]), min(r[1], r[3]))
            xi2, yi2 = self.index(max(r[0], r[2]), max(r[1], r[3]))
            return s_[xi1:xi2+1], s_[yi1:yi2+1]
        else: # use enveloping polygon
            xpos = self.gpos
            xis = []
            yis = []
            #replaced matplotlib Path by scipy spatial 
            #p = Path(array(r).reshape(-1,2))
            #inds = p.contains_points()
            inds = in_hull(xpos[:2,:].T,array(r).reshape(-1,2))
            for np in arange(self.size)[inds]: # np -- points in x2-circle
                xi, yi = self.index(xpos[0, np], xpos[1, np])
                xis += [xi]
                yis += [yi]
            if not (xis and yis): # if no points inside, take nearest to center
                center = array(r).reshape(-1,2).mean(0)
                return self.index(center[0], center[1])
            else:
                return array(xis), array(yis)
                #return arange(self.size)[inds]

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
        desc="number of grid points along x-axis")

    
    # Private trait for increment handling
    _increment = Any(0.1)
    
    #: The cell side length for the grid. This can either be a scalar (same 
    #: increments in all 3 dimensions) or a (3,) array of floats with 
    #: respective increments in x,y, and z-direction (in m). 
    #: Defaults to 0.1.
    increment = Property(desc="step size")
    
    def _get_increment(self):
        return self._increment
    
    def _set_increment(self, increment):
        if isscalar(increment):
            try:
                self._increment = absolute(float(increment))
            except:
                raise TraitError(args=self,
                                 name='increment', 
                                 info='Float or CArray(3,)',
                                 value=increment) 
        elif len(increment) == 3:
            self._increment = array(increment,dtype=float)
        else:
            raise(TraitError(args=self,
                             name='increment', 
                             info='Float or CArray(3,)',
                             value=increment))
    
    # Respective increments in x,y, and z-direction (in m).
    # Deprecated: Use :attr:`~RectGrid.increment` for this functionality
    increment3D = Property(desc="3D step sizes")
    
    def _get_increment3D(self):
        if isscalar(self._increment):
            return array([self._increment,self._increment,self._increment])
        else:
            return self._increment
    
    def _set_increment3D(self, inc):
        if not isscalar(inc) and len(inc) == 3:
            self._increment = array(inc,dtype=float)
        else:
            raise(TraitError(args=self,
                             name='increment3D', 
                             info='CArray(3,)',
                             value=inc))
    
    # internal identifier
    digest = Property(
        depends_on = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', \
        '_increment']
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

    @property_depends_on('digest')
    def _get_gpos ( self ):
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

    @cached_property
    def _get_digest( self ):
        return digest( self )

   

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
            raise ValueError("x-value out of range %f (%f, %f)" % \
                (x,self.x_min,self.x_max))
        if y < self.y_min or y > self.y_max:
            raise ValueError("y-value out of range %f (%f, %f)" % \
                (y,self.y_min,self.y_max))
        if z < self.z_min or z > self.z_max:
            raise ValueError("z-value out of range %f (%f, %f)" % \
                (z,self.z_min,self.z_max))
        xi = int(round((x-self.x_min)/self.increment3D[0]))
        yi = int(round((y-self.y_min)/self.increment3D[1]))
        zi = int(round((z-self.z_min)/self.increment3D[2]))
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



class Sector( HasPrivateTraits ):
    """
    Base class for sector types.
    
    Defines the common interface for all sector classes. This class
    may be used as a base for diverse sector implementaions. If used
    directly, it implements a sector encompassing the whole grid.
    """
    
    #: Boolean flag, if 'True' (default), grid points lying on the sector border are included.
    include_border = Bool(True, 
                          desc="include points on the border")    
    
    #: Absolute tolerance for sector border
    abs_tol = Float(1e-12,
                    desc="absolute tolerance for sector border")

    #: Boolean flag, if 'True' (default), the nearest grid point is returned if none is inside the sector.
    default_nearest = Bool(True, 
                          desc="return nearest grid point to center of none inside sector")

    def contains ( self, pos ):
        """
        Queries whether the coordinates in a given array lie within the 
        defined sector. 
        For this sector type, any position is valid.
        
        Parameters
        ----------
        pos : array of floats
            Array with the shape 3x[number of gridpoints] containing the
            grid positions
        
        Returns
        -------
        array of bools with as many entries as columns in pos
            Array indicating which of the given positions lie within the
            given sector                
        """
        return ones(pos.shape[1], dtype=bool)


class RectSector( Sector ):
    """
    Class for defining a rectangular sector.
    
    Can be used for 2D Grids for definining a rectangular sector or
    for 3D grids for a rectangular cylinder sector parallel to the z-axis.
    """
    
    #: The lower x position of the rectangle
    x_min = Float(-1.0,
                  desc="minimum x position of the rectangle")

    #: The upper x position of the rectangle
    x_max = Float(1.0,
                  desc="maximum x position of the rectangle")

    #: The lower y position of the rectangle
    y_min = Float(-1.0,
                  desc="minimum y position of the rectangle")
    
    #: The upper y position of the rectangle
    y_max = Float(1.0,
                  desc="maximum y position of the rectangle")

    def contains ( self, pos ):
        """
        Queries whether the coordinates in a given array lie within the 
        rectangular sector. 
        If no coordinate is inside, the nearest one to the rectangle center
        is returned if :attr:`~Sector.default_nearest` is True.
        
        Parameters
        ----------
        pos : array of floats
            Array with the shape 3x[number of gridpoints] containing the
            grid positions
        
        Returns
        -------
        array of bools with as many entries as columns in pos
            Array indicating which of the given positions lie within the
            given sector                
        """
        # make sure xmin is minimum etc
        xmin = min(self.x_min,self.x_max)
        xmax = max(self.x_min,self.x_max)
        ymin = min(self.y_min,self.y_max)
        ymax = max(self.y_min,self.y_max)
        
        abs_tol = self.abs_tol
        # get pos indices inside rectangle (* == and)
        if self.include_border:
            inds = (pos[0, :] - xmin > -abs_tol) * \
                   (pos[0, :] - xmax < abs_tol) * \
                   (pos[1, :] - ymin > -abs_tol) * \
                   (pos[1, :] - ymax < abs_tol)
        else:
            inds = (pos[0, :] - xmin > abs_tol) * \
                   (pos[0, :] - xmax < -abs_tol) * \
                   (pos[1, :] - ymin > abs_tol) * \
                   (pos[1, :] - ymax < -abs_tol)
        
        
        # if none inside, take nearest
        if ~inds.any() and self.default_nearest:
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            dr2 = (pos[0, :] - x)**2 + (pos[1, :] - y)**2
            inds[argmin(dr2)] = True
        
        return inds.astype(bool)


class CircSector( Sector ):
    """
    Class for defining a circular sector.
    
    Can be used for 2D Grids for definining a circular sector or
    for 3D grids for a cylindrical sector parallel to the z-axis.
    """
    
    #: x position of the circle center
    x = Float(0.0,
        desc="x position of the circle center")

    #: y position of the circle center
    y = Float(0.0,
        desc="y position of the circle center")
    
    #: radius of the circle
    r = Float(1.0,
        desc="radius of the circle")
        
    
    def contains ( self, pos ):
        """
        Queries whether the coordinates in a given array lie within the 
        circular sector. 
        If no coordinate is inside, the nearest one outside is returned
        if :attr:`~Sector.default_nearest` is True.
        
        Parameters
        ----------
        pos : array of floats
            Array with the shape 3x[number of gridpoints] containing the
            grid positions
        
        Returns
        -------
        array of bools with as many entries as columns in pos
            Array indicating which of the given positions lie within the
            given sector                
        """
        dr2 = (pos[0, :]-self.x)**2 + (pos[1, :]-self.y)**2
        # which points are in the circle?
        if self.include_border:
            inds = (dr2 - self.r**2) < self.abs_tol
        else:
            inds = (dr2 - self.r**2) < -self.abs_tol
        
        
        # if there's no poit inside
        if ~inds.any() and self.default_nearest: 
            inds[argmin(dr2)] = True
        
        return inds


class PolySector( Sector ):
    """
     Class for defining a polygon sector.
    
     Can be used for 2D Grids for definining a convex polygon sector.
    """
    # x1, y1, x2, y2, ... xn, yn :
    edges = List( Float ) 
    

    def contains ( self, pos ):
        """
        Queries whether the coordinates in a given array lie within the 
        defined sector. 
        For this sector type 
        
        Parameters
        ----------
        pos : array of floats
            Array with the shape 3x[number of gridpoints] containing the
            grid positions
        
        Returns
        -------
        array of bools with as many entries as columns in pos
            Array indicating which of the given positions lie within the
            given sector                
        """
        
        inds = in_poly(pos[:2,:].T, array(self.edges).reshape(-1,2), \
                           border = self.include_border ,tol = self.abs_tol)
        
        # if none inside, take nearest
        if ~inds.any() and self.default_nearest:
            dr2 = array(self.edges).reshape(-1,2).mean(0)
            inds[argmin(dr2)] = True
          
        return inds





class MultiSector(Sector):
    """
    Class for defining a sector consisting of multiple sectors.
    
    Can be used to sum over different sectors. Takes a list of sectors
    and returns the points contained in each sector.
    
    """
    
    #: List of :class:`acoular.grids.Sector` objects
    #: to be mixed.
    sectors = List(Instance(Sector)) 
    
    
    def contains ( self, pos ):
        """
        Queries whether the coordinates in a given array lie within any 
        of the sub-sectors. 
        
        Parameters
        ----------
        pos : array of floats
            Array with the shape 3x[number of gridpoints] containing the
            grid positions
        
        Returns
        -------
        array of bools with as many entries as columns in pos
            Array indicating which of the given positions lie within the
            sectors              
        """
        # initialize with only "False" entries
        inds = zeros(pos.shape[1], dtype=bool)
        
        # add points contained in each sector
        for sec in self.sectors:
            inds += sec.contains(pos)
        
        return inds.astype(bool)






