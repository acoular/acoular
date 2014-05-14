# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, C0103, R0901, R0902, R0903, R0904, W0232
# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1101, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) 2007-2014, Beamfpy Development Team.
#------------------------------------------------------------------------------
"""Implements support for two- and threedimensional grids

.. autosummary::
    :toctree: generated/

    Grid
    RectGrid
    RectGrid3D
    MicGeom

"""

# imports from other packages
from numpy import mgrid, s_, array, arange
from traits.api import HasPrivateTraits, Float, Property, File, \
CArray, List, property_depends_on, cached_property, on_trait_change
from traitsui.api import View
from traitsui.menu import OKCancelButtons
from os import path

from .internal import digest

class Grid( HasPrivateTraits ):
    """Virtual base class for grid geometries.
    
    Defines the common interface for all grid classes and
    provides facilities to query grid properties and related data. This class
    may be used as a base for specialized grid implementaions. It should not
    be used directly as it contains no real functionality.
    """

    #: overall number of grid points, readonly, is set automatically when
    #: other grid defining properties are set
    size = Property(
        desc="overall number of grid points")

    #: shape of grid, readonly, gives the shape as tuple, useful for cartesian
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
        """Calculates grid co-ordinates.
        
        Returns
        -------
        array of floats of shape (3, size)
            The grid point x, y, z-coordinates in one array.
        """
        return array([[0.],[0.],[0.]])

# This is not needed in the base class (?)
#    def extend (self) :
#        """
#        returns the x, y extension of the grid,
#        useful for the imshow function from pylab
#        """
#        pos = self.pos()
#        return (min(pos[0,:]), max(pos[0,:]), min(pos[1,:]), max(pos[1,:]))


class RectGrid( Grid ):
    """Provides a cartesian 2D grid for the beamforming results.
    
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
                ['x_max', 'y_max', 'z', 'increment', 'size~{grid size}', '|'],
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
        """Calculates grid co-ordinates.
        
        Returns
        -------
        array of floats of shape (3, size)
            The grid point x, y, z-coordinates in one array.
        """
        i = self.increment
        xi = 1j*round((self.x_max-self.x_min+i)/i)
        yi = 1j*round((self.y_max-self.y_min+i)/i)
        bpos = mgrid[self.x_min:self.x_max:xi, self.y_min:self.y_max:yi, \
            self.z:self.z+0.1]
        bpos.resize((3, self.size))
        return bpos

    def index ( self, x, y ):
        """Queries the indices for a grid point near a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        x,y : float
            The co-ordinates for which the indices is queried.

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
        """Queries the indices for a subdomain in the grid.
        
        Allows either rectagular or circular subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        x1, x2, y1, y2 : float
            If all four paramters are given, then a rectangular sector is
            assumed that is given by two corners (x1,y1) and (x2,y2). If
            only three parameters are given, then a circular sector is assumed
            that is given by its center (x1,y1) and the radius x2.

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
            dr2 = (xpos[0,:]-x1)**2 + (xpos[1,:]-y1)**2
            # array with true/false entries
            inds = dr2 <= x2**2 
            for np in arange(self.size)[inds]: # np -- points in x2-circle
                xi, yi = self.index(xpos[0,np], xpos[1,np])
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
        """The extension of the grid in pylab imshow compatible form.

        Returns
        -------
        4-tuple of floats
            The extent of the grid as a tuple of x_min, x_max, y_min, y_max)
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)

class RectGrid3D( RectGrid):
    """Provides a cartesian 3D grid for the beamforming results.
    
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

    # internal identifier
    digest = Property(
        depends_on = ['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', \
        'increment']
        )

    traits_view = View(
            [
                ['x_min', 'y_min', 'z_min', '|'],
                ['x_max', 'y_max', 'z_max', 'increment', \
                'size~{grid size}', '|'],
                '-[Map extension]'
            ]
        )

    @property_depends_on('nxsteps, nysteps, nzsteps')
    def _get_size ( self ):
        return self.nxsteps*self.nysteps*self.nzsteps

    @property_depends_on('nxsteps, nysteps, nzsteps')
    def _get_shape ( self ):
        return (self.nxsteps, self.nysteps, self.nzsteps)

    @property_depends_on('z_min, z_max, increment')
    def _get_nzsteps ( self ):
        i = abs(self.increment)
        if i != 0:
            return int(round((abs(self.z_max-self.z_min)+i)/i))
        return 1

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def pos ( self ):
        """Calculates grid co-ordinates.
        
        Returns
        -------
        array of floats of shape (3, size)
            The grid point x, y, z-coordinates in one array.
        """
        i = self.increment
        xi = 1j*round((self.x_max-self.x_min+i)/i)
        yi = 1j*round((self.y_max-self.y_min+i)/i)
        zi = 1j*round((self.z_max-self.z_min+i)/i)
        bpos = mgrid[self.x_min:self.x_max:xi, \
            self.y_min:self.y_max:yi, \
            self.z_min:self.z_max:zi]
        bpos.resize((3, self.size))
        return bpos

    def index ( self, x, y, z ):
        """Queries the indices for a grid point near a certain co-ordinate.

        This can be used to query results or co-ordinates at/near a certain
        co-ordinate.
        
        Parameters
        ----------
        x : float
        y : float
        z : float
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
        xi = round((x-self.x_min)/self.increment)
        yi = round((y-self.y_min)/self.increment)
        zi = round((z-self.z_min)/self.increment)
        return xi, yi, zi

    def indices ( self, x1, y1, z1, x2, y2, z2 ):
        """Queries the indices for a subdomain in the grid.
        
        Allows box-shaped subdomains. This can be used to
        mask or to query results from a certain sector or subdomain.
        
        Parameters
        ----------
        x1 : float
        y1 : float
        z1 : float
        x2 : float
        y2 : float
        z2 : float
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

class MicGeom( HasPrivateTraits ):
    """Provides the geometric arrangement of microphones in the mic. array.
    
    The geometric arrangement of microphones is read in from an 
    xml-source with element tag names 'pos' and attributes Name, x, y and z. 
    Can also be used with programmatically generated arrangements.
    """

    #: Name of the .xml-file from wich to read the data.
    from_file = File(filter=['*.xml'],
        desc="name of the xml file to import")

    #: Basename of the .xml-file, without the extension, readonly.
    basename = Property( depends_on = 'from_file',
        desc="basename of xml file")

    #: List that gives the indices of channels that should not be considered.
    #: Defaults to a blank list.
    invalid_channels = List(
        desc="list of invalid channels")

    #: Number of microphones in the array, readonly.
    num_mics = Property( depends_on = ['mpos', ],
        desc="number of microphones in the geometry")

    #: Positions as (3, num_mics) array of floats, may include also invalid
    #: microphones (if any). Set either automatically on change of the
    #: from_file argument or explicitely by assigning an array of floats.
    mpos_tot = CArray(
        desc="x, y, z position of all microphones")

    #: Positions as (3, num_mics) array of floats, without invalid
    #: microphones, readonly.
    mpos = Property( depends_on = ['mpos_tot', 'invalid_channels'],
        desc="x, y, z position of microphones")

    # internal identifier
    digest = Property( depends_on = ['mpos', ])

    traits_view = View(
        ['from_file',
        'num_mics~',
        '|[Microphone geometry]'
        ],
#        title='Microphone geometry',
        buttons = OKCancelButtons
                    )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    @cached_property
    def _get_basename( self ):
        return path.splitext(path.basename(self.from_file))[0]

    @cached_property
    def _get_mpos( self ):
        if len(self.invalid_channels)==0:
            return self.mpos_tot
        allr = range(self.mpos_tot.shape[-1])
        for channel in self.invalid_channels:
            if channel in allr:
                allr.remove(channel)
        return self.mpos_tot[:, array(allr)]

    @cached_property
    def _get_num_mics( self ):
        return self.mpos.shape[-1]

    @on_trait_change('basename')
    def import_mpos( self ):
        """import the microphone positions from .xml file,
        called when basename changes
        """
        if not path.isfile(self.from_file):
            # no file there
            self.mpos_tot = array([], 'd')
            self.num_mics = 0
            return
        import xml.dom.minidom
        doc = xml.dom.minidom.parse(self.from_file)
        names = []
        xyz = []
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('Name'))
            xyz.append(map(lambda a : float(el.getAttribute(a)), 'xyz'))
        self.mpos_tot = array(xyz, 'd').swapaxes(0, 1)
#        self.num_mics = self.mpos.shape[1]

