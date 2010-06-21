# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:25:10 2010

@author: sarradj
"""
#pylint: disable-msg=E0611, C0111, C0103, R0901, R0902, R0903, R0904, W0232
# imports from other packages
from numpy import mgrid, s_, array, isscalar, float32, float64, newaxis, sqrt
from enthought.traits.api import HasPrivateTraits, Float, Property, File, \
CArray, List, property_depends_on, cached_property, on_trait_change
from enthought.traits.ui.api import View
from enthought.traits.ui.menu import OKCancelButtons
from os import path

from internal import digest

#TODO: construct a base class for this
class RectGrid( HasPrivateTraits ):
    """
    constructs a quadratic 2D grid for the beamforming results
    that is on a plane perpendicular to the z-axis
    """

    x_min = Float(-1.0, 
        desc="minimum  x-value")

    x_max = Float(1.0, 
        desc="maximum  x-value")

    y_min = Float(-1.0, 
        desc="minimum  y-value")

    y_max = Float(1.0, 
        desc="maximum  y-value")

    z = Float(1.0, 
        desc="position on z-axis")

    # increment in x- and y- direction
    increment = Float(0.1, 
        desc="step size")

    # overall number of grid points (auto-set)
    size = Property( 
        desc="overall number of grid points")
        
    # shape of grid
    shape = Property(
        desc="grid shape as tuple")
    
    # number of grid points alog x-axis (auto-set)
    nxsteps = Property( 
        desc="number of grid points alog x-axis")

    # number of grid points alog y-axis (auto-set)
    nysteps = Property( 
        desc="number of grid points alog y-axis")

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
        """
        returns an (3, size) array with the grid point x, y, z-coordinates
        """
        i = self.increment
        xi = 1j*round((self.x_max-self.x_min+i)/i)
        yi = 1j*round((self.y_max-self.y_min+i)/i)
        bpos = mgrid[self.x_min:self.x_max:xi, self.y_min:self.y_max:yi, \
            self.z:self.z+0.1]
        bpos.resize((3, self.size))
        return bpos

    def index ( self, x, y ):
        """
        returns the indices for a certain x, y co-ordinate
        """
        if x < self.x_min or x > self.x_max:
            raise ValueError, "x-value out of range"
        if y  <  self.y_min or y > self.y_max:
            raise ValueError, "y-value out of range"
        xi = round((x-self.x_min)/self.increment)
        yi = round((y-self.y_min)/self.increment)
        return xi, yi

    def indices ( self, x1, y1, x2, y2 ):
        """
        returns the slices to index a recangular subdomain, 
        useful for inspecting subdomains in a result already calculated
        """
        xi1, yi1 = self.index(min(x1, x2), min(y1, y2))
        xi2, yi2 = self.index(max(x1, x2), max(y1, y2))
        return s_[xi1:xi2+1], s_[yi1:yi2+1]

    def extend (self) :
        """
        returns the x, y extension of the grid, 
        useful for the imshow function from pylab
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)

class RectGrid3D( RectGrid):
    """
    constructs a quadratic 3D grid for the beamforming results
    """

    z_min = Float(-1.0, 
        desc="minimum  z-value")

    z_max = Float(1.0, 
        desc="maximum  z-value")
    
    # number of grid points alog x-axis (auto-set)
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
        """
        returns an (3, size) array with the grid point x, y, z-coordinates
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
        """
        returns the indices for a certain x, y, z co-ordinate
        """
        if x < self.x_min or x > self.x_max:
            raise ValueError, "x-value out of range"
        if y < self.y_min or y > self.y_max:
            raise ValueError, "y-value out of range"
        if z < self.z_min or z > self.z_max:
            raise ValueError, "z-value out of range"
        xi = round((x-self.x_min)/self.increment)
        yi = round((y-self.y_min)/self.increment)
        zi = round((z-self.z_min)/self.increment)
        return xi, yi, zi

    def indices ( self, x1, y1, z1, x2, y2, z2 ):
        """
        returns the slices to index a rectangular subdomain, 
        useful for inspecting subdomains in a result already calculated
        """
        xi1, yi1, zi1 = self.index(min(x1, x2), min(y1, y2), min(z1, z2))
        xi2, yi2, zi2 = self.index(max(x1, x2), max(y1, y2), max(z1, z2))
        return s_[xi1:xi2+1], s_[yi1:yi2+1], s_[zi1:zi2+1]

class MicGeom( HasPrivateTraits ):
    """
    container for the geometric arrangement of microphones
    reads data from xml-source with element tag names 'pos'
    and attributes Name, x, y and z
    """

    # name of the .xml-file
    from_file = File(filter=['*.xml'], 
        desc="name of the xml file to import")

    # basename of the .xml-file
    basename = Property( depends_on = 'from_file', 
        desc="basename of xml file")
    
    # invalid channels  
    invalid_channels = List(
        desc="list of invalid channels")
    
    # number of mics
    num_mics = Property( depends_on = ['mpos', ], 
        desc="number of microphones in the geometry")

    # positions as (3, num_mics) array
    mpos_tot = CArray(
        desc="x, y, z position of all microphones")

    # positions as (3, num_mics) array
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
            self.mpos = array([], 'd')
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

class Environment( HasPrivateTraits ):
    """
    environment description base class
    """
    # internal identifier
    digest = Property

    traits_view = View(
        )
        
    def _get_digest( self ):
        return ''
    
    def r( self, c, gpos, mpos=0.0):
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]            
        mpos = mpos[:, newaxis, :]
        rmv = gpos[:, :, newaxis]-mpos
        rm = sqrt(sum(rmv*rmv, 0))
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        return rm

class UniformFlowEnvironment( Environment):
    """
    uniform flow enviroment
    """
    # the Mach number
    ma = Float(0.0, 
        desc="flow mach number")
        
    # the flow direction vector
    fdv = CArray( dtype=float64, shape=(3, ), value=array((1.0, 0, 0)), 
        desc="flow direction")

    # internal identifier
    digest = Property( 
        depends_on=['ma', 'fdv'], 
        )

    traits_view = View(
            [
                ['ma{flow Mach number}', 'fdv{flow vector}'], 
                '|[Uniform Flow]'
            ]
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    def r( self, c, gpos, mpos=0.0):
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]
        fdv = self.fdv/sqrt((self.fdv*self.fdv).sum())
        mpos = mpos[:, newaxis, :]
        rmv = gpos[:, :, newaxis]-mpos
        rm = sqrt(sum(rmv*rmv, 0))
        macostheta = (self.ma*sum(rmv.reshape((3, -1))*fdv[:, newaxis], 0)\
            /rm.reshape(-1)).reshape(rm.shape)        
        rm *= 1/(-macostheta + sqrt(macostheta*macostheta-self.ma*self.ma+1))
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        return rm
