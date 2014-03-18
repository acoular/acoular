# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, C0103, R0901, R0902, R0903, R0904, W0232
"""
grids.py: classes providing support for grids and microphone geometries

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

# imports from other packages
from numpy import mgrid, s_, array, isscalar, float32, float64, newaxis, \
sqrt, arange, pi, exp, sin, cos, arccos, zeros_like, empty, dot, hstack, \
vstack, identity
from numpy.linalg.linalg import norm
from scipy.integrate import ode
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
from traits.api import HasPrivateTraits, Float, Property, File, Int, \
CArray, List, property_depends_on, cached_property, on_trait_change, Trait
from traitsui.api import View
from traitsui.menu import OKCancelButtons
from os import path

from internal import digest

class Grid( HasPrivateTraits ):
    """
    base class for grid geometries
    """

    # overall number of grid points (auto-set)
    size = Property( 
        desc="overall number of grid points")

    # shape of grid
    shape = Property(
        desc="grid shape as tuple")
        
    # internal identifier
    digest = ''
    
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
        returns an (3, size) array with the grid point x, y, z-coordinates
        """
        return array([[0.],[0.],[0.]])
        
    def extend (self) :
        """
        returns the x, y extension of the grid, 
        useful for the imshow function from pylab
        """
        pos = self.pos()
        return (min(pos[0,:]), max(pos[0,:]), min(pos[1,:]), max(pos[1,:]))
        

class RectGrid( Grid ):
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

    # number of grid points along x-axis (auto-set)
    nxsteps = Property( 
        desc="number of grid points along x-axis")

    # number of grid points along y-axis (auto-set)
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
        xi = int((x-self.x_min)/self.increment+0.5)
        yi = int((y-self.y_min)/self.increment+0.5)
        return xi, yi

    def indices ( self, x1, y1, x2, y2=None ):
        """
        returns the slices to index a rectangular subdomain, or,
        alternatively, the indices of all grid points with distance 
        x2 or less to point (x1, y1)
        useful for inspecting subdomains in a result already calculated
        """
        if y2 is None: # only 3 values given -> use x,y,radius method
            xpos = self.pos()

            xis = []
            yis = []
            
            dr2 = (xpos[0,:]-x1)**2 + (xpos[1,:]-y1)**2
            inds = dr2 <= x2**2 # array with true/false entries
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
        """ calculates distances between grid points and mics or origin """
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
        """ 
        calculates virtual distances between grid points and mics or origin """
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

class FlowField( HasPrivateTraits ):
    """
    abstract base class for spatial flow field
    """
    digest = Property

    traits_view = View(
        )
        
    def _get_digest( self ):
        return ''
    
    def v( self, xx):
        """ 
        returns v and Jacobian at location xx
        """
        v = array( (0., 0., 0.) )
        dv = array( ((0.,0.,0.),(0.,0.,0.),(0.,0.,0.)) )      
        return -v,-dv                         
        
class OpenJet( FlowField ):
    """
    analytical approximation of the flow field of an open jet,
    see Albertson et al., 1950
    """
    # exit velocity
    v0 = Float(0.0, 
        desc="exit velocity")
        
    # nozzle center
    origin = CArray( dtype=float64, shape=(3, ), value=array((0., 0., 0.)), 
        desc="center of nozzle")

    # nozzle diameter
    D = Float(0.2, 
        desc="nozzle diameter")

    # internal identifier
    digest = Property( 
        depends_on=['v0', 'origin','D'], 
        )

    traits_view = View(
            [
                ['v0{exit velocity}', 'origin{jet origin}', 
                'D{nozzle diameter}'], 
                '|[open jet]'
            ]
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    # velocity and Jacobian, y and z components are neglected
    def v( self, xx):
        x,y,z = xx-self.origin
        r = sqrt(y*y+z*z)
        x1 = 0.081*x
        h1 = r+x1-0.5*self.D
        U = self.v0*exp(-h1*h1/(2*x1*x1))
        if h1<0.0:
            Udr = 0.0
            U = self.v0
        else:
            Udr = -h1*U/(x1*x1)
        if r>0.0:
            Udy = y*Udr/r
            Udz = z*Udr/r
        else:
            Udy = Udz = 0.0
        Udx = (h1*h1/(x*x1*x1)-h1/(x*x1))*U
        if h1<0.0:
            Udx = 0
        v = array( (U, 0., 0.) ) 
        dv = array( ((Udx,0.,0.),(Udy,0.,0.),(Udz,0.,0.)) )      
        return v,dv                         
       
def spiral_sphere(N,Om=2*pi,b=array((0,0,1))):
    """
    returns an array of unit vectors (N,3) giving equally distributed
    directions on a part of sphere given by the center direction b and
    the solid angle Om
    """
    # first produce 'equally' distributed directions in spherical coords
    o=4*pi/Om
    h = -1+ 2*arange(N)/(N*o-1.)
    theta = arccos(h)
    phi = zeros_like(theta)
    for i,hk in enumerate(h[1:]):
        phi[i+1] = phi[i]+3.6/sqrt(N*o*(1-hk*hk)) % (2*pi)
    # translate to cartesian coords
    xyz = vstack((sin(theta) * cos(phi),sin(theta) * sin(phi),cos(theta)))
    # mirror everything on a plane so that b points into the center
    a = xyz[:,0]
    b = b/norm(b)
    ab = (a-b)[:,newaxis]
    if norm(ab)<1e-10:
        return xyz
    # this is the Householder matrix for mirroring
    H = identity(3)-dot(ab,ab.T)/dot(ab.T,a)
    # actual mirroring
    return dot(H,xyz)    


class GeneralFlowEnvironment( Environment):
    """
    general flow enviroment
    """
    # the Mach number
    ff = Trait(FlowField, 
        desc="flow field")
        
    # exit velocity
    N = Int(100, 
        desc="number of rays per pi")

    # exit velocity
    Om = Float(pi, 
        desc="maximum spherical angle")
        
    # internal identifier
    digest = Property( 
        depends_on=['ff.digest','N','Om'], 
        )

    traits_view = View(
            [
                ['ff','N{max. number of rays}','Om{max. solid angle }'], 
                '|[General Flow]'
            ]
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    def r( self, c, gpos, mpos=0.0):
        """ 
        calculates virtual distances between grid points and mics or origin 
        """
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]

        # the DE system
        def f1(t, y, v):
            x = y[0:3]                            
            s = y[3:6]
            vv, dv = v(x)
            sa = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2])
            x = empty(6)
            x[0:3] = c*s/sa - vv # time reversal
            x[3:6] = dot(s, -dv) # time reversal
            return x
        
        # integration along a single ray 
        def fr(x0,n0, rmax, dt, v, xyz, t):
            s0 = n0 / (c+dot(v(x0)[0],n0))
            y0 = hstack((x0,s0))
            oo = ode(f1)
            oo.set_f_params(v)
            oo.set_integrator('vode',rtol=1e-2)
            oo.set_initial_value(y0,0)
            while oo.successful():
                xyz.append(oo.y[0:3])
                t.append(oo.t)
                if norm(oo.y[0:3]-x0)>rmax:
                    break
                oo.integrate(oo.t+dt)    

        print gpos.shape                
        gs2 = gpos.shape[-1]
        gt = empty((gs2,mpos.shape[-1]))
        vv = self.ff.v
        NN = int(sqrt(self.N))
        for micnum,x0 in enumerate(mpos.T):
            xe = gpos.mean(1) # center of grid
            r = x0[:,newaxis]-gpos
            rmax = sqrt((r*r).sum(0).max()) # maximum distance
            nv = spiral_sphere(self.N,pi,b=xe-x0)
            rstep = rmax/sqrt(self.N)
            rmax += rstep
            tstep = rstep/c
            xyz = []
            t = []
            lastind = 0
            for i,n0 in enumerate(nv.T):
                fr(x0,n0,rmax,tstep,vv,xyz,t)
                if i and i%NN==0:
                    if not lastind:
                        dd = ConvexHull(vstack((gpos.T,xyz)),incremental=True)
                    else:
                        dd.add_points(xyz[lastind:],restart=True)
                    lastind = len(xyz)
                    # ConvexHull includes grid ?
                    if dd.simplices.min()>=gs2:
                        break
            xyz = array(xyz)
            t = array(t)    
            li = LinearNDInterpolator(xyz,t)
            gt[:,micnum] = li(gpos.T)
        if gt.shape[1] == 1:
            gt = gt[:, 0]
    #        print gt[:,micnum].max(),gt[:,micnum].min(),gt[:,micnum].mean()   
        return c*gt #return distance along ray
