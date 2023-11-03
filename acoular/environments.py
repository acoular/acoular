# -*- coding: utf-8 -*-
#pylint: disable-msg=E0611, E1103, C0103, R0901, R0902, R0903, R0904, W0232
#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
"""Implements acoustic environments with and without flow.

.. autosummary::
    :toctree: generated/

    Environment
    UniformFlowEnvironment
    GeneralFlowEnvironment
    FlowField
    OpenJet
    RotatingFlow
    SlotJet

"""
import numba as nb
from numpy import array, isscalar, float32, float64, newaxis, \
sqrt, arange, pi, exp, sin, cos, arccos, zeros_like, empty, dot, hstack, \
vstack, identity, cross, sign, arctan2, matmul, sum, ascontiguousarray
from numpy.linalg.linalg import norm
from scipy.integrate import ode
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import ConvexHull
from traits.api import HasPrivateTraits, Float, Property, Int, \
CArray, cached_property, Trait, Dict

from .internal import digest

f64ro = nb.types.Array(nb.types.float64,2,'C',readonly=True)
f32ro = nb.types.Array(nb.types.float32,2,'C',readonly=True)

@nb.njit([(f64ro, f64ro), (f64ro, f32ro), (f32ro, f64ro),(f32ro, f32ro)],
                cache=True, fastmath=True)
def dist_mat(gpos,mpos):
    """computes distance matrix, accelerated with numba

    Args:
        gpos (3,N)
        mpos (3,M)

    Returns:
        (N,M) distance matrix
    """    
    _,M = mpos.shape
    _,N = gpos.shape
    rm = empty((N,M),dtype=gpos.dtype)
    TWO = rm.dtype.type(2.0) # make sure to have a float32 or float 64 literal
    m0 = mpos[0]
    m1 = mpos[1]
    m2 = mpos[2]
    for n in range(N):
        g0 = gpos[0,n]
        g1 = gpos[1,n]
        g2 = gpos[2,n]
        for m in range(M):
            rm[n,m] = sqrt((g0 - m0[m])**TWO + (g1 - m1[m])**TWO + (g2 - m2[m])**TWO)
    return rm


def cartToCyl(x, Q=identity(3)):
    """
    Returns the cylindrical coordinate representation of a input position 
    which was before transformed into a modified cartesian coordinate, which
    has flow into positive z direction.
    
    Parameters
    ----------
    x : float[3, nPoints]
        cartesian coordinates of n points
    Q : float[3,3]
        Orthogonal transformation matrix. If provided, the pos vectors are
        transformed via posiMod = Q * x, before transforming those modified
        coordinates into cylindrical ones. Default is identity matrix.
        
    Returns
    -------
    cylCoord : [3, nPoints]
        cylindrical representation of those n points with (phi, r, z)
    """
    if not (Q == identity(3)).all():
        x = matmul(Q, x)  # modified position vector
    cylCoord = array([arctan2(x[1], x[0]), sqrt(x[0]**2 + x[1]**2), x[2]])
    return cylCoord


def cylToCart(x, Q=identity(3)):
        """
        Returns the cartesian coordinate representation of a input position 
        which was before transformed into a cylindrical coordinate, which
        has flow into positive z direction.
        
        Parameters
        ----------
        x : float[3, nPoints]
            cylindrical representation of those n points with (phi, r, z)
            cartesian coordinates of n points
    
        Q : float[3,3]
        Orthogonal transformation matrix. If provided, the pos vectors are
        transformed via posiMod = Q * x, before transforming those modified
        coordinates into cylindrical ones. Default is identity matrix.
        
            
        Returns
        -------
        CartCoord : [3, nPoints]
        cartesian coordinates of n points
            
        """
        if not (Q == identity(3)).all():
            x = matmul(Q, x)  # modified position vector
        CartCoord = array([x[1]*sin(x[0]),x[1]*cos(x[0]) , x[2]])
        return CartCoord


class Environment( HasPrivateTraits ):
    """
    A simple acoustic environment without flow.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations.
    """
    # internal identifier
    digest = Property(
        depends_on=['c'], 
        )

    #: The speed of sound, defaults to 343 m/s
    c = Float(343., 
        desc="speed of sound")

    #: The region of interest (ROI), not needed for most types of environment
    roi = Trait(None,(None,CArray))

    def _get_digest( self ):
        return digest( self )

    def _r( self, gpos, mpos=0.0):
        """
        Calculates distances between grid point locations and microphone
        locations or the origin. Functionality may change in the future.

        Parameters
        ----------
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates. 
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        r : array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a one-dimensional array is returned.
        """
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float64)[:, newaxis]
        rm = dist_mat(ascontiguousarray(gpos),ascontiguousarray(mpos))
#        mpos = mpos[:, newaxis, :]
#        rmv = gpos[:, :, newaxis]-mpos
#        rm = sum(rmv*rmv, 0)**0.5
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        return rm

class UniformFlowEnvironment( Environment):
    """
    An acoustic environment with uniform flow.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations in a uniform flow
    field.
    """
    #: The Mach number, defaults to 0.
    ma = Float(0.0, 
        desc="flow mach number")

    #: The unit vector that gives the direction of the flow, defaults to
    #: flow in x-direction.
    fdv = CArray( dtype=float64, shape=(3, ), value=array((1.0, 0, 0)), 
        desc="flow direction")

    # internal identifier
    digest = Property(
        depends_on=['c', 'ma', 'fdv'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def _r( self, gpos, mpos=0.0):
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound. Functionality may change in the future.

        Parameters
        ----------
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a one-dimensional array is returned.
        """
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
    An abstract base class for a spatial flow field.
    """
    digest = Property

    def _get_digest( self ):
        return ''

    def v( self, xx):
        """
        Provides the flow field as a function of the location. This is
        implemented here for the possibly most simple case: a quiescent fluid.

        Parameters
        ----------
        xx : array of floats of shape (3, )
            Location in the fluid for which to provide the data.

        Returns
        -------
        tuple with two elements
            The first element in the tuple is the velocity vector and the
            second is the Jacobian of the velocity vector field, both at the
            given location.
        """
        v = array((0., 0., 0.))
        dv = array(((0., 0., 0.), (0., 0., 0.), (0., 0., 0.)))
        return -v, -dv

class SlotJet( FlowField ):
    """
    Provides an analytical approximation of the flow field of a slot jet, 
    see :ref:`Albertson et al., 1950<Albertson1950>`.
    """
    #: Exit velocity at jet origin, i.e. the nozzle. Defaults to 0.
    v0 = Float(0.0, 
        desc="exit velocity")

    #: Location of a point at the slot center line, 
    #: defaults to the co-ordinate origin.
    origin = CArray( dtype=float64, shape=(3, ), value=array((0., 0., 0.)), 
        desc="center of nozzle")

    #: Unit flow direction of the slot jet, defaults to (1,0,0).
    flow = CArray( dtype=float64, shape=(3, ), value=array((1., 0., 0.)), 
        desc="flow direction")

    #: Unit vector parallel to slot center plane, defaults to (0,1,0).
    plane = CArray( dtype=float64, shape=(3, ), value=array((0., 1., 0.)), 
        desc="slot center line direction")
        
    #: Width of the slot, defaults to 0.2 .
    B = Float(0.2, 
        desc="nozzle diameter")

    # internal identifier
    digest = Property(
        depends_on=['v0', 'origin', 'flow', 'plane', 'B'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def v( self, xx):
        """
        Provides the flow field as a function of the location. This is
        implemented here only for the component in the direction of :attr:`flow`;
        entrainment components are set to zero.

        Parameters
        ----------
        xx : array of floats of shape (3, )
            Location in the fluid for which to provide the data.

        Returns
        -------
        tuple with two elements
            The first element in the tuple is the velocity vector and the
            second is the Jacobian of the velocity vector field, both at the
            given location.
        """
        # TODO: better to make sure that self.flow and self.plane are indeed unit vectors before
        # normalize
        flow = self.flow/norm(self.flow)
        plane = self.plane/norm(self.plane)
        # additional axes of global co-ordinate system
        yy = -cross(flow,plane)
        zz = cross(flow,yy)
        # distance from slot exit plane
        xx1 = xx-self.origin
        # local co-ordinate system 
        x = dot(flow,xx1)
        y = dot(yy,xx1)
        x1 = 0.109*x
        h1 = abs(y)+sqrt(pi)*0.5*x1-0.5*self.B
        if h1 < 0.0:
            # core jet
            Ux = self.v0
            Udx = 0
            Udy = 0
        else:
            # shear layer
            Ux = self.v0*exp(-h1*h1/(2*x1*x1))
            Udx = (h1*h1/(x*x1*x1)-sqrt(pi)*0.5*h1/(x*x1))*Ux
            Udy = -sign(y)*h1*Ux/(x1*x1)
        # Jacobi matrix
        dU = array(((Udx,0,0),(Udy,0,0),(0,0,0))).T
        # rotation matrix
        R = array((flow,yy,zz)).T
        return dot(R,array((Ux,0,0))), dot(dot(R,dU),R.T)

class OpenJet( FlowField ):
    """
    Provides an analytical approximation of the flow field of an open jet, 
    see :ref:`Albertson et al., 1950<Albertson1950>`.

    Notes
    -----
    This is not a fully generic implementation and is limited to flow in the
    x-direction only. No other directions are possible at the moment and flow
    components in the other direction are zero.
    """
    #: Exit velocity at jet origin, i.e. the nozzle. Defaults to 0.
    v0 = Float(0.0, 
        desc="exit velocity")

    #: Location of the nozzle center, defaults to the co-ordinate origin.
    origin = CArray( dtype=float64, shape=(3, ), value=array((0., 0., 0.)), 
        desc="center of nozzle")

    #: Diameter of the nozzle, defaults to 0.2 .
    D = Float(0.2, 
        desc="nozzle diameter")

    # internal identifier
    digest = Property(
        depends_on=['v0', 'origin', 'D'], 
        )

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def v( self, xx):
        """
        Provides the flow field as a function of the location. This is
        implemented here only for a jet in `x`-direction and the `y`- and
        `z`-components are set to zero.

        Parameters
        ----------
        xx : array of floats of shape (3, )
            Location in the fluid for which to provide the data.

        Returns
        -------
        tuple with two elements
            The first element in the tuple is the velocity vector and the
            second is the Jacobian of the velocity vector field, both at the
            given location.
        """
        x, y, z = xx-self.origin
        r = sqrt(y*y+z*z)
        x1 = 0.081*x
        h1 = r+x1-0.5*self.D
        U = self.v0*exp(-h1*h1/(2*x1*x1))
        if h1 < 0.0:
            Udr = 0.0
            U = self.v0
        else:
            Udr = -h1*U/(x1*x1)
        if r > 0.0:
            Udy = y*Udr/r
            Udz = z*Udr/r
        else:
            Udy = Udz = 0.0
        Udx = (h1*h1/(x*x1*x1)-h1/(x*x1))*U
        if h1 < 0.0:
            Udx = 0

        # flow field
        v = array( (U, 0., 0.) )
        # Jacobi matrix
        dv = array( ((Udx, 0., 0.), (Udy, 0., 0.), (Udz, 0., 0.)) ).T
        return v, dv




class RotatingFlow( FlowField ):
    """
    Provides an analytical approximation of the flow field of a rotating fluid with constant flow. 


    """
    #: Exit velocity at jet origin, i.e. the nozzle. Defaults to 0.
    rpm = Float(0.0,
        desc="revolutions per minute of the virtual array; negative values for clockwise rotation")

    v0 = Float(0.0, 
        desc="flow velocity")

    #: Location of the nozzle center, defaults to the co-ordinate origin.
    origin = CArray( dtype=float64, shape=(3, ), value=array((0., 0., 0.)), 
        desc="center of nozzle")

    # internal identifier
    digest = Property(
        depends_on=['v0', 'origin', 'rpm'], 
        )

    # internal identifier
    omega = Property(
        depends_on=['rpm'], 
        )

    @cached_property
    def _get_omega(self):
        return 2 * pi * self.rpm / 60

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def v( self, xx):
        """
        Provides the rotating flow field around the z-Axis as a function of the location.

        Parameters
        ----------
        xx : array of floats of shape (3, )
            Location in the fluid for which to provide the data.

        Returns
        -------
        tuple with two elements
            The first element in the tuple is the velocity vector and the
            second is the Jacobian of the velocity vector field, both at the
            given location.
        """
        x, y, z = xx-self.origin

        # rotational speed
        omega = self.omega

        #velocity vector
        U = omega * y
        V = -omega * x
        W = self.v0 
        
        # flow field
        v = array( (U, V, W) )
        # Jacobi matrix
        dv = array( ((0, -omega, 0.), (omega, 0, 0.), (0., 0., 0.)) ).T
        return v, dv




def spiral_sphere(N, Om=2*pi, b=array((0, 0, 1))):    #change to 4*pi
    """
    Internal helper function for the raycasting that returns an array of
    unit vectors (N, 3) giving equally distributed directions on a part of
    sphere given by the center direction b and the solid angle Om.
    """
    # first produce 'equally' distributed directions in spherical coords
    o = 4*pi/Om 
    h = -1+ 2*arange(N)/(N*o-1.)
    theta = arccos(h)
    phi = zeros_like(theta)
    for i, hk in enumerate(h[1:]):
        phi[i+1] = phi[i]+3.6/sqrt(N*o*(1-hk*hk)) % (2*pi)
    # translate to cartesian coords
    xyz = vstack((sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)))
    # mirror everything on a plane so that b points into the center
    a = xyz[:, 0]
    b = b/norm(b)
    ab = (a-b)[:, newaxis]
    if norm(ab)<1e-10:
        return xyz
    # this is the Householder matrix for mirroring
    H = identity(3)-dot(ab, ab.T)/dot(ab.T, a)
    # actual mirroring
    return dot(H, xyz)

class GeneralFlowEnvironment(Environment):
    """
    An acoustic environment with a generic flow field.

    This class provides the facilities to calculate the travel time (distances)
    between grid point locations and microphone locations in a generic flow
    field with non-uniform velocities that depend on the location. The
    algorithm for the calculation uses a ray-tracing approach that bases on
    rays cast from every microphone position in multiple directions and traced
    backwards in time. The result is interpolated within a tetrahedal grid
    spanned between these rays.
    """
    #: The flow field, must be of type :class:`~acoular.environments.FlowField`.
    ff = Trait(FlowField, 
        desc="flow field")

    #: Number of rays used per solid angle :math:`\Omega`, defaults to 200.
    N = Int(200, 
        desc="number of rays per Om")

    #: The maximum solid angle used in the algorithm, defaults to :math:`\pi`.
    Om = Float(pi, 
        desc="maximum solid angle")

    # internal identifier
    digest = Property(
        depends_on=['c', 'ff.digest', 'N', 'Om'], 
        )

    # internal dictionary of interpolators
    idict = Dict

    @cached_property
    def _get_digest( self ):
        return digest( self )

    def _r( self, gpos, mpos=0.0):
        """
        Calculates the virtual distances between grid point locations and
        microphone locations or the origin. These virtual distances correspond
        to travel times of the sound along a ray that is traced through the
        medium. Functionality may change in the future.

        Parameters
        ----------
        gpos : array of floats of shape (3, N)
            The locations of points in the beamforming map grid in 3D cartesian
            co-ordinates.
        mpos : array of floats of shape (3, M), optional
            The locations of microphones in 3D cartesian co-ordinates. If not
            given, then only one microphone at the origin (0, 0, 0) is
            considered.

        Returns
        -------
        array of floats
            The distances in a twodimensional (N, M) array of floats. If M==1, 
            then only a one-dimensional array is returned.
        """
        c = self.c
        
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype = float32)[:, newaxis]

        gt = empty((gpos.shape[-1], mpos.shape[-1]))
        for micnum, x0 in enumerate(mpos.T):
            key = x0.tobytes() # make array hashable
            #todo: the interpolator also depends the roi, so idict keys should also depend on roi
            # OR the idict should be cleaned if roi changes
            try:
                li = self.idict[key] # fetch stored interpolator
            except KeyError:
                # if interpolator doesn't exist, construct it
                roi = gpos
                if self.roi is not None:
                    roi = self.roi
                li = self.get_interpolator(roi, x0)
                self.idict[key] = li
            # do the interpolation
            gt[:, micnum] = li(gpos.T)
        if gt.shape[1] == 1:
            gt = gt[:, 0]
        return c*gt #return distance along ray

    def get_interpolator( self, roi, x0 ):
        """
        gets an LinearNDInterpolator object

        Parameters
        ----------
        roi : array of floats of shape (3, N)
            The locations of points in the region of interest in 3D cartesian
            co-ordinates. Used to estimate the maximum distance and ROI
            extension and center.
        x0 : array of floats of shape (3)
            The location of the microphone in 3D cartesian co-ordinates. 

        Returns
        -------
        LinearNDInterpolator object
        """
        c = self.c

        # the DE system
        def f1(t, y, v):
            x = y[0:3]
            s = y[3:6]
            vv, dv = v(x)
            sa = sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2])
            x = empty(6)
            x[0:3] = c*s/sa - vv # time reversal
            x[3:6] = dot(s, -dv.T) # time reversal
            return x

        # integration along a single ray
        def fr(x0, n0, rmax, dt, v, xyz, t):
            s0 = n0 / (c+dot(v(x0)[0], n0))
            y0 = hstack((x0, s0))
            oo = ode(f1)
            oo.set_f_params(v)
            oo.set_integrator('vode', 
                              rtol=1e-4, # accuracy !
                              max_step=1e-4*rmax) # for thin shear layer
            oo.set_initial_value(y0, 0)
            while oo.successful():
                xyz.append(oo.y[0:3])
                t.append(oo.t)
                if norm(oo.y[0:3]-x0)>rmax:
                    break
                oo.integrate(oo.t+dt)

        gs2 = roi.shape[-1]
        vv = self.ff.v
        NN = int(sqrt(self.N))
        xe = roi.mean(1) # center of grid
        r = x0[:, newaxis]-roi
        rmax = sqrt((r*r).sum(0).max()) # maximum distance
        nv = spiral_sphere(self.N, self.Om, b=xe-x0)
        rstep = rmax/sqrt(self.N)
        rmax += rstep
        tstep = rstep/c
        xyz = []
        t = []
        lastind = 0
        for i, n0 in enumerate(nv.T):
            fr(x0, n0, rmax, tstep, vv, xyz, t)
            if i and i % NN == 0:
                if not lastind:
                    dd = ConvexHull(vstack((roi.T, xyz)), incremental=True)
                else:
                    dd.add_points(xyz[lastind:], restart=True)
                lastind = len(xyz)
                # ConvexHull includes grid if no grid points on hull
                if dd.simplices.min()>=gs2:
                    break
        xyz = array(xyz)
        t = array(t)
        return LinearNDInterpolator(xyz, t)


