# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implements acoustic environments with and without flow.

.. autosummary::
    :toctree: generated/

    Environment
    UniformFlowEnvironment
    GeneralFlowEnvironment
    FlowField
    OpenJet
    RotatingFlow
    SlotJet
    dist_mat
    cylToCart
    cartToCyl
    spiral_sphere
"""

from abc import abstractmethod
from warnings import warn

import numba as nb
from numpy import (
    arange,
    arccos,
    arctan2,
    array,
    ascontiguousarray,
    cos,
    cross,
    dot,
    empty,
    exp,
    float32,
    float64,
    hstack,
    identity,
    isscalar,
    matmul,
    newaxis,
    pi,
    sign,
    sin,
    sqrt,
    sum,  # noqa: A004
    vstack,
    zeros_like,
)
from scipy.integrate import ode
from scipy.interpolate import LinearNDInterpolator
from scipy.linalg import norm
from scipy.spatial import ConvexHull
from traits.api import (
    ABCHasStrictTraits,
    CArray,
    Dict,
    Float,
    HasStrictTraits,
    Instance,
    Int,
    Property,
    Union,
    cached_property,
)

from .internal import digest

f64ro = nb.types.Array(nb.types.float64, 2, 'C', readonly=True)
f32ro = nb.types.Array(nb.types.float32, 2, 'C', readonly=True)


@nb.njit([(f64ro, f64ro), (f64ro, f32ro), (f32ro, f64ro), (f32ro, f32ro)], cache=True, fastmath=True)
def dist_mat(gpos, mpos):  # pragma: no cover
    """
    Compute distance matrix. (accelerated with numba).

    Given an `(3, N)` array of the locations of points in the beamforming map grid in 3D cartesian
    coordinates and `(3, M)` array of the locations of microphones in 3D cartesian coordinates, the
    `(N, M)` matrix of the distances between each microphone and each point in the beamforming map
    grip.

    Parameters
    ----------
    gpos : :class:`numpy.ndarray` of :class:`floats<float>`
        The locations of `N` points in the beamforming map grid in 3D cartesian coordinates,
        shape `(3, N)`.

    mpos : :class:`numpy.ndarray` of :class:`floats<float>`
        The locations of `M` microphones in 3D cartesian coordinates, shape `(3, M)`.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`floats<float>`
        Matrix of the distances between each microphone and each point in the beamforming map grid,
        shape `(N, M)`.
    """
    _, M = mpos.shape
    _, N = gpos.shape
    rm = empty((N, M), dtype=gpos.dtype)
    TWO = rm.dtype.type(2.0)  # make sure to have a float32 or float 64 literal
    m0 = mpos[0]
    m1 = mpos[1]
    m2 = mpos[2]
    for n in range(N):
        g0 = gpos[0, n]
        g1 = gpos[1, n]
        g2 = gpos[2, n]
        for m in range(M):
            rm[n, m] = sqrt((g0 - m0[m]) ** TWO + (g1 - m1[m]) ** TWO + (g2 - m2[m]) ** TWO)
    return rm


def cartToCyl(x, Q=None):  # noqa: N802, N803
    r"""
    Return cylindrical coordinate representation of an input array in cartesian coordinate.

    Return the cylindrical coordinate representation of an input position which was before
    transformed into a modified cartesian coordinate, which has flow into positive z direction.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of :class:`floats<float>`
        Cartesian coordinates of `N` points, shape `(3, N)`.

    Q : :class:`numpy.ndarray` of :class:`floats<float>`, optional
        Orthogonal transformation matrix, shape `(3, 3)`. If provided, the positional vectors are
        transformed via ``new_x = Q * x``, before transforming those modified coordinates into
        cylindrical ones. Default is the identity matrix.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`floats<float>`
        Cylindrical representation of given `N` points in cartesian coodrinates as
        an array of shape `(3, N)` with new coordinates :math:`(\phi, r, z)`.
    """
    Q = identity(3) if Q is None else Q
    if not (Q == identity(3)).all():  # noqa: SIM300
        x = matmul(Q, x)  # modified position vector
    return array([arctan2(x[1], x[0]), sqrt(x[0] ** 2 + x[1] ** 2), x[2]])


def cylToCart(x, Q=None):  # noqa: N802, N803
    r"""
    Return cartesian coordinate representation of an input array in cylindrical coordinate.

    Return the cartesian coordinate representation of a input position which was before transformed
    into a cylindrical coordinate, which has flow into positive z direction.

    Parameters
    ----------
    x : :class:`numpy.ndarray` of :class:`floats<float>`
        Cylindrical coordinates of `N` points, shape `(3, N)`.

    Q : :class:`numpy.ndarray` of :class:`floats<float>`, optional
        Orthogonal transformation matrix, shape `(3, 3)`. If provided, the positional vectors are
        transformed via ``new_x = Q * x`` before transforming those modified coordinates into
        cartesian ones. Default is the identity matrix.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`floats<float>`
        Cartesian representation of given `N` points in cylindrical coodrinates as
        an array of shape `(3, N)` with coodinates :math:`(x, y, z)`.
    """
    Q = identity(3) if Q is None else Q
    if not (Q == identity(3)).all():  # noqa: SIM300
        x = matmul(Q, x)  # modified position vector
    return array([x[1] * sin(x[0]), x[1] * cos(x[0]), x[2]])


class Environment(HasStrictTraits):
    """
    A simple acoustic environment without flow.

    This class models an acoustic environment where the propagation of sound is considered to occur
    in a homogeneous medium without any flow effects (e.g., wind). It provides functionality for
    computing the travel time or distances between grid point locations and microphone locations.

    Notes
    -----
    - The :func:`dist_mat` function is used internally to compute the pairwise distances
      between grid points and microphone positions efficiently.
    - This class assumes a static, homogeneous environment without accounting
      for factors like temperature gradients, humidity, or atmospheric turbulence.
    """

    #: A unique identifier based on the environment properties. (read-only)
    digest = Property(depends_on=['c'])

    #: The speed of sound in the environment. Default is ``343.0``, which corresponds to the
    #: approximate speed of sound at 20°C in dry air at sea level, if the unit is m/s.
    c = Float(343.0, desc='speed of sound')

    #: The region of interest (ROI) for calculations. (Not needed for most types of environment.)
    #: Default is :obj:`None`.
    roi = Union(None, CArray)

    def _get_digest(self):
        return digest(self)

    def _r(self, gpos, mpos=0.0):
        # Compute the distance between two sets of points.
        #
        # The distance for each of the `N` points in ``gpos`` in 3-D space to each of the `M` points
        # in ``mpos``.
        #
        # See Also
        # --------
        # :func:`dist_mat`: Compute distance matrix.
        #
        # Parameters
        # ----------
        # gpos : :class:`numpy.ndarray` of :class:`floats<float>`
        #     The coordinates of the first set of points. Should be of shape `(N, 3)`,
        #     where `N` is the number of points.
        #
        # mpos : :class:`float` or :class:`numpy.ndarray` of :class:`floats<float>`, optional
        #     The coordinates of the second set of points. If a scalar is provided,
        #     it is treated as the origin ``(0, 0, 0)``. If an array is given,
        #     it should have shape `(M, 3)`, where `M` is the number of points.
        #
        # Returns
        # -------
        # rm : :class:`numpy.ndarray` of :class:`floats<float>`
        #     The distances between each point in ``gpos`` and ``mpos``.
        #     The result is an array of
        #
        #         - shape `(N,)` if ``mpos`` is a single point, or
        #         - shape `(N, M)` if ``mpos`` consists of multiple points.
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype=float64)[:, newaxis]
        rm = dist_mat(ascontiguousarray(gpos), ascontiguousarray(mpos))
        #        mpos = mpos[:, newaxis, :]
        #        rmv = gpos[:, :, newaxis]-mpos
        #        rm = sum(rmv*rmv, 0)**0.5
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        return rm


class UniformFlowEnvironment(Environment):
    """
    An acoustic environment with uniform flow.

    This class models an acoustic environment where sound propagates in a medium with uniform flow.
    It extends the :class:`Environment` class to account for the effects of flow on sound
    propagation, such as changes in travel times and distances due to advection by the flow field.

    The flow is assumed to be uniform and steady, characterized by its Mach number (:attr:`ma`)
    and direction (:attr:`fdv`).

    Notes
    -----
    The effective distance is adjusted by solving a flow-dependent relationship that accounts
    for the cosine of the angle between the flow direction and the propagation path.
    """

    #: The Mach number of the flow, defined as the ratio of the flow velocity to the speed of sound.
    #: Default is ``0.0``, which corresponds to no flow.
    ma = Float(0.0, desc='flow mach number')

    #: A unit vector specifying the direction of the flow in 3D Cartesian coordinates.
    #: Default is ``(1.0, 0, 0)``, which corresponds to flow in the x-direction.
    fdv = CArray(dtype=float64, shape=(3,), value=array((1.0, 0, 0)), desc='flow direction')

    #: A unique identifier based on the environment properties. (read-only)
    digest = Property(
        depends_on=['c', 'ma', 'fdv'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _r(self, gpos, mpos=0.0):
        # Compute the distance between two sets of points.
        #
        # This method calculates the distance between points ``gpos`` and ``mpos`` in a 3-D space,
        # with an additional adjustment based on a mass term and force vector. The result is
        # affected by the angle between the direction of movement and the force vector.
        #
        # Parameters
        # ----------
        # gpos : :class:`numpy.ndarray` of :class:`floats<float>`
        #     The 3-D coordinates of the first set of points, shape `(N, 3)`.
        #
        # mpos : :class:`float` or :class:`numpy.ndarray` of :class:`floats<float>`, optional
        #     The 3-D coordinates of the second set of points. If a scalar is provided, it is
        #     treated as the origin ``(0, 0, 0)``. If an array is given, it should have shape
        #     `(M, 3)`, where `M` is the number of points.
        #
        # Returns
        # -------
        # rm : :class:`numpy.ndarray` of :class:`floats<float>`
        #     The distances between each point in ``gpos`` and ``mpos``.
        #     The result is an array of
        #
        #         - shape `(N,)` if ``mpos`` is a single point, or
        #         - shape `(N, M)` if ``mpos`` consists of multiple points.
        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype=float32)[:, newaxis]
        fdv = self.fdv / sqrt((self.fdv * self.fdv).sum())
        mpos = mpos[:, newaxis, :]
        rmv = gpos[:, :, newaxis] - mpos
        rm = sqrt(sum(rmv * rmv, 0))
        macostheta = (self.ma * sum(rmv.reshape((3, -1)) * fdv[:, newaxis], 0) / rm.reshape(-1)).reshape(rm.shape)
        rm *= 1 / (-macostheta + sqrt(macostheta * macostheta - self.ma * self.ma + 1))
        if rm.shape[1] == 1:
            rm = rm[:, 0]
        return rm


class FlowField(ABCHasStrictTraits):
    """An abstract base class for a spatial flow field."""

    #: A unique identifier based on the field properties. (read-only)
    digest = Property

    @abstractmethod
    def _get_digest(self):
        pass

    @abstractmethod
    def v(self, xx):  # noqa: ARG002
        """
        Provide the flow field as a function of the location.

        Parameters
        ----------
        xx : :class:`numpy.ndarray` of :class:`floats<float>`
            Location in the fluid for which to provide the data, shape (3,).
        """


class SlotJet(FlowField):
    """
    Analytical approximation of the flow field of a slot jet.

    This class provides an analytical model of the velocity field for a slot jet,
    based on the description in :cite:`Albertson1950`. It describes the flow field
    originating from a slot nozzle, including the jet core and shear layer, and
    calculates the velocity field and its Jacobian matrix at a given location.

    Notes
    -----
    - The slot jet is divided into two regions: the jet core and the shear layer. The model
      distinguishes between these regions based on the non-dimensionalized distance from
      the slot exit plane.
    - The flow field is aligned with the direction of the :attr:`flow` vector,
      while the :attr:`plane` vector helps define the orientation of the slot.
    """

    #: Exit velocity at the :attr:`slot jet origin<origin>` (nozzle). Default is ``0.0``.
    v0 = Float(0.0, desc='exit velocity')

    #: The location of the slot nozzle center. Default is ``(0.0, 0.0, 0.0)``.
    origin = CArray(dtype=float64, shape=(3,), value=array((0.0, 0.0, 0.0)), desc='center of nozzle')

    #: Unit vector representing the flow direction. Default is ``(1.0, 0.0, 0.0)``.
    flow = CArray(dtype=float64, shape=(3,), value=array((1.0, 0.0, 0.0)), desc='flow direction')

    #: Unit vector parallel to the slot center plane, used to define the slot orientation.
    #: Default is ``(0.0, 1.0, 0.0)``.
    plane = CArray(dtype=float64, shape=(3,), value=array((0.0, 1.0, 0.0)), desc='slot center line direction')

    #: Width of the slot (slot diameter). Default is ``0.2``.
    B = Float(0.2, desc='nozzle diameter')

    #: Non-dimensional length of the zone of flow establishment (jet core length).
    #: Default is ``5.2``.
    l = Float(5.2, desc='flow establishment length')  # noqa: E741

    #: A unique identifier based on the field properties. (read-only)
    digest = Property(
        depends_on=['v0', 'origin', 'flow', 'plane', 'B', 'l'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def v(self, xx):
        """
        Compute the velocity field and its Jacobian matrix at a given location.

        This method provides the velocity vector and its Jacobian matrix at the given location
        ``xx`` in the fluid. The velocity is computed for the flow component in the direction
        of the slot jet, while the entrainment components are assumed to be zero.

        Parameters
        ----------
        xx : :class:`numpy.ndarray` of :class:`floats<float>`
            The 3D Cartesian coordinates of the location in the fluid where the velocity field
            is to be computed, shape `(3,)`.

        Returns
        -------
        velocity_vector : :class:`numpy.ndarray` of :class:`floats<float>`
            The velocity vector at the given location, shape `(3,)`.
        jacobian_matrix : :class:`numpy.ndarray` of :class:`floats<float>`
            The Jacobian matrix of the velocity vector field at the given location, shape `(3,)`.

        Notes
        -----
        - The velocity field is computed using a local coordinate system aligned with the flow
          direction and the slot orientation.
        - The velocity profile depends on whether the point lies within the jet core region or
          in the shear layer. In the jet core, the velocity is constant, while in the shear layer,
          it decays following a Gaussian distribution.
        - The Jacobian matrix provides the partial derivatives of the velocity vector components
          with respect to the spatial coordinates.
        """
        # normalize
        flow = self.flow / norm(self.flow)
        plane = self.plane / norm(self.plane)
        # additional axes of global coordinate system
        yy = -cross(flow, plane)
        zz = cross(flow, yy)
        # distance from slot exit plane
        xx1 = xx - self.origin
        # local coordinate system
        x = dot(flow, xx1)
        y = dot(yy, xx1)
        x1 = 0.5668 / self.l * x  # C1 in Albertson1950
        h1 = abs(y) + sqrt(pi) * 0.5 * x1 - 0.5 * self.B
        if h1 < 0.0:
            # core jet
            Ux = self.v0
            Udx = 0
            Udy = 0
        else:
            # shear layer
            Ux = self.v0 * exp(-h1 * h1 / (2 * x1 * x1))
            Udx = (h1 * h1 / (x * x1 * x1) - sqrt(pi) * 0.5 * h1 / (x * x1)) * Ux
            Udy = -sign(y) * h1 * Ux / (x1 * x1)
        # Jacobi matrix
        dU = array(((Udx, 0, 0), (Udy, 0, 0), (0, 0, 0))).T
        # rotation matrix
        R = array((flow, yy, zz)).T
        return dot(R, array((Ux, 0, 0))), dot(dot(R, dU), R.T)


class OpenJet(FlowField):
    """
    Analytical approximation of the flow field of an open jet.

    This class provides a simplified analytical model of the velocity field for an open jet, based
    on the description in :cite:`Albertson1950`. It calculates the velocity vector and its
    Jacobian matrix at a given location in the fluid domain, assuming flow in the x-direction only.

    Notes
    -----
    - This is not a fully generic implementation, and is limited to flow in the x-direction only.
      No other directions are possible at the moment and flow components in
      the other direction are zero.
    - The flow field transitions from the jet core to the shear layer, with velocity decay
      modeled using a Gaussian profile in the shear layer.

    Examples
    --------
    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> jet = ac.OpenJet(v0=10.0, D=0.4, l=6.2)
    >>> velocity, jacobian = jet.v(np.array((1.0, 0.1, 0.1)))
    >>> velocity
    array([9.62413564, 0.        , 0.        ])
    >>> jacobian
    array([[ -1.92660591, -23.25619062, -23.25619062],
           [  0.        ,   0.        ,   0.        ],
           [  0.        ,   0.        ,   0.        ]])
    """

    #: Exit velocity at the jet origin (nozzle). Default is ``0.0``.
    v0 = Float(0.0, desc='exit velocity')

    #: The location of the nozzle center. Default is ``(0.0, 0.0, 0.0)``.
    origin = CArray(dtype=float64, shape=(3,), value=array((0.0, 0.0, 0.0)), desc='center of nozzle')

    #: Diameter of the nozzle. Default is ``0.2``.
    D = Float(0.2, desc='nozzle diameter')

    #: Non-dimensional length of the zone of flow establishment (jet core length).
    #: Default is ``6.2``. :cite:`Albertson1950`
    l = Float(6.2, desc='flow establishment length')  # noqa: E741

    #: A unique identifier based on the field properties. (read-only)
    digest = Property(
        depends_on=['v0', 'origin', 'D', 'l'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def v(self, xx):
        """
        Compute the velocity field and its Jacobian matrix at a given location.

        This method calculates the velocity vector and its Jacobian matrix at the
        specified location ``xx`` in the fluid domain. The velocity is modeled only for the
        x-component of the flow, while the y- and z-components are assumed to be zero.

        Parameters
        ----------
        xx : :class:`numpy.ndarray` of :class:`floats<float>`
            The 3D Cartesian coordinates of the location in the fluid where the velocity
            field is to be computed, shape `(3,)`.

        Returns
        -------
        velocity_vector : :class:`numpy.ndarray` of :class:`floats<float>`
            The velocity vector at the specified location ``xx``, shape `(3,)`.
        jacobian_matrix : :class:`numpy.ndarray` of :class:`floats<float>`
            The Jacobian matrix of the velocity vector field at the specified location ``xx``,
            shape `(3, 3)`.

        Notes
        -----
        - The velocity field is determined based on whether the location is within the jet core
          region or in the shear layer. Within the jet core, the velocity is constant and equal
          to :attr:`v0`. In the shear layer, the velocity decays following a Gaussian distribution.
        - The Jacobian matrix provides the partial derivatives of the velocity components
          with respect to the spatial coordinates.
        - If the radial distance `r` from the jet axis is zero, the derivatives with respect
          to `y` and `z` are set to zero to avoid division by zero.
        """
        x, y, z = xx - self.origin
        r = sqrt(y * y + z * z)
        x1 = 0.5022 / self.l * x  # C2 in Albertson1950
        h1 = r + x1 - 0.5 * self.D
        U = self.v0 * exp(-h1 * h1 / (2 * x1 * x1))
        if h1 < 0.0:
            Udr = 0.0
            U = self.v0
        else:
            Udr = -h1 * U / (x1 * x1)
        if r > 0.0:
            Udy = y * Udr / r
            Udz = z * Udr / r
        else:
            Udy = Udz = 0.0
        Udx = (h1 * h1 / (x * x1 * x1) - h1 / (x * x1)) * U
        if h1 < 0.0:
            Udx = 0

        # flow field
        v = array((U, 0.0, 0.0))
        # Jacobi matrix
        dv = array(((Udx, 0.0, 0.0), (Udy, 0.0, 0.0), (Udz, 0.0, 0.0))).T
        return v, dv


class RotatingFlow(FlowField):
    """
    Analytical approximation of a rotating flow field with additional velocity component
    in z-direction.

    This class provides an analytical model for a fluid flow field with a
    rigid-body-like rotation about the z-axis. The flow combines rotational motion
    in the x-y plane and a constant velocity component in the z-direction.

    Notes
    -----
    - The rotation is assumed to be about the z-axis. The velocity components in the x-y plane are
      determined by the angular velocity :attr:`omega`, while the z-component is constant
      and set by :attr:`v0`.
    - The angular velocity :attr:`omega` is computed as: ``omega = 2 * pi * rps``,
      with the :attr:`rps` given in revolutions per second (i.e. Hz).

    Examples
    --------
    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> flow = RotatingFlow(rps=1, v0=1.0)
    >>> velocity, jacobian = flow.v(array((1.0, 1.0, 0.0)))
    >>> velocity
    array([-6.28318531,  6.28318531,  1.        ])
    >>> jacobian
    array([[ 0.        , -6.28318531,  0.        ],
           [ 6.28318531,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ]])
    """

    # Revolutions per minute (RPM). Default is ``0.0``.
    # Positive values indicate clockwise rotation of the flow.
    # This is contrary to the usual definition of the direction of rotation.
    # Deprecated! Please use the differently defined :attr:`rps` attribute instead.
    rpm = Property(desc='revolutions per minute of the flow; positive values for clockwise rotation')

    def _get_rpm(self):
        warn(
            'Deprecated use of "rpm" trait. Please use the "rps" trait instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        return -60 * self.rps

    def _set_rpm(self, rpm):
        warn(
            'Deprecated use of "rpm" trait. Please use the "rps" trait instead (divide rpm value by -60).',
            DeprecationWarning,
            stacklevel=2,
        )
        self.rps = -rpm / 60

    #: Rotational speed in revolutions per second. Negative values indicate clockwise
    #: rigid-body-like rotation of the flow. Default is ``0.0``.
    rps = Float(0.0, desc='rotational speed of the flow in Hz')

    #: Constant flow velocity in the z-direction. Default is ``0.0``.
    v0 = Float(0.0, desc='flow velocity')

    #: The location of the center of rotation.
    #: Default is ``(0.0, 0.0, 0.0)``.
    origin = CArray(dtype=float64, shape=(3,), value=array((0.0, 0.0, 0.0)), desc='center of rotation')

    #: A unique identifier based on the field properties. (read-only)
    digest = Property(
        depends_on=['v0', 'origin', 'rps'],
    )

    #: Angular velocity (in radians per second) of the rotation.
    #: This is a derived property based on :attr:`rps`.
    omega = Property(
        depends_on=['rps'],
    )

    @cached_property
    def _get_omega(self):
        return 2 * pi * self.rps

    @cached_property
    def _get_digest(self):
        return digest(self)

    def v(self, xx):
        """
        Compute the rotating flow field and its Jacobian matrix at a given location.

        This method calculates the velocity vector and its Jacobian matrix at the specified location
        ``xx`` in the fluid domain. The flow field consists of rotational components in the
        x-y plane and a constant velocity component in the z-direction.

        Parameters
        ----------
        xx : :class:`numpy.ndarray` of :class:`floats<float>`
            The 3D Cartesian coordinates of the location in the fluid where the velocity
            field is to be computed, shape `(3,)`.

        Returns
        -------
        velocity_vector : :class:`numpy.ndarray` of :class:`floats<float>`
            The velocity vector at the specified location ``xx``, shape `(3,)`. The components are:
                - U: Velocity in the x-direction (dependent on y-coordinate and :attr:`omega`).
                - V: Velocity in the y-direction (dependent on x-coordinate and :attr:`omega`).
                - W: Constant velocity in the z-direction (set by :attr:`v0`).
        jacobian_matrix : :class:`numpy.ndarray` of :class:`floats<float>`
            The Jacobian matrix of the velocity vector field at the specified location ``xx``.
            The matrix contains partial derivatives of each velocity component with
            respect to the spatial coordinates :math:`(x, y, z)`, shape `(3, 3)`.

        Notes
        -----
        The Jacobian matrix is constant for this flow field and represents the linear relationship
        between the velocity components and spatial coordinates in the x-y plane.
        """
        x, y, z = xx - self.origin

        # angular velocity
        omega = self.omega

        # velocity vector
        U = omega * -y
        V = omega * x
        W = self.v0

        # flow field
        v = array((U, V, W))
        # Jacobi matrix
        dv = array(((0.0, omega, 0.0), (-omega, 0.0, 0.0), (0.0, 0.0, 0.0))).T
        return v, dv


def spiral_sphere(N, Om=None, b=None):  # noqa: N803 # change to 4*pi
    """
    Generate unit vectors equally distributed over a sphere or a portion of it.

    Internal helper function for the raycasting that returns an array of unit
    vectors of shape `(N, 3)` giving equally distributed directions on a part
    of sphere given by the center direction ``b`` and the solid angle ``Om``.

    The function uses spherical coordinates to distribute the points, the converts them to Cartesian
    coordinates. It also applies a transformation to reflect the points about a plane so that the
    direction defined by the vector ``b`` points toward the center of the sphere.

    Parameters
    ----------
    N : :class:`int`
        The number of points to generate on the sphere.

    Om : :class:`float`, optional
        The solid angle in steradians to cover on the sphere. Default is ``2 * pi``,
        which corresponds to a hemisphere. Smaller values result in covering
        a smaller portion of the hemisphere.

    b : :class:`numpy.ndarray` of :class:`floats<float>`, optional
        A 3D unit vector specifying the desired center direction of the distribution.
        Points are mirrored such that this vector points toward the center of the sphere.
        Default is ``[0, 0, 1]``, which corresponds to the z-axis.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`floats<float>`
        An array of unit vectors representing points on the sphere, shape `(3, N)`.
        Each column corresponds to a 3D Cartesian coordinate of a point.

    Notes
    -----
    - The points are initially distributed using a spiral pattern in spherical coordinates.
      This ensures an approximately equal spacing between points over the specified portion
      of the sphere.
    - If a vector ``b`` is provided, the function mirrors the distribution using a
      Householder reflection so that ``b`` points toward the center.
    - The function avoids generating singularities at the poles by adjusting the spiral
      distribution formula.

    Examples
    --------
    Generate 100 points over a hemisphere:

    >>> from acoular.environments import spiral_sphere
    >>> points = spiral_sphere(100)
    >>> points.shape
    (3, 100)

    Generate 50 points over half a hemisphere with the z-axis as the center direction:

    >>> import numpy as np
    >>> points = spiral_sphere(50, Om=np.pi, b=array((0, 0, 1)))
    >>> points.shape
    (3, 50)

    Generate 200 points with a different direction vector:

    >>> points = spiral_sphere(200, b=array((1, 0, 0)))
    >>> points.shape
    (3, 200)
    """
    Om = 2 * pi if Om is None else Om
    b = array((0, 0, 1)) if b is None else b
    # first produce 'equally' distributed directions in spherical coords
    o = 4 * pi / Om
    h = -1 + 2 * arange(N) / (N * o - 1.0)
    theta = arccos(h)
    phi = zeros_like(theta)
    for i, hk in enumerate(h[1:]):
        phi[i + 1] = phi[i] + 3.6 / sqrt(N * o * (1 - hk * hk)) % (2 * pi)
    # translate to cartesian coords
    xyz = vstack((sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)))
    # mirror everything on a plane so that b points into the center
    a = xyz[:, 0]
    b = b / norm(b)
    ab = (a - b)[:, newaxis]
    if norm(ab) < 1e-10:
        return xyz
    # this is the Householder matrix for mirroring
    H = identity(3) - dot(ab, ab.T) / dot(ab.T, a)
    # actual mirroring
    return dot(H, xyz)


class GeneralFlowEnvironment(Environment):
    """
    An acoustic environment with a generic flow field.

    This class provides the facilities to calculate the travel time (distances) between grid point
    locations and microphone locations in a generic flow field with non-uniform velocities that
    depend on the location. The algorithm for the calculation uses a ray-tracing approach that bases
    on rays cast from every microphone position in multiple directions and traced backwards in time.
    The result is interpolated within a tetrahedal grid spanned between these rays.

    See Also
    --------
    :class:`scipy.interpolate.LinearNDInterpolator` :
        Piecewise linear interpolator in N > 1 dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> import acoular as ac
    >>>
    >>> # Instantiate the flow field
    >>> flow_field = ac.OpenJet(v0=10.0, D=0.4, l=3.121)
    >>>
    >>> # Create an instance of GeneralFlowEnvironment
    >>> environment = ac.GeneralFlowEnvironment(
    ...     ff=flow_field,  # Use the custom flow field
    ...     N=300,  # Number of rays
    ...     Om=np.pi,  # Maximum solid angle
    ... )
    """

    #: The flow field object describing the velocity field,
    #: which must be an instance of :class:`~acoular.environments.FlowField`.
    ff = Instance(FlowField, desc='flow field')

    #: The number of rays used per solid angle :math:`\Omega`. Defaults to ``200``.
    N = Int(200, desc='number of rays per Om')

    #: The maximum solid angle (in steradians) used in the ray-tracing algorithm.
    #: Default is :obj:`numpy.pi`.
    Om = Float(pi, desc='maximum solid angle')

    #: A unique identifier based on the environment properties. (read-only)
    digest = Property(
        depends_on=['c', 'ff.digest', 'N', 'Om'],
    )

    #: A dictionary for storing precomputed interpolators to optimize repeated calculations.
    #: (internal use)
    idict = Dict

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _r(self, gpos, mpos=0.0):
        c = self.c

        if isscalar(mpos):
            mpos = array((0, 0, 0), dtype=float32)[:, newaxis]

        gt = empty((gpos.shape[-1], mpos.shape[-1]))
        for micnum, x0 in enumerate(mpos.T):
            key = x0.tobytes()  # make array hashable
            try:
                li = self.idict[key]  # fetch stored interpolator
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
        return c * gt  # return distance along ray

    def get_interpolator(self, roi, x0):
        """
        Generate an interpolator for ray travel times based on a region of interest.

        This method computes the ray trajectories starting from a given microphone position (``x0``)
        through a region of interest (``roi``). The rays' paths are integrated numerically using a
        system of differential equations, and the resulting points are used to construct a
        convex hull. A linear interpolator is then created to estimate travel times for arbitrary
        points within the region.

        Parameters
        ----------
        roi : :class:`numpy.ndarray` of :class:`floats<float>`
            Array representing the region of interest (ROI), where each column corresponds
            to a point in the 3D space :math:`(x, y, z)`, shape `(3, M)`.

        x0 : :class:`numpy.ndarray` of :class:`floats<float>`
            Array representing the location of the microphone in 3D Cartesian coordinates,
            shape `(3,)`.

        Returns
        -------
        :class:`scipy.interpolate.LinearNDInterpolator` object
            A linear interpolator object for estimating travel times for 3D positions
            within the computed ray trajectories.
        """
        c = self.c

        # the DE system
        def f1(t, y, v):  # noqa: ARG001
            x = y[0:3]
            s = y[3:6]
            vv, dv = v(x)
            sa = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
            x = empty(6)
            x[0:3] = c * s / sa - vv  # time reversal
            x[3:6] = dot(s, -dv.T)  # time reversal
            return x

        # integration along a single ray
        def fr(x0, n0, rmax, dt, v, xyz, t):
            s0 = n0 / (c + dot(v(x0)[0], n0))
            y0 = hstack((x0, s0))
            oo = ode(f1)
            oo.set_f_params(v)
            oo.set_integrator(
                'vode',
                rtol=1e-4,  # accuracy !
                max_step=1e-4 * rmax,
            )  # for thin shear layer
            oo.set_initial_value(y0, 0)
            while oo.successful():
                xyz.append(oo.y[0:3])
                t.append(oo.t)
                if norm(oo.y[0:3] - x0) > rmax:
                    break
                oo.integrate(oo.t + dt)

        gs2 = roi.shape[-1]
        vv = self.ff.v
        NN = int(sqrt(self.N))
        xe = roi.mean(1)  # center of grid
        r = x0[:, newaxis] - roi
        rmax = sqrt((r * r).sum(0).max())  # maximum distance
        nv = spiral_sphere(self.N, self.Om, b=xe - x0)
        rstep = rmax / sqrt(self.N)
        rmax += rstep
        tstep = rstep / c
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
                if dd.simplices.min() >= gs2:
                    break
        xyz = array(xyz)
        t = array(t)
        return LinearNDInterpolator(xyz, t)
