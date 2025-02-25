# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implement support for multidimensional grids and integration sectors.

.. autosummary::
    :toctree: generated/

    Grid
    RectGrid
    RectGrid3D
    ImportGrid
    LineGrid
    MergeGrid
    Sector
    SingleSector
    RectSector
    RectSector3D
    CircSector
    PolySector
    ConvexSector
    MultiSector
    Polygon
    in_hull
"""

# imports from other packages
import xml.dom.minidom
from abc import abstractmethod

from numpy import (
    absolute,
    append,
    arange,
    argmin,
    array,
    asarray,
    concatenate,
    copysign,
    fabs,
    inf,
    isscalar,
    ma,
    mgrid,
    newaxis,
    ones,
    ones_like,
    s_,
    sum,  # noqa: A004
    tile,
    unique,
    where,
    zeros,
)
from scipy.linalg import norm

# from matplotlib.path import Path
from scipy.spatial import Delaunay
from traits.api import (
    ABCHasStrictTraits,
    Bool,
    CArray,
    File,
    Float,
    Instance,
    Int,
    List,
    Property,
    Str,
    Tuple,
    Union,
    cached_property,
    observe,
    on_trait_change,
    property_depends_on,
)
from traits.trait_errors import TraitError

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest, ldigest


def in_hull(p, hull, border=True, tol=0):
    """
    Test if points in ``p`` are in ``hull``, in- or excluding the border.

    Parameters
    ----------
    p : :class:`numpy.ndarray` of :class:`floats<float>`, shape `(N, K)`
        Coordinates of `N` points in `K` dimensions.

    hull : :class:`numpy.ndarray` of :class:`floats<float>`, shape `(M, K)`, or :class:`~scipy.spatial.Delaunay` object
        Coordinates of `M` points in `K` dimensions for which Delaunay triangulation will be
        computed.

    border : bool, optional
        Points in :attr:`p` on the border of :attr:`hull` will be kept in the return if `True`. If
        `False`, only points inside :attr:`hull` will be kept. Default is `True`.

    tol : :class:`float`, optional
        Tolerance allowed in the :meth:`inside-triangle check<scipy.spatial.Delaunay.find_simplex>`.
        Default is ``0``.

    Returns
    -------
    :class:`numpy.ndarray` of :class:`bools<bool>`
        An array of boolean values indicating which points in ``p`` are inside the hull, same
        shape as ``p``. Each entry is ``True`` if the corresponding point is inside the hull (or
        on the border, if ``border=True``), and ``False`` otherwise.

    Notes
    -----
    This function uses Delaunay triangulation to determine if a point is inside the convex hull,
    which is efficient and robust for arbitrary shapes in higher-dimensional spaces.

    Examples
    --------
    >>> from acoular.grids import in_hull
    >>> import numpy as np
    >>> from scipy.spatial import Delaunay
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> hull = Delaunay(points)
    >>> p = np.array([[0.5, 0.5], [2, 2]])
    >>> in_hull(p, hull)
    array([ True, False])
    """  # noqa W505
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    if border:
        return hull.find_simplex(p, tol=tol) >= 0
    return hull.find_simplex(p, tol=tol) > 0


def _det(xvert, yvert):
    xvert = asarray(xvert, dtype=float)
    yvert = asarray(yvert, dtype=float)
    x_prev = concatenate(([xvert[-1]], xvert[:-1]))
    y_prev = concatenate(([yvert[-1]], yvert[:-1]))
    return sum(yvert * x_prev - xvert * y_prev, axis=0)


class Polygon:
    """
    Create an object representing a general polygon in a 2D plane.

    This class allows defining a polygon by specifying the coordinates of its vertices and provides
    methods for checking whether a set of points lies inside the polygon, or if a point is closer to
    a side or vertex of the polygon.

    Parameters
    ----------
    x : array_like
        Array of x-coordinates of the vertices that define the polygon. These coordinates should
        form a closed shape (i.e., the last point should be the same as the first point).

    y : array_like
        Array of y-coordinates of the vertices that define the polygon. These coordinates should
        correspond to the x-coordinates, forming a closed shape.

    Attributes
    ----------
    x : :class:`numpy.ndarray`
        Array of x-coordinates of the polygon vertices.

    y : :class:`numpy.ndarray`
        Array of y-coordinates of the polygon vertices.
    """

    def __init__(self, x, y):
        if len(x) != len(y):
            msg = 'x and y must be equally sized.'
            raise IndexError(msg)
        self.x = asarray(x, dtype=float)
        self.y = asarray(y, dtype=float)
        # Closes the polygon if were open
        x1, y1 = x[0], y[0]
        xn, yn = x[-1], y[-1]
        if x1 != xn or y1 != yn:
            self.x = concatenate((self.x, [x1]))
            self.y = concatenate((self.y, [y1]))
        # Anti-clockwise coordinates
        if _det(self.x, self.y) < 0:
            self.x = self.x[::-1]
            self.y = self.y[::-1]

    def is_inside(self, xpoint, ypoint, smalld=1e-12):
        """
        Check if a point or set of points are inside the polygon.

        Parameters
        ----------
        xpoint : :class:`float` or array_like
            Array of x-coordinates of the points to be tested.

        ypoint : :class:`float` or array_like
            Array of y-coordinates of the points to be tested.

        smalld : :class:`float`, optional
            Tolerance used for floating point comparisons when checking if a point is exactly on a
            polygon's edge. The default value is ``1e-12``.

        Returns
        -------
        :class:`float` or array_like
            The distance from the point to the nearest point on the polygon. The values returned
            have the following meanings:
            - ``mindst < 0``: Point is outside the polygon.
            - ``mindst = 0``: Point is on an edge of the polygon.
            - ``mindst > 0``: Point is inside the polygon.

        Notes
        -----
        The method uses an improved algorithm based on Nordbeck and Rydstedt for determining
        whether a point is inside a polygon :cite:`SLOAN198545`.
        """
        xpoint = asarray(xpoint, dtype=float)
        ypoint = asarray(ypoint, dtype=float)
        # Scalar to array
        if xpoint.shape == ():
            xpoint = array([xpoint], dtype=float)
            ypoint = array([ypoint], dtype=float)
            scalar = True
        else:
            scalar = False
        # Check consistency
        if xpoint.shape != ypoint.shape:
            msg = 'x and y has different shapes'
            raise IndexError(msg)
        # If snear = True: Dist to nearest side < nearest vertex
        # If snear = False: Dist to nearest vertex < nearest side
        snear = ma.masked_all(xpoint.shape, dtype=bool)
        # Initialize arrays
        mindst = ones_like(xpoint, dtype=float) * inf
        j = ma.masked_all(xpoint.shape, dtype=int)
        x = self.x
        y = self.y
        n = len(x) - 1  # Number of sides/vertices defining the polygon
        # Loop over each side defining polygon
        for i in range(n):
            d = ones_like(xpoint, dtype=float) * inf
            # Start of side has coords (x1, y1)
            # End of side has coords (x2, y2)
            # Point has coords (xpoint, ypoint)
            x1 = x[i]
            y1 = y[i]
            x21 = x[i + 1] - x1
            y21 = y[i + 1] - y1
            x1p = x1 - xpoint
            y1p = y1 - ypoint
            # Points on infinite line defined by
            #     x = x1 + t * (x1 - x2)
            #     y = y1 + t * (y1 - y2)
            # where
            #     t = 0    at (x1, y1)
            #     t = 1    at (x2, y2)
            # Find where normal passing through (xpoint, ypoint) intersects
            # infinite line
            t = -(x1p * x21 + y1p * y21) / (x21**2 + y21**2)
            tlt0 = t < 0
            tle1 = (t >= 0) & (t <= 1)
            # Normal intersects side
            d[tle1] = (x1p[tle1] + t[tle1] * x21) ** 2 + (y1p[tle1] + t[tle1] * y21) ** 2
            # Normal does not intersects side
            # Point is closest to vertex (x1, y1)
            # Compute square of distance to this vertex
            d[tlt0] = x1p[tlt0] ** 2 + y1p[tlt0] ** 2
            # Store distances
            mask = d < mindst
            mindst[mask] = d[mask]
            j[mask] = i
            # Point is closer to (x1, y1) than any other vertex or side
            snear[mask & tlt0] = False
            # Point is closer to this side than to any other side or vertex
            snear[mask & tle1] = True
        if ma.count(snear) != snear.size:
            msg = 'Error computing distances'
            raise IndexError(msg)
        mindst **= 0.5
        # Point is closer to its nearest vertex than its nearest side, check if
        # nearest vertex is concave.
        # If the nearest vertex is concave then point is inside the polygon,
        # else the point is outside the polygon.
        jo = j.copy()
        jo[j == 0] -= 1
        area = _det([x[j + 1], x[j], x[jo - 1]], [y[j + 1], y[j], y[jo - 1]])
        mindst[~snear] = copysign(mindst, area)[~snear]
        # Point is closer to its nearest side than to its nearest vertex, check
        # if point is to left or right of this side.
        # If point is to left of side it is inside polygon, else point is
        # outside polygon.
        area = _det([x[j], x[j + 1], xpoint], [y[j], y[j + 1], ypoint])
        mindst[snear] = copysign(mindst, area)[snear]
        # Point is on side of polygon
        mindst[fabs(mindst) < smalld] = 0
        # If input values were scalar then the output should be too
        if scalar:
            mindst = float(mindst)
        return mindst


@deprecated_alias({'gpos': 'pos'})
class Grid(ABCHasStrictTraits):
    """
    Abstract base class for grid geometries.

    This class defines a common interface for all grid geometries and provides tools to query grid
    properties and related data. It is intended to serve as a base class for specialized grid
    implementations and should not be instantiated directly as it lacks concrete functionality.
    """

    #: The total number of grid points. This property is automatically calculated based on other
    #: defining attributes of the grid. (read-only)
    size = Property(desc='overall number of grid points')

    #: The shape of the grid, represented as a tuple. Primarily useful for Cartesian grids.
    #: (read-only)
    shape = Property(desc='grid shape as tuple')

    #: The grid positions represented as a (3, :attr:`size`) array of :class:`floats<float>`.
    #: (read-only)
    pos = Property(desc='x, y, z positions of grid points')

    #: A unique identifier for the grid, based on its properties. (read-only)
    digest = Property

    @abstractmethod
    def _get_digest(self):
        """Generate a unique digest for the grid."""

    # 'digest' is a placeholder for other properties in derived classes, necessary to trigger the
    # depends on mechanism
    @property_depends_on(['digest'])
    @abstractmethod
    def _get_size(self):
        """Return the number of grid points."""

    # 'digest' is a placeholder for other properties in derived classes
    @property_depends_on(['digest'])
    @abstractmethod
    def _get_shape(self):
        """Return the shape of the grid as a Tuple."""

    @property_depends_on(['digest'])
    @abstractmethod
    def _get_pos(self):
        """Return the grid positions as array of floats, shape (3, :attr:`size`)."""

    def subdomain(self, sector):
        """
        Return the indices for a subdomain in the grid.

        Allows arbitrary subdomains of type :class:`Sector`.

        Parameters
        ----------
        sector : :class:`Sector` object
            Sector describing the subdomain.

        Returns
        -------
        :class:`tuple`
            A 2-tuple of arrays of integers or :obj:`numpy.s_` objects that can be used to mask or
            select the specified subdomain from a grid-shaped array.

        Notes
        -----
        The :func:`numpy.where` method is used to determine the the indices.
        """
        xpos = self.pos
        # construct grid-shaped array with "True" entries where sector is
        xyi = sector.contains(xpos).reshape(self.shape)
        # return indices of "True" entries
        return where(xyi)


@deprecated_alias({'gpos': 'pos'}, read_only=True)
class RectGrid(Grid):
    """
    Provides a 2D Cartesian grid for beamforming results.

    This grid is composed of square or nearly square cells and lies on a plane perpendicular
    to the z-axis. It is defined by the lower and upper x- and y-limits and a constant z-coordinate.
    """

    #: The lower x-limit that defines the grid. Default is ``-1``.
    x_min = Float(-1.0, desc='minimum  x-value')

    #: The upper x-limit that defines the grid. Default is ``1``.
    x_max = Float(1.0, desc='maximum  x-value')

    #: The lower y-limit that defines the grid. Default is ``-1``.
    y_min = Float(-1.0, desc='minimum  y-value')

    #: The upper y-limit that defines the grid. Default is ``1``.
    y_max = Float(1.0, desc='maximum  y-value')

    #: The constant z-coordinate of the grid plane. Default is ``1.0``.
    z = Float(1.0, desc='position on z-axis')

    #: The side length of each cell. Default is ``0.1``.
    increment = Float(0.1, desc='step size')

    #: Number of grid points along x-axis. (read-only)
    nxsteps = Property(desc='number of grid points along x-axis')

    #: Number of grid points along y-axis. (read-only)
    nysteps = Property(desc='number of grid points along y-axis')

    #: A unique identifier for the grid, based on its properties. (read-only)
    digest = Property(
        depends_on=['x_min', 'x_max', 'y_min', 'y_max', 'z', 'increment'],
    )

    @property_depends_on(['nxsteps', 'nysteps'])
    def _get_size(self):
        return self.nxsteps * self.nysteps

    @property_depends_on(['nxsteps', 'nysteps'])
    def _get_shape(self):
        return (self.nxsteps, self.nysteps)

    @property_depends_on(['x_min', 'x_max', 'increment'])
    def _get_nxsteps(self):
        i = abs(self.increment)
        if i != 0:
            return int(round((abs(self.x_max - self.x_min) + i) / i))
        return 1

    @property_depends_on(['y_min', 'y_max', 'increment'])
    def _get_nysteps(self):
        i = abs(self.increment)
        if i != 0:
            return int(round((abs(self.y_max - self.y_min) + i) / i))
        return 1

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property_depends_on(['x_min', 'x_max', 'y_min', 'y_max', 'increment'])
    def _get_pos(self):
        bpos = mgrid[
            self.x_min : self.x_max : self.nxsteps * 1j,
            self.y_min : self.y_max : self.nysteps * 1j,
            self.z : self.z + 0.1,
        ]
        bpos.resize((3, self.size))
        return bpos

    def index(self, x, y):
        """
        Find the indices of a grid point near a given coordinate.

        Parameters
        ----------
        x : :class:`float`
            The x coordinate of interest.
        y : :class:`float`
            The y coordinate of interest.

        Returns
        -------
        :class:`tuple` of :class:`ints<int>`
            Indices corresponding to the nearest grid point.

        Raises
        ------
        ValueError
            If the coordinates are outside the grid boundaries.
        """
        if x < self.x_min or x > self.x_max:
            msg = 'x-value out of range'
            raise ValueError(msg)
        if y < self.y_min or y > self.y_max:
            msg = 'y-value out of range'
            raise ValueError(msg)
        xi = int((x - self.x_min) / self.increment + 0.5)
        yi = int((y - self.y_min) / self.increment + 0.5)
        return xi, yi

    def indices(self, *r):
        """
        Find the indices of a subdomain in the grid.

        Supports rectangular, circular, and polygonal subdomains.

        Parameters
        ----------
        r : :class:`tuple` of :class:`floats<float>`
            Defines the subdomain shape and dimensions:
                - If 3 values are provided: center ``(x1, y1)`` and radius ``r2`` define a circle.
                - If 4 values are provided: corners ``(x1, y1)`` and ``(x2, y2)`` define a
                  rectangle.
                - If more than 4 values are provided: vertices ``(xn, yn)`` define a polygon.

        Returns
        -------
        :class:`tuple`
            A 2-tuple of indices or slices corresponding to the subdomain.
        """
        if len(r) == 3:  # only 3 values given -> use x,y,radius method
            xpos = self.pos
            xis = []
            yis = []
            dr2 = (xpos[0, :] - r[0]) ** 2 + (xpos[1, :] - r[1]) ** 2
            # array with true/false entries
            inds = dr2 <= r[2] ** 2
            for np in arange(self.size)[inds]:  # np -- points in x2-circle
                xi, yi = self.index(xpos[0, np], xpos[1, np])
                xis += [xi]
                yis += [yi]
            if not (xis and yis):  # if no points in circle, take nearest one
                return self.index(r[0], r[1])
            return array(xis), array(yis)
        if len(r) == 4:  # rectangular subdomain - old functionality
            xi1, yi1 = self.index(min(r[0], r[2]), min(r[1], r[3]))
            xi2, yi2 = self.index(max(r[0], r[2]), max(r[1], r[3]))
            return s_[xi1 : xi2 + 1], s_[yi1 : yi2 + 1]
        xpos = self.pos
        xis = []
        yis = []
        # replaced matplotlib Path by numpy
        # p = Path(array(r).reshape(-1,2))
        # inds = p.contains_points()
        # inds = in_poly(xpos[:2,:].T,array(r).reshape(-1,2))
        poly = Polygon(array(r).reshape(-1, 2)[:, 0], array(r).reshape(-1, 2)[:, 1])
        dists = poly.is_inside(xpos[0, :], xpos[1, :])
        inds = dists >= 0
        for np in arange(self.size)[inds]:  # np -- points in x2-circle
            xi, yi = self.index(xpos[0, np], xpos[1, np])
            xis += [xi]
            yis += [yi]
        if not (xis and yis):  # if no points inside, take nearest to center
            center = array(r).reshape(-1, 2).mean(0)
            return self.index(center[0], center[1])
        return array(xis), array(yis)
        # return arange(self.size)[inds]

    def extend(self):
        """
        Return the grid's extension in :obj:`matplotlib.pyplot.imshow` compatible form.

        Returns
        -------
        :class:`tuple` of :class:`floats<float>`
            (:attr:`x_min`, :attr:`x_max`, :attr:`y_min`, :attr:`y_max`) representing the grid's
            extent.

        Notes
        -----
        - ``pylab.imhow`` is the same as :obj:`matplotlib.pyplot.imshow`. It's only using a
          different namespace.
        - The return of the method is ment for the ``extent`` parameter of
          :obj:`matplotlib.pyplot.imshow`.

        Examples
        --------
        >>> from acoular import RectGrid
        >>> grid = RectGrid()
        >>> grid.y_min = -5
        >>> grid.y_max = 5
        >>> grid.extend()
        (-1.0, 1.0, -5.0, 5.0)
        """
        return (self.x_min, self.x_max, self.y_min, self.y_max)


class RectGrid3D(RectGrid):
    """
    Provide a cartesian 3D grid for the beamforming results.

    The grid has cubic or nearly cubic cells. It is defined by lower and upper x-, y- and  z-limits.
    """

    #: The lower z-limit that defines the grid. Default is ``-1``.
    z_min = Float(-1.0, desc='minimum  z-value')

    #: The upper z-limit that defines the grid. Default is ``1``.
    z_max = Float(1.0, desc='maximum  z-value')

    #: Number of grid points along x-axis. (read-only)
    nxsteps = Property(desc='number of grid points along x-axis')

    #: Number of grid points along y-axis. (read-only)
    nysteps = Property(desc='number of grid points along y-axis')

    #: Number of grid points along z-axis. (read-only)
    nzsteps = Property(desc='number of grid points along z-axis')

    # Private trait for increment handling
    _increment = Union(Float(), CArray(shape=(3,), dtype=float), default_value=0.1, desc='step size')

    #: The cell side length for the grid. This can either be a scalar (same increments in all 3
    #: dimensions) or a (3,) array of :class:`tuple` of :class:`floats<float>` with respective
    #: increments in x-, y-, and z-direction. Default is ``0.1``.
    increment = Property(desc='step size')

    def _get_increment(self):
        return self._increment

    def _set_increment(self, increment):
        if isscalar(increment):
            try:
                self._increment = absolute(float(increment))
            except ValueError as ve:
                raise TraitError(args=self, name='increment', info='Float or CArray(3,)', value=increment) from ve
        elif len(increment) == 3:
            self._increment = array(increment, dtype=float)
        else:
            raise (TraitError(args=self, name='increment', info='Float or CArray(3,)', value=increment))

    #: A unique identifier for the grid, based on its properties. (read-only)
    digest = Property(
        depends_on=['x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max', '_increment'],
    )

    @property_depends_on(['nxsteps', 'nysteps', 'nzsteps'])
    def _get_size(self):
        return self.nxsteps * self.nysteps * self.nzsteps

    @property_depends_on(['nxsteps', 'nysteps', 'nzsteps'])
    def _get_shape(self):
        return (self.nxsteps, self.nysteps, self.nzsteps)

    @property_depends_on(['x_min', 'x_max', '_increment'])
    def _get_nxsteps(self):
        i = abs(self.increment) if isscalar(self.increment) else abs(self.increment[0])
        if i != 0:
            return int(round((abs(self.x_max - self.x_min) + i) / i))
        return 1

    @property_depends_on(['y_min', 'y_max', '_increment'])
    def _get_nysteps(self):
        i = abs(self.increment) if isscalar(self.increment) else abs(self.increment[1])
        if i != 0:
            return int(round((abs(self.y_max - self.y_min) + i) / i))
        return 1

    @property_depends_on(['z_min', 'z_max', '_increment'])
    def _get_nzsteps(self):
        i = abs(self.increment) if isscalar(self.increment) else abs(self.increment[2])
        if i != 0:
            return int(round((abs(self.z_max - self.z_min) + i) / i))
        return 1

    @property_depends_on('digest')
    def _get_pos(self):
        bpos = mgrid[
            self.x_min : self.x_max : self.nxsteps * 1j,
            self.y_min : self.y_max : self.nysteps * 1j,
            self.z_min : self.z_max : self.nzsteps * 1j,
        ]
        bpos.resize((3, self.size))
        return bpos

    @cached_property
    def _get_digest(self):
        return digest(self)

    def index(self, x, y, z):
        """
        Return the indices for a grid point near a certain coordinate.

        This can be used to query results or coordinates at or near a certain coordinate. Raises an
        exception if the given coordinate is outside the grid.

        Parameters
        ----------
        x, y, z : :class:`float`
            The coordinates for which the indices is queried.

        Returns
        -------
        3-:class:`tuple` of :class:`ints<int>`
            The indices that give the grid point nearest to the given x, y, z coordinates from an
            array with the same shape as the grid.

        Examples
        --------
        Check which of the points in a simple 8-point rectangular grid is closest to the point
        ``(0.5, 0.5, 1.0)``.

        >>> import acoular as ac
        >>>
        >>> grid = ac.RectGrid3D()
        >>> grid.increment = 2
        >>> grid.pos
        array([[-1., -1., -1., -1.,  1.,  1.,  1.,  1.],
               [-1., -1.,  1.,  1., -1., -1.,  1.,  1.],
               [-1.,  1., -1.,  1., -1.,  1., -1.,  1.]])
        >>>
        >>> grid.index(0.5, 0.5, 1.0)
        (1, 1, 1)
        """
        if x < self.x_min or x > self.x_max:
            msg = f'x-value out of range {x:f} ({self.x_min:f}, {self.x_max:f})'
            raise ValueError(msg)
        if y < self.y_min or y > self.y_max:
            msg = f'y-value out of range {y:f} ({self.y_min:f}, {self.y_max:f})'
            raise ValueError(msg)
        if z < self.z_min or z > self.z_max:
            msg = f'z-value out of range {z:f} ({self.z_min:f}, {self.z_max:f})'
            raise ValueError(msg)
        if isscalar(self.increment):
            incx = incy = incz = self.increment
        else:
            incx, incy, incz = self.increment
        xi = int(round((x - self.x_min) / incx))
        yi = int(round((y - self.y_min) / incy))
        zi = int(round((z - self.z_min) / incz))
        return xi, yi, zi

    def indices(self, x1, y1, z1, x2, y2, z2):
        """
        Return the indices for a subdomain in the grid.

        Allows box-shaped subdomains. This can be used to mask or to query results from a certain
        sector or subdomain.

        Parameters
        ----------
        x1, y1, z1, x2, y2, z2 : :class:`float`
            A box-shaped sector is assumed that is given by two corners ``(x1,y1,z1)`` and
            ``(x2,y2,z2)``.

        Returns
        -------
        3-:class:`tuple` of :obj:`numpy.s_` objects
            The indices that can be used to mask/select the grid subdomain from an array with the
            same shape as the grid.

        Examples
        --------
        Get the indices of the grid points of a simple 27-point rectangular grid which are located
        inside the first octant.

        >>> import acoular as ac
        >>>
        >>> grid = ac.RectGrid3D()
        >>> grid.increment = 1
        >>> grid.pos
        array([[-1., -1., -1., -1., -1., -1., -1., -1., -1.,  0.,  0.,  0.,  0.,
                 0.,  0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,
                 1.],
               [-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1., -1., -1., -1.,  0.,
                 0.,  0.,  1.,  1.,  1., -1., -1., -1.,  0.,  0.,  0.,  1.,  1.,
                 1.],
               [-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.,
                 0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1., -1.,  0.,
                 1.]])
        >>>
        >>> grid.indices(0, 0, 0, 1, 1, 1)
        (slice(1, 3, None), slice(1, 3, None), slice(1, 3, None))
        """
        xi1, yi1, zi1 = self.index(min(x1, x2), min(y1, y2), min(z1, z2))
        xi2, yi2, zi2 = self.index(max(x1, x2), max(y1, y2), max(z1, z2))
        return s_[xi1 : xi2 + 1], s_[yi1 : yi2 + 1], s_[zi1 : zi2 + 1]


@deprecated_alias({'from_file': 'file', 'gpos_file': 'pos'})
class ImportGrid(Grid):
    """
    Load a 3D grid from an XML file.

    This class is used to import a 3D grid defined in an XML file. The grid's
    positions and subgrid names are parsed and stored for further processing.
    """

    #: Name of the .xml-file from which to read the data.
    file = File(filter=['*.xml'], exists=True, desc='name of the xml file to import')

    _gpos = CArray(dtype=float, desc='x, y, z position of all Grid Points')

    #: Names of subgrids for each point.
    #: This is an optional property, typically used when grids are divided into named subregions.
    subgrids = CArray(desc='names of subgrids for each point')

    #: A unique identifier for the grid, based on its properties. (read-only)
    digest = Property(depends_on=['_gpos'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property_depends_on(['_gpos'])
    def _get_size(self):
        return self.pos.shape[-1]

    @property_depends_on(['_gpos'])
    def _get_shape(self):
        return (self.pos.shape[-1],)

    @property_depends_on(['_gpos'])
    def _get_pos(self):
        return self._gpos

    def _set_pos(self, pos):
        self._gpos = pos

    @on_trait_change('file')
    def import_gpos(self):
        """
        Import the grid point locations and subgrid names from an XML file.

        This method is automatically called whenever the :attr:`file` attribute changes.

        Notes
        -----
        The XML file should have elements with tag ``<pos>``, where each ``<pos>`` element
        contains attributes for ``x``, ``y``, ``z``, and optionally ``subgrid``.

        Examples
        --------
        Generate the positional data of two microphone grids:

        >>> import numpy as np
        >>>
        >>> # Grid 1: ten points aranged in a circle in the x-y plane at z=0
        >>> args = 2 * np.pi * np.arange(10) / 10
        >>> x1 = np.cos(args)
        >>> y1 = np.sin(args)
        >>> z1 = np.zeros_like(x1)
        >>> grid1 = np.vstack([x1, y1, z1]).T
        >>>
        >>> # Grid 2: nine points aranged in a mesh grid the the x-y plane at z=1
        >>> a = np.linspace(-1, 1, 3)
        >>> x2, y2 = np.meshgrid(a, a)
        >>> z2 = np.ones_like(x2)
        >>> grid2 = np.vstack([x2, y2, z2])

        Save the generated data in an XML file:

        >>> import xml.etree.cElementTree as ET
        >>>
        >>> # Create the root element for the XML document
        >>> root = ET.Element('root')
        >>>
        >>> # Loop over both grid 1 and grid 2, and create XML elements for each of their points
        >>> for num, grid in enumerate([grid1, grid2]):  # doctest: +SKIP
        ...     for x, y, z in grid:
        ...         # For each point in the grid, create a 'pos' XML element
        ...         ET.SubElement(root, 'pos', subgrid=str(num), x=str(x), y=str(y), z=str(z))
        >>>
        >>> # Create the final XML tree
        >>> tree = ET.ElementTree(root)
        >>>
        >>> # Write the XML tree to a file named 'two_grids.xml'
        >>> tree.write('two_grids.xml')  # doctest: +SKIP

        The ``two_grids.xml`` file will look like this:

        .. code-block:: xml

            <root>
                <pos subgrid="0" x="1.0" y="0.0" z="0.0" />
                <pos subgrid="0" x="0.8090169943749475" y="0.5877852522924731" z="0.0" />
                <pos subgrid="0" x="0.30901699437494745" y="0.9510565162951535" z="0.0" />
                <pos subgrid="0" x="-0.30901699437494734" y="0.9510565162951536" z="0.0" />
                <pos subgrid="0" x="-0.8090169943749473" y="0.5877852522924732" z="0.0" />
                <pos subgrid="0" x="-1.0" y="1.2246467991473532e-16" z="0.0" />
                <pos subgrid="0" x="-0.8090169943749475" y="-0.587785252292473" z="0.0" />
                <pos subgrid="0" x="-0.30901699437494756" y="-0.9510565162951535" z="0.0" />
                <pos subgrid="0" x="0.30901699437494723" y="-0.9510565162951536" z="0.0" />
                <pos subgrid="0" x="0.8090169943749473" y="-0.5877852522924732" z="0.0" />
                <pos subgrid="1" x="-1.0" y="0.0" z="1.0" />
                <pos subgrid="1" x="-1.0" y="0.0" z="1.0" />
                <pos subgrid="1" x="-1.0" y="0.0" z="1.0" />
                <pos subgrid="1" x="-1.0" y="-1.0" z="-1.0" />
                <pos subgrid="1" x="0.0" y="0.0" z="0.0" />
                <pos subgrid="1" x="1.0" y="1.0" z="1.0" />
                <pos subgrid="1" x="1.0" y="1.0" z="1.0" />
                <pos subgrid="1" x="1.0" y="1.0" z="1.0" />
                <pos subgrid="1" x="1.0" y="1.0" z="1.0" />
            </root>

        Importing the ``two_grids.xml`` file:

        >>> import acoular as ac
        >>>
        >>> grids = ac.ImportGrid(file='two_grids.xml')  # doctest: +SKIP
        >>> grids.size  # doctest: +SKIP
        19
        >>> grids.subgrids  # doctest: +SKIP
        array(['0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '1', '1',
               '1', '1', '1', '1', '1', '1'], dtype='<U1')
        >>> grids.gpos.T  # doctest: +SKIP
        array([[ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 8.09016994e-01,  5.87785252e-01,  0.00000000e+00],
               [ 3.09016994e-01,  9.51056516e-01,  0.00000000e+00],
               [-3.09016994e-01,  9.51056516e-01,  0.00000000e+00],
               [-8.09016994e-01,  5.87785252e-01,  0.00000000e+00],
               [-1.00000000e+00,  1.22464680e-16,  0.00000000e+00],
               [-8.09016994e-01, -5.87785252e-01,  0.00000000e+00],
               [-3.09016994e-01, -9.51056516e-01,  0.00000000e+00],
               [ 3.09016994e-01, -9.51056516e-01,  0.00000000e+00],
               [ 8.09016994e-01, -5.87785252e-01,  0.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00,  1.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00,  1.00000000e+00],
               [-1.00000000e+00,  0.00000000e+00,  1.00000000e+00],
               [-1.00000000e+00, -1.00000000e+00, -1.00000000e+00],
               [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
               [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
               [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00],
               [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00]])

        Consider two XML files, ``grid1.xml`` and ``grid2.xml`` containing different grids.

        >>> import acoular as ac
        >>>
        >>> grid = ac.ImportGrid(file='grid1.xml')  # doctest: +SKIP
        >>> grid.size  # doctest: +SKIP
        8
        >>> grid.file = 'grid2.xml'  # doctest: +SKIP
        >>> grid.size  # doctest: +SKIP
        12
        """
        doc = xml.dom.minidom.parse(self.file)
        names = []
        xyz = []
        for el in doc.getElementsByTagName('pos'):
            names.append(el.getAttribute('subgrid'))
            xyz.append([float(el.getAttribute(a)) for a in 'xyz'])
        self._gpos = array(xyz, 'd').swapaxes(0, 1)
        self.subgrids = array(names)


@deprecated_alias({'gpos': 'pos', 'numpoints': 'num_points'}, read_only=['gpos'])
class LineGrid(Grid):
    """
    Define a 3D grid for a line geometry.

    The :class:`LineGrid` class represents a grid where points are arranged linearly in 3D space.
    The grid is defined by a starting location (:attr:`loc`), a direction vector
    (:attr:`direction`), a total length (:attr:`length`), and the number of points
    (:attr:`num_points`) along the line.

    Notes
    -----
    - The distance between points is :attr:`length` ``/ (`` :attr:`num_points` ``- 1)``.
    - The direction vector is normalized to ensure consistency.

    Examples
    --------
    Create a line grid with 5 points along the x-axis, starting at (0, 0, 0), with a length of 4
    meters:

    >>> import acoular as ac
    >>> grid = ac.LineGrid()
    >>> grid.loc = (0.0, 0.0, 0.0)
    >>> grid.direction = (1.0, 0.0, 0.0)
    >>> grid.length = 4
    >>> grid.num_points = 5
    >>> grid.pos
    array([[0., 1., 2., 3., 4.],
           [0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.]])
    """

    #: Starting point of the grid in 3D space. Default is ``(0.0, 0.0, 0.0)``.
    loc = Tuple((0.0, 0.0, 0.0))

    #: A vector defining the orientation of the line in 3D space. Default is ``(1.0, 0.0, 0.0)``.
    direction = Tuple((1.0, 0.0, 0.0), desc='Line orientation ')

    #: Total length of the line. Default is ``1.0``.
    length = Float(1, desc='length of the line source')

    #: Number of grid points along the line. Default is ``1``.
    num_points = Int(1, desc='length of the line source')

    #: The total number of grid points. Automatically updated when other grid-defining attributes
    #: are set. (read-only)
    size = Property(desc='overall number of grid points')

    #: A (3, :attr:`size`) array containing the x, y, and z positions
    #: of the grid points. (read-only)
    pos = Property(desc='x, y, z positions of grid points')

    #: A unique identifier for the grid, based on its properties. (read-only)
    digest = Property(
        depends_on=['loc', 'direction', 'length', 'num_points', 'size'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property_depends_on(['num_points'])
    def _get_size(self):
        return self.pos.shape[-1]

    @property_depends_on(['num_points'])
    def _get_shape(self):
        return self.pos.shape[-1]

    @property_depends_on(['num_points', 'length', 'direction', 'loc'])
    def _get_pos(self):
        dist = self.length / (self.num_points - 1)
        loc = array(self.loc, dtype=float).reshape((3, 1))
        direc_n = array(self.direction) / norm(self.direction)
        pos = zeros((self.num_points, 3))
        for s in range(self.num_points):
            pos[s] = loc.T + direc_n * dist * s
        return pos.T


@deprecated_alias({'gpos': 'pos'}, read_only=True)
class MergeGrid(Grid):
    """
    Base class for merging multiple grid geometries.

    The `MergeGrid` class allows the combination of multiple grid geometries into a single unified
    grid. Each input grid is assigned a subdomain in the resulting grid, and all properties, such as
    positions and identifiers, are appropriately merged.

    Notes
    -----
    - The merged grid eliminates duplicate points based on their positions.
    - Each subgrid retains its original grid properties, such as digest and size.

    Examples
    --------
    Merging two simple grids:

    >>> import acoular as ac
    >>> grid1 = ac.LineGrid(loc=(0, 0, 0), direction=(1, 0, 0), length=1, num_points=3)
    >>> grid2 = ac.LineGrid(loc=(0, 0, 0), direction=(0, 1, 0), length=1, num_points=3)
    >>> merged_grid = ac.MergeGrid()
    >>> merged_grid.grids = [grid1, grid2]
    >>> merged_grid.size
    5
    >>> merged_grid.pos
    array([[0. , 0. , 0. , 0.5, 1. ],
           [0. , 0.5, 1. , 0. , 0. ],
           [0. , 0. , 0. , 0. , 0. ]])
    """

    #: A list of :class:`Grid` objects to be merged. Each grid is treated as a subdomain in the
    #: resulting merged grid.
    grids = List(desc='list of grids')

    #: A list of unique digests for each grid being merged. (read-only)
    grid_digest = Str(desc='digest of the merged grids')

    #: Names of subgrids corresponding to each point in the merged grid. (read-only)
    subgrids = Property(desc='names of subgrids for each point')

    #: A unique identifier for the grid, based on its properties. (read-only)
    digest = Property(depends_on=['grids', 'grid_digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @observe('grids.items.digest')
    def _set_sourcesdigest(self, event):  # noqa ARG002
        self.grid_digest = ldigest(self.grids)

    @property_depends_on(['digest'])
    def _get_size(self):
        return self.pos.shape[-1]

    @property_depends_on(['digest'])
    def _get_shape(self):
        return self.pos.shape[-1]

    @property_depends_on(['digest'])
    def _get_subgrids(self):
        subgrids = zeros((1, 0), dtype=str)
        for grid in self.grids:
            subgrids = append(subgrids, tile(grid.__class__.__name__ + grid.digest, grid.size))
        return subgrids[:, newaxis].T

    @property_depends_on(['digest'])
    def _get_pos(self):
        bpos = zeros((3, 0))
        for grid in self.grids:
            bpos = append(bpos, grid.pos, axis=1)
        return unique(bpos, axis=1)


class Sector(ABCHasStrictTraits):
    """
    Abstract base class for all sector types.

    The :class:`Sector` class defines the common interface for all sector implementations. It serves
    as the base class for creating diverse sector geometries, each capable of determining whether
    specific grid points fall within its bounds.

    When used directly, this class represents a sector encompassing the entire grid, meaning all
    positions are considered valid.

    Notes
    -----
    This class is designed to be subclassed. Derived classes should override the :meth:`contains`
    method to implement specific sector geometries (e.g., circular, rectangular, or polygon shapes).

    Examples
    --------
    Load example data and set different Sectors for integration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    def contains(self, pos):
        """
        Check whether the given coordinates lie within the sector's bounds.

        This method determines if each column of the input array :attr:`pos` corresponds to a point
        that falls within the sector. For this base class, all points are considered within the
        sector.

        Parameters
        ----------
        pos : :class:`numpy.ndarray`
            A 2D array with shape `(3, N)`, where `N` is the number of grid points. Each column
            represents the ``x, y, z`` coordinates of a grid point.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            A 1D array of length `N`, where each entry indicates whether the corresponding
            column in :attr:`pos` lies within the sector's bounds.

        Examples
        --------
        >>> import numpy as np
        >>> import acoular as ac
        >>> sector = ac.Sector()
        >>> positions = np.array([[0, 1], [0, 0], [0, 0]])  # Two grid points
        >>> sector.contains(positions)
        array([ True,  True])
        """
        return ones(pos.shape[1], dtype=bool)


class SingleSector(Sector):
    """
    Base class for single sector types.

    Defines the common interface for all single sector classes. This class can serve as a base for
    various single sector implementations. When used directly, it defines a sector that encompasses
    the whole grid. It includes attributes for handling border inclusion, tolerance for sector
    borders, and default behavior when no points are inside the sector.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: If ``True``, grid points lying on the sector border are included in the sector. Default is
    #: ``True``.
    include_border = Bool(True, desc='include points on the border')

    #: The absolute tolerance to apply when determining if a grid point lies on the sector border.
    #: Default is ``1e-12``.
    abs_tol = Float(1e-12, desc='absolute tolerance for sector border')

    #: If ``True``, the ``contains`` method (as in :meth:`RectSector.contains`,
    #: :meth:`RectSector3D.contains`, :meth:`CircSector.contains`, and :meth:`PolySector.contains`)
    #: returns the nearest grid point if no grid points are inside the sector. Default is ``True``.
    default_nearest = Bool(True, desc='``contains`` method return nearest grid point to center if none inside sector')


class RectSector(SingleSector):
    """
    Class for defining a rectangular sector.

    Defines a rectangular sector either for 2D grids (rectangle in the XY-plane) or for 3D grids
    (rectangular cylindrical sector parallel to the z-axis). The sector is bounded by the
    specified :attr:`x_min`, :attr:`x_max`, :attr:`y_min`, and :attr:`y_max` positions, defining the
    lower and upper bounds of the rectangle along the x and y axes.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: The minimum x position of the rectangle. Default is ``-1.0``.
    x_min = Float(-1.0, desc='minimum x position of the rectangle')

    #: The maximum x position of the rectangle. Default is ``1.0``.
    x_max = Float(1.0, desc='maximum x position of the rectangle')

    #: The minimum y position of the rectangle. Default is ``-1.0``.
    y_min = Float(-1.0, desc='minimum y position of the rectangle')

    #: The maximum y position of the rectangle. Default is ``1.0``.
    y_max = Float(1.0, desc='maximum y position of the rectangle')

    def contains(self, pos):
        """
        Check if the coordinates in a given array lie within the rectangular sector.

        If no coordinate is inside, the nearest one to the rectangle center is returned if
        :attr:`~SingleSector.default_nearest` is ``True``.

        Parameters
        ----------
        pos : array of :class:`floats<float>`
            A `(3, N)` array containing the positions of `N` grid points.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            An array of shape (N,) indicating which of the given positions
            lie within the given sector.

        Examples
        --------
        >>> import acoular as ac
        >>> grid = ac.RectGrid(increment=2)
        >>> sec = ac.RectSector(x_min=0, y_min=0)
        >>> sec.contains(grid.pos)
        array([False, False, False,  True])
        """
        # make sure xmin is minimum etc
        xmin = min(self.x_min, self.x_max)
        xmax = max(self.x_min, self.x_max)
        ymin = min(self.y_min, self.y_max)
        ymax = max(self.y_min, self.y_max)

        abs_tol = self.abs_tol
        # get pos indices inside rectangle (* == and)
        if self.include_border:
            inds = (
                (pos[0, :] - xmin > -abs_tol)
                * (pos[0, :] - xmax < abs_tol)
                * (pos[1, :] - ymin > -abs_tol)
                * (pos[1, :] - ymax < abs_tol)
            )
        else:
            inds = (
                (pos[0, :] - xmin > abs_tol)
                * (pos[0, :] - xmax < -abs_tol)
                * (pos[1, :] - ymin > abs_tol)
                * (pos[1, :] - ymax < -abs_tol)
            )

        # if none inside, take nearest
        if ~inds.any() and self.default_nearest:
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            dr2 = (pos[0, :] - x) ** 2 + (pos[1, :] - y) ** 2
            inds[argmin(dr2)] = True

        return inds.astype(bool)


class RectSector3D(RectSector):
    """
    Class for defining a cuboid sector.

    This class extends the :class:`RectSector` class to define a cuboid sector, which can be used
    for 3D grids. The cuboid sector is defined by its bounds along the x, y, and z axes.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: The lower z position of the cuboid. Default is ``-1.0``.
    z_min = Float(-1.0, desc='minimum z position of the cuboid')

    #: The upper z position of the cuboid.  Default is ``1.0``.
    z_max = Float(1.0, desc='maximum z position of the cuboid')

    def contains(self, pos):
        """
        Check if the coordinates in a given array lie within the cuboid sector.

        The method checks if the points in the provided position array are within the cuboid
        defined by the bounds along the x, y, and z axes. If no point is inside the sector, and if
        :attr:`~SingleSector.default_nearest` is ``True``, the nearest point to the center of the
        cuboid is returned.

        Parameters
        ----------
        pos : array of :class:`floats<float>`
            A (3, N) array containing the positions of N grid points, where each point is
            represented by its x, y, and z coordinates.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            A boolean array of shape shape ``(N,)`` indicating which of the given positions lie
            within the cuboid sector. ``True`` if the grid point is inside the cuboid,
            otherwise ``False``.

        Examples
        --------
        >>> import acoular as ac
        >>> grid = ac.RectGrid3D(increment=2)
        >>> sec = ac.RectSector3D(x_min=0, y_min=0, z_min=0)
        >>> sec.contains(grid.pos)
        array([False, False, False, False, False, False, False,  True])
        """
        # make sure xmin is minimum etc
        xmin = min(self.x_min, self.x_max)
        xmax = max(self.x_min, self.x_max)
        ymin = min(self.y_min, self.y_max)
        ymax = max(self.y_min, self.y_max)
        zmin = min(self.z_min, self.z_max)
        zmax = max(self.z_min, self.z_max)

        abs_tol = self.abs_tol
        # get pos indices inside rectangle (* == and)
        if self.include_border:
            inds = (
                (pos[0, :] - xmin > -abs_tol)
                * (pos[0, :] - xmax < abs_tol)
                * (pos[1, :] - ymin > -abs_tol)
                * (pos[1, :] - ymax < abs_tol)
                * (pos[2, :] - zmin > -abs_tol)
                * (pos[2, :] - zmax < abs_tol)
            )
        else:
            inds = (
                (pos[0, :] - xmin > abs_tol)
                * (pos[0, :] - xmax < -abs_tol)
                * (pos[1, :] - ymin > abs_tol)
                * (pos[1, :] - ymax < -abs_tol)
                * (pos[2, :] - zmin > abs_tol)
                * (pos[2, :] - zmax < -abs_tol)
            )

        # if none inside, take nearest
        if ~inds.any() and self.default_nearest:
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            dr2 = (pos[0, :] - x) ** 2 + (pos[1, :] - y) ** 2
            inds[argmin(dr2)] = True

        return inds.astype(bool)


class CircSector(SingleSector):
    """
    Class for defining a circular sector.

    Defines a circular sector, which can be used for both 2D grids (as a circle in the XY-plane) or
    for 3D grids (as a cylindrical sector parallel to the z-axis). The sector is defined by its
    center position (:attr:`x`, :attr:`y`) and its radius :attr:`r`.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: The x position of the circle center. Default is ``0.0``.
    x = Float(0.0, desc='x position of the circle center')

    #: The y position of the circle center. Default is ``0.0``.
    y = Float(0.0, desc='y position of the circle center')

    #: Radius of the circle. Default is ``1.0``.
    r = Float(1.0, desc='radius of the circle')

    def contains(self, pos):
        """
        Check if the coordinates in a given array lie within the circular sector.

        The method calculates the squared distance of each point from the center of the circle and
        checks if it lies within the sector, considering the sector's radius :attr:`r`. If no point
        is inside and :attr:`~SingleSector.default_nearest` is ``True``, the nearest point outside
        the sector will be returned.

        Parameters
        ----------
        pos : array of :class:`floats<float>`
            A (3, N) array containing the positions of N grid points, where each point is
            represented by its x, y, and z coordinates.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            A boolean array of shape shape (N,) indicating which of the given positions lie within
            the circular sector. ``True`` if the grid point is inside the circular sector,
            otherwise ``False``.

        Examples
        --------
        >>> import acoular as ac
        >>> grid = ac.RectGrid(increment=1)
        >>> grid.pos
        array([[-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
               [-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
        >>> sec = ac.CircSector(x=1, y=1, r=0.5)
        >>> sec.contains(grid.pos)
        array([False, False, False, False, False, False, False, False,  True])
        """
        dr2 = (pos[0, :] - self.x) ** 2 + (pos[1, :] - self.y) ** 2
        # which points are in the circle?
        inds = dr2 - self.r**2 < self.abs_tol if self.include_border else dr2 - self.r**2 < -self.abs_tol

        # if there's no point inside
        if ~inds.any() and self.default_nearest:
            inds[argmin(dr2)] = True

        return inds


class PolySector(SingleSector):
    """
    Class for defining a polygon sector.

    Inherits from :class:`SingleSector` and provides functionality to define a polygonal sector on a
    2D grid.

    Notes
    -----
    The polygon is specified by the :class:`Polygon` class.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: List of coordinates representing the polygon's vertices. The coordinates must define a closed
    #: polygon like ``x1, y1, x2, y2, ... xn, yn``.
    edges = List(Float)

    def contains(self, pos):
        """
        Check if the coordinates in a given array lie within the polygon sector.

        If no coordinate is inside, the nearest one to the rectangle center is returned if
        :attr:`~SingleSector.default_nearest` is ``True``.

        Parameters
        ----------
        pos : array of :class:`floats<float>`
            A (3, N) array containing the positions of N grid points where each point is represented
            by its x, y, and z coordinates.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            A boolean array of shape `(N,)` indicating which of the given positions lie within the
            polygon sector. ``True`` if the grid point is inside the polygon, otherwise ``False``.

        Examples
        --------
        >>> import acoular as ac
        >>> grid = ac.RectGrid(increment=1)
        >>> grid.pos
        array([[-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
               [-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
        >>> sec = ac.PolySector(edges=[0, 0, 1, 0, 1, 1, 0, 1])
        >>> sec.contains(grid.pos)
        array([False, False, False, False,  True,  True, False,  True,  True])
        """
        poly = Polygon(array(self.edges).reshape(-1, 2)[:, 0], array(self.edges).reshape(-1, 2)[:, 1])
        dists = poly.is_inside(pos[0, :], pos[1, :])
        inds = dists >= -self.abs_tol if self.include_border else dists > 0

        # if none inside, take nearest
        if ~inds.any() and self.default_nearest:
            dr2 = array(self.edges).reshape(-1, 2).mean(0)
            inds[argmin(dr2)] = True

        return inds


class ConvexSector(SingleSector):
    """
    Class for defining a convex hull sector.

    This class defines a convex hull sector for 2D grids. The sector is created using a list of edge
    coordinates :attr:`edges` which represent the vertices of a polygon. The convex hull is the
    smallest convex shape that contains all the given vertices.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: List of edge coordinates that define the convex hull. The coordinates must define a closed
    #: polygon that forms the convex hull like `x1, y1, x2, y2, ... xn, yn`.
    edges = List(Float)

    def contains(self, pos):
        """
        Check if the coordinates in a given array lie within the convex sector.

        If no coordinate is inside, the nearest one to the rectangle center is returned if
        :attr:`~SingleSector.default_nearest` is ``True``.

        Parameters
        ----------
        pos : array of :class:`floats<float>`
            Array containing the positions of N grid points, shape `(3, N)`.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            An array of shape `(N,)` indicating which of the given positions
            lie within the given sector.

        Examples
        --------
        >>> import acoular as ac
        >>> grid = ac.RectGrid(increment=1)
        >>> grid.pos
        array([[-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
               [-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
        >>> sec = ac.ConvexSector(edges=[0, 0, 1, 0, 1, 1, 0, 1])
        >>> sec.contains(grid.pos)
        array([False, False, False, False,  True,  True, False,  True,  True])
        """
        inds = in_hull(pos[:2, :].T, array(self.edges).reshape(-1, 2), border=self.include_border, tol=self.abs_tol)

        # if none inside, take nearest
        if ~inds.any() and self.default_nearest:
            dr2 = array(self.edges).reshape(-1, 2).mean(0)  # Use the centroid of the polygon as the "center"
            inds[argmin(dr2)] = True

        return inds


class MultiSector(Sector):
    """
    Class for defining a sector consisting of multiple sectors.

    This class allows the combination of several individual sectors into one.

    Examples
    --------
    Load example data and set diffrent Sectors for intergration in the
    :ref:`sector integration example<sector_integration_example>`.
    """

    #: List of :class:`Sector` objects to be mixed, each defining a different sector.
    sectors = List(Instance(Sector))

    def contains(self, pos):
        """
        Check if the coordinates in a given array lie within any of the sub-sectors.

        This method iterates over the list of sectors, checking if each point in the given position
        array lies within any of the defined sectors.

        Parameters
        ----------
        pos : array of :class:`floats<float>`
            A (3, N) array containing the positions of N grid points, where each point is
            represented by its x, y, and z coordinates.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`bools<bool>`
            A boolean array of shape `(N,)` indicating which of the given positions lie within any
            of the defined sectors. ``True`` if the grid point is inside the circular sector,
            ``False`` if otherwise.

        Examples
        --------
        >>> import acoular as ac
        >>> grid = ac.RectGrid(increment=1)
        >>> grid.pos
        array([[-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
               [-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
               [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.]])
        >>> sec1 = ac.RectSector(x_min=0, y_min=0)
        >>> sec2 = ac.CircSector(x=1, y=1, r=0.5)
        >>> multi_sec = ac.MultiSector(sectors=[sec1, sec2])
        >>> multi_sec.contains(grid.pos)
        array([False, False, False, False,  True,  True, False,  True,  True])
        """
        # initialize with only "False" entries
        inds = zeros(pos.shape[1], dtype=bool)

        # add points contained in each sector
        for sec in self.sectors:
            inds += sec.contains(pos)

        return inds.astype(bool)
