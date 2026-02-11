"""Utility classes intended for internal use in Acoular.

.. autosummary::
    :toctree: generated/

    get_file_basename
    find_basename
    mole_fraction_of_water_vapor
    Polygon
"""

from pathlib import Path

import numpy as np


def _det(xvert, yvert):
    xvert = np.asarray(xvert, dtype=float)
    yvert = np.asarray(yvert, dtype=float)
    x_prev = np.concatenate(([xvert[-1]], xvert[:-1]))
    y_prev = np.concatenate(([yvert[-1]], yvert[:-1]))
    return np.sum(yvert * x_prev - xvert * y_prev, axis=0)


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
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        # Closes the polygon if it were open
        x1, y1 = x[0], y[0]
        xn, yn = x[-1], y[-1]
        if x1 != xn or y1 != yn:
            self.x = np.concatenate((self.x, [x1]))
            self.y = np.concatenate((self.y, [y1]))
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
        xpoint = np.asarray(xpoint, dtype=float)
        ypoint = np.asarray(ypoint, dtype=float)
        # Scalar to array
        if xpoint.shape == ():
            xpoint = np.array([xpoint], dtype=float)
            ypoint = np.array([ypoint], dtype=float)
            scalar = True
        else:
            scalar = False
        # Check consistency
        if xpoint.shape != ypoint.shape:
            msg = 'x and y have different shapes'
            raise IndexError(msg)
        # If snear = True: Dist to nearest side < nearest vertex
        # If snear = False: Dist to nearest vertex < nearest side
        snear = np.ma.masked_all(xpoint.shape, dtype=bool)
        # Initialize arrays
        mindst = np.ones_like(xpoint, dtype=float) * np.inf
        j = np.ma.masked_all(xpoint.shape, dtype=int)
        x = self.x
        y = self.y
        n = len(x) - 1  # Number of sides/vertices defining the polygon
        # Loop over each side defining polygon
        for i in range(n):
            d = np.ones_like(xpoint, dtype=float) * np.inf
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
        if np.ma.count(snear) != snear.size:
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
        mindst[~snear] = np.copysign(mindst, area)[~snear]
        # Point is closer to its nearest side than to its nearest vertex, check
        # if point is to left or right of this side.
        # If point is to left of side it is inside polygon, else point is
        # outside polygon.
        area = _det([x[j], x[j + 1], xpoint], [y[j], y[j + 1], ypoint])
        mindst[snear] = np.copysign(mindst, area)[snear]
        # Point is on side of polygon
        mindst[np.fabs(mindst) < smalld] = 0
        # If input values were scalar then the output should be too
        if scalar:
            mindst = float(mindst[0])
        return mindst


def get_file_basename(file, alternative_basename='void'):
    """Return the basename of the file.

    Parameters
    ----------
    file : str
        File path.

    Returns
    -------
    str
        Basename of the file.
    """
    return Path(file).stem if file else alternative_basename


def find_basename(source, alternative_basename='void'):
    """Return the basename of the original source.

    Traverses the source chain of the object and returns the basename of the original source.
    If the source object does not have a basename, uses the alternative basename.

    Parameters
    ----------
    source : instance
        :class:`~acoular.base.Generator` derived object
    alternative_basename : str
        Alternative basename to use if the source object does not have a basename.


    Returns
    -------
    str
        Basename of the original source.
    """
    while source:
        basename = getattr(source, 'basename', None)
        if basename is not None:
            return basename
        source = getattr(source, 'source', None)
    return alternative_basename


def mole_fraction_of_water_vapor(h, t, p=101325):
    r"""Mole fraction of water vapor in the air for real gases.

    Calculates the mole fraction of water vapor in air from the relative humidity,
    based on the equations provided in the appendix of :cite:`Cramer1993` and
    the enhancement factors from :cite:`Davis1992`.

    Parameters
    ----------
    h : float
        Relative humidity as a fraction [0,1].
    t : float
        Thermodynamic temperature in K.
    p : float
        Atmospheric pressure in Pa (default is the standard pressure 101325 Pa).

    Returns
    -------
    float
        Mole fraction of water vapor.

    Notes
    -----
    The mole fraction is calculated as:

    .. math::
        x_w = h \cdot f \cdot \frac{p_{sv}}{p},

    where:
      - :math:`h` is the relative humidity as a fraction [0,1].
      - :math:`f` is the enhancement factor:

        .. math::
            f = 1.00062 + 3.14 \times 10^{-8} \cdot p + 5.6 \times 10^{-7} \cdot t^2.

      - :math:`p_{sv}` is the saturation vapor pressure of water vapor in air:

        .. math::
            p_{sv} = \exp(A \cdot t^2 + B \cdot t + C + \frac{D}{t}),

        with the updated coefficients from :cite:`Davis1992`:

        .. math::
            A = 1.2378847 \times 10^{-5}, \\
            B = -1.9121316 \times 10^{-2}, \\
            C = 33.93711047, \\
            D = -6.3431645 \times 10^3.

    """
    f = 1.00062 + 3.14 * 10 ** (-8) * p + 5.6 * 10 ** (-7) * t**2  # enhancement factor
    # Saturation vapor pressure using updated coefficients from Davis (1992)
    A = 1.2378847 * 10 ** (-5)  # noqa: N806
    B = -1.9121316 * 10 ** (-2)  # noqa: N806
    C = 33.93711047  # noqa: N806
    D = -6.3431645 * 10**3  # noqa: N806
    p_sv = np.exp(A * t**2 + B * t + C + D / t)  # p_sv in Pa
    return h * f * p_sv / p
