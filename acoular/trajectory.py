# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements the definition of trajectories.

.. autosummary::
    :toctree: generated/

    Trajectory
"""

# imports from other packages
from numpy import arange, array, r_, sort
from scipy.interpolate import splev, splprep
from traits.api import Dict, Float, HasStrictTraits, Property, Tuple, cached_property, property_depends_on

# acoular imports
from .internal import digest


class Trajectory(HasStrictTraits):
    """
    Represents a trajectory as a continuous curve derived from sampled points.

    The :class:`Trajectory` class computes a smooth, continuous path through a set of discrete
    points in space and time using spline interpolation. It also supports evaluating the trajectory
    and its derivatives at arbitrary time instants.

    It can be used to:
        - define the traveling path of a moving sound source, e.g. for microphone array data
          simulation (see :class:`~acoular.sources.MovingPointSource`)
        - move a source grid along a certain path to create a fixed focus
          (see :class:`~acoular.tbeamform.BeamformerTimeTraj`
          and :class:`~acoular.tbeamform.BeamformerCleantTraj`)

    Exemplary use can also be seen in the
    :ref:`rotating point source example<rotating_point_source>`.

    See Also
    --------
    :class:`~acoular.sources.MovingPointSource` : Model a point source moving along a trajectory.
    :class:`~acoular.sources.MovingPointSourceDipole` :
        Model a point source dipole moving along a trajectory.
    :class:`~acoular.sources.MovingLineSource` : Model a line source moving along a trajectory.
    :class:`~acoular.tbeamform.BeamformerCleantTraj` :
        Beamformer implementing the CLEAN method :cite:`Kujawski2020` in time domain
        for moving sources with known trajectory.
    :class:`~acoular.tbeamform.BeamformerTimeTraj` :
        Basic time domain beamformer with time signal output for a grid moving along a trajectory.
    :func:`scipy.interpolate.splprep` : Underlying spline generation function.
    :func:`scipy.interpolate.splev` : Used for evaluating the spline.

    Notes
    -----
    - Spline interpolation provides a smooth trajectory that passes through all sampled points.
      The interpolation order is adjusted automatically based on the number of points.
    - The trajectory can be used in simulations where a source's motion must be modeled
      continuously.

    Examples
    --------
    Create a trajectory and evaluate positions and velocities:

    >>> from acoular import Trajectory
    >>> points = {0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0), 2.0: (2.0, 1.0, 0.0)}
    >>> tr = Trajectory(points=points)
    >>>
    >>> tr.location(1.5)  # Position at t=1.5
    [array(1.5), array(0.375), array(0.)]
    >>>
    >>> for pos in tr.traj(0.0, 2.0, 0.5):  # Positions every 0.5 seconds
    ...     print(pos)
    (np.float64(0.0), np.float64(0.0), np.float64(0.0))
    (np.float64(0.5), np.float64(-0.125), np.float64(0.0))
    (np.float64(1.0), np.float64(0.0), np.float64(0.0))
    (np.float64(1.5), np.float64(0.375), np.float64(0.0))
    """  # noqa W505

    #: Dictionary mapping time instants (keys, as floats) to sampled ``(x, y, z)`` positions
    #: (values, as tuples of floats) along the trajectory.
    points = Dict(
        key_trait=Float,
        value_trait=Tuple(Float, Float, Float),
        desc='sampled positions along the trajectory',
    )

    #: Automatically determined tuple ``(t_min, t_max)`` representing the start and end times of the
    #: trajectory, based on the keys in :attr:`points`.
    interval = Property()
    # t_min, t_max tuple

    #: Internal representation of the spline, generated using :func:`scipy.interpolate.splprep`.
    tck = Property()

    #: A unique identifier for the trajectory, based on its points. (read-only)
    digest = Property(depends_on=['points[]'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @property_depends_on(['points[]'])
    def _get_interval(self):
        return sort(list(self.points.keys()))[r_[0, -1]]

    @property_depends_on(['points[]'])
    def _get_tck(self):
        t = sort(list(self.points.keys()))
        xp = array([self.points[i] for i in t]).T
        k = min(3, len(self.points) - 1)
        tcku = splprep(xp, u=t, s=0, k=k)
        return tcku[0]

    def location(self, t, der=0):
        """
        Compute the trajectory's position or derivatives at specified times.

        Parameters
        ----------
        t : :class:`float` or array of :class:`floats<float>`
            Time instant(s) at which to compute the position(s) or derivative(s).
        der : :class:`int`, optional
            Order of the derivative to compute:
                - ``0`` for positions (default),
                - ``1`` for velocities,
                - ``2`` for accelerations, etc.

        Returns
        -------
        :class:`numpy.ndarray`
            ``(x, y, z)`` arrays representing the trajectory's position (or derivative) at the given
            time(s). The shape matches that of ``t``.

        Examples
        --------
        >>> import acoular as ac
        >>>
        >>> points = {0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 2.0, 0.0), 2.0: (2.0, 4.0, 0.0)}
        >>> tr = ac.Trajectory(points=points)
        >>> tr.location(1.0)  # Position at t=1.0
        [array(1.), array(2.), array(0.)]
        >>> tr.location([0.5, 1.5], der=1)  # Velocity at t=0.5 and t=1.5
        [array([1., 1.]), array([2., 2.]), array([0., 0.])]
        """
        return splev(t, self.tck, der)

    def traj(self, t_start, t_end=None, delta_t=None, der=0):
        """
        Interate through trajectory positions or derivatives at specified intervals.

        Parameters
        ----------
        t_start : :class:`float`
            Start time for the trajectory. Default is earliest key in :attr:`points`.
        t_end : :class:`float`, optional
            End time of the trajectory. Default is the latest key in :attr:`points`.
        delta_t : :class:`float`, optional
            Time interval between consecutive points to yield. Default is the value of ``t_start``.
        der : int, optional
            Order of the derivative to compute:
                - ``0`` for positions (default),
                - ``1`` for velocities,
                - ``2`` for accelerations, etc.

        Yields
        ------
        :class:`tuple` of :class:`floats<float>`
            ``(x, y, z)`` positions or derivatives at the specified time intervals.

        Notes
        -----
        The function precomputes all interpolated locations for efficiency and yields them
        sequentially.

        Examples
        --------
        Create a trajectory and iterate through the positions in the :attr:`interval`:

        >>> import acoular as ac
        >>>
        >>> points = {0.0: (0.0, 0.0, 0.0), 1.0: (1.0, 0.0, 0.0), 2.0: (2.0, 1.0, 0.0)}
        >>> tr = ac.Trajectory(points=points)
        >>> for pos in tr.traj(0.0, 2.0, 0.5):
        ...     print(pos)
        (np.float64(0.0), np.float64(0.0), np.float64(0.0))
        (np.float64(0.5), np.float64(-0.125), np.float64(0.0))
        (np.float64(1.0), np.float64(0.0), np.float64(0.0))
        (np.float64(1.5), np.float64(0.375), np.float64(0.0))
        """
        if not delta_t:
            delta_t = t_start
            t_start, t_end = self.interval
        if not t_end:
            t_end = self.interval[1]
        # all locations are fetched in one go because that is much faster further improvement could
        # be possible if interpolated locations are fetched in blocks
        yield from zip(*self.location(arange(t_start, t_end, delta_t), der))
