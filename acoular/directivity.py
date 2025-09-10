# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""
Implements methods required for directivity shared by source
"""

import numpy as np
import scipy.linalg as spla
from traits.api import CArray, Enum, Float, HasTraits, Instance, Property, Str, cached_property

# acoular imports
from .internal import digest
from .microphones import MicGeom
from .sources import PointSource


def get_radiation_angles(direction, mpos, sourceposition):
    r"""
    Calculate the azimuthal and elevation angles between the microphones and the source.

    The function computes the azimuth (``azi``) and elevation (``ele``) angles between each
    microphone position and the source position, taking into account the orientation of the
    spherical harmonics provided by the parameter ``direction``.

    Parameters
    ----------
    direction : :class:`numpy.ndarray` of shape ``(3,)``
        Unit vector representing the spherical harmonic orientation. It should be a 3-element array
        corresponding to the ``x``, ``y``, and ``z`` components of the direction.
    mpos : :class:`numpy.ndarray` of shape ``(3, N)``
        Microphone positions in a 3D Cartesian coordinate system. The array should have 3 rows (the
        ``x``, ``y`` and ``z`` coordinates) and ``N`` columns (one for each microphone).
    sourceposition : :class:`numpy.ndarray` of shape ``(3,)``
        Position of the source in a 3D Cartesian coordinate system. It should be a 3-element array
        corresponding to the ``x``, ``y``, and ``z`` coordinates of the source.

    Returns
    -------
    azi : :class:`numpy.ndarray` of shape ``(N,)``
        Azimuth angles in radians between the microphones and the source. The range of the values is
        :math:`[0, 2\pi)`.
    ele : :class:`numpy.ndarray` of shape ``(N,)``
        Elevation angles in radians between the microphones and the source. The range of the values
        is :math:`[0, \pi]`.

    See Also
    --------
    :func:`numpy.linalg.norm` :
        Computes the norm of a vector.
    :func:`numpy.arctan2` :
        Computes the arctangent of two variables, preserving quadrant information.

    Notes
    -----
    - The function accounts for a coordinate system transformation where the ``z``-axis in Acoular
      corresponds to the ``y``-axis in spherical coordinates, and the ``y``-axis in Acoular
      corresponds to the ``z``-axis in spherical coordinates.
    - The elevation angle (``ele``) is adjusted to the range :math:`[0, \pi]` by adding
      :math:`\pi/2` after the initial calculation.

    Examples
    --------
    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> direction = [1, 0, 0]
    >>> mpos = np.array([[1, 2], [0, 0], [0, 1]])  # Two microphones
    >>> sourceposition = [0, 0, 0]
    >>> azi, ele = ac.sources.get_radiation_angles(direction, mpos, sourceposition)
    >>> azi
    array([0.       , 5.8195377])
    >>> ele
    array([4.71238898, 4.71238898])
    """
    # direction of the Spherical Harmonics
    direc = np.array(direction, dtype=float)
    direc = direc / spla.norm(direc)
    # distances
    source_to_mic_vecs = mpos - np.array(sourceposition).reshape((3, 1))
    source_to_mic_vecs[2] *= -1  # invert z-axis (acoular)    #-1
    # z-axis (acoular) -> y-axis (spherical)
    # y-axis (acoular) -> z-axis (spherical)
    # theta
    ele = np.arctan2(np.sqrt(source_to_mic_vecs[0] ** 2 + source_to_mic_vecs[2] ** 2), source_to_mic_vecs[1])
    ele += np.arctan2(np.sqrt(direc[0] ** 2 + direc[2] ** 2), direc[1])
    ele += np.pi * 0.5  # convert from [-pi/2, pi/2] to [0,pi] range
    # phi
    azi = np.arctan2(source_to_mic_vecs[2], source_to_mic_vecs[0])
    azi += np.arctan2(direc[2], direc[0])
    azi = np.mod(azi, 2 * np.pi)
    return azi, ele


class DirectivityCalculator(HasTraits):
    """
    A baseclass with an execute method which calculates the directivity based on directions of source and recievers

    TODO: This would only work with directivity which needs only a fwd and direction vector at the moment
          - Need to rethink the interface to allow for more complex directivity which may vary between dimensions
    """

    #: Vector defining the forward direction of the object we are working out directivity from. Default is (0, 0, 1)
    fwd_directions = CArray(shape=(3, None), default=np.array([[0.0], [0.0], [1.0]]), desc='Forward directions of object we are calulating directivity for')

    #: Vector defining the direction of the object we are working out direction to. Default is (0, 0, 1)
    object_directions = CArray(shape=(3, None), default=np.array([[0.0], [0.0], [1.0]]), desc='Directions of the other objects to which we are calculating the directivity')

    # Method which returns a scalar value to be used to attenuate the signal
    def __call__(self):
        return 1


class PointSourceDirectional(PointSource):
    """
    Define a fixed point source with directivity.

    The :class:`PointSourceDirectional` class simulates a fixed point source with directivity.

    The generated output is available via the :meth:`result` generator.

    See Also
    --------
    :class:`acoular.sources.PointSource` : For modeling stationary point sources.

    Notes
    -----
    - Directivity of the sources to be simulated later on, currently this class just
      calculates directions between sources and recievers.
    - Currently direction of the microphones are hardcoded.
    """

    #: Vectors defining the local orientation of the source relative to global space
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientation = CArray(shape=(3, 3), desc='source orientation matrix')

    def _validate_orientation(self):
        if not np.allclose(np.dot(self.orientation, self.orientation.T), np.eye(3), atol=1e-12):
            raise ValueError('Orientation matrix must be orthogonal.')

    # Rotation speed in radians/sec - default is 0
    rot_speed = Float(0.0)

    #: Behavior of the signal for negative time indices. Currently only supports `loop`. Default is
    #: ``'loop'``.
    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

    # Type of DirecetivityCalculator used to calculate directivity which will be instantiated later
    dir_calc = Instance(DirectivityCalculator)

    #: A unique identifier for the current state of the source, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'mics.digest',
            'signal.digest',
            'loc',
            'env.digest',
            'start_t',
            'start',
            'up',
            'forward_vec',
            'up_vec',
            'right_vec',
            'rot_speed'
            'prepadding',
            'src_directivity_calc',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def _calc_rotation_matrix(self, sample_index):
        # Calculates the 3D rotation matrix for a specific audio sample assuming the rotation is constant
        # TODO: ensure this works with large numbers
        time = sample_index / self.sample_freq
        rot_angle = self.rot_speed * time

        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)

        # build rotation matrix around the y axis
        rot_mats = np.array([[cos_a,                np.zeros_like(cos_a),   sin_a],
                             [np.zeros_like(cos_a), np.ones_like(cos_a),    np.zeros_like(cos_a)],
                             [-sin_a,               np.zeros_like(cos_a),   cos_a]])

        if rot_mats.ndim == 3:
            rot_mats = rot_mats.transpose(2, 0, 1)

        return rot_mats

    def result(self, num=128):
        """
        Generate output signal at microphones in blocks.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to yield. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`~PointSource.num_channels`) containing the signal
            detected at the microphones. The last block may have fewer samples if
            :attr:`~PointSource.num_samples` is not a multiple of ``num``.

        Raises
        ------
        :obj:`IndexError`
            If no more samples are available from the source.

        Notes
        -----
        If samples are needed for times earlier than the source's :attr:`~PointSource.start_t`, the
        signal is taken from the end of the signal array, effectively looping the signal for
        negative indices.
        """
        self._validate_orientation()
        self._validate_locations()

        # calculate directions in global space
        gdirs_to_source = np.array(self.loc).reshape((3, 1)) - self.mics.pos

        # directions of microphones relative to sources ref and up vector
        rdirs_to_mic = self.orientation @ -gdirs_to_source

        # direction of sources in relative to microphones
        # @TODO set these values somewhere - later microphone in its own class?
        # @TODO currently all microphones have the same orientation
        mic_trans_mat = np.eye(3)
        rdirs_to_source = mic_trans_mat @ gdirs_to_source

        # @TODO do something with the directions, store these as an attribute?

        # generate output
        signal = self.signal.usignal(self.up)
        out = np.empty((num, self.num_channels))
        # distances
        rm = self.env._r(np.array(self.loc).reshape((3, 1)), self.mics.pos).reshape(1, -1)
        ind = (-rm / self.env.c - self.start_t + self.start) * self.sample_freq

        i = 0
        n = self.num_samples

        # object directions do not change once set
        self.dir_calc.object_directions = -gdirs_to_source

        while n:
            n -= 1
            try:
                self.dir_calc.fwd_directions = (self._calc_rotation_matrix(ind[0,:]) @ self.orientation[2]).T
                coeffs = self.dir_calc()

                out[i] = (signal[np.array(0.5 + ind * self.up, dtype=np.int64)] * coeffs) / rm
                ind += 1.0
                i += 1
                if i == num:
                    yield out
                    out = np.zeros((num, self.num_channels))
                    i = 0
            except IndexError:
                break
        yield out[:i]


class MicGeomDirectional(MicGeom):
    """
    Extension of MicGeom where directivity can be specified for each microphone
    """

    # @TODO none of the XML parsing mechanics has been implemented
    # @TODO decide how to store/represent each directivity

    # @TODO don't store these as strings
    #: Array containing directivity for each microphone, including invalid ones
    directivities_total = CArray(dtype=Str, desc='directivity for each microphone')

    # @TODO don't store these as strings
    #: Array containing directivity for each microphone, excluding those in invalid_channels (read-only)
    directivities = Property(depends_on=['pos_total', 'invalid_channels'], desc='directivity for each microphone')

    #: Vectors defining the local orientation of the microphones relative to global space
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientations_total = CArray(dtype=float, shape=(None,3,3), desc='orientations for each microphone')

    orientations = Property(depends_on=['orientations_total', 'invalid_channels'],
                            desc='orientation for each microphone')
