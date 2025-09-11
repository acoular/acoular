# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""
Implements methods required for directivity shared by source
"""

from abc import abstractmethod
import numpy as np
import scipy.linalg as spla
from traits.api import CArray, Enum, Float, ABCHasStrictTraits, Instance, List, Property, Str, cached_property

# acoular imports
from .internal import digest
from .microphones import MicGeom
from .sources import PointSource


def get_angle_to_target(src_locs, src_orientations, target_locs):
    """
    This function computes the azimuths and elevations of target points to the source points
    relative to the sources orientation.

    Parameters
    ----------
    src_locs : :class:`np.array`
        These are the (x, y, z) coordinates of the sources - this array must be supplied as shape
        (N, 3) where N is the number of sources.

    src_orientation : :class:`np.array`
        These are the right, up and forward vectors of the sources - these vectors must be orthogonal
        and normalised. The layout of the matrix is shown below. The shape of the array must be (N,3,3) where N is the number of sources.
        [
            [
                [right_x,  right_y,    right_z],
                [up_x,     up_y,       up_z],
                [fwd_x,    fwd_y,      fwd_z]
            ],
            [...], [...]
        ]

    target_locs : :class:`np.array`
        These are the (x, y, z) coordinates of the targets the angles are calculated for - this array must be
        supplied as shape(M,3) where M is the number of targets.

    Yields
    ------
    :class:`Tuple(np.ndarray, np.ndarray)`
        A returns a tuple of two np.ndarrays. The first array returns the azimuths of targets relative to the sources,
        the second tuple returns the elevation of the targets to the sources.

    """

    directions = target_locs - src_locs # in global space
    # Convert these directions to local coordinate space of the src
    # @TODO check this logic - 3x3 orientation matrix should be the transformation matrix if each dimension is normalised?
    directions = np.matvec(src_orientations, directions)
    azimuth = np.arctan2(directions[:,0], directions[:,2])
    elevation = np.arctan2(directions[:,1], np.hypot(directions[:,0], directions[:,2]))

    return azimuth, elevation


class Directivity(ABCHasStrictTraits):
    """
    Abstract base class for directivity calculation.
    """
    #: Vectors defining the global orientation of the object
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientation = CArray(shape=(3, 3), desc='source orientation matrix', default=np.eye(3))

    def _validate_orientation(self):
        if not np.allclose(np.dot(self.orientation, self.orientation.T), np.eye(3), atol=1e-12):
            raise ValueError('Orientation matrix must be orthogonal.')

    #: Vector defining the direction of the object we are working out direction to. Default is (0, 0, 1)
    target_directions = CArray(shape=(3, None), default=np.array([[0.0], [0.0], [1.0]]), desc='Directions of the other objects to which the directivity is to be calculated.')

    coefficients = Property(desc='Directivity coefficients', depends_on=['orientation', 'object_directions'])


class OmniDirectivity(Directivity):
    @cached_property
    def _get_coefficients(self):
        return np.ones(self.target_directions.shape[1], dtype=float)


class CardioidDirectivity(Directivity):
    @cached_property
    def _get_coefficients(self):
        obj_dir_norm = self.object_directions / np.linalg.norm(self.object_directions, axis=0, keepdims=True)
        return (self.orientation[2].reshape(3, 1).T @ obj_dir_norm + 1) / 2


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

    # Type of DirecetivityCalculator used to calculate directivity which will be instantiated later
    dir_calc = Instance(Directivity)

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
        self.dir_calc._validate_orientation()
        self._validate_locations()

        # object directions do not change once set
        self.dir_calc.target_directions = self.mics.pos - np.array(self.loc).reshape(3, 1)

        # -----------------------------------------------------------------------------------------
        # For now lets do speherical harmonic stuff in here and we will move it
        # later to somewhere more sensible.
        # We need to:
        # - Work out angle of the source from the mics (also the inverse for directional sources
        #   but for now will assume omnidirectional source)
        # - calculate spherical harmonics for the modes (need to specify number of order somewhere
        #   with 1st until we clean and move this)
        # - based on source direction work out the attenuation for each mode and output to a channel
        #
        # Later on:
        # - clean up - move this somewhere else work out tidy way to do this
        # - frequency dependant directionality
        # -----------------------------------------------------------------------------------------

        if isinstance(self.mics, MicGeomDirectional):
            # @TODO change from lists and speed up by vectorising and to not iterate over each mic
            azimuths = np.empty(shape=self.mics.num_mics)
            elevations = np.empty(shape=self.mics.num_mics)

            mic_pos = self.mics.pos.T
            src_pos = np.array(self.loc).reshape(1, 3)

            azimuths, elevations = get_angle_to_target(mic_pos, self.mics.orientations, src_pos)

            print(azimuths, elevations)

        # generate output
        signal = self.signal.usignal(self.up)
        out = np.empty((num, self.num_channels))
        # distances
        rm = self.env._r(np.array(self.loc).reshape((3, 1)), self.mics.pos).reshape(1, -1)
        ind = (-rm / self.env.c - self.start_t + self.start) * self.sample_freq

        i = 0
        n = self.num_samples

        while n:
            n -= 1
            try:
                print(self._calc_rotation_matrix(ind[0,:]).shape)
                self.dir_calc.orientation = (self._calc_rotation_matrix(ind[0,:]) @ self.dir_calc.orientation).T
                coeffs = self.dir_calc()
                #coeffs = np.ones(self.mics.num_mics)

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


class RotatingPointSource(PointSourceDirectional):
    # Rotation speed in radians/sec - default is 0
    rot_speed = Float(0.0)

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

class MicGeomDirectional(MicGeom):
    """
    Extension of MicGeom where directivity can be specified for each microphone
    """

    # @TODO none of the XML parsing mechanics has been implemented
    # @TODO decide how to store/represent each directivity

    # @TODO don't store these as strings
    #: Array containing directivity for each microphone, including invalid ones
    directivities_total = List(Enum('omni', 'cardioid'), desc='directivity for each microphone')

    # @TODO don't store these as strings
    #: Array containing directivity for each microphone, excluding those in invalid_channels (read-only)
    directivities = Property(depends_on=['pos_total', 'invalid_channels'], desc='directivity for each microphone')

    def _get_directivities(self):
        return [self.directivities_total[i] for i in self._valid_channels]

    #: Vectors defining the local orientation of the microphones relative to global space
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientations_total = CArray(dtype=float, shape=(None,3,3), desc='orientations for each microphone')

    orientations = Property(depends_on=['orientations_total', 'invalid_channels'],
                            desc='orientation for each microphone')

    def _get_orientations(self):
        return self.orientations_total[self._valid_channels]
