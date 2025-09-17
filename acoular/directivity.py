# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------

"""
Implements methods required for directivity shared by source
"""

from abc import abstractmethod
import numpy as np
import scipy.linalg as spla
from scipy.special import factorial, sph_harm_y_all
from traits.api import CArray, Enum, Float, Int, ABCHasStrictTraits, Instance, List, Property, Str, cached_property

# acoular imports
from .internal import digest
from .microphones import MicGeom
from .sources import PointSource


def cart2sph(coordinates):
    x = coordinates[:,0]
    y = coordinates[:,1]
    z = coordinates[:,2]

    hypot_xz = np.hypot(x, z)

    azimuth = np.mod(np.arctan2(x, z), 2 * np.pi)
    r = np.hypot(hypot_xz, y)
    elevation = np.arccos(y / r)

    return azimuth, elevation, r

def num_channels_for_sph_degree(n):
    return (n+1)**2

def sph_harm_complex_to_real(n, m, harm_array):
    result = None
    if m == 0:
        result = harm_array[n, m]
    elif m < 0:
        result = np.sqrt(2) * (-1 ** m) * harm_array[n, -m].imag
    elif m > 0:
        result = np.sqrt(2) * (-1 ** m) * harm_array[n, m].real
    return result

def sn3n_norm_factor(n, m):
    delta = 1 if m == 0 else 0
    return np.sqrt((2 - delta) * (factorial(n - np.abs(m)) / factorial(n + np.abs(m))))

def squash_sph_harm_array(harm_array):
    original_shape = harm_array.shape
    n_max = original_shape[0] - 1
    num_coeffs = num_channels_for_sph_degree(n_max)

    # Preserve trailing dimensions if present
    trailing_shape = original_shape[2:] if len(original_shape) > 2 else ()
    result_shape = (num_coeffs,) + trailing_shape
    result = np.empty(result_shape)

    # Transforming the complex spherical harmonics to real spherical harmonics
    i = 0
    for n in range(n_max + 1):
        for m in range(-n, n+1):
            result[i] = sph_harm_complex_to_real(n, m, harm_array)
            result *= sn3n_norm_factor(n, m)
            i+=1

    return result


class Directivity(ABCHasStrictTraits):
    """
    Abstract base class for directivity calculation.
    """
    #: Vectors defining the global orientation of the object
    #: These vectors must be orthogonal to each other
    #: self.orientation[0] = right_vec
    #: self.orientation[1] = up_vec
    #: self.orientation[2] = forward_vec
    orientation = CArray(shape=(3, 3), desc='source orientation matrix', value=np.eye(3))

    def _validate_orientation(self):
        if not np.allclose(np.dot(self.orientation, self.orientation.T), np.eye(3), atol=1e-12):
            raise ValueError('Orientation matrix must be orthogonal.')

    # TODO: QUESTION: Is it good to have target_directions and orientation as attributes? Shouldn't they be arguments to get_coefficients or similar?
    target_directions = Property(desc='Directions of the other objects to which the directivity is to be calculated.')

    _target_directions = CArray(shape=(3, None), default=np.array([[0.0], [0.0], [1.0]]))

    def _set_target_directions(self, value):
        self._target_directions = value / spla.norm(value, axis=0, keepdims=True)

    def _get_target_directions(self):
        return self._target_directions

    coefficients = Property(desc='Directivity coefficients', depends_on=['orientation', 'target_directions'])

    @abstractmethod
    def _get_coefficients(self):
        """
        Abstract method to calculate directivity coefficients.
        """
        pass


class OmniDirectivity(Directivity):
    @cached_property
    def _get_coefficients(self):
        return np.ones(self.target_directions.shape[1], dtype=float)


class CardioidDirectivity(Directivity):
    @cached_property
    def _get_coefficients(self):
        return 0.5 * (1.0 + np.dot(self.orientation[2], self.target_directions))


class SphericalHarmonicDirectivity(Directivity):
    """
    Define directivity for all orders of spherical harmonics given a degree.
    """
    n = Int(1)

    @cached_property
    def _get_coefficients(self):
        target_directions_local = np.matvec(self.orientation.reshape(-1, 3, 3), self.target_directions.T.reshape(-1, 3))
        azimuth, elevation, _ = cart2sph(target_directions_local)
        sph_harms = sph_harm_y_all(self.n, self.n, elevation, azimuth)
        sph_harms = squash_sph_harm_array(sph_harms)
        return sph_harms


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

    directivity = Instance(Directivity)

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
        self._validate_locations()

        # object directions do not change once set
        self.directivity.target_directions = self.mics.pos - np.array(self.loc).reshape(3, 1)

        # additional output channels may be added if we are using sph_harm recievers
        additional_channels = 0
        # initialising mic_coeffs here in the case we are not using MicGeomDirectional
        mic_coeffs = np.ones(self.mics.num_mics)

        # If we are using MicGeomDirectional - we work out the mic directivity coefficients here
        # If SpehericalHarmonicDirectivity is used we will add additional channels to the output
        if isinstance(self.mics, MicGeomDirectional):
            mic_pos = self.mics.pos
            src_pos = np.array(self.loc)

            # if sph_harm is the only directivity all coeffs are generated from that object:
            if self.mics.num_mics == 1 and isinstance(self.mics.directivities[0], SphericalHarmonicDirectivity):
                sph_harm_calc = self.mics.directivities[0]
                sph_harm_calc.target_directions = (src_pos - mic_pos[0]).reshape(3, 1)
                # create additional channels based on the spherical harmonics order
                additional_channels = num_channels_for_sph_degree(sph_harm_calc.n) - 1
                mic_coeffs = sph_harm_calc.coefficients.squeeze()
            # otherwise each directivity is used for each individual mic coeff
            else:
                # @Note - Is there a way to speed this up to prevent iterating over individual Directivities
                for m, directivity in enumerate(self.mics.directivities):
                    if isinstance(directivity, SphericalHarmonicDirectivity):
                        raise RuntimeError(f'SphericalHarmonicDirectivity can currently only be used if it is the only mic in the Geom')
                    else:
                        directivity.target_directions = (src_pos - mic_pos[m]).reshape(3, 1)
                        coeff = directivity.coefficients
                        assert(len(coeff) == 1)
                        mic_coeffs[m] = coeff

        # generate output
        signal = self.signal.usignal(self.up)
        out = np.empty((num, self.num_channels + additional_channels))
        # distances
        rm = self.env._r(np.array(self.loc).reshape((3, 1)), self.mics.pos).reshape(1, -1)
        ind = (-rm / self.env.c - self.start_t + self.start) * self.sample_freq

        # incorporate directivity
        rm /= self.directivity.coefficients

        i = 0
        n = self.num_samples

        while n:
            n -= 1
            try:
                out[i] = (signal[np.array(0.5 + ind * self.up, dtype=np.int64)] / rm) * mic_coeffs
                ind += 1.0
                i += 1
                if i == num:
                    yield out
                    out = np.zeros((num, self.num_channels + additional_channels))
                    i = 0
            except IndexError:
                break
        yield out[:i]


class RotatingPointSource(PointSourceDirectional):
    # Rotation speed in radians/sec - default is 0
    rot_speed = Float(0.0)

    def result(self, num=128):
        self._validate_locations()

        # object directions do not change once set
        self.directivity.target_directions = self.mics.pos - np.array(self.loc).reshape(3, 1)

        # additional output channels may be added if we are using sph_harm recievers
        additional_channels = 0
        # initialising mic_coeffs here in the case we are not using MicGeomDirectional
        mic_coeffs = np.ones(self.mics.num_mics)

        # If we are using MicGeomDirectional - we work out the mic directivity coefficients here
        # If SpehericalHarmonicDirectivity is used we will add additional channels to the output
        if isinstance(self.mics, MicGeomDirectional):
            mic_pos = self.mics.pos
            src_pos = np.array(self.loc)

            # if sph_harm is the only directivity all coeffs are generated from that object:
            if self.mics.num_mics == 1 and isinstance(self.mics.directivities[0], SphericalHarmonicDirectivity):
                sph_harm_calc = self.mics.directivities[0]
                sph_harm_calc.target_directions = (src_pos - mic_pos[0]).reshape(3, 1)
                # create additional channels based on the spherical harmonics order
                additional_channels = num_channels_for_sph_degree(sph_harm_calc.n) - 1
                mic_coeffs = sph_harm_calc.coefficients.squeeze()
            # otherwise each directivity is used for each individual mic coeff
            else:
                # @Note - Is there a way to speed this up to prevent iterating over individual Directivities
                for m, directivity in enumerate(self.mics.directivities):
                    if isinstance(directivity, SphericalHarmonicDirectivity):
                        raise RuntimeError(f'SphericalHarmonicDirectivity can currently only be used if it is the only mic in the Geom')
                    else:
                        directivity.target_directions = (src_pos - mic_pos[m]).reshape(3, 1)
                        coeff = directivity.coefficients
                        assert(len(coeff) == 1)
                        mic_coeffs[m] = coeff

        # generate output
        signal = self.signal.usignal(self.up)
        out = np.empty((num, self.num_channels + additional_channels))
        # distances
        rm = self.env._r(np.array(self.loc).reshape((3, 1)), self.mics.pos).reshape(1, -1)
        ind = (-rm / self.env.c - self.start_t + self.start) * self.sample_freq

        i = 0
        n = self.num_samples

        # for constant rotation, the rotation matrix in each step stays the same
        rad = self.rot_speed / self.sample_freq
        sin_a, cos_a = np.sin(rad), np.cos(rad)
        rotation_matrix = np.array([[cos_a, 0.0, sin_a],
                                    [0.0, 1.0, 0.0],
                                    [-sin_a, 0.0, cos_a]])

        while n:
            n -= 1
            try:
                self.directivity.orientation = rotation_matrix @ self.directivity.orientation
                out[i] = (signal[np.array(0.5 + ind * self.up, dtype=np.int64)] * self.directivity.coefficients / rm) * mic_coeffs
                ind += 1.0
                i += 1
                if i == num:
                    yield out
                    out = np.zeros((num, self.num_channels + additional_channels))
                    i = 0
            except IndexError:
                break
        yield out[:i]


# @TODO none of the XML parsing mechanics has been implemented
class MicGeomDirectional(MicGeom):
    """
    Extension of MicGeom where directivity can be specified for each microphone
    """

    #: Array containing directivity for each microphone, including invalid ones
    directivities_total = List(Instance(Directivity, ()), desc='directivity for each microphone')

    directivities = Property(depends_on=['directivities_total', 'invalid_channels'], desc='directivity for each microphone')

    def _get_directivities(self):
        return [self.directivities_total[i] for i in self._valid_channels]

    def _get_orientations(self):
        return self.orientations_total[self._valid_channels]
