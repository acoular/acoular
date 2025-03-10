# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Measured multichannel data management and simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    TimeSamples
    MaskedTimeSamples
    PointSource
    PointSourceDipole
    SphericalHarmonicSource
    LineSource
    MovingPointSource
    MovingPointSourceDipole
    MovingLineSource
    UncorrelatedNoiseSource
    SourceMixer
    PointSourceConvolve
    spherical_hn1
    get_radiation_angles
    get_modes
"""

# imports from other packages

import contextlib
from os import path
from warnings import warn

import numba as nb
from numpy import any as npany
from numpy import (
    arange,
    arctan2,
    array,
    ceil,
    complex128,
    cross,
    dot,
    empty,
    int64,
    mod,
    newaxis,
    ones,
    pi,
    real,
    repeat,
    sqrt,
    tile,
    uint32,
    zeros,
)
from numpy import min as npmin
from numpy.fft import fft, ifft
from scipy.linalg import norm
from scipy.special import sph_harm, spherical_jn, spherical_yn
from traits.api import (
    Any,
    Bool,
    CArray,
    CInt,
    Delegate,
    Dict,
    Enum,
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
)

from .base import SamplesGenerator

# acoular imports
from .calib import Calib
from .deprecation import deprecated_alias
from .environments import Environment
from .h5files import H5FileBase, _get_h5file_class
from .internal import digest, ldigest
from .microphones import MicGeom
from .signals import NoiseGenerator, SignalGenerator
from .tools.utils import get_file_basename
from .tprocess import TimeConvolve
from .trajectory import Trajectory


@nb.njit(cache=True, error_model='numpy')  # pragma: no cover
def _fill_mic_signal_block(out, signal, rm, ind, blocksize, num_channels, up, prepadding):
    if prepadding:
        for b in range(blocksize):
            for m in range(num_channels):
                if ind[0, m] < 0:
                    out[b, m] = 0
                else:
                    out[b, m] = signal[int(0.5 + ind[0, m])] / rm[0, m]
            ind += up
    else:
        for b in range(blocksize):
            for m in range(num_channels):
                out[b, m] = signal[int(0.5 + ind[0, m])] / rm[0, m]
            ind += up
    return out


def spherical_hn1(n, z):
    r"""
    Compute the spherical Hankel function of the first kind.

    The spherical Hankel function of the first kind, :math:`h_n^{(1)}(z)`, is defined as

    .. math:: h_n^{(1)}(z) = j_n(z) + i \cdot y_n(z)

    with the complex unit :math:`i`, the spherical Bessel function of the first kind as

    .. math:: j_n(z) = \sqrt{\frac{\pi}{2z}} J_{n + 1/2}(z),

    and the spherical Bessel function of the second kind as

    .. math:: y_n(z) = \sqrt{\frac{\pi}{2z}} Y_{n + 1/2}(z),

    where :math:`Y_n` is the Bessel function of the second kind.

    Parameters
    ----------
    n : :class:`int`, array_like
        Order of the spherical Hankel function. Must be a non-negative integer.
    z : complex or :class:`float`, array_like
        Argument of the spherical Hankel function. Can be real or complex.

    Returns
    -------
    complex or :class:`numpy.ndarray`
        Value of the spherical Hankel function of the first kind for the
        given order ``n`` and argument ``z``. If ``z`` is array-like, an array
        of the same shape is returned.

    See Also
    --------
    :func:`scipy.special.spherical_jn` : Computes the spherical Bessel function of the first kind.
    :func:`scipy.special.spherical_yn` : Computes the spherical Bessel function of the second kind.

    Notes
    -----
    - The function relies on :func:`scipy.special.spherical_jn` for the spherical Bessel function of
      the first kind and :func:`scipy.special.spherical_yn` for the spherical Bessel function of the
      second kind.
    - The input ``n`` must be a non-negative integer; otherwise, the behavior is undefined.

    Examples
    --------
    >>> import acoular as ac
    >>>
    >>> ac.sources.spherical_hn1(0, 1.0)
    np.complex128(0.8414709848078965-0.5403023058681398j)
    >>> ac.sources.spherical_hn1(1, [1.0, 2.0])
    array([0.30116868-1.38177329j, 0.43539777-0.350612j  ])
    """
    return spherical_jn(n, z, derivative=False) + 1j * spherical_yn(n, z, derivative=False)


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
    direc = array(direction, dtype=float)
    direc = direc / norm(direc)
    # distances
    source_to_mic_vecs = mpos - array(sourceposition).reshape((3, 1))
    source_to_mic_vecs[2] *= -1  # invert z-axis (acoular)    #-1
    # z-axis (acoular) -> y-axis (spherical)
    # y-axis (acoular) -> z-axis (spherical)
    # theta
    ele = arctan2(sqrt(source_to_mic_vecs[0] ** 2 + source_to_mic_vecs[2] ** 2), source_to_mic_vecs[1])
    ele += arctan2(sqrt(direc[0] ** 2 + direc[2] ** 2), direc[1])
    ele += pi * 0.5  # convert from [-pi/2, pi/2] to [0,pi] range
    # phi
    azi = arctan2(source_to_mic_vecs[2], source_to_mic_vecs[0])
    azi += arctan2(direc[2], direc[0])
    azi = mod(azi, 2 * pi)
    return azi, ele


def get_modes(lOrder, direction, mpos, sourceposition=None):  # noqa: N803
    """
    Calculate the spherical harmonic radiation pattern at microphone positions.

    This function computes the spherical harmonic radiation pattern values at each
    microphone position for a given maximum spherical harmonic order (``lOrder``),
    orientation (``direction``), and optional source position (``sourceposition``).

    Parameters
    ----------
    lOrder : :class:`int`
        The maximum order of spherical harmonics to compute. The resulting modes will include all
        orders up to and including ``lOrder``.
    direction : :class:`numpy.ndarray` of shape ``(3,)``
        Unit vector representing the orientation of the spherical harmonics. Should contain the
        ``x``, ``y``, and ``z`` components of the direction.
    mpos : :class:`numpy.ndarray` of shape ``(3, N)``
        Microphone positions in a 3D Cartesian coordinate system. The array should have 3 rows (the
        ``x``, ``y`` and ``z`` coordinates) and ``N`` columns (one for each microphone).
    sourceposition : :class:`numpy.ndarray` of shape ``(3,)``, optional
        Position of the source in a 3D Cartesian coordinate system. If not provided, it defaults to
        the origin ``[0, 0, 0]``.

    Returns
    -------
    :class:`numpy.ndarray` of shape ``(N, (lOrder+1) ** 2)``
        Complex values representing the spherical harmonic radiation pattern at each microphone
        position (``N`` microphones) for each spherical harmonic mode.

    See Also
    --------
    :func:`get_radiation_angles` :
        Computes azimuth and elevation angles between microphones and the source.
    :obj:`scipy.special.sph_harm` : Computes spherical harmonic values.

    Notes
    -----
    - The azimuth (``azi``) and elevation (``ele``) angles between the microphones and the source
      are calculated using the :func:`get_radiation_angles` function.
    - Spherical harmonics (``sph_harm``) are computed for each mode ``(l, m)``, where ``l`` is the
      degree (ranging from ``0`` to ``lOrder``) and ``m`` is the order
      (ranging from ``-l`` to ``+l``).
    - For negative orders (`m < 0`), the conjugate of the spherical harmonic is computed and scaled
      by the imaginary unit ``1j``.

    Examples
    --------
    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> lOrder = 2
    >>> direction = [0, 0, 1]  # Orientation along z-axis
    >>> mpos = np.array([[1, -1], [1, -1], [0, 0]])  # Two microphones
    >>> sourcepos = [0, 0, 0]  # Source at origin
    >>>
    >>> modes = ac.sources.get_modes(lOrder, direction, mpos, sourcepos)
    >>> modes.shape
    (2, 9)
    """
    sourceposition = sourceposition if sourceposition is not None else array([0, 0, 0])
    azi, ele = get_radiation_angles(direction, mpos, sourceposition)  # angles between source and mics
    modes = zeros((azi.shape[0], (lOrder + 1) ** 2), dtype=complex128)
    i = 0
    for lidx in range(lOrder + 1):
        for m in range(-lidx, lidx + 1):
            modes[:, i] = sph_harm(m, lidx, azi, ele)
            if m < 0:
                modes[:, i] = modes[:, i].conj() * 1j
            i += 1
    return modes


@deprecated_alias({'name': 'file'})
class TimeSamples(SamplesGenerator):
    """
    Container for processing time data in ``*.h5`` or NumPy array format.

    The :class:`TimeSamples` class provides functionality for loading, managing, and accessing
    time-domain data stored in HDF5 files or directly provided as a NumPy array. This data can be
    accessed iteratively through the :meth:`result` method, which returns chunks of the time data
    for further processing.

    See Also
    --------
    :class:`acoular.sources.MaskedTimeSamples` :
        Extends the functionality of class :class:`TimeSamples` by enabling the definition of start
        and stop samples as well as the specification of invalid channels.

    Notes
    -----
    - If a calibration object is provided, calibrated time-domain data will be returned.
    - Metadata from the :attr:`HDF5 file<file>` can be accessed through the :attr:`metadata`
      attribute.

    Examples
    --------
    Data can be loaded from a HDF5 file as follows:

    >>> from acoular import TimeSamples
    >>> file = <some_h5_file.h5>  # doctest: +SKIP
    >>> ts = TimeSamples(file=file)  # doctest: +SKIP
    >>> print(f'number of channels: {ts.num_channels}')  # doctest: +SKIP
    number of channels: 56 # doctest: +SKIP

    Alternatively, the time data can be specified directly as a NumPy array. In this case, the
    :attr:`data` and :attr:`~acoular.base.Generator.sample_freq` attributes must be set manually.

    >>> import numpy as np
    >>> data = np.random.rand(1000, 4)
    >>> ts = TimeSamples(data=data, sample_freq=51200)

    Chunks of the time data can be accessed iteratively via the :meth:`result` generator. The last
    block will be shorter than the block size if the number of samples is not a multiple of the
    block size.

    >>> blocksize = 512
    >>> generator = ts.result(num=blocksize)
    >>> for block in generator:
    ...     print(block.shape)
    (512, 4)
    (488, 4)
    """

    #: Full path to the ``.h5`` file containing time-domain data.
    file = File(filter=['*.h5'], exists=True, desc='name of data file')

    #: Basename of the ``.h5`` file, set automatically from the :attr:`file` attribute.
    basename = Property(depends_on=['file'], desc='basename of data file')

    #: Calibration data, an instance of the :class:`~acoular.calib.Calib` class.
    #: (optional; if provided, the time data will be calibrated.)
    calib = Instance(Calib, desc='Calibration data')

    #: Number of input channels in the time data, set automatically based on the
    #: :attr:`loaded data<file>` or :attr:`specified array<data>`.
    num_channels = CInt(0, desc='number of input channels')

    #: Total number of time-domain samples, set automatically based on the :attr:`loaded data<file>`
    #: or :attr:`specified array<data>`.
    num_samples = CInt(0, desc='number of samples')

    #: A 2D NumPy array containing the time-domain data, shape (:attr:`num_samples`,
    #: :attr:`num_channels`).
    data = Any(transient=True, desc='the actual time data array')

    #: HDF5 file object.
    h5f = Instance(H5FileBase, transient=True)

    #: Metadata loaded from the HDF5 file, if available.
    metadata = Dict(desc='metadata contained in .h5 file')

    # Checksum over first data entries of all channels
    _datachecksum = Property()

    #: A unique identifier for the samples, based on its properties. (read-only)
    digest = Property(
        depends_on=['basename', 'calib.digest', '_datachecksum', 'sample_freq', 'num_channels', 'num_samples']
    )

    def _get__datachecksum(self):
        return self.data[0, :].sum()

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return get_file_basename(self.file)

    @on_trait_change('basename')
    def _load_data(self):
        # Open the .h5 file and set attributes.
        if self.h5f is not None:
            with contextlib.suppress(OSError):
                self.h5f.close()
        file = _get_h5file_class()
        self.h5f = file(self.file)
        self._load_timedata()
        self._load_metadata()

    @on_trait_change('data')
    def _load_shapes(self):
        # Set :attr:`num_channels` and :attr:`num_samples` from data.
        if self.data is not None:
            self.num_samples, self.num_channels = self.data.shape

    def _load_timedata(self):
        # Loads timedata from :attr:`.h5 file<file>`. Only for internal use.
        self.data = self.h5f.get_data_by_reference('time_data')
        self.sample_freq = self.h5f.get_node_attribute(self.data, 'sample_freq')

    def _load_metadata(self):
        # Loads :attr:`metadata` from :attr:`.h5 file<file>`. Only for internal use.
        self.metadata = {}
        if '/metadata' in self.h5f:
            self.metadata = self.h5f.node_to_dict('/metadata')

    def result(self, num=128):
        """
        Generate blocks of time-domain data iteratively.

        The :meth:`result` method is a Python generator that yields blocks of time-domain data
        of the specified size. Data is either read from an HDF5 file (if :attr:`file` is set)
        or from a NumPy array (if :attr:`data` is directly provided). If a calibration object
        is specified, the returned data is calibrated.

        Parameters
        ----------
        num : :class:`int`, optional
            The size of each block to be yielded, representing the number of time-domain
            samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`num_channels`) representing a block of
            time-domain data. The last block may have fewer than ``num`` samples if the total number
            of samples is not a multiple of ``num``.

        Raises
        ------
        :obj:`OSError`
            If no samples are available (i.e., :attr:`num_samples` is ``0``).
        :obj:`ValueError`
            If the calibration data does not match the number of channels.

        Warnings
        --------
        A deprecation warning is raised if the calibration functionality is used directly in
        :class:`TimeSamples`. Instead, the :class:`~acoular.calib.Calib` class should be used as a
        separate processing block.

        Examples
        --------
        Create a generator and access blocks of data:

        >>> import numpy as np
        >>> from acoular.sources import TimeSamples
        >>> ts = TimeSamples(data=np.random.rand(1000, 4), sample_freq=51200)
        >>> generator = ts.result(num=256)
        >>> for block in generator:
        ...     print(block.shape)
        (256, 4)
        (256, 4)
        (256, 4)
        (232, 4)

        Note that the last block may have fewer that ``num`` samples.
        """
        if self.num_samples == 0:
            msg = 'no samples available'
            raise OSError(msg)
        self._datachecksum  # trigger checksum calculation # noqa: B018
        i = 0
        if self.calib:
            warn(
                'The use of the calibration functionality in TimeSamples is deprecated and will be removed in \
                       Acoular 25.10. Use the Calib class as an additional processing block instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if self.calib.num_mics == self.num_channels:
                cal_factor = self.calib.data[newaxis]
            else:
                msg = f'calibration data not compatible: {self.calib.num_mics:d}, {self.num_channels:d}'
                raise ValueError(msg)
            while i < self.num_samples:
                yield self.data[i : i + num] * cal_factor
                i += num
        else:
            while i < self.num_samples:
                yield self.data[i : i + num]
                i += num


@deprecated_alias(
    {
        'numchannels_total': 'num_channels_total',
        'numsamples_total': 'num_samples_total',
        'numchannels': 'num_channels',
        'numsamples': 'num_samples',
    },
    read_only=['numchannels', 'numsamples'],
)
class MaskedTimeSamples(TimeSamples):
    """
    Container to process and manage time-domain data with support for masking samples and channels.

    The :class:`MaskedTimeSamples` class extends the functionality of :class:`TimeSamples` by
    allowing the definition of :attr:`start` and :attr:`stop` indices for valid samples and by
    supporting invalidation of specific channels. This makes it suitable for use cases where only a
    subset of the data is of interest, such as analyzing specific time segments or excluding faulty
    sensor channels.

    See Also
    --------
    :class:`acoular.sources.TimeSamples` : The parent class for managing unmasked time-domain data.

    Notes
    -----
    Channels specified in :attr:`invalid_channels` are excluded from processing and not included in
    the generator output.

    Examples
    --------
    Data can be loaded from a HDF5 file and invalid channels can be specified as follows:

    >>> from acoular import MaskedTimeSamples
    >>> file = <some_h5_file.h5>  # doctest: +SKIP
    >>> ts = MaskedTimeSamples(file=file, invalid_channels=[0, 1])  # doctest: +SKIP
    >>> print(f'number of valid channels: {ts.num_channels}')  # doctest: +SKIP
    number of valid channels: 54 # doctest: +SKIP

    Alternatively, the time data can be specified directly as a numpy array.
    In this case, the :attr:`data` and :attr:`sample_freq` attributes must be set manually.

    >>> from acoular import MaskedTimeSamples
    >>> import numpy as np
    >>> data = np.random.rand(1000, 4)
    >>> ts = MaskedTimeSamples(data=data, sample_freq=51200)

    Chunks of the time data can be accessed iteratively via the :meth:`result` generator:

    >>> block_size = 512
    >>> generator = ts.result(num=block_size)
    >>> for block in generator:
    ...     print(block.shape)
    (512, 4)
    (488, 4)
    """

    #: Index of the first sample to be considered valid. Default is ``0``.
    start = CInt(0, desc='start of valid samples')

    #: Index of the last sample to be considered valid. If ``None``, all remaining samples from the
    #: :attr:`start` index onward are considered valid. Default is ``None``.
    stop = Union(None, CInt, desc='stop of valid samples')

    #: List of channel indices to be excluded from processing. Default is ``[]``.
    invalid_channels = List(int, desc='list of invalid channels')

    #: A mask or index array representing valid channels. Automatically updated based on the
    #: :attr:`invalid_channels` and :attr:`num_channels_total` attributes.
    channels = Property(depends_on=['invalid_channels', 'num_channels_total'], desc='channel mask')

    #: Total number of input channels, including invalid channels. (read-only).
    num_channels_total = CInt(0, desc='total number of input channels')

    #: Total number of samples, including invalid samples. (read-only).
    num_samples_total = CInt(0, desc='total number of samples per channel')

    #: Number of valid input channels after excluding :attr:`invalid_channels`. (read-only)
    num_channels = Property(
        depends_on=['invalid_channels', 'num_channels_total'], desc='number of valid input channels'
    )

    #: Number of valid time-domain samples, based on :attr:`start` and :attr:`stop` indices.
    #: (read-only)
    num_samples = Property(
        depends_on=['start', 'stop', 'num_samples_total'], desc='number of valid samples per channel'
    )

    #: A unique identifier for the samples, based on its properties. (read-only)
    digest = Property(depends_on=['basename', 'start', 'stop', 'calib.digest', 'invalid_channels', '_datachecksum'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_channels(self):
        if len(self.invalid_channels) == 0:
            return slice(0, None, None)
        allr = [i for i in range(self.num_channels_total) if i not in self.invalid_channels]
        return array(allr)

    @cached_property
    def _get_num_channels(self):
        if len(self.invalid_channels) == 0:
            return self.num_channels_total
        return len(self.channels)

    @cached_property
    def _get_num_samples(self):
        sli = slice(self.start, self.stop).indices(self.num_samples_total)
        return sli[1] - sli[0]

    @on_trait_change('basename')
    def _load_data(self):
        # Open the .h5 file and set attributes.
        if not path.isfile(self.file):
            # no file there
            self.sample_freq = 0
            msg = f'No such file: {self.file}'
            raise OSError(msg)
        if self.h5f is not None:
            with contextlib.suppress(OSError):
                self.h5f.close()
        file = _get_h5file_class()
        self.h5f = file(self.file)
        self._load_timedata()
        self._load_metadata()

    @on_trait_change('data')
    def _load_shapes(self):
        # Set :attr:`num_channels` and num_samples from :attr:`~acoular.sources.TimeSamples.data`.
        if self.data is not None:
            self.num_samples_total, self.num_channels_total = self.data.shape

    def _load_timedata(self):
        # Loads timedata from .h5 file. Only for internal use.
        self.data = self.h5f.get_data_by_reference('time_data')
        self.sample_freq = self.h5f.get_node_attribute(self.data, 'sample_freq')
        (self.num_samples_total, self.num_channels_total) = self.data.shape

    def result(self, num=128):
        """
        Generate blocks of valid time-domain data iteratively.

        The :meth:`result` method is a Python generator that yields blocks of valid time-domain data
        based on the specified :attr:`start` and :attr:`stop` indices and the valid channels. Data
        can be calibrated if a calibration object, given by :attr:`calib`, is provided.

        Parameters
        ----------
        num : :class:`int`, optional
            The size of each block to be yielded, representing the number of time-domain samples
            per block. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`num_channels`) representing a block of valid
            time-domain data. The last block may have fewer than ``num`` samples if the
            :attr:`number of valid samples<num_samples>` is not a multiple of ``num``.

        Raises
        ------
        :obj:`OSError`
            If no valid samples are available (i.e., :attr:`start` and :attr:`stop` indices result
            in an empty range).
        :obj:`ValueError`
            If the :attr:`calibration data<calib>` is incompatible with the
            :attr:`number of valid channels<num_channels>`.

        Warnings
        --------
        A deprecation warning is raised if the calibration functionality is used directly in
        :class:`MaskedTimeSamples`. Instead, the :class:`acoular.calib.Calib` class should be used
        as a separate processing block.

        Examples
        --------
        Access valid data in blocks:

        >>> import numpy as np
        >>> from acoular.sources import MaskedTimeSamples
        >>>
        >>> data = np.random.rand(1000, 4)
        >>> ts = MaskedTimeSamples(data=data, start=100, stop=900)
        >>>
        >>> generator = ts.result(num=256)
        >>> for block in generator:
        ...     print(block.shape)
        (256, 4)
        (256, 4)
        (256, 4)
        (32, 4)

        Note that the last block may have fewer that ``num`` samples.
        """
        sli = slice(self.start, self.stop).indices(self.num_samples_total)
        i = sli[0]
        stop = sli[1]
        cal_factor = 1.0
        if i >= stop:
            msg = 'no samples available'
            raise OSError(msg)
        self._datachecksum  # trigger checksum calculation # noqa: B018
        if self.calib:
            warn(
                'The use of the calibration functionality in MaskedTimeSamples is deprecated and will be removed in \
                       Acoular 25.10. Use the Calib class as an additional processing block instead.',
                DeprecationWarning,
                stacklevel=2,
            )
            if self.calib.num_mics == self.num_channels_total:
                cal_factor = self.calib.data[self.channels][newaxis]
            elif self.calib.num_mics == self.num_channels:
                cal_factor = self.calib.data[newaxis]
            elif self.calib.num_mics == 0:
                warn('No calibration data used.', Warning, stacklevel=2)
            else:
                msg = f'calibration data not compatible: {self.calib.num_mics:d}, {self.num_channels:d}'
                raise ValueError(msg)
        while i < stop:
            yield self.data[i : min(i + num, stop)][:, self.channels] * cal_factor
            i += num


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples'}, read_only=True)
class PointSource(SamplesGenerator):
    """
    Define a fixed point source emitting a signal, intended for simulations.

    The :class:`PointSource` class models a stationary sound source that generates a signal
    detected by microphones. It includes support for specifying the source's location, handling
    signal behaviors for pre-padding, and integrating environmental effects on sound propagation.
    The output is being generated via the :meth:`result` generator.

    See Also
    --------
    :class:`acoular.signals.SignalGenerator` : For defining custom emitted signals.
    :class:`acoular.microphones.MicGeom` : For specifying microphone geometries.
    :class:`acoular.environments.Environment` : For modeling sound propagation effects.

    Notes
    -----
    - The signal is adjusted to account for the distances between the source and microphones.
    - The :attr:`prepadding` attribute allows control over how the signal behaves for time indices
      before :attr:`start_t`.
    - Environmental effects such as sound speed are included through the :attr:`env` attribute.

    Examples
    --------
    To define a point source emitting a signal at a specific location, we first programmatically set
    a microphone geomertry as in :class:`~acoular.microphones.MicGeom`:

    >>> import numpy as np
    >>>
    >>> # Generate a (3,3) grid of points in the x-y plane
    >>> x = np.linspace(-1, 1, 3)  # Generate 3 points for x, from -1 to 1
    >>> y = np.linspace(-1, 1, 3)  # Generate 3 points for y, from -1 to 1
    >>>
    >>> # Create a meshgrid for 3D coordinates, with z=0 for all points
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.zeros_like(X)  # Set all z-values to 0
    >>>
    >>> # Stack the coordinates into a single (3,9) array
    >>> points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    >>> points
    array([[-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
           [-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    Now, to set the actual point source (``ps``), we define a microphone geomerity (``mg``), using
    the positional data from ``points``, and a sine generator (``sg``) with a total number of 6
    samples.

    >>> from acoular import PointSource, SineGenerator, MicGeom
    >>> mg = MicGeom(pos_total=points)
    >>> sg = SineGenerator(freq=1000, sample_freq=51200, num_samples=6)
    >>> ps = PointSource(signal=sg, loc=(0.5, 0.5, 1.0), mics=mg)

    We choose a blocksize of 4 and generate the output signal at the microphones in blocks:

    >>> for block in ps.result(num=4):
    ...     print(block.shape)
    (4, 9)
    (2, 9)

    The first block has shape (4,9) for 4 samples and 9 microphones. The second block has shape
    (2,9), since of a total of 6 samples only 2 remained.
    """

    #: Instance of the :class:`~acoular.signals.SignalGenerator` class defining the emitted signal.
    signal = Instance(SignalGenerator)

    #: Coordinates ``(x, y, z)`` of the source in a left-oriented system. Default is
    #: ``(0.0, 0.0, 1.0)``.
    loc = Tuple((0.0, 0.0, 1.0), desc='source location')

    #: Number of output channels, automatically set based on the :attr:`microphone geometry<mics>`.
    num_channels = Delegate('mics', 'num_mics')

    #: :class:`~acoular.microphones.MicGeom` object defining the positions of the microphones.
    mics = Instance(MicGeom, desc='microphone geometry')

    def _validate_locations(self):
        dist = self.env._r(array(self.loc).reshape((3, 1)), self.mics.pos)
        if npany(dist < 1e-7):
            warn('Source and microphone locations are identical.', Warning, stacklevel=2)

    #: An :class:`~acoular.environments.Environment` or derived object providing sound propagation
    #: details, such as :attr:`speed of sound in the medium<acoular.environments.Environment.c>`.
    #: Default is :class:`~acoular.environments.Environment`.
    env = Instance(Environment, args=())

    #: Start time of the signal in seconds. Default is ``0.0``.
    start_t = Float(0.0, desc='signal start time')

    #: Start time of data acquisition at the microphones in seconds. Default is ``0.0``.
    start = Float(0.0, desc='sample start time')

    #: Behavior of the signal for negative time indices,
    #: i.e. if (:attr:`start` ``<`` :attr:`start_t`):
    #:
    #: - ``'loop'``: Repeat the :attr:`signal` from its end.
    #: - ``'zeros'``: Use zeros, recommended for deterministic signals.
    #:
    #: Default is ``'loop'``.
    prepadding = Enum('loop', 'zeros', desc='Behaviour for negative time indices.')

    #: Internal upsampling factor for finer signal resolution. Default is ``16``.
    up = Int(16, desc='upsampling factor')

    #: Total number of samples in the emitted signal, derived from the :attr:`signal` generator.
    num_samples = Delegate('signal')

    #: Sampling frequency of the signal, derived from the :attr:`signal` generator.
    sample_freq = Delegate('signal')

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
            'prepadding',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """
        Generate output signal at microphones in blocks, incorporating propagation effects.

        The :meth:`result` method provides a generator that yields blocks of the signal detected at
        microphones. The signal is adjusted for the distances between the source and microphones, as
        well as any environmental propagation effects.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to be yielded. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`num_channels`) containing the signal detected at
            the microphones. The last block may have fewer samples if :attr:`num_samples` is not a
            multiple of ``num``.

        Raises
        ------
        :obj:`ValueError`
            If the source and a microphone are located at the same position.
        :obj:`RuntimeError`
            If signal processing or propagation cannot be performed.
        """
        self._validate_locations()
        N = int(ceil(self.num_samples / num))  # number of output blocks
        signal = self.signal.usignal(self.up)
        out = empty((num, self.num_channels))
        # distances
        rm = self.env._r(array(self.loc).reshape((3, 1)), self.mics.pos).reshape(1, -1)
        # emission time relative to start_t (in samples) for first sample
        ind = (-rm / self.env.c - self.start_t + self.start) * self.sample_freq * self.up

        if self.prepadding == 'zeros':
            # number of blocks where signal behaviour is amended
            pre = -int(npmin(ind[0]) // (self.up * num))
            # amend signal for first blocks
            # if signal stops during prepadding, terminate
            if pre >= N:
                for _nb in range(N - 1):
                    out = _fill_mic_signal_block(out, signal, rm, ind, num, self.num_channels, self.up, True)
                    yield out

                blocksize = self.num_samples % num or num
                out = _fill_mic_signal_block(out, signal, rm, ind, blocksize, self.num_channels, self.up, True)
                yield out[:blocksize]
                return
            else:
                for _nb in range(pre):
                    out = _fill_mic_signal_block(out, signal, rm, ind, num, self.num_channels, self.up, True)
                    yield out

        else:
            pre = 0

        # main generator
        for _nb in range(N - pre - 1):
            out = _fill_mic_signal_block(out, signal, rm, ind, num, self.num_channels, self.up, False)
            yield out

        # last block of variable size
        blocksize = self.num_samples % num or num
        out = _fill_mic_signal_block(out, signal, rm, ind, blocksize, self.num_channels, self.up, False)
        yield out[:blocksize]


class SphericalHarmonicSource(PointSource):
    """
    Define a fixed spherical harmonic source emitting a signal.

    The :class:`SphericalHarmonicSource` class models a stationary sound source that emits a signal
    with spatial properties represented by spherical harmonics. This source can simulate
    directionality and orientation in sound emission, making it suitable for advanced acoustic
    simulations.

    The output is being generated via the :meth:`result` generator.
    """

    #: Order of the spherical harmonic representation. Default is ``0``.
    lOrder = Int(0, desc='Order of spherical harmonic')  # noqa: N815

    #: Coefficients of the spherical harmonic modes for the given :attr:`lOrder`.
    alpha = CArray(desc='coefficients of the (lOrder,) spherical harmonic mode')

    #: Vector defining the orientation of the spherical harmonic source. Default is
    #: ``(1.0, 0.0, 0.0)``.
    direction = Tuple((1.0, 0.0, 0.0), desc='Spherical Harmonic orientation')

    #: Behavior of the signal for negative time indices. Currently only supports `loop`. Default is
    #: ``'loop'``.
    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

    # Unique identifier for the current state of the source, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'mics.digest',
            'signal.digest',
            'loc',
            'env.digest',
            'start_t',
            'start',
            'up',
            'alpha',
            'lOrder',
            'prepadding',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def transform(self, signals):
        """
        Apply spherical harmonic transformation to input signals.

        The :meth:`transform` method modifies the input signals using the spherical harmonic modes,
        taking into account the specified coefficients (:attr:`alpha`), order (:attr:`lOrder`), and
        source orientation (:attr:`direction`).

        Parameters
        ----------
        signals : :class:`numpy.ndarray`
            Input signal array of shape (:attr:`~PointSouce.num_samples`,
            :attr:`~PointSouce.num_channels`).

        Returns
        -------
        :class:`numpy.ndarray`
            Transformed signal array of the same shape as ``signals``.

        See Also
        --------
        :func:`get_modes` : Method for computing spherical harmonic modes.

        Notes
        -----
        - The spherical harmonic modes are computed using the :func:`get_modes` function, which
          requires the microphone positions, source position, and source orientation.
        - The transformation applies the spherical harmonic coefficients (:attr:`alpha`) to the
          signal in the frequency domain.
        """
        Y_lm = get_modes(
            lOrder=self.lOrder,
            direction=self.direction,
            mpos=self.mics.pos,
            sourceposition=array(self.loc),
        )
        return real(ifft(fft(signals, axis=0) * (Y_lm @ self.alpha), axis=0))

    def result(self, num=128):
        """
        Generate output signal at microphones in blocks, incorporating propagation effects.

        The :meth:`result` method provides a generator that yields blocks of the signal detected at
        microphones. The signal is adjusted for the distances between the source and microphones, as
        well as any environmental propagation effects.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to be yielded. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`~PointSource.num_channels`) containing the signal
            detected at the microphones. The last block may have fewer samples if
            :attr:`~PointSource.num_samples` is not a multiple of ``num``.

        Raises
        ------
        :obj:`IndexError`
            If no more samples are available from the signal source.
        """
        # If signal samples are needed for te < t_start, then samples are taken
        # from the end of the calculated signal.

        signal = self.signal.usignal(self.up)
        # emission time relative to start_t (in samples) for first sample
        rm = self.env._r(array(self.loc).reshape((3, 1)), self.mics.pos)
        ind = (-rm / self.env.c - self.start_t + self.start) * self.sample_freq + pi / 30
        i = 0
        n = self.num_samples
        out = empty((num, self.num_channels))
        while n:
            n -= 1
            try:
                out[i] = signal[array(0.5 + ind * self.up, dtype=int64)] / rm
                ind += 1
                i += 1
                if i == num:
                    yield self.transform(out)
                    i = 0
            except IndexError:  # if no more samples available from the source
                break
        if i > 0:  # if there are still samples to yield
            yield self.transform(out[:i])


class MovingPointSource(PointSource):
    """
    Define a moving :class:`point source<PointSource>` emitting a :attr:`~PointSource.signal`.

    The :class:`MovingPointSource` class models a sound source that follows a
    :attr:`specified trajectory<trajectory>` while emitting a :attr:`~PointSource.signal`.
    This allows for the simulation of dynamic acoustic scenarios,
    e.g. sources changing position over time such as vehicles in motion.

    See Also
    --------
    :class:`acoular.sources.PointSource` : For modeling stationary point sources.
    :class:`acoular.trajectory.Trajectory` : For specifying source motion paths.
    """

    #: Determines whether convective amplification is considered. When ``True``, the amplitude of
    #: the signal is adjusted based on the relative motion between the source and microphones.
    #: Default is ``False``.
    conv_amp = Bool(False, desc='determines if convective amplification is considered')

    #: Instance of the :class:`~acoular.trajectory.Trajectory` class specifying the source's motion.
    #: The trajectory defines the source's position and velocity at any given time.
    trajectory = Instance(Trajectory, desc='trajectory of the source')

    #: Behavior of the signal for negative time indices. Currently only supports ``'loop'``.
    #: Default is ``'loop'``.
    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

    #: A unique identifier for the current state of the source, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'mics.digest',
            'signal.digest',
            'loc',
            'conv_amp',
            'env.digest',
            'start_t',
            'start',
            'trajectory.digest',
            'prepadding',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """
        Generate the output signal at microphones in blocks, accounting for source motion.

        The :meth:`result` method provides a generator that yields blocks of the signal received at
        microphones. It incorporates the :attr:`source's trajectory<trajectory>`, convective
        amplification (if enabled), and environmental propagation effects.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to be yielded. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`~PointSource.num_channels`) containing the signal
            detected at the microphones. The last block may have fewer samples if
            :attr:`~PointSource.num_samples` is not a multiple of ``num``.

        Raises
        ------
        :obj:`IndexError`
            If no more samples are available from the signal source.

        Notes
        -----
        - The method iteratively solves for the emission times of the signal at each microphone
          using the Newton-Raphson method.
        - Convective amplification is applied if :attr:`conv_amp` ``= True``, modifying the signal's
          amplitude based on the relative motion between the source and microphones.
        - The signal's emission time is calculated relative to the trajectory's position and
          velocity at each step.
        """
        # If signal samples are needed for te < t_start, then samples are taken
        # from the end of the calculated signal.

        signal = self.signal.usignal(self.up)
        out = empty((num, self.num_channels))
        # shortcuts and initial values
        m = self.mics
        t = self.start * ones(m.num_mics)
        i = 0
        epslim = 0.1 / self.up / self.sample_freq
        c0 = self.env.c
        tr = self.trajectory
        n = self.num_samples
        while n:
            n -= 1
            eps = ones(m.num_mics)
            te = t.copy()  # init emission time = receiving time
            j = 0
            # Newton-Rhapson iteration
            while abs(eps).max() > epslim and j < 100:
                loc = array(tr.location(te))
                rm = loc - m.pos  # distance vectors to microphones
                rm = sqrt((rm * rm).sum(0))  # absolute distance
                loc /= sqrt((loc * loc).sum(0))  # distance unit vector
                der = array(tr.location(te, der=1))
                Mr = (der * loc).sum(0) / c0  # radial Mach number
                eps = (te + rm / c0 - t) / (1 + Mr)  # discrepancy in time
                te -= eps
                j += 1  # iteration count
            t += 1.0 / self.sample_freq
            # emission time relative to start time
            ind = (te - self.start_t + self.start) * self.sample_freq
            if self.conv_amp:
                rm *= (1 - Mr) ** 2
            try:
                out[i] = signal[array(0.5 + ind * self.up, dtype=int64)] / rm
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:  # if no more samples available from the source
                break
        if i > 0:  # if there are still samples to yield
            yield out[:i]


class PointSourceDipole(PointSource):
    """
    Define a fixed point source with dipole characteristics.

    The :class:`PointSourceDipole` class simulates a fixed point source with dipole characteristics
    by superimposing two nearby inversely phased monopoles. This is particularly useful for
    acoustic simulations where dipole sources are required.

    The generated output is available via the :meth:`result` generator.

    See Also
    --------
    :class:`acoular.sources.PointSource` : For modeling stationary point sources.

    Notes
    -----
    The dipole's output is calculated as the superposition of two monopoles: one shifted forward and
    the other backward along the :attr:`direction` vector, with inverse phases. This creates the
    characteristic dipole radiation pattern.
    """

    #: Vector defining the orientation of the dipole lobes and the distance between the inversely
    #: phased monopoles. The magnitude of the vector determines the monopoles' separation:
    #:
    #: - ``distance = [lowest wavelength in spectrum] * [magnitude] * 1e-5``
    #:
    #: Use vectors with magnitudes on the order of ``1.0`` or smaller for best results.
    #: Default is ``(0.0, 0.0, 1.0)`` (z-axis orientation).
    #:
    #: **Note:** Use vectors with order of magnitude around ``1.0`` or less for good results.
    direction = Tuple((0.0, 0.0, 1.0), desc='dipole orientation and distance of the inversely phased monopoles')

    #: Behavior of the signal for negative time indices. Currently only supports ``'loop'``.
    #: Default is ``'loop'``.
    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

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
            'direction',
            'prepadding',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

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
        # If signal samples are needed for te < t_start, then samples are taken
        # from the end of the calculated signal.

        mpos = self.mics.pos
        # position of the dipole as (3,1) vector
        loc = array(self.loc, dtype=float).reshape((3, 1))
        # direction vector from tuple
        direc = array(self.direction, dtype=float) * 1e-5
        direc_mag = sqrt(dot(direc, direc))

        # normed direction vector
        direc_n = direc / direc_mag

        c = self.env.c

        # distance between monopoles as function of c, sample freq, direction vector
        dist = c / self.sample_freq * direc_mag

        # vector from dipole center to one of the monopoles
        dir2 = (direc_n * dist / 2.0).reshape((3, 1))

        signal = self.signal.usignal(self.up)
        out = empty((num, self.num_channels))

        # distance from dipole center to microphones
        rm = self.env._r(loc, mpos)

        # distances from monopoles to microphones
        rm1 = self.env._r(loc + dir2, mpos)
        rm2 = self.env._r(loc - dir2, mpos)

        # emission time relative to start_t (in samples) for first sample
        ind1 = (-rm1 / c - self.start_t + self.start) * self.sample_freq
        ind2 = (-rm2 / c - self.start_t + self.start) * self.sample_freq

        i = 0
        n = self.num_samples
        while n:
            n -= 1
            try:
                # subtract the second signal b/c of phase inversion
                out[i] = (
                    rm
                    / dist
                    * (
                        signal[array(0.5 + ind1 * self.up, dtype=int64)] / rm1
                        - signal[array(0.5 + ind2 * self.up, dtype=int64)] / rm2
                    )
                )
                ind1 += 1.0
                ind2 += 1.0

                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break

        yield out[:i]


class MovingPointSourceDipole(PointSourceDipole, MovingPointSource):
    """
    Define a moving point source with dipole characteristics.

    This class extends the functionalities of :class:`PointSourceDipole` and
    :class:`MovingPointSource` to simulate a dipole source that moves along a
    :attr:`defined trajectory<MovingPointSource.trajectory>`. It incorporates both rotational and
    translational dynamics for the dipole lobes, allowing simulation of complex directional sound
    sources.

    Key Features:
        - Combines dipole characteristics with source motion.
        - Supports rotation of the dipole directivity via the :attr:`rvec` attribute.
        - Calculates emission times using Newton-Raphson iteration.

    See Also
    --------
    :class:`acoular.sources.PointSourceDipole` : For stationary dipole sources.
    :class:`acoular.sources.MovingPointSource` :
        For moving point sources without dipole characteristics.
    """

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
            'direction',
        ],
    )

    #: A reference vector, perpendicular to the x and y-axis of moving source, defining the axis of
    #: rotation for the dipole directivity. If set to ``(0, 0, 0)``, the dipole is only translated
    #: along the :attr:`~MovingPointSource.trajectory` without rotation. Default is ``(0, 0, 0)``.
    rvec = CArray(dtype=float, shape=(3,), value=array((0, 0, 0)), desc='reference vector')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def get_emission_time(self, t, direction):
        """
        Calculate the emission time and related properties for a moving source.

        Parameters
        ----------
        t : :class:`numpy.ndarray`
            The current receiving time at the microphones.
        direction : :class:`float` or :class:`numpy.ndarray`
            Direction vector for the source's dipole directivity.

        Returns
        -------
        tuple
            A tuple containing:

            - te : :class:`numpy.ndarray`
                Emission times for each microphone.
            - rm : :class:`numpy.ndarray`
                Distances from the source to each microphone.
            - Mr : :class:`numpy.ndarray`
                Radial Mach numbers for the source's motion.
            - xs : :class:`numpy.ndarray`
                Source coordinates at the calculated emission times.

        Warnings
        --------
        Ensure that the maximum iteration count (``100``) is sufficient for convergence in all
        scenarios, especially for high Mach numbers or long trajectories.

        Notes
        -----
        The emission times are computed iteratively using the Newton-Raphson method. The iteration
        terminates when the time discrepancy (``eps``) is below a threshold (``epslim``)
        or after 100 iterations.
        """
        eps = ones(self.mics.num_mics)
        epslim = 0.1 / self.up / self.sample_freq
        te = t.copy()  # init emission time = receiving time
        j = 0
        # Newton-Rhapson iteration
        while abs(eps).max() > epslim and j < 100:
            xs = array(self.trajectory.location(te))
            loc = xs.copy()
            loc += direction
            rm = loc - self.mics.pos  # distance vectors to microphones
            rm = sqrt((rm * rm).sum(0))  # absolute distance
            loc /= sqrt((loc * loc).sum(0))  # distance unit vector
            der = array(self.trajectory.location(te, der=1))
            Mr = (der * loc).sum(0) / self.env.c  # radial Mach number
            eps = (te + rm / self.env.c - t) / (1 + Mr)  # discrepancy in time
            te -= eps
            j += 1  # iteration count
        return te, rm, Mr, xs

    def get_moving_direction(self, direction, time=0):
        """
        Calculate the moving direction of the dipole source along its trajectory.

        This method computes the updated direction vector for the dipole source, considering both
        translation along the trajectory and rotation defined by the :attr:`reference vector<rvec>`.
        If the reference vector is ``(0, 0, 0)``, only translation is applied. Otherwise, the method
        incorporates rotation into the calculation.

        Parameters
        ----------
        direction : :class:`numpy.ndarray`
            The initial direction vector of the dipole, specified as a 3-element
            array representing the orientation of the dipole lobes.
        time : :class:`float`, optional
            The time at which the trajectory position and velocity are evaluated. Defaults to ``0``.

        Returns
        -------
        :class:`numpy.ndarray`
            The updated direction vector of the dipole source after translation
            and, if applicable, rotation. The output is a 3-element array.

        Notes
        -----
        - The method computes the translation direction vector based on the trajectory's velocity at
          the specified time.
        - If the :attr:`reference vector<rvec>` is non-zero, the method constructs a rotation matrix
          to compute the new dipole direction based on the trajectory's motion and the
          reference vector.
        - The rotation matrix ensures that the new dipole orientation adheres
          to the right-hand rule and remains orthogonal.
        """
        trajg1 = array(self.trajectory.location(time, der=1))[:, 0][:, newaxis]
        rflag = (self.rvec == 0).all()  # flag translation vs. rotation
        if rflag:
            return direction
        dx = array(trajg1.T)  # direction vector (new x-axis)
        dy = cross(self.rvec, dx)  # new y-axis
        dz = cross(dx, dy)  # new z-axis
        RM = array((dx, dy, dz)).T  # rotation matrix
        RM /= sqrt((RM * RM).sum(0))  # column normalized
        newdir = dot(RM, direction)
        return cross(newdir[:, 0].T, self.rvec.T).T

    def result(self, num=128):
        """
        Generate the output signal at microphones in blocks.

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

        Notes
        -----
        Radial Mach number adjustments are applied if :attr:`~MovingPointSource.conv_amp` is
        enabled.
        """
        # If signal samples are needed for te < t_start, then samples are taken
        # from the end of the calculated signal.
        mpos = self.mics.pos

        # direction vector from tuple
        direc = array(self.direction, dtype=float) * 1e-5
        direc_mag = sqrt(dot(direc, direc))
        # normed direction vector
        direc_n = direc / direc_mag
        c = self.env.c
        # distance between monopoles as function of c, sample freq, direction vector
        dist = c / self.sample_freq * direc_mag * 2

        # vector from dipole center to one of the monopoles
        dir2 = (direc_n * dist / 2.0).reshape((3, 1))

        signal = self.signal.usignal(self.up)
        out = empty((num, self.num_channels))
        # shortcuts and initial values
        m = self.mics
        t = self.start * ones(m.num_mics)

        i = 0
        n = self.num_samples
        while n:
            n -= 1
            te, rm, Mr, locs = self.get_emission_time(t, 0)
            t += 1.0 / self.sample_freq
            # location of the center
            loc = array(self.trajectory.location(te), dtype=float)[:, 0][:, newaxis]
            # distance of the dipoles from the center
            diff = self.get_moving_direction(dir2, te)

            # distance of sources
            rm1 = self.env._r(loc + diff, mpos)
            rm2 = self.env._r(loc - diff, mpos)

            ind = (te - self.start_t + self.start) * self.sample_freq
            if self.conv_amp:
                rm *= (1 - Mr) ** 2
                rm1 *= (1 - Mr) ** 2  # assume that Mr is the same for both poles
                rm2 *= (1 - Mr) ** 2
            try:
                # subtract the second signal b/c of phase inversion
                out[i] = (
                    rm
                    / dist
                    * (
                        signal[array(0.5 + ind * self.up, dtype=int64)] / rm1
                        - signal[array(0.5 + ind * self.up, dtype=int64)] / rm2
                    )
                )
                i += 1
                if i == num:
                    yield out
                    i = 0
            except IndexError:
                break
        yield out[:i]


class LineSource(PointSource):
    """
    Define a fixed line source with a signal.

    The :class:`LineSource` class models a fixed line source composed of multiple monopole sources
    arranged along a specified direction. Each monopole can have its own source strength, and the
    coherence between them can be controlled.

    Key Features:
        - Specify the :attr:`orientation<direction>`, :attr:`length`, and
          :attr:`number<num_sources>` of monopoles in the line source.
        - Control the :attr:`source strength<source_strength>` of individual monopoles.
        - Support for :attr:`coherent or incoherent<coherence>` monopole sources.

    The output signals at microphones are generated block-wise using the :meth:`result` generator.

    See Also
    --------
    :class:`acoular.sources.PointSource` : For modeling stationary point sources.

    Notes
    -----
    For incoherent sources, a unique seed is set for each monopole to generate independent signals.
    """

    #: Vector to define the orientation of the line source. Default is ``(0.0, 0.0, 1.0)``.
    direction = Tuple((0.0, 0.0, 1.0), desc='Line orientation ')

    #: Vector to define the length of the line source in meters. Default is ``1.0``.
    length = Float(1, desc='length of the line source')

    #: Number of monopole sources in the line source. Default is ``1``.
    num_sources = Int(1)

    #: Strength coefficients for each monopole source.
    source_strength = CArray(desc='coefficients of the source strength')

    #: Coherence mode for the monopoles (``'coherent'`` or ``'incoherent'``).
    coherence = Enum('coherent', 'incoherent', desc='coherence mode')

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
            'direction',
            'source_strength',
            'coherence',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """
        Generate the output signal at microphones in blocks.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to yield. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`~PointSource.num_channels`) containing
            the signal detected at the microphones. The last block may have fewer samples
            if :attr:`~PointSource.num_samples` is not a multiple of ``num``.
        """
        # If signal samples are needed for te < t_start, then samples are taken
        # from the end of the calculated signal.

        mpos = self.mics.pos

        # direction vector from tuple
        direc = array(self.direction, dtype=float)
        # normed direction vector
        direc_n = direc / norm(direc)
        c = self.env.c

        # distance between monopoles in the line
        dist = self.length / self.num_sources

        # blocwise output
        out = zeros((num, self.num_channels))

        # distance from line start position to microphones
        loc = array(self.loc, dtype=float).reshape((3, 1))

        # distances from monopoles in the line to microphones
        rms = empty((self.num_channels, self.num_sources))
        inds = empty((self.num_channels, self.num_sources))
        signals = empty((self.num_sources, len(self.signal.usignal(self.up))))
        # for every source - distances
        for s in range(self.num_sources):
            rms[:, s] = self.env._r((loc.T + direc_n * dist * s).T, mpos)
            inds[:, s] = (-rms[:, s] / c - self.start_t + self.start) * self.sample_freq
            # new seed for every source
            if self.coherence == 'incoherent':
                self.signal.seed = s + abs(int(hash(self.digest) // 10e12))
            self.signal.rms = self.signal.rms * self.source_strength[s]
            signals[s] = self.signal.usignal(self.up)
        i = 0
        n = self.num_samples
        while n:
            n -= 1
            try:
                for s in range(self.num_sources):
                    # sum sources
                    out[i] += signals[s, array(0.5 + inds[:, s].T * self.up, dtype=int64)] / rms[:, s]

                inds += 1.0
                i += 1
                if i == num:
                    yield out
                    out = zeros((num, self.num_channels))
                    i = 0
            except IndexError:
                break

        yield out[:i]


class MovingLineSource(LineSource, MovingPointSource):
    """
    A moving :class:`line source<LineSource>` with an arbitrary signal.

    The :class:`MovingLineSource` class models a :class:`line source<LineSource>` composed of
    multiple monopoles that move along a :attr:`~MovingPointSource.trajectory`. It supports
    :attr:`coherent and incoherent<LineSource.coherence>` sources and considers Doppler effects due
    to motion.

    Key Features:
        - Specify the :attr:`~MovingPointSource.trajectory` and rotation of the
          :class:`line source<LineSource>`.
        - Compute emission times considering motion and source :attr:`~LineSource.direction`.
        - Generate block-wise microphone output with moving source effects.

    See Also
    --------
    :class:`acoular.sources.LineSource` :
        For :class:`line sources<LineSource>` consisting of
        :attr:`coherent or incoherent<LineSource.coherence>` monopoles.
    :class:`acoular.sources.MovingPointSource` :
        For moving point sources without dipole characteristics.
    """

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
            'direction',
        ],
    )

    #: A reference vector, perpendicular to the x and y-axis of moving source, defining the axis of
    #: rotation for the line source directivity. If set to ``(0, 0, 0)``, the line source is only
    #: translated along the :attr:`~MovingPointSource.trajectory` without rotation. Default is
    #: ``(0, 0, 0)``.
    rvec = CArray(dtype=float, shape=(3,), value=array((0, 0, 0)), desc='reference vector')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def get_moving_direction(self, direction, time=0):
        """
        Calculate the moving direction of the line source along its trajectory.

        This method computes the updated direction vector for the line source,
        considering both translation along the :attr:`~MovingPointSource.trajectory` and rotation
        defined by the :attr:`reference vector<rvec>`. If the :attr:`reference vector<rvec>` is
        `(0, 0, 0)`, only translation is applied. Otherwise, the method incorporates rotation
        into the calculation.

        Parameters
        ----------
        direction : :class:`numpy.ndarray`
            The initial direction vector of the line source, specified as a
            3-element array representing the orientation of the line.
        time : :class:`float`, optional
            The time at which the :attr:`~MovingPointSource.trajectory` position and velocity
            are evaluated. Defaults to ``0``.

        Returns
        -------
        :class:`numpy.ndarray`
            The updated direction vector of the line source after translation and,
            if applicable, rotation. The output is a 3-element array.

        Notes
        -----
        - The method computes the translation direction vector based on the
          :attr:`~MovingPointSource.trajectory`'s velocity at the specified time.
        - If the :attr:`reference vector<rvec>` is non-zero, the method constructs a
          rotation matrix to compute the new line source direction based on the
          :attr:`~MovingPointSource.trajectory`'s motion and the :attr:`reference vector<rvec>`.
        - The rotation matrix ensures that the new orientation adheres to the
          right-hand rule and remains orthogonal.
        """
        trajg1 = array(self.trajectory.location(time, der=1))[:, 0][:, newaxis]
        rflag = (self.rvec == 0).all()  # flag translation vs. rotation
        if rflag:
            return direction
        dx = array(trajg1.T)  # direction vector (new x-axis)
        dy = cross(self.rvec, dx)  # new y-axis
        dz = cross(dx, dy)  # new z-axis
        RM = array((dx, dy, dz)).T  # rotation matrix
        RM /= sqrt((RM * RM).sum(0))  # column normalized
        newdir = dot(RM, direction)
        return cross(newdir[:, 0].T, self.rvec.T).T

    def get_emission_time(self, t, direction):
        """
        Calculate the emission time for a moving line source based on its trajectory.

        This method computes the time at which sound waves are emitted from the line source
        at a specific point along its :attr:`~MovingPointSource.trajectory`. It also determines the
        distances from the source to each microphone and calculates the radial Mach number, which
        accounts for the Doppler effect due to the motion of the source.

        Parameters
        ----------
        t : :class:`float`
            The current receiving time at the microphones, specified in seconds.
        direction : :class:`numpy.ndarray`
            The current direction vector of the line source, specified as a 3-element array
            representing the orientation of the line.

        Returns
        -------
        te : :class:`numpy.ndarray`
            The computed emission times for each microphone, specified as an array of floats.
        rm : :class:`numpy.ndarray`
            The distances from the line source to each microphone, represented as an
            array of absolute distances.
        Mr : :class:`numpy.ndarray`
            The radial Mach number, which accounts for the Doppler effect, calculated for
            each microphone.
        xs : :class:`numpy.ndarray`
            The position of the line source at the computed emission time, returned as
            a 3-element array.

        Notes
        -----
        - This method performs Newton-Raphson iteration to find the emission time where
          the sound wave from the source reaches the microphones.
        - The distance between the line source and microphones is computed using
          Euclidean geometry.
        - The radial Mach number (``Mr``) is calculated using the velocity of the source
          and the speed of sound in the medium (:attr:`~acoular.environments.Environment.c`).
        - The method iterates until the difference between the computed emission time and
          the current time is sufficiently small (within a defined threshold).
        """
        eps = ones(self.mics.num_mics)
        epslim = 0.1 / self.up / self.sample_freq
        te = t.copy()  # init emission time = receiving time
        j = 0
        # Newton-Rhapson iteration
        while abs(eps).max() > epslim and j < 100:
            xs = array(self.trajectory.location(te))
            loc = xs.copy()
            loc += direction
            rm = loc - self.mics.pos  # distance vectors to microphones
            rm = sqrt((rm * rm).sum(0))  # absolute distance
            loc /= sqrt((loc * loc).sum(0))  # distance unit vector
            der = array(self.trajectory.location(te, der=1))
            Mr = (der * loc).sum(0) / self.env.c  # radial Mach number
            eps = (te + rm / self.env.c - t) / (1 + Mr)  # discrepancy in time
            te -= eps
            j += 1  # iteration count
        return te, rm, Mr, xs

    def result(self, num=128):
        """
        Generate the output signal at microphones in blocks.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to yield. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`~PointSource.num_channels`) containing
            the signal detected at the microphones. The last block may have fewer samples
            if :attr:`~PointSource.num_samples` is not a multiple of ``num``.
        """
        # If signal samples are needed for te < t_start, then samples are taken
        # from the end of the calculated signal.
        mpos = self.mics.pos

        # direction vector from tuple
        direc = array(self.direction, dtype=float)
        # normed direction vector
        direc_n = direc / norm(direc)

        # distance between monopoles in the line
        dist = self.length / self.num_sources
        dir2 = (direc_n * dist).reshape((3, 1))

        # blocwise output
        out = zeros((num, self.num_channels))

        # distances from monopoles in the line to microphones
        rms = empty((self.num_channels, self.num_sources))
        inds = empty((self.num_channels, self.num_sources))
        signals = empty((self.num_sources, len(self.signal.usignal(self.up))))
        # coherence
        for s in range(self.num_sources):
            # new seed for every source
            if self.coherence == 'incoherent':
                self.signal.seed = s + abs(int(hash(self.digest) // 10e12))
            self.signal.rms = self.signal.rms * self.source_strength[s]
            signals[s] = self.signal.usignal(self.up)
        mpos = self.mics.pos

        # shortcuts and initial values
        m = self.mics
        t = self.start * ones(m.num_mics)
        i = 0
        n = self.num_samples
        while n:
            n -= 1
            t += 1.0 / self.sample_freq
            te1, rm1, Mr1, locs1 = self.get_emission_time(t, 0)
            # trajg1 = array(self.trajectory.location( te1, der=1))[:,0][:,newaxis]

            # get distance and ind for every source in the line
            for s in range(self.num_sources):
                diff = self.get_moving_direction(dir2, te1)
                te, rm, Mr, locs = self.get_emission_time(t, tile((diff * s).T, (self.num_channels, 1)).T)
                loc = array(self.trajectory.location(te), dtype=float)[:, 0][:, newaxis]
                diff = self.get_moving_direction(dir2, te)
                rms[:, s] = self.env._r((loc + diff * s), mpos)
                inds[:, s] = (te - self.start_t + self.start) * self.sample_freq

            if self.conv_amp:
                rm *= (1 - Mr) ** 2
                rms[:, s] *= (1 - Mr) ** 2  # assume that Mr is the same
            try:
                # subtract the second signal b/c of phase inversion
                for s in range(self.num_sources):
                    # sum sources
                    out[i] += signals[s, array(0.5 + inds[:, s].T * self.up, dtype=int64)] / rms[:, s]

                i += 1
                if i == num:
                    yield out
                    out = zeros((num, self.num_channels))
                    i = 0
            except IndexError:
                break
        yield out[:i]


@deprecated_alias({'numchannels': 'num_channels'}, read_only=True)
class UncorrelatedNoiseSource(SamplesGenerator):
    """
    Simulate uncorrelated white or pink noise signals at multiple channels.

    The :class:`UncorrelatedNoiseSource` class generates noise signals (e.g., white or pink noise)
    independently at each channel. It supports a user-defined random seed for reproducibility and
    adapts the number of channels based on the provided microphone geometry. The output is
    generated block-by-block through the :meth:`result` generator.

    See Also
    --------
    :class:`acoular.signals.SignalGenerator` : For defining noise types and properties.
    :class:`acoular.microphones.MicGeom` : For specifying microphone geometries.

    Notes
    -----
    - The type of noise is defined by the :attr:`signal` attribute, which must be an instance of
      a :class:`~acoular.signals.SignalGenerator`-derived class that supports a ``seed`` parameter.
    - Each channel generates independent noise, with optional pre-defined random seeds via the
      :attr:`seed` attribute.
    - If no seeds are provided, they are generated automatically based on the number of channels
      and the signal seed.

    Examples
    --------
    To simulate uncorrelated white noise at multiple channels:

    >>> from acoular import UncorrelatedNoiseSource, WNoiseGenerator, MicGeom
    >>> import numpy as np
    >>>
    >>> # Define microphone geometry
    >>> mic_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]).T  # Three microphones
    >>> mics = MicGeom(pos_total=mic_positions)
    >>>
    >>> # Define white noise generator
    >>> noise_gen = WNoiseGenerator(sample_freq=51200, num_samples=1024, rms=1.0, seed=42)
    >>>
    >>> # Create the noise source
    >>> noise_source = UncorrelatedNoiseSource(signal=noise_gen, mics=mics)
    >>>
    >>> # Generate noise output block-by-block
    >>> for block in noise_source.result(num=256):
    ...     print(block.shape)
    (256, 3)
    (256, 3)
    (256, 3)
    (256, 3)

    The output blocks contain noise signals for each of the 3 channels. The number of blocks
    depends on the total number of samples and the block size.
    """

    #: Instance of a :class:`~acoular.signals.NoiseGenerator`-derived class. For example:
    #:    - :class:`~acoular.signals.WNoiseGenerator` for white noise.
    #:    - :class:`~acoular.signals.PNoiseGenerator` for pink noise.
    signal = Instance(NoiseGenerator, desc='type of noise')

    #: Array of random seed values for generating uncorrelated noise at each channel. If left empty,
    #: seeds will be automatically generated as ``np.arange(self.num_channels) + signal.seed``. The
    #: size of the array must match the :attr:`number of output channels<num_channels>`.
    seed = CArray(dtype=uint32, desc='random seed values')

    #: Number of output channels, automatically determined by the number of microphones
    #: defined in the :attr:`mics` attribute. Corresponds to the number of uncorrelated noise
    #: signals generated.
    num_channels = Delegate('mics', 'num_mics')

    #: :class:`~acoular.microphones.MicGeom` object specifying the positions of microphones.
    #: This attribute is used to define the microphone geometry and the
    #: :attr:`number of channels<num_channels>`.
    mics = Instance(MicGeom, desc='microphone geometry')

    #: Start time of the generated noise signal in seconds. Determines the time offset for the noise
    #: output relative to the start of data acquisition. Default is ``0.0``.
    start_t = Float(0.0, desc='signal start time')

    #: Start time of data acquisition at the microphones in seconds. This value determines when the
    #: generated noise begins relative to the acquisition process.  Default is ``0.0``.
    start = Float(0.0, desc='sample start time')

    #: Total number of samples in the noise signal, derived from the :attr:`signal` generator.
    #: This value determines the length of the output signal for all channels.
    num_samples = Delegate('signal')

    #: Sampling frequency of the generated noise signal in Hz, derived from the :attr:`signal`
    #: generator. This value defines the temporal resolution of the noise output.
    sample_freq = Delegate('signal')

    #: A unique identifier for the current state of the source, based on its properties. (read-only)
    digest = Property(
        depends_on=[
            'mics.digest',
            'signal.digest',
            'seed',
            'loc',
            'start_t',
            'start',
        ],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """
        Generate uncorrelated noise signals at microphones in blocks.

        The :meth:`result` method produces a Python generator that yields blocks of noise signals
        generated independently for each channel. This method supports customizable block sizes and
        ensures that the last block may have fewer samples if the total number of samples is not an
        exact multiple of the block size.

        Parameters
        ----------
        num : :class:`int`, optional
            Number of samples per block to be yielded. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`num_channels`) containing uncorrelated noise
            signals. The last block may be shorter if the total number of samples is not a
            multiple of ``num``.

        Raises
        ------
        :obj:`ValueError`
            If the shape of the :attr:`seed` array does not match the number of channels.

        Notes
        -----
        - Each channel's noise signal is generated using a unique random seed.
        - The type and characteristics of the noise are defined by the :attr:`signal` attribute.
        """
        Noise = self.signal.__class__
        # create or get the array of random seeds
        if not self.seed.size > 0:
            seed = arange(self.num_channels) + self.signal.seed
        elif self.seed.shape == (self.num_channels,):
            seed = self.seed
        else:
            msg = f'Seed array expected to be of shape ({self.num_channels:d},), but has shape {self.seed.shape}.'
            raise ValueError(msg)
        # create array with [num_channels] noise signal tracks
        signal = array(
            [
                Noise(seed=s, num_samples=self.num_samples, sample_freq=self.sample_freq, rms=self.signal.rms).signal()
                for s in seed
            ],
        ).T

        n = num
        while n <= self.num_samples:
            yield signal[n - num : n, :]
            n += num
        else:
            if (n - num) < self.num_samples:
                yield signal[n - num :, :]
            else:
                return


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples'}, read_only=True)
class SourceMixer(SamplesGenerator):
    """
    Combine signals from multiple sources by mixing their outputs.

    The :class:`SourceMixer` class takes signals generated by multiple
    :class:`~acoular.base.SamplesGenerator` instances and combines them into
    a single mixed output. The signals are weighted (if weights are provided)
    and added block-by-block, supporting efficient streaming.

    See Also
    --------
    :class:`acoular.base.SamplesGenerator` : Base class for signal generators.

    Notes
    -----
    - All sources must have the same sampling frequency, number of channels,
      and number of samples for proper mixing.
    - The weights for the sources can be specified to control their relative
      contributions to the mixed output. If no weights are provided, all sources
      are equally weighted.

    Examples
    --------
    Mix a stationary point source emitting a sine signal with two pink noise emitting point sources
    circling it and white noise for each channel:

    >>> import numpy as np
    >>> import acoular as ac
    >>>
    >>> # Generate positional microphone data for a 3x3 grid in the x-y plane at z=0
    >>> mic_positions = []
    >>> for i in range(3):
    ...     for j in range(3):
    ...         mic_positions.append([i - 1, j - 1, 0])  # Center the grid at the origin
    >>>
    >>> # Convert positions to the format required by MicGeom
    >>> mg = ac.MicGeom(pos_total=np.array(mic_positions).T)
    >>>
    >>> # Generate positional data for trajectories of two moving sources
    >>> # Trajectory 1: Circle in x-y plane at z=1
    >>> args = 2 * np.pi * np.arange(10) / 10  # Discrete points around the circle
    >>> x = np.cos(args)
    >>> y = np.sin(args)
    >>> z = np.ones_like(x)  # Constant height at z=1
    >>>
    >>> locs1 = np.array([x, y, z])
    >>> # Map time indices to positions for Trajectory 1
    >>> points1 = {time: tuple(pos) for time, pos in enumerate(locs1.T)}
    >>> tr1 = ac.Trajectory(points=points1)
    >>>
    >>> # Trajectory 2: Same circle but with a 180-degree phase shift
    >>> locs2 = np.roll(locs1, 5, axis=1)  # Shift the positions by half the circle
    >>> # Map time indices to positions for Trajectory 2
    >>> points2 = {time: tuple(pos) for time, pos in enumerate(locs2.T)}
    >>> tr2 = ac.Trajectory(points=points2)
    >>>
    >>> # Create signal sources
    >>> # Pink noise sources with different RMS values and random seeds
    >>> pinkNoise1 = ac.PNoiseGenerator(sample_freq=51200, num_samples=1024, rms=1.0, seed=42)
    >>> pinkNoise2 = ac.PNoiseGenerator(sample_freq=51200, num_samples=1024, rms=0.5, seed=24)
    >>>
    >>> # Moving sources emitting pink noise along their respective trajectories
    >>> pinkSource1 = ac.MovingPointSource(trajectory=tr1, signal=pinkNoise1, mics=mg)
    >>> pinkSource2 = ac.MovingPointSource(trajectory=tr2, signal=pinkNoise2, mics=mg)
    >>>
    >>> # White noise source generating uncorrelated noise for each microphone channel
    >>> whiteNoise = ac.WNoiseGenerator(sample_freq=51200, num_samples=1024, rms=1.0, seed=73)
    >>> whiteSources = ac.UncorrelatedNoiseSource(signal=whiteNoise, mics=mg)
    >>>
    >>> # Stationary point source emitting a sine wave
    >>> sineSignal = ac.SineGenerator(freq=1200, sample_freq=51200, num_samples=1024)
    >>> sineSource = ac.PointSource(signal=sineSignal, loc=(0, 0, 1), mics=mg)
    >>>
    >>> # Combine all sources in a SourceMixer with specified weights
    >>> sources = [pinkSource1, pinkSource2, whiteSources, sineSource]
    >>> mixer = ac.SourceMixer(sources=sources, weights=[1.0, 1.0, 0.3, 2.0])
    >>>
    >>> # Generate and process the mixed output block by block
    >>> for block in mixer.result(num=256):  # Generate blocks of 256 samples
    ...     print(block.shape)
    Pink noise filter depth set to maximum possible value of 10.
    Pink noise filter depth set to maximum possible value of 10.
    (256, 9)
    (256, 9)
    (256, 9)
    (256, 9)

    The output contains blocks of mixed signals. Each block is a combination of
    the four signals, weighted according to the provided weights.
    """

    #: List of :class:`~acoular.base.SamplesGenerator` instances to be mixed.
    #: Each source provides a signal that will be combined in the output.
    #: All sources must have the same sampling frequency, number of channels,
    #: and number of samples. The list must contain at least one source.
    sources = List(Instance(SamplesGenerator, ()))

    #: Sampling frequency of the mixed signal in Hz. Derived automatically from the
    #: first source in :attr:`sources`. If no sources are provided, default is ``0``.
    sample_freq = Property(depends_on=['sdigest'])

    #: Number of channels in the mixed signal. Derived automatically from the
    #: first source in :attr:`sources`. If no sources are provided, default is ``0``.
    num_channels = Property(depends_on=['sdigest'])

    #: Total number of samples in the mixed signal. Derived automatically from
    #: the first source in :attr:`sources`. If no sources are provided, default is ``0``.
    num_samples = Property(depends_on=['sdigest'])

    #: Array of amplitude weights for the sources. If not set, all sources are equally weighted.
    #: The size of the weights array must match the number of sources in :attr:`sources`.
    #: For example, with two sources, ``weights = [1.0, 0.5]`` would mix the first source at
    #: full amplitude and the second source at half amplitude.
    weights = CArray(desc='channel weights')

    #: Internal identifier for the combined state of all sources, used to track
    #: changes in the sources for reproducibility and caching.
    sdigest = Str()

    @observe('sources.items.digest')
    def _set_sources_digest(self, event):  # noqa ARG002
        self.sdigest = ldigest(self.sources)

    #: A unique identifier for the current state of the source,
    #: based on the states of the sources and the weights. (read-only)
    digest = Property(depends_on=['sdigest', 'weights'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sample_freq(self):
        return self.sources[0].sample_freq if self.sources else 0

    @cached_property
    def _get_num_channels(self):
        return self.sources[0].num_channels if self.sources else 0

    @cached_property
    def _get_num_samples(self):
        return self.sources[0].num_samples if self.sources else 0

    def validate_sources(self):
        """
        Ensure that all sources are compatible for mixing.

        This method checks that all sources in :attr:`sources` have the same
        sampling frequency, number of channels, and number of samples. A
        :class:`ValueError` is raised if any mismatch is detected.

        Raises
        ------
        :obj:`ValueError`
            If any source has incompatible attributes.
        """
        if len(self.sources) < 1:
            msg = 'Number of sources in SourceMixer should be at least 1.'
            raise ValueError(msg)
        for s in self.sources[1:]:
            if self.sample_freq != s.sample_freq:
                msg = f'Sample frequency of {s} does not fit'
                raise ValueError(msg)
            if self.num_channels != s.num_channels:
                msg = f'Channel count of {s} does not fit'
                raise ValueError(msg)
            if self.num_samples != s.num_samples:
                msg = f'Number of samples of {s} does not fit'
                raise ValueError(msg)

    def result(self, num):
        """
        Generate uncorrelated the mixed signal at microphones in blocks.

        The :meth:`result` method combines signals from all sources block-by-block,
        applying the specified weights to each source. The output blocks contain
        the mixed signal for all channels.

        Parameters
        ----------
        num : :class:`int`
            Number of samples per block to be yielded.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`num_channels`) containing the mixed
            signal. The last block may have fewer samples if the total number of samples
            is not a multiple of ``num``.

        Raises
        ------
        :obj:`ValueError`
            If the sources are not compatible for mixing.
        """
        # check whether all sources fit together
        self.validate_sources()

        gens = [i.result(num) for i in self.sources[1:]]
        weights = self.weights.copy()
        if weights.size == 0:
            weights = array([1.0 for j in range(len(self.sources))])
        assert weights.shape[0] == len(self.sources)
        for temp in self.sources[0].result(num):
            temp *= weights[0]
            sh = temp.shape[0]
            for j, g in enumerate(gens):
                temp1 = next(g) * weights[j + 1]
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[: temp1.shape[0]]
                temp += temp1[: temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break


class PointSourceConvolve(PointSource):
    """
    Blockwise convolution of a source signal with an impulse response (IR).

    The :class:`PointSourceConvolve` class extends :class:`PointSource` to simulate the effects of
    sound propagation through a room or acoustic environment by convolving the input signal with a
    specified :attr:`convolution kernel<kernel>` (the IR).

    The convolution is performed block-by-block to allow efficient streaming
    and processing of large signals.

    See Also
    --------
    :class:`PointSource` : Base class for point sources.
    :class:`acoular.tprocess.TimeConvolve` : Class used for performing time-domain convolution.

    Notes
    -----
    - The input :attr:`convolution kernel<kernel>` must be provided as a time-domain array.
    - The second dimension of :attr:`kernel` must either be ``1`` (a single kernel applied to all
      channels) or match the :attr:`number of channels<acoular.base.Generator.num_channels>`
      in the output.
    - Convolution is performed using the :class:`~acoular.tprocess.TimeConvolve` class.

    Examples
    --------
    Convolve a stationary sine wave source with a room impulse response (RIR):

    >>> import numpy as np
    >>> import acoular as ac
    >>>
    >>> # Define microphone geometry: 4 microphones in a 2x2 grid at z=0
    >>> mic_positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]).T
    >>> mg = ac.MicGeom(pos_total=mic_positions)
    >>>
    >>> # Generate a sine wave signal
    >>> sine_signal = ac.SineGenerator(freq=1000, sample_freq=48000, num_samples=1000)
    >>>
    >>> # Define an impulse response kernel (example: 100-tap random kernel)
    >>> kernel = np.random.randn(100, 1)  # One kernel for all channels
    >>>
    >>> # Create the convolving source
    >>> convolve_source = PointSourceConvolve(
    ...     signal=sine_signal,
    ...     loc=(0, 0, 1),  # Source located at (0, 0, 1)
    ...     kernel=kernel,
    ...     mics=mg,
    ... )
    >>>
    >>> # Generate the convolved signal block by block
    >>> for block in convolve_source.result(num=256):  # 256 samples per block
    ...     print(block.shape)
    (256, 4)
    (256, 4)
    (256, 4)
    (256, 4)
    (75, 4)

    The last block has fewer samples.
    """

    #: Convolution kernel in the time domain.
    #: The array must either have one column (a single kernel applied to all channels)
    #: or match the number of output channels in its second dimension.
    kernel = CArray(dtype=float, desc='Convolution kernel.')

    #: Start time of the signal in seconds. Default is ``0.0``.
    start_t = Enum(0.0, desc='signal start time')

    #: Start time of the data acquisition the the microphones in seconds. Default is ``0.0``.
    start = Enum(0.0, desc='sample start time')

    #: Behavior for negative time indices. Default is ``None``.
    prepadding = Enum(None, desc='Behavior for negative time indices.')

    #: Upsampling factor for internal use. Default is ``None``.
    up = Enum(None, desc='upsampling factor')

    #: Unique identifier for the current state of the source,
    #: based on microphone geometry, input signal, source location, and kernel. (read-only)
    digest = Property(depends_on=['mics.digest', 'signal.digest', 'loc', 'kernel'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """
        Generate the convolved signal at microphones in blocks.

        The :meth:`result` method produces blocks of the output signal
        by convolving the input signal with the specified kernel. Each block
        contains the signal for all output channels (microphones).

        Parameters
        ----------
        num : :class:`int`, optional
            The number of samples per block to yield. Default is ``128``.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D array of shape (``num``, :attr:`~PointSource.num_channels`) containing
            the convolved signal for all microphones. The last block may
            contain fewer samples if the total number of samples is not
            a multiple of ``num``.

        Notes
        -----
        - The input signal is expanded to match the number of microphones, if necessary.
        - Convolution is performed using the :class:`~acoular.tprocess.TimeConvolve` class
          to ensure efficiency.
        """
        data = repeat(self.signal.signal()[:, newaxis], self.mics.num_mics, axis=1)
        source = TimeSamples(
            data=data,
            sample_freq=self.sample_freq,
            num_samples=self.num_samples,
            num_channels=self.mics.num_mics,
        )
        time_convolve = TimeConvolve(
            source=source,
            kernel=self.kernel,
        )
        yield from time_convolve.result(num)
