# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Measured multichannel data management and simulation of acoustic sources.

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
    """Spherical Hankel Function of the First Kind."""
    return spherical_jn(n, z, derivative=False) + 1j * spherical_yn(n, z, derivative=False)


def get_radiation_angles(direction, mpos, sourceposition):
    """Returns azimuthal and elevation angles between the mics and the source.

    Parameters
    ----------
    direction : array of floats
        Spherical Harmonic orientation
    mpos : array of floats
        x, y, z position of microphones
    sourceposition : array of floats
        position of the source

    Returns
    -------
    azi, ele : array of floats
        the angle between the mics and the source

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
    """Returns Spherical Harmonic Radiation Pattern at the Microphones.

    Parameters
    ----------
    lOrder : int
        Maximal order of spherical harmonic
    direction : array of floats
        Spherical Harmonic orientation
    mpos : array of floats
        x, y, z position of microphones
    sourceposition : array of floats
        position of the source

    Returns
    -------
    modes : array of floats
        the radiation values at each microphone for each mode

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
    """Container for processing time data in `*.h5` or NumPy array format.

    This class loads measured data from HDF5 files and provides information about this data. It also
    serves as an interface where the data can be accessed (e.g. for use in a block chain) via the
    :meth:`result` generator.

    Examples
    --------
    Data can be loaded from a HDF5 file as follows:

    >>> from acoular import TimeSamples
    >>> file = <some_h5_file.h5>  # doctest: +SKIP
    >>> ts = TimeSamples(file=file)  # doctest: +SKIP
    >>> print(f'number of channels: {ts.num_channels}')  # doctest: +SKIP
    number of channels: 56 # doctest: +SKIP

    Alternatively, the time data can be specified directly as a numpy array.
    In this case, the :attr:`data` and :attr:`sample_freq` attributes must be set manually.

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

    See Also
    --------
    acoular.sources.MaskedTimeSamples:
        Extends the functionality of class :class:`TimeSamples` by enabling the definition of start
        and stop samples as well as the specification of invalid channels.
    """

    #: Full name of the .h5 file with data.
    file = File(filter=['*.h5'], exists=True, desc='name of data file')

    #: Basename of the .h5 file with data, is set automatically.
    basename = Property(depends_on=['file'], desc='basename of data file')

    #: Calibration data, instance of :class:`~acoular.calib.Calib` class, optional .
    calib = Instance(Calib, desc='Calibration data')

    #: Number of channels, is set automatically / read from file.
    num_channels = CInt(0, desc='number of input channels')

    #: Number of time data samples, is set automatically / read from file.
    num_samples = CInt(0, desc='number of samples')

    #: The time data as array of floats with dimension (num_samples, num_channels).
    data = Any(transient=True, desc='the actual time data array')

    #: HDF5 file object
    h5f = Instance(H5FileBase, transient=True)

    #: Provides metadata stored in HDF5 file object
    metadata = Dict(desc='metadata contained in .h5 file')

    # Checksum over first data entries of all channels
    _datachecksum = Property()

    # internal identifier
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
        """Open the .h5 file and set attributes."""
        if self.h5f is not None:
            with contextlib.suppress(OSError):
                self.h5f.close()
        file = _get_h5file_class()
        self.h5f = file(self.file)
        self._load_timedata()
        self._load_metadata()

    @on_trait_change('data')
    def _load_shapes(self):
        """Set num_channels and num_samples from data."""
        if self.data is not None:
            self.num_samples, self.num_channels = self.data.shape

    def _load_timedata(self):
        """Loads timedata from .h5 file. Only for internal use."""
        self.data = self.h5f.get_data_by_reference('time_data')
        self.sample_freq = self.h5f.get_node_attribute(self.data, 'sample_freq')

    def _load_metadata(self):
        """Loads metadata from .h5 file. Only for internal use."""
        self.metadata = {}
        if '/metadata' in self.h5f:
            self.metadata = self.h5f.node_to_dict('/metadata')

    def result(self, num=128):
        """Python generator that yields the output block-wise.

        Reads the time data either from a HDF5 file or from a numpy array given
        by :attr:`data` and iteratively returns a block of size `num` samples.
        Calibrated data is returned if a calibration object is given by :attr:`calib`.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Container for processing time data in `*.h5` or NumPy array format.

    This class loads measured data from HDF5 files and provides information about this data. It
    supports storing information about (in)valid samples and (in)valid channels and allows to
    specify a start and stop index for the valid samples. It also serves as an interface where the
    data can be accessed (e.g. for use in a block chain) via the :meth:`result` generator.

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

    #: Index of the first sample to be considered valid.
    start = CInt(0, desc='start of valid samples')

    #: Index of the last sample to be considered valid.
    stop = Union(None, CInt, desc='stop of valid samples')

    #: Channels that are to be treated as invalid.
    invalid_channels = List(int, desc='list of invalid channels')

    #: Channel mask to serve as an index for all valid channels, is set automatically.
    channels = Property(depends_on=['invalid_channels', 'num_channels_total'], desc='channel mask')

    #: Number of channels (including invalid channels), is set automatically.
    num_channels_total = CInt(0, desc='total number of input channels')

    #: Number of time data samples (including invalid samples), is set automatically.
    num_samples_total = CInt(0, desc='total number of samples per channel')

    #: Number of valid channels, is set automatically.
    num_channels = Property(
        depends_on=['invalid_channels', 'num_channels_total'], desc='number of valid input channels'
    )

    #: Number of valid time data samples, is set automatically.
    num_samples = Property(
        depends_on=['start', 'stop', 'num_samples_total'], desc='number of valid samples per channel'
    )

    # internal identifier
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
        # """ open the .h5 file and set attributes
        # """
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
        """Set num_channels and num_samples from data."""
        if self.data is not None:
            self.num_samples_total, self.num_channels_total = self.data.shape

    def _load_timedata(self):
        """Loads timedata from .h5 file. Only for internal use."""
        self.data = self.h5f.get_data_by_reference('time_data')
        self.sample_freq = self.h5f.get_node_attribute(self.data, 'sample_freq')
        (self.num_samples_total, self.num_channels_total) = self.data.shape

    def result(self, num=128):
        """Python generator that yields the output block-wise.

        Reads the time data either from a HDF5 file or from a numpy array given
        by :attr:`data` and iteratively returns a block of size `num` samples.
        Calibrated data is returned if a calibration object is given by :attr:`calib`.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Yields
        ------
        numpy.ndarray
            Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to define a fixed point source with an arbitrary signal.
    This can be used in simulations.

    The output is being generated via the :meth:`result` generator.
    """

    #:  Emitted signal, instance of the :class:`~acoular.signals.SignalGenerator` class.
    signal = Instance(SignalGenerator)

    #: Location of source in (`x`, `y`, `z`) coordinates (left-oriented system).
    loc = Tuple((0.0, 0.0, 1.0), desc='source location')

    #: Number of channels in output, is set automatically /
    #: depends on used microphone geometry.
    num_channels = Delegate('mics', 'num_mics')

    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Instance(MicGeom, desc='microphone geometry')

    def _validate_locations(self):
        dist = self.env._r(array(self.loc).reshape((3, 1)), self.mics.pos)
        if npany(dist < 1e-7):
            warn('Source and microphone locations are identical.', Warning, stacklevel=2)

    #: :class:`~acoular.environments.Environment` or derived object,
    #: which provides information about the sound propagation in the medium.
    env = Instance(Environment(), Environment)

    #: Start time of the signal in seconds, defaults to 0 s.
    start_t = Float(0.0, desc='signal start time')

    #: Start time of the data acquisition at microphones in seconds,
    #: defaults to 0 s.
    start = Float(0.0, desc='sample start time')

    #: Signal behaviour for negative time indices, i.e. if :attr:`start` < :attr:start_t.
    #: `loop` take values from the end of :attr:`signal.signal()` array.
    #: `zeros` set source signal to zero, advisable for deterministic signals.
    #: defaults to `loop`.
    prepadding = Enum('loop', 'zeros', desc='Behaviour for negative time indices.')

    #: Upsampling factor, internal use, defaults to 16.
    up = Int(16, desc='upsampling factor')

    #: Number of samples, is set automatically /
    #: depends on :attr:`signal`.
    num_samples = Delegate('signal')

    #: Sampling frequency of the signal, is set automatically /
    #: depends on :attr:`signal`.
    sample_freq = Delegate('signal')

    # internal identifier
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to define a fixed Spherical Harmonic Source with an arbitrary signal.
    This can be used in simulations.

    The output is being generated via the :meth:`result` generator.
    """

    #: Order of spherical harmonic source
    lOrder = Int(0, desc='Order of spherical harmonic')  # noqa: N815

    alpha = CArray(desc='coefficients of the (lOrder,) spherical harmonic mode')

    #: Vector to define the orientation of the SphericalHarmonic.
    direction = Tuple((1.0, 0.0, 0.0), desc='Spherical Harmonic orientation')

    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

    # internal identifier
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
        Y_lm = get_modes(
            lOrder=self.lOrder,
            direction=self.direction,
            mpos=self.mics.pos,
            sourceposition=array(self.loc),
        )
        return real(ifft(fft(signals, axis=0) * (Y_lm @ self.alpha), axis=0))

    def result(self, num=128):
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to define a point source with an arbitrary
    signal moving along a given trajectory.
    This can be used in simulations.

    The output is being generated via the :meth:`result` generator.
    """

    #: Considering of convective amplification
    conv_amp = Bool(False, desc='determines if convective amplification is considered')

    #: Trajectory of the source,
    #: instance of the :class:`~acoular.trajectory.Trajectory` class.
    #: The start time is assumed to be the same as for the samples.
    trajectory = Instance(Trajectory, desc='trajectory of the source')

    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

    # internal identifier
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to define a fixed point source with an arbitrary signal and
    dipole characteristics via superposition of two nearby inversely
    phased monopoles.
    This can be used in simulations.

    The output is being generated via the :meth:`result` generator.
    """

    #: Vector to define the orientation of the dipole lobes. Its magnitude
    #: governs the distance between the monopoles
    #: (dist = [lowest wavelength in spectrum] x [magnitude] x 1e-5).
    #: Note: Use vectors with order of magnitude around 1.0 or less
    #: for good results.
    direction = Tuple((0.0, 0.0, 1.0), desc='dipole orientation and distance of the inversely phased monopoles')

    prepadding = Enum('loop', desc='Behaviour for negative time indices.')

    # internal identifier
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    # internal identifier
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

    #: Reference vector, perpendicular to the x and y-axis of moving source.
    #: rotation source directivity around this axis
    rvec = CArray(dtype=float, shape=(3,), value=array((0, 0, 0)), desc='reference vector')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def get_emission_time(self, t, direction):
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
        """Function that yields the moving coordinates along the trajectory."""
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to define a fixed Line source with an arbitrary signal.
    This can be used in simulations.

    The output is being generated via the :meth:`result` generator.
    """

    #: Vector to define the orientation of the line source
    direction = Tuple((0.0, 0.0, 1.0), desc='Line orientation ')

    #: Vector to define the length of the line source in m
    length = Float(1, desc='length of the line source')

    #: number of monopol sources in the line source
    num_sources = Int(1)

    #: source strength for every monopole
    source_strength = CArray(desc='coefficients of the source strength')

    #:coherence
    coherence = Enum('coherent', 'incoherent', desc='coherence mode')

    # internal identifier
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    # internal identifier
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

    #: Reference vector, perpendicular to the x and y-axis of moving source.
    #: rotation source directivity around this axis
    rvec = CArray(dtype=float, shape=(3,), value=array((0, 0, 0)), desc='reference vector')

    @cached_property
    def _get_digest(self):
        return digest(self)

    def get_moving_direction(self, direction, time=0):
        """Function that yields the moving coordinates along the trajectory."""
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to simulate white or pink noise as uncorrelated signal at each
    channel.

    The output is being generated via the :meth:`result` generator.
    """

    #: Type of noise to generate at the channels.
    #: The `~acoular.signals.SignalGenerator`-derived class has to
    # feature the parameter "seed" (i.e. white or pink noise).
    signal = Instance(NoiseGenerator, desc='type of noise')

    #: Array with seeds for random number generator.
    #: When left empty, arange(:attr:`num_channels`) + :attr:`signal`.seed
    #: will be used.
    seed = CArray(dtype=uint32, desc='random seed values')

    #: Number of channels in output; is set automatically /
    #: depends on used microphone geometry.
    num_channels = Delegate('mics', 'num_mics')

    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Instance(MicGeom, desc='microphone geometry')

    #: Start time of the signal in seconds, defaults to 0 s.
    start_t = Float(0.0, desc='signal start time')

    #: Start time of the data acquisition at microphones in seconds,
    #: defaults to 0 s.
    start = Float(0.0, desc='sample start time')

    #: Number of samples is set automatically /
    #: depends on :attr:`signal`.
    num_samples = Delegate('signal')

    #: Sampling frequency of the signal; is set automatically /
    #: depends on :attr:`signal`.
    sample_freq = Delegate('signal')

    # internal identifier
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
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Mixes the signals from several sources."""

    #: List of :class:`~acoular.base.SamplesGenerator` objects
    #: to be mixed.
    sources = List(Instance(SamplesGenerator, ()))

    #: Sampling frequency of the signal.
    sample_freq = Property(depends_on=['sdigest'])

    #: Number of channels.
    num_channels = Property(depends_on=['sdigest'])

    #: Number of samples.
    num_samples = Property(depends_on=['sdigest'])

    #: Amplitude weight(s) for the sources as array. If not set,
    #: all source signals are equally weighted.
    #: Must match the number of sources in :attr:`sources`.
    weights = CArray(desc='channel weights')

    # internal identifier
    sdigest = Str()

    @observe('sources.items.digest')
    def _set_sources_digest(self, event):  # noqa ARG002
        self.sdigest = ldigest(self.sources)

    # internal identifier
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
        """Validates if sources fit together."""
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
        """Python generator that yields the output block-wise.
        The outputs from the sources in the list are being added.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
    """Class to blockwise convolve an arbitrary source signal with a room impulse response."""

    #: Convolution kernel in the time domain. The second dimension of the kernel array
    #: has to be either 1 or match :attr:`~SamplesGenerator.num_channels`.
    #: If only a single kernel is supplied, it is applied to all channels.
    kernel = CArray(dtype=float, desc='Convolution kernel.')

    # ------------- overwrite traits that are not supported by this class -------------

    #: Start time of the signal in seconds, defaults to 0 s.
    start_t = Enum(0.0, desc='signal start time')

    #: Start time of the data acquisition at microphones in seconds,
    #: defaults to 0 s.
    start = Enum(0.0, desc='sample start time')

    #: Signal behaviour for negative time indices, i.e. if :attr:`start` < :attr:start_t.
    #: `loop` take values from the end of :attr:`signal.signal()` array.
    #: `zeros` set source signal to zero, advisable for deterministic signals.
    #: defaults to `loop`.
    prepadding = Enum(None, desc='Behaviour for negative time indices.')

    #: Upsampling factor, internal use, defaults to 16.
    up = Enum(None, desc='upsampling factor')

    # internal identifier
    digest = Property(
        depends_on=['mics.digest', 'signal.digest', 'loc', 'kernel'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num=128):
        """Python generator that yields the output at microphones block-wise.

        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .

        Returns
        -------
        Samples in blocks of shape (num, num_channels).
            The last block may be shorter than num.

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
