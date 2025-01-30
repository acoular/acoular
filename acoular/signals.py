# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements signal generators for the simulation of acoustic sources.

.. autosummary::
    :toctree: generated/

    SignalGenerator
    PeriodicSignalGenerator
    NoiseGenerator
    WNoiseGenerator
    PNoiseGenerator
    FiltWNoiseGenerator
    SineGenerator
    GenericSignalGenerator

"""

# imports from other packages
from abc import abstractmethod
from warnings import warn

from numpy import arange, array, log, pi, repeat, sin, sqrt, tile, zeros
from numpy.random import RandomState
from scipy.signal import resample, sosfilt, tf2sos
from traits.api import (
    ABCHasStrictTraits,
    Bool,
    CArray,
    CInt,
    Delegate,
    Float,
    Instance,
    Int,
    Property,
    cached_property,
)

# acoular imports
from .base import SamplesGenerator
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'numsamples': 'num_samples'})
class SignalGenerator(ABCHasStrictTraits):
    """Virtual base class for a simple one-channel signal generator.

    Defines the common interface for all SignalGenerator classes. This class
    may be used as a base for specialized SignalGenerator implementations. It
    should not be used directly as it contains no real functionality.
    """

    #: Sampling frequency of the signal.
    sample_freq = Float(1.0, desc='sampling frequency')

    #: Number of samples to generate.
    num_samples = CInt

    # internal identifier
    digest = Property(depends_on=['sample_freq', 'num_samples'])

    @abstractmethod
    def _get_digest(self):
        """Returns the internal identifier."""

    @abstractmethod
    def signal(self):
        """Deliver the signal."""

    def usignal(self, factor):
        """Delivers the signal resampled with a multiple of the sampling freq.

        Uses fourier transform method for resampling (from scipy.signal).

        Parameters
        ----------
        factor : integer
            The factor defines how many times the new sampling frequency is
            larger than :attr:`sample_freq`.

        Returns
        -------
        array of floats
            The resulting signal of length `factor` * :attr:`num_samples`.
        """
        return resample(self.signal(), factor * self.num_samples)


class PeriodicSignalGenerator(SignalGenerator):
    """
    Abstract base class for periodic signal generators.

    Defines the common interface for all :class:`SignalGenerator`-derived classes with periodic
    signals. This class may be used as a base for class handling periodic signals that can be
    characterized by their frequency, phase and amplitude. It should not be used directly as it
    contains no real functionality.
    """

    #: Frequency of the signal, float, defaults to 1000.0.
    freq = Float(1000.0, desc='Frequency')

    #: Phase of the signal (in radians), float, defaults to 0.0.
    phase = Float(0.0, desc='Phase')

    #: Amplitude of the signal. Defaults to 1.0.
    amplitude = Float(1.0)

    # internal identifier
    digest = Property(depends_on=['amplitude', 'num_samples', 'sample_freq', 'freq', 'phase'])

    @abstractmethod
    def _get_digest(self):
        """Returns the internal identifier."""

    @abstractmethod
    def signal(self):
        """Deliver the signal."""


class NoiseGenerator(SignalGenerator):
    """Abstract base class for noise signal generators.

    Defines the common interface for all :class:`SignalGenerator` classes with noise signals. This
    class may be used as a base for class handling noise signals that can be characterized by their
    RMS amplitude. It should not be used directly as it contains no real functionality.
    """

    #: RMS amplitude of the signal.
    rms = Float(1.0, desc='rms amplitude')

    #: Seed for random number generator, defaults to 0.
    #: This parameter should be set differently for different instances
    #: to guarantee statistically independent (non-correlated) outputs.
    seed = Int(0, desc='random seed value')

    # internal identifier
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples'])

    @abstractmethod
    def _get_digest(self):
        """Returns the internal identifier."""

    @abstractmethod
    def signal(self):
        """Deliver the signal."""


class WNoiseGenerator(NoiseGenerator):
    """White noise signal generator."""

    # internal identifier
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.num_samples`.
        """
        rnd_gen = RandomState(self.seed)
        return self.rms * rnd_gen.standard_normal(self.num_samples)


class PNoiseGenerator(NoiseGenerator):
    """Pink noise signal generator.

    Simulation of pink noise is based on the Voss-McCartney algorithm.
    Ref.:

      * S.J. Orfanidis: Signal Processing (2010), pp. 729-733
      * online discussion: http://www.firstpr.com.au/dsp/pink-noise/

    The idea is to iteratively add larger-wavelength noise to get 1/f
    characteristic.
    """

    #: "Octave depth" -- higher values for 1/f spectrum at low frequencies,
    #: but longer calculation, defaults to 16.
    depth = Int(16, desc='octave depth')

    # internal identifier
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'depth'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        nums = self.num_samples
        depth = self.depth
        # maximum depth depending on number of samples
        max_depth = int(log(nums) / log(2))

        if depth > max_depth:
            depth = max_depth
            print(f'Pink noise filter depth set to maximum possible value of {max_depth:d}.')

        rnd_gen = RandomState(self.seed)
        s = rnd_gen.standard_normal(nums)
        for _ in range(depth):
            ind = 2**_ - 1
            lind = nums - ind
            dind = 2 ** (_ + 1)
            s[ind:] += repeat(rnd_gen.standard_normal(nums // dind + 1), dind)[:lind]
        # divide by sqrt(depth+1.5) to get same overall level as white noise
        return self.rms / sqrt(depth + 1.5) * s


class FiltWNoiseGenerator(WNoiseGenerator):
    """Filtered white noise signal following an autoregressive (AR), moving-average
    (MA) or autoregressive moving-average (ARMA) process.

    The desired frequency response of the filter can be defined by specifying
    the filter coefficients :attr:`ar` and :attr:`ma`.
    The RMS value specified via the :attr:`rms` attribute belongs to the white noise
    signal and differs from the RMS value of the filtered signal.
    For numerical stability at high orders, the filter is a combination of second order
    sections (sos).
    """

    ar = CArray(value=array([]), dtype=float, desc='autoregressive coefficients (coefficients of the denominator)')

    ma = CArray(value=array([]), dtype=float, desc='moving-average coefficients (coefficients of the numerator)')

    # internal identifier
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'ar', 'ma'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def handle_empty_coefficients(self, coefficients):
        if coefficients.size == 0:
            return array([1.0])
        return coefficients

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.num_samples`.
        """
        rnd_gen = RandomState(self.seed)
        ma = self.handle_empty_coefficients(self.ma)
        ar = self.handle_empty_coefficients(self.ar)
        sos = tf2sos(ma, ar)
        ntaps = ma.shape[0]
        sdelay = round(0.5 * (ntaps - 1))
        wnoise = self.rms * rnd_gen.standard_normal(
            self.num_samples + sdelay,
        )  # create longer signal to compensate delay
        return sosfilt(sos, x=wnoise)[sdelay:]


class SineGenerator(PeriodicSignalGenerator):
    """Sine signal generator with adjustable amplitude, frequency and phase."""

    # internal identifier
    digest = Property(depends_on=['num_samples', 'sample_freq', 'amplitude', 'freq', 'phase'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.num_samples`.
        """
        t = arange(self.num_samples, dtype=float) / self.sample_freq
        return self.amplitude * sin(2 * pi * self.freq * t + self.phase)


@deprecated_alias({'rms': 'amplitude'})
class GenericSignalGenerator(SignalGenerator):
    """Generate signal from output of :class:`~acoular.base.SamplesGenerator` object.

    This class can be used to inject arbitrary signals into Acoular processing
    chains. For example, it can be used to read signals from a HDF5 file or create any signal
    by using the :class:`acoular.sources.TimeSamples` class.

    Example
    -------
    >>> import numpy as np
    >>> from acoular import TimeSamples, GenericSignalGenerator
    >>> data = np.random.rand(1000, 1)
    >>> ts = TimeSamples(data=data, sample_freq=51200)
    >>> sig = GenericSignalGenerator(source=ts)
    """

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    #: Amplitude of the signal. Defaults to 1.0.
    amplitude = Float(1.0)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    _num_samples = CInt(0)

    #: Number of samples to generate. Is set to source.num_samples by default.
    num_samples = Property()

    def _get_num_samples(self):
        if self._num_samples:
            return self._num_samples
        return self.source.num_samples

    def _set_num_samples(self, num_samples):
        self._num_samples = num_samples

    #: Boolean flag, if 'True' (default), signal track is repeated if requested
    #: :attr:`num_samples` is higher than available sample number
    loop_signal = Bool(True)

    # internal identifier
    digest = Property(
        depends_on=['source.digest', 'loop_signal', 'num_samples', 'amplitude'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """Deliver the signal.

        Returns
        -------
        array of floats
            The resulting signal as an array of length :attr:`~GenericSignalGenerator.num_samples`.

        """
        block = 1024
        if self.source.num_channels > 1:
            warn(
                'Signal source has more than one channel. Only channel 0 will be used for signal.',
                Warning,
                stacklevel=2,
            )
        nums = self.num_samples
        track = zeros(nums)

        # iterate through source generator to fill signal track
        for i, temp in enumerate(self.source.result(block)):
            start = block * i
            stop = start + len(temp[:, 0])
            if nums > stop:
                track[start:stop] = temp[:, 0]
            else:  # exit loop preliminarily if wanted signal samples are reached
                track[start:nums] = temp[: nums - start, 0]
                break

        # if the signal should be repeated after finishing and there are still samples open
        if self.loop_signal and (nums > stop):
            # fill up empty track with as many full source signals as possible
            nloops = nums // stop
            if nloops > 1:
                track[stop : stop * nloops] = tile(track[:stop], nloops - 1)
            # fill up remaining empty track
            res = nums % stop  # last part of unfinished loop
            if res > 0:
                track[stop * nloops :] = track[:res]
        return self.amplitude * track
