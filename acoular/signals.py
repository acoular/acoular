# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
Implements signal generators for the simulation of acoustic sources.

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
    """
    ABC for a simple one-channel signal generator.

    This ABC defines the common interface and attributes for all signal generator implementations.
    It provides a template for generating one-channel signals with specified amplitude,
    sampling frequency, and duration. Subclasses should implement the core functionality,
    including signal generation and computation of the internal identifier.

    See Also
    --------
    :func:`scipy.signal.resample` : Used for resampling signals in the :meth:`usignal` method.

    Notes
    -----
    This class should not be instantiated directly. Instead, use a subclass that
    implements the required methods for signal generation.
    """

    #: Sampling frequency of the signal in Hz. Default is ``1.0``.
    sample_freq = Float(1.0, desc='sampling frequency')

    #: The number of samples to generate for the signal.
    num_samples = CInt

    #: A unique checksum identifier based on the object properties. (read-only)
    digest = Property(depends_on=['sample_freq', 'num_samples'])

    @abstractmethod
    def _get_digest(self):
        """Return the internal identifier."""

    @abstractmethod
    def signal(self):
        """
        Generate and return the signal.

        This method must be implemented by subclasses to provide the generated signal
        as a 1D array of samples.
        """

    def usignal(self, factor):
        """
        Resample the signal at a higher sampling frequency.

        This method uses Fourier transform-based resampling to deliver the signal at a
        sampling frequency that is a multiple of the original :attr:`sample_freq`.
        The resampled signal has a length of ``factor * num_samples``.

        Parameters
        ----------
        factor : int
            The resampling factor. Defines how many times larger the new sampling frequency is
            compared to the original :attr:`sample_freq`.

        Returns
        -------
        :class:`numpy.ndarray`
            The resampled signal as a 1D array of floats.

        Notes
        -----
        This method relies on the :func:`scipy.signal.resample` function for resampling.

        Examples
        --------
        Resample a signal by a factor of 4:

        >>> from acoular import SineGenerator  # Class extending SignalGenerator
        >>> sg = SineGenerator(sample_freq=100.0, num_samples=1000)
        >>> resampled_signal = sg.usignal(4)
        >>> len(resampled_signal)
        4000
        """
        return resample(self.signal(), factor * self.num_samples)


class PeriodicSignalGenerator(SignalGenerator):
    """
    Abstract base class for periodic signal generators.

    The :class:`PeriodicSignalGenerator` class defines the common interface
    for all :class:`SignalGenerator`-derived classes with periodic signals.
    This class may be used as a base for class handling periodic signals
    that can be characterized by their frequency, phase and amplitude.

    It should not be used directly as it contains no real functionality.

    See Also
    --------
    SineGenerator : Generate a sine signal.
    """

    #: The frequency of the signal. Default is ``1000.0``.
    freq = Float(1000.0, desc='Frequency')

    #: The phase of the signal (in radians). Default is ``0.0``.
    phase = Float(0.0, desc='Phase')

    #: The amplitude of the signal. Default is ``1.0``.
    amplitude = Float(1.0)

    #: Internal identifier based on generator properties. (read-only)
    digest = Property(depends_on=['amplitude', 'num_samples', 'sample_freq', 'freq', 'phase'])

    @abstractmethod
    def _get_digest(self):
        """Return the internal identifier."""

    @abstractmethod
    def signal(self):
        """Deliver the signal."""


class NoiseGenerator(SignalGenerator):
    """
    Abstract base class for noise signal generators.

    The :class:`NoiseGenerator` class defines the common interface for all :class:`SignalGenerator`
    classes with noise signals. This class may be used as a base for class handling noise signals
    that can be characterized by their RMS amplitude.

    It should not be used directly as it contains no real functionality.

    See Also
    --------
    :class:`acoular.signals.PNoiseGenerator` : For pink noise generation.
    :class:`acoular.signals.WNoiseGenerator` : For pink white generation.
    :class:`acoular.sources.UncorrelatedNoiseSource` : For per-channel noise generation.
    """

    #: Root mean square (RMS) amplitude of the signal. For a point source,
    #: this corresponds to the RMS amplitude at a distance of 1 meter. Default is ``1.0``.
    rms = Float(1.0, desc='rms amplitude')

    #: Seed for random number generator. Default is ``0``.
    #: This parameter should be set differently for different instances
    #: to guarantee statistically independent (non-correlated) outputs.
    seed = Int(0, desc='random seed value')

    #: Internal identifier based on generator properties. (read-only)
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples'])

    @abstractmethod
    def _get_digest(self):
        """Return the internal identifier."""

    @abstractmethod
    def signal(self):
        """Generate and deliver the periodic signal."""


class WNoiseGenerator(NoiseGenerator):
    """
    White noise signal generator.

    This class generates white noise signals with a specified
    :attr:`root mean square (RMS)<SignalGenerator.rms>` amplitude,
    :attr:`number of samples<SignalGenerator.num_samples>`, and
    :attr:`sampling frequency<SignalGenerator.sample_freq>`. The white noise is generated using a
    :obj:`random number generator<numpy.random.RandomState.standard_normal>` initialized with a
    :attr:`user-defined seed<seed>` for reproducibility.

    See Also
    --------
    :obj:`numpy.random.RandomState.standard_normal` :
        Used here to generate normally distributed noise.
    :class:`acoular.signals.PNoiseGenerator` : For pink noise generation.
    :class:`acoular.sources.UncorrelatedNoiseSource` : For per-channel noise generation.

    Examples
    --------
    Generate white noise with an RMS amplitude of 1.0 and 0.5:

    >>> from acoular import WNoiseGenerator
    >>> from numpy import mean
    >>>
    >>> # White noise with RMS of 1.0
    >>> gen1 = WNoiseGenerator(rms=1.0, num_samples=1000, seed=42)
    >>> signal1 = gen1.signal()
    >>>
    >>> # White noise with RMS of 0.5
    >>> gen2 = WNoiseGenerator(rms=0.5, num_samples=1000, seed=24)
    >>> signal2 = gen2.signal()
    >>>
    >>> mean(signal1) > mean(signal2)
    np.True_

    Ensure different outputs with different seeds:

    >>> gen1 = WNoiseGenerator(num_samples=3, seed=42)
    >>> gen2 = WNoiseGenerator(num_samples=3, seed=73)
    >>> gen1.signal() == gen2.signal()
    array([False, False, False])
    """

    # internal identifier
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """
        Generate and deliver the white noise signal.

        The signal is created using a Gaussian distribution with mean 0 and variance 1,
        scaled by the :attr:`RMS<SignalGenerator.rms>` amplitude of the object.

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D array of floats containing the generated white noise signal.
            The length of the array is equal to :attr:`~SignalGenerator.num_samples`.
        """
        rnd_gen = RandomState(self.seed)
        return self.rms * rnd_gen.standard_normal(self.num_samples)


class PNoiseGenerator(NoiseGenerator):
    """
    Generate pink noise signal.

    The :class:`PNoiseGenerator` class generates pink noise signals,
    which exhibit a :math:`1/f` power spectral density. Pink noise is characterized by
    equal energy per octave, making it useful in various applications such as audio testing,
    sound synthesis, and environmental noise simulations.

    The pink noise simulation is based on the Voss-McCartney algorithm, which iteratively adds
    noise with increasing wavelength to achieve the desired :math:`1/f` characteristic.

    See Also
    --------
    :class:`acoular.signals.WNoiseGenerator` : For white noise generation.
    :class:`acoular.sources.UncorrelatedNoiseSource` : For per-channel noise generation.

    References
    ----------
    - S.J. Orfanidis: Signal Processing (2010), pp. 729-733 :cite:`Orfanidis2010`
    - Online discussion: http://www.firstpr.com.au/dsp/pink-noise/
    """

    #: "Octave depth" of the pink noise generation. Higher values result in a better approximation
    #: of the :math:`1/f` spectrum at low frequencies but increase computation time. The  maximum
    #: allowable value depends on the :attr:`number of samples<SignalGenerator.num_samples>`.
    #: Default is ``16``.
    depth = Int(16, desc='octave depth')

    #: A unique checksum identifier based on the object properties. (read-only)
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'depth'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """
        Generate and deliver the pink noise signal.

        The signal is computed using the Voss-McCartney algorithm, which generates noise
        with a :math:`1/f` power spectral density. The method ensures that the output has the
        desired :attr:`RMS<SignalGenerator.rms>` amplitude and spectrum.

        Returns
        -------
        :class:`numpy.ndarray`
            A 1D array of floats containing the generated pink noise signal. The length
            of the array is equal to :attr:`~SignalGenerator.num_samples`.

        Notes
        -----
        - The "depth" parameter controls the number of octaves included in the pink noise
          simulation. If the specified depth exceeds the maximum possible value based on
          the number of samples, it is automatically adjusted, and a warning is printed.
        - The output signal is scaled to have the same overall level as white noise by dividing
          the result by ``sqrt(depth + 1.5)``.
        """
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
    """
    Generate filtered white noise using an **AR**, **MA**, or **ARMA** process.

        - **AR:** autoregressive (:attr:`ar`)
        - **MA:** moving-average (:attr:`ma`)
        - **ARMA:** autoregressive moving-average

    This class extends the :class:`WNoiseGenerator` class to apply a digital filter to white noise,
    producing a signal with specific characteristics based on the provided filter coefficients.
    The desired filter is defined using the autoregressive coefficients (:attr:`ar`) and
    moving-average coefficients (:attr:`ma`).

    The filter is implemented as a series of second-order sections (sos) for numerical stability,
    especially at high filter orders.

    See Also
    --------
    :class:`WNoiseGenerator` : For white noise generation.

    Notes
    -----
    - The output signal is adjusted for the group delay introduced by the filter, ensuring
      proper alignment of the filtered signal.
    - The RMS value specified in the :attr:`~NoiseGenerator.rms` attribute corresponds to the
      original white noise signal and not the filtered output.

    Examples
    --------
    Generate filtered white noise using :attr:`AR<ar>` and :attr:`MA<ma>` coefficients:

    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> # Define AR and MA coefficients
    >>> ar = np.array([1.0, -0.5])
    >>> ma = np.array([0.5, 0.5])
    >>>
    >>> # Create generator
    >>> gen = ac.FiltWNoiseGenerator(
    ...     rms=1.0,
    ...     seed=42,
    ...     sample_freq=1000,
    ...     num_samples=10000,
    ...     ar=ar,
    ...     ma=ma,
    ... )
    >>>
    >>> # Generate signal
    >>> signal = gen.signal()
    >>> print(signal[:10])  # Print the first 10 samples
    [0.24835708 0.30340346 0.40641385 1.28856612 1.2887213  0.41021549
     0.87764567 1.61214661 0.95505348 0.51406957]
    """

    #: A :class:`numpy.ndarray` of autoregressive coefficients (denominator). Default is ``[]``,
    #: which results in no AR filtering (i.e., all-pole filter is ``[1.0]``).
    ar = CArray(value=array([]), dtype=float, desc='autoregressive coefficients (coefficients of the denominator)')

    #: A :class:`numpy.ndarray` of moving-average coefficients (numerator). Default is ``[]``,
    #: which results in no MA filtering (i.e., all-zero filter is ``[1.0]``).
    ma = CArray(value=array([]), dtype=float, desc='moving-average coefficients (coefficients of the numerator)')

    #: A unique checksum identifier based on the object properties. (read-only)
    digest = Property(depends_on=['rms', 'seed', 'sample_freq', 'num_samples', 'ar', 'ma'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def handle_empty_coefficients(self, coefficients):
        """
        Handle empty filter coefficient arrays by returning a default value.

        This method ensures that both the autoregressive (:attr:`ar`) and moving-average
        (:attr:`ma`) coefficients are non-empty before filtering. If a coefficient array is empty,
        it is replaced with a default array containing a single value of ``1.0``.

        Parameters
        ----------
        coefficients : :class:`numpy.ndarray`
            Array of filter coefficients to check.

        Returns
        -------
        :class:`numpy.ndarray`
            The original array if it is non-empty, or a default array containing ``[1.0]``
            if the input array is empty.
        """
        if coefficients.size == 0:
            return array([1.0])
        return coefficients

    def signal(self):
        """
        Generate and return the filtered white noise signal.

        This method creates a white noise signal with the specified
        :attr:`RMS value<NoiseGenerator.rms>` and :attr:`~NoiseGenerator.seed`, then filters it
        using the autoregressive (:attr:`ar`) and moving-average (:attr:`ma`) coefficients.
        The filtering process compensates for group delay introduced by the filter.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`floats<float>`
            An array representing the filtered white noise signal. The length of the returned array
            is equal to :attr:`the number of samples<SignalGenerator.num_samples>`.
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
    r"""
    Generate a sine signal.

    The :class:`SineGenerator` class extends the :class:`PeriodicSignalGenerator` class and
    generates a sinusoidal signal based on specified :attr:`~PeriodicSignalGenerator.amplitude`,
    :attr:`frequency<PeriodicSignalGenerator.freq>`, and :attr:`~PeriodicSignalGenerator.phase`.

    This generator is commonly used for creating test signals in signal processing, acoustics,
    and control systems.

    The signal is defined as

    .. math::

        s(t) = A \sin(2 \pi f t + \phi)

    where:
        - :math:`A` is the amplitude,
        - :math:`f` is the frequency,
        - :math:`\phi` is the phase,
        - :math:`t` is the time (computed from the
          :attr:`sampling frequency<SignalGenerator.sample_freq>` and the
          :attr:`number of samples<SignalGenerator.num_samples>`).

    See Also
    --------
    PeriodicSignalGenerator : Base class for periodic signal generators.

    Examples
    --------
    Generate a sine wave signal:

    >>> import acoular as ac
    >>>
    >>> gen = ac.SineGenerator(
    ...     amplitude=2.0,
    ...     freq=50.0,
    ...     phase=0.0,
    ...     num_samples=1000,
    ...     sample_freq=1000,
    ... )
    >>> signal = gen.signal()
    >>> signal[:5]  # The first 5 samples
    array([0.        , 0.61803399, 1.1755705 , 1.61803399, 1.90211303])

    Generate a sine wave with a phase shift (arguably a cosine wave):

    >>> import numpy as np
    >>>
    >>> gen = ac.SineGenerator(
    ...     amplitude=1.0,
    ...     freq=100.0,
    ...     phase=np.pi / 2,
    ...     num_samples=500,
    ...     sample_freq=2000,
    ... )
    >>> signal = gen.signal()
    >>> signal[:5]  # The first 5 samples
    array([1.        , 0.95105652, 0.80901699, 0.58778525, 0.30901699])
    """

    #: A unique checksum identifier based on the object properties. (read-only)
    digest = Property(depends_on=['num_samples', 'sample_freq', 'amplitude', 'freq', 'phase'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        r"""
        Generate and return the sine wave signal.

        The method computes the sine wave based on the specified
        :attr:`~PeriodicSignalGenerator.amplitude`, :attr:`frequency<PeriodicSignalGenerator.freq>`,
        and :attr:`~PeriodicSignalGenerator.phase`. The time values are determined by the
        :attr:`sampling frequency<SignalGenerator.sample_freq>` and the
        :attr:`number of samples<SignalGenerator.num_samples>`.

        Returns
        -------
        :class:`numpy.ndarray` of :class:`floats<float>`
            A 1D array representing the sine wave signal.
            The length of the array is equal to :attr:`~SignalGenerator.num_samples`.

        Notes
        -----
        The generator supports high-frequency and high-resolution signals,
        limited by the Nyquist criterion.
        """
        t = arange(self.num_samples, dtype=float) / self.sample_freq
        return self.amplitude * sin(2 * pi * self.freq * t + self.phase)


@deprecated_alias({'rms': 'amplitude'})
class GenericSignalGenerator(SignalGenerator):
    """
    Generate signals from a :class:`~acoular.base.SamplesGenerator` or derived object.

    The :class:`GenericSignalGenerator` class enables the integration of arbitrary signals into
    Acoular processing chains. The signal is fetched from a specified data source and optionally
    scaled by an amplitude factor. It supports looping the signal to match the desired number of
    samples and can handle signals with multiple channels (only the first channel is used).

    Common use cases include:
        - Injecting custom or pre-recorded signals from HDF5 files.
        - Creating signals using the :class:`~acoular.sources.TimeSamples` class.
        - Generating a continuous or repeated signal for simulations.

    Notes
    -----
    If the signal source has more than one channel, only channel 0 is used.

    Examples
    --------
    Inject a random signal into a processing chain:

    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> data = np.random.rand(1000, 1)
    >>> ts = ac.TimeSamples(data=data, sample_freq=51200)
    >>> sig = ac.GenericSignalGenerator(source=ts)
    >>> output_signal = sig.signal()
    """

    #: The data source from which the signal is fetched.
    #: This can be any object derived from :class:`SamplesGenerator`.
    source = Instance(SamplesGenerator)

    #: Scaling factor applied to the generated signal. Defaults to ``1.0``.
    amplitude = Float(1.0)

    #: Sampling frequency of the output signal, as provided by the :attr:`source` object.
    sample_freq = Delegate('source')

    _num_samples = CInt(0)

    #: The number of samples to generate. Default is the number of samples available in the
    #: :attr:`source` (``source.num_samples``). If set explicitly, it can exceed the source length,
    #: in which case the signal will loop if :attr:`loop_signal` is ``True``.
    num_samples = Property()

    def _get_num_samples(self):
        if self._num_samples:
            return self._num_samples
        return self.source.num_samples

    def _set_num_samples(self, num_samples):
        self._num_samples = num_samples

    #: If ``True`` (default), the signal is repeated to meet the requested :attr:`num_samples`.
    #: If ``False``, the signal stops once the source data is exhausted.
    loop_signal = Bool(True)

    #: A unique checksum identifier based on the object properties. (read-only)
    digest = Property(
        depends_on=['source.digest', 'loop_signal', 'num_samples', 'amplitude'],
    )

    @cached_property
    def _get_digest(self):
        return digest(self)

    def signal(self):
        """
        Deliver the signal from the specified source.

        Returns
        -------
        :class:`numpy.array` of :class:`floats<float>`
            The resulting signal, scaled by the :attr:`amplitude` attribute, with a length
            matching :attr:`~GenericSignalGenerator.num_samples`.

        Warnings
        --------
        A warning is raised if the source has more than one channel.
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
