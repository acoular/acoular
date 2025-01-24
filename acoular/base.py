# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements base classes for signal processing blocks in Acoular.

The classes in this module are abstract base classes that provide a common interface for all classes
that generate an output via the generator :meth:`result` in block-wise manner. They are not intended
to be used directly, but to be subclassed by classes that implement the actual signal processing.

.. autosummary::
    :toctree: generated/

    Generator
    SamplesGenerator
    SpectraGenerator
    InOut
    TimeOut
    SpectraOut
    TimeInOut
"""

from abc import abstractmethod

from traits.api import (
    ABCHasStrictTraits,
    CArray,
    CInt,
    Delegate,
    Float,
    Instance,
    Property,
    cached_property,
)

# acoular imports
from .deprecation import deprecated_alias
from .internal import digest


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples'})
class Generator(ABCHasStrictTraits):
    """Interface for any generating signal processing block.

    It provides a common interface for all classes, which generate an output via the generator
    :meth:`result` in block-wise manner. It has a common set of traits that are used by all classes
    that implement this interface. This includes the sampling frequency of the signal
    (:attr:`sample_freq`), the number of samples (:attr:`num_samples`), and the number of channels
    (:attr:`num_channels`). A private trait :attr:`digest` is used to store the internal identifier
    of the object, which is a hash of the object's attributes.
    This is used to check if the object's internal state has changed.

    """

    #: Sampling frequency of the signal, defaults to 1.0
    sample_freq = Float(1.0, desc='sampling frequency')

    #: Number of signal samples
    num_samples = CInt

    #: Number of channels
    num_channels = CInt

    # internal identifier
    digest = Property(depends_on=['sample_freq', 'num_samples', 'num_channels'])

    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def result(self, num):
        """Python generator that yields the output block-wise.

        This method needs to be implemented by the derived classes.

        Parameters
        ----------
        num : int
            The size of the first dimension of the blocks to be yielded

        Yields
        ------
        numpy.ndarray
            Two-dimensional output data block of shape (num, ...)
        """


class SamplesGenerator(Generator):
    """Interface for any generating multi-channel time domain signal processing block.

    It provides a common interface for all SamplesGenerator classes, which generate an output via
    the generator :meth:`result` in block-wise manner. This class has no real functionality on its
    own and should not be used directly.

    """

    # internal identifier
    digest = Property(depends_on=['sample_freq', 'num_samples', 'num_channels'])

    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block)

        Yields
        ------
        numpy.ndarray
            The two-dimensional time-data block of shape (num, num_channels).
        """


class SpectraGenerator(Generator):
    """Interface for any generating multi-channel signal frequency domain processing block.

    It provides a common interface for all SpectraGenerator classes, which generate an output via
    the generator :meth:`result` in block-wise manner. This class has no real functionality on its
    own and should not be used directly.

    """

    #: Number of frequencies
    num_freqs = CInt

    #: 1-D array of frequencies
    freqs = CArray

    #: The length of the block used to calculate the spectra
    block_size = CInt

    # internal identifier
    digest = Property(depends_on=['sample_freq', 'num_samples', 'num_channels', 'num_freqs', 'block_size'])

    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def result(self, num=1):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the number of snapshots to be yielded.
            Defaults to 1.

        Yields
        ------
        numpy.ndarray
            A two-dimensional block of shape (num, num_channels * num_freqs).
        """


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples'}, read_only=True)
class TimeOut(SamplesGenerator):
    """Abstract base class for any signal processing block that receives data from any
    :attr:`source` domain and returns time domain signals.

    It provides a base class that can be used to create signal processing blocks that receive data
    from any generating :attr:`source` and generates a time signal output via the generator
    :meth:`result` in block-wise manner.
    """

    #: Data source; :class:`~acoular.base.Generator` or derived object.
    source = Instance(Generator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of channels in output, as given by :attr:`source`.
    num_channels = Delegate('source')

    #: Number of samples in output, as given by :attr:`source`.
    num_samples = Delegate('source')

    # internal identifier
    digest = Property(depends_on=['source.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def result(self, num):
        """Python generator that processes the source data and yields the time-signal block-wise.

        This method needs to be implemented by the derived classes.

        Parameters
        ----------
        num : int
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block)

        Yields
        ------
        numpy.ndarray
            Two-dimensional output data block of shape (num, num_channels)
        """


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples', 'numfreqs': 'num_freqs'}, read_only=True)
class SpectraOut(SpectraGenerator):
    """Abstract base class for any signal processing block that receives data from any
    :attr:`source` domain and returns frequency domain signals.

    It provides a base class that can be used to create signal processing blocks that receive data
    from any generating :attr:`source` domain and generates a frequency domain output via the
    generator :meth:`result` in block-wise manner.
    """

    #: Data source; :class:`~acoular.base.Generator` or derived object.
    source = Instance(Generator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of channels in output, as given by :attr:`source`.
    num_channels = Delegate('source')

    #: Number of snapshots in output, as given by :attr:`source`.
    num_samples = Delegate('source')

    #: Number of frequencies in output, as given by :attr:`source`.
    num_freqs = Delegate('source')

    #: 1-D array of frequencies, as given by :attr:`source`.
    freqs = Delegate('source')

    #: The size of the block used to calculate the spectra
    block_size = Delegate('source')

    # internal identifier
    digest = Property(depends_on=['source.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def result(self, num=1):
        """Python generator that processes the source data and yields the output block-wise.

        This method needs to be implemented by the derived classes.

        num : integer
            This parameter defines the the number of snapshots to be yielded.
            Defaults to 1.

        Yields
        ------
        numpy.ndarray
            A two-dimensional block of shape (num, num_channels * num_freqs).
        """


@deprecated_alias({'numchannels': 'num_channels', 'numsamples': 'num_samples'}, read_only=True)
class InOut(SamplesGenerator, SpectraGenerator):
    """Abstract base class for any signal processing block that receives data from any
    :attr:`source` domain and returns signals in the same domain.

    It provides a base class that can be used to create signal processing blocks that receive data
    from any generating :attr:`source` and generates an output via the generator :meth:`result` in
    block-wise manner.
    """

    #: Data source; :class:`~acoular.base.Generator` or derived object.
    source = Instance(Generator)

    #: Sampling frequency of output signal, as given by :attr:`source`.
    sample_freq = Delegate('source')

    #: Number of channels in output, as given by :attr:`source`.
    num_channels = Delegate('source')

    #: Number of frequencies in output, as given by :attr:`source`.
    num_freqs = Delegate('source')

    #: Number of samples / snapshots in output, as given by :attr:`source`.
    num_samples = Delegate('source')

    # internal identifier
    digest = Property(depends_on=['source.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @abstractmethod
    def result(self, num):
        """Python generator that processes the source data and yields the output block-wise.

        This method needs to be implemented by the derived classes.

        Parameters
        ----------
        num : int
            The size of the first dimension of the blocks to be yielded

        Yields
        ------
        numpy.ndarray
            Two-dimensional output data block of shape (num, ...)
        """


class TimeInOut(TimeOut):
    """Deprecated alias for :class:`~acoular.base.TimeOut`.

    .. deprecated:: 24.10
        Using :class:`~acoular.base.TimeInOut` is deprecated and will be removed in Acoular 25.07.
        Use :class:`~acoular.base.TimeOut` instead.
    """

    #: Data source; :class:`~acoular.base.SamplesGenerator` or derived object.
    source = Instance(SamplesGenerator)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import warnings

        warnings.warn(
            'TimeInOut is deprecated and will be removed in Acoular 25.07. Use TimeOut instead.',
            DeprecationWarning,
            stacklevel=2,
        )
