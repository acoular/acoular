# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""
General purpose blockwise processing methods independent of the domain (time or frequency).

.. autosummary::
    :toctree: generated/

    Average
    Cache
    SampleSplitter
    TimeAverage
    TimeCache
    SamplesBuffer
"""

import threading
from collections import deque
from inspect import currentframe
from warnings import warn

import numpy as np
from traits.api import Any, Array, Bool, Dict, Enum, Instance, Int, Property, Union, cached_property, on_trait_change

# acoular imports
from .base import Generator, InOut
from .configuration import config
from .deprecation import deprecated_alias
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest
from .tools.utils import find_basename


class LockedGenerator:
    """
    Thread-safe wrapper for an iterator or generator.

    The :class:`LockedGenerator` class ensures that calls to the ``__next__`` method of the
    given iterator or generator are thread-safe, preventing race conditions when accessed by
    multiple threads simultaneously.

    It achieves thread safety by using a lock to serialize access to the underlying
    iterator or generator.

    Parameters
    ----------
    it : iterator or generator
        The iterator or generator to be made thread-safe.

    See Also
    --------
    :class:`acoular.process.SampleSplitter` :
        Distribute data from a source to several following objects in a block-wise manner.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __next__(self):
        """Fetch the next item from the iterator or generator in a thread-safe manner."""
        with self.lock:
            return self.it.__next__()


@deprecated_alias({'naverage': 'num_per_average', 'numsamples': 'num_samples'}, read_only=['numsamples'])
class Average(InOut):
    """
    Calculate the average across consecutive time samples or frequency snapshots.

    The average operation is performed differently depending on the source type.
    If the source is a time domain source (e.g. derived from
    :class:`~acoular.base.SamplesGenerator`), the average is calculated
    over a certain number of time samples given by :attr:`num_per_average`.
    If the source is a frequency domain source (e.g. derived from
    :class:`~acoular.base.SpectraGenerator`), the average is calculated
    over a certain number of frequency snapshots given by :attr:`num_per_average`.

    See Also
    --------
    :class:`acoular.base.InOut` :
        Receive data from any source domain and return signals in the same domain.

    Examples
    --------
    To estimate the RMS of a white noise (time-domain) signal, the average of the squared
    signal can be calculated:

    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> signal = ac.WNoiseGenerator(sample_freq=51200, num_samples=51200, rms=2.0).signal()
    >>> ts = ac.TimeSamples(data=signal[:, np.newaxis], sample_freq=51200)
    >>> tp = ac.TimePower(source=ts)
    >>> avg = ac.Average(source=tp, num_per_average=512)
    >>> mean_squared_value = next(avg.result(num=1))
    >>> rms = np.sqrt(mean_squared_value)[0, 0]
    >>> print(rms)
    1.9985200025816718

    Here, each evaluation of the generator created by the :meth:`result` method of the
    :class:`Average` object via the :meth:`next` function returns :code:`num=1` average across a
    snapshot of 512 time samples.

    If the source is a frequency domain source, the average is calculated over a certain number
    of frequency snapshots, defined by :attr:`num_per_average`.

    >>> fft = ac.RFFT(source=ts, block_size=64)
    >>> ps = ac.AutoPowerSpectra(source=fft)
    >>> avg = ac.Average(source=ps, num_per_average=16)
    >>> mean_power = next(avg.result(num=1))
    >>> print(np.sqrt(mean_power.sum()))
    2.0024960894399295

    Here, the generator created by the :meth:`result` method of the :class:`Average` object
    returns the average across 16 snapshots in the frequency domain.
    """

    #: The number of samples (time domain source) or snapshots (frequency domain source)
    #: to average over. Default is ``64``.
    num_per_average = Int(64, desc='number of samples/snapshots to average over')

    #: The sampling frequency of the output signal. It is set automatically as
    #: (:attr:`~acoular.base.Generator.sample_freq` ``/`` :attr:`num_per_average`).
    sample_freq = Property(depends_on=['source.sample_freq', 'num_per_average'])

    #: The number of samples (time domain) or snapshots (frequency domain) of the output signal.
    #: It is set automatically as
    #: (:attr:`~acoular.base.Generator.num_samples` ``/`` :attr:`num_per_average`).
    num_samples = Property(depends_on=['source.num_samples', 'num_per_average'])

    #: A unique identifier based on the class properties.
    digest = Property(depends_on=['source.digest', 'num_per_average'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sample_freq(self):
        if self.source:
            return 1.0 * self.source.sample_freq / self.num_per_average
        return None

    @cached_property
    def _get_num_samples(self):
        if self.source:
            return self.source.num_samples / self.num_per_average
        return None

    def result(self, num):
        """
        Generate averaged output blocks from the source data.

        This method implements a Python generator that yields blocks of averaged data
        from the source. The averaging is performed over :attr:`num_per_average` samples
        (for time-domain sources) or snapshots (for frequency-domain sources).
        The size of the blocks yielded is defined by the ``num`` parameter.

        Parameters
        ----------
        num : :class:`int`
            The number of averaged blocks to yield at a time. Each block contains the average over
            :attr:`num_per_average` time samples or frequency snapshots. The last block may be
            shorter than the specified size if the remaining data is insufficient.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D NumPy array of shape ``(num, num_channels)``, where ``num`` is the number
            of averaged blocks requested, and ``num_channels`` corresponds to the number of channels
            in the source, as specified by :attr:`~acoular.base.Generator.num_channels`.
            Each entry in the array is the average over :attr:`num_per_average` samples/snapshots.

        Notes
        -----
        - The averaging operation depends on the source type:
            - For time-domain sources (e.g., derived from :class:`~acoular.base.SamplesGenerator`),
              the average is calculated over :attr:`num_per_average` time samples.
            - For frequency-domain sources (e.g., derived from
              :class:`~acoular.base.SpectraGenerator`), the average is calculated over
              :attr:`num_per_average` frequency snapshots.
        - The generator will stop yielding when the source data is exhausted.
        - If the source provides fewer than ``num * num_per_average`` samples,
          the final block may be smaller than the requested ``num`` size.
        """
        nav = self.num_per_average
        for temp in self.source.result(num * nav):
            ns, nc = temp.shape
            nso = int(ns / nav)
            if nso > 0:
                yield temp[: nso * nav].reshape((nso, -1, nc)).mean(axis=1)


class Cache(InOut):
    """
    Cache the output of a source in a file to avoid redundant computations.

    The :class:`Cache` class stores the output of a source (derived from
    :class:`~acoular.base.Generator`) in a cache file within the Acoular cache directory.
    This enables faster reuse of precomputed data by avoiding time-consuming recalculations.
    The cache behavior is managed through the :class:`~acoular.configuration.Config` class by
    setting the :attr:`~acoular.configuration.Config.global_caching` attribute.

    The class intelligently determines whether to use the cached data, update it,
    or bypass caching based on the global caching configuration and the state of the cache file.
    The caching mechanism supports scenarios such as:

    - Reading from a complete or incomplete cache.
    - Overwriting an existing cache.
    - Operating in a read-only or no-cache mode.

    See Also
    --------
    :class:`acoular.base.InOut` : Receive data from any source domain and return signals in the same
                                  domain.

    Examples
    --------
    Caching the output of an FFT computation:

    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> ac.config.h5library = 'tables'
    >>> data = np.random.rand(1024, 1)
    >>> ts = ac.TimeSamples(data=data, sample_freq=51200)
    >>> fft = ac.RFFT(source=ts, block_size=1024)
    >>> cache = ac.Cache(source=fft)  # cache the output of the FFT in cache file
    >>> for block in cache.result(num=1):  # read the cached data block-wise
    ...     print(block.shape)
    [('void_cache.h5', 1)]
    (1, 513)

    Disabling caching globally:

    >>> ac.config.global_caching = 'none'

    Changing the cache directory:

    >>> ac.config.cache_dir = '/path/to/cache_dir'  # doctest: +SKIP
    """

    # The basename for the cache file.
    # Derived from the :attr:`digest` property and used to uniquely identify the cache file.
    basename = Property(depends_on=['digest'])

    # The HDF5 cache file instance.
    # This is used to store or retrieve cached data in the Acoular cache directory.
    h5f = Instance(H5CacheFileBase, transient=True)

    #: A unique identifier based on the cache properties.
    digest = Property(depends_on=['source.digest'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        return find_basename(self.source)

    def _pass_data(self, num):
        yield from self.source.result(num)

    def _write_data_to_cache(self, num):
        nodename = 'tc_' + self.digest
        for i, data in enumerate(self.source.result(num)):
            if i == 0:
                self.h5f.create_extendable_array(nodename, (0, data.shape[1]), data.dtype.name)
                ac = self.h5f.get_data_by_reference(nodename)
                self.h5f.set_node_attribute(ac, 'sample_freq', self.sample_freq)
                self.h5f.set_node_attribute(ac, 'complete', False)
            self.h5f.append_data(ac, data)
            self.h5f.flush()
            yield data
        self.h5f.set_node_attribute(ac, 'complete', True)

    def _get_data_from_cache(self, num):
        nodename = 'tc_' + self.digest
        ac = self.h5f.get_data_by_reference(nodename)
        i = 0
        while i < ac.shape[0]:
            yield ac[i : i + num]
            i += num

    def _get_data_from_incomplete_cache(self, num):
        nodename = 'tc_' + self.digest
        ac = self.h5f.get_data_by_reference(nodename)
        i = 0
        nblocks = 0
        while i + num <= ac.shape[0]:
            yield ac[i : i + num]
            nblocks += 1
            i += num
        self.h5f.remove_data(nodename)
        for j, data in enumerate(self.source.result(num)):
            if j == 0:
                self.h5f.create_extendable_array(nodename, (0, data.shape[1]), data.dtype.name)
                ac = self.h5f.get_data_by_reference(nodename)
                self.h5f.set_node_attribute(ac, 'sample_freq', self.sample_freq)
                self.h5f.set_node_attribute(ac, 'complete', False)
            self.h5f.append_data(ac, data)
            if j >= nblocks:
                self.h5f.flush()
                yield data
        self.h5f.set_node_attribute(ac, 'complete', True)

    # result generator: delivers input, possibly from cache
    def result(self, num):
        """
        Generate data blocks from the source, using cache when available.

        This method acts as a Python generator that yields blocks of output data from the source,
        reading from the cache file when possible. The size of the data blocks is determined by the
        ``num`` parameter. The caching mechanism helps prevent redundant calculations by storing and
        reusing the source's output.

        Parameters
        ----------
        num : :class:`int`
            The number of time samples or frequency snapshots per block to yield.
            The final block may be smaller if there is insufficient data.

        Yields
        ------
        :class:`numpy.ndarray`
            A 2D NumPy array of shape ``(num, num_channels)`` representing the output data.
            Each block is either retrieved from the cache file or generated by the source
            and cached dynamically during processing.

        Notes
        -----
        - The behavior of the caching mechanism depends on the
          :attr:`~acoular.configuration.Config.global_caching` setting:

            - ``'none'``: Bypasses caching and directly retrieves data from the source.
            - ``'readonly'``: Reads data from the cache if available; otherwise,
              retrieves data from the source without caching.
            - ``'overwrite'``: Replaces any existing cache with newly computed data.

        - If the cache file is incomplete or corrupted, the method may generate new data
          from the source to update the cache unless the caching mode is ``'readonly'``.
        - The cache node name is based on the source's :attr:`digest` attribute.
        """
        if config.global_caching == 'none':
            generator = self._pass_data
        else:
            nodename = 'tc_' + self.digest
            H5cache.get_cache_file(self, self.basename)
            if not self.h5f:
                generator = self._pass_data
            elif self.h5f.is_cached(nodename):
                generator = self._get_data_from_cache
                if config.global_caching == 'overwrite':
                    self.h5f.remove_data(nodename)
                    generator = self._write_data_to_cache
                elif not self.h5f.get_data_by_reference(nodename).attrs.__contains__('complete'):
                    if config.global_caching == 'readonly':
                        generator = self._pass_data
                    else:
                        generator = self._get_data_from_incomplete_cache
                elif not self.h5f.get_data_by_reference(nodename).attrs['complete']:
                    if config.global_caching == 'readonly':
                        warn(
                            f"Cache file is incomplete for nodename {nodename}. With config.global_caching='readonly', \
                            the cache file will not be used!",
                            Warning,
                            stacklevel=1,
                        )
                        generator = self._pass_data
                    else:
                        generator = self._get_data_from_incomplete_cache
            elif not self.h5f.is_cached(nodename):
                generator = self._write_data_to_cache
                if config.global_caching == 'readonly':
                    generator = self._pass_data
        yield from generator(num)


class SampleSplitter(InOut):
    """
    Distribute data from a source to multiple connected objects in a block-wise manner.

    The :class:`SampleSplitter` class is designed to manage the distribution of data blocks from a
    single source object, derived from :class:`~acoular.base.Generator`, to multiple target
    objects, also derived from :class:`~acoular.base.Generator`. Each connected target object
    is assigned a dedicated buffer to hold incoming data blocks. These buffers operate in a
    first-in-first-out (FIFO) manner, ensuring efficient and parallelized data handling.

    This class is particularly useful when distributing data blocks from a streaming source
    to multiple downstream processing objects.

    Each registered target object maintains its own dedicated block buffer, allowing for independent
    data management. The buffer size can be customized per object, and different overflow handling
    strategies can be configured, such as raising an error, issuing a warning, or discarding old
    data. This ensures efficient parallel data processing, making it well-suited for complex
    workflows.

    Notes
    -----
    - Buffers are dynamically created and managed for each registered object.
    - Buffer overflow behavior can be set individually for each target object.

    Examples
    --------
    Consider a time-domain signal stream where the FFT spectra and signal power are calculated
    block-by-block and in parallel using the :class:`~acoular.fprocess.RFFT`,
    :class:`~acoular.tprocess.TimePower`, and :class:`~acoular.process.Average` objects.
    The :class:`SampleSplitter` is responsible for distributing incoming data blocks to the buffers
    of the :class:`~acoular.fprocess.RFFT` and :class:`~acoular.tprocess.TimePower` objects whenever
    either object requests data via the :meth:`result` generator.

    For the :class:`~acoular.tprocess.TimePower` object, the buffer size is set to 10 blocks.
    If the buffer is full, an error is raised, as the buffer overflow treatment is set to
    ``'error'``. For the :class:`~acoular.fprocess.RFFT` object, the buffer size is limited to 1
    block, and the overflow treatment is set to ``'none'``. This setup helps reduce latency in FFT
    calculations, which may take longer than signal power calculations. If new data arrives and the
    :class:`~acoular.fprocess.RFFT` buffer is full, the :class:`SampleSplitter` will discard the
    oldest block, ensuring that the :class:`~acoular.fprocess.RFFT`
    object always receives the most recent block of data.

    >>> import acoular as ac
    >>> import numpy as np
    >>>
    >>> # create a time domain signal source
    >>> ts = ac.TimeSamples(data=np.random.rand(1024, 1), sample_freq=51200)
    >>>
    >>> # create the sample splitter object
    >>> ss = ac.SampleSplitter(source=ts)
    >>>
    >>> # create the FFT spectra and further objects that receive the data
    >>> fft = ac.RFFT(source=ss, block_size=64)
    >>> pow = ac.TimePower(source=ss)
    >>> avg = ac.Average(source=pow, num_per_average=64)
    >>>
    >>> # register the subsequent processing block objects at the sample splitter
    >>> ss.register_object(fft, buffer_size=1, buffer_overflow_treatment='none')
    >>> ss.register_object(pow, buffer_size=10, buffer_overflow_treatment='error')

    After object registration, the ``SampleSplitter`` object is ready to distribute the data to the
    object buffers. The block buffers can be accessed via the :attr:`block_buffer` attribute of the
    ``SampleSplitter`` object.

    >>> ss.block_buffer.values()
    dict_values([deque([], maxlen=1), deque([], maxlen=10)])

    Calling the result method of the FFT object will start the data collection and distribution
    process.

    >>> generator = fft.result(num=1)
    >>> fft_res = next(generator)

    Although we haven't called the result method of the signal power object, one data block is
    already available in the buffer.

    >>> print(len(ss.block_buffer[pow]))
    1

    To remove registered objects from the :class:`SampleSplitter`, use the :meth:`remove_object`
    method.

    >>> ss.remove_object(pow)
    >>> print(len(ss.block_buffer))
    1
    """

    #: A dictionary containing block buffers for registered objects.
    #: Keys are the registered objects, and values are deque structures holding data blocks.
    block_buffer = Dict(key_trait=Instance(Generator))

    #: The maximum number of blocks each buffer can hold.
    #: Can be set globally for all objects or individually using a dictionary.
    buffer_size = Union(
        Int,
        Dict(key_trait=Instance(Generator), value_trait=Int),
        default_value=100,
    )

    #: Defines behavior when a buffer exceeds its maximum size.
    buffer_overflow_treatment = Dict(
        key_trait=Instance(Generator),
        value_trait=Enum('error', 'warning', 'none'),
        desc='defines buffer overflow behaviour.',
    )

    # A shadow trait to monitor if source deliver samples or is empty.
    _source_generator_exist = Bool(False)

    # A shadow trait to monitor if buffer of objects with overflow treatment = 'error'
    # or warning is overfilled. Error will be raised in all threads.
    _buffer_overflow = Bool(False)

    # Helper Trait holds source generator
    _source_generator = Instance(LockedGenerator)

    def _create_block_buffer(self, obj, buffer_size=None):
        if buffer_size is None:
            buffer_size = self.buffer_size if isinstance(self.buffer_size, int) else self.buffer_size[obj]
        self.block_buffer[obj] = deque([], maxlen=buffer_size)

    def _create_buffer_overflow_treatment(self, obj, buffer_overflow_treatment=None):
        if buffer_overflow_treatment is None:
            buffer_overflow_treatment = 'error'
        self.buffer_overflow_treatment[obj] = buffer_overflow_treatment

    def _clear_block_buffer(self, obj):
        self.block_buffer[obj].clear()

    def _remove_block_buffer(self, obj):
        del self.block_buffer[obj]

    def _remove_buffer_overflow_treatment(self, obj):
        del self.buffer_overflow_treatment[obj]

    def _assert_obj_registered(self, obj):
        if obj not in self.block_buffer:
            msg = f'calling object {obj} is not registered.'
            raise OSError(msg)

    def _get_objs_to_inspect(self):
        return [obj for obj in self.buffer_overflow_treatment if self.buffer_overflow_treatment[obj] != 'none']

    def _inspect_buffer_levels(self, inspect_objs):
        for obj in inspect_objs:
            if len(self.block_buffer[obj]) == self.buffer_size:
                if self.buffer_overflow_treatment[obj] == 'error':
                    self._buffer_overflow = True
                elif self.buffer_overflow_treatment[obj] == 'warning':
                    warn(f'overfilled buffer for object: {obj} data will get lost', UserWarning, stacklevel=1)

    def _create_source_generator(self, num):
        for obj in self.block_buffer:
            self._clear_block_buffer(obj)
        self._buffer_overflow = False  # reset overflow bool
        self._source_generator = LockedGenerator(self.source.result(num))
        self._source_generator_exist = True  # indicates full generator

    def _fill_block_buffers(self):
        next_block = next(self._source_generator)
        [self.block_buffer[obj].appendleft(next_block) for obj in self.block_buffer]

    @on_trait_change('buffer_size')
    def _change_buffer_size(self):  #
        for obj in self.block_buffer:
            self._remove_block_buffer(obj)
            self._create_block_buffer(obj)

    def register_object(self, *objects_to_register, buffer_size=None, buffer_overflow_treatment=None):
        """
        Register one or more target objects to the :class:`SampleSplitter` object.

        This method creates and configures block buffers for the specified target objects, enabling
        them to receive data blocks from the :class:`SampleSplitter`. Each registered object is
        assigned a dedicated buffer with customizable size and overflow behavior.

        Parameters
        ----------
        objects_to_register : :class:`~acoular.base.Generator` or list of :class:`~acoular.base.Generator`
            A single object or a list of objects derived from :class:`~acoular.base.Generator` to be
            registered as targets for data distribution.
        buffer_size : :class:`int`, optional
            The maximum number of data blocks each object's buffer can hold. If not specified,
            the default buffer size (100 blocks) is used, or a globally defined size if
            ``buffer_size`` is a dictionary.
        buffer_overflow_treatment : :attr:`str`, optional
            Defines the behavior when a buffer exceeds its maximum size. Options are:

            - ``'error'``: Raises an :obj:`IOError` when the buffer overflows.
            - ``'warning'``: Issues a warning and may result in data loss.
            - ``'none'``: Silently discards the oldest data blocks to make room for new ones.
              If not specified, the default behavior is ``'error'``.

        Raises
        ------
        :obj:`OSError`
            If any of the specified objects is already registered.
        """  # noqa: W505
        for obj in objects_to_register:
            if obj not in self.block_buffer:
                self._create_block_buffer(obj, buffer_size)
                self._create_buffer_overflow_treatment(obj, buffer_overflow_treatment)
            else:
                msg = f'object {obj} is already registered.'
                raise OSError(msg)

    def remove_object(self, *objects_to_remove):
        """
        Unregister one or more objects from the :class:`SampleSplitter`.

        This method removes the specified objects and their associated block buffers from the
        :class:`SampleSplitter`. If no objects are specified, all currently registered objects
        are unregistered, effectively clearing all buffers.

        Parameters
        ----------
        objects_to_remove : :class:`~acoular.base.Generator` or list of :class:`~acoular.base.Generator`, optional
            A single object or a list of objects derived from :class:`~acoular.base.Generator` to be
            removed from the :class:`SampleSplitter`.
            If no objects are provided, all registered objects will be removed.

        Raises
        ------
        :obj:`KeyError`
            If any of the specified objects are not currently registered.

        Notes
        -----
        - Once an object is removed, it will no longer receive data from the
          :class:`SampleSplitter`.
        - Removing an object also clears its associated buffer.
        """  # noqa: W505
        if not objects_to_remove:
            objects_to_remove = list(self.block_buffer.keys())
        for obj in objects_to_remove:
            self._remove_block_buffer(obj)
            self._remove_buffer_overflow_treatment(obj)

    def result(self, num):
        """
        Yield data blocks from the buffer to the calling object.

        This generator method retrieves blocks of data for the calling object, either
        from its dedicated block buffer or by processing new data from the source.
        If the buffer is empty, new data blocks are generated and distributed to
        all registered objects in a block-wise manner.

        Parameters
        ----------
        num : :class:`int`
            The size of each block to be yielded, defined as the number of samples per block.

        Yields
        ------
        :class:`numpy.ndarray`
            Blocks of data with shape ``(num, num_channels)``.
            The last block may be shorter than ``num`` if the source data is exhausted.

        Raises
        ------
        :obj:`OSError`
            If the calling object is not registered with the :class:`SampleSplitter`.
        :obj:`OSError`
            If the block buffer reaches its maximum size and the overflow handling
            policy is set to ``'error'``.

        Notes
        -----
        - If the block buffer is empty, new data is fetched from the source and distributed to all
          registered objects.
        - Buffer overflow behavior is controlled by the :attr:`buffer_overflow_treatment` attribute,
          which can be set to ``'error'``, ``'warning'``, or ``'none'``.
        """
        calling_obj = currentframe().f_back.f_locals['self']
        self._assert_obj_registered(calling_obj)
        objs_to_inspect = self._get_objs_to_inspect()

        if not self._source_generator_exist:
            self._create_source_generator(num)

        while not self._buffer_overflow:
            if self.block_buffer[calling_obj]:
                yield self.block_buffer[calling_obj].pop()
            else:
                self._inspect_buffer_levels(objs_to_inspect)
                try:
                    self._fill_block_buffers()
                except StopIteration:
                    self._source_generator_exist = False
                    return
        else:
            msg = 'Maximum size of block buffer is reached!'
            raise OSError(msg)


class TimeAverage(Average):
    """
    Calculate the average of the signal.

    .. deprecated:: 24.10
        The use of :class:`~acoular.process.TimeAverage` is deprecated
        and will be removed in Acoular version 25.07.
        Please use :class:`~acoular.process.Average` instead for future compatibility.

    Alias for :class:`~acoular.process.Average`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using TimeAverage is deprecated and will be removed in Acoular version 25.07. Use Average instead.',
            DeprecationWarning,
            stacklevel=2,
        )


class TimeCache(Cache):
    """
    Cache source signals in cache file.

    .. deprecated:: 24.10
        The use of :class:`~acoular.process.TimeCache` is deprecated
        and will be removed in Acoular version 25.07.
        Please use :class:`~acoular.process.Cache` instead for future compatibility.

    Alias for :class:`~acoular.process.Cache`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using TimeCache is deprecated and will be removed in Acoular version 25.07. Use Cache instead.',
            DeprecationWarning,
            stacklevel=2,
        )


class SamplesBuffer(InOut):
    """
    Handle buffering of samples from a source.

    The :class:`SamplesBuffer` class buffers samples from a source and provides them in blocks of a
    specified size. It supports various use cases for efficient handling of sample data.
    Below is an example demonstrating its functionality.

    Examples
    --------
    Suppose we want to draw blocks of 16 samples from the source, while ensuring that the buffer
    always holds twice that number (32 samples). The following code achieves this behavior:

    >>> import acoular as ac
    >>> import numpy as np
    >>> # create a white noise source with 512 samples
    >>> source = ac.TimeSamples(
    ...     data=ac.WNoiseGenerator(
    ...         sample_freq=64,
    ...         num_samples=512,
    ...     ).signal()[:, np.newaxis],
    ...     sample_freq=64,
    ... )
    >>> # create a buffer with a size of 32 samples
    >>> buffer = ac.process.SamplesBuffer(source=source, length=32)
    >>> # get the first block of 16 samples
    >>> block = next(buffer.result(num=16))
    >>> np.testing.assert_array_equal(block, source.data[:16])

    In the example above, the buffer initially collects blocks of the specified size from the
    source. It then returns the first block of 16 samples. With subsequent calls to the
    :meth:`result` method, the buffer refills and returns additional blocks of 16 samples.

    In some cases, you may wish to retrieve a different number of samples from the source than you
    want to return. This can be achieved by setting the :attr:`source_num` attribute. For example,
    in the :class:`~acoular.tbeamform.BeamformerTimeTraj` class, the number of time samples varies
    based on the expected delay for moving sources, while still adhering to the desired block size
    for the buffer.

    The :attr:`shift_index_by` attribute controls how the buffer updates its index when retrieving
    data. If set to ``'num'``, the buffer returns :attr:`result_num` samples but forgets ``'num'``
    samples from the buffer.
    If set to :attr:`result_num`, the buffer will return and forget the same number of samples.

    >>> buffer = ac.process.SamplesBuffer(source=source, length=32, result_num=20, shift_index_by='num')
    >>> block_sizes = []
    >>> block_sizes.append(
    ...     next(buffer.result(num=16)).shape[0]
    ... )  # this time, the buffer will return 20 samples, but the buffer will only forget the first 16 samples
    >>> buffer.result_num = 24
    >>> block_sizes.append(
    ...     next(buffer.result(num=16)).shape[0]
    ... )  # this time, the buffer will return 24 samples, but the buffer will only forget the first 16 samples
    >>> np.testing.assert_array_equal(block_sizes, [20, 24])
    """  # noqa: W505

    #: The number of samples that the buffer can hold.
    length = Int(desc='number of samples that fit in the buffer')

    #: The number of samples per block to obtain from the source. If set to ``None``, the number of
    #: samples will be determined by the ``num`` argument of the :meth:`result` method.
    source_num = Union(
        None,
        Int(),
        default_value=None,
        desc='number of samples to return from the source. If "None", use "num" argument of result method',
    )

    #: The number of samples to return from the buffer. If set to ``None``, the number of
    #: samples will be determined by the ``num`` argument of the :meth:`result` method.
    result_num = Union(
        None,
        Int(),
        default_value=None,
        desc="number of samples to return from the buffer. If 'None', use 'num' argument of result method",
    )

    #: Index shift value for the buffer.
    #:
    #: - If set to ``'result_num'``, the buffer will return and forget :attr:`result_num` samples.
    #: - If set to ``'num'``, the buffer will return :attr:`result_num` samples but forget ``num``
    #:   samples.
    shift_index_by = Enum(
        ('result_num', 'num'),
        desc=(
            'index shift value for the buffer. If "result_num", use "result_num" trait.'
            ' If "num", use "num" argument of result method'
        ),
    )

    #: The current filling level of the buffer, i.e., how many samples are currently available.
    level = Property(desc='current filling level of buffer')

    #: The data type of the elements in the buffer.
    dtype = Any(desc='data type of the buffer')

    # Flag indicating if the source is empty (for internal use).
    _empty_source = Bool(False, desc='flag to indicate that the source is empty')

    # The actual buffer holding the samples for processing.
    _buffer = Array(shape=(None, None), desc='buffer for block processing')

    # The current index position in the buffer.
    _index = Int(desc='current index in buffer')

    def _get_level(self):
        return self._buffer.shape[0] - self._index

    def _create_new_buffer(self):
        self._buffer = np.zeros((self.length, self.num_channels), dtype=self.dtype)
        self._index = self.length
        self._empty_source = False

    def _write_to_buffer(self, data):
        ns = data.shape[0]
        self._buffer[0 : (self.length - ns)] = self._buffer[-(self.length - ns) :]
        self._buffer[-ns:, :] = data.astype(self.dtype)
        self._index -= ns

    def increase_buffer(self, num):
        """
        Increase the size of the buffer by a specified number of samples.

        This method expands the buffer by appending additional samples, effectively increasing
        its capacity. The new samples are initialized to zero. The index of the buffer is adjusted
        accordingly to accommodate the increase.

        Parameters
        ----------
        num : :class:`int`
            The number of samples by which to increase the buffer size.
        """
        ar = np.zeros((num, self.num_channels), dtype=self._buffer.dtype)
        self._buffer = np.concatenate((ar, self._buffer), axis=0)
        self._index += num
        self.length += num

    def read_from_buffer(self, num):
        """
        Read a specified number of samples from the buffer.

        This method retrieves samples from the buffer, ensuring that the requested number of samples
        is returned. If the buffer contains fewer samples than requested, the method will return all
        available samples. The index of the buffer is updated based on the :attr:`shift_index_by`
        setting.

        Parameters
        ----------
        num : :class:`int`
            The number of samples to read from the buffer.

        Returns
        -------
        :class:`numpy.ndarray`
            A block of samples (array) from the buffer.

        Notes
        -----
        - If the :attr:`result_num` attribute is set, it determines the number of samples to return.
        - The method ensures the buffer index is adjusted according to the :attr:`shift_index_by`
          setting. Options are:

            - ``'result_num'``: The index will shift by the number of samples returned.
            - ``'num'``: The index will shift by the number of samples requested (``num``).
        """
        rnum = num if self.result_num is None else self.result_num
        rnum = rnum if self.level >= rnum else self.level
        data = self._buffer[self._index : self._index + rnum]
        if self.shift_index_by == 'result_num':
            self._index += rnum
        else:
            self._index += num
        return data

    def fill_buffer(self, snum):
        """
        Fill the buffer with samples from the source.

        The :meth:`fill_buffer` method collects samples from the source and writes them to the
        buffer. It continues to fill the buffer until there are enough samples available, or the
        source runs out of data. If the buffer reaches its maximum capacity, additional samples are
        discarded. The buffer will only contain the most recent data, and its index will be updated
        accordingly.

        Parameters
        ----------
        snum : :class:`int`
            The number of samples to retrieve from the source in each iteration.

        Yields
        ------
        :obj:`None`
            This method is a generator and yields control back after filling the buffer.

        Notes
        -----
        - The method ensures that the buffer is filled with the required number of samples,
          adjusting the buffer size if necessary (via the :meth:`increase_buffer` method) when more
          space is needed.
        - Once the buffer is filled, it yields control and resumes only when the buffer is ready for
          more data.
        """
        source_generator = self.source.result(snum)
        while not self._empty_source:
            while self._index >= snum:
                if self.result_num is not None:
                    while self.result_num > self.length:
                        self.increase_buffer(snum)
                try:
                    self._write_to_buffer(next(source_generator))
                except StopIteration:
                    self._empty_source = True
                    break
            yield

    def result(self, num):
        """
        Return blocks of samples from the buffer.

        The :meth:`result` method retrieves blocks of samples from the buffer and yields them to the
        calling process. The number of samples per block is determined by the ``num`` argument, but
        can also be influenced by other attributes like `result_num` (if set). If the buffer is not
        yet filled, it will continue to collect samples from the source until the buffer contains
        enough data. Once the buffer is full, it will return the requested blocks of samples.

        Parameters
        ----------
        num : :class:`int`
            The number of samples to return in each block.
            This value specifies the size of the blocks to be yielded from the buffer.

        Yields
        ------
        :class:`numpy.ndarray`
            A block of samples from the buffer. The size of the block is determined by the ``num``
            parameter or the :attr:`result_num` attribute, depending on the buffer's configuration.

        Notes
        -----
        - If :attr:`result_num` is set, the method will use it to determine the number of samples
          returned instead of the ``num`` parameter.
        - If the buffer is empty or does not have enough samples, it will attempt to fill the buffer
          by collecting data from the source. If there are not enough samples available from the
          source, the method will yield whatever samples are left in the buffer.
        """
        self._create_new_buffer()
        snum = num
        if self.source_num is not None:
            snum = self.source_num
        for _ in self.fill_buffer(snum):
            if self.level > 0:
                yield self.read_from_buffer(num)
            else:
                break
        while self.level > 0:
            yield self.read_from_buffer(num)
