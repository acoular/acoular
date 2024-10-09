# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements general purpose blockwise processing methods independent of the domain (time or frequency).

.. autosummary::
    :toctree: generated/

    Average
    Cache
    SampleSplitter
    TimeAverage
    TimeCache
"""

import threading
from collections import deque
from inspect import currentframe
from warnings import warn

from traits.api import Bool, Dict, Instance, Int, Property, Trait, cached_property, on_trait_change

from .base import Generator, InOut
from .configuration import config
from .h5cache import H5cache
from .h5files import H5CacheFileBase
from .internal import digest


class LockedGenerator:
    """Creates a Thread Safe Iterator.
    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __next__(self):
        with self.lock:
            return self.it.__next__()


class Average(InOut):
    """Calculates the average across consecutive time samples or frequency snapshots.

    The average operation is performed differently depending on the source type.
    If the source is a time domain source (e.g. derived from :class:`~acoular.base.SamplesGenerator`),
    the average is calculated over a certain number of time samples given by :attr:`naverage`.
    If the source is a frequency domain source (e.g. derived from :class:`~acoular.base.SpectraGenerator`),
    the average is calculated over a certain number of snapshots given by :attr:`naverage`.

    Examples
    --------
        For estimate the RMS of a white noise (time-domain) signal, the average of the squared signal can be calculated:

        >>> import acoular as ac
        >>> import numpy as np
        >>>
        >>> signal = ac.WNoiseGenerator(sample_freq=51200, numsamples=51200, rms=2.0).signal()
        >>> ts = ac.TimeSamples(data=signal[:, np.newaxis], sample_freq=51200)
        >>> tp = ac.TimePower(source=ts)
        >>> avg = ac.Average(source=tp, naverage=512)
        >>> mean_squared_value = next(avg.result(num=1))
        >>> rms = np.sqrt(mean_squared_value)[0, 0]
        >>> print(rms)
        1.9985200025816718

        Here, each evaluation of the generator created by the :meth:`result` method of the :class:`Average` object
        via the :meth:`next` function returns :code:`num=1` average across a snapshot of 512 samples.

        If the source is a frequency domain source, the average is calculated over a certain number of
        snapshots, defined by :attr:`naverage`.

        >>> fft = ac.RFFT(source=ts, block_size=64)
        >>> ps = ac.AutoPowerSpectra(source=fft)
        >>> avg = ac.Average(source=ps, naverage=16)
        >>> mean_power = next(avg.result(num=1))
        >>> print(np.sqrt(mean_power.sum()))
        2.0024960894399295

        Here, the generator created by the :meth:`result` method of the :class:`Average` object
        returns the average across 16 snapshots in the frequency domain.

    """

    #: Number of samples (time domain source) or snapshots (frequency domain source)
    #: to average over, defaults to 64.
    naverage = Int(64, desc='number of samples to average over')

    #: Sampling frequency of the output signal, is set automatically.
    sample_freq = Property(depends_on='source.sample_freq, naverage')

    #: Number of samples (time domain) or snapshots (frequency domain) of the output signal.
    #: Is set automatically.
    numsamples = Property(depends_on='source.numsamples, naverage')

    # internal identifier
    digest = Property(depends_on=['source.digest', '__class__', 'naverage'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_sample_freq(self):
        if self.source:
            return 1.0 * self.source.sample_freq / self.naverage
        return None

    @cached_property
    def _get_numsamples(self):
        if self.source:
            return self.source.numsamples / self.naverage
        return None

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Average of the output of source.
            Yields samples in blocks of shape (num, numchannels).
            The last block may be shorter than num.

        """
        nav = self.naverage
        for temp in self.source.result(num * nav):
            ns, nc = temp.shape
            nso = int(ns / nav)
            if nso > 0:
                yield temp[: nso * nav].reshape((nso, -1, nc)).mean(axis=1)


class Cache(InOut):
    """Caches source output in cache file.

    This class is used to cache the output of a :class:`acoular.base.Generator` derived source
    object in a cache file to circumvent time-consuming re-calculation.
    The cache file is created in the Acoular cache directory.

    Examples
    --------
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
    [('_cache.h5', 1)]
    (1, 513)

    The caching behaviour can be controlled by the :class:`~acoular.configuration.Config` instance
    via the :attr:`~acoular.configuration.Config.global_caching` attribute.
    To turn off caching, set :attr:`~acoular.configuration.Config.global_caching` to 'none' before
    running the code. The cache file directory can be obtained (and set) via the
    :attr:`~acoular.configuration.Config.cache_dir`

    >>> ac.config.global_caching = 'none'

    """

    # basename for cache
    basename = Property(depends_on='digest')

    # hdf5 cache file
    h5f = Instance(H5CacheFileBase, transient=True)

    # internal identifier
    digest = Property(depends_on=['source.digest', '__class__'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    @cached_property
    def _get_basename(self):
        obj = self.source  # start with source
        basename = 'void'  # if no file source is found
        while obj:
            if 'basename' in obj.all_trait_names():  # at original source?
                basename = obj.basename  # get the name
                break
            try:
                obj = obj.source  # traverse down until original data source
            except AttributeError:
                obj = None
        return basename

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
        """Python generator that yields the output from cache block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, numchannels).
            The last block may be shorter than num.
            Echos the source output, but reads it from cache
            when available and prevents unnecassary recalculation.

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
                            "Cache file is incomplete for nodename %s. With config.global_caching='readonly', the cache file will not be used!"
                            % str(nodename),
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
    """Distributes data blocks from source to several following objects.
    A separate block buffer is created for each registered object in
    (:attr:`block_buffer`) .
    """

    #: dictionary with block buffers (dict values) of registered objects (dict
    #: keys).
    block_buffer = Dict(key_trait=Instance(Generator))

    #: max elements/blocks in block buffers.
    buffer_size = Int(100)

    #: defines behaviour in case of block_buffer overflow. Can be set individually
    #: for each registered object.
    #:
    #: * 'error': an IOError is thrown by the class
    #: * 'warning': a warning is displayed. Possibly leads to lost blocks of data
    #: * 'none': nothing happens. Possibly leads to lost blocks of data
    buffer_overflow_treatment = Dict(
        key_trait=Instance(Generator),
        value_trait=Trait('error', 'warning', 'none'),
        desc='defines buffer overflow behaviour.',
    )

    # shadow trait to monitor if source deliver samples or is empty
    _source_generator_exist = Bool(False)

    # shadow trait to monitor if buffer of objects with overflow treatment = 'error'
    # or warning is overfilled. Error will be raised in all threads.
    _buffer_overflow = Bool(False)

    # Helper Trait holds source generator
    _source_generator = Trait()

    def _create_block_buffer(self, obj):
        self.block_buffer[obj] = deque([], maxlen=self.buffer_size)

    def _create_buffer_overflow_treatment(self, obj):
        self.buffer_overflow_treatment[obj] = 'error'

    def _clear_block_buffer(self, obj):
        self.block_buffer[obj].clear()

    def _remove_block_buffer(self, obj):
        del self.block_buffer[obj]

    def _remove_buffer_overflow_treatment(self, obj):
        del self.buffer_overflow_treatment[obj]

    def _assert_obj_registered(self, obj):
        if obj not in self.block_buffer:
            raise OSError('calling object %s is not registered.' % obj)

    def _get_objs_to_inspect(self):
        return [obj for obj in self.buffer_overflow_treatment if self.buffer_overflow_treatment[obj] != 'none']

    def _inspect_buffer_levels(self, inspect_objs):
        for obj in inspect_objs:
            if len(self.block_buffer[obj]) == self.buffer_size:
                if self.buffer_overflow_treatment[obj] == 'error':
                    self._buffer_overflow = True
                elif self.buffer_overflow_treatment[obj] == 'warning':
                    warn('overfilled buffer for object: %s data will get lost' % obj, UserWarning, stacklevel=1)

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

    def register_object(self, *objects_to_register):
        """Function that can be used to register objects that receive blocks from this class."""
        for obj in objects_to_register:
            if obj not in self.block_buffer:
                self._create_block_buffer(obj)
                self._create_buffer_overflow_treatment(obj)

    def remove_object(self, *objects_to_remove):
        """Function that can be used to remove registered objects."""
        for obj in objects_to_remove:
            self._remove_block_buffer(obj)
            self._remove_buffer_overflow_treatment(obj)

    def result(self, num):
        """Python generator that yields the output block-wise from block-buffer.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Samples in blocks of shape (num, numchannels).
            Delivers a block of samples to the calling object.
            The last block may be shorter than num.

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
    """Calculates average of the signal (Alias for :class:`acoular.process.Average`).

    .. deprecated:: 24.10
        Using :class:`~acoular.process.TimeAverage` is deprecated and will be removed in Acoular
        version 25.07. Use :class:`~acoular.process.Average` instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using TimeAverage is deprecated and will be removed in Acoular version 25.07. Use Average instead.',
            DeprecationWarning,
            stacklevel=2,
        )


class TimeCache(Cache):
    """Caches source signals in cache file (Alias for :class:`acoular.process.Cache`).

    .. deprecated:: 24.10
        Using :class:`~acoular.process.TimeCache` is deprecated and will be removed in Acoular
        version 25.07. Use :class:`~acoular.process.Cache` instead.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warn(
            'Using TimeCache is deprecated and will be removed in Acoular version 25.07. Use Cache instead.',
            DeprecationWarning,
            stacklevel=2,
        )
