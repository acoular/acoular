# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements general purpose blockwise processing methods.

.. autosummary::
    :toctree: generated/

    TimeAverage
    Average
    TimeCumAverage
    CumAverage
    TimeCache
    SampleSplitter
"""

import threading
from collections import deque
from inspect import currentframe
from warnings import warn

import numpy as np
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

    def __next__(self):  # this function implementation is not python 2 compatible!
        with self.lock:
            return self.it.__next__()


class TimeAverage(InOut):
    """Calculates time-dependent average of the signal."""

    #: Number of samples to average over, defaults to 64.
    naverage = Int(64, desc='number of samples to average over')

    #: Sampling frequency of the output signal, is set automatically.
    sample_freq = Property(depends_on='source.sample_freq, naverage')

    #: Number of samples of the output signal, is set automatically.
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


Average = TimeAverage
Average.__doc__ = """Alias for :class:`acoular.process.TimeAverage`."""
Average.__name__ = 'Average'


class TimeCumAverage(InOut):
    """Calculates cumulative average of the signal, useful for Leq."""

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).

        Returns
        -------
        Cumulative average of the output of source.
            Yields samples in blocks of shape (num, numchannels).
            The last block may be shorter than num.

        """
        count = (np.arange(num) + 1)[:, np.newaxis]
        for i, temp in enumerate(self.source.result(num)):
            ns, nc = temp.shape
            if not i:
                accu = np.zeros((1, nc))
            temp = (accu * (count[0] - 1) + np.cumsum(temp, axis=0)) / count[:ns]
            accu = temp[-1]
            count += ns
            yield temp


CumAverage = TimeCumAverage
CumAverage.__doc__ = """Alias for :class:`acoular.process.TimeCumAverage`."""
CumAverage.__name__ = 'CumAverage'


class TimeCache(InOut):
    """Caches time signal in cache file."""

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
        self.h5f.create_extendable_array(nodename, (0, self.numchannels), 'float32')
        ac = self.h5f.get_data_by_reference(nodename)
        self.h5f.set_node_attribute(ac, 'sample_freq', self.sample_freq)
        self.h5f.set_node_attribute(ac, 'complete', False)
        for data in self.source.result(num):
            self.h5f.append_data(ac, data)
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
        self.h5f.create_extendable_array(nodename, (0, self.numchannels), 'float32')
        ac = self.h5f.get_data_by_reference(nodename)
        self.h5f.set_node_attribute(ac, 'sample_freq', self.sample_freq)
        self.h5f.set_node_attribute(ac, 'complete', False)
        for j, data in enumerate(self.source.result(num)):
            self.h5f.append_data(ac, data)
            if j >= nblocks:
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
