# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements blockwise processing methods in that are generally applicable in time and frequency domain.

.. autosummary::
    :toctree: generated/

    Power
    BlockAverage

"""

import multiprocessing

from traits.api import Either, Instance, Int, Property, cached_property

from acoular.fprocess import FreqInOut
from acoular.internal import digest
from acoular.tprocess import TimeInOut

CPU_COUNT = multiprocessing.cpu_count()


class Power(FreqInOut, TimeInOut):
    """Calculates power of the signal.

    The class can be used to calculate the power of the signal in the frequency domain.
    and in the time domain.
    """

    #: Data source; either of :class:`~acoular.fprocess.FreqInOut` or
    # :class:`acoular.tprocess.TimeInOut` derived object.
    source = Either(Instance(FreqInOut), Instance(TimeInOut), desc='data source')

    def _fresult(self, num):
        for temp in self.source.result(num):
            yield (temp * temp.conjugate()).real

    def _tresult(self, num):
        for temp in self.source.result(num):
            yield temp * temp

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            If the source yields frequency data, num corresponds to the number of frequencies.
            If the source yields time data, num corresponds to the number of samples per block.

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, numchannels, numfreq).
            The last block may be shorter than num.

        """
        if isinstance(self.source, FreqInOut):
            yield from self._fresult(num)
        else:
            yield from self._tresult(num)


class BlockAverage(TimeInOut, FreqInOut):
    """Averages frequency data over a number of blocks.

    The class can be used to average frequency data over a number of blocks.
    """

    #: Data source; either of :class:`~acoular.fprocess.FreqInOut` or
    # :class:`acoular.tprocess.TimeInOut` derived object.
    source = Either(Instance(FreqInOut), Instance(TimeInOut), desc='data source')

    #: Number of frequency spectra to average over, defaults to :code:`None`. Averages
    # as long as the source yields data.
    numaverage = Either(None, Int, desc='number of frequency spectra to average over')

    # internal identifier
    digest = Property(depends_on=['source.digest', 'numaverage'])

    @cached_property
    def _get_digest(self):
        return digest(self)

    def result(self, num):
        """Python generator that yields the output block-wise.

        Parameters
        ----------
        num : integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of frequencies per block).

        Yields
        ------
        numpy.ndarray
            Yields blocks of shape (num, numchannels).
        """
        i = 0
        data_avg = None
        for data in self.source.result(num):
            data_avg = data.copy() if i == 0 else (i - 1) / i * data_avg + 1 / i * data
            if self.numaverage == 0:
                yield data_avg
                continue
            i += 1
            if (self.numaverage is not None) and (i % self.numaverage == 0):
                yield data_avg
                data_avg = None
                i = 0
        if self.numaverage is None:
            yield data_avg
