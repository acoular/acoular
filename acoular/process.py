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
        blocksize = (num -1) * 2
        for temp in self.source.result(num):
            yield (temp * temp.conjugate()).real / blocksize

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
    """Averages data over a number of blocks.

    The class can be used to average blocks of data.
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

    def _result_noaverage(self, num):
        yield from self.source.result(num)

    def _result_fullaverage(self, num):
        """Averages the data over all blocks the source yields."""
        for i, data in enumerate(self.source.result(num)):
            if i == 0:
                data_avg = data.copy()
            else:
                n = data.shape[0]
                data_avg[:n] += data
            i += 1
        yield data_avg / i

    def _result_numaverage(self, num):
        """Averages the data over a fixed number of blocks."""
        i = 0
        for data in self.source.result(num):
            if i == 0:
                data_avg = data.copy()
            else:
                n = data.shape[0]
                data_avg[:n] += data
            if i == self.numaverage:
                yield data_avg / i
                i = 0
                continue
            i += 1
        yield data_avg / i

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
        if self.numaverage is None:
            yield from self._result_fullaverage(num)
        elif self.numaverage == 0:
            yield from self._result_noaverage(num)
        else:
            yield from self._result_numaverage(num)
