"""Utility classes intended for internal use in Acoular.

.. autosummary::
    :toctree: generated/

    SamplesBuffer
"""

import numpy as np
from traits.api import Any, Array, Bool, Enum, Int, Property, Union

from acoular.process import InOut


class SamplesBuffer(InOut):
    """Handles buffering of samples from a source.

    This class is used to buffer samples from a source and provide them in blocks
    of a specified size. There are several usecases for this class, as demonstrated in
    the following.

    Examples
    --------
    Let us assume we want to draw blocks of 16 samples from our source, but we want to make sure
    that we always have twice the number of samples in the buffer. We can achieve this simple behaviour
    by using the following code:

    >>> import acoular as ac
    >>> import numpy as np
    >>> # create a white noise source with 512 samples
    >>> source = ac.TimeSamples(
    ...     data=ac.WNoiseGenerator(
    ...         sample_freq=64,
    ...         numsamples=512,
    ...     ).signal()[:, np.newaxis],
    ...     sample_freq=64,
    ... )
    >>> # create a buffer with a size of 32 samples
    >>> buffer = ac.tools.SamplesBuffer(source=source, length=32)
    >>> # get the first block of 16 samples
    >>> block = next(buffer.result(num=16))
    >>> np.testing.assert_array_equal(block, source.data[:16])

    Here, on the first call to the result method, the buffer will fill up by collecting blocks with same size
    from the source. The buffer will then return the first block of 16 samples. On the next call to the result
    method, the buffer will be filled again and returns the next block of 16 samples.

    In some cases, we might want to draw a different number of samples from the source than we want to return.
    This can be achieved by setting the `source_num` trait of the buffer. A special case is the return of a variable
    number of samples. This is the case, for example, in the class :class:`~acoular.tbeamform.BeamformerTimeTraj`,
    in which a different number of time samples is required from the buffer for further delay-and-sum processing
    depending on the expected delay, which can be vary for moving sources. At the same time, however, only 'num'
    samples should be written to and removed from the buffer. This behavior can be achieved by setting the
    `shift_index_by` trait to 'num' and by setting the `result_num` trait to the number of samples that should be
    returned by the result function.

    >>> buffer = ac.tools.SamplesBuffer(source=source, length=32, result_num=20, shift_index_by='num')
    >>> block_sizes = []
    >>> block_sizes.append(
    ...     next(buffer.result(num=16)).shape[0]
    ... )  # this time, the buffer will return 20 samples, but the buffer will only forget the first 16 samples
    >>> buffer.result_num = 24
    >>> block_sizes.append(
    ...     next(buffer.result(num=16)).shape[0]
    ... )  # this time, the buffer will return 24 samples, but the buffer will only forget the first 16 samples
    >>> np.testing.assert_array_equal(block_sizes, [20, 24])
    """

    #: number of samples that fit in the buffer
    length = Int(desc='number of samples that fit in the buffer')

    #: number of samples per block to obtain from the source. If 'None', use 'num' argument of result method
    source_num = Union(
        None,
        Int(),
        default_value=None,
        desc='number of samples to return from the source. If "None", use "num" argument of result method',
    )

    #: number of samples to return from the buffer. If 'None', use 'num' argument of result method
    result_num = Union(
        None,
        Int(),
        default_value=None,
        desc="number of samples to return from the buffer. If 'None', use 'num' argument of result method",
    )

    #: index shift value for the buffer. If "result_num", buffer will return and forget 'result_num' samples.
    #: If "num", buffer will return 'result_num' samples but will forget 'num' samples
    shift_index_by = Enum(
        ('result_num', 'num'),
        desc=(
            'index shift value for the buffer. If "result_num", use "result_num" trait.'
            ' If "num", use "num" argument of result method'
        ),
    )

    #: current filling level of buffer
    level = Property(desc='current filling level of buffer')

    #: data type of the buffer elements
    dtype = Any(desc='data type of the buffer')

    # flag to indicate that the source is empty, for internal use
    _empty_source = Bool(False, desc='flag to indicate that the source is empty')

    # the buffer for processing
    _buffer = Array(shape=(None, None), desc='buffer for block processing')

    # current index in buffer
    _index = Int(desc='current index in buffer')

    def _get_level(self):
        return self._buffer.shape[0] - self._index

    def _create_new_buffer(self):
        self._buffer = np.zeros((self.length, self.numchannels), dtype=self.dtype)
        self._index = self.length
        self._empty_source = False

    def _write_to_buffer(self, data):
        ns = data.shape[0]
        self._buffer[0 : (self.length - ns)] = self._buffer[-(self.length - ns) :]
        self._buffer[-ns:, :] = data.astype(self.dtype)
        self._index -= ns

    def increase_buffer(self, num):
        """Increase the buffer by 'num' samples.

        Returns
        -------
        None
        """
        ar = np.zeros((num, self.numchannels), dtype=self._buffer.dtype)
        self._buffer = np.concatenate((ar, self._buffer), axis=0)
        self._index += num
        self.length += num

    def read_from_buffer(self, num):
        """Read samples from the buffer.

        Parameters
        ----------
        num : int
            number of samples to read from the buffer.

        Returns
        -------
        numpy.ndarray
            block of samples from the buffer

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
        """Fill the buffer with samples from the source.

        Parameters
        ----------
        snum : int
            number of samples to return from the source.

        Yields
        ------
        None
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
        """Return blocks of samples from the buffer.

        Parameters
        ----------
        num : int
            number of samples to return.

        Yields
        ------
        numpy.ndarray
            block of samples from the buffer
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
