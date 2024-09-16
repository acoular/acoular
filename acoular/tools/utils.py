"""Utility classes intended for internal use in Acoular."""

import numpy as np
from traits.api import CArray, HasPrivateTraits, Instance, Int, Property

from acoular import SamplesGenerator


class SamplesBuffer(HasPrivateTraits):
    
    source = Instance(SamplesGenerator)

    buffer = CArray(dtype=float, shape=(None, None), value=np.array([]), desc='buffer for block processing')

    index = Int(desc='current index in buffer')

    numsamples = Property(desc='size of the buffer in samples')

    numchannels = Property(desc='number of channels in the buffer')

    def create_new_buffer(self, numsamples, numchannels, dtype=float):
        self.buffer = np.zeros((numsamples, numchannels), dtype=dtype)
        self.index = self.numsamples

    def increase_buffer(self, num):
        ar = np.zeros((num, self.numchannels), dtype=self.buffer.dtype)
        self.buffer = np.concatenate((ar, self.buffer), axis=0)
        self.bufferIndex += num

    def fill_buffer(self, block):
        ns = block.shape[0]
        self.buffer[0 : (self.size - ns)] = self.buffer[-(self.size - ns) :]
        self.buffer[-ns:, :] = block
        self.bufferIndex -= ns

    def get_from_buffer(self, num):
        self.bufferIndex += num
        return self.buffer[self.index - num : self.index]

    def _get_numsamples(self):
        return self.buffer.shape[0]

    def _get_numchannels(self):
        return self.buffer.shape[1]
