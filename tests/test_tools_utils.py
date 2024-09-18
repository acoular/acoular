import acoular as ac
import numpy as np
import pytest


class TestSamplesBuffer:
    def create_time_data(self, numsamples, sample_freq=64):
        return ac.TimeSamples(
            data=ac.WNoiseGenerator(
                sample_freq=sample_freq,
                numsamples=numsamples,
            ).signal()[:, np.newaxis],
            sample_freq=sample_freq,
        )

    @pytest.mark.parametrize('num', [1, 64, 127, 128, 129, 256])
    def test_num_args(self, num):
        buffer_size = 512
        source = self.create_time_data(numsamples=512 + 10)
        for snum in [128, None]:
            buffer = ac.tools.utils.SamplesBuffer(source=source, length=buffer_size, source_num=snum)
            data = ac.tools.return_result(source, num=num)
            data_comp = ac.tools.return_result(buffer, num=num)
            np.testing.assert_array_equal(data, data_comp)

    def test_increase_buffer(self):
        buffer_size = 50
        source = self.create_time_data(numsamples=512 + 10)
        data = ac.tools.return_result(source, num=256)
        # test buffer
        buffer = ac.tools.utils.SamplesBuffer(source=source, length=buffer_size, source_num=25)
        gen = buffer.result(num=25)
        collected_data = np.zeros_like(data)
        i = 0
        for block in gen:
            end = block.shape[0]
            collected_data[i : i + end] = block
            i += block.shape[0]
            # here we increase the num returned by the buffer
            buffer.result_num = 51  # greater than buffer size
        np.testing.assert_array_equal(data, collected_data)
