import acoular as ac
import numpy as np
import pytest


@pytest.mark.parametrize('numchannels', [1, 2], ids=['1ch', '2ch'])
@pytest.mark.parametrize('num', [1, 64, 127, 128, 129, 256])
def test_num_args(num, create_time_data_source, numchannels):
    buffer_size = 512
    time_data_source = create_time_data_source(numchannels=numchannels, numsamples=buffer_size + 10)
    for snum in [128, None]:
        buffer = ac.tools.utils.SamplesBuffer(source=time_data_source, length=buffer_size, source_num=snum)
        data = ac.tools.return_result(time_data_source, num=num)
        data_comp = ac.tools.return_result(buffer, num=num)
        np.testing.assert_array_equal(data, data_comp)

@pytest.mark.parametrize('numchannels', [1, 2], ids=['1ch', '2ch'])
def test_increase_buffer(create_time_data_source, numchannels):
    buffer_size = 50
    time_data_source = create_time_data_source(numchannels=numchannels, numsamples=512 + 10)
    data = ac.tools.return_result(time_data_source, num=256)
    # test buffer
    buffer = ac.tools.utils.SamplesBuffer(source=time_data_source, length=buffer_size, source_num=25)
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
