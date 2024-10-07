import os
from pathlib import Path

import acoular as ac
import numpy as np
import pytest

src = ac.TimeSamples(data=np.random.random((10, 2)))


def remove_cache():
    cache_file = Path(ac.config.cache_dir).absolute() / '_cache.h5'
    if cache_file.exists():
        os.remove(cache_file)


class TestCache:
    @pytest.mark.parametrize('lib, sample_freq', [('pytables', 1), ('h5py', 2)])
    def test_cache_time(self, lib, sample_freq):
        remove_cache()
        ac.config.h5library = lib
        src.sample_freq = sample_freq
        tc = ac.Cache(source=src)
        ac.tools.return_result(tc, num=1)  # here, the result is cached
        cached_result = ac.tools.return_result(tc, num=1)  # here, the result is read from cache
        nocache_result = ac.tools.return_result(src, num=1)
        assert np.allclose(nocache_result, cached_result)

    @pytest.mark.parametrize('lib, sample_freq', [('pytables', 3), ('h5py', 4)])
    def test_cache_spectra(self, lib, sample_freq):
        remove_cache()
        ac.config.h5library = lib
        src.sample_freq = sample_freq
        fft = ac.RFFT(source=src, block_size=2)
        tc = ac.Cache(source=fft)
        ac.tools.return_result(tc, num=1)  # here, the result is cached
        cached_result = ac.tools.return_result(tc, num=1)  # here, the result is read from cache
        nocache_result = ac.tools.return_result(fft, num=1)
        assert np.allclose(nocache_result, cached_result)
