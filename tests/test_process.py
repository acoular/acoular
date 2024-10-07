import os
from pathlib import Path

import acoular as ac
import numpy as np
import pytest

CACHE_DIR = Path(ac.config.cache_dir).absolute()


class TestCache:
    """Test the block-wise caching mechanism.

    To allow concurrent testing without writing interference, the cache_dir is set to a different directories.
    """

    def _set_h5lib_and_cache_dir(self, lib, cache_dir):
        ac.config.h5library = lib
        ac.config.cache_dir = cache_dir
        # remove cache file if it already exists to avoid interference between tests
        cache_file = Path(ac.config.cache_dir) / '_cache.h5'
        if cache_file.exists():
            os.remove(cache_file)

    def create_source_time(self, sample_freq):
        rng = np.random.default_rng(1)
        ts = ac.TimeSamples(data=rng.random((10, 2)), sample_freq=sample_freq)
        return ac.Cache(source=ts)

    def create_source_spectra(self, sample_freq, **kwargs):
        rng = np.random.default_rng(1)
        ts = ac.TimeSamples(data=rng.random((10, 2)), sample_freq=sample_freq)
        return ac.Cache(source=ac.RFFT(source=ts, **kwargs))

    @pytest.mark.parametrize('lib, sample_freq', [('pytables', 1), ('h5py', 2)])
    def test_cache_time(self, lib, sample_freq):
        self._set_h5lib_and_cache_dir(lib, str(CACHE_DIR / (lib + str(sample_freq))))
        tc = self.create_source_time(sample_freq)
        ac.tools.return_result(tc, num=1)  # here, the result is cached
        cached_result = ac.tools.return_result(tc, num=1)  # here, the result is read from cache with the same object
        nocache_result = ac.tools.return_result(tc.source, num=1)
        assert np.allclose(nocache_result, cached_result)
        # with new tc object
        tc2 = self.create_source_time(sample_freq)
        assert np.allclose(nocache_result, ac.tools.return_result(tc2, num=1))

    @pytest.mark.parametrize('lib, sample_freq', [('pytables', 3), ('h5py', 4)])
    def test_cache_spectra(self, lib, sample_freq):
        self._set_h5lib_and_cache_dir(lib, str(CACHE_DIR / (lib + str(sample_freq))))
        tc = self.create_source_spectra(sample_freq, block_size=2)
        ac.tools.return_result(tc, num=1)  # here, the result is cached
        cached_result = ac.tools.return_result(tc, num=1)  # here, the result is read from cache with the same object
        nocache_result = ac.tools.return_result(tc.source, num=1)
        assert np.allclose(nocache_result, cached_result)
        # with new tc object
        tc2 = self.create_source_spectra(sample_freq, block_size=2)
        assert np.allclose(nocache_result, ac.tools.return_result(tc2, num=1))
