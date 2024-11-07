# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Implements snapshot testing of frequency beamformers."""

import tempfile

import acoular as ac
import numpy as np
import pytest
from acoular.h5files import H5CacheFileH5py, H5CacheFileTables
from pytest_cases import fixture, parametrize, parametrize_with_cases

TEST_PARAMS_F_NUM = [
    pytest.param(8000, 1, id='8kHz_oct')
    ]

def expected_results_no_cache(h5library):
    cache_file_type = H5CacheFileTables if h5library == 'pytables' else H5CacheFileH5py
    # (caching_flag, cached): expected h5f
    return {
        ('none', True): None,
        ('none', False): None,
        ('individual', True): cache_file_type,
        ('individual', False): None,
        ('all', True): cache_file_type,
        ('all', False): cache_file_type,
        ('readonly', True): None,
        ('readonly', False): None,
        ('overwrite', True): cache_file_type,
        ('overwrite', False): cache_file_type
    }


@fixture
@pytest.mark.parametrize('h5library', ['h5py', 'pytables'])
@pytest.mark.parametrize('caching_flag', [
    'none', 'individual', 'all', 'readonly', 'overwrite'])
def file_cache_options(h5library, caching_flag):
    ac.config.cache_dir = tempfile.mkdtemp()
    ac.config.h5library = h5library
    ac.config.global_caching = caching_flag
    return ac.config.h5library, ac.config.global_caching

@parametrize('cached', [False, True], scope='function')
def case_BeamformerBase_cached(source_case, cached):
    return ac.BeamformerBase(cached=cached,
        freq_data=source_case.freq_data, steer=source_case.steer)

@parametrize_with_cases("beamformer", cases=case_BeamformerBase_cached)
def test_beamformer_no_filecache(mocker, beamformer, file_cache_options):

    # Mock the global configuration attributes in Acoular
    #mocker.patch('acoular.config.h5library', h5library)
    # mocker.patch('acoular.config.global_caching', caching_flag)
    # assert ac.config.h5library == h5library

    h5library, caching_flag = file_cache_options
    expected_result = expected_results_no_cache(h5library)
    assert beamformer.h5f is None
    beamformer.synthetic(8000, 0)
    exp_res = expected_result[(caching_flag, beamformer.cached)]
    if exp_res is None:
        assert beamformer.h5f is None
    else:
        assert beamformer.h5f is not None
        assert type(beamformer.h5f) == exp_res
        #beamformer.h5f.close()

def test_time_cache(file_cache_options, time_data_source):
    tc = ac.Cache(source=time_data_source)
    h5library, caching_flag = file_cache_options
    expected_result = expected_results_no_cache(h5library)
    exp_res = expected_result[(caching_flag, True)]

    block_size = 32
    # unfinished evaluation
    block_c = next(tc.result(block_size))
    block_nc = next(time_data_source.result(block_size))
    np.testing.assert_array_almost_equal(block_c, block_nc)

    # finished evaluation
    block_c = ac.tools.return_result(tc, num=block_size)
    block_nc = ac.tools.return_result(time_data_source, num=block_size)
    np.testing.assert_array_almost_equal(block_c, block_nc)

    # assert file
    if exp_res is None:
        assert tc.h5f is None
    else:
        assert tc.h5f is not None
        assert type(tc.h5f) == exp_res
