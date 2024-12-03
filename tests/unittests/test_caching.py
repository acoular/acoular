# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Integration tests for all objects that can use Acoular's HDF5 filecache functionality."""

import re
from pathlib import Path

import acoular as ac
import numpy as np
import pytest
from acoular.h5files import H5CacheFileH5py, H5CacheFileTables
from pytest_cases import fixture, parametrize_with_cases

from tests.cases.test_caching_cases import Caching


def delete_traits_cache(obj):
    """Delete traits cache from object."""
    for key in list(obj.__dict__.keys()):
        if key.startswith('_traits_cache'):
            obj.__dict__.pop(key)


def expected_results(h5library):
    """
    Expected results for test `test_filecache_created` when no cache file exists before the test.

    Parameters
    ----------
    h5library : str
        HDF5 library to use for cache files ('pytables' or 'h5py').

    Returns
    -------
    dict
        Expected results when no cache file exists. Keys are tuples of caching_flag and cached.
        Values are the expected type of the cache file.
    """
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
        ('overwrite', False): cache_file_type,
    }


@fixture
@pytest.mark.parametrize('h5library', ['pytables', 'h5py'])
@pytest.mark.parametrize('caching_flag', ['none', 'individual', 'all', 'readonly', 'overwrite'])
def file_cache_options(tmp_path, h5library, caching_flag):
    """Fixture to set up file cache options.

    Parameters
    ----------
    tmp_path : pytest.fixture
        Temporary directory that is removed after the test (built-in pytest fixture).
    h5library : str
        HDF5 library to use for cache files ('pytables' or 'h5py').
    caching_flag : str
        Cache flag to use ('none', 'individual', 'all', 'readonly', 'overwrite').
    """
    ac.config.cache_dir = str(tmp_path)
    ac.config.h5library = h5library
    ac.config.global_caching = caching_flag
    return ac.config.h5library, ac.config.global_caching


@parametrize_with_cases('case', cases=Caching)
def test_filecache_created(case, file_cache_options):
    """Test if cache file is created when it should be.

    Parameters
    ----------
    case : pytest.fixture (tuple)
        Test case with Acoular object, calculation function, and cached attribute value.
    file_cache_options : pytest.fixture (tuple)
        File cache options with HDF5 library and caching
    """
    h5library, caching_flag = file_cache_options
    expected_result = expected_results(h5library)
    objs, calc, cached = case

    obj1, obj2, obj3 = objs

    expected_error = (
        isinstance(obj1, ac.PointSpreadFunction)
        and caching_flag not in ['readonly', 'none']
        and obj1.calcmode == 'readonly'
    )

    # run calculation
    if expected_error:
        # ValueError is expected if PSF calcmode is 'readonly' and no cache file is present
        with pytest.raises(ValueError, match=re.escape("Cannot calculate missing PSF (points) in 'readonly' mode.")):
            result = calc(obj1)
    else:
        result = calc(obj1)
        if caching_flag == 'overwrite':
            result = result[...]  # origiinal array will be overwritten (and closed)

    # check if cache file is created
    exp_res = expected_result[(caching_flag, cached)]
    if exp_res is None:
        assert obj1.h5f is None
    else:
        assert obj1.h5f is not None
        assert type(obj1.h5f) is exp_res
        assert Path(obj1.h5f.filename).exists()

    # create copy and check if it loads the cache file
    if not expected_error and exp_res is not None:
        calc(obj2)
        assert Path(obj2.h5f.filename).samefile(Path(obj1.h5f.filename)), 'Not the same cache file loaded.'

    # disable cache and check if result matches uncached result
    if not expected_error and not isinstance(obj1, ac.Cache) and exp_res is not None:
        ac.config.global_caching = 'none'
        result_uncache = calc(obj2)
        np.testing.assert_allclose(result, result_uncache)
