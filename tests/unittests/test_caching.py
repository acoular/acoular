# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Integration tests for all objects that can use Acoular's HDF5 filecache functionality."""

import re
from pathlib import Path

import acoular as ac
from acoular.h5cache import H5cache
from acoular.h5files import H5CacheFileH5py, H5CacheFileTables

from tests.cases.test_caching_cases import Caching

import numpy as np
import pytest
from pytest_cases import fixture, parametrize_with_cases


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
def file_cache_options(tmp_path, monkeypatch, h5library, caching_flag):
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
    monkeypatch.setattr(ac.config, 'cache_dir', str(tmp_path))
    monkeypatch.setattr(ac.config, 'h5library', h5library)
    monkeypatch.setattr(ac.config, 'global_caching', caching_flag)
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
        and obj1.calcmode == 'readonly'
        and (caching_flag in ['overwrite', 'all'] or (caching_flag == 'individual' and obj1.cached))
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
        assert obj1._h5f is None
    else:
        assert obj1._h5f is not None
        assert type(obj1._h5f) is exp_res
        assert Path(obj1._h5f.filename).exists()

    # create copy and check if it loads the cache file
    if not expected_error and exp_res is not None:
        calc(obj2)
        assert Path(obj2._h5f.filename).samefile(Path(obj1._h5f.filename)), 'Not the same cache file loaded.'

    # disable cache and check if result matches uncached result
    if not expected_error and not isinstance(obj1, ac.Cache) and exp_res is not None:
        ac.config.global_caching = 'none'
        result_uncache = calc(obj2)
        np.testing.assert_allclose(result, result_uncache)


@pytest.mark.parametrize('h5library', ['pytables', 'h5py'])
@pytest.mark.parametrize('mode', ['none', 'individual', 'all', 'readonly', 'overwrite'])
@pytest.mark.parametrize('source_type', ['average', 'rfft'])
def test_cache_empty_source(tmp_path, monkeypatch, h5library, mode, source_type):
    monkeypatch.setattr(ac.config, 'cache_dir', str(tmp_path))
    monkeypatch.setattr(ac.config, 'h5library', h5library)
    monkeypatch.setattr(ac.config, 'global_caching', mode)
    source = ac.TimeSamples(data=np.ones((1, 1)), sample_freq=48000)
    if source_type == 'average':
        source = ac.Average(source=source, num_per_average=2)
    else:
        source = ac.RFFT(source=source, block_size=128)
    cache = ac.Cache(source=source)

    try:
        assert list(source.result(1)) == []
        for _ in range(2):
            assert list(cache.result(1)) == []
            if cache._h5f is not None:
                assert not cache._h5f.is_cached('tc_' + cache.digest)
    finally:
        if cache._h5f is not None:
            cache._h5f.close()


@pytest.mark.parametrize('h5library', ['pytables', 'h5py'])
@pytest.mark.parametrize('complete', [None, False])
def test_cache_empty_source_incomplete_cache(tmp_path, monkeypatch, h5library, complete):
    monkeypatch.setattr(ac.config, 'cache_dir', str(tmp_path))
    monkeypatch.setattr(ac.config, 'h5library', h5library)
    monkeypatch.setattr(ac.config, 'global_caching', 'all')
    source = ac.TimeSamples(data=np.ones((1, 1)), sample_freq=48000)
    cache = ac.Cache(source=ac.Average(source=source, num_per_average=2))
    H5cache.get_cache_file(cache, cache.basename)
    nodename = 'tc_' + cache.digest

    try:
        cache._h5f.create_extendable_array(nodename, (0, 1), 'float64')
        if complete is not None:
            node = cache._h5f.get_data_by_reference(nodename)
            cache._h5f.set_node_attribute(node, 'complete', complete)
        for _ in range(2):
            assert list(cache.result(1)) == []
            assert not cache._h5f.is_cached(nodename)
    finally:
        cache._h5f.close()


@pytest.mark.parametrize(
    ('trait', 'method'), [('csm', 'calc_csm'), ('eigvals', '_calc_eigvals'), ('eigvecs', '_calc_eigvecs')]
)
@pytest.mark.parametrize(('mode', 'cached'), [('none', False), ('none', True), ('individual', False)])
def test_spectra_disabled_cache_skips_file_preparation(monkeypatch, mocker, trait, method, mode, cached):
    monkeypatch.setattr(ac.config, 'global_caching', mode)
    obj = ac.PowerSpectra(cached=cached)
    expected = np.ones((1, 1, 1))
    compute = mocker.patch.object(ac.PowerSpectra, method, return_value=expected)
    filecache = mocker.patch('acoular.h5cache.HDF5Cache.get_cache_file')

    assert obj._get_filecache(trait) is expected
    compute.assert_called_once_with()
    filecache.assert_not_called()


@pytest.mark.parametrize('trait', ['csm', 'eigvals', 'eigvecs'])
@pytest.mark.parametrize('mode', ['none', 'individual', 'all', 'readonly', 'overwrite'])
def test_spectra_getters_delegate_cache_policy(monkeypatch, mocker, trait, mode):
    monkeypatch.setattr(ac.config, 'global_caching', mode)
    obj = ac.PowerSpectra(cached=False)
    expected = np.ones((1, 1, 1))
    handler = mocker.patch.object(ac.PowerSpectra, '_get_filecache', return_value=expected)

    assert getattr(obj, trait) is expected
    handler.assert_called_once_with(trait)


@pytest.fixture(params=['pytables', 'h5py'])
def cached_spectra(request, tmp_path, monkeypatch):
    monkeypatch.setattr(ac.config, 'cache_dir', str(tmp_path))
    monkeypatch.setattr(ac.config, 'h5library', request.param)
    monkeypatch.setattr(ac.config, 'global_caching', 'all')
    source = ac.TimeSamples(data=np.random.default_rng(1).normal(size=(256, 2)), sample_freq=48000)
    return ac.PowerSpectra(source=source, block_size=128, cached=False)


@pytest.mark.parametrize('trait', ['csm', 'eigvals', 'eigvecs'])
def test_spectra_readonly_reuses_cached_result(cached_spectra, monkeypatch, mocker, trait):
    expected = getattr(cached_spectra, trait)[:].copy()
    monkeypatch.setattr(ac.config, 'global_caching', 'readonly')
    for method in ('calc_csm', '_calc_eigvals', '_calc_eigvecs'):
        mocker.patch.object(ac.PowerSpectra, method, side_effect=AssertionError('unexpected computation on cache hit'))
    obj = ac.PowerSpectra(source=cached_spectra.source, block_size=128, cached=False)

    np.testing.assert_array_equal(getattr(obj, trait)[:], expected)


@pytest.mark.parametrize('trait', ['eigvals', 'eigvecs'])
def test_spectra_readonly_does_not_store_missing_result(cached_spectra, monkeypatch, trait):
    csm = cached_spectra.csm[:].copy()
    expected = np.linalg.eigh(csm)[0 if trait == 'eigvals' else 1]
    nodename = trait + '_' + cached_spectra.digest
    assert not cached_spectra._h5f.is_cached(nodename)
    monkeypatch.setattr(ac.config, 'global_caching', 'readonly')
    obj = ac.PowerSpectra(source=cached_spectra.source, block_size=128, cached=False)

    np.testing.assert_allclose(getattr(obj, trait), expected)
    assert not obj._h5f.is_cached(nodename)


@pytest.mark.parametrize('mode', ['none', 'individual', 'all', 'readonly', 'overwrite'])
def test_imported_spectra_bypasses_file_cache(monkeypatch, mocker, mode):
    monkeypatch.setattr(ac.config, 'global_caching', mode)
    csm = np.array([[[2.0, 0.0], [0.0, 1.0]]], dtype=complex)
    obj = ac.PowerSpectraImport(csm=csm)
    handler = mocker.patch.object(
        ac.PowerSpectra, '_get_filecache', side_effect=AssertionError('unexpected disk caching')
    )
    eigvals, eigvecs = np.linalg.eigh(csm)

    np.testing.assert_array_equal(obj.csm, csm)
    np.testing.assert_allclose(obj.eigvals, eigvals)
    np.testing.assert_allclose(obj.eigvecs, eigvecs)
    handler.assert_not_called()


@pytest.mark.parametrize('h5library', ['pytables', 'h5py'])
def test_timesamples_file_change(tmp_path, monkeypatch, h5library):
    """Test that changing the file attribute of a TimeSamples object works correctly.

    This test verifies the fix for issue #474, where changing the file attribute
    of a TimeSamples object that already had a file set would throw a KeyError.
    The bug was that _decrease_file_reference_counter expected a Path object but
    received a string when the file reference counter was being decreased for a
    cached object whose source's file changed.

    Parameters
    ----------
    tmp_path : pytest.fixture
        Temporary directory for cache files (built-in pytest fixture).
    """
    # Configure caching
    monkeypatch.setattr(ac.config, 'cache_dir', str(tmp_path))
    monkeypatch.setattr(ac.config, 'global_caching', 'all')
    monkeypatch.setattr(ac.config, 'h5library', h5library)

    # Generate some random time data
    rng = np.random.default_rng(1)
    file1, file2 = tmp_path / 'timesamples1.h5', tmp_path / 'timesamples2.h5'
    ts1 = ac.TimeSamples(data=rng.standard_normal((1024, 4)), sample_freq=51200)
    ac.WriteH5(source=ts1, file=file1).save()
    ts2 = ac.TimeSamples(data=rng.standard_normal((1024, 4)), sample_freq=51200)
    ac.WriteH5(source=ts2, file=file2).save()

    ts = ac.TimeSamples(file=file1)
    ps = ac.PowerSpectra(source=ts, cached=True, block_size=128)

    # Trigger calculation to create initial cache file
    _ = ps.csm[:]

    # Verify cache file was created
    assert ps._h5f is not None
    first_cache_file = Path(ps._h5f.filename)
    assert first_cache_file.exists()
    ts.file = file2

    # Trigger calculation again with the new file
    # This should open a new cache file and properly decrease the reference counter for the old one
    _ = ps.csm[:]

    # Verify that a new cache file was created (different basename due to different source digest)
    assert ps._h5f is not None
    second_cache_file = Path(ps._h5f.filename)
    assert second_cache_file.exists()

    # The cache files should be different because the source digest changed
    assert first_cache_file != second_cache_file, 'Cache file should have changed when source file changed'
