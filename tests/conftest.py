import shutil
import tempfile
from pathlib import Path

import acoular as ac
import pytest

from tests.setup import SetupSourceCase


@pytest.fixture(scope='module')
def use_temporary_cache_dir():
    ac.config.cache_dir = tempfile.mkdtemp()
    print(f'Use temporary cache directory: {ac.config.cache_dir}')
    return Path(ac.config.cache_dir)


@pytest.fixture(scope='module', autouse=True)
def clear_cache(use_temporary_cache_dir):
    cache_dir = use_temporary_cache_dir
    yield
    # Remove the cache directory again after the tests
    print(f'Remove temporary cache directory: {ac.config.cache_dir}')
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)


@pytest.fixture
def source_case():
    ac.config.global_caching = 'none'  # to make sure that nothing is cached
    return SetupSourceCase()
