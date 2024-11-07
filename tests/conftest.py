from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pytest
from pytest_cases import fixture, parametrize

from tests.utils import SetupMovingSourceCase, SetupSourceCase


@pytest.fixture(scope='session')
def test_config():
    @dataclass
    class Config:
        testdir: Path = Path(__file__).parent
        moduledir: Path = Path(__file__).parent.parent
        reference_data: Path = Path(__file__).parent / 'reference_data'
    return Config()

@pytest.fixture(scope='session')
def source_case(test_config):
    blocksize = 128
    numsamples = blocksize * 54 * 2 # blocksize * number of microphones * 2
    return SetupSourceCase(
        test_config=test_config, numsamples=numsamples, blocksize=blocksize)

@pytest.fixture(scope='session')
def moving_source_case():
    return SetupMovingSourceCase()

@fixture
@parametrize('numchannels', [1, 2], ids=['1ch', '2ch'])
def time_data_source(moving_source_case, numchannels):
    time_data = deepcopy(moving_source_case.source)
    time_data.stop = 50
    time_data.invalid_channels = [i for i in range(time_data.numchannels) if i >= numchannels]
    return time_data


# @pytest.fixture(scope='session', autouse=True)
# def clear_cache(use_temporary_cache_dir):  # TODO: can this be done with built-in tmp_path fixture?
#     cache_dir = use_temporary_cache_dir
#     yield
#     # Remove the cache directory again after the tests
#     print(f'Remove temporary cache directory: {ac.config.cache_dir}')
#     if cache_dir.exists() and cache_dir.is_dir():
#         shutil.rmtree(cache_dir)

# @pytest.fixture(scope='session')
# def use_temporary_cache_dir():  # TODO: use built-in tmp_path fixture
#     ac.config.cache_dir = tempfile.mkdtemp()
#     print(f'Use temporary cache directory: {ac.config.cache_dir}')
#     return Path(ac.config.cache_dir)

