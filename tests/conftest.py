import shutil
import tempfile
from pathlib import Path

import acoular as ac
import pytest

some_object = []

@pytest.fixture
def add_element():
    some_object.append(len(some_object)+1)
    return some_object

@pytest.fixture(scope='session')
def test_config():
    class Config:
        def __init__(self):
            self.testdir = Path(__file__).parent
            self.moduledir = Path(__file__).parent.parent
            self.reference_data = self.testdir / 'reference_data'

    return Config()


@pytest.fixture(scope='session', autouse=True)
def clear_cache(use_temporary_cache_dir):  # TODO: can this be done with built-in tmp_path fixture?
    cache_dir = use_temporary_cache_dir
    yield
    # Remove the cache directory again after the tests
    print(f'Remove temporary cache directory: {ac.config.cache_dir}')
    if cache_dir.exists() and cache_dir.is_dir():
        shutil.rmtree(cache_dir)


@pytest.fixture(scope='session')
def use_temporary_cache_dir():  # TODO: use built-in tmp_path fixture
    ac.config.cache_dir = tempfile.mkdtemp()
    print(f'Use temporary cache directory: {ac.config.cache_dir}')
    return Path(ac.config.cache_dir)


class SetupSourceCase:
    def __init__(self, test_config, numsamples, blocksize=128):
        self.calib = ac.Calib(from_file=test_config.moduledir / 'examples' / 'data' / 'example_calib.xml')
        self.source = ac.MaskedTimeSamples(
            name=test_config.moduledir / 'examples' / 'data' / 'example_data.h5',
            invalid_channels=[1, 7],
            start=0,
            stop=numsamples,
            calib=self.calib,
        )
        self.mics = ac.MicGeom(
            from_file=test_config.moduledir / 'acoular' / 'xml' / 'array_56.xml', invalid_channels=[1, 7]
        )
        self.grid = ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.07)
        self.env = ac.Environment(c=346.04)
        self.steer = ac.SteeringVector(grid=self.grid, mics=self.mics, env=self.env)
        self.freq_data = ac.PowerSpectra(
            source=self.source,
            window='Hanning',
            overlap='50%',
            block_size=blocksize,
            cached=False,
        )

@pytest.fixture
def source_case(test_config):
    ac.config.global_caching = 'none'  # to make sure that nothing is cached
    blocksize = 128
    numsamples = blocksize * 54 * 2 # blocksize * number of microphones * 2
    return SetupSourceCase(
        test_config=test_config, numsamples=numsamples, blocksize=blocksize)

@pytest.fixture
def source_case_short(test_config):
    ac.config.global_caching = 'none'
    return SetupSourceCase(test_config=test_config, numsamples=50)
