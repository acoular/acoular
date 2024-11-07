import acoular as ac
import numpy as np
import pytest
import tables as tb
from pytest_cases import fixture, parametrize

from tests.utils import SetupMovingSourceCase, SetupStationarySourceCase


@pytest.hookimpl()
def pytest_sessionfinish(session): # noqa ARG001
    print("\nTest session finished!")
    tb.file._open_files.close_all()

@fixture(scope='session')
def create_time_data_source():  # fixture factory
    def _create_time_data_source(numchannels, numsamples):
        rng = np.random.RandomState(1)
        return ac.TimeSamples(sample_freq=51200, data=rng.randn(numsamples, numchannels))
    return _create_time_data_source

@fixture(scope='session')
@parametrize('numchannels', [1, 2], ids=['1ch', '2ch'])
def time_data_source(create_time_data_source, numchannels):
    return create_time_data_source(numchannels=numchannels, numsamples=50)


# Stationary Source Cases

@fixture(scope='session')
def create_source_case():
    def _create_source_case(blocksize=None, numsamples=None, grid=None, invalid_channels=None):
        return SetupStationarySourceCase(
            numsamples=numsamples,
            blocksize=blocksize,
            grid=grid,
            invalid_channels=invalid_channels,
        )
    return _create_source_case


@fixture(scope='session')
def regression_source_case(create_source_case):
    return create_source_case(
        grid=ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.07),
        numsamples=128 * 54 * 2,
        blocksize=128,
        invalid_channels=[1, 7],
    )

@fixture(scope='session')
def small_source_case(create_source_case):
    return create_source_case(
        numsamples=128,
        blocksize=128,
        invalid_channels=[i for i in range(56) if i % 2 != 0],
        grid=ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.2),
    )

# moving source cases

@pytest.fixture(scope='session')
def moving_source_case():
    return SetupMovingSourceCase()
