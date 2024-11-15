import acoular as ac
import numpy as np
import pytest
import sounddevice as sd
import tables as tb
from pytest_cases import fixture, parametrize

from tests.utils import SetupMovingSourceCase, SetupStationarySourceCase


@pytest.hookimpl()
def pytest_sessionfinish(session):  # noqa ARG001
    """Close all open files after the test session.

    This hook is called after the test session is finished and is used to get rid of the
    annoying UnclosedFile warnings from pytables that cannot be suppressed by filterwarnings.
    """
    tb.file._open_files.close_all()


@fixture(scope='session')
def create_time_data_source():
    """Fixture factory for creating time data sources of certain length and number of channels."""

    def _create_time_data_source(numchannels, numsamples):
        rng = np.random.RandomState(1)
        return ac.TimeSamples(sample_freq=51200, data=rng.randn(numsamples, numchannels))

    return _create_time_data_source


@fixture(scope='session')
@parametrize('numchannels', [1, 2], ids=['1ch', '2ch'])
def time_data_source(create_time_data_source, numchannels):
    """Default time data source fixture for testing generator classes."""
    return create_time_data_source(numchannels=numchannels, numsamples=50)


# Stationary Source Cases


@fixture(scope='session')
def create_source_case():
    """Fixture factory for creating spatially stationary source cases with certain parameters."""

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
    """Fixture for snapshot testing of beamformer classes."""
    return create_source_case(
        grid=ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.07),
        numsamples=128 * 54 * 2,
        blocksize=128,
        invalid_channels=[1, 7],
    )


@fixture(scope='session')
def small_source_case(create_source_case):
    """Fixture for faster testing of the caching (uses fewer microphones and coarser grid)."""
    return create_source_case(
        numsamples=128,
        blocksize=128,
        invalid_channels=[1, 7] + [i for i in range(56) if i % 3 == 0],  # use every third microphone
        grid=ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.2),
    )


# moving source cases


@pytest.fixture(scope='session')
def moving_source_case():
    return SetupMovingSourceCase()


@pytest.fixture()
def _mock_have_no_lib(mocker):
    """Mock behaviour: library does not exist."""
    mocker.patch('importlib.util.find_spec').return_value = None


@pytest.fixture()
def setup_sounddevice_mocks(mocker):
    """Helper function to set up necessary mocks for tests."""
    num = 128
    mocker.patch('sounddevice.query_devices').return_value = {
        'default_samplerate': 44100,
        'max_input_channels': 2,
        'index': 0,
    }
    mock_stream_instance = mocker.patch('sounddevice.InputStream').return_value.__enter__.return_value
    mock_stream_instance.read.return_value = (np.zeros((num, 2)), False)  # (data, overflow)
    return num


@pytest.fixture()
def sounddevice_properties(setup_sounddevice_mocks):
    """Retrieve default device properties."""
    default_device = sd.query_devices(kind='input')
    num = setup_sounddevice_mocks
    return default_device['default_samplerate'], default_device['max_input_channels'], default_device['index'], num
