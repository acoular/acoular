import numpy as np
import pytest
import sounddevice as sd


@pytest.fixture
def mock_have_no_lib(mocker):
    """Mock sounddevice module to raise ImportError."""
    mocker.patch('importlib.util.find_spec').return_value = None


@pytest.fixture
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


@pytest.fixture
def sounddevice_properties(setup_sounddevice_mocks):
    """Retrieve default device properties."""
    default_device = sd.query_devices(kind='input')
    num = setup_sounddevice_mocks
    return default_device['default_samplerate'], default_device['max_input_channels'], default_device['index'], num
