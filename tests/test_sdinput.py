import numpy as np
import sounddevice as sd
from acoular import SoundDeviceSamplesGenerator


def get_default_device_properties():
    """Retrieve default device properties."""
    default_device = sd.query_devices(kind='input')
    return default_device['default_samplerate'], default_device['max_input_channels'], default_device['index']


def setup_mocks(mocker):
    """Helper function to set up necessary mocks for tests."""
    mock_query_devices = mocker.patch('sounddevice.query_devices')
    mock_query_devices.return_value = {'default_samplerate': 44100, 'max_input_channels': 2, 'index': 0}
    mock_input_stream = mocker.patch('sounddevice.InputStream')
    return mock_input_stream.return_value.__enter__.return_value


def test_default_device_properties(mocker):
    """Test that the default samplerate and input channels from sounddevice are
    correctly transferred to SoundDeviceSamplesGenerator class."""

    setup_mocks(mocker)
    default_samplerate, max_input_channels, device_index = get_default_device_properties()
    sdev = SoundDeviceSamplesGenerator(device=device_index)
    assert sdev.sample_freq == default_samplerate
    assert sdev.device_properties()['default_samplerate'] == default_samplerate
    assert sdev.device_properties()['max_input_channels'] == max_input_channels


def test_result(mocker):
    """Test that one signal block of samples is collected with correct shape."""
    num = 128

    mock_stream_instance = setup_mocks(mocker)
    mock_stream_instance.read.return_value = (np.zeros((num, 2)), False)  # (data, overflow)

    _, max_input_channels, device_index = get_default_device_properties()
    sdev = SoundDeviceSamplesGenerator(device=device_index, numchannels=min(2, max_input_channels))
    sdev.numchannels = min(2, max_input_channels)  # Ensure numchannels is <= max_input_channels

    block = next(sdev.result(num))
    assert block.shape == (num, sdev.numchannels), f'Expected shape {(num, sdev.numchannels)}, but got {block.shape}'
