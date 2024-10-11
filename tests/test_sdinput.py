import sounddevice as sd
from acoular import SoundDeviceSamplesGenerator


def get_default_device_properties():
    """Retrieve default device properties."""
    default_device = sd.query_devices(kind='input')
    return default_device['default_samplerate'], default_device['max_input_channels'], default_device['index']


def test_default_device_properties():
    """Test that the default samplerate and input channels from sounddevice are
    correctly transferred to SoundDeviceSamplesGenerator class."""

    # Get the default device properties
    default_samplerate, max_input_channels, device_index = get_default_device_properties()

    # Initialize the SoundDeviceSamplesGenerator
    sdev = SoundDeviceSamplesGenerator(device=device_index)

    # Assert that the values are transferred correctly
    assert sdev.sample_freq == default_samplerate
    assert sdev.device_properties()['default_samplerate'] == default_samplerate
    assert sdev.device_properties()['max_input_channels'] == max_input_channels

def test_collect():
    """Test that the default samplerate, input channels are transferred and that one block of samples is collected with correct shape."""
    num = 128  # Number of samples per block
    _, max_input_channels, device_index = get_default_device_properties()

    sdev = SoundDeviceSamplesGenerator(device=device_index, numchannels=min(2, max_input_channels))
    sdev.numchannels = min(2, max_input_channels)  # Ensure numchannels is <= max_input_channels
    block = next(sdev.result(num))
    assert block.shape == (num, sdev.numchannels), f'Expected shape {(num, sdev.numchannels)}, but got {block.shape}'
