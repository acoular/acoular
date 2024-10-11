import acoular as ac
import pytest


def test_default_sounddevice_properties(sounddevice_properties):
    """Test that the default samplerate and input channels from sounddevice are
    correctly transferred to ac.SoundDeviceSamplesGenerator class."""

    default_samplerate, max_input_channels, device_index, num = sounddevice_properties
    sdev = ac.SoundDeviceSamplesGenerator(device=device_index)
    assert sdev.sample_freq == default_samplerate
    assert sdev.device_properties()['default_samplerate'] == default_samplerate
    assert sdev.device_properties()['max_input_channels'] == max_input_channels


def test_result(sounddevice_properties):
    """Test that one signal block of samples is collected with correct shape."""
    _, max_input_channels, device_index, num = sounddevice_properties
    sdev = ac.SoundDeviceSamplesGenerator(device=device_index, numchannels=min(2, max_input_channels))
    sdev.numchannels = min(2, max_input_channels)  # Ensure numchannels is <= max_input_channels
    block = next(sdev.result(num))
    assert sdev.running
    assert block.shape == (num, sdev.numchannels), f'Expected shape {(num, sdev.numchannels)}, but got {block.shape}'


def test_import_error(mock_have_no_lib):  # noqa ARG001 (uses fixture from conftest.py)
    """Test that ac.SoundDeviceSamplesGenerator raises ImportError if sounddevice is not available."""
    del ac.config.__dict__['_traits_cache_have_sounddevice']  # remove cached property
    with pytest.raises(ImportError):
        ac.SoundDeviceSamplesGenerator()
