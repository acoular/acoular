# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Unit tests for acoular.sdinput module."""

import acoular as ac
import pytest


@pytest.fixture
def sounddevice_samples_generator(sounddevice_properties):
    """Fixture to create an instance of ac.SoundDeviceSamplesGenerator."""
    _, max_input_channels, device_index, num = sounddevice_properties
    sdev = ac.SoundDeviceSamplesGenerator(device=device_index, num_channels=min(2, max_input_channels))
    return sdev, num


def test_default_sounddevice_properties(sounddevice_properties):
    """Test that the default samplerate and input channels from sounddevice are
    correctly transferred to ac.SoundDeviceSamplesGenerator class.
    """
    default_samplerate, max_input_channels, device_index, num = sounddevice_properties
    sdev = ac.SoundDeviceSamplesGenerator(device=device_index)
    assert sdev.sample_freq == default_samplerate
    assert sdev.device_properties()['default_samplerate'] == default_samplerate
    assert sdev.device_properties()['max_input_channels'] == max_input_channels


def test_result_infinite(sounddevice_samples_generator):
    """Test that one signal block of samples is collected with correct shape."""
    sdev, num = sounddevice_samples_generator
    sdev.num_samples = -1
    for i, block in enumerate(sdev.result(num)):
        assert sdev.running
        assert block.shape == (
            num,
            sdev.num_channels,
        ), f'Expected shape {(num, sdev.num_channels)}, but got {block.shape}'
        if i > 1:
            break


def test_result_finite(sounddevice_samples_generator):
    """Test that one signal block of samples is collected with correct shape."""
    sdev, num = sounddevice_samples_generator
    sdev.num_samples = num * 2
    for block in sdev.result(num):
        assert sdev.running
        assert block.shape == (
            num,
            sdev.num_channels,
        ), f'Expected shape {(num, sdev.num_channels)}, but got {block.shape}'
    assert not sdev.running


def test_set_sample_freq(sounddevice_properties):  # noqa ARG001 (uses fixture from conftest.py)
    """Test that the sample frequency can be set.

    Parameters
    ----------
    sounddevice_properties : tuple
        Tuple with default samplerate, max_input_channels, device_index, num
    """
    sdev = ac.SoundDeviceSamplesGenerator(device=0)
    default_sample_freq = sdev.sample_freq
    new_sample_freq = default_sample_freq * 2
    sdev.sample_freq = new_sample_freq
    assert sdev.sample_freq == new_sample_freq


def test_import_error(_mock_have_no_lib):  # noqa ARG001 (uses fixture from conftest.py)
    """Test that ac.SoundDeviceSamplesGenerator raises ImportError if sounddevice is not available.

    Parameters
    ----------
    _mock_have_no_lib : pytest.fixture
        Mock behaviour: sounddevice library does not exist.
    """
    del ac.config.__dict__['_traits_cache_have_sounddevice']  # remove cached property
    with pytest.raises(ImportError):
        ac.SoundDeviceSamplesGenerator()
