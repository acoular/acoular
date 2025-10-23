from pathlib import Path

import acoular as ac
import numpy as np
import pytest
from pytest_cases import parametrize


def test_td_dir(tmp_path, create_time_data_source):
    """Test that global time data directory is used when no filename is given."""
    ac.config.td_dir = tmp_path
    time_data = create_time_data_source(num_channels=2, num_samples=1)
    h5 = ac.WriteH5(source=time_data)
    h5.save()
    assert (tmp_path / Path(h5.file).name).exists()


@parametrize('h5library', ['pytables', 'h5py'])
@parametrize('data', ['test', 10, np.random.randn(3)], ids=['str', 'scalar', 'array'])
@parametrize('acoular_cls', [ac.TimeSamples, ac.MaskedTimeSamples])
def test_metadata(tmp_path, create_time_data_source, acoular_cls, h5library, data):
    """Test that metadata can be saved in the h5 file and loaded correctly."""
    ac.config.h5library = h5library
    # save metadata in h5 file
    ac.config.td_dir = tmp_path
    time_data = create_time_data_source(num_channels=2, num_samples=1)
    h5 = ac.WriteH5(source=time_data, metadata={'test': data})
    h5.save()
    # load metadata from h5 file
    ts = acoular_cls(file=h5.file)
    if isinstance(data, (float, int)):
        assert ts.metadata['test'][()] == data
    if isinstance(data, str):
        assert ts.metadata['test'][()] == data.encode()
    else:
        np.testing.assert_allclose(ts.metadata['test'][()], data)


def test_mixer_sample_freq_mismatch(create_time_data_source):
    """Test Mixer raises ValueError when sources have different sample frequencies.

    Parameters
    ----------
    create_time_data_source : callable
        Fixture factory for creating time data sources.
    """
    # Create primary source with sample_freq=51200
    primary_source = create_time_data_source(num_channels=2, num_samples=100)
    
    # Create secondary source with different sample_freq
    secondary_source = ac.TimeSamples(sample_freq=44100, data=np.random.randn(100, 2))
    
    # Create Mixer and expect ValueError
    mixer = ac.Mixer(source=primary_source, sources=[secondary_source])
    
    with pytest.raises(ValueError, match='Sample frequency.*does not fit'):
        mixer.validate_sources()


def test_mixer_channel_count_mismatch(create_time_data_source):
    """Test Mixer raises ValueError when sources have different channel counts.

    This test covers lines 1498-1503 of tprocess.py by verifying that validate_sources
    raises a ValueError when additional sources have a different number of channels
    than the primary source.

    Parameters
    ----------
    create_time_data_source : callable
        Fixture factory for creating time data sources.
    """
    # Create primary source with 2 channels
    primary_source = create_time_data_source(num_channels=2, num_samples=100)
    
    # Create secondary source with 3 channels
    secondary_source = create_time_data_source(num_channels=3, num_samples=100)
    
    # Create Mixer and expect ValueError
    mixer = ac.Mixer(source=primary_source, sources=[secondary_source])
    
    with pytest.raises(ValueError, match='Channel count.*does not fit'):
        mixer.validate_sources()


def test_mixer_stop_iteration_handling(create_time_data_source):
    """Test Mixer stops when one of the additional sources ends early.

    Parameters
    ----------
    create_time_data_source : callable
        Fixture factory for creating time data sources.
    """
    # Create primary source with 200 samples
    primary_source = create_time_data_source(num_channels=2, num_samples=200)
    
    # Create secondary source with only 50 samples (shorter)
    secondary_source = create_time_data_source(num_channels=2, num_samples=50)
    
    # Create Mixer
    mixer = ac.Mixer(source=primary_source, sources=[secondary_source])
    
    # Collect all results
    results = []
    for block in mixer.result(num=64):
        results.append(block)
    
    # Total samples should be limited by the shorter source (50 samples)
    total_samples = sum(block.shape[0] for block in results)
    assert total_samples == 50


def test_mixer_shape_handling_with_truncation(create_time_data_source):
    """Test Mixer handles shape mismatches and truncates appropriately.

    Parameters
    ----------
    create_time_data_source : callable
        Fixture factory for creating time data sources.
    """
    # Create primary source with 75 samples (not a multiple of block size)
    primary_source = create_time_data_source(num_channels=2, num_samples=75)
    
    # Create secondary source with 60 samples (shorter, also not a multiple)
    secondary_source = create_time_data_source(num_channels=2, num_samples=60)
    
    # Create Mixer
    mixer = ac.Mixer(source=primary_source, sources=[secondary_source])
    
    # Process with block size of 64
    results = []
    for block in mixer.result(num=64):
        results.append(block)
    
    # Should get 1 block with 60 samples (limited by secondary_source)
    assert len(results) == 1
    assert results[0].shape[0] == 60


def test_mixer_addition_correctness():
    """Test that Mixer correctly adds signals from multiple sources.

    Parameters
    ----------
    create_time_data_source : callable
        Fixture factory for creating time data sources.
    """
    # Create sources with known data
    # Need to make a copy since Mixer modifies arrays in-place
    rng = np.random.RandomState(42)
    data1 = rng.randn(100, 2)
    data2 = rng.randn(100, 2)
    
    # Store expected result before creating sources (as Mixer modifies in-place)
    expected = data1 + data2
    
    source1 = ac.TimeSamples(sample_freq=51200, data=data1.copy())
    source2 = ac.TimeSamples(sample_freq=51200, data=data2.copy())
    
    # Create Mixer
    mixer = ac.Mixer(source=source1, sources=[source2])
    
    # Get mixed result
    mixed_data = []
    for block in mixer.result(num=100):
        mixed_data.append(block.copy())  # Copy to avoid modification after yield
    
    mixed_data = np.vstack(mixed_data)
    
    # Verify the sum is correct
    np.testing.assert_allclose(mixed_data, expected)

