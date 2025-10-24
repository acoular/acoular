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


@parametrize('mismatch_type', ['sample_freq', 'num_channels'])
def test_mixer_validation_errors(create_time_data_source, mismatch_type):
    """Test Mixer raises ValueError when sources have incompatible properties."""
    primary_source = create_time_data_source(num_channels=2, num_samples=100)

    if mismatch_type == 'sample_freq':
        secondary_source = ac.TimeSamples(sample_freq=44100, data=np.random.randn(100, 2))
        error_match = 'Sample frequency.*does not fit'
    else:
        secondary_source = create_time_data_source(num_channels=3, num_samples=100)
        error_match = 'Channel count.*does not fit'

    mixer = ac.Mixer(source=primary_source, sources=[secondary_source])

    with pytest.raises(ValueError, match=error_match):
        mixer.validate_sources()


@parametrize(
    'primary_samples,secondary_samples,block_size,expected_samples',
    [
        (200, 50, 64, 50),  # Test StopIteration handling - secondary ends early
        (75, 60, 64, 60),  # Test shape truncation - non-aligned lengths
        (500, 100, 64, 100),  # Test early termination with larger difference
    ],
)
def test_mixer_result_handling(
    create_time_data_source, primary_samples, secondary_samples, block_size, expected_samples
):
    """Test Mixer handles different source lengths and shape mismatches."""
    primary_source = create_time_data_source(num_channels=2, num_samples=primary_samples)
    secondary_source = create_time_data_source(num_channels=2, num_samples=secondary_samples)
    mixer = ac.Mixer(source=primary_source, sources=[secondary_source])

    results = []
    for block in mixer.result(num=block_size):
        results.append(block)

    total_samples = sum(block.shape[0] for block in results)
    assert total_samples == expected_samples

    assert all(b.shape[0] > 0 for b in results), 'All blocks should have samples'


def test_mixer_addition_correctness():
    """Test that Mixer correctly adds signals from multiple sources."""
    rng = np.random.RandomState(42)
    data1 = rng.randn(100, 2)
    data2 = rng.randn(100, 2)
    expected = data1 + data2

    source1 = ac.TimeSamples(sample_freq=51200, data=data1.copy())
    source2 = ac.TimeSamples(sample_freq=51200, data=data2.copy())
    mixer = ac.Mixer(source=source1, sources=[source2])

    mixed_data = []
    for block in mixer.result(num=100):
        mixed_data.append(block.copy())
    mixed_data = np.vstack(mixed_data)

    np.testing.assert_allclose(mixed_data, expected)


@parametrize('filter_cls', [ac.FiltOctave, ac.FiltFiltOctave])
def test_filt_octave_band_frequency_too_high(create_time_data_source, filter_cls):
    """Test octave filters raise ValueError when band frequency is too high."""
    source = create_time_data_source(num_channels=2, num_samples=100)
    filt = filter_cls(source=source, band=20000.0)

    with pytest.raises(ValueError, match='band frequency too high'):
        _ = filt.sos


@parametrize('weight', ['A', 'C', 'Z'])
def test_filt_freq_weight_types(create_time_data_source, weight):
    """Test FiltFreqWeight correctly generates SOS coefficients for different weight types."""
    source = create_time_data_source(num_channels=2, num_samples=100)
    filt = ac.FiltFreqWeight(source=source, weight=weight)
    sos = filt.sos

    assert sos is not None
    assert isinstance(sos, np.ndarray)
    assert sos.ndim == 2
    assert sos.shape[1] == 6
    assert sos.shape[0] >= 1


def test_filt_freq_weight_z_flat_response():
    """Test that Z-weighting produces a flat frequency response."""
    rng = np.random.RandomState(42)
    data = rng.randn(100, 1)
    source = ac.TimeSamples(sample_freq=51200, data=data.copy())
    filt = ac.FiltFreqWeight(source=source, weight='Z')

    output = []
    for block in filt.result(num=100):
        output.append(block)
    output = np.vstack(output)

    # Z-weighting should pass signal unchanged (flat response)
    np.testing.assert_allclose(output, data, rtol=1e-3, atol=1e-6)
