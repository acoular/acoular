"""Tests for processing sources whose length is an indefinite stream."""

import acoular as ac

import numpy as np
import pytest
from traits.trait_errors import TraitError


class IndefiniteSamples(ac.SamplesGenerator):
    """Small deterministic stand-in for a live input."""

    sample_freq = 1.0
    num_samples = -1
    num_channels = 1

    def result(self, num):
        offset = 0
        while True:
            yield np.arange(offset, offset + num)[:, np.newaxis]
            offset += num


def test_streaming_blocks_preserve_indefinite_length(monkeypatch):
    source = IndefiniteSamples()

    masked = ac.MaskedTimeOut(source=source, start=3)
    assert masked.num_samples == -1
    np.testing.assert_array_equal(next(masked.result(4)).ravel(), [3.0, 4.0, 5.0, 6.0])

    average = ac.Average(source=source, num_per_average=2)
    assert average.num_samples == -1
    assert next(average.result(1)).shape == (1, 1)

    monkeypatch.setattr(ac.config, 'global_caching', 'all')
    cache = ac.Cache(source=source)
    assert next(cache.result(4)).shape == (4, 1)
    assert cache._h5f is None


def test_time_convolve_streams_indefinite_source():
    source = ac.MaskedTimeOut(source=IndefiniteSamples(), start=3)
    convolve = ac.TimeConvolve(source=source, kernel=np.array([1.0]))
    np.testing.assert_allclose(next(convolve.result(4)).ravel(), [3.0, 4.0, 5.0, 6.0], atol=1e-15)


@pytest.mark.parametrize(
    ('block', 'kwargs'),
    [
        (ac.TimeReverse, {}),
        (ac.FiltFiltOctave, {'band': 0.1}),
    ],
)
def test_full_signal_blocks_reject_indefinite_source(block, kwargs):
    with pytest.raises(ValueError, match='finite source'):
        next(block(source=IndefiniteSamples(), **kwargs).result(4))


def test_power_spectra_rejects_indefinite_source():
    spectra = ac.PowerSpectra(source=IndefiniteSamples(), cached=False, block_size=128)
    with pytest.raises(ValueError, match='finite source'):
        _ = spectra.csm


def test_write_h5_can_bound_an_indefinite_source(tmp_path):
    writer = ac.WriteH5(source=IndefiniteSamples(), file=tmp_path / 'stream.h5', num_samples_write=3)
    blocks = list(writer.result(4))
    assert sum(block.shape[0] for block in blocks) == 3


def test_write_wav_requires_scale_for_indefinite_source(tmp_path):
    writer = ac.WriteWAV(source=IndefiniteSamples(), file=tmp_path / 'stream.wav', channels=[0])
    with pytest.raises(ValueError, match='max_val'):
        next(writer.result(4))


def test_signal_generator_rejects_indefinite_length():
    with pytest.raises(TraitError):
        ac.WNoiseGenerator(num_samples=-1)


def test_masked_time_out_rejects_end_relative_stream_selection():
    masked = ac.MaskedTimeOut(source=IndefiniteSamples(), stop=4)
    with pytest.raises(ValueError, match='indefinite streams'):
        _ = masked.num_samples
