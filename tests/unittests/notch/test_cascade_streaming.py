"""Tests for CascadeNotchFilter streaming zero-phase mode.

Verifies that the streaming API (_reset_streaming / _filter_block_streaming /
_flush_streaming) produces output consistent with the batch zero-phase path
(_result_zero_phase).
"""

from acoular import CascadeNotchFilter

from tests.unittests.notch.notch_helpers import MockFreqSource, MockSamplesGenerator, generate_tonal_signal

import numpy as np
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cascade(signal, sample_freq, frequencies, block_size=4096,
                   mode=None, freq_source=None):
    """Build a CascadeNotchFilter with zero_phase=True."""
    source = MockSamplesGenerator(signal, sample_freq)
    S, M = frequencies.shape
    kwargs = dict(
        num_sources=S,
        harmonics_per_source=M,
        frequencies=frequencies,
        pole_radius=0.95,
        zero_phase=True,
        block_size=block_size,
        source=source,
    )
    if mode is not None:
        kwargs['mode'] = mode
    if freq_source is not None:
        kwargs['freq_source'] = freq_source
    return CascadeNotchFilter(**kwargs)


def _batch_output(signal, sample_freq, frequencies, block_size=4096,
                  mode=None, freq_source=None):
    """Run the batch zero-phase path and return the full output array."""
    filt = _build_cascade(signal, sample_freq, frequencies, block_size,
                          mode, freq_source)
    return np.vstack(list(filt.result(block_size)))


def _streaming_output(signal, sample_freq, frequencies, block_size=4096,
                      mode=None, freq_source=None, freq_data=None):
    """Run the streaming zero-phase path and return the full output array.

    *freq_data* is the full (num_samples, S*M) array; it is sliced into
    blocks internally so the same data feeds both batch and streaming paths.
    """
    filt = _build_cascade(signal, sample_freq, frequencies, block_size,
                          mode, freq_source=None)  # freq_source unused in streaming
    filt._reset_streaming()

    num_samples = signal.shape[0]
    blocks = []
    for start in range(0, num_samples, block_size):
        end = min(start + block_size, num_samples)
        data_block = signal[start:end]
        freq_block = freq_data[start:end] if freq_data is not None else None
        out = filt._filter_block_streaming(data_block, freq_block=freq_block)
        if out is not None:
            blocks.append(out)

    tail = filt._flush_streaming()
    blocks.extend(tail)

    if blocks:
        return np.vstack(blocks)
    return np.empty((0, signal.shape[1]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCascadeStreamingBasic:
    """Basic streaming API behaviour."""

    def test_requires_zero_phase(self, standard_params):
        """_reset_streaming raises RuntimeError when zero_phase is False."""
        sample_freq = standard_params['sample_freq']
        signal = np.random.randn(4096, 1)
        source = MockSamplesGenerator(signal, sample_freq)
        filt = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=1,
            frequencies=np.array([[100.0]]),
            pole_radius=0.95,
            zero_phase=False,
            source=source,
        )
        with pytest.raises(RuntimeError, match="zero_phase"):
            filt._reset_streaming()

    def test_pipeline_latency_default(self, standard_params):
        """Default (num_lookahead_blocks=1): first call None, then output."""
        sample_freq = standard_params['sample_freq']
        block_size = 1024
        _S, _M = 2, 3  # 6 filters, but unified cascade → 1 block latency
        num_blocks = 5
        num_samples = block_size * num_blocks
        signal = np.random.randn(num_samples, 1)

        filt = _build_cascade(
            signal, sample_freq,
            frequencies=np.array([[100, 200, 300], [150, 300, 450]], dtype=float),
            block_size=block_size,
        )
        filt._reset_streaming()

        results = []
        for i in range(num_blocks):
            start = i * block_size
            out = filt._filter_block_streaming(signal[start:start + block_size])
            results.append(out)

        # Only the first call returns None (1-block lookahead)
        assert results[0] is None, "Block 0 should be None (buffering)"

        # All subsequent calls should produce output
        for i in range(1, num_blocks):
            assert results[i] is not None, f"Block {i} should produce output"
            assert results[i].shape == (block_size, 1)

    def test_pipeline_latency_multi_lookahead(self, standard_params):
        """num_lookahead_blocks=3: first 3 calls None, then output."""
        sample_freq = standard_params['sample_freq']
        block_size = 1024
        L = 3
        num_blocks = L + 3
        num_samples = block_size * num_blocks
        signal = np.random.randn(num_samples, 1)

        filt = _build_cascade(
            signal, sample_freq,
            frequencies=np.array([[100, 200, 300]], dtype=float),
            block_size=block_size,
        )
        filt.num_lookahead_blocks = L
        filt._reset_streaming()

        results = []
        for i in range(num_blocks):
            start = i * block_size
            out = filt._filter_block_streaming(signal[start:start + block_size])
            results.append(out)

        # First L calls return None
        for i in range(L):
            assert results[i] is None, f"Block {i} should be None (filling)"

        # After that, output flows
        for i in range(L, num_blocks):
            assert results[i] is not None, f"Block {i} should produce output"
            assert results[i].shape == (block_size, 1)

    def test_flush_returns_remaining_blocks(self, standard_params):
        """_flush_streaming returns up to num_lookahead_blocks blocks."""
        sample_freq = standard_params['sample_freq']
        block_size = 1024
        _S, _M = 1, 2  # 2 filters
        L = 1
        num_blocks = 4
        num_samples = block_size * num_blocks
        signal = np.random.randn(num_samples, 1)

        filt = _build_cascade(
            signal, sample_freq,
            frequencies=np.array([[100.0, 200.0]]),
            block_size=block_size,
        )
        filt.num_lookahead_blocks = L
        filt._reset_streaming()

        for i in range(num_blocks):
            start = i * block_size
            filt._filter_block_streaming(signal[start:start + block_size])

        tail = filt._flush_streaming()
        assert len(tail) == L, f"flush should return {L} block(s)"
        for blk in tail:
            assert blk.shape == (block_size, 1)

    def test_flush_multi_lookahead(self, standard_params):
        """_flush_streaming with num_lookahead_blocks=3."""
        sample_freq = standard_params['sample_freq']
        block_size = 1024
        L = 3
        num_blocks = L + 2
        num_samples = block_size * num_blocks
        signal = np.random.randn(num_samples, 1)

        filt = _build_cascade(
            signal, sample_freq,
            frequencies=np.array([[100.0, 200.0]]),
            block_size=block_size,
        )
        filt.num_lookahead_blocks = L
        filt._reset_streaming()

        for i in range(num_blocks):
            start = i * block_size
            filt._filter_block_streaming(signal[start:start + block_size])

        tail = filt._flush_streaming()
        assert len(tail) == L, f"flush should return {L} blocks"


class TestCascadeStreamingVsBatch:
    """Compare streaming output against batch zero-phase output."""

    def test_static_mode_single_filter(self, standard_params):
        """Streaming ≈ batch for simplest case (S=1, M=1, static)."""
        sample_freq = standard_params['sample_freq']
        block_size = 2048
        duration = 1.0
        int(duration * sample_freq)

        signal = generate_tonal_signal(
            f0=200, harmonics=[1], duration=duration,
            sample_freq=sample_freq, num_channels=2, snr_db=30,
        )
        frequencies = np.array([[200.0]])

        batch = _batch_output(signal, sample_freq, frequencies, block_size)
        stream = _streaming_output(signal, sample_freq, frequencies, block_size)

        # Trim to common length (streaming may lose tail samples)
        n = min(batch.shape[0], stream.shape[0])
        # Allow boundary tolerance — skip first and last block
        inner = slice(block_size, n - block_size)

        assert_allclose(
            stream[inner], batch[inner],
            atol=0.05, rtol=0.01,
            err_msg="Streaming and batch static outputs diverge in interior",
        )

    def test_static_mode_multi_filter(self, standard_params):
        """Streaming ≈ batch for S=1, M=3 static cascade."""
        sample_freq = standard_params['sample_freq']
        block_size = 2048
        duration = 1.0

        signal = generate_tonal_signal(
            f0=100, harmonics=[1, 2, 3], duration=duration,
            sample_freq=sample_freq, num_channels=1, snr_db=30,
        )
        frequencies = np.array([[100.0, 200.0, 300.0]])

        batch = _batch_output(signal, sample_freq, frequencies, block_size)
        stream = _streaming_output(signal, sample_freq, frequencies, block_size)

        n = min(batch.shape[0], stream.shape[0])
        # Skip first block (pipeline fill) and last block (flush boundary)
        inner = slice(block_size, n - block_size)
        if inner.start < inner.stop:
            assert_allclose(
                stream[inner], batch[inner],
                atol=0.1, rtol=0.02,
                err_msg="Streaming and batch multi-filter static outputs diverge",
            )

    def test_external_mode(self, standard_params):
        """Streaming ≈ batch for external mode with known frequency trajectories."""
        sample_freq = standard_params['sample_freq']
        block_size = 2048
        duration = 1.0
        num_samples = int(duration * sample_freq)

        # Swept tone: 100→120 Hz + 200→240 Hz
        np.arange(num_samples) / sample_freq
        freq_traj_1 = np.linspace(100, 120, num_samples)
        freq_traj_2 = np.linspace(200, 240, num_samples)
        phase_1 = 2 * np.pi * np.cumsum(freq_traj_1) / sample_freq
        phase_2 = 2 * np.pi * np.cumsum(freq_traj_2) / sample_freq
        signal = (np.sin(phase_1) + 0.7 * np.sin(phase_2))[:, np.newaxis]

        frequencies = np.array([[100.0, 200.0]])
        freq_data = np.column_stack([freq_traj_1, freq_traj_2])  # (N, 2)

        # Batch path needs a freq_source
        batch = _batch_output(
            signal, sample_freq, frequencies, block_size,
            mode='external',
            freq_source=MockFreqSource(freq_data),
        )
        # Streaming path uses freq_data sliced per block
        stream = _streaming_output(
            signal, sample_freq, frequencies, block_size,
            mode='external',
            freq_data=freq_data,
        )

        n = min(batch.shape[0], stream.shape[0])
        # Skip first block (pipeline fill) and last block (flush boundary)
        inner = slice(block_size, n - block_size)
        if inner.start < inner.stop:
            assert_allclose(
                stream[inner], batch[inner],
                atol=0.15, rtol=0.05,
                err_msg="Streaming and batch external-mode outputs diverge",
            )


class TestCascadeStreamingSuppression:
    """Verify that streaming output actually suppresses tonal content."""

    def test_tonal_suppression_static(self, standard_params):
        """Streaming zero-phase cascade suppresses target frequencies."""
        sample_freq = standard_params['sample_freq']
        block_size = 2048
        duration = 1.0

        signal = generate_tonal_signal(
            f0=150, harmonics=[1, 2], duration=duration,
            sample_freq=sample_freq, num_channels=1, snr_db=30,
        )
        frequencies = np.array([[150.0, 300.0]])

        stream = _streaming_output(signal, sample_freq, frequencies, block_size)

        # Measure suppression in a central segment
        n = stream.shape[0]
        seg = slice(n // 4, 3 * n // 4)

        for freq in [150.0, 300.0]:
            fft_in = np.fft.rfft(signal[seg, 0])
            fft_out = np.fft.rfft(stream[seg, 0])
            freqs = np.fft.rfftfreq(signal[seg, 0].shape[0], 1 / sample_freq)
            idx = np.argmin(np.abs(freqs - freq))

            power_in = np.abs(fft_in[idx]) ** 2
            power_out = np.abs(fft_out[idx]) ** 2
            if power_in > 0 and power_out > 0:
                suppression_db = 10 * np.log10(power_in / power_out)
                assert suppression_db > 15, (
                    f"Streaming cascade should suppress {freq} Hz by >15 dB, "
                    f"got {suppression_db:.1f} dB"
                )

    def test_multi_channel_streaming(self, standard_params):
        """Streaming processes multiple channels independently."""
        sample_freq = standard_params['sample_freq']
        block_size = 2048
        duration = 1.0
        num_channels = 3

        signal = generate_tonal_signal(
            f0=200, harmonics=[1], duration=duration,
            sample_freq=sample_freq, num_channels=num_channels, snr_db=30,
        )
        frequencies = np.array([[200.0]])

        stream = _streaming_output(signal, sample_freq, frequencies, block_size)

        assert stream.shape[1] == num_channels
        # Each channel should be finite
        assert np.all(np.isfinite(stream))
