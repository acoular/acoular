"""Tests for StaticNotchFilter class.

This module tests the second-order IIR notch filter implementation,
verifying mathematical correctness, stability, and acoular compatibility.
"""

from acoular import NotchFilter

from tests.unittests.notch.notch_helpers import MockSamplesGenerator, compute_fft_power_db, generate_tonal_signal

import numpy as np
import pytest
from scipy.signal import freqz


class TestStaticNotchFilterBasics:
    """Test basic functionality and interface compliance."""

    def test_initialization(self, standard_params):
        """Test that StaticNotchFilter initializes with correct parameters."""
        sample_freq = standard_params['sample_freq']
        signal = generate_tonal_signal(
            f0=440, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Create filter
        filter_obj = NotchFilter(
            f_notch=440.0,
            pole_radius=0.95,
            source=source
        )

        # Verify traits are set
        assert filter_obj.f_notch == 440.0
        assert filter_obj.pole_radius == 0.95
        assert filter_obj.sample_freq == sample_freq
        assert filter_obj.num_channels == 1

    def test_result_generator(self, standard_params):
        """Test that result() yields blocks of filtered data."""
        sample_freq = standard_params['sample_freq']
        signal = generate_tonal_signal(
            f0=440, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=440.0,
            pole_radius=0.95,
            source=source
        )

        # Request blocks and verify shape
        block_size = 512
        blocks = list(filter_obj.result(block_size))

        assert len(blocks) > 0, "Should yield at least one block"

        # Check first block shape
        assert blocks[0].shape[1] == 1, "Should preserve channel count"
        assert blocks[0].shape[0] <= block_size, "Block size should not exceed requested"

        # Verify total samples match
        total_samples = sum(block.shape[0] for block in blocks)
        assert total_samples == signal.shape[0], "Should yield all samples"


class TestStaticNotchFilterSuppression:
    """Test tonal suppression at notch frequency."""

    def test_suppression_at_notch_frequency(self, standard_params):
        """Test >20 dB suppression at f_notch."""
        sample_freq = standard_params['sample_freq']
        f_notch = 440.0

        # Generate clean sine at notch frequency
        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)[:, 0]

        # Measure power at notch frequency
        input_power_db = compute_fft_power_db(signal[:, 0], sample_freq, f_notch)
        output_power_db = compute_fft_power_db(output, sample_freq, f_notch)

        suppression_db = input_power_db - output_power_db

        assert suppression_db > 20, \
            f"Expected >20 dB suppression at {f_notch} Hz, got {suppression_db:.1f} dB"

    def test_minimal_suppression_away_from_notch(self, standard_params):
        """Test that frequencies away from notch pass relatively unaffected."""
        sample_freq = standard_params['sample_freq']
        f_notch = 440.0
        f_test = 880.0  # One octave above notch

        # Generate sine at test frequency
        signal = generate_tonal_signal(
            f0=f_test, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)[:, 0]

        # Measure power at test frequency
        input_power_db = compute_fft_power_db(signal[:, 0], sample_freq, f_test)
        output_power_db = compute_fft_power_db(output, sample_freq, f_test)

        suppression_db = input_power_db - output_power_db

        # Should have minimal suppression (<5 dB for one octave separation)
        assert suppression_db < 5, \
            f"Expected <5 dB suppression at {f_test} Hz (away from notch), got {suppression_db:.1f} dB"

    def test_selective_suppression_multi_harmonic(self, standard_params):
        """Test that only target frequency is suppressed in multi-harmonic signal."""
        sample_freq = standard_params['sample_freq']
        f0 = 100.0
        f_notch = 200.0  # Second harmonic

        # Generate signal with 3 harmonics
        signal = generate_tonal_signal(
            f0=f0, harmonics=[1, 2, 3], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)[:, 0]

        # Check suppression at each harmonic
        for h in [1, 2, 3]:
            freq = f0 * h
            input_power = compute_fft_power_db(signal[:, 0], sample_freq, freq)
            output_power = compute_fft_power_db(output, sample_freq, freq)
            suppression = input_power - output_power

            if freq == f_notch:
                assert suppression > 20, \
                    f"Expected >20 dB suppression at notch ({freq} Hz), got {suppression:.1f} dB"
            else:
                # Allow up to 5 dB for frequencies one octave away
                assert suppression < 5, \
                    f"Expected <5 dB suppression at {freq} Hz (not notch), got {suppression:.1f} dB"


class TestStaticNotchFilterStability:
    """Test filter stability and numerical robustness."""

    def test_no_divergence(self, standard_params):
        """Test that filter output remains bounded (no divergence)."""
        sample_freq = standard_params['sample_freq']

        # Generate test signal
        signal = generate_tonal_signal(
            f0=440, harmonics=[1, 2, 3], duration=2.0,
            sample_freq=sample_freq, num_channels=1, snr_db=20
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=440.0,
            pole_radius=0.95,
            source=source
        )

        # Collect output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        # Check for NaN or Inf
        assert np.all(np.isfinite(output)), "Output contains NaN or Inf values"

        # Check that output is bounded (should be similar magnitude to input)
        input_max = np.max(np.abs(signal))
        output_max = np.max(np.abs(output))

        assert output_max < 10 * input_max, \
            f"Output diverged: max output {output_max:.2f} >> max input {input_max:.2f}"

    def test_edge_case_nyquist(self, standard_params):
        """Test filter stability near Nyquist frequency."""
        sample_freq = standard_params['sample_freq']
        f_nyquist = sample_freq / 2
        f_notch = f_nyquist * 0.9  # 90% of Nyquist

        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=source
        )

        # Should not crash or produce NaN
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        assert np.all(np.isfinite(output)), \
            "Filter unstable near Nyquist frequency"

    def test_edge_case_low_frequency(self, standard_params):
        """Test filter stability at low frequencies."""
        sample_freq = standard_params['sample_freq']
        f_notch = 10.0  # Very low frequency

        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=source
        )

        # Should not crash or produce NaN
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        assert np.all(np.isfinite(output)), \
            "Filter unstable at low frequencies"


class TestStaticNotchFilterFrequencyResponse:
    """Test theoretical frequency response validation."""

    def test_frequency_response(self, standard_params):
        """Test frequency response matches theoretical transfer function.

        Validates theoretical H(z) using scipy.signal.freqz to compute
        exact frequency response from filter coefficients.
        """
        sample_freq = standard_params['sample_freq']
        f_notch = 1000.0
        pole_radius = 0.95

        # Generate minimal signal (just for initialization)
        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=0.1,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Create filter
        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=pole_radius,
            source=source
        )

        # Get filter coefficients
        b, a = filter_obj.coefficients

        # Compute theoretical frequency response
        w, h = freqz(b, a, worN=8192, fs=sample_freq)

        # Convert to dB magnitude
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)  # Add small value to avoid log(0)

        # Find indices for test frequencies
        def find_freq_idx(freq):
            return np.argmin(np.abs(w - freq))

        idx_notch = find_freq_idx(f_notch)
        idx_half = find_freq_idx(f_notch / 2)
        idx_double = find_freq_idx(f_notch * 2)

        # Verify at f_notch (1000 Hz): should have deep notch (< -20 dB)
        assert magnitude_db[idx_notch] < -20, \
            f"Expected < -20 dB at notch frequency ({f_notch} Hz), got {magnitude_db[idx_notch]:.1f} dB"

        # Verify at f_notch/2 (500 Hz): minimal impact (> -3 dB)
        assert magnitude_db[idx_half] > -3, \
            f"Expected > -3 dB at f/2 ({f_notch/2} Hz), got {magnitude_db[idx_half]:.1f} dB"

        # Verify at 2×f_notch (2000 Hz): minimal impact (> -3 dB)
        assert magnitude_db[idx_double] > -3, \
            f"Expected > -3 dB at 2×f ({f_notch*2} Hz), got {magnitude_db[idx_double]:.1f} dB"


class TestStaticNotchFilterPerformance:
    """Test performance against PROJECT.md requirements."""

    @pytest.mark.parametrize("pole_radius", [0.9, 0.95, 0.99])
    def test_tonal_suppression_performance(self, standard_params, pole_radius):
        """Test >25 dB tonal suppression requirement from PROJECT.md.

        Tests multiple pole_radius values:
        - r=0.9: narrower notch
        - r=0.95: standard value
        - r=0.99: very narrow notch, maximum suppression

        Validates against PROJECT.md requirement: "Tonal suppression: 25-30 dB
        at harmonic frequencies"
        """
        sample_freq = standard_params['sample_freq']
        f_notch = 1000.0

        # Generate pure 1000 Hz tone (infinite SNR for cleaner measurement)
        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Create filter
        filter_obj = NotchFilter(
            f_notch=f_notch,
            pole_radius=pole_radius,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)[:, 0]

        # Measure input and output power at 1000 Hz via FFT
        fft_in = np.fft.rfft(signal[:, 0])
        fft_out = np.fft.rfft(output)

        # Find bin corresponding to 1000 Hz
        bin_idx = int(f_notch * len(signal[:, 0]) / sample_freq)

        # Power: abs(fft[bin_idx])**2
        power_in = np.abs(fft_in[bin_idx])**2
        power_out = np.abs(fft_out[bin_idx])**2

        # Compute suppression in dB
        # Add small epsilon to avoid log(0) if perfect suppression
        suppression_db = 10 * np.log10((power_in + 1e-20) / (power_out + 1e-20))

        # Assert PROJECT.md requirement: >25 dB suppression
        assert suppression_db > 25, \
            f"Expected >25 dB suppression (PROJECT.md requirement), got {suppression_db:.1f} dB for r={pole_radius}"
