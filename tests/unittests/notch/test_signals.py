"""Tests for synthetic signal generator."""

from tests.unittests.notch.notch_helpers import generate_tonal_signal

import numpy as np


class TestGenerateTonalSignal:
    """Test suite for tonal signal generator with harmonics."""

    def test_output_shape_single_channel(self):
        """Test that output has correct shape for single channel."""
        duration = 1.0
        sample_freq = 16000
        num_channels = 1

        signal = generate_tonal_signal(
            f0=100,
            harmonics=[1, 2, 3],
            duration=duration,
            sample_freq=sample_freq,
            num_channels=num_channels
        )

        expected_samples = int(duration * sample_freq)
        assert signal.shape == (expected_samples, num_channels)
        assert signal.dtype == np.float64

    def test_output_shape_multi_channel(self):
        """Test that output has correct shape for multiple channels."""
        duration = 0.5
        sample_freq = 8000
        num_channels = 4

        signal = generate_tonal_signal(
            f0=200,
            harmonics=[1],
            duration=duration,
            sample_freq=sample_freq,
            num_channels=num_channels
        )

        expected_samples = int(duration * sample_freq)
        assert signal.shape == (expected_samples, num_channels)
        assert signal.dtype == np.float64

    def test_frequency_content_fundamental(self):
        """Test that FFT shows peak at fundamental frequency."""
        f0 = 100
        duration = 1.0
        sample_freq = 16000

        signal = generate_tonal_signal(
            f0=f0,
            harmonics=[1],
            duration=duration,
            sample_freq=sample_freq,
            num_channels=1,
            snr_db=40  # High SNR for clear peak
        )

        # Compute FFT
        fft = np.fft.rfft(signal[:, 0])
        freqs = np.fft.rfftfreq(len(signal), 1/sample_freq)
        magnitude = np.abs(fft)

        # Find peak frequency
        peak_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_idx]

        # Peak should be within ±1 Hz of f0
        assert abs(peak_freq - f0) < 1.0

    def test_frequency_content_harmonics(self):
        """Test that FFT shows peaks at all harmonic frequencies."""
        f0 = 100
        harmonics = [1, 2, 3]
        duration = 1.0
        sample_freq = 16000

        signal = generate_tonal_signal(
            f0=f0,
            harmonics=harmonics,
            duration=duration,
            sample_freq=sample_freq,
            num_channels=1,
            snr_db=40  # High SNR for clear peaks
        )

        # Compute FFT
        fft = np.fft.rfft(signal[:, 0])
        freqs = np.fft.rfftfreq(len(signal), 1/sample_freq)
        magnitude = np.abs(fft)

        # Check for peaks at each harmonic frequency
        for harmonic in harmonics:
            expected_freq = f0 * harmonic
            # Find peak near expected frequency
            freq_mask = (freqs > expected_freq - 5) & (freqs < expected_freq + 5)
            local_magnitude = magnitude[freq_mask]
            local_freqs = freqs[freq_mask]

            if len(local_magnitude) > 0:
                peak_idx = np.argmax(local_magnitude)
                peak_freq = local_freqs[peak_idx]
                # Peak should be within ±1 Hz
                assert abs(peak_freq - expected_freq) < 1.0

    def test_snr_measurement(self):
        """Test that measured SNR is close to specified SNR."""
        f0 = 100
        harmonics = [1, 2, 3]
        duration = 2.0  # Longer duration for better SNR estimate
        sample_freq = 16000
        target_snr_db = 20

        signal = generate_tonal_signal(
            f0=f0,
            harmonics=harmonics,
            duration=duration,
            sample_freq=sample_freq,
            num_channels=1,
            snr_db=target_snr_db
        )

        # Compute FFT
        fft = np.fft.rfft(signal[:, 0])
        freqs = np.fft.rfftfreq(len(signal), 1/sample_freq)
        magnitude_squared = np.abs(fft) ** 2

        # Identify signal bins (harmonics) and noise bins
        signal_power = 0
        signal_bins = set()

        for harmonic in harmonics:
            expected_freq = f0 * harmonic
            # Find bin closest to harmonic
            bin_idx = np.argmin(np.abs(freqs - expected_freq))
            signal_bins.add(bin_idx)
            signal_power += magnitude_squared[bin_idx]

        # Noise power from all other bins (excluding DC and signal bins)
        noise_bins = [i for i in range(1, len(magnitude_squared)) if i not in signal_bins]
        noise_power = np.mean(magnitude_squared[noise_bins]) * len(noise_bins)

        # Calculate SNR in dB
        measured_snr_db = 10 * np.log10(signal_power / noise_power)

        # Should be within ±2 dB of target
        assert abs(measured_snr_db - target_snr_db) < 2.0

    def test_edge_case_no_harmonics(self):
        """Test with empty harmonics list (pure noise)."""
        signal = generate_tonal_signal(
            f0=100,
            harmonics=[],
            duration=0.5,
            sample_freq=8000,
            num_channels=1,
            snr_db=20
        )

        expected_samples = int(0.5 * 8000)
        assert signal.shape == (expected_samples, 1)

        # Signal should be noise-only (no strong peaks)
        fft = np.fft.rfft(signal[:, 0])
        magnitude = np.abs(fft)
        # Check that no single frequency dominates (all similar magnitude)
        max_mag = np.max(magnitude[1:])  # Exclude DC
        mean_mag = np.mean(magnitude[1:])
        # Ratio should be small for noise
        assert max_mag / mean_mag < 5.0

    def test_edge_case_infinite_snr(self):
        """Test with infinite SNR (no noise)."""
        signal = generate_tonal_signal(
            f0=100,
            harmonics=[1],
            duration=1.0,
            sample_freq=16000,
            num_channels=1,
            snr_db=np.inf
        )

        # Should have a pure sinusoid
        expected_samples = int(1.0 * 16000)
        assert signal.shape == (expected_samples, 1)

        # All channels should be identical
        t = np.arange(expected_samples) / 16000
        expected = np.sin(2 * np.pi * 100 * t)
        np.testing.assert_allclose(signal[:, 0], expected, rtol=1e-10)

    def test_multi_channel_consistency(self):
        """Test that all channels have same signal (different noise)."""
        signal = generate_tonal_signal(
            f0=100,
            harmonics=[1, 2],
            duration=1.0,
            sample_freq=16000,
            num_channels=3,
            snr_db=30
        )

        # All channels should have same frequency content
        for ch in range(3):
            fft = np.fft.rfft(signal[:, ch])
            freqs = np.fft.rfftfreq(len(signal), 1/16000)
            magnitude = np.abs(fft)

            # Check for peak at 100 Hz
            freq_mask = (freqs > 95) & (freqs < 105)
            local_freqs = freqs[freq_mask]
            local_magnitude = magnitude[freq_mask]
            peak_idx = np.argmax(local_magnitude)
            peak_freq = local_freqs[peak_idx]

            assert abs(peak_freq - 100) < 1.0
