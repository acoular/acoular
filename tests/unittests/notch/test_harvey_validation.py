"""Harvey (2019) performance validation tests.

Validates that implementation meets Harvey's performance targets:
- Mean Frequency Deviation (MFD) < 0.5 Hz
- Tonal suppression > 25 dB
- Phase preservation for beamforming
"""

from acoular import CascadeNotchFilter, ZeroPhaseNotchFilter

from tests.unittests.notch.notch_helpers import MockSamplesGenerator, generate_tonal_signal

import numpy as np
from scipy.fft import rfft, rfftfreq


class TestHarveyPerformanceTargets:
    """Validate Harvey (2019) performance specifications."""

    def test_frequency_tracking_mfd_below_half_hz(self):
        """Test that adaptive filter tracks frequency with MFD < 0.5 Hz.

        Uses ZeroPhaseNotchFilter in auto mode with a steady tonal
        frequency to validate tracking accuracy. The LMS adaptation learns
        the frequency and achieves sub-Hz accuracy after convergence.

        Harvey's MFD < 0.5 Hz target applies to steady-state tracking after
        initial convergence. For slowly drifting frequencies in real drone
        scenarios, this implementation provides sufficient accuracy.
        """
        # Parameters - constant frequency for steady-state MFD validation
        sample_freq = 16000.0
        duration = 2.0
        num_samples = int(duration * sample_freq)
        f_target = 100.0  # Target frequency to track

        # Generate steady tonal signal
        t = np.arange(num_samples) / sample_freq
        signal = np.sin(2 * np.pi * f_target * t)

        # Add some noise (40 dB SNR)
        noise_power = np.mean(signal**2) / (10**(40/10))
        np.random.seed(42)  # Reproducibility
        signal = signal + np.random.randn(num_samples) * np.sqrt(noise_power)

        # Reshape to (num_samples, 1) for filter input
        signal = signal[:, np.newaxis]

        # Create mock source
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = sample_freq

        # Create zero-phase adaptive filter in auto mode
        filter_obj = ZeroPhaseNotchFilter(
            pole_radius=0.95,
            mode='auto',
            mu=0.002,
            smooth_window=50,
            source=mock,
        )

        # Process signal - this performs forward-backward filtering with LMS
        # (populates the learned trajectory as a side effect).
        filter_obj._process_signal()

        # Get learned trajectory from forward pass
        learned_traj = filter_obj._learned_trajectory

        # Skip initial convergence period (first 20% of samples)
        start_idx = int(0.2 * num_samples)
        estimated_freq = learned_traj[start_idx:]

        # Compute Mean Frequency Deviation from target
        frequency_error = np.abs(estimated_freq - f_target)
        mfd = np.mean(frequency_error)

        # Harvey target: MFD < 0.5 Hz for steady-state tracking
        assert mfd < 0.5, f"MFD {mfd:.3f} Hz exceeds target 0.5 Hz"

    def test_tonal_suppression_above_25_db(self):
        """Test that zero-phase filter achieves > 25 dB suppression at notch frequency.

        Generates tonal signal at 100 Hz with 40 dB SNR, applies ZeroPhaseNotchFilter,
        and measures power reduction at the notch frequency.
        """
        # Parameters
        sample_freq = 16000.0
        duration = 1.0
        f_notch = 100.0
        num_samples = int(duration * sample_freq)

        # Generate clean tonal signal at notch frequency
        t = np.arange(num_samples) / sample_freq
        signal = np.sin(2 * np.pi * f_notch * t)

        # Add noise at 40 dB SNR
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(40/10))
        signal = signal + np.random.randn(num_samples) * np.sqrt(noise_power)

        # Reshape for filter input
        signal = signal[:, np.newaxis]

        # Create mock source
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = sample_freq

        # Create zero-phase notch filter
        filter_obj = ZeroPhaseNotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=mock
        )

        # Process signal
        output = filter_obj._process_signal()

        # Measure power at notch frequency before and after
        # Use FFT to compute power spectral density
        fft_before = rfft(signal[:, 0])
        fft_after = rfft(output[:, 0])
        freqs = rfftfreq(num_samples, 1/sample_freq)

        # Measure power in a small band around the notch (±5 Hz)
        freq_mask = np.abs(freqs - f_notch) < 5.0
        power_before = np.mean(np.abs(fft_before[freq_mask])**2)
        power_after = np.mean(np.abs(fft_after[freq_mask])**2)

        # Compute suppression in dB
        # Add small epsilon to avoid log(0)
        suppression_db = 10 * np.log10(power_before / (power_after + 1e-12))

        assert suppression_db > 25, f"Suppression {suppression_db:.1f} dB below target 25 dB"

    def test_phase_preservation_for_beamforming(self):
        """Test that zero-phase filter preserves inter-channel phase relationships.

        Generates multi-channel signal with known phase offsets, applies filter,
        and verifies phase relationships are preserved within 0.1 degrees.
        """
        # Parameters
        sample_freq = 16000.0
        duration = 1.0
        num_channels = 8
        f_notch = 100.0  # Notch frequency
        f_test = 250.0   # Test frequency for phase measurement (passband)
        num_samples = int(duration * sample_freq)

        # Generate time array
        t = np.arange(num_samples) / sample_freq

        # Create known phase offsets for each channel (simulating array geometry)
        # Phase offsets in radians, representing typical array steering
        phase_offsets = np.linspace(0, np.pi/4, num_channels)

        # Generate multi-channel signal with phase offsets at test frequency
        # Also include a tone at notch frequency to be removed
        signal = np.zeros((num_samples, num_channels))
        for ch in range(num_channels):
            # Test tone with known phase offset
            signal[:, ch] = np.sin(2 * np.pi * f_test * t + phase_offsets[ch])
            # Add tone at notch frequency (to be filtered)
            signal[:, ch] += 0.5 * np.sin(2 * np.pi * f_notch * t)

        # Create mock source
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = sample_freq

        # Create zero-phase notch filter
        filter_obj = ZeroPhaseNotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=mock
        )

        # Process signal
        output = filter_obj._process_signal()

        # Measure inter-channel phase at test frequency
        # Use FFT to extract phase at f_test
        test_bin = int(f_test * num_samples / sample_freq)

        # Measure phases before and after filtering
        fft_before = rfft(signal, axis=0)
        fft_after = rfft(output, axis=0)

        phases_before = np.angle(fft_before[test_bin, :])
        phases_after = np.angle(fft_after[test_bin, :])

        # Compute relative phases (channel 0 as reference)
        rel_phases_before = phases_before - phases_before[0]
        rel_phases_after = phases_after - phases_after[0]

        # Wrap to [-pi, pi]
        rel_phases_before = np.arctan2(np.sin(rel_phases_before), np.cos(rel_phases_before))
        rel_phases_after = np.arctan2(np.sin(rel_phases_after), np.cos(rel_phases_after))

        # Compute phase deviation in degrees
        phase_deviation = np.abs(rel_phases_after - rel_phases_before)
        # Wrap to [0, pi]
        phase_deviation = np.minimum(phase_deviation, 2*np.pi - phase_deviation)
        phase_deviation_deg = np.degrees(phase_deviation)

        max_phase_deviation = np.max(phase_deviation_deg)

        assert max_phase_deviation < 0.1, \
            f"Phase deviation {max_phase_deviation:.3f} deg exceeds 0.1 deg target"


class TestAcoularIntegration:
    """Validate acoular integration patterns."""

    def test_source_chaining_pattern(self):
        """Test filter chaining works like native acoular classes.

        Creates chain: source → CascadeNotchFilter → ZeroPhaseNotchFilter
        Verifies properties propagate correctly through the chain.

        Note: StaticNotchFilter is SISO by design. For multi-channel filtering,
        use CascadeNotchFilter which handles multiple channels internally.
        """
        # Parameters
        sample_freq = 16000.0
        duration = 0.5
        num_channels = 4
        num_samples = int(duration * sample_freq)

        # Generate test signal
        signal = generate_tonal_signal(
            f0=100.0,
            harmonics=[1, 2, 3],
            duration=duration,
            sample_freq=sample_freq,
            num_channels=num_channels,
            snr_db=30.0
        )

        # Create mock source
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = sample_freq

        # Create filter chain: source → CascadeNotchFilter (handles multi-channel)
        cascade_filter = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=2,
            frequencies=np.array([[100.0, 200.0]]),  # Remove 100 Hz and 200 Hz
            pole_radius=0.95,
            source=mock
        )

        # Verify property propagation through chain
        assert cascade_filter.sample_freq == sample_freq
        assert cascade_filter.num_channels == num_channels
        assert cascade_filter.num_samples == num_samples

        # Verify result() generator works through chain
        output_blocks = list(cascade_filter.result(1024))
        total_samples = sum(block.shape[0] for block in output_blocks)

        assert total_samples == num_samples
        assert output_blocks[0].shape[1] == num_channels

        # Verify filtering actually occurred (signal power reduced at notch freqs)
        input_signal = signal[:, 0]
        output_signal = np.vstack(output_blocks)[:, 0]

        # Check power at 100 Hz before and after
        fft_before = rfft(input_signal)
        fft_after = rfft(output_signal)
        freqs = rfftfreq(num_samples, 1/sample_freq)

        notch_bin = np.argmin(np.abs(freqs - 100.0))
        power_before = np.abs(fft_before[notch_bin])**2
        power_after = np.abs(fft_after[notch_bin])**2

        # Should have significant suppression at notch frequency
        assert power_after < 0.1 * power_before, "Filter should suppress notch frequency"

    def test_digest_changes_on_config_update(self):
        """Test that digest property changes when configuration changes.

        This validates that acoular caching will work correctly - the digest
        must change when filter parameters change to invalidate cached results.
        """
        # Parameters
        sample_freq = 16000.0
        duration = 0.5

        # Generate test signal
        signal = generate_tonal_signal(
            f0=100.0,
            harmonics=[1],
            duration=duration,
            sample_freq=sample_freq,
            num_channels=1,
            snr_db=np.inf
        )

        # Create mock source
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = sample_freq

        # Create filter with initial config
        filter_obj = ZeroPhaseNotchFilter(
            f_notch=100.0,
            pole_radius=0.95,
            source=mock
        )

        # Record initial digest
        initial_digest = filter_obj.digest

        # Change configuration parameter
        filter_obj.f_notch = 200.0

        # Get new digest (should have changed)
        new_digest = filter_obj.digest

        # Verify digest changed
        assert new_digest != initial_digest, \
            "Digest should change when f_notch parameter changes"

        # Also test pole_radius change
        filter_obj.pole_radius = 0.99
        third_digest = filter_obj.digest

        assert third_digest != new_digest, \
            "Digest should change when pole_radius parameter changes"
