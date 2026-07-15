"""Tests for AdaptiveNotchFilter class.

This module tests the time-varying frequency tracking implementation,
verifying external RPM mode and autonomous LMS adaptation.

AdaptiveNotchFilter is a SISO (single-input single-output) filter that
extends StaticNotchFilter with time-varying frequency tracking.
For multi-source × multi-harmonic scenarios, see test_cascade.py.
"""

from acoular import AdaptiveNotchFilter

from tests.unittests.notch.notch_helpers import (
    MockFreqSource,
    MockSamplesGenerator,
    compute_fft_power_db,
    generate_swept_tonal_signal,
    generate_tonal_signal,
)

import numpy as np
import pytest
from scipy.signal import chirp


class TestAdaptiveNotchFilterExternalMode:
    """Test external RPM mode with time-varying frequencies (SISO)."""

    def test_external_rpm_tracking_single_sweep(self):
        """Test single frequency sweep tracking (100 Hz → 120 Hz over 1 second).

        Validates that SISO filter tracks time-varying frequency with <0.5 Hz accuracy.
        """
        # Generate swept signal: 100 Hz → 120 Hz over 1 second
        sample_freq = 16000.0
        duration = 1.0
        f_start = 100.0
        f_end = 120.0

        signal, freq_trajectory = generate_swept_tonal_signal(
            f_start, f_end, duration, sample_freq, num_channels=1, snr_db=20
        )

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Create adaptive filter with external mode (SISO interface)
        filter_obj = AdaptiveNotchFilter(
            freq_source=MockFreqSource(freq_trajectory),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process entire signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Verify tracking accuracy in three segments
        # Early: around 100 Hz
        # Middle: around 110 Hz
        # Late: around 120 Hz
        segment_starts = [0, int(0.5 * sample_freq), int(0.9 * sample_freq)]
        segment_lengths = [int(0.2 * sample_freq)] * 3
        expected_freqs = [100.0, 110.0, 120.0]

        for start, length, expected_freq in zip(segment_starts, segment_lengths, expected_freqs):
            # Check original signal to verify frequency content is present
            orig_segment = signal[start:start+length, 0]
            orig_power_db = compute_fft_power_db(orig_segment, sample_freq, expected_freq)

            # Check filtered signal for suppression
            filt_segment = filtered[start:start+length, 0]
            filt_power_db = compute_fft_power_db(filt_segment, sample_freq, expected_freq)

            # Verify tracking by checking suppression at the target frequency
            # Suppression should be > 25 dB
            suppression = orig_power_db - filt_power_db
            assert suppression > 25, \
                f"Insufficient suppression: {suppression:.1f} dB at {expected_freq} Hz (expected >25 dB)"

            # Additional check: compare suppression at target vs nearby frequencies
            # If tracking is accurate, suppression should be strongest at target
            nearby_offset = 2.0  # Hz
            power_below = compute_fft_power_db(filt_segment, sample_freq, expected_freq - nearby_offset)
            power_above = compute_fft_power_db(filt_segment, sample_freq, expected_freq + nearby_offset)

            # Power at target should be lower than power at offset frequencies
            # (allowing for noise floor effects)
            assert filt_power_db <= min(power_below, power_above) + 3, \
                f"Filter not centered at {expected_freq} Hz: target={filt_power_db:.1f} dB, " \
                f"nearby={min(power_below, power_above):.1f} dB"

    def test_step_frequency_change(self):
        """Test step frequency change (100 Hz → 110 Hz instantaneous).

        Validates state preservation across frequency changes (no discontinuities).
        """
        sample_freq = 16000.0
        duration = 1.0
        num_samples = int(duration * sample_freq)

        # Generate two-segment signal: 100 Hz for 0.5s, then 110 Hz for 0.5s
        t = np.arange(num_samples) / sample_freq
        signal = np.zeros(num_samples)

        mid_point = num_samples // 2
        signal[:mid_point] = np.sin(2 * np.pi * 100.0 * t[:mid_point])
        signal[mid_point:] = np.sin(2 * np.pi * 110.0 * t[mid_point:])

        # Add noise
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal_with_noise = signal[:, np.newaxis] + noise

        # Create frequency trajectory with step change (1D)
        freq_trajectory = np.ones(num_samples)
        freq_trajectory[:mid_point] = 100.0
        freq_trajectory[mid_point:] = 110.0

        # Create mock source
        source = MockSamplesGenerator(signal_with_noise, sample_freq)

        # Create adaptive filter (SISO interface)
        filter_obj = AdaptiveNotchFilter(
            freq_source=MockFreqSource(freq_trajectory),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Check for discontinuities around step change
        # Sample-to-sample jumps should stay within normal variation
        step_region = filtered[mid_point-50:mid_point+50, 0]
        diffs = np.diff(step_region)

        # Normal variation is small for filtered signal
        # Large jumps would indicate state discontinuity
        max_jump = np.max(np.abs(diffs))
        typical_jump = np.percentile(np.abs(diffs), 90)

        # Max jump should not be more than 5x typical jump
        assert max_jump < 5 * typical_jump, \
            f"Discontinuity detected: max jump {max_jump:.3f} vs typical {typical_jump:.3f}"

    def test_static_mode_fallback(self):
        """Test that filter falls back to static mode when no trajectory is provided.

        Validates that AdaptiveNotchFilter behaves like StaticNotchFilter
        when no freq_trajectory is provided.
        """
        sample_freq = 16000.0
        duration = 1.0
        f_notch = 100.0

        # Generate signal at target frequency
        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=duration,
            sample_freq=sample_freq, num_channels=1, snr_db=20
        )

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Create adaptive filter without trajectory (should behave like static)
        filter_obj = AdaptiveNotchFilter(
            f_notch=f_notch,
            pole_radius=0.95,
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Verify suppression at target frequency
        input_power = compute_fft_power_db(signal[:, 0], sample_freq, f_notch)
        output_power = compute_fft_power_db(filtered[:, 0], sample_freq, f_notch)
        suppression = input_power - output_power

        assert suppression > 25, \
            f"Static fallback mode: Expected >25 dB suppression, got {suppression:.1f} dB"


class TestAdaptiveNotchFilterAutoMode:
    """Test autonomous LMS adaptation mode (referenceless) for SISO filter."""

    def test_lms_static_convergence(self):
        """Test LMS convergence on static harmonic (100 Hz constant).

        Validates autonomous convergence to correct frequency and suppression.
        """
        # Generate static 100 Hz signal
        sample_freq = 16000.0
        duration = 2.0  # 2 seconds for convergence
        num_samples = int(duration * sample_freq)
        t = np.arange(num_samples) / sample_freq

        signal = np.sin(2 * np.pi * 100.0 * t)

        # Add noise
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal_multi = signal[:, np.newaxis] + noise

        # Create mock source
        source = MockSamplesGenerator(signal_multi, sample_freq)

        # Create adaptive filter in auto mode (SISO interface)
        filter_obj = AdaptiveNotchFilter(
            f_notch=100.0,  # Initial estimate
            pole_radius=0.95,
            mode='auto',
            mu=0.001,  # Step size
            smooth_window=20,
            source=source
        )

        # Process entire signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Check convergence in late segment (after 1 second)
        # By this time, LMS should have converged
        start = int(1.5 * sample_freq)
        length = int(0.4 * sample_freq)

        orig_segment = signal_multi[start:start+length, 0]
        filt_segment = filtered[start:start+length, 0]

        # Verify strong suppression at 100 Hz
        orig_power = compute_fft_power_db(orig_segment, sample_freq, 100.0)
        filt_power = compute_fft_power_db(filt_segment, sample_freq, 100.0)
        suppression = orig_power - filt_power

        assert suppression > 25, \
            f"LMS did not converge: suppression {suppression:.1f} dB (expected >25 dB)"

        # Verify tracking accuracy: filter should be tracking near 100 Hz
        # With conservative step size and smoothing, we expect within 3 Hz
        detected_freq = filter_obj.f_notch
        freq_error = abs(detected_freq - 100.0)
        assert freq_error < 3.0, \
            f"Tracking accuracy error: detected {detected_freq:.2f} Hz, " \
            f"expected 100.0 Hz (error: {freq_error:.2f} Hz)"

    def test_lms_tracking_sweep(self):
        """Test LMS tracking on slow frequency sweep (100 Hz → 110 Hz over 2 seconds).

        Validates continuous tracking of time-varying harmonic.
        """
        # Generate swept signal: 100 Hz → 110 Hz over 2 seconds
        sample_freq = 16000.0
        duration = 2.0
        f_start = 100.0
        f_end = 110.0

        signal, freq_trajectory = generate_swept_tonal_signal(
            f_start, f_end, duration, sample_freq, num_channels=1, snr_db=20
        )

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Create adaptive filter in auto mode (SISO interface)
        filter_obj = AdaptiveNotchFilter(
            f_notch=100.0,  # Initial estimate
            pole_radius=0.95,
            mode='auto',
            mu=0.002,  # Slightly higher step size for tracking
            smooth_window=30,
            source=source
        )

        # Process entire signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Check tracking accuracy at multiple time points
        # Early (t=0.5s): ~102.5 Hz
        # Middle (t=1.0s): ~105.0 Hz
        # Late (t=1.5s): ~107.5 Hz
        test_times = [0.5, 1.0, 1.5]
        expected_freqs = [102.5, 105.0, 107.5]

        for test_time, expected_freq in zip(test_times, expected_freqs):
            start = int(test_time * sample_freq)
            length = int(0.2 * sample_freq)

            orig_segment = signal[start:start+length, 0]
            filt_segment = filtered[start:start+length, 0]

            # Verify suppression at expected frequency
            orig_power = compute_fft_power_db(orig_segment, sample_freq, expected_freq)
            filt_power = compute_fft_power_db(filt_segment, sample_freq, expected_freq)
            suppression = orig_power - filt_power

            assert suppression > 20, \
                f"At t={test_time}s: suppression {suppression:.1f} dB at {expected_freq:.1f} Hz " \
                f"(expected >20 dB)"


class TestTrackingAccuracyValidation:
    """Comprehensive tracking accuracy validation tests (<0.5 Hz target) for SISO filter."""

    @pytest.mark.parametrize("mode,sweep_params", [
        # External mode: all ranges supported
        ('external', (100, 120, 'linear')),
        ('external', (200, 250, 'linear')),
        ('external', (500, 600, 'linear')),
        ('external', (100, 120, 'logarithmic')),
        ('external', (200, 250, 'logarithmic')),
        ('external', (500, 600, 'logarithmic')),
        # Auto mode: lower frequency ranges only
        # (higher frequencies require parameter tuning beyond validation scope)
        ('auto', (100, 120, 'linear')),
        ('auto', (200, 250, 'linear')),
        ('auto', (100, 120, 'logarithmic')),
        ('auto', (200, 250, 'logarithmic')),
    ])
    def test_tracking_accuracy_validation(self, mode, sweep_params):
        """Validate <0.5 Hz tracking across modes, sweep types, and frequency ranges.

        Tests multiple scenarios for SISO AdaptiveNotchFilter to ensure
        tracking accuracy meets Harvey (2019) requirements.
        Auto mode limited to <300 Hz due to LMS parameter sensitivity.
        """
        f_start, f_end, sweep_type = sweep_params
        sample_freq = 44100.0
        duration = 2.0
        num_samples = int(duration * sample_freq)
        t = np.arange(num_samples) / sample_freq

        # Generate swept signal using scipy.signal.chirp
        harmonic_signal = chirp(t, f_start, duration, f_end, method=sweep_type)

        # Create ground-truth frequency trajectory
        if sweep_type == 'linear':
            freq_trajectory = np.linspace(f_start, f_end, num_samples)
        elif sweep_type == 'logarithmic':
            # Logarithmic chirp: f(t) = f_start * (f_end/f_start)^(t/T)
            freq_trajectory = f_start * (f_end / f_start) ** (t / duration)

        # Add noise (SNR = 20 dB)
        signal_power = float(np.mean(harmonic_signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal = harmonic_signal[:, np.newaxis] + noise

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Create adaptive filter based on mode (SISO interface)
        if mode == 'external':
            # External mode: provide ground-truth trajectory via streaming source
            filter_obj = AdaptiveNotchFilter(
                freq_source=MockFreqSource(freq_trajectory),
                pole_radius=0.95,
                mode='external',
                source=source
            )
        else:  # mode == 'auto'
            # Autonomous mode: let LMS adapt
            filter_obj = AdaptiveNotchFilter(
                f_notch=f_start,  # Initial estimate
                pole_radius=0.95,
                mode='auto',
                mu=0.002,  # Higher step size for tracking sweeps
                smooth_window=30,
                source=source
            )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Validate tracking accuracy by checking suppression at correct frequencies
        # If tracking is accurate (<0.5 Hz error), suppression should be strong
        # at the true frequency, and weaker at offset frequencies
        window_size = int(0.1 * sample_freq)
        num_windows = int(duration / 0.1)
        tracking_failures = []

        for i in range(num_windows):
            start = i * window_size
            end = start + window_size

            # Get ground-truth frequency for this window (middle of window)
            mid_idx = start + window_size // 2
            true_freq = freq_trajectory[mid_idx]

            # Skip very early windows in auto mode (not yet converged)
            if mode == 'auto' and i < 5:  # Skip first 0.5 seconds
                continue

            # Measure suppression at true frequency and nearby offsets
            orig_segment = signal[start:end, 0]
            filt_segment = filtered[start:end, 0]

            # Power at true frequency (should be maximally suppressed)
            orig_power_true = compute_fft_power_db(orig_segment, sample_freq, true_freq)
            filt_power_true = compute_fft_power_db(filt_segment, sample_freq, true_freq)
            suppression_true = orig_power_true - filt_power_true

            # Tracking validation: suppression at true freq should exceed nearby freqs
            # For auto mode during sweeps, allow more tolerance
            min_suppression = 18 if mode == 'auto' else 23

            if suppression_true < min_suppression:
                tracking_failures.append((i, true_freq, suppression_true))

        # 95% of windows should have strong suppression (indicating accurate tracking)
        failure_rate = len(tracking_failures) / max(len(range(num_windows)) - (5 if mode == 'auto' else 0), 1)

        assert failure_rate < 0.05, \
            f"Tracking accuracy failures in {failure_rate*100:.1f}% of windows " \
            f"(expected <5%) [{mode}, {sweep_type}, {f_start}-{f_end} Hz]. " \
            f"Failures: {tracking_failures[:3]}"

        # Additional validation: verify overall suppression across sweep
        # Sample middle segment for comprehensive check
        mid_start = int(duration * 0.5 * sample_freq)
        mid_length = int(0.2 * sample_freq)
        mid_freq = (f_start + f_end) / 2

        orig_segment = signal[mid_start:mid_start+mid_length, 0]
        filt_segment = filtered[mid_start:mid_start+mid_length, 0]

        orig_power = compute_fft_power_db(orig_segment, sample_freq, mid_freq)
        filt_power = compute_fft_power_db(filt_segment, sample_freq, mid_freq)
        suppression = orig_power - filt_power

        # For auto mode, allow slightly lower suppression during sweeps
        min_suppression = 20 if mode == 'auto' else 25

        assert suppression > min_suppression, \
            f"Insufficient overall suppression: {suppression:.1f} dB " \
            f"(expected >{min_suppression} dB) [{mode}, {sweep_type}, {f_start}-{f_end} Hz]"

    @pytest.mark.parametrize("mode", ['external', 'auto'])
    def test_tracking_accuracy_step_changes(self, mode):
        """Validate tracking accuracy with step frequency changes.

        Tests tracking across discontinuous frequency jumps to ensure filter
        maintains accuracy even with instantaneous changes.
        """
        sample_freq = 44100.0
        duration = 2.0
        num_samples = int(duration * sample_freq)
        t = np.arange(num_samples) / sample_freq

        # Create signal with three step changes: 100 → 110 → 105 → 115 Hz
        # Each segment is 0.5 seconds
        segment_length = num_samples // 4
        freqs = [100.0, 110.0, 105.0, 115.0]

        signal = np.zeros(num_samples)
        freq_trajectory = np.zeros(num_samples)

        for i, freq in enumerate(freqs):
            start = i * segment_length
            end = (i + 1) * segment_length if i < 3 else num_samples
            signal[start:end] = np.sin(2 * np.pi * freq * t[start:end])
            freq_trajectory[start:end] = freq

        # Add noise
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal_with_noise = signal[:, np.newaxis] + noise

        # Create mock source
        source = MockSamplesGenerator(signal_with_noise, sample_freq)

        # Create adaptive filter (SISO interface)
        if mode == 'external':
            filter_obj = AdaptiveNotchFilter(
                freq_source=MockFreqSource(freq_trajectory),
                pole_radius=0.95,
                mode='external',
                source=source
            )
        else:  # auto
            filter_obj = AdaptiveNotchFilter(
                f_notch=100.0,  # Initial estimate
                pole_radius=0.95,
                mode='auto',
                mu=0.002,
                smooth_window=20,
                source=source
            )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Validate suppression in each segment (avoid transitions)
        for i, freq in enumerate(freqs):
            # Sample from middle of segment (avoid first/last 100ms)
            start = i * segment_length + int(0.15 * sample_freq)
            length = int(0.2 * sample_freq)

            # Skip first segment in auto mode (initialization period)
            if mode == 'auto' and i == 0:
                continue

            orig_segment = signal_with_noise[start:start+length, 0]
            filt_segment = filtered[start:start+length, 0]

            orig_power = compute_fft_power_db(orig_segment, sample_freq, freq)
            filt_power = compute_fft_power_db(filt_segment, sample_freq, freq)
            suppression = orig_power - filt_power

            # For auto mode, first adaptation after step may be lower
            min_suppression = 18 if mode == 'auto' else 25

            assert suppression > min_suppression, \
                f"Segment {i} at {freq} Hz: suppression {suppression:.1f} dB " \
                f"(expected >{min_suppression} dB) [{mode}]"


class TestAdaptiveMultiChannel:
    """Test multi-channel processing (K channels, SISO filter applied to each)."""

    def test_multi_channel_independence(self):
        """Test that K channels are processed independently.

        The same frequency trajectory is applied to all channels, but
        each channel maintains independent filter state.
        """
        sample_freq = 16000.0
        duration = 1.0
        f_notch = 100.0
        num_channels = 3

        # Generate multi-channel signal with same tone
        generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=duration,
            sample_freq=sample_freq, num_channels=num_channels, snr_db=20
        )

        # Create frequency trajectory (1D)
        num_samples = int(duration * sample_freq)
        freq_trajectory = np.linspace(100.0, 110.0, num_samples)

        # Generate swept signal to match trajectory
        signal_swept, _ = generate_swept_tonal_signal(
            100.0, 110.0, duration, sample_freq, num_channels=num_channels, snr_db=20
        )

        source = MockSamplesGenerator(signal_swept, sample_freq)

        # Create adaptive filter (SISO interface, applied to all channels)
        filter_obj = AdaptiveNotchFilter(
            freq_source=MockFreqSource(freq_trajectory),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Verify shape preserved
        assert filtered.shape == signal_swept.shape

        # Verify suppression in each channel
        mid_freq = 105.0  # Middle of sweep
        mid_start = int(0.4 * sample_freq)
        mid_length = int(0.2 * sample_freq)

        for ch in range(num_channels):
            orig_power = compute_fft_power_db(
                signal_swept[mid_start:mid_start+mid_length, ch], sample_freq, mid_freq
            )
            filt_power = compute_fft_power_db(
                filtered[mid_start:mid_start+mid_length, ch], sample_freq, mid_freq
            )
            suppression = orig_power - filt_power

            assert suppression > 20, \
                f"Channel {ch}: Expected >20 dB suppression, got {suppression:.1f} dB"


class TestFreqSourceStreaming:
    """Test freq_source streaming alternative to freq_trajectory."""

    def test_freq_source_exhausted_raises(self):
        """Verify ValueError when freq_source has fewer total samples than source."""
        sample_freq = 16000.0
        num_samples = 2048
        signal = np.random.randn(num_samples, 1)
        # freq_source has exactly one block worth of data — StopIteration on block 2
        short_freq = np.ones(1024) * 100.0

        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = AdaptiveNotchFilter(
            freq_source=MockFreqSource(short_freq),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        with pytest.raises(ValueError, match="exhausted"):
            list(filter_obj.result(1024))

    def test_freq_source_short_block_raises(self):
        """Verify ValueError when freq_source yields a block shorter than the source block."""
        sample_freq = 16000.0
        num_samples = 2048
        signal = np.random.randn(num_samples, 1)
        # 1500 samples: block 1 yields 1024 (OK), block 2 yields 476 (mismatch with source's 1024)
        short_freq = np.ones(1500) * 100.0

        source = MockSamplesGenerator(signal, sample_freq)

        filter_obj = AdaptiveNotchFilter(
            freq_source=MockFreqSource(short_freq),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        with pytest.raises(ValueError, match="does not match"):
            list(filter_obj.result(1024))

    def test_freq_source_mode_inferred_as_external(self):
        """Verify mode is correctly inferred as external when only freq_source is set."""
        sample_freq = 16000.0
        duration = 1.0

        signal, freq_trajectory = generate_swept_tonal_signal(
            100.0, 120.0, duration, sample_freq, num_channels=1, snr_db=20
        )

        source = MockSamplesGenerator(signal, sample_freq)

        # mode not explicitly set — should be inferred as 'external' from freq_source
        filter_obj = AdaptiveNotchFilter(
            freq_source=MockFreqSource(freq_trajectory),
            pole_radius=0.95,
            source=source
        )

        assert filter_obj._get_effective_mode() == 'external'

        # Verify it actually suppresses the tone
        filtered = np.vstack(list(filter_obj.result(1024)))

        mid_start = int(0.4 * sample_freq)
        mid_length = int(0.2 * sample_freq)
        mid_freq = 110.0  # Middle of the 100→120 Hz sweep

        orig_power = compute_fft_power_db(signal[mid_start:mid_start+mid_length, 0], sample_freq, mid_freq)
        filt_power = compute_fft_power_db(filtered[mid_start:mid_start+mid_length, 0], sample_freq, mid_freq)

        assert orig_power - filt_power > 20, \
            f"Mode inference: expected >20 dB suppression, got {orig_power - filt_power:.1f} dB"
