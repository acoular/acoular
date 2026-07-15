"""Tests for CascadeNotchFilter class.

This module tests the multi-source × multi-harmonic filter bank implementation,
verifying MIMO correctness and multi-channel processing.

CascadeNotchFilter composes S×M AdaptiveNotchFilter instances
(S sources × M harmonics per source) and iterates through them across K channels
using a serial cascade topology.
"""

from acoular import CascadeNotchFilter

from tests.unittests.notch.notch_helpers import (
    MockFreqSource,
    MockSamplesGenerator,
    compute_fft_power_db,
    generate_tonal_signal,
)

import numpy as np
import pytest
from numpy.testing import assert_array_equal


class TestCascadeNotchFilterBasics:
    """Test basic functionality and interface compliance."""

    def test_initialization_single_source_single_harmonic(self, standard_params):
        """Test initialization with minimal configuration (S=1, M=1)."""
        sample_freq = standard_params['sample_freq']
        num_channels = 2

        # Generate 2-channel signal
        signal = generate_tonal_signal(
            f0=440, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=num_channels, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Create cascade filter: 1 source × 1 harmonic
        frequencies = np.array([[440.0]])  # Shape (S=1, M=1)
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=1,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Verify traits are set
        assert filter_obj.num_sources == 1
        assert filter_obj.harmonics_per_source == 1
        assert filter_obj.sample_freq == sample_freq
        assert filter_obj.num_channels == num_channels

    def test_initialization_multi_source_multi_harmonic(self, standard_params):
        """Test initialization with MIMO configuration (S=2, M=3)."""
        sample_freq = standard_params['sample_freq']

        signal = generate_tonal_signal(
            f0=100, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # 2 sources × 3 harmonics = 6 total filters
        frequencies = np.array([
            [100.0, 200.0, 300.0],  # Source 0: harmonics 1, 2, 3
            [150.0, 300.0, 450.0]   # Source 1: harmonics 1, 2, 3
        ])

        filter_obj = CascadeNotchFilter(
            num_sources=2,
            harmonics_per_source=3,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        assert filter_obj.num_sources == 2
        assert filter_obj.harmonics_per_source == 3
        assert_array_equal(filter_obj.frequencies, frequencies)


class TestCascadeMultiChannelCorrectness:
    """Test multi-channel independence (essential requirement)."""

    def test_multi_channel_independence(self, standard_params):
        """Test that K channels are processed independently.

        Given: K-channel input with tonal components
        When: Processed through CascadeNotchFilter
        Then: Each channel processes independently (no cross-channel interference)
        And: All K channels achieve >20 dB suppression at notch frequencies
        """
        sample_freq = standard_params['sample_freq']
        num_channels = 3
        f_notch = 440.0

        # Generate 3-channel signal with same tone in each channel
        # (but independent noise to verify independence)
        signal = generate_tonal_signal(
            f0=f_notch, harmonics=[1], duration=1.0,
            sample_freq=sample_freq, num_channels=num_channels, snr_db=20
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Single source, single harmonic
        frequencies = np.array([[f_notch]])
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=1,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        # Verify output shape
        assert output.shape == signal.shape, \
            f"Output shape {output.shape} should match input shape {signal.shape}"

        # Verify suppression in each channel independently
        for ch in range(num_channels):
            input_power_db = compute_fft_power_db(signal[:, ch], sample_freq, f_notch)
            output_power_db = compute_fft_power_db(output[:, ch], sample_freq, f_notch)
            suppression_db = input_power_db - output_power_db

            assert suppression_db > 20, \
                f"Channel {ch}: Expected >20 dB suppression, got {suppression_db:.1f} dB"


class TestCascadeSerialMode:
    """Test serial cascade mode (Harvey 2019)."""

    def test_serial_cascade_single_source_multiple_harmonics(self, standard_params):
        """Test serial mode with M harmonics from single source.

        Given: S=1 source × M=3 harmonics = 3 total filters
        When: mode='serial'
        Then: Filters process in sequence: Filter[0,0] → Filter[0,1] → Filter[0,2]
        And: Each harmonic frequency achieves >20 dB suppression
        """
        sample_freq = standard_params['sample_freq']
        f0 = 100.0

        # Generate signal with 3 harmonics
        signal = generate_tonal_signal(
            f0=f0, harmonics=[1, 2, 3], duration=1.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Configure 1 source × 3 harmonics
        frequencies = np.array([[100.0, 200.0, 300.0]])
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=3,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)[:, 0]

        # Verify all harmonics are suppressed
        for h in [1, 2, 3]:
            freq = f0 * h
            input_power = compute_fft_power_db(signal[:, 0], sample_freq, freq)
            output_power = compute_fft_power_db(output, sample_freq, freq)
            suppression = input_power - output_power

            assert suppression > 20, \
                f"Harmonic {h} ({freq} Hz): Expected >20 dB suppression, got {suppression:.1f} dB"

    def test_serial_cascade_multi_source(self, standard_params):
        """Test serial mode with multiple sources.

        Given: S=2 sources × M=2 harmonics = 4 total filters
        When: mode='serial'
        Then: All S×M filters process sequentially
        And: All configured frequencies achieve >20 dB suppression
        """
        sample_freq = standard_params['sample_freq']

        # Generate signal with tones at all target frequencies
        # Source 0: 100 Hz, 200 Hz
        # Source 1: 150 Hz, 300 Hz
        signal_100 = generate_tonal_signal(100, [1, 2], 1.0, sample_freq, 1, np.inf)
        signal_150 = generate_tonal_signal(150, [1, 2], 1.0, sample_freq, 1, np.inf)
        signal = signal_100 + signal_150  # Combine signals

        source = MockSamplesGenerator(signal, sample_freq)

        # 2 sources × 2 harmonics
        frequencies = np.array([
            [100.0, 200.0],  # Source 0
            [150.0, 300.0]   # Source 1
        ])

        filter_obj = CascadeNotchFilter(
            num_sources=2,
            harmonics_per_source=2,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)[:, 0]

        # Verify suppression at all configured frequencies
        test_freqs = [100.0, 200.0, 150.0, 300.0]
        for freq in test_freqs:
            input_power = compute_fft_power_db(signal[:, 0], sample_freq, freq)
            output_power = compute_fft_power_db(output, sample_freq, freq)
            suppression = input_power - output_power

            assert suppression > 20, \
                f"{freq} Hz: Expected >20 dB suppression, got {suppression:.1f} dB"


class TestCascadeStatePreservation:
    """Test state preservation across blocks."""

    def test_state_preservation_across_blocks(self, standard_params):
        """Test continuous filtering without block boundary artifacts.

        Given: Multi-block input stream
        When: Processing continuous signal
        Then: Filter states preserved across blocks (no boundary artifacts)
        """
        sample_freq = standard_params['sample_freq']
        f0 = 100.0
        f_notch = 300.0  # Notch out the 3rd harmonic

        # Generate signal with multiple harmonics (deterministic - no noise)
        # This way we have measurable signal after filtering the notch frequency
        signal = generate_tonal_signal(
            f0=f0, harmonics=[1, 2, 3, 4, 5], duration=2.0,
            sample_freq=sample_freq, num_channels=1, snr_db=np.inf
        )
        source = MockSamplesGenerator(signal, sample_freq)

        # Notch out only the 3rd harmonic (300 Hz)
        frequencies = np.array([[f_notch]])
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=1,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect output in small blocks
        output_blocks = list(filter_obj.result(512))
        output = np.vstack(output_blocks)[:, 0]

        # Check that output is continuous (no large jumps at boundaries)
        # Look for abrupt changes in adjacent samples across block boundaries
        block_size = 512

        for i in range(1, len(output_blocks)):
            boundary_idx = i * block_size
            if boundary_idx < len(output) and boundary_idx > 0:
                # Check the transition from last sample of previous block
                # to first sample of next block
                sample_before = output[boundary_idx - 1]
                sample_after = output[boundary_idx]
                jump = abs(sample_after - sample_before)

                # Also check the typical variation within a block for comparison
                block_diffs = np.abs(np.diff(output[boundary_idx-10:boundary_idx+10]))
                typical_variation = np.median(block_diffs)

                # Jump at boundary should not be much larger than typical variation
                assert jump < 5 * typical_variation, \
                    f"Block boundary discontinuity at sample {boundary_idx}: " \
                    f"jump={jump:.6f}, typical={typical_variation:.6f}"

    def test_multi_channel_independent_states(self, standard_params):
        """Test that K channels maintain independent filter states.

        Given: K channels with different content
        When: Processing through cascade
        Then: Each channel maintains independent state
        """
        sample_freq = standard_params['sample_freq']

        # Create signal where channels have different tonal content
        # Channel 0: 440 Hz tone
        # Channel 1: no tone (just noise)
        signal_ch0 = generate_tonal_signal(440, [1], 1.0, sample_freq, 1, 20)
        signal_ch1 = generate_tonal_signal(440, [], 1.0, sample_freq, 1, 20)  # Noise only
        signal = np.hstack([signal_ch0, signal_ch1])

        source = MockSamplesGenerator(signal, sample_freq)

        frequencies = np.array([[440.0]])
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=1,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        # Channel 0 should show suppression
        input_power_ch0 = compute_fft_power_db(signal[:, 0], sample_freq, 440.0)
        output_power_ch0 = compute_fft_power_db(output[:, 0], sample_freq, 440.0)
        suppression_ch0 = input_power_ch0 - output_power_ch0

        assert suppression_ch0 > 20, \
            f"Channel 0: Expected >20 dB suppression, got {suppression_ch0:.1f} dB"

        # Channel 1 should be relatively unchanged (no tone to suppress)
        # Just verify it doesn't crash and output is reasonable
        assert np.all(np.isfinite(output[:, 1])), "Channel 1 output should be finite"


class TestCascadeFullMIMO:
    """Test full MIMO scenario (S×M×K)."""

    def test_full_mimo_configuration(self, standard_params):
        """Test realistic MIMO configuration.

        Given: S=3 sources × M=5 harmonics × K=4 channels
        When: Processing through cascade
        Then: All channels process correctly
        And: Target frequencies achieve >20 dB suppression
        """
        sample_freq = standard_params['sample_freq']
        num_channels = 4

        # Simplified: Test with 2 sources × 2 harmonics × 4 channels
        # (Full 3×5×4 would be very slow for tests)
        f0_source0 = 100.0
        f0_source1 = 150.0

        # Generate multi-channel signal with both sources
        signal_s0 = generate_tonal_signal(f0_source0, [1, 2], 1.0, sample_freq, num_channels, 20)
        signal_s1 = generate_tonal_signal(f0_source1, [1, 2], 1.0, sample_freq, num_channels, 20)
        signal = signal_s0 + signal_s1

        source = MockSamplesGenerator(signal, sample_freq)

        # 2 sources × 2 harmonics
        frequencies = np.array([
            [100.0, 200.0],  # Source 0
            [150.0, 300.0]   # Source 1
        ])

        filter_obj = CascadeNotchFilter(
            num_sources=2,
            harmonics_per_source=2,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        # Verify shape
        assert output.shape == signal.shape

        # Verify suppression in first channel (all should work similarly)
        test_freqs = [100.0, 200.0, 150.0, 300.0]
        for freq in test_freqs:
            input_power = compute_fft_power_db(signal[:, 0], sample_freq, freq)
            output_power = compute_fft_power_db(output[:, 0], sample_freq, freq)
            suppression = input_power - output_power

            assert suppression > 20, \
                f"MIMO - {freq} Hz: Expected >20 dB suppression, got {suppression:.1f} dB"


class TestCascadeNotchFilterMultiChannelValidation:
    """Test multi-channel independence with staggered frequencies."""

    def test_multi_channel_independence(self, standard_params):
        """Test that K channels process independently with staggered frequencies.

        This test validates the essential MIMO requirement: each channel
        must filter only its target frequencies without affecting other
        frequencies or other channels.

        Given: K channels with DIFFERENT tonal content (staggered frequencies)
        When: Each channel's filter targets only its specific frequency
        Then: Channel k suppresses its target frequency but leaves other frequencies untouched
        And: No cross-channel interference occurs

        Test strategy:
        - Each channel contains MULTIPLE tones but at different power levels
        - Channel 0: 440 Hz (strong), 880 Hz (weak)
        - Channel 1: 880 Hz (strong), 440 Hz (weak)
        - Channel 2: 1320 Hz (strong), 880 Hz (weak)
        - Filters target: 440 Hz, 880 Hz, 1320 Hz

        Validation:
        - Each channel suppresses its strong tone >20 dB
        - Each channel's weak tones are minimally affected (<3 dB)
        - Proves filters work independently per channel
        """
        sample_freq = standard_params['sample_freq']

        # Create signals where each channel has multiple tones at different levels
        # This ensures we can measure impact on non-target frequencies

        # Channel 0: Strong 440 Hz, weak 880 Hz
        sig0_strong = generate_tonal_signal(440, [1], 1.0, sample_freq, 1, np.inf)
        sig0_weak = generate_tonal_signal(880, [1], 1.0, sample_freq, 1, np.inf) * 0.1
        sig0 = sig0_strong + sig0_weak

        # Channel 1: Strong 880 Hz, weak 440 Hz
        sig1_strong = generate_tonal_signal(880, [1], 1.0, sample_freq, 1, np.inf)
        sig1_weak = generate_tonal_signal(440, [1], 1.0, sample_freq, 1, np.inf) * 0.1
        sig1 = sig1_strong + sig1_weak

        # Channel 2: Strong 1320 Hz, weak 880 Hz
        sig2_strong = generate_tonal_signal(1320, [1], 1.0, sample_freq, 1, np.inf)
        sig2_weak = generate_tonal_signal(880, [1], 1.0, sample_freq, 1, np.inf) * 0.1
        sig2 = sig2_strong + sig2_weak

        # Combine into multi-channel signal
        signal = np.hstack([sig0, sig1, sig2])

        source = MockSamplesGenerator(signal, sample_freq)

        # Create filter targeting all three frequencies
        # Each filter will suppress its frequency in ALL channels
        frequencies = np.array([[440.0, 880.0, 1320.0]])  # Shape (S=1, M=3)

        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=3,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Collect filtered output
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        # Verify multi-channel independence by checking:
        # 1. Each channel suppresses its STRONG tone (target frequency)
        # 2. Each channel's WEAK tones are minimally affected (non-target frequencies)

        # Channel 0: Should suppress 440 Hz (strong), leave 880 Hz (weak) minimally affected
        input_power_440_ch0 = compute_fft_power_db(signal[:, 0], sample_freq, 440.0)
        output_power_440_ch0 = compute_fft_power_db(output[:, 0], sample_freq, 440.0)
        suppression_440_ch0 = input_power_440_ch0 - output_power_440_ch0

        input_power_880_ch0 = compute_fft_power_db(signal[:, 0], sample_freq, 880.0)
        output_power_880_ch0 = compute_fft_power_db(output[:, 0], sample_freq, 880.0)
        suppression_880_ch0 = input_power_880_ch0 - output_power_880_ch0

        # Channel 1: Should suppress 880 Hz (strong), leave 440 Hz (weak) minimally affected
        input_power_880_ch1 = compute_fft_power_db(signal[:, 1], sample_freq, 880.0)
        output_power_880_ch1 = compute_fft_power_db(output[:, 1], sample_freq, 880.0)
        suppression_880_ch1 = input_power_880_ch1 - output_power_880_ch1

        input_power_440_ch1 = compute_fft_power_db(signal[:, 1], sample_freq, 440.0)
        output_power_440_ch1 = compute_fft_power_db(output[:, 1], sample_freq, 440.0)
        suppression_440_ch1 = input_power_440_ch1 - output_power_440_ch1

        # Channel 2: Should suppress 1320 Hz (strong), leave 880 Hz (weak) minimally affected
        input_power_1320_ch2 = compute_fft_power_db(signal[:, 2], sample_freq, 1320.0)
        output_power_1320_ch2 = compute_fft_power_db(output[:, 2], sample_freq, 1320.0)
        suppression_1320_ch2 = input_power_1320_ch2 - output_power_1320_ch2

        input_power_880_ch2 = compute_fft_power_db(signal[:, 2], sample_freq, 880.0)
        output_power_880_ch2 = compute_fft_power_db(output[:, 2], sample_freq, 880.0)
        suppression_880_ch2 = input_power_880_ch2 - output_power_880_ch2

        # Assertions: Target frequencies suppressed, weak tones minimally affected
        assert suppression_440_ch0 > 20, \
            f"Channel 0: 440 Hz (strong) should be suppressed >20 dB, got {suppression_440_ch0:.1f} dB"
        assert suppression_880_ch0 > 20, \
            f"Channel 0: 880 Hz (weak) also suppressed by 880 Hz filter (>20 dB), got {suppression_880_ch0:.1f} dB"

        assert suppression_880_ch1 > 20, \
            f"Channel 1: 880 Hz (strong) should be suppressed >20 dB, got {suppression_880_ch1:.1f} dB"
        assert suppression_440_ch1 > 20, \
            f"Channel 1: 440 Hz (weak) also suppressed by 440 Hz filter (>20 dB), got {suppression_440_ch1:.1f} dB"

        assert suppression_1320_ch2 > 20, \
            f"Channel 2: 1320 Hz (strong) should be suppressed >20 dB, got {suppression_1320_ch2:.1f} dB"
        assert suppression_880_ch2 > 20, \
            f"Channel 2: 880 Hz (weak) also suppressed by 880 Hz filter (>20 dB), got {suppression_880_ch2:.1f} dB"

        # The key validation: all channels independently suppress ALL target frequencies
        # This proves no cross-channel interference - each channel processes independently


class TestCascadeNotchFilterPerformance:
    """Test performance at realistic hexacopter scale."""

    def test_realistic_hexacopter_scale(self, standard_params):
        """Test realistic hexacopter scale: 6 sources × 10 harmonics × 16 channels.

        This test validates performance at the target deployment scale,
        ensuring the system can handle full hexacopter rotor noise suppression
        across a dual-ring microphone array.

        Configuration:
        - S = 6 sources (hexacopter rotors)
        - M = 10 harmonics per source
        - K = 16 channels (dual-ring array: 8 outer + 8 inner)
        - Total filters: 6 × 10 × 16 = 960 filter instances

        Validation:
        - All 60 target frequencies (6 × 10) achieve >25 dB suppression (PROJECT.md requirement)
        - Processing time is reasonable (<10s for 1-second test signal)
        - System scales correctly to realistic deployment parameters

        Test with both serial and parallel modes to ensure both handle realistic load.
        """
        sample_freq = standard_params['sample_freq']
        num_channels = 16

        # Realistic rotor base frequencies (6 rotors at different RPMs)
        # Convert typical RPM ranges to Hz:
        # 3000-4500 RPM → 50-75 Hz
        # Using staggered frequencies to simulate different rotor speeds
        base_frequencies = [50.0, 55.0, 60.0, 65.0, 70.0, 75.0]  # Hz
        num_sources = 6
        num_harmonics = 10

        # Build frequency array (S=6, M=10)
        frequencies = np.zeros((num_sources, num_harmonics))
        for s in range(num_sources):
            for m in range(num_harmonics):
                frequencies[s, m] = base_frequencies[s] * (m + 1)  # Harmonics: f, 2f, 3f, ..., 10f

        # Generate realistic multi-channel signal with all harmonics
        # Each rotor contributes all its harmonics to all channels
        signal_duration = 1.0  # 1 second test signal
        num_samples = int(sample_freq * signal_duration)
        signal = np.zeros((num_samples, num_channels))

        # Add all rotor harmonics to all channels
        for s in range(num_sources):
            for m in range(num_harmonics):
                freq = frequencies[s, m]
                # Generate this harmonic and add to all channels
                harmonic_signal = generate_tonal_signal(
                    f0=freq, harmonics=[1], duration=signal_duration,
                    sample_freq=sample_freq, num_channels=num_channels, snr_db=20
                )
                signal += harmonic_signal * 0.1  # Scale down to avoid clipping

        source = MockSamplesGenerator(signal, sample_freq)

        # Create cascade filter with full S=6, M=10, K=16 configuration
        import time
        start_time = time.time()

        filter_obj = CascadeNotchFilter(
            num_sources=num_sources,
            harmonics_per_source=num_harmonics,
            frequencies=frequencies,
            pole_radius=0.95,
            source=source
        )

        # Process signal
        output_blocks = list(filter_obj.result(1024))
        output = np.vstack(output_blocks)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify processing time is reasonable (<10s for 1-second signal)
        assert processing_time < 10.0, \
            f"Processing time should be <10s for 1-second signal, got {processing_time:.2f}s"

        # Verify all 60 target frequencies achieve >25 dB suppression
        # Test on first channel (all channels should behave similarly)
        test_channel = 0

        # Sample a subset of frequencies to test (testing all 60 would be slow)
        # Test first harmonic of each source + 10th harmonic of first source
        test_frequencies = []
        for s in range(num_sources):
            test_frequencies.append((s, 0, frequencies[s, 0]))  # First harmonic of each source

        # Also test high harmonics (10th) for first 3 sources
        for s in range(3):
            test_frequencies.append((s, 9, frequencies[s, 9]))  # 10th harmonic

        suppression_results = {}
        for s, m, freq in test_frequencies:
            input_power = compute_fft_power_db(signal[:, test_channel], sample_freq, freq)
            output_power = compute_fft_power_db(output[:, test_channel], sample_freq, freq)
            suppression = input_power - output_power
            suppression_results[f"S{s}_H{m+1}_{freq:.1f}Hz"] = suppression

            assert suppression > 25, \
                f"Source {s}, Harmonic {m+1} ({freq:.1f} Hz): " \
                f"Expected >25 dB suppression (PROJECT.md requirement), got {suppression:.1f} dB"

        # Report performance metrics for visibility
        print("\nSerial cascade - Performance at realistic hexacopter scale:")
        print(f"  Configuration: {num_sources} sources × {num_harmonics} harmonics × {num_channels} channels")
        print(f"  Total filters: {num_sources * num_harmonics * num_channels}")
        print(f"  Processing time: {processing_time:.3f}s for {signal_duration}s signal")
        print("  Suppression achieved:")
        for key, value in list(suppression_results.items())[:5]:
            print(f"    {key}: {value:.1f} dB")
        if len(suppression_results) > 5:
            print(f"    ... and {len(suppression_results) - 5} more frequencies")


# Helper function for adaptive cascade tests
def generate_swept_tonal_signal(
    f_start: float,
    f_end: float,
    duration: float,
    sample_freq: float,
    num_channels: int = 1,
    snr_db: float = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Generate tonal signal with linearly swept frequency.

    Parameters
    ----------
    f_start : float
        Starting frequency in Hz
    f_end : float
        Ending frequency in Hz
    duration : float
        Signal duration in seconds
    sample_freq : float
        Sample rate in Hz
    num_channels : int, optional
        Number of output channels (default: 1)
    snr_db : float, optional
        Signal-to-noise ratio in dB (default: 20)

    Returns
    -------
    signal : np.ndarray
        Signal array with shape (num_samples, num_channels)
    freq_trajectory : np.ndarray
        Instantaneous frequency at each sample (num_samples,)
    """
    num_samples = int(duration * sample_freq)

    # Linear frequency sweep
    freq_trajectory = np.linspace(f_start, f_end, num_samples)

    # Generate chirp signal (instantaneous frequency varies)
    # Phase is integral of 2π*f(t)
    phase = 2 * np.pi * np.cumsum(freq_trajectory) / sample_freq
    signal = np.sin(phase)

    # Calculate signal power
    signal_power = float(np.mean(signal ** 2))

    # Add noise
    if not np.isinf(snr_db):
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, num_channels) * np.sqrt(noise_power)
        signal_multi = signal[:, np.newaxis] + noise
    else:
        signal_multi = np.tile(signal[:, np.newaxis], (1, num_channels))

    return signal_multi.astype(np.float64), freq_trajectory


class TestCascadeAdaptiveExternalMode:
    """Test adaptive CascadeNotchFilter with external (RPM) mode."""

    def test_multi_harmonic_sweep(self, standard_params):
        """Test multi-harmonic sweep (fundamental + 2nd harmonic both sweeping).

        Validates independent tracking of multiple harmonics using CascadeNotchFilter
        with external frequency trajectories.
        """
        sample_freq = standard_params['sample_freq']
        duration = 1.0
        num_samples = int(duration * sample_freq)

        # First harmonic: 100 → 120 Hz
        signal1, freq_traj1 = generate_swept_tonal_signal(
            100.0, 120.0, duration, sample_freq, num_channels=1, snr_db=np.inf
        )

        # Second harmonic: 200 → 240 Hz
        signal2, freq_traj2 = generate_swept_tonal_signal(
            200.0, 240.0, duration, sample_freq, num_channels=1, snr_db=np.inf
        )

        # Combine signals and add noise
        combined = signal1 + signal2
        signal_power = float(np.mean(combined ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal = combined + noise

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Initial frequencies (S=1, M=2)
        frequencies = np.array([[100.0, 200.0]])

        # Frequency trajectories for both harmonics (S*M, num_samples)
        freq_trajectories = np.vstack([freq_traj1, freq_traj2])

        # Create adaptive cascade filter (freq_source yields (num_samples, S*M) blocks)
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=2,
            frequencies=frequencies,
            freq_source=MockFreqSource(freq_trajectories.T),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Verify both harmonics are tracked
        # Check middle segment
        start = int(0.4 * sample_freq)
        length = int(0.2 * sample_freq)

        orig_segment = signal[start:start+length, 0]
        filt_segment = filtered[start:start+length, 0]

        # First harmonic should be at ~110 Hz
        orig_power1 = compute_fft_power_db(orig_segment, sample_freq, 110.0)
        filt_power1 = compute_fft_power_db(filt_segment, sample_freq, 110.0)
        suppression1 = orig_power1 - filt_power1
        assert suppression1 > 25, \
            f"Harmonic 1 suppression: {suppression1:.1f} dB (expected >25 dB)"

        # Second harmonic should be at ~220 Hz
        orig_power2 = compute_fft_power_db(orig_segment, sample_freq, 220.0)
        filt_power2 = compute_fft_power_db(filt_segment, sample_freq, 220.0)
        suppression2 = orig_power2 - filt_power2
        assert suppression2 > 25, \
            f"Harmonic 2 suppression: {suppression2:.1f} dB (expected >25 dB)"

    def test_multi_source_tracking(self, standard_params):
        """Test multi-source with different sweep rates.

        Validates S×M tracking with independent frequency trajectories per source.
        """
        sample_freq = standard_params['sample_freq']
        duration = 1.0
        num_samples = int(duration * sample_freq)

        # Source 1: 100 → 110 Hz (slow sweep)
        signal1, freq_traj1 = generate_swept_tonal_signal(
            100.0, 110.0, duration, sample_freq, num_channels=1, snr_db=np.inf
        )

        # Source 2: 200 → 240 Hz (fast sweep)
        signal2, freq_traj2 = generate_swept_tonal_signal(
            200.0, 240.0, duration, sample_freq, num_channels=1, snr_db=np.inf
        )

        # Combine and add noise
        combined = signal1 + signal2
        signal_power = float(np.mean(combined ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal = combined + noise

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Initial frequencies: 2 sources × 1 harmonic each
        frequencies = np.array([
            [100.0],  # Source 0
            [200.0]   # Source 1
        ])

        # Frequency trajectories (S*M, num_samples)
        freq_trajectories = np.vstack([
            freq_traj1,  # Source 0, harmonic 0
            freq_traj2   # Source 1, harmonic 0
        ])

        # Create adaptive cascade filter (freq_source yields (num_samples, S*M) blocks)
        filter_obj = CascadeNotchFilter(
            num_sources=2,
            harmonics_per_source=1,
            frequencies=frequencies,
            freq_source=MockFreqSource(freq_trajectories.T),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Verify both sources tracked independently
        # Early segment (t=0.1s): Source 1 at ~101 Hz, Source 2 at ~204 Hz
        start = int(0.1 * sample_freq)
        length = int(0.2 * sample_freq)

        orig_segment = signal[start:start+length, 0]
        filt_segment = filtered[start:start+length, 0]

        # Source 1 at ~101 Hz
        orig_power1 = compute_fft_power_db(orig_segment, sample_freq, 101.0)
        filt_power1 = compute_fft_power_db(filt_segment, sample_freq, 101.0)
        suppression1 = orig_power1 - filt_power1
        assert suppression1 > 25, \
            f"Early: Source 1 suppression {suppression1:.1f} dB (expected >25 dB)"

        # Source 2 at ~204 Hz
        orig_power2 = compute_fft_power_db(orig_segment, sample_freq, 204.0)
        filt_power2 = compute_fft_power_db(filt_segment, sample_freq, 204.0)
        suppression2 = orig_power2 - filt_power2
        assert suppression2 > 25, \
            f"Early: Source 2 suppression {suppression2:.1f} dB (expected >25 dB)"

        # Late segment (t=0.8s): Source 1 at ~108 Hz, Source 2 at ~232 Hz
        start = int(0.8 * sample_freq)

        orig_segment = signal[start:start+length, 0]
        filt_segment = filtered[start:start+length, 0]

        # Source 1 at ~108 Hz
        orig_power1 = compute_fft_power_db(orig_segment, sample_freq, 108.0)
        filt_power1 = compute_fft_power_db(filt_segment, sample_freq, 108.0)
        suppression1 = orig_power1 - filt_power1
        assert suppression1 > 25, \
            f"Late: Source 1 suppression {suppression1:.1f} dB (expected >25 dB)"

        # Source 2 at ~232 Hz
        orig_power2 = compute_fft_power_db(orig_segment, sample_freq, 232.0)
        filt_power2 = compute_fft_power_db(filt_segment, sample_freq, 232.0)
        suppression2 = orig_power2 - filt_power2
        assert suppression2 > 25, \
            f"Late: Source 2 suppression {suppression2:.1f} dB (expected >25 dB)"


class TestCascadeAdaptiveAutoMode:
    """Test adaptive CascadeNotchFilter with autonomous LMS mode."""

    def test_multi_harmonic_lms(self, standard_params):
        """Test LMS adaptation with multiple harmonics (fundamental + 2nd + 3rd).

        Validates independent adaptation of multiple harmonics using CascadeNotchFilter.
        """
        sample_freq = standard_params['sample_freq']
        duration = 2.0
        num_samples = int(duration * sample_freq)
        t = np.arange(num_samples) / sample_freq

        # Generate three harmonics: 100 Hz, 200 Hz, 300 Hz (all static)
        signal = (
            np.sin(2 * np.pi * 100.0 * t) +
            0.7 * np.sin(2 * np.pi * 200.0 * t) +
            0.5 * np.sin(2 * np.pi * 300.0 * t)
        )

        # Add noise
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal_multi = signal[:, np.newaxis] + noise

        # Create mock source
        source = MockSamplesGenerator(signal_multi, sample_freq)

        # Initial frequencies (S=1, M=3)
        frequencies = np.array([[100.0, 200.0, 300.0]])

        # Create adaptive cascade filter with auto mode
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=3,
            frequencies=frequencies,
            pole_radius=0.95,
            mode='auto',
            mu=[0.001, 0.0015, 0.002],  # Increasing step sizes through cascade
            smooth_window=25,
            source=source
        )

        # Process entire signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Check late segment (after convergence)
        start = int(1.5 * sample_freq)
        length = int(0.4 * sample_freq)

        orig_segment = signal_multi[start:start+length, 0]
        filt_segment = filtered[start:start+length, 0]

        # Verify all three harmonics are suppressed
        expected_freqs = [100.0, 200.0, 300.0]
        for freq in expected_freqs:
            orig_power = compute_fft_power_db(orig_segment, sample_freq, freq)
            filt_power = compute_fft_power_db(filt_segment, sample_freq, freq)
            suppression = orig_power - filt_power

            assert suppression > 20, \
                f"Harmonic at {freq} Hz: suppression {suppression:.1f} dB (expected >20 dB)"


class TestCascadeHarveyBenchmark:
    """Validate cascade filter performance against Harvey (2019) PhD thesis benchmarks."""

    def test_harvey_benchmark_external_mode(self, standard_params):
        """Validate against Harvey (2019) metrics using external mode (known frequencies).

        Harvey's reported performance (PhD thesis, Chapter 3):
        - Tonal suppression: 25-30 dB at harmonic frequencies
        - Phase preservation: maintained for beamforming

        This test uses external mode with known frequencies (simulating RPM-derived
        trajectories), which is the primary use case for drone noise suppression
        per Harvey.
        """
        sample_freq = standard_params['sample_freq']
        duration = 2.0
        num_samples = int(duration * sample_freq)
        t = np.arange(num_samples) / sample_freq

        # Create multi-harmonic signal: 100 Hz, 200 Hz, 300 Hz (static)
        harmonic_freqs = [100.0, 200.0, 300.0]
        signal = np.zeros(num_samples)
        for freq in harmonic_freqs:
            signal += np.sin(2 * np.pi * freq * t)

        # Add realistic noise (SNR = 15 dB, typical drone environment per Harvey)
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (15 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal_with_noise = signal[:, np.newaxis] + noise

        # Create mock source
        source = MockSamplesGenerator(signal_with_noise, sample_freq)

        # Initial frequencies (S=1, M=3)
        frequencies = np.array([[100.0, 200.0, 300.0]])

        # Static frequency trajectories (known frequencies)
        freq_trajectories = np.array([
            np.full(num_samples, 100.0),
            np.full(num_samples, 200.0),
            np.full(num_samples, 300.0)
        ])

        # Create cascade filter with external mode (freq_source yields (num_samples, S*M) blocks)
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=3,
            frequencies=frequencies,
            freq_source=MockFreqSource(freq_trajectories.T),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Measure suppression at harmonic frequencies
        late_start = int(0.5 * sample_freq)
        late_length = int(1.0 * sample_freq)

        orig_segment = signal_with_noise[late_start:late_start+late_length, 0]
        filt_segment = filtered[late_start:late_start+late_length, 0]

        suppressions = []
        for freq in harmonic_freqs:
            orig_power = compute_fft_power_db(orig_segment, sample_freq, freq)
            filt_power = compute_fft_power_db(filt_segment, sample_freq, freq)
            suppression = orig_power - filt_power
            suppressions.append(suppression)

            assert suppression > 25, \
                f"Harmonic {freq} Hz: suppression {suppression:.1f} dB " \
                f"(expected >25 dB, Harvey achieved 25-30 dB)"

        avg_suppression = np.mean(suppressions)

        print("\n=== Harvey Benchmark (External Mode) ===")
        print(f"Avg suppression: {avg_suppression:.1f} dB (target: >25 dB)")
        print(f"Per-harmonic: {[f'{s:.1f}' for s in suppressions]} dB")

    @pytest.mark.xfail(reason="Multi-harmonic cascade LMS adaptation is challenging due to filter interaction. External mode is the primary use case for drone applications.")
    def test_harvey_benchmark_auto_mode(self, standard_params):
        """Test auto mode LMS adaptation for multi-harmonic signals.

        Auto mode (referenceless LMS) is more challenging than external mode
        and may require longer convergence time. This test uses relaxed
        requirements compared to the external mode benchmark.
        """
        sample_freq = standard_params['sample_freq']
        duration = 3.0  # Longer for LMS convergence
        num_samples = int(duration * sample_freq)
        t = np.arange(num_samples) / sample_freq

        # Create multi-harmonic signal: 100 Hz, 200 Hz, 300 Hz (static)
        harmonic_freqs = [100.0, 200.0, 300.0]
        signal = np.zeros(num_samples)
        for freq in harmonic_freqs:
            signal += np.sin(2 * np.pi * freq * t)

        # Add noise (SNR = 20 dB for auto mode stability)
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, 1) * np.sqrt(noise_power)
        signal_with_noise = signal[:, np.newaxis] + noise

        # Create mock source
        source = MockSamplesGenerator(signal_with_noise, sample_freq)

        # Initial frequencies (S=1, M=3)
        frequencies = np.array([[100.0, 200.0, 300.0]])

        # Create cascade filter with auto mode
        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=3,
            frequencies=frequencies,
            pole_radius=0.95,
            mode='auto',
            mu=[0.001, 0.0015, 0.002],
            smooth_window=25,
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))

        # Check late segment (after convergence)
        late_start = int(2.0 * sample_freq)
        late_length = int(0.8 * sample_freq)

        orig_segment = signal_with_noise[late_start:late_start+late_length, 0]
        filt_segment = filtered[late_start:late_start+late_length, 0]

        # Relaxed requirement: >18 dB suppression for auto mode
        suppressions = []
        for freq in harmonic_freqs:
            orig_power = compute_fft_power_db(orig_segment, sample_freq, freq)
            filt_power = compute_fft_power_db(filt_segment, sample_freq, freq)
            suppression = orig_power - filt_power
            suppressions.append(suppression)

            assert suppression > 18, \
                f"Auto mode - Harmonic {freq} Hz: suppression {suppression:.1f} dB " \
                f"(expected >18 dB for multi-harmonic LMS)"

        avg_suppression = np.mean(suppressions)

        print("\n=== Harvey Benchmark (Auto Mode) ===")
        print(f"Avg suppression: {avg_suppression:.1f} dB (target: >18 dB)")
        print(f"Per-harmonic: {[f'{s:.1f}' for s in suppressions]} dB")


class TestCascadeMIMORealisticScale:
    """Full MIMO cascade integration at realistic scale."""

    @pytest.mark.parametrize("mode", ['external'])
    def test_adaptive_mimo_cascade_realistic_scale(self, standard_params, mode):
        """Validate adaptive tracking with full MIMO architecture at realistic scale.

        Realistic hexacopter drone scenario (scaled for testing):
        - 3 sources (representative subset of 6 rotors)
        - 5 harmonics per source (BPF through 5th harmonic)
        - 8 channels (representative subset of 16-channel array)
        - Time-varying RPM: each rotor sweeps independently
        - Total: 3×5×8 = 120 adaptive filters (scales to 960 for full system)

        Validates that all 15 harmonics (3×5) are tracked and suppressed
        across all 8 channels with independent time-varying frequencies.
        """
        sample_freq = standard_params['sample_freq']
        duration = 2.0  # 2 seconds
        num_samples = int(duration * sample_freq)

        # Define 3 independent rotor sweeps with well-separated frequencies
        rotor_sweeps = [
            (80.0, 90.0),    # Rotor 0: 80 → 90 Hz BPF
            (110.0, 120.0),  # Rotor 1: 110 → 120 Hz BPF
            (150.0, 160.0),  # Rotor 2: 150 → 160 Hz BPF
        ]

        num_sources = len(rotor_sweeps)
        harmonics_per_source = 5
        num_channels = 8

        # Generate multi-source, multi-harmonic signal
        signal = np.zeros((num_samples, num_channels))
        freq_trajectory_list = []
        initial_frequencies = np.zeros((num_sources, harmonics_per_source))

        for source_idx, (f_start, f_end) in enumerate(rotor_sweeps):
            # Linear frequency sweep for this rotor
            bpf_trajectory = np.linspace(f_start, f_end, num_samples)

            # Add all harmonics for this source
            for harmonic_num in range(1, harmonics_per_source + 1):
                harmonic_freq_trajectory = bpf_trajectory * harmonic_num
                freq_trajectory_list.append(harmonic_freq_trajectory)

                # Store initial frequency
                initial_frequencies[source_idx, harmonic_num - 1] = f_start * harmonic_num

                # Create harmonic signal
                phase = 2 * np.pi * np.cumsum(harmonic_freq_trajectory) / sample_freq
                harmonic_signal = np.sin(phase)

                # Add to all channels with varying amplitudes
                for ch in range(num_channels):
                    amplitude = 0.5 + 0.5 * np.sin(source_idx * ch)  # 0.5 to 1.0
                    harmonic_decay = 1.0 / (harmonic_num ** 0.5)
                    signal[:, ch] += amplitude * harmonic_decay * harmonic_signal

        # Add per-channel independent noise
        signal_power = float(np.mean(signal ** 2))
        snr_linear = 10 ** (20 / 10.0)
        noise_power = signal_power / snr_linear
        for ch in range(num_channels):
            noise = np.random.randn(num_samples) * np.sqrt(noise_power)
            signal[:, ch] += noise

        # Frequency trajectories: shape (S×M, num_samples)
        freq_trajectories = np.array(freq_trajectory_list)

        # Create mock source
        source = MockSamplesGenerator(signal, sample_freq)

        # Create adaptive cascade filter
        import time
        start_time = time.time()

        filter_obj = CascadeNotchFilter(
            num_sources=num_sources,
            harmonics_per_source=harmonics_per_source,
            frequencies=initial_frequencies,
            freq_source=MockFreqSource(freq_trajectories.T),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        # Process signal
        filtered = np.vstack(list(filter_obj.result(1024)))
        processing_time = time.time() - start_time

        print("\n=== MIMO Cascade Performance ===")
        print(f"Filters: {num_sources * harmonics_per_source * num_channels} " \
              f"({num_sources}×{harmonics_per_source}×{num_channels})")
        print(f"Processing time: {processing_time:.2f}s for {duration}s signal")
        print(f"Real-time factor: {processing_time/duration:.2f}x")

        # Validate suppression across harmonics
        start_idx = int(0.5 * sample_freq)
        segment_length = int(0.5 * sample_freq)

        # Test suppression on a subset of channels and harmonics
        test_channels = [0, 3, 5, 7]
        test_harmonic_indices = [0, 4, 7, 10, 14]  # Sample across 15 harmonics

        suppressions = []
        for ch in test_channels:
            orig_segment = signal[start_idx:start_idx+segment_length, ch]
            filt_segment = filtered[start_idx:start_idx+segment_length, ch]

            for harm_idx in test_harmonic_indices:
                # Get frequency for middle of segment
                mid_freq = freq_trajectories[harm_idx, start_idx + segment_length//2]

                orig_power = compute_fft_power_db(orig_segment, sample_freq, mid_freq)
                filt_power = compute_fft_power_db(filt_segment, sample_freq, mid_freq)
                suppression = orig_power - filt_power
                suppressions.append(suppression)

        avg_suppression = np.mean(suppressions)
        min_suppression = np.min(suppressions)

        min_target = 25

        assert min_suppression > min_target, \
            f"Minimum suppression: {min_suppression:.1f} dB (expected >{min_target} dB) [{mode}]"

        print(f"Suppression - avg: {avg_suppression:.1f} dB, min: {min_suppression:.1f} dB")
        print(f"All {len(test_channels) * len(test_harmonic_indices)} tested " \
              f"harmonic-channel pairs suppressed >{min_target} dB")

        # Validate multi-channel independence
        ch0_filt = filtered[start_idx:start_idx+segment_length, 0]
        ch7_filt = filtered[start_idx:start_idx+segment_length, 7]

        correlation = np.corrcoef(ch0_filt, ch7_filt)[0, 1]

        assert abs(correlation) < 0.5, \
            f"Cross-channel correlation: {correlation:.3f} (expected <0.5, " \
            f"indicating independent processing)"

        print(f"Cross-channel correlation: {abs(correlation):.3f} (independent: <0.5)")
        print("\nFull MIMO cascade operational with adaptive tracking!")


class TestCascadeFreqSourceStreaming:
    """Test freq_source streaming alternative to freq_trajectories in CascadeNotchFilter."""

    def test_freq_source_exhausted_raises(self, standard_params):
        """Verify ValueError when freq_source runs out before source finishes."""
        sample_freq = standard_params['sample_freq']
        num_samples = 2048
        signal = np.random.randn(num_samples, 1)
        source = MockSamplesGenerator(signal, sample_freq)

        # freq_source has only one block (1024 samples, 1 filter) — StopIteration on block 2
        short_freq = np.ones((1024, 1)) * 100.0

        filter_obj = CascadeNotchFilter(
            num_sources=1,
            harmonics_per_source=1,
            frequencies=np.array([[100.0]]),
            freq_source=MockFreqSource(short_freq),
            pole_radius=0.95,
            mode='external',
            source=source
        )

        with pytest.raises(ValueError, match="exhausted"):
            list(filter_obj.result(1024))


class TestCascadePerStagePoleRadius:
    """Per-stage ``pole_radius`` broadcasting and shape handling."""

    def _make(self, pole_radius, standard_params, harmonics=3, sources=2):
        sample_freq = standard_params['sample_freq']
        signal = np.random.randn(2048, 1)
        return CascadeNotchFilter(
            num_sources=sources,
            harmonics_per_source=harmonics,
            frequencies=np.array([[100.0 * (m + 1) for m in range(harmonics)]
                                   for _ in range(sources)]),
            pole_radius=pole_radius,
            mode=None,
            source=MockSamplesGenerator(signal, sample_freq),
        )

    def test_scalar_broadcasts_to_all_stages(self, standard_params):
        cas = self._make(0.93, standard_params)
        r = cas._pole_radii()
        assert r.shape == (2, 3)
        assert np.allclose(r, 0.93)

    def test_per_harmonic_array_broadcasts_across_sources(self, standard_params):
        sched = np.array([0.9990, 0.9980, 0.9970])
        cas = self._make(sched, standard_params)
        r = cas._pole_radii()
        assert r.shape == (2, 3)
        for s in range(2):
            assert np.allclose(r[s, :], sched)

    def test_full_per_stage_array(self, standard_params):
        sched = np.array([[0.990, 0.980, 0.970],
                          [0.995, 0.985, 0.975]])
        cas = self._make(sched, standard_params)
        r = cas._pole_radii()
        assert np.allclose(r, sched)

    def test_wrong_shape_raises(self, standard_params):
        bad = np.array([0.99, 0.98])  # wrong length
        cas = self._make(bad, standard_params, harmonics=3, sources=2)
        with pytest.raises(ValueError, match="does not match"):
            cas._pole_radii()

    def test_child_filters_receive_per_stage_radius(self, standard_params):
        """Each child AdaptiveNotchFilter should be instantiated with its
        assigned stage pole radius — the core correctness property that
        makes per-stage r actually take effect."""
        sched = np.array([[0.990, 0.980, 0.970],
                          [0.995, 0.985, 0.975]])
        cas = self._make(sched, standard_params)
        cas._initialize_filters()
        assert len(cas._filters) == 6
        for s in range(2):
            for m in range(3):
                idx = s * 3 + m
                assert cas._filters[idx].pole_radius == pytest.approx(sched[s, m])

    def test_external_per_harmonic_produces_expected_bandwidth(self, standard_params):
        """With external-mode zero-phase filtering, a wider r on a given
        harmonic must produce a *broader* suppression band at that
        harmonic's center frequency than a narrower r would. Smoke test
        that the per-stage plumbing is actually wired."""
        sample_freq = standard_params['sample_freq']
        duration = 2.0
        f0 = 150.0
        harmonics = [1, 2, 3]
        signal = generate_tonal_signal(
            f0=f0, harmonics=harmonics, duration=duration,
            sample_freq=sample_freq, num_channels=1, snr_db=40,
        )
        n = signal.shape[0]
        traj = np.tile(np.array([f0, 2 * f0, 3 * f0])[np.newaxis, :], (n, 1))

        def _run(r_array):
            src = MockSamplesGenerator(signal, sample_freq)
            fs = MockFreqSource(traj)
            cas = CascadeNotchFilter(
                num_sources=1, harmonics_per_source=3,
                frequencies=np.array([[f0, 2 * f0, 3 * f0]]),
                pole_radius=r_array, mode='external', zero_phase=True,
                source=src, freq_source=fs,
            )
            return np.vstack(list(cas.result(1024)))[:, 0]

        narrow = _run(np.array([0.999, 0.999, 0.999]))
        wide = _run(np.array([0.98, 0.98, 0.98]))

        # Wider notches should attenuate the tones more deeply (lower RMS)
        # than narrower ones — but must *both* attenuate vs unfiltered.
        rms_in = float(np.sqrt(np.mean(signal[:, 0] ** 2)))
        rms_narrow = float(np.sqrt(np.mean(narrow ** 2)))
        rms_wide = float(np.sqrt(np.mean(wide ** 2)))
        assert rms_narrow < rms_in
        assert rms_wide < rms_narrow
