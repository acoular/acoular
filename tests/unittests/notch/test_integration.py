"""Integration tests for NotchFilterBase with signal generator.

This module tests that NotchFilterBase correctly integrates with signal
generators and properly propagates properties through the acoular interface.
These tests establish patterns for testing TimeOut subclasses.
"""

from acoular.base import TimeOut

from tests.unittests.notch.notch_helpers import MockSamplesGenerator, generate_tonal_signal

import numpy as np
import pytest


class ConcreteNotchFilter(TimeOut):
    """Concrete implementation of NotchFilterBase for testing.

    This minimal implementation provides a pass-through result() method
    that simply yields the source data unchanged. Used for testing the
    base class interface without actual filtering logic.
    """

    def result(self, num):
        """Pass-through generator that yields source data unchanged.

        Parameters
        ----------
        num : int
            Number of samples per block

        Yields
        ------
        np.ndarray
            Blocks of unfiltered source data
        """
        for block in self.source.result(num):
            yield block


class TestNotchFilterBaseIntegration:
    """Test suite for NotchFilterBase integration with signal generator."""

    def test_property_propagation_single_channel(self, standard_params):
        """Test that properties propagate correctly from source to base.

        Verifies that num_channels, num_samples, and sample_freq are
        properly delegated from the source to the NotchFilterBase instance.
        """
        # Create test signal
        signal = generate_tonal_signal(
            f0=standard_params['f0'],
            harmonics=[1, 2, 3],
            duration=standard_params['duration'],
            sample_freq=standard_params['sample_freq'],
            num_channels=1,
            snr_db=np.inf  # No noise for simpler test
        )

        # Create mock source with signal
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = standard_params['sample_freq']

        # Create concrete filter instance
        base = ConcreteNotchFilter(source=mock)

        # Verify properties propagate correctly
        assert base.num_channels == mock.num_channels
        assert base.num_samples == mock.num_samples
        assert base.sample_freq == mock.sample_freq

        # Verify values match expected
        expected_samples = int(standard_params['duration'] * standard_params['sample_freq'])
        assert base.num_channels == 1
        assert base.num_samples == expected_samples
        assert base.sample_freq == standard_params['sample_freq']

    def test_property_propagation_multi_channel(self, standard_params):
        """Test property propagation with multi-channel signal."""
        num_channels = 4
        signal = generate_tonal_signal(
            f0=standard_params['f0'],
            harmonics=[1, 2, 3],
            duration=standard_params['duration'],
            sample_freq=standard_params['sample_freq'],
            num_channels=num_channels,
            snr_db=np.inf
        )

        # Create mock source
        mock = MockSamplesGenerator()
        mock.set_signal(signal)
        mock.sample_freq = standard_params['sample_freq']

        # Create concrete filter instance
        base = ConcreteNotchFilter(source=mock)

        # Verify multi-channel properties
        assert base.num_channels == num_channels
        assert base.num_channels == mock.num_channels
        assert base.num_samples == mock.num_samples
        assert base.sample_freq == mock.sample_freq

    def test_abstract_result_method(self):
        """Test that result() method is abstract.

        NotchFilterBase inherits from TimeOut and should not implement
        result() directly. Attempting to instantiate NotchFilterBase
        directly should fail because it doesn't implement the abstract
        result() method.
        """
        # Create minimal mock source
        mock = MockSamplesGenerator()
        mock.set_signal(np.zeros((100, 1)))
        mock.sample_freq = 16000.0

        # Attempting to instantiate NotchFilterBase directly should fail
        # because it doesn't implement the abstract result() method
        with pytest.raises(TypeError) as exc_info:
            TimeOut(source=mock)

        # Verify the error message mentions the abstract method
        assert "abstract" in str(exc_info.value).lower()
        assert "result" in str(exc_info.value).lower()

    def test_source_update_propagates(self, standard_params):
        """Test that changing source properties updates base properties.

        Verifies that the delegate pattern correctly tracks changes
        to the source object's properties.
        """
        # Create initial signal
        signal1 = generate_tonal_signal(
            f0=100,
            harmonics=[1],
            duration=1.0,
            sample_freq=16000,
            num_channels=2,
            snr_db=np.inf
        )

        mock = MockSamplesGenerator()
        mock.set_signal(signal1)
        mock.sample_freq = 16000.0

        base = ConcreteNotchFilter(source=mock)

        # Record initial values
        initial_channels = base.num_channels
        initial_samples = base.num_samples

        assert initial_channels == 2
        assert initial_samples == 16000

        # Update source with new signal
        signal2 = generate_tonal_signal(
            f0=100,
            harmonics=[1],
            duration=0.5,
            sample_freq=16000,
            num_channels=3,
            snr_db=np.inf
        )

        mock.set_signal(signal2)

        # Properties should update via delegation
        assert base.num_channels == 3
        assert base.num_samples == 8000

    def test_different_sample_frequencies(self):
        """Test that different sample frequencies are handled correctly."""
        for sample_freq in [8000, 16000, 44100, 48000]:
            signal = generate_tonal_signal(
                f0=100,
                harmonics=[1],
                duration=0.1,
                sample_freq=sample_freq,
                num_channels=1,
                snr_db=np.inf
            )

            mock = MockSamplesGenerator()
            mock.set_signal(signal)
            mock.sample_freq = float(sample_freq)

            base = ConcreteNotchFilter(source=mock)

            assert base.sample_freq == sample_freq
            assert base.num_samples == int(0.1 * sample_freq)
