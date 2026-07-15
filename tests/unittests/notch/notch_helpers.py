"""Shared helpers, mock sources and signal generators for the notch-filter tests.

Merges the standalone ``signals.py`` and ``conftest.py`` helpers from the
original notchfilter package into a single importable module so the ported
tests can pull everything from one place.
"""

from acoular.base import SamplesGenerator

import numpy as np
from traits.api import CArray, Float

# Constants for power calculations
DB_TO_LINEAR_FACTOR = 10.0
NOISE_ONLY_POWER = 1.0


# ---------------------------------------------------------------------------
# Tonal signal generation
# ---------------------------------------------------------------------------


def generate_tonal_signal(f0, harmonics, duration, sample_freq, num_channels=1, snr_db=20):
    """Generate multi-channel tonal signal with harmonics and noise.

    Creates a time-domain signal consisting of sinusoidal tones at specified
    harmonic frequencies with additive white Gaussian noise at a given SNR.

    Parameters
    ----------
    f0 : float
        Fundamental frequency in Hz.
    harmonics : list of int
        Harmonic numbers to include (e.g. ``[1, 2, 3]`` for f0, 2*f0, 3*f0).
    duration : float
        Signal duration in seconds.
    sample_freq : float
        Sample rate in Hz.
    num_channels : int, optional
        Number of output channels (default 1).
    snr_db : float, optional
        Signal-to-noise ratio in dB (default 20). Use ``np.inf`` for no noise.

    Returns
    -------
    numpy.ndarray
        Signal array with shape ``(num_samples, num_channels)`` and dtype float64.
    """
    num_samples = int(duration * sample_freq)
    t = np.arange(num_samples) / sample_freq

    signal = _generate_harmonics(f0, harmonics, t)
    signal_power = _calculate_signal_power(signal, harmonics)
    signal_multi = _add_noise_and_broadcast(signal, signal_power, snr_db, num_samples, num_channels)

    return signal_multi.astype(np.float64)


def _generate_harmonics(f0, harmonics, t):
    """Generate sum of harmonic sinusoids."""
    signal = np.zeros(len(t))
    for h in harmonics:
        freq = f0 * h
        signal += np.sin(2 * np.pi * freq * t)
    return signal


def _calculate_signal_power(signal, harmonics):
    """Calculate mean-square power of signal, handling noise-only case."""
    if len(harmonics) > 0:
        return float(np.mean(signal**2))
    return NOISE_ONLY_POWER


def _add_noise_and_broadcast(signal, signal_power, snr_db, num_samples, num_channels):
    """Add noise at specified SNR and broadcast to multiple channels."""
    if not np.isinf(snr_db):
        snr_linear = 10 ** (snr_db / DB_TO_LINEAR_FACTOR)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, num_channels) * np.sqrt(noise_power)
        return signal[:, np.newaxis] + noise
    return np.tile(signal[:, np.newaxis], (1, num_channels))


def generate_swept_tonal_signal(f_start, f_end, duration, sample_freq, num_channels=1, snr_db=20):
    """Generate tonal signal with a linearly swept frequency.

    Returns
    -------
    signal : numpy.ndarray
        Signal array with shape ``(num_samples, num_channels)``.
    freq_trajectory : numpy.ndarray
        Instantaneous frequency at each sample, shape ``(num_samples,)``.
    """
    num_samples = int(duration * sample_freq)

    freq_trajectory = np.linspace(f_start, f_end, num_samples)
    phase = 2 * np.pi * np.cumsum(freq_trajectory) / sample_freq
    signal = np.sin(phase)

    signal_power = float(np.mean(signal**2))

    if not np.isinf(snr_db):
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.random.randn(num_samples, num_channels) * np.sqrt(noise_power)
        signal_multi = signal[:, np.newaxis] + noise
    else:
        signal_multi = np.tile(signal[:, np.newaxis], (1, num_channels))

    return signal_multi.astype(np.float64), freq_trajectory


# ---------------------------------------------------------------------------
# Mock sources
# ---------------------------------------------------------------------------


class MockSamplesGenerator(SamplesGenerator):
    """Mock source providing pre-generated data in blocks.

    Supports two usage patterns:

    1. Constructor initialisation::

           source = MockSamplesGenerator(data, sample_freq)

    2. Deferred ``set_signal()``::

           source = MockSamplesGenerator()
           source.sample_freq = 16000.0
           source.set_signal(signal)
    """

    sample_freq = Float(16000.0)
    _signal_data = CArray(dtype=np.float64, shape=(None, None))

    def __init__(self, data=None, sample_freq=None, **kwargs):
        super().__init__(**kwargs)
        if data is not None:
            self._signal_data = np.asarray(data, dtype=np.float64)
        if sample_freq is not None:
            self.sample_freq = float(sample_freq)

    def _get_num_samples(self):
        return self._signal_data.shape[0] if self._signal_data.size > 0 else 0

    def _get_num_channels(self):
        return self._signal_data.shape[1] if self._signal_data.size > 0 else 1

    num_samples = property(_get_num_samples)
    num_channels = property(_get_num_channels)

    def set_signal(self, signal):
        """Set the signal data."""
        self._signal_data = np.asarray(signal, dtype=np.float64)

    def result(self, num):
        """Yield blocks of signal data."""
        start = 0
        total_samples = self._signal_data.shape[0]
        while start < total_samples:
            end = min(start + num, total_samples)
            yield self._signal_data[start:end, :]
            start = end


class MockFreqSource(SamplesGenerator):
    """Mock streaming frequency source for testing.

    Wraps a pre-computed frequency array and yields it in blocks following the
    ``result(num)`` streaming protocol.  Works for both 1-D
    (AdaptiveNotchFilter) and 2-D (CascadeNotchFilter) data.
    """

    sample_freq = Float(1.0)
    _freq_data = CArray(dtype=np.float64)

    def __init__(self, freq_data, **kwargs):
        super().__init__(**kwargs)
        self._freq_data = np.asarray(freq_data)

    def _get_num_samples(self):
        return self._freq_data.shape[0] if self._freq_data.size > 0 else 0

    def _get_num_channels(self):
        return self._freq_data.shape[1] if self._freq_data.ndim == 2 else 1

    num_samples = property(_get_num_samples)
    num_channels = property(_get_num_channels)

    def result(self, num):
        """Yield frequency blocks of size *num* along the first axis."""
        for i in range(0, self._freq_data.shape[0], num):
            yield self._freq_data[i : i + num]


# ---------------------------------------------------------------------------
# FFT analysis helpers
# ---------------------------------------------------------------------------


def compute_fft_power_db(signal, sample_freq, freq):
    """Compute power at a specific frequency using FFT (in dB)."""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_freq)

    idx = np.argmin(np.abs(freqs - freq))
    power_linear = np.abs(fft[idx]) ** 2 / len(signal) ** 2

    if power_linear > 0:
        return 10 * np.log10(power_linear)
    return -np.inf


def compute_fft_peak_frequency(signal, sample_freq, expected_freq, search_width=5.0):
    """Find peak frequency near an expected frequency using FFT."""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_freq)

    search_mask = (freqs >= expected_freq - search_width) & (freqs <= expected_freq + search_width)
    search_freqs = freqs[search_mask]
    search_fft = fft[search_mask]

    peak_idx = np.argmax(np.abs(search_fft))
    return search_freqs[peak_idx]


def compute_phase_at_frequency(signal, sample_freq, freq):
    """Compute phase at a specific frequency using FFT (in radians)."""
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1 / sample_freq)

    idx = np.argmin(np.abs(freqs - freq))
    return np.angle(fft[idx])
