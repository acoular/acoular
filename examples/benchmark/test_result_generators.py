# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
"""Benchmarks for the result-generator ownership guarantees (Issue #654)."""

import acoular as ac

import numpy as np

NUM_SAMPLES = 8192
NUM_CHANNELS = 8
FFT_BLOCK_SIZE = 256
RESULT_BLOCK_SIZE = 192
SOURCE_NUM_SAMPLES = 1024
SOURCE_BLOCK_SIZE = 128

DATA = np.random.default_rng(0).standard_normal((NUM_SAMPLES, NUM_CHANNELS))
SILENCE = np.zeros_like(DATA)
MICS = ac.MicGeom(
    pos_total=np.array(
        [
            [-0.1, 0.1, -0.1, 0.1],
            [-0.1, -0.1, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0],
        ],
    ),
)
TRAJECTORY = ac.Trajectory(
    points={
        0.0: (-0.1, 0.0, 1.0),
        0.1: (0.1, 0.0, 1.0),
    },
)
INTEGRATION_GRID = ac.RectGrid(x_min=0.0, x_max=0.1, y_min=0.0, y_max=0.1, z=1.0, increment=0.1)


def _time_samples(data=DATA):
    return ac.TimeSamples(data=data, sample_freq=51200)


def _drain(generator, num):
    for _ in generator.result(num):
        pass


def _signal():
    return ac.WNoiseGenerator(sample_freq=51200, num_samples=SOURCE_NUM_SAMPLES, seed=0)


def _point_source(source_class):
    return source_class(signal=_signal(), mics=MICS)


def test_rfft():
    """Fresh output allocation for spectral blocks."""
    _drain(ac.RFFT(source=_time_samples(), block_size=FFT_BLOCK_SIZE), num=4)


def test_irfft_buffer():
    """Stable output blocks from IRFFT's SamplesBuffer path."""
    rfft = ac.RFFT(source=_time_samples(), block_size=FFT_BLOCK_SIZE)
    _drain(ac.IRFFT(source=rfft), num=RESULT_BLOCK_SIZE)


def test_mixer():
    """Output allocation instead of mutating the primary source block."""
    _drain(
        ac.Mixer(source=_time_samples(SILENCE), sources=[_time_samples(SILENCE)]),
        num=FFT_BLOCK_SIZE,
    )


def test_source_mixer():
    """Output allocation instead of mutating a source in a source mix."""
    _drain(ac.SourceMixer(sources=[_time_samples(SILENCE), _time_samples(SILENCE)]), num=FFT_BLOCK_SIZE)


def test_masked_timeout():
    """Independent blocks from the buffered MaskedTimeOut path."""
    _drain(ac.MaskedTimeOut(source=_time_samples(), start=13, stop=NUM_SAMPLES - 19), num=FFT_BLOCK_SIZE)


def test_time_reverse():
    """Independent blocks from the reversed result buffer."""
    _drain(ac.TimeReverse(source=_time_samples()), num=FFT_BLOCK_SIZE)


def test_octave_filter_bank():
    """Independent output blocks from the filter-bank workspace."""
    _drain(ac.OctaveFilterBank(source=_time_samples(), lband=30, hband=34), num=FFT_BLOCK_SIZE)


def test_point_source():
    """Fresh blocks from a stationary point source."""
    _drain(_point_source(ac.PointSource), num=SOURCE_BLOCK_SIZE)


def test_point_source_dipole():
    """Fresh blocks from a stationary dipole source."""
    _drain(_point_source(ac.PointSourceDipole), num=SOURCE_BLOCK_SIZE)


def test_moving_point_source_dipole():
    """Fresh blocks from a moving dipole source."""
    _drain(
        ac.MovingPointSourceDipole(signal=_signal(), mics=MICS, trajectory=TRAJECTORY),
        num=SOURCE_BLOCK_SIZE,
    )


def test_integrator_sector_time():
    """Independent blocks from time-domain sector integration."""
    _drain(
        ac.IntegratorSectorTime(
            source=_time_samples(DATA[:, : INTEGRATION_GRID.size]),
            grid=INTEGRATION_GRID,
            sectors=[np.array([0.05, 0.05, 0.2])],
        ),
        num=FFT_BLOCK_SIZE,
    )


def test_rfft_irfft_chain():
    """End-to-end throughput for a representative frequency-domain chain."""
    rfft = ac.RFFT(source=_time_samples(), block_size=FFT_BLOCK_SIZE)
    _drain(ac.TimePower(source=ac.IRFFT(source=rfft)), num=RESULT_BLOCK_SIZE)
