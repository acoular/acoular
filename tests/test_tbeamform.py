from pathlib import Path

import numpy as np
import pytest
from acoular import (
    BeamformerCleant,
    BeamformerCleantSq,
    BeamformerCleantSqTraj,
    BeamformerCleantTraj,
    BeamformerTime,
    BeamformerTimeSq,
    BeamformerTimeSqTraj,
    BeamformerTimeTraj,
    MaskedTimeSamples,
    MicGeom,
    MovingPointSource,
    RectGrid,
    SteeringVector,
    Trajectory,
    WNoiseGenerator,
    WriteH5,
    config,
)

config.global_caching = 'none'

# if this flag is set to True, new time data will be simulated and
WRITE_NEW_REFERENCE_DATA = False
# new beamformer results are generated for comparison during testing. Should always be False. Only set to
# true, if it is necessary to recalculate the data, due to wanted changes of the Beamformers (or MovingPointSource).
testdir = Path(__file__).parent
# Parameters
FNAME = testdir / 'reference_data' / 'beamformer_traj_time_data.h5'
SFREQ = 6000
SPEED = 10  # km/h
SEED = 1
D = 0.5
SOURCE_POS = (0.0, 0.0, D)
passby_dist = 0.5  # distance that the source is passing in front of array
CONV_AMP = True

# create linear mic geom
MGEOM = MicGeom()
N = 5
L = 0.5
MGEOM.mpos_tot = np.zeros((3, N), dtype=np.float64)
win = np.sin(np.arange(N) * np.pi / (N - 1))
b = 0.4
MGEOM.mpos_tot[0] = np.linspace(-L, L, N) * (1 - b) / (win * b + 1 - b)

# Monopole Trajectory
t_passby = passby_dist / SPEED / 3.6
nsamples = int(t_passby * SFREQ)
TRAJ = Trajectory()  # source center
TRAJ.points[0] = (-passby_dist / 2 + SOURCE_POS[0], SOURCE_POS[1], SOURCE_POS[2])
TRAJ.points[t_passby] = (+passby_dist / 2, SOURCE_POS[1], SOURCE_POS[2])

steer_types = ['classic', 'inverse', 'true level', 'true location']
bf_time_classes = [
    BeamformerTime,
    BeamformerTimeSq,
    BeamformerCleant,
    BeamformerCleantSq,
    BeamformerTimeTraj,
    BeamformerTimeSqTraj,
    BeamformerCleantTraj,
    BeamformerCleantSqTraj,
]
TEST_PARAMS = [(bf, steer) for bf in bf_time_classes for steer in steer_types]


def create_test_time_data(nsamples):
    """
    creates test data for a single moving monopole emitting white noise

    Parameters
    ----------
    speed : float
        source velocity in m/s.

    Returns
    -------
    None.

    """
    n1 = WNoiseGenerator(sample_freq=SFREQ, numsamples=nsamples, seed=SEED)
    p1 = MovingPointSource(signal=n1, mics=MGEOM, trajectory=TRAJ, conv_amp=CONV_AMP)
    wh5 = WriteH5(source=p1, name=FNAME)
    print(50 * '#')
    print(f'create {FNAME} ...')
    print(f'num samples: {nsamples}, pass-by time: {t_passby}s')
    print(50 * '#')
    wh5.save()


def get_beamformer_traj_result(Beamformer, steer_type, num=32):
    """
    returns the result for a given Beamformer class

    Parameters
    ----------
    Beamformer : cls
        trajectory beamformer.
    num : int, optional
        number of samples to return. The default is 32.

    Returns
    -------
    array
        first block returned by the trajectory beamformers result() function.

    """
    ## with moving grid
    ts = MaskedTimeSamples(name=FNAME, stop=48)
    gMoving = RectGrid(x_min=-0.1, x_max=0.1, y_min=0, y_max=0, z=0, increment=0.1)
    stMoving = SteeringVector(grid=gMoving, mics=MGEOM, steer_type=steer_type)
    bt = Beamformer(source=ts, trajectory=TRAJ, steer=stMoving)
    if hasattr(bt, 'n_iter'):
        bt.n_iter = 2
    for data in bt.result(num):
        yield data.astype(np.float32)


def get_beamformer_time_result(Beamformer, steer_type, num=32):
    """
    returns the result for a given time Beamformer class

    Parameters
    ----------
    Beamformer : cls
        time beamformer.
    num : int, optional
        number of samples to return. The default is 32.

    Returns
    -------
    array
        first block returned by the time beamformers result() function.

    """
    ## with moving grid
    ts = MaskedTimeSamples(name=FNAME, stop=48)
    gfixed = RectGrid(x_min=-0.1, x_max=0.1, y_min=0, y_max=0, z=D, increment=0.1)
    stfixed = SteeringVector(grid=gfixed, mics=MGEOM, steer_type=steer_type)
    bt = Beamformer(source=ts, steer=stfixed)
    if hasattr(bt, 'n_iter'):
        bt.n_iter = 2
    for data in bt.result(num):
        yield data.astype(np.float32)


@pytest.mark.parametrize('beamformer, steer_type', TEST_PARAMS)
def test_beamformer_traj_result(beamformer, steer_type):
    """compare results of trajectory beamformers against previous
    results from .h5 file"""
    if beamformer.__name__.endswith('Traj'):
        gen = get_beamformer_traj_result(beamformer, steer_type=steer_type)
    else:
        gen = get_beamformer_time_result(beamformer, steer_type=steer_type)
    for i, actual_data in enumerate(gen):
        name = testdir / 'reference_data' / f'{beamformer.__name__}_{steer_type.replace(" ", "")}_{i}.npy'
        if WRITE_NEW_REFERENCE_DATA:
            np.save(name, actual_data)
        ref_data = np.load(name)
        np.testing.assert_allclose(actual_data, ref_data, rtol=5e-5, atol=5e-6)
