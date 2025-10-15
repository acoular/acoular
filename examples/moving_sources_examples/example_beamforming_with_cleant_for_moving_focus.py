# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
.. _beamforming_with_cleant_for_moving_focus:

Beamforming with deconvolution (CLEANT) for moving focus
========================================================

This example demonstrates how to use Acoular to perform moving-focus beamforming
with CLEANT-based deconvolution for moving sources.

A constellation of four moving white-noise sources passes a microphone array, and
the beamforming maps are computed along a moving focus trajectory that follows
the sources. CLEANT deconvolution is applied to each time block to enhance the
spatial resolution of the beamforming results.
"""


# %%
# =======================
# Setting up the example
# =======================
#
# At the beginning of this examples, we make the necessary imports
# and define the simulation parameters.

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

# %%
# We define the trajectory of the moving focus and the parameters for
# the moving sources. The trajectory is a straight line along the x-axis at a
# fixed height (z = 10 m) and y = 0. The sources move at a speed of 60 km/h
# and pass by the array over a duration of 0.6 seconds.

LX = 5.0  # m, distance to start position on x-axis
LZ = 10.0  # m, distance between array and source plane
V_KMH = 60.0  # km/h, source velocity
VX_M = V_KMH / 3.6  # m/s, source velocity converted to meters per second
T_PASS = 2 * LX / VX_M  # s, total pass-by duration
tr0 = ac.Trajectory(points={0: (-LX, 0, LZ), T_PASS: (LX, 0, LZ)})  # moving focus trajectory

# %%
# Next, we define parameters for the microphone array, the processing blocks,
# and the frequency band of interest.
#
# The example uses a 64-microphone array.

mics = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml')
plt.scatter(*mics.pos[:2])
plt.grid()

# %%
# The sampling frequency is set to 12.8 kHz, and the total
# number of samples is calculated based on the pass duration.

SFREQ = 12800  # Hz, sampling frequency
NUM_SAMPLES = int(T_PASS * SFREQ)  # total number of samples

# %%
# The processing block size is set to 1/16 of the total number of samples.

BLOCK_SIZE = int(NUM_SAMPLES / 16)  # number of samples per processing block

# %%
# The frequency band of interest is set to 4000 Hz, which
# corresponds to a third-octave band used for analysis.

FREQ = 4000  # Hz, third-octave band used for analysis

# %%
# Beamforming will be performed using a rectangular grid in the x-y plane,
# centered at the origin (0, 0) and extending from -2 to 2 meters in both `x` and `y` directions.
# The grid increment is set to 0.1 meters.
#
# *Generally, acoular works with dimensionless units, but since we defined the
# trajectory in meters above, we will use the same unit for the grid.*

rgm = ac.RectGrid(x_min=-2, x_max=2, y_min=-2, y_max=2, z=0, increment=0.1)

# %%
# The steering vector is created for the grid and the microphone array.
# This vector is used to steer the beamformer towards the desired focus trajectory.

st = ac.SteeringVector(grid=rgm, mics=mics)

# %%
# The `WNoiseGenerator` is used to create white-noise signal generators for the sources.
# Each source has a different root mean square (RMS) value, which is set to
# 0.564 multiplied by a factor that decreases with each source.


def wngen(nseed):
    rms = 0.564 * 10 ** (-0.5 * (nseed - 1))
    return ac.WNoiseGenerator(sample_freq=SFREQ, num_samples=NUM_SAMPLES, seed=nseed, rms=rms)


sigs = [wngen(1 + i) for i in range(4)]

# %%
# Source positions (relative to trajectory centerline)
# The four sources are positioned at the corners of a square spanning an area of 2 by 2 meters.
# For each source position, a trajectory is created that is offset from the moving focus trajectory.
# The sources move linearly along the x-axis, passing by the microphone array.

source_pos = [(1, 1, 0), (1, -1, 0), (-1, -1, 0), (-1, 1, 0)]


def create_trajectory(x, y, z):
    # Linear trajectory offset by source position
    return ac.Trajectory(points={0: (-LX + x, y, LZ + z), T_PASS: (LX + x, y, LZ + z)})


trs = [create_trajectory(*sp) for sp in source_pos]

# %%
# The sources are combined with their respective trajectories using the
# `MovingPointSource` class. The `conv_amp` parameter is set to `True`
# to account for amplitude changes due to distance variations.


def mpsgen(sig, tr):
    return ac.MovingPointSource(signal=sig, mics=mics, trajectory=tr, conv_amp=True)


mps = [mpsgen(sig, tr) for sig, tr in zip(sigs, trs)]

# %%
# Finally, the sources are mixed together using the `SourceMixer` class.
# This class combines the individual source signals into a single mixed signal
# that can be processed further down the chain.

source_mixer = ac.SourceMixer(sources=mps)

# %%
# =============================
# Building the processing chain
# =============================
#
# We now connect the processing blocks into a chain that can be executed by
# Acoular. The chain is:
#   SourceMixer → FiltFiltOctave → BeamformerCleantSqTraj → Average → Cache
#
# - **FiltFiltOctave:** zero-phase third-octave filtering around `FREQ`.
# - **BeamformerCleantSqTraj:** computes (squared) beamforming along the provided
#   moving-focus trajectory and applies CLEANT deconvolution to improve
#   spatial resolution.
# - **Average:** groups frames into blocks of `BLOCK_SIZE` samples to produce one
#   map per block.
# - **Cache:** stores the averaged maps so that repeated access is fast and
#   deterministic.

fi = ac.FiltFiltOctave(source=source_mixer, band=FREQ, fraction='Third octave')
bt = ac.BeamformerCleantSqTraj(source=fi, steer=st, trajectory=tr0, conv_amp=True)
avgt = ac.Average(source=bt, num_per_average=BLOCK_SIZE)
cacht = ac.Cache(source=avgt)

# %%
# In the following step, we execute the processing chain and retrieve
# a small number of averaged beamforming maps for visualization.
#
# The `num` argument controls how many averaged blocks to return. Use `num=None`
# to return all cached blocks, or set `num` to an integer to limit the number of
# returned maps (useful for quick visual checks).
res = ac.tools.return_result(cacht, num=1)

# %%
# -----------------------------
# Plotting the time-series maps
# -----------------------------
#
# Each element in `res` is a flattened array containing the beamforming map for
# a single averaged block. We reshape it to the grid shape and convert to sound
# pressure level (dB) via `ac.L_p` for display.
#
# We choose a dynamic colormap range per frame: [max-15 dB, max] to emphasize
# peaks while preserving local contrast. The extent for `imshow` is taken from
# the grid so the axes are in meters.

if not np.any(res):
    print(
        'No results returned by the processing chain. Try increasing `num` or '
        'ensure the processing chain produced cached averages.'
    )
else:
    n = len(res)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))  # Execute the processing chain and return a (small) sequence of averaged maps.

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    # compute a time label for each averaged block (start time of block)
    times = [(i * BLOCK_SIZE) / SFREQ for i in range(n)]

    for idx, r in enumerate(res):
        r0 = r.reshape(rgm.shape)
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]

        level = ac.L_p(r0)
        mx = level.max()

        im = ax.imshow(
            np.transpose(level),
            origin='lower',
            vmax=mx,
            vmin=mx - 15,
            interpolation='nearest',
            extent=rgm.extent,
            cmap='viridis',
        )

        ax.set_title(f't = {times[idx]:.2f} s')
        ax.set_xlabel('x / m')
        ax.set_ylabel('y / m')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='$L_p$ / dB')

    # Hide any unused subplots
    total_axes = nrows * ncols
    for i in range(n, total_axes):
        row = i // ncols
        col = i % ncols
        axes[row][col].axis('off')

    plt.tight_layout()
    plt.show()
