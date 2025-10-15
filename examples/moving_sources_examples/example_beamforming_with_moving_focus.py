# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
.. _beamforming_with_moving_focus:

Beamforming with moving focus
=============================

Demonstrates time-domain beamforming with moving sources and a moving focus grid.

This example shows how to set up a moving-focus trajectory and
compute time-domain beamforming maps along that trajectory. The script is
split into well-documented sections for parameters, measurement setup, synthetic
data generation, processing-chain construction, plotting and a CLI-based main
entry point with a `--quick` option for short demo runs.
"""

import argparse
from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

# =====================
# Parameters for example
# =====================
LX = 5.0  # m, distance to start position on x-axis
LZ = 10.0  # m, distance between array and source plane
V_KMH = 60.0  # km/h, source velocity
VX_M = V_KMH / 3.6  # m/s, source velocity converted to meters per second
T_PASS = 2 * LX / VX_M  # s, total pass-by duration
SFREQ = 12800  # Hz, sampling frequency
NUM_SAMPLES = int(T_PASS * SFREQ)  # total number of samples
BLOCK_SIZE = int(NUM_SAMPLES / 16)  # number of samples per processing block
FREQ = 4000  # Hz, third-octave band used for analysis


def mic_geometry():
    """Return the MicGeom used for the examples."""
    return ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml')


def default_grid_and_trajectory(lx=LX, lz=LZ):
    """Create and return a RectGrid, SteeringVector and trajectory for the moving focus."""
    rgm = ac.RectGrid(x_min=-2, x_max=2, y_min=-2, y_max=2, z=0, increment=0.1)
    st = ac.SteeringVector(grid=rgm, mics=mic_geometry())
    tr0 = ac.Trajectory(points={0: (-lx, 0, lz), T_PASS: (lx, 0, lz)})
    return rgm, st, tr0


def generate_synth_data(micgeom, sfreq=SFREQ, num_samples=NUM_SAMPLES, lx=LX, lz=LZ):
    """Generate synthetic data for four moving white noise sources.

    Returns (trajectories, SourceMixer).
    """
    nseed = 1
    convamp = True
    rms_base = 0.564
    sigs = [ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=nseed + i, rms=rms_base * 10**(-0.5 * i)) for i in range(4)]
    source_pos = [(1, 1, 0), (1, -1, 0), (-1, -1, 0), (-1, 1, 0)]

    def create_trajectory(sp):
        return ac.Trajectory(points={0: (-lx + sp[0], sp[1], lz + sp[2]), T_PASS: (lx + sp[0], sp[1], lz + sp[2])})

    trs = [create_trajectory(sp) for sp in source_pos]
    mps = [ac.MovingPointSource(signal=sigs[i], mics=micgeom, trajectory=trs[i], conv_amp=convamp) for i in range(4)]
    source_mixer = ac.SourceMixer(sources=mps)
    return trs, source_mixer


def build_processing_chain(source_mixer, grid, steer, trajectory, freq=FREQ, block_size=BLOCK_SIZE):
    """Build and return the processing pipeline for the moving-focus example."""
    # Band-limit to a third-octave band around `freq`.
    fi = ac.FiltFiltOctave(source=source_mixer, band=freq, fraction='Third octave')

    # Time-domain trajectory beamforming. The `BeamformerTimeSqTraj` object
    # computes squared time-domain beamforming values along the provided
    # trajectory. Setting `r_diag=True` uses only the diagonal of the
    # cross-spectral matrix when appropriate for stability.
    bt = ac.BeamformerTimeSqTraj(source=fi, steer=steer, trajectory=trajectory, r_diag=True)

    # Average frames into larger blocks for stable plotting and performance.
    avgt = ac.Average(source=bt, num_per_average=block_size)

    # Cache averaged blocks to speed up repeated access.
    return ac.Cache(source=avgt)


def plot_results(results, grid, block_size=BLOCK_SIZE, sfreq=SFREQ, cmap='viridis'):
    n = len(results)
    if n == 0:
        print('No results to plot')
        return
    nsize = int(np.ceil(np.sqrt(n)))
    plt.figure(figsize=(8, 7))
    for i, r in enumerate(results):
        r0 = r.reshape(grid.shape)
        plt.subplot(nsize, nsize, i + 1)
        mx = ac.L_p(r0.max())
        plt.imshow(ac.L_p(np.transpose(r0)), vmax=mx, vmin=mx - 15, interpolation='nearest', extent=grid.extend(), origin='lower', cmap=cmap)
        plt.title(f"t = {(i) * block_size / sfreq:.2f}s")
        plt.colorbar()
    plt.tight_layout()
    plt.show()


def main(quick=False):
    micgeom = mic_geometry()
    grid, steer, traj = default_grid_and_trajectory()

    num_samples = NUM_SAMPLES
    block_size = BLOCK_SIZE
    if quick:
        num_samples = max(1024, int(NUM_SAMPLES / 32))
        block_size = max(64, int(block_size / 8))

    trs, source_mixer = generate_synth_data(micgeom, sfreq=SFREQ, num_samples=num_samples, lx=LX, lz=LZ)
    proc = build_processing_chain(source_mixer, grid, steer, traj, freq=FREQ, block_size=block_size)
    res = ac.tools.return_result(proc, num=1)
    plot_results(res, grid, block_size=block_size, sfreq=SFREQ)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Moving sources with moving focus example')
    parser.add_argument('--quick', action='store_true', help='Run a shorter quick demo (faster)')
    args = parser.parse_args()
    main(quick=args.quick)
