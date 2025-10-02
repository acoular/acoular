# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
.. _beamforming_in_the_time_domain_with_fixed_focus:

Beamforming in the time domain with fixed focus
===============================================

The examples demonstrates time-domain beamforming with moving sources and a fixed focus grid.
"""

from pathlib import Path
import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

# %% Parameters for synthetic scenario
lx = 5  # m, distance to start position on x-axis
lz = 10 # m, distance between array and source plane
v = 60  # km/h, source velocity
vx_m = v / 3.6  # m/s, source velocity converted to meters per second
t_pass = 2 * lx / vx_m  # s, total pass-by duration
sfreq = 12800  # Hz, sampling frequency
num_samples = int(t_pass * sfreq)  # total number of samples
block_size = int(num_samples / 16)  # number of samples per processing block
freq = 4000  # Hz, third-octave band used for analysis

# %% Microphone array configuration
micgeom = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'tub_vogel64.xml')

# %% Define rectangular grid for fixed focus beamforming
rg = ac.RectGrid(x_min=-4, x_max=4, y_min=-2, y_max=2, z=lz, increment=0.1)
st = ac.SteeringVector(grid=rg, mics=micgeom)

# %% Synthetic data generation function
def generate_synth_data(micgeom, sfreq, num_samples, lx, lz):
    """
    Generates synthetic data for four moving white noise sources.
    Returns trajectories and a SourceMixer object.
    """
    nseed = 1
    convamp = True
    # Create four white noise signals with different RMS values
    sigs = [ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=nseed + i, rms=0.564 * 10**(-0.5 * i)) for i in range(4)]
    # Source positions (relative to origin)
    source_pos = [(1, 1, 0), (1, -1, 0), (-1, -1, 0), (-1, 1, 0)]
    def create_trajectory(sp):
        # Linear trajectory for each source
        return ac.Trajectory(points={0: (-lx + sp[0], sp[1], lz + sp[2]), t_pass: (lx + sp[0], sp[1], lz + sp[2])})
    trs = [create_trajectory(sp) for sp in source_pos]
    # Create MovingPointSource objects
    mps = [ac.MovingPointSource(signal=sigs[i], mics=micgeom, trajectory=trs[i], conv_amp=convamp) for i in range(4)]
    # Mix all sources
    source_mixer = ac.SourceMixer(sources=mps)
    return trs, source_mixer

# %% Generate synthetic moving source data
trs, source_mixer = generate_synth_data(micgeom, sfreq, num_samples, lx, lz)

# %% Processing chain for fixed focus beamforming
fi = ac.FiltFiltOctave(source=source_mixer, band=freq, fraction='Third octave')
bt = ac.BeamformerTimeSq(source=fi, steer=st, r_diag=True)
avgt = ac.Average(source=bt, num_per_average=block_size)
cacht = ac.Cache(source=avgt)

# %% Run processing and plot results
res = ac.tools.return_result(cacht, num=1)
nsize = int(np.ceil(np.sqrt(len(res))))
plt.figure(figsize=(8, 7))
for i, r in enumerate(res):
    r0 = r.reshape(rg.shape)
    plt.subplot(nsize, nsize, i + 1)
    mx = ac.L_p(r0.max())
    plt.imshow(ac.L_p(np.transpose(r0)), vmax=mx, vmin=mx - 15, interpolation='nearest', extent=rg.extend(), origin='lower')
    plt.title(f"t = {(i) * block_size / sfreq:.2f}s")
    plt.colorbar()
plt.tight_layout()
plt.show()
