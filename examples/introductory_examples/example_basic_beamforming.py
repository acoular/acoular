# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""Basic Beamforming -- Generate a map of three sources.
=====================================================

Loads the simulated signals from the `three_sources.h5` file, analyzes them with Conventional
Beamforming and generates a map of the three sources.

.. note:: The `three_sources.h5` file must be generated first by running the
    :doc:`example_three_sources` example.
"""

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt

# Apply Acoular's matplotlib style
plt.style.use(Path(ac.__file__).parent / 'plots.mplstyle')

micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
datafile = Path('three_sources.h5')
assert datafile.exists(), 'Data file not found, run example_three_sources.py first'

mg = ac.MicGeom(file=micgeofile)
ts = ac.TimeSamples(file=datafile)
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)
bb = ac.BeamformerBase(freq_data=ps, steer=st)
pm = bb.synthetic(8000, 3)
Lm = ac.L_p(pm)

# %%
# Plot the beamforming map
fig, ax = plt.subplots()
im = ax.imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extent, interpolation='bicubic')
plt.colorbar(im, ax=ax, label='$L_p$ / dB')
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_title('Beamforming Map')

# %%
# Plot the microphone geometry
fig, ax = plt.subplots()
ax.plot(mg.pos[0], mg.pos[1], 'o')
ax.axis('equal')
ax.set_xlabel('x / m')
ax.set_ylabel('y / m')
ax.set_title('Microphone Array Geometry')
plt.show()
