# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""Lazy evaluation -- Build first, calculate on demand.
=======================================================

Demonstrates how Acoular objects stay idle until a result is actually requested.
The example covers both frequency-domain beamforming and a time-domain
processing chain.
"""

# Import packages.
from pathlib import Path

import acoular as ac

# %%
# Define shared geometry and input data.
micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
datafile = Path('three_sources.h5')
mg = ac.MicGeom(file=micgeofile)

if not datafile.exists():
    ac.demo.create_three_sources(mg, h5savefile=datafile)

ts = ac.TimeSamples(file=datafile)
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)

# %%
# Build the frequency-domain processing objects.
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# Request the cross-spectral matrix.
csm = ps.csm

# %%
# Request a single frequency line from the lazy result object.
result = bb.result
one_line = result[10]

# %%
# Trigger the beamforming map calculation.
pm = bb.synthetic(8000, 3)

# %%
# Change a parameter and request the result again.
ps.block_size = 256
pm_updated = bb.synthetic(8000, 3)

# %%
# Build a lazy time-domain processing chain.
bt = ac.BeamformerTime(source=ts, steer=st)
ft = ac.FiltOctave(source=bt, band=4000)
pt = ac.TimePower(source=ft)
avg = ac.Average(source=pt, num_per_average=1024)

# %%
# Create the generator.
blocks = avg.result(num=1)

# %%
# Pull the first block from the generator.
first_map = next(blocks)

# %%
# Add explicit caching to the lazy chain.
cached_avg = ac.Cache(source=avg)
first_cached_map = next(cached_avg.result(num=1))
