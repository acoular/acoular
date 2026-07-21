# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""Caching -- Reuse calculated results.
======================================

Shows how to configure Acoular's file-based cache, how object-level caching
interacts with the global caching mode, and how to explicitly cache a lazy
time-domain processing chain.
"""

# Import packages.
from pathlib import Path

import acoular as ac

# %%
# Configure cache directory, backend, and global mode.
cache_dir = Path('cache')
ac.config.cache_dir = str(cache_dir)
ac.config.h5library = 'pytables'
ac.config.global_caching = 'individual'

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
# Use object-level caching controls.
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
ps.cached = False

bb = ac.BeamformerBase(freq_data=ps, steer=st, cached=True)
pm = bb.synthetic(8000, 3)

# %%
# Check the generated cache file.
cache_file = cache_dir / 'three_sources_cache.h5'

# %%
# Force recalculation in overwrite mode.
ac.config.global_caching = 'overwrite'
pm_overwrite = bb.synthetic(8000, 3)
ac.config.global_caching = 'individual'

# %%
# Explicitly cache a time-domain processing chain.
bt = ac.BeamformerTime(source=ts, steer=st)
ft = ac.FiltOctave(source=bt, band=4000)
pt = ac.TimePower(source=ft)
avg = ac.Average(source=pt, num_per_average=1024)
cached_avg = ac.Cache(source=avg)
first_map = next(cached_avg.result(num=1))
