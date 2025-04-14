"""Grids
=====================================================

Description...
"""


#%%
# First import Acoular

import acoular as ac
from pathlib import Path
import acoular as ac
import matplotlib.pyplot as plt


#%% 
# ================
# Measurement Setup
# ================

micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
datafile = Path('three_sources.h5')
assert datafile.exists(), 'Data file not found, run example_three_sources.py first'

mg = ac.MicGeom(file=micgeofile)
ts = ac.TimeSamples(file=datafile)
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
st = ac.SteeringVector(mics=mg)
bb = ac.BeamformerBase(freq_data=ps, steer=st)


#%% 
#================
#Rectangular Grid
#================

rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)

for i in range(3):
    rg.increment = 0.01 * (i + 1)
    pm = bb.synthetic(8000, 3)
    Lm = ac.L_p(pm)

rg.size # size of the grid
rg.nxsteps
rg.nysteps

# plot mit 3 subplots




#%% 
#========
#3D Grid
#========

rg = ac.RectGrid3D(

)
st.grid = rg

for i in range(3):
    rg.increment = 0.01 * (i + 1)
    pm = bb.synthetic(8000, 3)
    Lm = ac.L_p(pm)