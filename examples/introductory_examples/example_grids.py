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
import numpy as np # for array


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

#%%
# Basic RectGrid Usage

# Create a RectGrid object
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)

print(rg)

#%%
# RectGrid Properties

# Access grid properties
print(f"Grid size: {rg.size}")  # Total number of grid points
print(f"Number of x steps: {rg.nxsteps}")
print(f"Number of y steps: {rg.nysteps}")
print(f"Grid shape: {rg.shape}")  # (nxsteps, nysteps)
print(f"Grid positions:\n{rg.pos[:, :5]}")  # Display the first 5 positions


#%%
# RectGrid Subdomain

# Create a rectangular sector
rect_sector = ac.RectSector(x_min=-0.1, x_max=0.1, y_min=-0.1, y_max=0.1)

# Get the indices within the sector
subdomain_indices = rg.subdomain(rect_sector)
print("Subdomain indices:", subdomain_indices)

# You can use these indices to extract the corresponding grid points
x_indices, y_indices = subdomain_indices
subdomain_points = rg.pos[:, x_indices * rg.nysteps + y_indices]
print("Subdomain points:\n", subdomain_points[:, :5])


#%%
# Coordinate Mapping

# Find the grid index closest to a given coordinate
x_coord = 0.05
y_coord = -0.05
index_of_coord = rg.index(x_coord, y_coord)
print(f"Index of ({x_coord}, {y_coord}): {index_of_coord}")

# Get indices of a rectangular subdomain
rect_subdomain_indices = rg.indices(-0.1, -0.1, 0.1, 0.1)
print("Rectangular subdomain indices:", rect_subdomain_indices)

# Get indices of a circular subdomain
circ_subdomain_indices = rg.indices(0, 0, 0.15)  # x, y, radius
print("Circular subdomain indices:", circ_subdomain_indices)

# Get indices of a polygonal subdomain
poly_subdomain_indices = rg.indices(-0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1)
print("Polygonal subdomain indices:", poly_subdomain_indices)

#%%
# Visualization

# Calculate beamforming output (dummy data for demonstration)
# Replace this with your actual beamforming calculation
freq = 8000
num_points = rg.nxsteps * rg.nysteps
pm = np.random.rand(num_points)
Lm = ac.L_p(pm.reshape(rg.shape).T)  # Beamforming map in dB


# Create the plot
plt.figure(figsize=(8, 6))
plt.imshow(
    Lm,
    origin="lower",
    extent=(rg.x_min, rg.x_max, rg.y_min, rg.y_max),
    cmap="jet",
    interpolation="nearest",
)
plt.title("Beamforming Map")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.colorbar(label="Sound Pressure Level (dB)")
plt.grid(False)
plt.show()

#%%
# Extend

# Get the grid's extent for imshow
extent = rg.extend()
print(f"Grid extent: {extent}")