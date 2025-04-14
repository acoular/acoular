"""Grids
=====================================================

Description...
"""


# %%
# First import Acoular

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np  # for array

# %%
# =================
# Measurement Setup
# =================

micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
datafile = Path('three_sources.h5')
assert datafile.exists(), 'Data file not found, run example_three_sources.py first'

mg = ac.MicGeom(file=micgeofile)
ts = ac.TimeSamples(file=datafile)
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
st = ac.SteeringVector(mics=mg)
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# ================
# Rectangular Grid
# ================

# %%
# Basic RectGrid usage

# Create a RectGrid object
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)

# %%
# The RectGrid class has several properties that can be easily accessed

# Access grid properties
print(f'Grid size: {rg.size}')  # Total number of grid points
print(f'Number of x steps: {rg.nxsteps}') # Grid points along the x axis
print(f'Number of y steps: {rg.nysteps}') # Grid points along the y axis
print(f'Grid shape: {rg.shape}')  # (nxsteps, nysteps)
print(f'Grid positions:\n{rg.pos[:, :5]}')  # Display the first 5 grid point positions

# %%
# There is one more characteristic of grids which can be useful: the extend.

# Get the grid's extent (useful for plt.imshow)
print(f'Grid extent: {rg.extend()}')

# %%
# A RectGrid can be split up into subdomains.
# Here we will look at a subdomain in the first quadrant.

# For reasons of clarity and comprehensibility we use a much smaller grid
rg.increment = 0.1

# First, create a rectangular sector in the first quadrant 
rect_sector = ac.RectSector(x_min=0, x_max=1, y_min=0, y_max=1)

# Get the indices within the sector
subdomain_indices = rg.subdomain(rect_sector)
print('Subdomain indices:', subdomain_indices)

# You can use these indices to extract the corresponding grid points
x_indices, y_indices = subdomain_indices
subdomain_points = rg.pos[:, x_indices * rg.nysteps + y_indices]
print('Subdomain points:\n', subdomain_points)


# %%
# Coordinate mapping

# First, we reset the grid increment
rg.increment = 0.01

# Find the grid index closest to a given coordinate
x_coord = 0.05
y_coord = -0.05
index_of_coord = rg.index(x_coord, y_coord)
print(f'Index of ({x_coord}, {y_coord}): {index_of_coord}')

# Get indices of a rectangular subdomain
rect_subdomain_indices = rg.indices(-0.1, -0.1, 0.1, 0.1)
print('Rectangular subdomain indices:', rect_subdomain_indices)

# Get indices of a circular subdomain
circ_subdomain_indices = rg.indices(0, 0, 0.15)  # x, y, radius
print('Circular subdomain indices:', circ_subdomain_indices)

# Get indices of a polygonal subdomain
poly_subdomain_indices = rg.indices(-0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1)
print('Polygonal subdomain indices:', poly_subdomain_indices)

# %%
# To procede to beamforming from here, we first need to pass the grid to the steering vector.

# Pass grid to steering vector
st.grid = rg

# %%
# Now we can calcualte the beamforming output and visualize it in a plot.
# Here we do so for three different grid increments.

for i in range(3):
    # Increase grid increment
    rg.increment += 0.01 * i

    # Calculate beamforming output
    pm = bb.synthetic(8000, 3)
    Lm = ac.L_p(pm)

    # Create the plot
    plt.figure(i)
    plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extend(), interpolation='bicubic')
    plt.title(f'Beamforming map\n(Grid increment: {rg.increment})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Sound pressure level')

plt.show()

# %%
# Note that the results are more precise the smaller the incerment.

# %%
# =======
# 3D Grid
# =======

# %%
# Basic RectGrid3D usage

# Create a RectGrid3D object
rg3d = ac.RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=-0.3, z_max=-0.1, increment=0.02)
print(rg3d.size)

# Apply a different increment for each dimension
rg3d.increment = (0.02, 0.02, 0.01)
print(rg3d.size)

# %%
# 3D Grid Properties

# Access 3D grid properties
print(f'3D Grid size: {rg3d.size}')
print(f'Number of x steps: {rg3d.nxsteps}')
print(f'Number of y steps: {rg3d.nysteps}')
print(f'Number of z steps: {rg3d.nzsteps}')
print(f'3D Grid shape: {rg3d.shape}')
print(f'3D Grid positions:\n{rg3d.pos[:, :5]}')

# %%
# 3D Visualization

# Calculate beamforming output (dummy 3D data)
# Replace this with your actual 3D beamforming calculation
num_x_points = rg3d.nxsteps
num_y_points = rg3d.nysteps
num_z_points = rg3d.nzsteps
num_points = rg3d.size
map_3d = np.random.rand(num_x_points * num_y_points * num_z_points).reshape((num_x_points, num_y_points, num_z_points))
# Sum over the z-axis to get a 2D map
map_2d_z_sum = np.sum(map_3d, axis=2)
map_2d_y_sum = np.sum(map_3d, axis=1)
map_2d_x_sum = np.sum(map_3d, axis=0)

# Display the result
fig = plt.figure(figsize=(12, 5))

# plot the results
# XY view
ax1 = fig.add_subplot(1, 3, 1)
mx = ac.L_p(map_2d_z_sum.max())
ax1.imshow(
    ac.L_p(map_2d_z_sum.T),
    vmax=mx,
    vmin=mx - 20,
    origin='lower',
    interpolation='nearest',
    extent=(rg3d.x_min, rg3d.x_max, rg3d.y_min, rg3d.y_max),
)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Top view (xy)')

# XZ view
ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(
    ac.L_p(map_2d_y_sum.T),
    vmax=mx,
    vmin=mx - 20,
    origin='lower',
    interpolation='nearest',
    extent=(rg3d.x_min, rg3d.x_max, rg3d.z_min, rg3d.z_max),
)
ax2.set_xlabel('x')
ax2.set_ylabel('z')
ax2.set_title('Side view (xz)')

# YZ view
ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(
    ac.L_p(map_2d_x_sum.T),
    vmax=mx,
    vmin=mx - 20,
    origin='lower',
    interpolation='nearest',
    extent=(rg3d.y_min, rg3d.y_max, rg3d.z_min, rg3d.z_max),
)
ax3.set_xlabel('y')
ax3.set_ylabel('z')
ax3.set_title('Front view (yz)')

plt.tight_layout()
plt.show()
