"""Grids
=====================================================

This example illustrates the use of Acoular's grid functionalities.
It demonstrates grid creation, property access, subdomains, and
non-uniform grid generation. A measurement setup is also included
to provide context for beamforming applications.
"""

# First import Acoular (and other modules)
from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

# %%
# =================
# Measurement Setup
# =================
#
# This section simulates a basic measurement setup.
# It's included to demonstrate how grids are typically
# used within an Acoular workflow.

# Define the microphone geometry using an XML file.
micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
# Load time sample data from an HDF5 file.
# This file should contain the recorded microphone signals.
datafile = Path('three_sources.h5')
# Ensure the data file exists; if not, raise an error.
assert datafile.exists(), 'Data file not found, run example_three_sources.py first'

# Create a MicGeom object to represent the microphone geometry.
mg = ac.MicGeom(file=micgeofile)
# Create a TimeSamples object to load the microphone data.
ts = ac.TimeSamples(file=datafile)
# Create a PowerSpectra object to calculate the frequency spectra of the signals.
# This is a crucial step in beamforming.
ps = ac.PowerSpectra(source=ts, block_size=128, window='Hanning')
# Create a SteeringVector object to define the steering vectors.
# Steering vectors are essential for beamforming, as they account
# for the time delays between microphones.
# Note that the steering vector also needs a grid as input.
# We will pass it later when we proceed to beamforming.
st = ac.SteeringVector(mics=mg)
# Create a BeamformerBase object, the base class for beamforming algorithms.
# It takes the frequency data and steering vectors as input.
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# ================
# Rectangular Grid
# ================
#
# This section demonstrates how to create and use RectGrid objects in Acoular.
# RectGrid is used to define the spatial points where the beamforming output will be calculated.

# %%
# Basic RectGrid usage
#
# Create a RectGrid object.
# This defines a rectangular grid in the x-y plane.
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)

# %%
# The RectGrid class has several properties that can be easily accessed

# Access grid properties
print(f'Grid size: {rg.size}')  # Total number of grid points
print(f'Number of x steps: {rg.nxsteps}')  # Grid points along the x axis
print(f'Number of y steps: {rg.nysteps}')  # Grid points along the y axis
print(f'Grid shape: {rg.shape}')  # (nxsteps, nysteps)
print(f'Grid positions:\n{rg.pos[:, :5]}')  # Display the first 5 grid point positions

# %%
# There is one more characteristic of grids which can be useful: the extent.
#
# The "extent" of a grid is useful for visualization, particularly when using
# :obj:`matplotlib.pyplot.imshow`. It defines the boundaries of the grid for the plot.

# Get the grid's extent (useful for plt.imshow)
print(f'Grid extent: {rg.extend()}')

# %%
# A RectGrid can be split up into subdomains.
#
# Subdomains allow you to focus the analysis on specific regions of interest within the grid.
# This is helpful for computational efficiency and focusing on relevant areas.
# Here we will look at a subdomain in the first quadrant.
# For reasons of clarity and comprehensibility we use a much smaller grid
rg.increment = 0.1  # Increase grid spacing for fewer points

# First, create a rectangular sector in the first quadrant using the RectSector class.
rect_sector = ac.RectSector(x_min=0, x_max=1, y_min=0, y_max=1)

# Get the indices of the grid points that fall within the rectangular sector.
subdomain_indices = rg.subdomain(rect_sector)
print('Subdomain indices:', subdomain_indices)

# You can use these indices to extract the corresponding grid points from the grid.
# This allows you to work with only the points within the subdomain.
x_indices, y_indices = subdomain_indices
subdomain_points = rg.pos[:, x_indices * rg.nysteps + y_indices]
print('Subdomain points:\n', subdomain_points)

# %%
# Coordinate Mapping
#
# Acoular provides methods to map between coordinates and grid indices. This is useful for finding
# the grid point closest to a given location or for defining subdomains based on coordinates.

# Find the grid index closest to a given coordinate.
x_coord = 0.05
y_coord = -0.05
index_of_coord = rg.index(x_coord, y_coord)
print(f'Index of ({x_coord}, {y_coord}): {index_of_coord}')

# %%
# Get indices of various types of subdomains: rectangular, circular, and polygonal.

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
# Visualization
#
# This section demonstrates how to visualize beamforming results on the grid.

# We first need to pass the grid to the steering vector.
st.grid = rg

# %%
# Now we can calcualte the beamforming output and visualize it in a plot.
# Here we do so for three different grid increments.

for i in range(3):
    # Increase grid increment
    rg.increment = 0.01 * (i + 1)

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
#
# This section briefly introduces 3D grids.
# For more detailed examples of 3D beamforming,
# see the 'example_3d_beamforming.py' example.
#
# Basic RectGrid3D Usage

# Create a RectGrid3D object
rg3d = ac.RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=-0.3, z_max=-0.1, increment=0.02)
print(rg3d.size)

# Apply a different increment for each dimension
rg3d.increment = (0.02, 0.02, 0.01)
print(rg3d.size)

# %%
# 3D Grid Properties
#
# Accessing properties of a 3D grid is similar to 2D,
# but now we also have 'nzsteps' and the 3D 'shape'.

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
