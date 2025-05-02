# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Grids
=====

This example will guide you through using Acoular's grid functionalities.
We'll cover grid creation, accessing grid properties, working with subdomains,
and generating non-uniform grids. To give you a practical context, we'll
also include a basic measurement setup, showing how grids fit into
beamforming applications.
"""
# %%
# Let's start by importing Acoular and the other modules we'll need.

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

# %%
# =================
# Measurement Setup
# =================
#
# First, let's set up a simulated measurement.
# This will illustrate how grids are typically used within a standard Acoular workflow.

# We'll begin by defining the microphone geometry using an XML file.
micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
# Now, we'll create a MicGeom object to represent our microphone geometry.
mg = ac.MicGeom(file=micgeofile)
# Next, we'll generate time sample data.
pa = ac.demo.create_three_sources_2d(mg)
# To prepare for beamforming, we need to calculate the frequency spectra of the signals.
# We'll do this using the PowerSpectra object.
ps = ac.PowerSpectra(source=pa, block_size=128, window='Hanning')
# Next, we'll create a SteeringVector object.
# Steering vectors are crucial for beamforming; they handle the time delays between microphones.
# Keep in mind that the steering vector requires a grid as input, which we'll provide later.
st = ac.SteeringVector(mics=mg)
# Finally, let's create a BeamformerBase object.
# This is the foundation for our beamforming algorithms,
# and it takes the frequency data and steering vectors as input.
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# ================
# Rectangular Grid
# ================
#
# In this section, we'll explore how to create and use RectGrid objects in Acoular.
# RectGrid helps us define the spatial points where the beamforming output will be calculated.

# %%
# **Basic RectGrid usage**
#
# Let's start by creating a RectGrid object.
# This will define a rectangular grid in the x-y plane.
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.2, increment=0.01)

# %%
# The RectGrid class provides several properties that you can easily access.

# Let's access some of these grid properties:
print(f'Grid size: {rg.size}')  # Total number of grid points
print(f'Number of x steps: {rg.nxsteps}')  # Grid points along the x axis
print(f'Number of y steps: {rg.nysteps}')  # Grid points along the y axis
print(f'Grid shape: {rg.shape}')  # (nxsteps, nysteps)
print(f'Grid positions:\n{rg.pos[:, :5]}')  # Display the first 5 grid point positions

# %%
# Another useful grid characteristic is its *extent*.
# The *extent* is particularly helpful for visualization, especially with matplotlib's
# :obj:`~matplotlib.pyplot.imshow` function.
# It defines the grid's boundaries for plotting.

# Let's get the grid's extent:
print(f'Grid extent: {rg.extend()}')

# %%
# **Beamforming with rectangular grids**
#
# Now, let's visualize some beamforming results on our grid.

# First, we need to pass the grid to the steering vector.
st.grid = rg

# %%
# We can now calculate the beamforming output and display it in a plot.

fig = plt.figure(figsize=(12, 5))

# We'll do this for three different grid increments to see the effect of resolution.
for i in range(3):
    # Increase grid increment
    rg.increment = 0.01 * (i + 1)

    # Here, we'll calculate the beamforming output.
    # We'll calculate it for the third-octave band around a frequency of 8000.
    pm = bb.synthetic(8000, 3)
    Lm = ac.L_p(pm)

    # Create the plot
    ax = fig.add_subplot(1, 3, i + 1)
    ax.imshow(
        Lm.T,
        origin='lower',
        vmin=Lm.max() - 10,
        extent=rg.extend(),
    )
    ax.set_title(f'Beamforming Results on Rectangular Grid\n(grid increment: {rg.increment})')
    ax.set_xlabel('x / m')
    ax.set_ylabel('y / m')

plt.tight_layout()
plt.show()

# %%
# Note how the results become less precise as the grid increment increases.
#
# =======
# 3D Grid
# =======
#
# Let's briefly introduce 3D grids.
#
# **Basic RectGrid3D usage**

# To create a 3D grid, we use the RectGrid3D object.
rg3d = ac.RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=-0.3, z_max=-0.1, increment=0.01)
print('3D-grid size:', rg3d.size)

# We can also apply different increments for each dimension if needed.
rg3d.increment = (0.02, 0.02, 0.01)
print('3D-grid size with bigger incerement in the z-dimension:', rg3d.size)

# %%
# **3D-grid properties**
#
# 3D rectangular grids hold a range of properties similar to those of their 2D counterparts.
# The main difference is that we now have 'nzsteps' and a 3D 'shape'.
# Accessing these properties also is similar to the 2D-grids case.
# Let's take a look:

# Access 3D grid properties
print(f'3D Grid size: {rg3d.size}')
print(f'Number of x steps: {rg3d.nxsteps}')
print(f'Number of y steps: {rg3d.nysteps}')
print(f'Number of z steps: {rg3d.nzsteps}')
print(f'3D Grid shape: {rg3d.shape}')
print(f'3D Grid positions:\n{rg3d.pos[:, :5]}')

# %%
# **Beamforming with 3D grids**
#
# For the visualize 3D beamforming results, we'll first mix up our examples data a little and
# generate time sample data of three white-noise point sources with diffrent x-, y-, and
# z-coodinates each.

# Here, we generate the 3D time samples data.
# We use the same microphone geometry we used for the 2D grids.
pa = ac.demo.create_three_sources_3d(mg)
# Again, we plug the sample data into a PowerSpectra object.
# This time we use a block size of 128 and a Bartlett window for a change.
ps = ac.PowerSpectra(source=pa, block_size=128, window='Bartlett')
# Now, we need a 3D beamforming method. We'll use BeamformerCleansc,
# which is a more advanced technique than BeamformerBase.
bc = ac.BeamformerCleansc(freq_data=ps, steer=st)

# %%
# Note, that we needn't change the steering vector.
# However we need to change the grid attributed to it, which currently still is the 2D grid.

# Here, we change the steering vector's grid to the 3D grid.
st.grid = rg3d
# And we change the grid increment to a unifrom value.
rg3d.increment = 0.01

# %%
# Now we are ready to compute the beamforming output.
# This time, we'll calculate it for a frequency of exactly 8000.

map_3d = bc.synthetic(8000, 1)
# Note that the output of 3D beamforming is a 3D field:
print('3D beamforming output shape:', map_3d.shape)

# %%
# Let's display the results in a 3D plot.

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Plot the microphone geometry.
ax.scatter(*mg.pos, marker='o', label='Microphones')

# Plot the 3D grid.
ax.scatter(*rg3d.pos, s=0.05, c='k', marker='.', label='3D Grid Points')

# Finde the indices of the nonzero entries in the output map.
indices = np.nonzero(map_3d)
# Find the corresponding grid points to the indices.
pos = [p.reshape(map_3d.shape)[indices] for p in rg3d.pos]
# Plot the points with color mapping according to their beamforming intensity.
scatter = ax.scatter(*pos, c=map_3d[indices], marker='^', label='Beamforming Results')
# Add a color bar to indicate the beamforming intensity.s
cbar = fig.colorbar(scatter)
cbar.set_label('$L_p$ / dB')

ax.set(xlabel='x / m', ylabel='y / m', zlabel='z / m')
ax.set_title('Beamforming Results on 3D Grid')
ax.legend()
plt.show()

# %%
# =========
# Line Grid
# =========
#
# Let's explore the LineGrid class, which is useful for analyzing sound sources
# along a straight line in 3D space.

# %%
# **Basic LineGrid usage**
#
# First, let's create a line grid along the x-axis.
line_grid = ac.LineGrid()
line_grid.loc = (-0.2, 0.0, 0.0)  # Start at -0.2 m
line_grid.direction = (1.0, 0.0, 0.0)  # Point along x-axis
line_grid.length = 0.4  # 40 cm long
line_grid.num_points = 100  # 100 points along the line

# %%
# **Line grid properties**
#
# The LineGrid class provides several properties that you can access:

# Basic properties
print(f'Line grid size: {line_grid.size}')  # Total number of grid points
print(f'Line grid length: {line_grid.length} m')  # Total length of the line
print(f'Number of points: {line_grid.num_points}')  # Number of points along the line
print(f'Start location: {line_grid.loc}')  # Starting point of the line
print(f'Direction vector: {line_grid.direction}')  # Direction of the line
print(f'First 5 grid positions:\n{line_grid.pos[:, :5]}')  # First 5 positions

# %%
# **Beamforming with line grids**
#
# Let's set up a simple measurement scenario to demonstrate beamforming with a line grid.

# Create a microphone line geometry.
pos_total = np.zeros((3, 32))
pos_total[0, :] = np.linspace(-0.3, 0.3, 32) # 32 microphones along a 60 cm line
line_mg = ac.MicGeom(pos_total=pos_total)

# Generate test data with three sources along the line
line_pa = ac.demo.create_three_sources_1d(line_mg)

# %%
# Calculate the power spectrum
ps = ac.PowerSpectra(source=line_pa, block_size=128, window='Hanning')
freqs = ps.fftfreq()  # Get the FFT frequencies

# %%
# Change the steering vector grid and microphone geometry to the line grid and line microphone
# geometry.
st.grid = line_grid
st.mics = line_mg

# %%
# Change the frequency data to the line frequency data.
bc.freq_data = ps

# %%
# Calculate the beamforming output for a range of frequencies
freq_range = np.logspace(2, 4, 100)  # Frequencies from 100 Hz to 10 kHz
results = np.zeros((len(freq_range), line_grid.num_points))

for i, freq in enumerate(freq_range):
    pm = bc.synthetic(freq, 3)  # Use 1/3 octave band
    results[i, :] = ac.L_p(pm)

# %%
# Create the visualization
plt.figure(figsize=(10, 6))
plt.pcolormesh(
    line_grid.pos[0, :],  # x positions
    freq_range,  # frequencies
    results,
    shading='auto',
    cmap='viridis',
)
plt.colorbar(label='$L_p$ / dB')
plt.xlabel('Position along line / m')
plt.ylabel('Frequency / Hz')
plt.title('Beamforming Results along Line Grid')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()