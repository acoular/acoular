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
    ax.set_title(f'Beamforming map\n(grid increment: {rg.increment})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

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
# Let's display the results in a plot.

fig = plt.figure()

# Calculate the beamforming output.
# Note: For true 3D beamforming, the result is a 3D field.
#       This time, we'll calculate it for a frequency of exactly 8000.
map_3d = bc.synthetic(8000, 1)
print(map_3d.shape)
print(map_3d)

# To visualize this 3D data, we need to reshape it to 3D.
map_3d = map_3d.reshape(rg3d.shape)
print(map_3d.shape)

# %%
# Unterbrechung

# # Now, we'll make a couple of handy definitions that will help us with plotting soon.
# # Here, we define a list for labeling the plots.
# xyz = ['x-y', 'x-z', 'y-z']
# # Here, another one to manage the extents.
# extents = [
#     (rg3d.x_min, rg3d.x_max, rg3d.y_min, rg3d.y_max),
#     (rg3d.x_min, rg3d.x_max, rg3d.z_min, rg3d.z_max),
#     (rg3d.y_min, rg3d.y_max, rg3d.z_min, rg3d.z_max),
# ]
# # And here, the max value for scaling the plots.
# mx = ac.L_p(np.max(map_3d))

# for i in range(3):
#     # To display this 3D result in 2D, we can integrate (sum) along one axis.
#     # This gives us a projection of the 3D sound field onto a 2D plane.
#     map_2d_sum = np.sum(map_3d, axis=2 - i)  # Sum along z-, y- or x-axis (in this order)

#     ax = fig.add_subplot(1, 3, i + 1)
#     ax.imshow(
#         ac.L_p(map_2d_sum.T),  # Transpose for correct orientation
#         vmax=mx,
#         vmin=mx - 20,
#         origin='lower',
#         interpolation='nearest',
#         extent=extents[i],
#     )
#     ax.set_xlabel(xyz[i][0])
#     ax.set_ylabel(xyz[i][2])
#     ax.set_title(f'{xyz[i]} view\n(grid increment: {rg3d.increment})')

ax = fig.add_subplot(projection='3d')

ax.plot(*mg.pos, 'o', label='mics')

ax.scatter(*map_3d)

# gpos = rg3d.pos.reshape((3, rg3d.nxsteps, rg3d.nysteps))
# ax.plot_wireframe(gpos[0], gpos[1], gpos[2], color='k', lw=0.2, label='grid')

ax.set(xlabel='x', ylabel='y', zlabel='z')
fig.legend()

plt.tight_layout()
plt.show()

# %%
# =========
# Line Grid
# =========

# Here, we create a microphone line geometry.
pos_total = np.zeros((3, 32))
pos_total[0, :] = np.linspace(-0.3, 0.3, 32)
line_mg = ac.MicGeom(pos_total=pos_total)
