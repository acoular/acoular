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

# Let's start by importing Acoular and the other modules we'll need.
import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

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

# Next, we'll generate time sample data from an HDF5 file.
sfreq = 51200
duration = 1
num_samples = duration * sfreq

n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=1)
n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=2, rms=0.7)
n3 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=3, rms=0.5)
p1 = ac.PointSource(signal=n1, mics=mg, loc=(-0.1, -0.1, -0.25))
p2 = ac.PointSource(signal=n2, mics=mg, loc=(0.15, 0, -0.2))
p3 = ac.PointSource(signal=n3, mics=mg, loc=(0, 0.1, -0.15))
pa = ac.SourceMixer(sources=[p1, p2, p3])

# To prepare for beamforming, we need to calculate the frequency spectra of the signals.
# We'll do this using the PowerSpectra object.
ps = ac.PowerSpectra(source=pa, block_size=128, window='Hanning')
# Next, we'll create a SteeringVector object.
# Steering vectors are crucial for beamforming; they handle the time delays between microphones.
# Keep in mind that the steering vector requires a grid as input, which we'll provide later.
st = ac.SteeringVector(mics=mg)
# Finally, let's create a BeamformerBase object.
# This is the foundation for our beamforming algorithms, and it takes the frequency data and steering vectors as input.
bb = ac.BeamformerBase(freq_data=ps, steer=st)

# %%
# ================
# Rectangular Grid
# ================
#
# In this section, we'll explore how to create and use RectGrid objects in Acoular.
# RectGrid helps us define the spatial points where the beamforming output will be calculated.

# %%
# Basic RectGrid usage
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
# Another useful grid characteristic is its "extent".
# The "extent" is particularly helpful for visualization, especially with matplotlib's `imshow` function.
# It defines the grid's boundaries for plotting.
# Let's get the grid's extent:
print(f'Grid extent: {rg.extend()}')

# %%
# Now, let's see how to divide a RectGrid into subdomains.
# Subdomains allow us to focus our analysis on specific regions of interest within the grid.
# This can improve computational efficiency and help us concentrate on the most relevant areas.
# Here, we'll look at a subdomain in the first quadrant.
# To keep things clear, we'll use a smaller grid for this example.
rg.increment = 0.1  # Increase grid spacing for fewer points

# First, we'll create a rectangular sector in the first quadrant using the RectSector class.
rect_sector = ac.RectSector(x_min=0, x_max=1, y_min=0, y_max=1)

# Next, we'll get the indices of the grid points that fall within this rectangular sector.
subdomain_indices = rg.subdomain(rect_sector)
print('Subdomain indices:', subdomain_indices)

# Using these indices, we can extract the corresponding grid points from the grid.
# This allows us to work only with the points inside our subdomain.
x_indices, y_indices = subdomain_indices
subdomain_points = rg.pos[:, x_indices * rg.nysteps + y_indices]
print('Subdomain points:\n', subdomain_points)

# %%
# Coordinate Mapping
#
# Acoular provides handy methods for mapping between coordinates and grid indices.
# This is useful for finding the grid point closest to a given location
# or for defining subdomains based on coordinate values.
# Let's find the grid index closest to a specific coordinate:
x_coord = 0.05
y_coord = -0.05
index_of_coord = rg.index(x_coord, y_coord)
print(f'Index of ({x_coord}, {y_coord}): {index_of_coord}')

# %%
# We can also get indices for various types of subdomains: rectangular, circular, and polygonal.
# Let's try that:
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
# Now, let's visualize some beamforming results on our grid.
# First, we need to pass the grid to the steering vector.
st.grid = rg

# %%
# We can now calculate the beamforming output and display it in a plot.
fig = plt.figure(figsize=(12, 5))

# We'll do this for three different grid increments to see the effect of resolution.
for i in range(3):
    # Increase grid increment
    rg.increment = 0.01 * (i+1)

    # Calculate beamforming output
    pm = bb.synthetic(8000, 3)
    Lm = ac.L_p(pm)

    # Create the plot
    ax = fig.add_subplot(1, 3, i+1)
    ax.imshow(
        Lm.T,
        origin='lower',
        vmin=Lm.max() - 10,
        extent=rg.extend(),
        interpolation='bicubic',
    )
    ax.set_title(f'Beamforming map\n(Grid increment: {rg.increment})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

plt.tight_layout()
plt.show()

# %%
# Notice how the results become more precise as the grid increment decreases.

# %%
# =======
# 3D Grid
# =======
#
# Let's briefly introduce 3D grids.
# For more in-depth examples of 3D beamforming, check out the 'example_3d_beamforming.py' example.
#
# Basic RectGrid3D Usage

# To create a 3D grid, we use the RectGrid3D object.
rg3d = ac.RectGrid3D(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z_min=-0.3, z_max=-0.1, increment=0.01)
print(rg3d.size)

# We can also apply different increments for each dimension if needed.
rg3d.increment = (0.02, 0.02, 0.01)
print(rg3d.size)

# %%
# 3D Grid Properties
#
# Accessing properties of a 3D grid is similar to 2D grids.
# The main difference is that we now have 'nzsteps' and a 3D 'shape'.
# Let's take a look:
# Access 3D grid properties
print(f'3D Grid size: {rg3d.size}')
print(f'Number of x steps: {rg3d.nxsteps}')
print(f'Number of y steps: {rg3d.nysteps}')
print(f'Number of z steps: {rg3d.nzsteps}')
print(f'3D Grid shape: {rg3d.shape}')
print(f'3D Grid positions:\n{rg3d.pos[:, :5]}')

# %%
# 3D Visualization
#
# Let's visualize 3D beamforming results using our example data.
# For this, we'll need a 3D beamforming method. We'll use BeamformerBase,
# but in a real-world scenario, you might choose a more advanced technique.

# First, ensure the steering vector is set up for 3D grids.
st.grid = rg3d

# Let's display the results.
fig = plt.figure(figsize=(12, 10))

for j in range(3):
    # Again we will look at the affect of different increments.
    rg3d.increment = 0.01 * (i+1)

    # Calculate the beamforming output.
    # Note: For true 3D beamforming, the result is a 3D field.
    #       Here, we'll calculate it for the third-octave band around a frequency of 8000.
    map_3d = bb.synthetic(8000, 3)  # shape will be (number of grid points)

    #  To visualize this 3D data, we need to reshape it to 3D.
    map_3d = map_3d.reshape(rg3d.shape)

    # Let's define a handy list for labeling the plots...
    xyz = ['x-y', 'x-z', 'y-z']
    # and one to manage the extents.
    extents = [
        (rg3d.x_min, rg3d.x_max, rg3d.y_min, rg3d.y_max),
        (rg3d.x_min, rg3d.x_max, rg3d.z_min, rg3d.z_max),
        (rg3d.y_min, rg3d.y_max, rg3d.z_min, rg3d.z_max)
    ]
    # And the max value for scaling the plots
    mx = ac.L_p(np.max(map_3d))

    for i in range(3):
        # To display this 3D result in 2D, we can integrate (sum) along one axis.
        # This gives us a projection of the 3D sound field onto a 2D plane.
        map_2d_sum = np.sum(map_3d, axis=2-i) # Sum along z-, y- or x-axis (in this order)

        ax = fig.add_subplot(3, 3, (3*j)+i+1)
        ax.imshow(
            ac.L_p(map_2d_sum.T),  # Transpose for correct orientation
            vmax=mx,
            vmin=mx - 20,
            origin='lower',
            interpolation='nearest',
            extent=extents[i],
        )
        ax.set_xlabel(xyz[i][0])
        ax.set_ylabel(xyz[i][2])
        ax.set_title(f'Top view ({xyz[i]})')

plt.tight_layout()
plt.show()