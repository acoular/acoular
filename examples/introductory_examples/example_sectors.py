# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Sectors
=======

This example demonstrates how to use a sector for spatial integration in Acoular.
It uses the airfoil-in-open-jet dataset, places a rectangular sector behind the airfoil,
integrates the beamforming result over this sector, and plots the normal and the third-octave
spectra using the :func:`~acoular.tools.barspectrum` function.
"""

# %%
# First, we import the modules and fucntions we are going to use in this example.

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

from acoular.tools import barspectrum
from acoular.tools.helpers import get_calib_file, get_time_data_file

# %%
# ================
# Data Preparation
# ================
#
# Let's start by making sure we have the data we need.
# If the calibration or measurement files aren't present, we'll download them automatically.
# This way, you can run the example out of the box!

cfreq = 4000  # Center frequency for demonstration

# Download example data if necessary
calib_file = get_calib_file()
time_data_file = get_time_data_file()

ts = ac.MaskedTimeSamples(
    file=time_data_file,
    invalid_channels=[1, 7],
    start=0,
    stop=16000,
)
calib = ac.Calib(source=ts, file=calib_file, invalid_channels=[1, 7])
mics = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_56.xml', invalid_channels=[1, 7])
grid = ac.RectGrid(x_min=-0.6, x_max=0.0, y_min=-0.3, y_max=0.3, z=-0.68, increment=0.05)
env = ac.Environment(c=346.04)
st = ac.SteeringVector(grid=grid, mics=mics, env=env)
f = ac.PowerSpectra(source=calib, window='Hanning', overlap='50%', block_size=128)
bb = ac.BeamformerBase(freq_data=f, steer=st)

airfoil = np.array([[-0.2, -0.5], [0.1, -0.1]])


# %%
# ===============
# Define a Sector
# ===============
#
# Now let's define a sector. Think of a sector as a region in the sound map where you want to focus
# your analysis. Here, we'll create a rectangular sector that covers the trailing edge of the
# airfoil. This is often where interesting noise sources are found in aeroacoustic measurements.

sector = ac.RectSector(x_min=-0.3, x_max=-0.2, y_min=-0.15, y_max=0.15)


# %%
# Let's see how our sector and airfoil look together. We'll plot the airfoil (as a rectangle for
# illustration), the sector (as a light blue rectangle), and the grid points (where the sound field
# is calculated). This helps you see exactly what region you're integrating over.

plt.figure(figsize=(7, 5))

# Plot airfoil and sector as a filled rectangles
plt.fill_between(airfoil[0], *airfoil[1], label='Airfoil')
plt.fill_between([sector.x_min, sector.x_max], sector.y_min, sector.y_max, alpha=0.5, label='Sector')

# Scatter the grid points.
plt.scatter(*grid.pos[0:2], c='lightgray', s=10, label='Grid Points')

plt.xlabel('x / m')
plt.ylabel('y / m')
plt.title('Airfoil, sector, and grid')
plt.legend()
plt.show()


# %%
# =====================
# Integrate Over Sector
# =====================
#
# Now for the main event: integrating the beamforming result over our sector! This step gives
# you the total sound pressure level (SPL) coming from just your region of interest.
freqs = f.fftfreq()
bf_sector = bb.integrate(sector)
spl_sector = ac.L_p(bf_sector)
spl_sector = np.where(spl_sector > 0, spl_sector, 0)  # Keep positve entries only

print(freqs.shape, spl_sector.shape)
print(freqs)
print(spl_sector)

plt.figure(figsize=(8, 5))

plt.plot(freqs / 1000, spl_sector)

plt.xlabel('Frequency / kHz')
plt.ylabel('$L_p$ / dB')
plt.title('Integrated spectrum over sector')
plt.grid()
plt.show()


# %%
# ========================
# Third-octave Barspectrum
# ========================
#
# Let's convert our integrated spectrum into third-octave bands
# using Acoular's :func:`~acoular.tools.barspectrum` function.

f_borders, spl_bars, f_center = barspectrum(spl_sector, freqs, 3, bar=True)

plt.figure(figsize=(8, 5))

plt.fill_between(f_borders, spl_bars)

plt.xscale('log')
plt.xlabel('Frequency / Hz')
plt.ylabel('$L_p$ / dB')
plt.title('Integrated spectrum over sector (third octave)')
plt.grid()
plt.show()


# %%
# =============
# Other Sectors
# =============
#
# Acoular provides several other sector types for flexible spatial integration:
#
# - :class:`~acoular.grids.RectSector`: Rectangle in the x-y plane (as used above).
# - :class:`~acoular.grids.CircSector`: Circular sector in the x-y plane.
# - :class:`~acoular.grids.PolySector`: Arbitrary polygonal sector in the x-y plane.
# - :class:`~acoular.grids.ConvexSector`: Convex hull sector in the x-y plane.
# - :class:`~acoular.grids.RectSector3D`: 3D cuboid sector (for 3D grids).
# - :class:`~acoular.grids.MultiSector`: Combines multiple sectors of any type.
#
# All sector types can be used with the integrate function
# to sum or average results over the selected region.
