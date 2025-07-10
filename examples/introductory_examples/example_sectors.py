# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Sectors
=======

This example demonstrates how to use a sector for spatial integration in Acoular.
It uses the airfoil-in-open-jet dataset, places a rectangular sector on the trailing edge of the airfoil,
integrates the beamforming result over this sector, and plots the normal and the third-octave
spectra using the :func:`~acoular.tools.helpers.barspectrum` function.
"""


# %%
# ===========
# Preparation
# ===========
#
# First, we import the modules and functions we are going to use in this example.

from pathlib import Path

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np
from acoular.tools import barspectrum
from acoular.tools.helpers import get_data_file

# %%
# Also, we make sure we have the example data we need.
#
# If the calibration or measurement files aren't present, we will download them automatically.
# This way, you can run the example out of the box!

# Download example data if necessary
time_data_file = get_data_file('example_data.h5')
calib_file = get_data_file('example_calib.xml')

ts = ac.MaskedTimeSamples(
    file=time_data_file,
    invalid_channels=[1, 7],
    start=0,
    stop=16000,
)
calib = ac.Calib(source=ts, file=calib_file, invalid_channels=[1, 7])
mics = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_56.xml', invalid_channels=[1, 7])
grid = ac.RectGrid(x_min=-0.6, x_max=0.0, y_min=-0.3, y_max=0.3, z=-0.68, increment=0.1)
env = ac.Environment(c=346.04)
st = ac.SteeringVector(grid=grid, mics=mics, env=env)
f = ac.PowerSpectra(source=calib, window='Hanning', overlap='50%', block_size=128)
bb = ac.BeamformerBase(freq_data=f, steer=st)


# %%
# ===============
# Define a Sector
# ===============
#
# Let us start by defining a sector.
#
# One can think of a sector as a region in the sound map to which the analysis is constrained.
#
# Here, we will create a rectangular sector that covers the trailing edge of the airfoil.
# Oftentimes, this is where interesting noise sources reside in practice.

sector = ac.RectSector(x_min=-0.3, x_max=-0.25, y_min=-0.05, y_max=0.05)

# %%
# To see how our sector and airfoil look together, we visulaize the airfoil's front and trailing
# edges (as dashed lines), the sector (as a light blue rectangle), and the grid points (where the
# sound field is calculated). This helps us see exactly what region we are integrating over later.
#
# Also we calculate the beamforming results for a frequency of 4000 Hz
# to see why the airfoil's trailing edge is of such interest.

# Calculate beamforming results for 4000 Hz
map = bb.synthetic(4000)
spl_map = ac.L_p(map)

# Display beamforming results for 4000 Hz
plt.imshow(spl_map.T, origin='lower', vmin=spl_map.max() - 10, extent=grid.extend(), interpolation='bicubic')
# Plot airfoil's edges as dashed lines (we is they are at x = -0.33 and x = -0.29)
plt.vlines([-0.33, -0.29], ymin=-0.3, ymax=0.3, linestyles='--', label='Airfoil edges')
# Plot sector as a filled rectangle
plt.fill_between([sector.x_min, sector.x_max], sector.y_min, sector.y_max, alpha=0.5, label='Sector')
# Scatter the grid points.
plt.scatter(*grid.pos[0:2], c='lightgray', s=10, label='Grid Points')

cbar = plt.colorbar()
cbar.set_label('$L_p$ / dB')
plt.xlabel('x / m')
plt.ylabel('y / m')
plt.title('Airfoil, sector, grid, and beamforming map')
plt.legend()
plt.show()


# %%
# =======================
# Grid Points of Interest
# =======================
#
# Acoular uses sector classes to define regions of interest in your grid. Each sector
# class provides a method called :meth:`~acoular.grids.Sector.contains` that takes a
# grid's :attr:`~acoular.grids.Grid.pos` attribute as parameter and checks which grid
# points are inside the sector.
#
# For `N` grids points the :meth:`~acoular.grids.Sector.contains` method returns a
# boolean mask of length `N`. Grid points inside the sector are represented as ``True``.

mask = sector.contains(grid.pos)
print('x: {}, y: {}'.format(*grid.pos[:2, mask]))
print(mask)

# %%
# We see that the one grid point represented as ``True`` in the mask corresponds to the
# ``(-0.3, 0.0)`` point in the x-y plane. From looking at the figure above and at the sector's
# definition, we know that this grid point does not lie `inside` the sector but on its border.
# To prevent this inclusion of the bordering points, we may change the sector's
# :attr:`~acoular.grids.SingleSector.include_border` attribute to ``False``.

sector.include_border = False
mask = sector.contains(grid.pos)
print('x: {}, y: {}'.format(*grid.pos[:2, mask]))

# %%
# Note that the same point is still contained by the sector. This is because of the
# :attr:`~acoular.grids.SingleSector.default_nearest` attribute, which – if no grid
# points lie inside the sector – picks the grid point closest to the sector's center.
# By setting it to ``False`` we can use only points that definitely lie `inside` the
# sector – which are none in this case.

sector.default_nearest = False
mask = sector.contains(grid.pos)
print('x: {}, y: {}'.format(*grid.pos[:2, mask]))

# %%
# Also, the tolerance for deciding whether a grid point lies on a sector's border can be
# changed by varying the sector's :attr:`~acoular.grids.SingleSector.abs_tol` attribute.
#
# By defining which grid points lie inside the sector this way, we determine which points
# the beamformer's :meth:`~acoular.fbeamform.BeamformerBase.integrate` method and Acoular's
# :func:`~acoular.fbeamform.integrate` function sum over.

# %%
# =====================
# Integrate Over Sector
# =====================
#
# Now for the main event: integrating the beamforming result over our sector. This step gives
# us the total sound pressure level (SPL) coming from just our region of interest. We can use
# the beamformer's :meth:`~acoular.fbeamform.BeamformerBase.integrate` method or Acoular's
# :func:`~acoular.fbeamform.L_p` function to get the sector's SPL.
#
# Right now, the sector does not contain any grid points and therefore the SPL will be zero for
# all frequencies. To reinsert the ``(-0.3, 0.0)`` point into the sector and get beamforming
# results, we need to change either the :attr:`~acoular.grids.SingleSector.default_nearest`
# attribute or the :attr:`~acoular.grids.SingleSector.include_border` attribute back to ``True``.
#
# We look at the spectrum of the SPL over the power spectrum's FFT frequencies.

sector.default_nearest = True

freqs = f.fftfreq()
bf_sector = bb.integrate(sector)
spl_sector = ac.L_p(bf_sector)
spl_sector = np.where(spl_sector > 0, spl_sector, 0)  # Keep positive entries only

plt.figure(figsize=(8, 5))

plt.plot(freqs / 1000, spl_sector)

plt.xlabel('Frequency / kHz')
plt.ylabel('$L_p$ / dB')
plt.title('Integrated spectrum over sector')
plt.grid()
plt.show()


# %%
# ========================
# Third-Octave Barspectrum
# ========================
#
# Using Acoular's :func:`~acoular.tools.helpers.barspectrum` function
# we can easily convert our integrated spectrum into third-octave bands.

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
# All sector types can be used with the integrate function to sum or average
# results over the selected region.
