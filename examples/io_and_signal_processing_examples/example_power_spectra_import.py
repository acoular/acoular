# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Cross-spectral matrix import
============================

This example demonstrates how to import a cross-spectral matrix from a external source by means of
the :class:`acoular.spectra.PowerSpectraImport` class. The CSM is created numerically for a
frequency of 8 kHz.
"""

from pathlib import Path

import acoular as ac
import numpy as np

# %%
# Define the source positions and rms values of three sources as well as the microphone positions
loc1 = (-0.1, -0.1, -0.3)
loc2 = (0.15, 0, -0.3)
loc3 = (0, 0.1, -0.3)
rms = np.array([1, 0.7, 0.5])

micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
mg = ac.MicGeom(file=micgeofile)

# %%
# Obtain the transfer function of the monopole sources by using the SteeringVector object
st_src = ac.SteeringVector(grid=ac.ImportGrid(pos=np.array([loc1, loc2, loc3]).T), mics=mg)
H = st_src.transfer(8000).T  # transfer functions for 8000 Hz
H_h = H.transpose().conjugate()  # H hermetian

# %%
# Calculate the cross-spectral matrix for the three sources without noise

Q = np.diag(rms) ** 2  # matrix containing the source strength
csm = (H @ Q.astype(complex) @ H_h)[np.newaxis]  # calculate csm

# %%
# Import the cross-spectral matrix using the PowerSpectraImport object

ps_import = ac.PowerSpectraImport(csm=csm.copy(), frequencies=8000)

# %%
# Calculate the Beamforming result for the imported cross-spectral matrix

rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=mg)
bb = ac.BeamformerBase(freq_data=ps_import, steer=st, r_diag=False, cached=False)
pm = bb.synthetic(8000, 0)
Lm = ac.L_p(pm)

# %%
# Show the source map
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extend(), interpolation='bicubic')
plt.colorbar()
plt.show()
