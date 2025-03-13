# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Airfoil in open jet -- Covariance matrix fitting (CMF).
=======================================================

Demonstrates the inverse CMF method with different solvers.
Uses measured data in file example_data.h5, calibration in file example_calib.xml,
microphone geometry in array_56.xml (part of Acoular).
"""

import urllib
from pathlib import Path

import acoular as ac

# %%
# The 4 kHz third-octave band is used for the example.

cfreq = 4000
num = 3

# %% Obtain necessary data

calib_file = Path('../data/example_calib.xml')
if not calib_file.exists():
    calib_file = Path().cwd() / 'example_calib.xml'
    if not calib_file.exists():
        print('Cannot find calibration file. Downloading...')
        url = 'https://github.com/acoular/acoular/tree/master/examples/data/example_calib.xml'
        urllib.request.urlretrieve(url, calib_file)
    print(f'Calibration file location: {calib_file}')

time_data_file = Path('../data/example_data.h5')
if not time_data_file.exists():
    time_data_file = Path().cwd() / 'example_data.h5'
    if not time_data_file.exists():
        print('Cannot find example_data.h5 file. Downloading...')
        url = 'https://github.com/acoular/acoular/tree/master/examples/data/example_data.h5'
        time_data_file, _ = urllib.request.urlretrieve(url, time_data_file)
    print(f'Time data file location: {time_data_file}')

# %%
# Setting up the processing chain for :class:`acoular.fbeamform.BeamformerCMF` methods.
#
# .. hint::
#    A step-by-step explanation for setting up the processing chain is given in the example
#    :doc:`example_airfoil_in_open_jet_steering_vectors`.

ts = ac.MaskedTimeSamples(
    file=time_data_file,
    invalid_channels=[1, 7],
    start=0,
    stop=16000,
)
calib = ac.Calib(source=ts, file=calib_file, invalid_channels=[1, 7])
mics = ac.MicGeom(file=Path(ac.__file__).parent / 'xml' / 'array_56.xml', invalid_channels=[1, 7])
grid = ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=-0.68, increment=0.05)
env = ac.Environment(c=346.04)
st = ac.SteeringVector(grid=grid, mics=mics, env=env)
f = ac.PowerSpectra(source=calib, window='Hanning', overlap='50%', block_size=128)
b = ac.BeamformerCMF(freq_data=f, steer=st, alpha=1e-8)

# %%
# Plot result maps for CMF with different solvers from `SciPy <https://scipy.org/>`_ and
# `scikit-learn <https://scikit-learn.org/stable/>`_, including:
#
# * LassoLars
# * LassoLarsBIC
# * OMPCV
# * NNLS
# * fmin_l_bfgs_b

import matplotlib.pyplot as plt

plt.figure(1, (10, 7))  # no of figure
i1 = 1  # no of subplot

for method in ('LassoLars', 'LassoLarsBIC', 'OMPCV', 'NNLS', 'fmin_l_bfgs_b'):
    b.method = method
    plt.subplot(2, 3, i1)
    i1 += 1
    map = b.synthetic(cfreq, 1)
    mx = ac.L_p(map.max())
    plt.imshow(ac.L_p(map.T), vmax=mx, vmin=mx - 15, origin='lower', interpolation='nearest', extent=grid.extend())
    plt.colorbar(shrink=0.5)
    plt.title(b.method)

plt.tight_layout()
plt.show()
