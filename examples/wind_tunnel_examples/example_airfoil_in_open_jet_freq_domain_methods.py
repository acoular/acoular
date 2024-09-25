# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Airfoil in open jet -- Frequency domain beamforming methods.
============================================================

Demonstrates different microphone array methods operating in the frequency domain.
Uses measured data in file example_data.h5, calibration in file example_calib.xml,
microphone geometry in array_56.xml (part of Acoular).

"""

import urllib
from pathlib import Path

import acoular as ac

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
# The 4 kHz third-octave band is used for the example.

cfreq = 4000
num = 3

# %%
# Setting up the processing chain for the frequency domain methods.
#
# .. hint::
#    An in-depth explanation for setting up the processing chain is given in the example :doc:`example_airfoil_in_open_jet_steering_vectors`.


ts = ac.MaskedTimeSamples(
    name=time_data_file,
    invalid_channels=[1, 7],
    start=0,
    stop=16000,
    calib=ac.Calib(from_file=calib_file),
)
mics = ac.MicGeom(from_file=Path(ac.__file__).parent / 'xml' / 'array_56.xml', invalid_channels=[1, 7])
grid = ac.RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, z=0.68, increment=0.05)
env = ac.Environment(c=346.04)
st = ac.SteeringVector(grid=grid, mics=mics, env=env)
f = ac.PowerSpectra(source=ts, window='Hanning', overlap='50%', block_size=128)


# %%
# Here, different frequency domain beamformers defined in the module :mod:`acoular.fbeamform` are used and the corresponding result maps are calculated by
# evaluating the :meth:`acoular.fbeamform.BeamformerBase.synthetic` method with the desired frequency and bandwidth.

bb = ac.BeamformerBase(freq_data=f, steer=st, r_diag=True)
bc = ac.BeamformerCapon(freq_data=f, steer=st, cached=False)
be = ac.BeamformerEig(freq_data=f, steer=st, r_diag=True, n=54)
bm = ac.BeamformerMusic(freq_data=f, steer=st, n=6)
bd = ac.BeamformerDamas(freq_data=f, steer=st, r_diag=True, n_iter=100)
bdp = ac.BeamformerDamasPlus(freq_data=f, steer=st, r_diag=True, n_iter=100)
bo = ac.BeamformerOrth(freq_data=f, steer=st, r_diag=True, eva_list=list(range(38, 54)))
bs = ac.BeamformerCleansc(freq_data=f, steer=st, r_diag=True)
bcmf = ac.BeamformerCMF(freq_data=f, steer=st, method='LassoLarsBIC')
bl = ac.BeamformerClean(freq_data=f, steer=st, r_diag=True, n_iter=100)
bf = ac.BeamformerFunctional(freq_data=f, steer=st, r_diag=False, gamma=4)
bgib = ac.BeamformerGIB(freq_data=f, steer=st, method='LassoLars', n=10)

# %%
# Plot result maps for different beamformers in frequency domain

from pylab import colorbar, figure, imshow, show, subplot, tight_layout, title

figure(1, (10, 6))
i1 = 1  # no of subplot
for b in (bb, bc, be, bm, bl, bo, bs, bd, bcmf, bf, bdp, bgib):
    subplot(3, 4, i1)
    i1 += 1
    map = b.synthetic(cfreq, num)
    mx = ac.L_p(map.max())
    imshow(ac.L_p(map.T), origin='lower', vmin=mx - 15, interpolation='nearest', extent=grid.extend())
    colorbar()
    title(b.__class__.__name__)
    tight_layout()
show()
