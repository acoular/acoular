# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Airfoil in open jet -- Time domain beamforming methods.
=======================================================

Demonstrates different microphone array methods operating in the time domain.
Uses measured data in file example_data.h5, calibration in file example_calib.xml,
microphone geometry in array_56.xml (part of Acoular).
"""

from pathlib import Path

import acoular as ac
import numpy as np
from acoular.tools.helpers import get_data_file

# %%
# The 4 kHz third-octave band is used for the example.

cfreq = 4000
num = 3

# %%
# Obtain necessary data

time_data_file = get_data_file('example_data.h5')
calib_file = get_data_file('example_calib.xml')


# %%
# Setting up the processing chain for the time domain methods.
#
# .. hint::
#    An in-depth explanation for setting up the time data, microphone geometry, environment and
#    steering vector is given in the example :doc:`example_airfoil_in_open_jet_steering_vectors`.

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

# %%
# First, classic delay-and-sum beamforming in time domain is set up using
# the :class:`acoular.tbeamform.BeamformerTime` class.
# To produce an image of the sound sources, the beamformer time signal output for each grid-point
# is zero-phase filtered, squared and block-wise averaged over time.
# The result is cached to disk to prevent recalculation.

bt = ac.BeamformerTime(source=calib, steer=st)
ft = ac.FiltFiltOctave(source=bt, band=cfreq)
pt = ac.TimePower(source=ft)
avgt = ac.Average(source=pt, num_per_average=1024)
cacht = ac.Cache(source=avgt)  # cache to prevent recalculation

# %%
# Second, by using the :class:`acoular.tbeamform.BeamformerTimeSq` class, the squared output of the
# beamformer is calculated directly. It also allows for the removal of the autocorrelation, which is
# similar to the removal of the cross spectral matrix diagonal.

fi = ac.FiltFiltOctave(source=calib, band=cfreq)
bts = ac.BeamformerTimeSq(source=fi, steer=st, r_diag=True)
avgts = ac.Average(source=bts, num_per_average=1024)
cachts = ac.Cache(source=avgts)  # cache to prevent recalculation

# %%
# Third, CLEAN deconvolution in the time domain (CLEAN-T) is applied, using the
# :class:`acoular.tbeamform.BeamformerCleant` class.
fct = ac.FiltFiltOctave(source=calib, band=cfreq)
bct = ac.BeamformerCleant(source=fct, steer=st, n_iter=20, damp=0.7)
ptct = ac.TimePower(source=bct)
avgct = ac.Average(source=ptct, num_per_average=1024)
cachct = ac.Cache(source=avgct)  # cache to prevent recalculation

# %%
# Finally, squared signals with autocorrelation removal can be obtained by using the
# :class:`acoular.tbeamform.BeamformerCleantSq` class.
fcts = ac.FiltFiltOctave(source=calib, band=cfreq)
bcts = ac.BeamformerCleantSq(source=fcts, steer=st, n_iter=20, damp=0.7, r_diag=True)
avgcts = ac.Average(source=bcts, num_per_average=1024)
cachcts = ac.Cache(source=avgcts)  # cache to prevent recalculation

# %%
# Plot result maps for different beamformers in time domain

import matplotlib.pyplot as plt

ftitles = ['BeamformerTime', 'BeamformerTimeSq', 'BeamformerCleant', 'BeamformerCleantSq']
i2 = 1  # no of figure
i1 = 1  # no of subplot
for b in (cacht, cachts, cachct, cachcts):
    # first, plot time-dependent result (block-wise)
    fig = plt.figure(i2, (7, 7))
    fig.suptitle(f'{ftitles[i2 - 1]}: block-wise source maps (f={cfreq} Hz)')
    i2 += 1
    res = np.zeros(grid.size)  # init accumulator for average
    i3 = 1  # no of subplot
    for r in b.result(1):  # one single block
        plt.subplot(4, 4, i3)
        i3 += 1
        res += r[0]  # average accum.
        map = r[0].reshape(grid.shape)
        mx = ac.L_p(map.max())
        plt.imshow(ac.L_p(map.T), vmax=mx, vmin=mx - 15, origin='lower', interpolation='nearest', extent=grid.extent)
        plt.title(f'{(i3 - 1) * 1024}')
    res /= i3 - 1  # average
    plt.tight_layout()

    # second, plot overall result (average over all blocks)
    fig = plt.figure(10, (8, 2))
    fig.suptitle(f'Averaged source maps (f={cfreq} Hz)')
    plt.subplot(1, 4, i1)
    i1 += 1
    map = res.reshape(grid.shape)
    mx = ac.L_p(map.max())
    plt.imshow(ac.L_p(map.T), vmax=mx, vmin=mx - 15, origin='lower', interpolation='nearest', extent=grid.extent)
    plt.colorbar(shrink=0.5)
    plt.title(('BeamformerTime', 'BeamformerTimeSq', 'BeamformerCleant', 'BeamformerCleantSq')[i2 - 2])
plt.tight_layout()
plt.show()
