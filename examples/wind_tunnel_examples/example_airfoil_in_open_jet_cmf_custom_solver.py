# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Airfoil in open jet -- Covariance matrix fitting with custom solvers.
=====================================================================

Demonstrates how to pass custom solver instances to
:class:`~acoular.fbeamform.BeamformerCMF`.
Uses measured data in file example_data.h5, calibration in file example_calib.xml,
microphone geometry in array_56.xml (part of Acoular).
"""

from pathlib import Path

import acoular as ac
from acoular.solvers import ISTACV, IntermediateResultCallback, PylopsLeastSquaresSolver
from acoular.tools.helpers import get_data_file

if not ac.config.have_pylops:
    msg = 'This example requires PyLops. Install acoular with optional PyLops dependencies.'
    raise ImportError(msg)

# %%
# The 4 kHz third-octave band is used for the example.

cfreq = 4000
num = 3

# %%
# Obtain necessary data

time_data_file = get_data_file('example_data.h5')
calib_file = get_data_file('example_calib.xml')

# %%
# Setting up the processing chain for :class:`~acoular.fbeamform.BeamformerCMF`.
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

# %%
# Configure custom solvers:
#
# * a plain FISTA solver
# * an ISTA solver with cross-validated regularization

save_iters = [1, 10, 100, 1000]
intermediate_callback = IntermediateResultCallback(n_iter=save_iters)

fista_solver = PylopsLeastSquaresSolver(
    method='FISTA',
    options={
        'niter': 1000,
        'eps': 1e-8,
        'alpha': None,
        'tol': 0.0,
        'show': False,
    },
)

cv_solver = ISTACV(
    solver=PylopsLeastSquaresSolver(
        method='ISTA',
        options={
            'niter': 1000,
            'tol': 1e-10,
            'show': False,
        },
        callback=intermediate_callback,
    ),
    num_grid=10,
    reg_ratio=1e-2,
    cv=5,
)

# %%
# Create beamformers and compute maps.

bf_fista = ac.BeamformerCMF(freq_data=f, steer=st, solver=fista_solver)
bf_cv = ac.BeamformerCMF(freq_data=f, steer=st, solver=cv_solver)

# Reset callback before running to ensure clean state
intermediate_callback.x_steps = []
intermediate_callback.i_steps = []
intermediate_callback.x_by_iter = {}

# Run one single-frequency solve to capture intermediate iterates from final ISTACV solve.
# Note: CV performs multiple solves (one per fold), but callback captures only the final solve.
# Intermediate results are automatically in absolute scale (callback handles rescaling).
map_cv_single = bf_cv.synthetic(cfreq, 1)

intermediate_maps = {
    iiter: intermediate_callback.x_by_iter[iiter].reshape(grid.shape)
    for iiter in save_iters
    if iiter in intermediate_callback.x_by_iter
}

map_fista = bf_fista.synthetic(cfreq, num)
map_cv = bf_cv.synthetic(cfreq, num)

# %%
# Plot result maps.

import matplotlib.pyplot as plt

plt.figure(1, (10, 4.5))

plt.subplot(1, 2, 1)
mx_fista = ac.L_p(map_fista.max())
plt.imshow(
    ac.L_p(map_fista.T),
    vmax=mx_fista,
    vmin=mx_fista - 15,
    origin='lower',
    interpolation='nearest',
    extent=grid.extent,
)
plt.colorbar(shrink=0.6)
plt.title('CMF + custom FISTA solver')

plt.subplot(1, 2, 2)
mx_cv = ac.L_p(map_cv.max())
plt.imshow(
    ac.L_p(map_cv.T),
    vmax=mx_cv,
    vmin=mx_cv - 15,
    origin='lower',
    interpolation='nearest',
    extent=grid.extent,
)
plt.colorbar(shrink=0.6)
plt.title('CMF + ISTACV-wrapped solver')

print(cv_solver.output.keys())

# %%
# Plot intermediate ISTACV results at selected iterations.

plt.figure(2, (10, 8))

for iplot, iiter in enumerate(save_iters, start=1):
    plt.subplot(2, 2, iplot)
    if iiter in intermediate_maps:
        map_iter = intermediate_maps[iiter]
        mx_iter = ac.L_p(map_iter.max())
        plt.imshow(
            ac.L_p(map_iter.T),
            vmax=mx_iter,
            vmin=mx_iter - 15,
            origin='lower',
            interpolation='nearest',
            extent=grid.extent,
        )
        plt.colorbar(shrink=0.6)
    plt.title(f'ISTACV iteration {iiter}')

plt.tight_layout()
plt.savefig('example_airfoil_in_open_jet_cmf_custom_solver.png', dpi=300)
plt.show()
