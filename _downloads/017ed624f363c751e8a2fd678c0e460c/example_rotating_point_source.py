# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
.. _rotating_point_source:

Rotating point source
=====================

Demonstrates the use of acoular for a point source moving on a circle trajectory.
Uses synthesized data.

Four different methods are compared:

* fixed focus time domain beamforming
* fixed focus frequency domain beamforming
* moving focus time domain beamforming
* moving focus time domain deconvolution
"""

import acoular as ac
import matplotlib.pyplot as plt
import numpy as np

# %%
# First, we make some important definitions:
#
# * the frequency of interest (114 Hz),
# * 1/3 octave band for later analysis,
# * the sampling frequency (3072 Hz),
# * the array radius (3.0),
# * the radius of the source trajectory (2.5),
# * the distance of the source trajectory from the array (4),
# * the revolutions per second (15/60),
# * the total number of revolutions (1.5).


freq = 6144.0 * 3 / 128.0
num = 3
sfreq = 6144.0 / 2
r = 3.0
R = 2.5
Z = 4
rps = 15.0 / 60.0
U = 1.5

# %%
# Construct the trajectory for the source. The source moves on a circle trajectory in
# anti-clockwise direction.


tr = ac.Trajectory()
tr1 = ac.Trajectory()
tmax = U / rps
delta_t = 1.0 / rps / 16.0  # 16 steps per revolution
for t in np.arange(0, tmax * 1.001, delta_t):
    i = t * rps * 2 * np.pi  # angle
    # define points for trajectory spline
    tr.points[t] = (R * np.cos(i), R * np.sin(i), Z)  # anti-clockwise rotation
    tr1.points[t] = (R * np.cos(i), R * np.sin(i), Z)  # anti-clockwise rotation

# %%
# Define a circular microphone array with 28 microphones.

m = ac.MicGeom()
m.pos_total = np.array(
    [
        (r * np.sin(2 * np.pi * i + np.pi / 4), r * np.cos(2 * np.pi * i + np.pi / 4), 0)
        for i in np.linspace(0.0, 1.0, 28, False)
    ],
).T

# %%
# Define the different source signals

num_samples = int(sfreq * tmax)
n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples)
s1 = ac.SineGenerator(sample_freq=sfreq, num_samples=num_samples, freq=freq)

# %%
# Define the moving source and one fixed source and mix their signals.
# The simulation output is cached by the :class:`~acoular.process.Cache` class.

p0 = ac.MovingPointSource(signal=s1, mics=m, trajectory=tr1)
# t = p0 # use only moving source
p1 = ac.PointSource(signal=n1, mics=m, loc=(0, R, Z))
t = ac.Mixer(source=p0, sources=[p1])
cached_mix = ac.Cache(source=t)

# t = p1 # use only fix source

# %%
# Optionally, save the signal of channel 0 and 14 to a wave file.

# ww = WriteWAV(source = t)
# ww.channels = [0,14]
# ww.save()

# %%
# Define the evaluation grid and the steering vector.

g = ac.RectGrid(x_min=-3.0, x_max=+3.0, y_min=-3.0, y_max=+3.0, z=Z, increment=0.3)
st = ac.SteeringVector(grid=g, mics=m)

# %%
# Plot the scene

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, num=1)
ax.plot(m.pos[0], m.pos[1], m.pos[2], 'o', label='mics')
gpos = g.pos.reshape((3, g.nxsteps, g.nysteps))
ax.plot_wireframe(gpos[0], gpos[1], gpos[2], color='k', lw=0.2, label='grid')
txyz = np.array(list(tr.traj(0, 4.1, 0.1)))
ax.plot(*txyz.T, 'k', label='trajectory')
ax.set(xlabel='x', ylabel='y', zlabel='z')
fig.legend()

# %%
# Fixed focus time domain beamforming
# -----------------------------------

fi = ac.FiltFiltOctave(source=cached_mix, band=freq, fraction='Third octave')
bt = ac.BeamformerTimeSq(source=fi, steer=st, r_diag=True)
avgt = ac.Average(source=bt, num_per_average=int(sfreq * tmax / 16))  # 16 single images
cacht = ac.Cache(source=avgt)  # cache to prevent recalculation

# %%
# Plot single frames. Note that the look direction is _towards_ the array. If you want a
# look direction _from_ the array (like a photo camera would do), the image needs to be mirrored.

plt.figure(2, (8, 7))
i = 1
map2 = np.zeros(g.shape)  # accumulator for average
for res in cacht.result(1):
    res0 = res[0].reshape(g.shape)
    map2 += res0  # average
    i += 1
    plt.subplot(4, 4, i)
    mx = ac.L_p(res0.max())
    plt.imshow(
        ac.L_p(np.transpose(res0)), vmax=mx, vmin=mx - 10, interpolation='nearest', extent=g.extend(), origin='lower'
    )
    plt.colorbar()
map2 /= i

plt.subplot(4, 4, 1)
plt.text(0.4, 0.25, 'fixed\nfocus', fontsize=15, ha='center')
plt.axis('off')
plt.tight_layout()


# %%
# Moving focus time domain beamforming
# ------------------------------------
# New grid needed, the trajectory starts at origin and is oriented towards +x
# thus, with the circular movement assumed, the center of rotation is at (0,2.5)

g1 = ac.RectGrid(
    x_min=-3.0,
    x_max=+3.0,
    y_min=-1.0,
    y_max=+5.0,
    z=0,
    increment=0.3,
)  # grid point of origin is at trajectory (thus z=0)
st1 = ac.SteeringVector(grid=g1, mics=m)
# beamforming with trajectory (rvec axis perpendicular to trajectory)
bts = ac.BeamformerTimeSqTraj(source=fi, steer=st1, trajectory=tr, rvec=np.array((0, 0, 1.0)))
avgts = ac.Average(source=bts, num_per_average=int(sfreq * tmax / 16))  # 16 single images
cachts = ac.Cache(source=avgts)  # cache to prevent recalculation

# %%
# Plot the scene with moving grid. We show three example positions of the grid when it get moved and
# swiveled along the trajectory.

fig, ax = plt.subplots(subplot_kw={'projection': '3d'}, num=3)
ax.plot(m.pos[0], m.pos[1], m.pos[2], 'o', label='mics')
# translation and direction of trajectory
for loc, dx, co in zip(tr.traj(0, 1.3, 0.6), tr.traj(0, 1.3, 0.6, der=1), 'krg'):
    dy = np.cross(bts.rvec, dx)  # new y-axis
    dz = np.cross(dx, dy)  # new z-axis
    RM = np.array((dx, dy, dz)).T  # rotation matrix
    RM /= np.sqrt((RM * RM).sum(0))  # column normalized
    tpos = np.dot(RM, g1.pos) + np.array(loc)[:, np.newaxis]  # rotation+translation
    gpos = tpos.reshape((3, g.nxsteps, g.nysteps))
    ax.plot_wireframe(gpos[0], gpos[1], gpos[2], color=co, lw=0.2)
    ax.plot(loc[0], loc[1], loc[2], 'o', color=co, label='grid origin')
txyz = np.array(list(tr.traj(0, 4.1, 0.1)))
ax.plot(*txyz.T, 'k', label='trajectory')
ax.set(xlabel='x', ylabel='y', zlabel='z')
fig.legend()

# %%
# Plot single frames

plt.figure(4, (8, 7))
i = 1
map3 = np.zeros(g1.shape)  # accumulator for average
for res in cachts.result(1):
    res0 = res[0].reshape(g1.shape)
    map3 += res0  # average
    i += 1
    plt.subplot(4, 4, i)
    mx = ac.L_p(res0.max())
    plt.imshow(
        ac.L_p(np.transpose(res0)), vmax=mx, vmin=mx - 10, interpolation='nearest', extent=g1.extend(), origin='lower'
    )
    plt.colorbar()
map3 /= i

plt.subplot(4, 4, 1)
plt.text(0.4, 0.25, 'moving\nfocus', fontsize=15, ha='center')
plt.axis('off')
plt.tight_layout()

# %%
# Moving focus time domain deconvolution
# --------------------------------------
# beamforming with trajectory (rvec axis perpendicular to trajectory)

bct = ac.BeamformerCleantSqTraj(source=fi, steer=st1, trajectory=tr, rvec=np.array((0, 0, 1.0)), n_iter=5)
avgct = ac.Average(source=bct, num_per_average=int(sfreq * tmax / 16))  # 16 single images
cachct = ac.Cache(source=avgct)  # cache to prevent recalculation

# %%
# Plot single frames

plt.figure(5, (8, 7))
i = 1
map4 = np.zeros(g1.shape)  # accumulator for average
for res in cachct.result(1):
    res0 = res[0].reshape(g1.shape)
    map4 += res0  # average
    i += 1
    plt.subplot(4, 4, i)
    mx = ac.L_p(res0.max())
    plt.imshow(
        ac.L_p(np.transpose(res0)), vmax=mx, vmin=mx - 10, interpolation='nearest', extent=g1.extend(), origin='lower'
    )
    plt.colorbar()
map4 /= i

plt.subplot(4, 4, 1)
plt.text(0.4, 0.25, 'moving\nfocus\ndeconvolution', fontsize=15, ha='center')
plt.axis('off')
plt.tight_layout()

# %%
# Fixed focus frequency domain beamforming
# ----------------------------------------

f = ac.PowerSpectra(
    source=cached_mix,
    window='Hanning',
    overlap='50%',
    block_size=128,
)
b = ac.BeamformerBase(freq_data=f, steer=st, r_diag=True)
map1 = b.synthetic(freq, num)


# %%
# Compare all four methods

plt.figure(6, (10, 3))
for i, map, tit in zip(
    (1, 2, 3, 4),
    (map1, map2, map3, map4),
    (
        'frequency domain\n fixed focus',
        'time domain\n fixed focus',
        'time domain\n fixed focus',
        'time domain\n deconvolution (moving focus)',
    ),
):
    plt.subplot(1, 4, i)
    mx = ac.L_p(map.max())
    plt.imshow(
        ac.L_p(np.transpose(map)), vmax=mx, vmin=mx - 10, interpolation='nearest', extent=g.extend(), origin='lower'
    )
    plt.colorbar(shrink=0.4)
    plt.title(tit)
plt.tight_layout()
plt.show()
