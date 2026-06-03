# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Three sources -- Generate synthetic microphone array data.
==========================================================

Generates a test data set for three sources.

The simulation generates the sound pressure at 64 microphones that are
arrangend in the 'array64' geometry which is part of the package. The sound
pressure signals are sampled at 51200 Hz for a duration of 1 second.
The simulated signals are stored in a HDF5 file named 'three_sources.h5'.

Source location (relative to array center) and levels:

====== =============== ======
Source Location        Level
====== =============== ======
1      (-0.1,-0.1,0.3) 1 Pa
2      (0.15,0,0.3)    0.7 Pa
3      (0,0.1,0.3)     0.5 Pa
====== =============== ======


"""

from pathlib import Path
import matplotlib.pyplot as plt
import acoular as ac
import numpy as np


sfreq = 51200
block_size = 512
duration = 1
num_samples = duration * sfreq
micgeofile = Path(ac.__file__).parent / 'xml' / 'array_64.xml'
h5savefile = Path('three_sources.h5')

m = ac.MicGeom(file=micgeofile)
n1 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=1)
n2 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=2, rms=0.7)
n3 = ac.WNoiseGenerator(sample_freq=sfreq, num_samples=num_samples, seed=3, rms=0.5)
p1 = ac.PointSource(signal=n1, mics=m, loc=(-0.1, -0.1, -0.3))
p2 = ac.PointSource(signal=n2, mics=m, loc=(0.15, 0, -0.3))
p3 = ac.PointSource(signal=n3, mics=m, loc=(0, 0.1, -0.3))
p = ac.Mixer(source=p1, sources=[p2, p3])
fft = ac.RFFT(source=p, block_size=block_size, window='Rectangular')
rg = ac.RectGrid(x_min=-0.2, x_max=0.2, y_min=-0.2, y_max=0.2, z=-0.3, increment=0.01)
st = ac.SteeringVector(grid=rg, mics=m)
mf = ac.MaskedFreqOut(source=fft, freqs=[200, 4000]) # only for 4 kHz
sbl = ac.BeamformerSBL(
    source=mf,
    method="SBL1",
    steer=st,
    options={'n_iter':100}
)

out = next(sbl.result(num=1))
maps = out.reshape(out.shape[0], sbl.num_freqs, rg.size)


plt.figure(1)
Lm = ac.L_p(maps[0,0].reshape(rg.shape))
plt.title("200 Hz")
plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extent)
plt.colorbar()
plt.savefig('three_sources_sbl_200Hz.png', dpi=200, bbox_inches='tight')

plt.figure(2)
Lm = ac.L_p(maps[0,1].reshape(rg.shape))
plt.title("4 kHz")
plt.imshow(Lm.T, origin='lower', vmin=Lm.max() - 10, extent=rg.extent)
plt.colorbar()
plt.savefig('three_sources_sbl_4kHz.png', dpi=200, bbox_inches='tight')


