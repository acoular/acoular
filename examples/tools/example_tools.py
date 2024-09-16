# ------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
# ------------------------------------------------------------------------------
# %%
"""
Tools -- Demonstrates barspectrum tool of Acoular
=================================================

A noise source emitting white noise is simulated. Its spectrum at a single
microphone is plotted by using functions from acoular.tools.
"""

import acoular as ac
import numpy as np
from acoular.tools import barspectrum

#  Set up a single microphone at (0,0,0)
m = ac.MicGeom(mpos_tot=np.array([[0, 0, 0]]))

# Create a noise source
sample_freq = 12800  # sample frequency
n1 = ac.WNoiseGenerator(sample_freq=sample_freq, numsamples=10 * sample_freq, seed=1)

t = ac.PointSource(signal=n1, mics=m, loc=(1, 0, 1))

# create power spectrum
f = ac.PowerSpectra(source=t, window='Hanning', overlap='50%', block_size=4096, cached=False)

# get spectrum data
spectrum_data = np.real(f.csm[:, 0, 0])  # get power spectrum from cross-spectral matrix
freqs = f.fftfreq()  # FFT frequencies

# use barspectrum from acoular.tools to create third octave plot data
band = 3  # octave: 1 ;   1/3-octave: 3 (for plotting)
(f_borders, p, f_center) = barspectrum(spectrum_data, freqs, band, bar=True)
(f_borders_, p_, f_center_) = barspectrum(spectrum_data, freqs, band, bar=False)


# %%
# create figure with barspectra

from pylab import figure, grid, legend, plot, real, show, title, xlabel, xlim, xscale, xticks, ylabel, ylim

figure(figsize=(20, 6))
title('Powerspectrum')
plot(f_borders, ac.L_p(p), label='bar=True')
plot(f_borders_, ac.L_p(p_), label='bar=False')
xlim(f_borders[0] * 2 ** (-1.0 / 6), f_borders[-1] * 2 ** (1.0 / 6))
ylim(50, 90)
xscale('symlog')
label_freqs = [str(int(_)) for _ in f_center]  # create string labels
xticks(f_center, label_freqs)
xlabel('f in Hz')
ylabel('SPL in dB')
grid(True)
legend()
show()
