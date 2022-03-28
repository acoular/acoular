# -*- coding: utf-8 -*-
"""
Example "Airfoil in open jet -- Beamforming".

Demonstrates barspectrum tool of Acoular.

A noise source emitting white noise is simulated. Its spectrum at a single 
microphone is plotted by using functions from acoular.tools.

Copyright (c)  Acoular Development Team.
All rights reserved.
"""

from acoular import WNoiseGenerator, PointSource, PowerSpectra, MicGeom, L_p
from acoular.tools import barspectrum
from numpy import array
from pylab import figure,plot,show,xlim,ylim,xscale,xticks,xlabel,ylabel,\
    grid,real, title, legend

# constants
sfreq= 12800 # sample frequency
band = 3 # octave: 1 ;   1/3-octave: 3 (for plotting)


# set up microphone at (0,0,0)
m = MicGeom()
m.mpos_tot = array([[0,0,0]])


# create noise source
n1 = WNoiseGenerator(sample_freq = sfreq, 
                     numsamples = 10*sfreq, 
                     seed = 1)

t = PointSource(signal = n1, 
                mics = m,  
                loc = (1, 0, 1))


# create power spectrum
f = PowerSpectra(time_data = t, 
                 window = 'Hanning', 
                 overlap = '50%', 
                 block_size = 4096)

# get spectrum data
spectrum_data = real(f.csm[:,0,0]) # get power spectrum from cross-spectral matrix
freqs = f.fftfreq() # FFT frequencies


# use barspectrum from acoular.tools to create third octave plot data
(f_borders, p, f_center) = barspectrum(spectrum_data, freqs, band, bar=True)
(f_borders_, p_, f_center_) = barspectrum(spectrum_data, freqs, band,bar=False)


# create figure with barspectra
figure(figsize=(20, 6))
title("Powerspectrum")
plot(f_borders,L_p(p), label="bar=True")
plot(f_borders_,L_p(p_),label="bar=False")
xlim(f_borders[0]*2**(-1./6),f_borders[-1]*2**(1./6))
ylim(50,90)
xscale('symlog')
label_freqs = [str(int(_)) for _ in f_center] # create string labels
xticks(f_center,label_freqs)
xlabel('f in Hz')
ylabel('SPL in dB')
grid(True)
legend()
show()