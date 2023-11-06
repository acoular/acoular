#------------------------------------------------------------------------------
# Copyright (c) Acoular Development Team.
#------------------------------------------------------------------------------
#
# demonstrates band pass filter characteristics and shows
# - unfiiltered power spectrum
# - three adjacent octave bands filtered spectra
# - sum of all octave band spectra
# 
# ----------------------------------------------------------------

from acoular import (
    config,
    WNoiseGenerator,
    MicGeom,
    PointSource,
    FiltFiltOctave,
    PowerSpectra,
    tools
)
import numpy as np
import matplotlib.pyplot as pl
config.global_caching = 'none'
numsamples = 51200*10
n1 = WNoiseGenerator(sample_freq=51200, numsamples=numsamples, seed=1)
m1 = MicGeom(mpos_tot=[[1], [1], [1]])
p1 = PointSource(signal=n1, mics=m1)
f1 = FiltFiltOctave(source = p1, band = 10, fraction = 'Octave')
hs = 0.0
res0 = 0.0
for i in np.arange(2.1,3.0,0.3):
    #f1.order = 2
    f1.band = 10**i
    res = (tools.return_result(f1)**2).sum()
    res0 += res
    ps = PowerSpectra(time_data=f1,block_size=8192)
    print(res/numsamples,ps.csm[:,0,0].real.sum())
    x = ps.fftfreq()
    y = 10*np.log10(ps.csm[:,0,0].real)
    pl.semilogx(x,y)
    hs = hs + ps.csm[:,0,0].real
y = 10 * np.log10(abs(hs))
pl.semilogx(x,y)
ps = PowerSpectra(time_data=p1,block_size=8192)
x = ps.fftfreq()
y = 10*np.log10(ps.csm[:,0,0].real)
pl.semilogx(x,y)
pl.ylim(-45,-35)
pl.xlim(70,1400)
pl.show()
