# -*- coding: utf-8 -*-
"""
Beamfpy library

testing and timing the different beamformers in 2D

(c) Ennes Sarradj 2007-2010, all rights reserved
"""
import beamfpy
print beamfpy.__file__

from beamfpy import td_dir, L_p, TimeSamples, Calib, MicGeom, \
EigSpectra, RectGrid, BeamformerBase, BeamformerCapon, BeamformerMusic, \
BeamformerEig, BeamformerOrth, BeamformerCleansc, BeamformerDamas
from os import path

def test():
    """ test function """
    # set up input data 
    cal = Calib(from_file=path.join(td_dir, 'calib_06_05_2008.xml'))
    t = TimeSamples( \
        name=path.join(td_dir, '2008-05-16_11-36-00_468000.h5'), calib=cal)
    m = MicGeom(from_file=path.join( \
        path.split(beamfpy.__file__)[0], 'xml', 'array_56.xml'))
    
    # set up spectra and grids
    f = EigSpectra(time_data = t, window='Hanning', overlap='50%', 
                   block_size=128, ind_low=15, ind_high=30, calib=cal)
    g = RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, \
                 z=0.68, increment=0.005)
    g_coarse = RectGrid(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, \
                        z=0.68, increment=0.05) 
    
    print f.csm[25,1,1]
    # different beamformers
    bb = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04)
    bc = BeamformerCapon(freq_data=f, grid=g, mpos=m, c=346)
    be = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346, n=50)
    bm = BeamformerMusic(freq_data=f, grid=g, mpos=m, r_diag=True, c=346, n=2)
    bb1 = BeamformerBase(freq_data=f, grid=g_coarse, mpos=m, r_diag=True, \
                        c=346.04)
    bd = BeamformerDamas(beamformer=bb1, n_iter=100)
    bo = BeamformerOrth(beamformer=be, eva_list=range(40, 56))
    bs = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346)
    
    # plot results, print timing
    from pylab import subplot, imshow, show, colorbar, plot, figure
    from time import time
    
    i = 1
    for b in (bb, bc, bm, bd, bo, bs):
        subplot(2, 3, i)
        ti1 = time()
        mapd = L_p(b.synthetic(8000, 1).T)
        print b.__class__.__name__, time()-ti1
        imshow(mapd, vmin=mapd.max()-20, interpolation='nearest', \
                extent=b.grid.extend())
        colorbar()
        i += 1
    
    figure(2)
    plot(L_p(bo.integrate((-0.3, -0.1, -0.1, 0.1)))[f.ind_low:f.ind_high])
    plot(L_p(bs.integrate((-0.3, -0.1, -0.1, 0.1)))[f.ind_low:f.ind_high])
    plot(L_p(bd.integrate((-0.3, -0.1, -0.1, 0.1)))[f.ind_low:f.ind_high])
    show()

if __name__ == "__main__":
    test()


