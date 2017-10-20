# -*- coding: utf-8 -*-
"""
This script is a help for checking if the new NUMBA functions are correctly 
integrated into acoular.
One has to make a savefile (see 'all_bfWeave.sav') for both, the old acoular 
version an the new one. In section '#%% Compare Weave vs Numba' both versions 
are  compared.

This script uses essentially 'example3.py', so therefor 'example_data.h5' and 
'example_calib.xml' are needed.


Copyright (c) 2006-2015 The Acoular developers.
All rights reserved.
"""

# imports from acoular
import acoular
from acoular import L_p, TimeSamples, Calib, MicGeom, EigSpectra,\
RectGrid3D, BeamformerBase, BeamformerFunctional, BeamformerEig, BeamformerOrth, \
BeamformerCleansc, BeamformerCapon, BeamformerMusic, BeamformerCMF, PointSpreadFunction, BeamformerClean, BeamformerDamas

# other imports
from os import path
#from mayavi import mlab
from numpy import amax
#from cPickle import dump, load
from pickle import dump, load

# see example3
t = TimeSamples(name='example_data.h5')
cal = Calib(from_file='example_calib.xml')
m = MicGeom(from_file=path.join(\
    path.split(acoular.__file__)[0], 'xml', 'array_56.xml'))
g = RectGrid3D(x_min=-0.6, x_max=-0.0, y_min=-0.3, y_max=0.3, \
    z_min=0.48, z_max=0.88, increment=0.1)
f = EigSpectra(time_data=t, window='Hanning', overlap='50%', block_size=128, ind_low=5, ind_high=15)
csm = f.csm[:]
eva = f.eva[:]
eve = f.eve[:]

#""" Creating the beamformers
bb1Rem = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='classic')
bb2Rem = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='inverse')
bb3Rem = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true level')
bb4Rem = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true location')
bb1Full = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic')
bb2Full = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse')
bb3Full = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level')
bb4Full = BeamformerBase(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location')
Lbb1Rem = L_p(bb1Rem.synthetic(4000,1))
Lbb2Rem = L_p(bb2Rem.synthetic(4000,1))
Lbb3Rem = L_p(bb3Rem.synthetic(4000,1))
Lbb4Rem = L_p(bb4Rem.synthetic(4000,1))
Lbb1Full = L_p(bb1Full.synthetic(4000,1))
Lbb2Full = L_p(bb2Full.synthetic(4000,1))
Lbb3Full = L_p(bb3Full.synthetic(4000,1))
Lbb4Full = L_p(bb4Full.synthetic(4000,1))

bf1Rem = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='classic', gamma=3)
bf2Rem = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='inverse', gamma=3)
bf3Rem = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true level', gamma=3)
bf4Rem = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true location', gamma=3)
bf1Full = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic', gamma=3)
bf2Full = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse', gamma=3)
bf3Full = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level', gamma=3)
bf4Full = BeamformerFunctional(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location', gamma=3)
Lbf1Rem = L_p(bf1Rem.synthetic(4000,1))
Lbf2Rem = L_p(bf2Rem.synthetic(4000,1))
Lbf3Rem = L_p(bf3Rem.synthetic(4000,1))
Lbf4Rem = L_p(bf4Rem.synthetic(4000,1))
Lbf1Full = L_p(bf1Full.synthetic(4000,1))
Lbf2Full = L_p(bf2Full.synthetic(4000,1))
Lbf3Full = L_p(bf3Full.synthetic(4000,1))
Lbf4Full = L_p(bf4Full.synthetic(4000,1))

bca1Full = BeamformerCapon(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic')
bca2Full = BeamformerCapon(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse')
bca3Full = BeamformerCapon(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level')
bca4Full = BeamformerCapon(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location')
Lbca1Full = L_p(bca1Full.synthetic(4000,1))
Lbca2Full = L_p(bca2Full.synthetic(4000,1))
Lbca3Full = L_p(bca3Full.synthetic(4000,1))
Lbca4Full = L_p(bca4Full.synthetic(4000,1))

be1Rem = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='classic', n=12)
be2Rem = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='inverse', n=12)
be3Rem = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true level', n=12)
be4Rem = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true location', n=12)
be1Full = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic', n=12)
be2Full = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse', n=12)
be3Full = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level', n=12)
be4Full = BeamformerEig(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location', n=12)
Lbe1Rem = L_p(be1Rem.synthetic(4000,1))
Lbe2Rem = L_p(be2Rem.synthetic(4000,1))
Lbe3Rem = L_p(be3Rem.synthetic(4000,1))
Lbe4Rem = L_p(be4Rem.synthetic(4000,1))
Lbe1Full = L_p(be1Full.synthetic(4000,1))
Lbe2Full = L_p(be2Full.synthetic(4000,1))
Lbe3Full = L_p(be3Full.synthetic(4000,1))
Lbe4Full = L_p(be4Full.synthetic(4000,1))

bm1Full = BeamformerMusic(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic', n=12)
bm2Full = BeamformerMusic(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse', n=12)
bm3Full = BeamformerMusic(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level', n=12)
bm4Full = BeamformerMusic(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location', n=12)
Lbm1Full = L_p(bm1Full.synthetic(4000,1))
Lbm2Full = L_p(bm2Full.synthetic(4000,1))
Lbm3Full = L_p(bm3Full.synthetic(4000,1))
Lbm4Full = L_p(bm4Full.synthetic(4000,1))

bcsc1Rem = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='classic')
bcsc2Rem = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='inverse')
bcsc3Rem = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true level')
bcsc4Rem = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true location')
bcsc1Full = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic')
bcsc2Full = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse')
bcsc3Full = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level')
bcsc4Full = BeamformerCleansc(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location')
Lbcsc1Rem = L_p(bcsc1Rem.synthetic(4000,1))
Lbcsc2Rem = L_p(bcsc2Rem.synthetic(4000,1))
Lbcsc3Rem = L_p(bcsc3Rem.synthetic(4000,1))
Lbcsc4Rem = L_p(bcsc4Rem.synthetic(4000,1))
Lbcsc1Full = L_p(bcsc1Full.synthetic(4000,1))
Lbcsc2Full = L_p(bcsc2Full.synthetic(4000,1))
Lbcsc3Full = L_p(bcsc3Full.synthetic(4000,1))
Lbcsc4Full = L_p(bcsc4Full.synthetic(4000,1))

bort1Rem = BeamformerOrth(beamformer=be1Rem, eva_list=list(range(4,8)))
bort2Rem = BeamformerOrth(beamformer=be2Rem, eva_list=list(range(4,8)))
bort3Rem = BeamformerOrth(beamformer=be3Rem, eva_list=list(range(4,8)))
bort4Rem = BeamformerOrth(beamformer=be4Rem, eva_list=list(range(4,8)))
bort1Full = BeamformerOrth(beamformer=be1Full, eva_list=list(range(4,8)))
bort2Full = BeamformerOrth(beamformer=be2Full, eva_list=list(range(4,8)))
bort3Full = BeamformerOrth(beamformer=be3Full, eva_list=list(range(4,8)))
bort4Full = BeamformerOrth(beamformer=be4Full, eva_list=list(range(4,8)))
Lbort1Rem = L_p(bort1Rem.synthetic(4000,1))
Lbort2Rem = L_p(bort2Rem.synthetic(4000,1))
Lbort3Rem = L_p(bort3Rem.synthetic(4000,1))
Lbort4Rem = L_p(bort4Rem.synthetic(4000,1))
Lbort1Full = L_p(bort1Full.synthetic(4000,1))
Lbort2Full = L_p(bort2Full.synthetic(4000,1))
Lbort3Full = L_p(bort3Full.synthetic(4000,1))
Lbort4Full = L_p(bort4Full.synthetic(4000,1))

bcmf1Rem = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='classic')
bcmf2Rem = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='inverse')
bcmf3Rem = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true level')
bcmf4Rem = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=True, c=346.04, steer='true location')
bcmf1Full = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='classic')
bcmf2Full = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='inverse')
bcmf3Full = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true level')
bcmf4Full = BeamformerCMF(freq_data=f, grid=g, mpos=m, r_diag=False, c=346.04, steer='true location')
Lbcmf1Rem = L_p(bcmf1Rem.synthetic(4000,1))
Lbcmf2Rem = L_p(bcmf2Rem.synthetic(4000,1))
Lbcmf3Rem = L_p(bcmf3Rem.synthetic(4000,1))
Lbcmf4Rem = L_p(bcmf4Rem.synthetic(4000,1))
Lbcmf1Full = L_p(bcmf1Full.synthetic(4000,1))
Lbcmf2Full = L_p(bcmf2Full.synthetic(4000,1))
Lbcmf3Full = L_p(bcmf3Full.synthetic(4000,1))
Lbcmf4Full = L_p(bcmf4Full.synthetic(4000,1))

##==============================================================================
## There are various variations to calculate the psf: Need to be checked individually
## #psfSingle = PointSpreadFunction(grid=g, mpos=m, calcmode='single')
## #LPsfSingle = L_p(psfSingle.psf[:])
## 
## #psfBlock = PointSpreadFunction(grid=g, mpos=m, calcmode='block')
## #LPsfBlock = L_p(psfBlock.psf[:])
## 
## #psfFull = PointSpreadFunction(grid=g, mpos=m, calcmode='full')
## #LPsfFull = L_p(psfFull.psf[:])
##
## #all_bf = (LPsfFull,)
##==============================================================================
psf1 = PointSpreadFunction(grid=g, mpos=m, c=346.04, steer='classic')
psf2 = PointSpreadFunction(grid=g, mpos=m, c=346.04, steer='inverse')
psf3 = PointSpreadFunction(grid=g, mpos=m, c=346.04, steer='true level')
psf4 = PointSpreadFunction(grid=g, mpos=m, c=346.04, steer='true location')
Lpsf1 = L_p(psf1.psf[:])
Lpsf2 = L_p(psf2.psf[:])
Lpsf3 = L_p(psf3.psf[:])
Lpsf4 = L_p(psf4.psf[:])

bcpsf1Rem = BeamformerClean(beamformer=bb1Rem)
bcpsf2Rem = BeamformerClean(beamformer=bb2Rem)
bcpsf3Rem = BeamformerClean(beamformer=bb3Rem)
bcpsf4Rem = BeamformerClean(beamformer=bb4Rem)
bcpsf1Full = BeamformerClean(beamformer=bb1Full)
bcpsf2Full = BeamformerClean(beamformer=bb2Full)
bcpsf3Full = BeamformerClean(beamformer=bb3Full)
bcpsf4Full = BeamformerClean(beamformer=bb4Full)
Lbcpsf1Rem = L_p(bcpsf1Rem.synthetic(4000,1))
Lbcpsf2Rem = L_p(bcpsf2Rem.synthetic(4000,1))
Lbcpsf3Rem = L_p(bcpsf3Rem.synthetic(4000,1))
Lbcpsf4Rem = L_p(bcpsf4Rem.synthetic(4000,1))
Lbcpsf1Full = L_p(bcpsf1Full.synthetic(4000,1))
Lbcpsf2Full = L_p(bcpsf2Full.synthetic(4000,1))
Lbcpsf3Full = L_p(bcpsf3Full.synthetic(4000,1))
Lbcpsf4Full = L_p(bcpsf4Full.synthetic(4000,1))

bd1Rem = BeamformerDamas(beamformer=bb1Rem, n_iter=100)
bd2Rem = BeamformerDamas(beamformer=bb2Rem, n_iter=100)
bd3Rem = BeamformerDamas(beamformer=bb3Rem, n_iter=100)
bd4Rem = BeamformerDamas(beamformer=bb4Rem, n_iter=100)
bd1Full = BeamformerDamas(beamformer=bb1Full, n_iter=100)
bd2Full = BeamformerDamas(beamformer=bb2Full, n_iter=100)
bd3Full = BeamformerDamas(beamformer=bb3Full, n_iter=100)
bd4Full = BeamformerDamas(beamformer=bb4Full, n_iter=100)
Lbd1Rem = L_p(bd1Rem.synthetic(4000,1))
Lbd2Rem = L_p(bd2Rem.synthetic(4000,1))
Lbd3Rem = L_p(bd3Rem.synthetic(4000,1))
Lbd4Rem = L_p(bd4Rem.synthetic(4000,1))
Lbd1Full = L_p(bd1Full.synthetic(4000,1))
Lbd2Full = L_p(bd2Full.synthetic(4000,1))
Lbd3Full = L_p(bd3Full.synthetic(4000,1))
Lbd4Full = L_p(bd4Full.synthetic(4000,1))



all_bf = (Lbb1Rem, Lbb2Rem, Lbb3Rem, Lbb4Rem, Lbb1Full, Lbb2Full, Lbb3Full, Lbb4Full,
          Lbf1Rem, Lbf2Rem, Lbf3Rem, Lbf4Rem, Lbf1Full, Lbf2Full, Lbf3Full, Lbf4Full,
          Lbca1Full, Lbca2Full, Lbca3Full, Lbca4Full,
          Lbe1Rem, Lbe2Rem, Lbe3Rem, Lbe4Rem, Lbe1Full, Lbe2Full, Lbe3Full, Lbe4Full,
          Lbm1Full, Lbm2Full, Lbm3Full, Lbm4Full,
          Lbcsc1Rem, Lbcsc2Rem, Lbcsc3Rem, Lbcsc4Rem, Lbcsc1Full, Lbcsc2Full, Lbcsc3Full, Lbcsc4Full,
          Lbort1Rem, Lbort2Rem, Lbort3Rem, Lbort4Rem, Lbort1Full, Lbort2Full, Lbort3Full, Lbort4Full,
          Lbcmf1Rem, Lbcmf2Rem, Lbcmf3Rem, Lbcmf4Rem, Lbcmf1Full, Lbcmf2Full, Lbcmf3Full, Lbcmf4Full, 
          Lpsf1, Lpsf2, Lpsf3, Lpsf4,
          Lbcpsf1Rem, Lbcpsf2Rem, Lbcpsf3Rem, Lbcpsf4Rem, Lbcpsf1Full, Lbcpsf2Full, Lbcpsf3Full, Lbcpsf4Full,
          Lbd1Rem, Lbd2Rem, Lbd3Rem, Lbd4Rem, Lbd1Full, Lbd2Full, Lbd3Full, Lbd4Full)

fi = open('all_bfWeave.sav','w')  # This file saves the outputs of the current acoular version
#fi = open('all_bfNumba.sav','w')  # This file saves the outputs of the new acoular version, which has to be validated

dump(all_bf,fi,-1) # uses newest pickle protocol -1 (default = 0)
fi.close()
#"""

#%% Compare Weave vs Numba
fi = open('all_bfWeave.sav','r')
all_bfWeave = load(fi)
fi.close()

fi = open('all_bfNumba.sav','r')
all_bfNumba = load(fi)
fi.close()

# remove all negative levels
err = []  # keep in mind that these are levels!!!
for cnt in range(len(all_bfNumba)):
    all_bfNumba[cnt][all_bfNumba[cnt] < 0] = all_bfWeave[cnt][all_bfWeave[cnt] < 0] = 1e-20
    relDiff = (all_bfWeave[cnt] - all_bfNumba[cnt]) / (all_bfWeave[cnt] + all_bfNumba[cnt]) * 2
    err.append(amax(amax(amax(abs(relDiff), 0), 0), 0))
